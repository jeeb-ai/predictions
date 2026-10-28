"""Read-only, public Polymarket US market data with immutable response snapshots.

Only the documented US gateway is reachable through this adapter.  There are
no credentials, account methods, order endpoints, or trading operations here.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

import requests

from predictions.data import TEAM_ALIASES, TEAMS, _atomic_write

API_BASE = "https://gateway.polymarket.us"
FEES_URL = "https://docs.polymarket.us/fees.md"
SPORTS_RULES_URL = "https://docs.polymarket.us/faqs/sports-faqs.md"
FEE_EFFECTIVE_AT = "2026-10-19T17:00:00+00:00"
STANDARD_TAKER_THETA = Decimal("0.0695")


def _slug(value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) > 250
        or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", value)
    ):
        raise ValueError("A market slug must contain lowercase letters, digits, and hyphens only")
    return value


def _decimal(value, name: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite decimal")
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{name} must be a finite decimal") from exc
    if not number.is_finite():
        raise ValueError(f"{name} must be a finite decimal")
    return number


def _timestamp(value, name: str) -> str:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Missing or invalid {name} timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{name} timestamp must include a timezone")
    return parsed.astimezone(timezone.utc).isoformat()


def _team(side: dict) -> str:
    team = side.get("team")
    if not isinstance(team, dict) or str(team.get("league", "")).casefold() != "nfl":
        raise ValueError("Only documented NFL team winner markets are supported")
    code = str(team.get("abbreviation", "")).upper()
    code = TEAM_ALIASES.get(code, code)
    if code not in TEAMS:
        raise ValueError("Market contains an unknown NFL team abbreviation")
    return code


def _levels(entries, *, complement: bool) -> list[dict]:
    if not isinstance(entries, list):
        raise ValueError("Market book must contain explicit bids and offers depth")
    quantities: dict[Decimal, Decimal] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("px"), dict):
            raise ValueError("Malformed market book depth entry")
        if entry["px"].get("currency") != "USD":
            raise ValueError("Only USD market books are supported")
        price = _decimal(entry["px"].get("value"), "book price")
        quantity = _decimal(entry.get("qty"), "book quantity")
        if not 0 < price < 1 or quantity <= 0:
            raise ValueError(
                "Book prices must be strictly between zero and one with positive depth"
            )
        if complement:
            price = 1 - price
        quantities[price] = quantities.get(price, Decimal(0)) + quantity
    return [
        {"price": str(price), "quantity": str(quantity)}
        for price, quantity in sorted(quantities.items())
    ]


def normalize_quote(market: dict, book: dict, *, outcome: str, source: dict) -> dict:
    """Normalize one verified NFL winner book; never use display/outcome prices.

    The exchange has one YES instrument.  Buying synthetic NO consumes its
    YES bids at the complementary price; it is not an independent token book.
    Empty depth is retained so callers can record a no-liquidity PASS.
    """
    if outcome not in {"yes", "no"}:
        raise ValueError("outcome must be yes or no")
    if not isinstance(market, dict) or not isinstance(book, dict):
        raise ValueError("Market details and book must be JSON objects")
    slug = _slug(market.get("slug"))
    if book.get("marketSlug") != slug:
        raise ValueError("Market identity does not match the order book")
    if (
        market.get("sportsMarketType") != "football_team_full_game_winner"
        or market.get("sportsMarketTypeV2") != "SPORTS_MARKET_TYPE_MONEYLINE"
    ):
        raise ValueError("Only full-game NFL winner moneyline contracts are supported")
    if (
        market.get("active") is not True
        or market.get("closed") is not False
        or market.get("status") != "MARKET_STATUS_OPEN"
        or book.get("state") != "MARKET_STATE_OPEN"
    ):
        raise ValueError("The market and book must both be open")
    sides = market.get("marketSides")
    if (
        not isinstance(sides, list)
        or len(sides) != 2
        or any(not isinstance(side, dict) or type(side.get("long")) is not bool for side in sides)
    ):
        raise ValueError("Market must explicitly identify two long/short sides")
    by_long = {side["long"]: side for side in sides}
    if set(by_long) != {False, True} or any(side.get("tradable") is not True for side in sides):
        raise ValueError("Market sides must be distinct and tradable")
    yes_team, no_team = _team(by_long[True]), _team(by_long[False])
    if yes_team == no_team:
        raise ValueError("Market must identify two different NFL teams")
    description = market.get("description")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("The market's actual settlement terms are required")
    lowered = description.casefold()
    if (
        "overtime is included" not in lowered
        or re.search(r"tie[^.]*\$0\.50", lowered) is None
        or "last fair market price" not in lowered
    ):
        raise ValueError("Unsupported or ambiguous NFL overtime, tie, or cancellation terms")
    theta = _decimal(market.get("feeCoefficient"), "market fee coefficient")
    if theta != STANDARD_TAKER_THETA:
        raise ValueError(
            "Market fee coefficient differs from the verified standard US fee schedule"
        )
    quote_at = _timestamp(book.get("transactTime"), "book transactTime")
    received_at = _timestamp(source["book"].get("fetched_at"), "book received_at")
    if datetime.fromisoformat(quote_at) < datetime.fromisoformat(FEE_EFFECTIVE_AT):
        raise ValueError("This quote predates the verified US fee schedule")
    # Both sides are validated, even when only one is requested.
    yes_asks = _levels(book.get("offers"), complement=False)
    no_asks = _levels(book.get("bids"), complement=True)
    if yes_asks and no_asks and Decimal(yes_asks[0]["price"]) + Decimal(no_asks[0]["price"]) < 1:
        raise ValueError("Crossed book: best YES bid exceeds best YES offer")
    quantity = _decimal(market.get("minimumTradeQty"), "minimum trade quantity")
    tick = _decimal(market.get("orderPriceMinTickSize"), "price tick")
    if quantity <= 0 or not 0 < tick < 1:
        raise ValueError("Missing or invalid minimum trade quantity/price tick")
    identifier = str(market.get("id", "")).strip()
    if not identifier:
        raise ValueError("Market ID is required")
    identity = {
        "market_sha256": source["market"]["sha256"],
        "book_sha256": source["book"]["sha256"],
        "outcome": outcome,
        "received_at": received_at,
    }
    return {
        "schema_version": 1,
        "quote_id": hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
        "liquidity_key": hashlib.sha256(
            f"{source['book']['sha256']}:{identifier}:{outcome}".encode()
        ).hexdigest(),
        "venue": "polymarket_us",
        "market_id": identifier,
        "market_slug": slug,
        "outcome": outcome,
        "ask_levels": yes_asks if outcome == "yes" else no_asks,
        "quote_at": quote_at,
        "received_at": received_at,
        "market_status": "open",
        "game_start_time": _timestamp(market.get("gameStartTime"), "game start"),
        "yes_team": yes_team,
        "no_team": no_team,
        "minimum_trade_quantity": str(quantity),
        "price_tick": str(tick),
        "settlement": {
            "rule_type": "nfl_winner_including_overtime",
            "win_payout": "1",
            "loss_payout": "0",
            "tie_payout": "0.5",
            "cancellation": "last_fair_market_price",
            "description": description,
            "rules_url": SPORTS_RULES_URL,
            "terms_sha256": hashlib.sha256(description.encode()).hexdigest(),
        },
        "fee_schedule": {
            "type": "polymarket_us_standard_taker",
            "theta": str(theta),
            "effective_at": FEE_EFFECTIVE_AT,
            "rounding": "ROUND_HALF_EVEN",
            "currency": "USD",
            "source_url": FEES_URL,
            "note": "Per-fill banker's rounding capped by rounded cumulative exact fee; L2 depth hides individual maker fill partitions. No rebates assumed.",
        },
        "source": source,
    }


class PolymarketUSProvider:
    """Public US gateway adapter; every network request is unauthenticated GET."""

    def __init__(self, cache_dir: Path, timeout: float = 30, retries: int = 3):
        if (
            not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
            or not isinstance(retries, int)
            or isinstance(retries, bool)
            or retries < 0
        ):
            raise ValueError("timeout must be positive/finite and retries nonnegative")
        self.cache_dir = Path(cache_dir)
        self.timeout, self.retries = timeout, retries

    def _get(self, path: str, params: dict | None = None) -> tuple[dict, dict]:
        url = API_BASE + path
        for attempt in range(self.retries + 1):
            response = None
            try:
                response = requests.get(
                    url, params=params, timeout=self.timeout, allow_redirects=False
                )
                if (
                    response.status_code == 429 or response.status_code >= 500
                ) and attempt < self.retries:
                    retry_after = response.headers.get("Retry-After", "")
                    try:
                        delay = min(max(float(retry_after), 0), 8)
                        if not math.isfinite(delay):
                            raise ValueError
                    except ValueError:
                        delay = min(2**attempt, 8)
                    time.sleep(delay)
                    continue
                response.raise_for_status()
                if 300 <= response.status_code < 400:
                    raise ValueError("Redirects are not allowed for the fixed US gateway")
                content = response.content
                payload = json.loads(content)
                if not isinstance(payload, dict):
                    raise ValueError("US market gateway returned a non-object JSON response")
                break
            except (requests.ConnectionError, requests.Timeout):
                if attempt == self.retries:
                    raise
                time.sleep(min(2**attempt, 8))
            finally:
                if response is not None:
                    response.close()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(content).hexdigest()
        raw_path = self.cache_dir / f"{digest}.json"
        if raw_path.exists():
            if hashlib.sha256(raw_path.read_bytes()).hexdigest() != digest:
                raise ValueError("Existing immutable market snapshot is corrupt")
        else:
            _atomic_write(raw_path, content)
        fetched = datetime.now(timezone.utc)
        snapshot_id = fetched.strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:8]
        metadata = {
            "snapshot_id": snapshot_id,
            "url": response.url or url,
            "sha256": digest,
            "fetched_at": fetched.isoformat(),
            "path": str(raw_path),
            "venue": "polymarket_us",
        }
        _atomic_write(
            self.cache_dir / f"{snapshot_id}.metadata.json",
            (json.dumps(metadata, sort_keys=True, indent=2) + "\n").encode(),
        )
        return payload, metadata

    def list_markets(self, limit: int = 100, offset: int = 0) -> list[dict]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("limit must be an integer between 1 and 100")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be a nonnegative integer")
        payload, _ = self._get("/v1/markets", {"limit": limit, "offset": offset})
        if not isinstance(payload.get("markets"), list):
            raise ValueError("US gateway response lacks a markets list")
        return payload["markets"]

    def list_nfl_markets(self, limit: int = 20, offset: int = 0) -> list[dict]:
        markets, _ = self.list_nfl_market_page(limit=limit, offset=offset)
        return markets

    def list_nfl_market_page(self, limit: int = 20, offset: int = 0) -> tuple[list[dict], int]:
        """Return winner markets and the raw event count used for pagination.

        Filtering can yield no winner markets on a nonempty event page. Callers
        must use the event count, rather than the filtered list, to detect the
        end of the league endpoint's pagination.
        """
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("limit must be an integer between 1 and 100")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be a nonnegative integer")
        payload, _ = self._get("/v2/leagues/nfl/events", {"limit": limit, "offset": offset})
        if not isinstance(payload.get("events"), list):
            raise ValueError("US gateway response lacks an NFL events list")
        markets = [
            market
            for event in payload["events"]
            for market in event.get("markets", [])
            if market.get("sportsMarketType") == "football_team_full_game_winner"
        ]
        return markets, len(payload["events"])

    def get_quote(self, market_slug: str, outcome: str = "yes") -> dict:
        slug = _slug(market_slug)
        if outcome not in {"yes", "no"}:
            raise ValueError("outcome must be yes or no")
        details, market_source = self._get(f"/v1/market/slug/{slug}")
        book, book_source = self._get(f"/v1/markets/{slug}/book")
        return normalize_quote(
            details.get("market"),
            book.get("marketData"),
            outcome=outcome,
            source={"market": market_source, "book": book_source},
        )

    def get_settlement(self, market_slug: str) -> dict:
        """Read a final settlement only; provisional/closed markets are rejected."""
        slug = _slug(market_slug)
        details, market_source = self._get(f"/v1/market/slug/{slug}")
        market = details.get("market")
        if not isinstance(market, dict) or market.get("slug") != slug:
            raise ValueError("Settlement market identity is missing or mismatched")
        if (
            market.get("sportsMarketType") != "football_team_full_game_winner"
            or market.get("sportsMarketTypeV2") != "SPORTS_MARKET_TYPE_MONEYLINE"
            or market.get("status") != "MARKET_STATUS_RESOLVED"
        ):
            raise ValueError("Only finally resolved NFL winner markets can be settled")
        result, settlement_source = self._get(f"/v1/markets/{slug}/settlement")
        if result.get("slug") != slug:
            raise ValueError("Settlement response has a mismatched market slug")
        price = _decimal(result.get("settlement"), "settlement price")
        if not 0 <= price <= 1:
            raise ValueError("Settlement price must be between zero and one")
        return {
            "venue": "polymarket_us",
            "market_id": str(market.get("id", "")),
            "market_slug": slug,
            "market_status": "resolved",
            "settlement_price_yes": str(price),
            "settlement_price_no": str(1 - price),
            "observed_at": settlement_source["fetched_at"],
            "source": {"market": market_source, "settlement": settlement_source},
        }
