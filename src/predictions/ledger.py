"""Append-only, local paper positions; this module has no network execution path.

SQLite transactions reserve displayed liquidity and record canonical payloads in
a hash chain. The head hash can be retained externally to detect truncation;
without an external anchor a database owner can rewrite an entire local file.
Paper fills use the conservative documented fee upper bound, not an assertion
that a real exchange fill or commission occurred.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

TABLES = {
    "forecast": ("paper_forecasts", "forecast_id"),
    "quote": ("paper_quotes", "quote_id"),
    "evaluation": ("paper_evaluations", "evaluation_id"),
    "paper_fill": ("paper_trades", "paper_trade_id"),
    "settlement": ("paper_settlements", "settlement_id"),
}
GENESIS_HASH = "0" * 64


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _time(value, field: str) -> datetime:
    try:
        result = (
            value
            if isinstance(value, datetime)
            else datetime.fromisoformat(value.replace("Z", "+00:00"))
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a timezone-aware timestamp") from exc
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone")
    return result.astimezone(timezone.utc)


def _decimal(value, field: str) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{field} must be a finite decimal")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{field} must be a finite decimal") from exc
    if not result.is_finite():
        raise ValueError(f"{field} must be a finite decimal")
    return result


def _text(payload: dict, field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing or invalid {field}")
    return value


def _canonical(payload: dict) -> str:
    if not isinstance(payload, dict):
        raise ValueError("Ledger payload must be an object")
    try:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("Ledger payload must contain finite JSON values") from exc


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _hash(value, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"{field} must be a SHA-256 hexadecimal digest")
    return value


def _levels(items: list) -> dict[Decimal, Decimal]:
    if not isinstance(items, list):
        raise ValueError("Ask levels must be a list")
    levels = {}
    previous = Decimal("0")
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Every ask level must be an object")
        price = _decimal(item.get("price"), "ask price")
        quantity = _decimal(item.get("quantity"), "ask quantity")
        if not 0 < price < 1 or quantity <= 0 or price < previous:
            raise ValueError("Ask levels require ascending prices in (0,1) and positive quantities")
        levels[price] = levels.get(price, Decimal("0")) + quantity
        previous = price
    return levels


def _validate_forecast(payload: dict) -> None:
    for field in ["forecast_id", "game_id", "home_team", "away_team", "model_id", "model_version"]:
        _text(payload, field)
    for field in ["model_sha256", "features_sha256"]:
        _hash(payload.get(field), field)
    kickoff = _time(payload.get("kickoff"), "kickoff")
    as_of = _time(payload.get("as_of"), "as_of")
    if as_of >= kickoff or payload["home_team"] == payload["away_team"]:
        raise ValueError("Forecast must be pregame with distinct teams")
    if payload.get("probability_scope") != "conditional_on_decisive_result":
        raise ValueError("Unsupported forecast probability scope")
    home = _decimal(payload.get("home_win_probability"), "home probability")
    away = _decimal(payload.get("away_win_probability"), "away probability")
    if not (0 <= home <= 1 and 0 <= away <= 1) or abs(home + away - 1) > Decimal("1e-12"):
        raise ValueError("Forecast probabilities must be complementary numbers in [0,1]")
    if payload.get("generated_at") is not None:
        generated = _time(payload["generated_at"], "generated_at")
        if generated < as_of or generated >= kickoff:
            raise ValueError("Forecast generation must occur before kickoff and after its cutoff")
    for field in ["feature_cutoff_at", "model_training_cutoff_at"]:
        if payload.get(field) is not None and _time(payload[field], field) > as_of:
            raise ValueError(f"{field} cannot postdate forecast as_of")
    for field in [
        "model_train_max_result_available_at",
        "history_available_through",
        "advanced_history_available_through",
        "home_qb_proxy_available_at",
        "away_qb_proxy_available_at",
    ]:
        if payload.get(field) is not None and _time(payload[field], field) >= as_of:
            raise ValueError(f"{field} must strictly precede forecast as_of")


def _validate_quote(payload: dict) -> None:
    for field in ["quote_id", "market_id", "yes_team", "no_team"]:
        _text(payload, field)
    if payload.get("venue") != "polymarket_us" or payload.get("outcome") not in {"yes", "no"}:
        raise ValueError("Only Polymarket US binary quotes are supported")
    if payload["yes_team"] == payload["no_team"]:
        raise ValueError("Quote teams must be distinct")
    quote_at = _time(payload.get("quote_at"), "quote_at")
    received = _time(payload.get("received_at"), "received_at")
    _time(payload.get("game_start_time"), "game_start_time")
    if quote_at > received:
        raise ValueError("Quote timestamp cannot postdate receipt")
    _levels(payload.get("ask_levels"))
    source = payload.get("source")
    if not isinstance(source, dict) or not isinstance(source.get("book"), dict):
        raise ValueError("Quote requires its raw book source provenance")
    _hash(source["book"].get("sha256"), "book source hash")
    if not isinstance(payload.get("fee_schedule"), dict) or not isinstance(
        payload.get("settlement"), dict
    ):
        raise ValueError("Quote requires fee and settlement terms")


def _retained_sources(source: dict, names: list[str]) -> dict:
    retained = {}
    for name in names:
        item = source.get(name)
        if not isinstance(item, dict) or not item.get("path"):
            continue
        path = Path(item["path"])
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item.get(
            "sha256"
        ):
            raise ValueError("Raw snapshot is missing or its checksum differs")
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError("Retained raw snapshot must be a JSON object")
        retained[name] = value
    return retained


def _verify_quote_sources(payload: dict) -> str:
    """Reconcile executable fields to retained raw US responses when available."""
    retained = _retained_sources(payload["source"], ["market", "book"])
    if set(retained) != {"market", "book"}:
        return "raw_snapshots_not_retained"
    from predictions.markets import normalize_quote

    market = retained["market"].get("market", retained["market"])
    book = retained["book"].get("marketData", retained["book"])
    expected = normalize_quote(market, book, outcome=payload["outcome"], source=payload["source"])
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(f"Quote {field} disagrees with its retained raw snapshots")
    return "raw_snapshots_reconciled"


def _verify_settlement_evidence(evidence: dict, trade: dict, price: Decimal) -> str:
    if not any(
        key in evidence
        for key in ["venue", "market_status", "settlement_price_yes", "settlement_price_no"]
    ):
        return "operator_supplied"
    if (
        evidence.get("venue") != "polymarket_us"
        or evidence.get("market_status") != "resolved"
        or evidence.get("market_id") != trade["market_id"]
        or evidence.get("market_slug") != trade["quote"].get("market_slug")
    ):
        raise ValueError("Provider settlement does not match the paper position")
    yes = _decimal(evidence.get("settlement_price_yes"), "provider YES settlement price")
    no = _decimal(evidence.get("settlement_price_no"), "provider NO settlement price")
    if not 0 <= yes <= 1 or no != 1 - yes or price != (yes if trade["outcome"] == "yes" else no):
        raise ValueError("Price override disagrees with the provider settlement")
    source = evidence.get("source")
    if not isinstance(source, dict) or not isinstance(source.get("settlement"), dict):
        raise ValueError("Provider settlement requires source provenance")
    _hash(source["settlement"].get("sha256"), "settlement source hash")
    retained = _retained_sources(source, ["market", "settlement"])
    if set(retained) != {"market", "settlement"}:
        return "provider_raw_snapshots_not_retained"
    market = retained["market"].get("market", retained["market"])
    result = retained["settlement"]
    if (
        market.get("status") != "MARKET_STATUS_RESOLVED"
        or market.get("sportsMarketType") != "football_team_full_game_winner"
        or market.get("sportsMarketTypeV2") != "SPORTS_MARKET_TYPE_MONEYLINE"
        or str(market.get("id")) != trade["market_id"]
        or market.get("slug") != evidence["market_slug"]
        or result.get("slug") != evidence["market_slug"]
        or _decimal(result.get("settlement"), "raw settlement price") != yes
        or _time(evidence["observed_at"], "observed_at")
        != _time(source["settlement"].get("fetched_at"), "source fetched_at")
    ):
        raise ValueError("Provider settlement disagrees with its retained raw snapshots")
    return "public_us_settlement_snapshot"


def _reconcile(evaluation: dict, forecast: dict, quote: dict) -> None:
    """Recalculate linked EV rather than trusting an editable candidate flag."""
    from predictions.ev import evaluate_expected_value

    assumptions = evaluation.get("assumptions")
    if not isinstance(assumptions, dict):
        raise ValueError("Evaluation requires explicit assumptions")
    expected = evaluate_expected_value(
        forecast,
        quote,
        selected_team=_text(evaluation, "selected_team"),
        quantity=evaluation.get("quantity"),
        as_of=_time(evaluation.get("as_of"), "as_of"),
        tie_probability=assumptions.get("tie_probability"),
        completion_probability=assumptions.get("completion_probability"),
        cancellation_settlement_price=assumptions.get("cancellation_settlement_price"),
        max_quote_age_seconds=evaluation.get("max_quote_age_seconds", 60),
        max_forecast_age_seconds=evaluation.get("max_forecast_age_seconds", 3600),
        min_ev_per_contract=evaluation.get("min_ev_per_contract", "0"),
    )
    for field, value in expected.items():
        if evaluation.get(field) != value:
            raise ValueError(f"Evaluation {field} does not match its linked forecast and quote")


class PaperLedger:
    """Transactional local journal; no exchange orders or real-money accounts."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS paper_forecasts (
                    forecast_id TEXT PRIMARY KEY, payload_json TEXT NOT NULL, payload_sha256 TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS paper_quotes (
                    quote_id TEXT PRIMARY KEY, payload_json TEXT NOT NULL, payload_sha256 TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS paper_evaluations (
                    evaluation_id TEXT PRIMARY KEY,
                    forecast_id TEXT NOT NULL REFERENCES paper_forecasts(forecast_id),
                    quote_id TEXT NOT NULL REFERENCES paper_quotes(quote_id),
                    payload_json TEXT NOT NULL, payload_sha256 TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS paper_trades (
                    paper_trade_id TEXT PRIMARY KEY,
                    evaluation_id TEXT NOT NULL REFERENCES paper_evaluations(evaluation_id),
                    forecast_id TEXT NOT NULL REFERENCES paper_forecasts(forecast_id),
                    quote_id TEXT NOT NULL REFERENCES paper_quotes(quote_id),
                    idempotency_key TEXT UNIQUE NOT NULL, liquidity_key TEXT NOT NULL,
                    payload_json TEXT NOT NULL, payload_sha256 TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS paper_settlements (
                    settlement_id TEXT PRIMARY KEY,
                    paper_trade_id TEXT UNIQUE NOT NULL REFERENCES paper_trades(paper_trade_id),
                    idempotency_key TEXT UNIQUE NOT NULL,
                    payload_json TEXT NOT NULL, payload_sha256 TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS ledger_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT UNIQUE NOT NULL,
                    event_type TEXT NOT NULL, entity_id TEXT NOT NULL, recorded_at TEXT NOT NULL,
                    previous_hash TEXT NOT NULL, payload_json TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL, event_hash TEXT UNIQUE NOT NULL);
            """)
            for table in [*[item[0] for item in TABLES.values()], "ledger_events"]:
                for action in ["UPDATE", "DELETE"]:
                    connection.execute(
                        f"CREATE TRIGGER IF NOT EXISTS {table}_no_{action.lower()} BEFORE {action} ON {table} "
                        "BEGIN SELECT RAISE(ABORT, 'paper ledger is append-only'); END"
                    )
            unique_keys = {
                "paper_forecasts": ["forecast_id"],
                "paper_quotes": ["quote_id"],
                "paper_evaluations": ["evaluation_id"],
                "paper_trades": ["paper_trade_id", "idempotency_key"],
                "paper_settlements": ["settlement_id", "paper_trade_id", "idempotency_key"],
                "ledger_events": ["sequence", "event_id", "event_hash"],
            }
            # INSERT OR REPLACE bypasses ordinary DELETE triggers when a client
            # has recursive_triggers disabled. Reject replacement before it runs.
            for table, keys in unique_keys.items():
                condition = " OR ".join(f"{key}=NEW.{key}" for key in keys)
                connection.execute(
                    f"CREATE TRIGGER IF NOT EXISTS {table}_no_replace BEFORE INSERT ON {table} "
                    f"WHEN EXISTS (SELECT 1 FROM {table} WHERE {condition}) "
                    "BEGIN SELECT RAISE(ABORT, 'paper ledger is append-only'); END"
                )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    @staticmethod
    def _get(connection, kind: str, identity: str) -> dict:
        table, key = TABLES[kind]
        row = connection.execute(
            f"SELECT payload_json FROM {table} WHERE {key}=?", (identity,)
        ).fetchone()
        if row is None:
            raise ValueError(f"Unknown {kind}: {identity}")
        return json.loads(row[0])

    @staticmethod
    def _append_event(connection, kind: str, payload: dict) -> None:
        _, key = TABLES[kind]
        text = _canonical(payload)
        latest = connection.execute(
            "SELECT recorded_at,event_hash FROM ledger_events ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        now = _utcnow()
        if latest is not None and now < _time(latest["recorded_at"], "previous event time"):
            raise ValueError("Ledger clock moved backwards")
        event = {
            "event_id": uuid.uuid4().hex,
            "event_type": kind,
            "entity_id": payload[key],
            "recorded_at": now.isoformat(),
            "previous_hash": latest["event_hash"] if latest else GENESIS_HASH,
            "payload_sha256": _digest(text),
        }
        event_hash = _digest(_canonical(event))
        connection.execute(
            "INSERT INTO ledger_events(event_id,event_type,entity_id,recorded_at,previous_hash,payload_json,payload_sha256,event_hash) VALUES (?,?,?,?,?,?,?,?)",
            (
                *[
                    event[name]
                    for name in [
                        "event_id",
                        "event_type",
                        "entity_id",
                        "recorded_at",
                        "previous_hash",
                    ]
                ],
                text,
                event["payload_sha256"],
                event_hash,
            ),
        )

    @staticmethod
    def _insert(connection, kind: str, payload: dict, **references) -> dict:
        table, key = TABLES[kind]
        text = _canonical(payload)
        identity = _text(payload, key)
        row = connection.execute(
            f"SELECT payload_json FROM {table} WHERE {key}=?", (identity,)
        ).fetchone()
        if row is not None:
            if row[0] != text:
                raise ValueError(f"Existing {kind} ID has different content")
            return json.loads(row[0])
        columns = [key, *references, "payload_json", "payload_sha256"]
        values = [identity, *references.values(), text, _digest(text)]
        connection.execute(
            f"INSERT INTO {table}({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
            values,
        )
        PaperLedger._append_event(connection, kind, payload)
        return json.loads(text)

    def _record(self, kind: str, payload: dict) -> dict:
        _canonical(payload)
        if kind == "forecast":
            _validate_forecast(payload)
        elif kind == "quote":
            _validate_quote(payload)
            _verify_quote_sources(payload)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                references = {}
                if kind == "evaluation":
                    forecast_id, quote_id = (
                        _text(payload, "forecast_id"),
                        _text(payload, "quote_id"),
                    )
                    forecast = self._get(connection, "forecast", forecast_id)
                    quote = self._get(connection, "quote", quote_id)
                    _reconcile(payload, forecast, quote)
                    references = {"forecast_id": forecast_id, "quote_id": quote_id}
                result = self._insert(connection, kind, payload, **references)
                connection.commit()
                return result
            except BaseException:
                connection.rollback()
                raise

    def record_forecast(self, forecast: dict) -> dict:
        return self._record("forecast", forecast)

    def record_quote(self, quote: dict) -> dict:
        return self._record("quote", quote)

    def record_evaluation(self, evaluation: dict) -> dict:
        return self._record("evaluation", evaluation)

    def paper_fill(
        self,
        evaluation_id: str,
        *,
        idempotency_key: str,
        recorded_at=None,
        max_exposure=None,
    ) -> dict:
        """Reserve quoted levels and append a simulated fill, never an exchange order."""
        _text({"idempotency_key": idempotency_key}, "idempotency_key")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                previous = connection.execute(
                    "SELECT payload_json FROM paper_trades WHERE idempotency_key=?",
                    (idempotency_key,),
                ).fetchone()
                if previous is not None:
                    result = json.loads(previous[0])
                    if result["evaluation_id"] != evaluation_id:
                        raise ValueError("Idempotency key belongs to another evaluation")
                    connection.commit()
                    return result
                evaluation = self._get(connection, "evaluation", evaluation_id)
                forecast = self._get(connection, "forecast", evaluation["forecast_id"])
                quote = self._get(connection, "quote", evaluation["quote_id"])
                quote_verification = _verify_quote_sources(quote)
                _reconcile(evaluation, forecast, quote)
                if (
                    evaluation.get("mode") != "paper_only"
                    or evaluation.get("status") != "paper_candidate"
                ):
                    raise ValueError("Only a validated paper_candidate evaluation can fill")
                now = _utcnow()
                recorded = now if recorded_at is None else _time(recorded_at, "recorded_at")
                if abs((recorded - now).total_seconds()) > 5:
                    raise ValueError("Paper fills cannot be backdated or future-dated")
                kickoff = _time(forecast["kickoff"], "kickoff")
                if now >= kickoff or recorded >= kickoff:
                    raise ValueError("Cannot paper-fill a started game")
                max_age = evaluation.get("max_quote_age_seconds", 60)
                if not isinstance(max_age, int) or isinstance(max_age, bool) or max_age <= 0:
                    raise ValueError("Invalid quote freshness policy")
                for field, value in [
                    ("evaluation as_of", evaluation["as_of"]),
                    ("quote_at", quote["quote_at"]),
                    ("received_at", quote["received_at"]),
                    ("forecast as_of", forecast["as_of"]),
                ]:
                    timestamp = _time(value, field)
                    if timestamp > now:
                        raise ValueError(f"{field} cannot postdate the paper fill")
                    if timestamp > recorded:
                        raise ValueError("Paper fill cannot predate the information used")
                    if field != "forecast as_of" and now - timestamp > timedelta(seconds=max_age):
                        raise ValueError("Quote or evaluation is stale at paper-fill time")
                forecast_age = evaluation.get("max_forecast_age_seconds", 3600)
                if (
                    not isinstance(forecast_age, int)
                    or isinstance(forecast_age, bool)
                    or forecast_age <= 0
                ):
                    raise ValueError("Invalid forecast freshness policy")
                if now - _time(forecast["as_of"], "forecast as_of") > timedelta(
                    seconds=forecast_age
                ):
                    raise ValueError("Forecast is stale at paper-fill time")
                liquidity_key = _digest(
                    _canonical(
                        {
                            "market_id": quote["market_id"],
                            "outcome": quote["outcome"],
                            "book_sha256": quote["source"]["book"]["sha256"],
                        }
                    )
                )
                available = _levels(quote["ask_levels"])
                existing = connection.execute(
                    "SELECT payload_json FROM paper_trades WHERE liquidity_key=?", (liquidity_key,)
                ).fetchall()
                used: dict[Decimal, Decimal] = {}
                for row in existing:
                    for fill in json.loads(row[0])["fills"]:
                        price = _decimal(fill["price"], "fill price")
                        used[price] = used.get(price, Decimal("0")) + _decimal(
                            fill["quantity"], "fill quantity"
                        )
                for fill in evaluation["fills"]:
                    price = _decimal(fill["price"], "fill price")
                    used[price] = used.get(price, Decimal("0")) + _decimal(
                        fill["quantity"], "fill quantity"
                    )
                if any(
                    quantity > available.get(price, Decimal("0"))
                    for price, quantity in used.items()
                ):
                    raise ValueError("Displayed snapshot liquidity has already been reserved")
                if max_exposure is not None:
                    limit = _decimal(max_exposure, "max_exposure")
                    if limit <= 0:
                        raise ValueError("max_exposure must be positive")
                    rows = connection.execute(
                        "SELECT t.payload_json FROM paper_trades t LEFT JOIN paper_settlements s "
                        "ON t.paper_trade_id=s.paper_trade_id WHERE s.paper_trade_id IS NULL"
                    ).fetchall()
                    exposure = sum(
                        (_decimal(json.loads(row[0])["total_cost"], "total_cost") for row in rows),
                        Decimal("0"),
                    )
                    if exposure + _decimal(evaluation["total_cost"], "total_cost") > limit:
                        raise ValueError("Paper fill exceeds the explicit open-exposure limit")
                result = {
                    **evaluation,
                    "paper_trade_id": "paper-" + uuid.uuid4().hex,
                    "idempotency_key": idempotency_key,
                    "recorded_at": recorded.isoformat(),
                    "mode": "paper_only",
                    "execution": "simulated_displayed_liquidity",
                    "quote_verification": quote_verification,
                    "liquidity_key": liquidity_key,
                    "fee_kind": "estimated_upper_bound",
                    "max_exposure": str(max_exposure) if max_exposure is not None else None,
                    "forecast": forecast,
                    "quote": quote,
                }
                self._insert(
                    connection,
                    "paper_fill",
                    result,
                    evaluation_id=evaluation_id,
                    forecast_id=evaluation["forecast_id"],
                    quote_id=evaluation["quote_id"],
                    idempotency_key=idempotency_key,
                    liquidity_key=liquidity_key,
                )
                connection.commit()
                return result
            except BaseException:
                connection.rollback()
                raise

    def settle(
        self,
        paper_trade_id: str,
        *,
        settlement_price,
        settled_at,
        evidence: dict,
        idempotency_key: str,
    ) -> dict:
        """Append explicit evidence-backed settlement; cancellation is never a refund default."""
        price = _decimal(settlement_price, "settlement_price")
        if not 0 <= price <= 1:
            raise ValueError("Settlement price must be in [0,1]")
        timestamp = _time(settled_at, "settled_at")
        _text({"idempotency_key": idempotency_key}, "idempotency_key")
        if not isinstance(evidence, dict) or not any(
            isinstance(evidence.get(key), str) and evidence[key].strip()
            for key in ["reference", "source_url"]
        ):
            raise ValueError("Settlement requires an evidence reference")
        observed = _time(evidence.get("observed_at"), "evidence observed_at")
        settlement_type = evidence.get("settlement_type")
        expected_price = {"win": Decimal("1"), "loss": Decimal("0"), "tie": Decimal("0.5")}
        if settlement_type is not None and settlement_type not in {*expected_price, "cancellation"}:
            raise ValueError("Unsupported settlement_type")
        if settlement_type in expected_price and price != expected_price[settlement_type]:
            raise ValueError("Settlement price does not match the declared outcome")
        request = {
            "paper_trade_id": paper_trade_id,
            "settlement_price": str(price),
            "settled_at": timestamp.isoformat(),
            "evidence": evidence,
            "idempotency_key": idempotency_key,
        }
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                previous = connection.execute(
                    "SELECT payload_json FROM paper_settlements WHERE idempotency_key=?",
                    (idempotency_key,),
                ).fetchone()
                if previous is not None:
                    result = json.loads(previous[0])
                    if any(result[key] != value for key, value in request.items()):
                        raise ValueError("Settlement idempotency key has different content")
                    connection.commit()
                    return result
                trade = self._get(connection, "paper_fill", paper_trade_id)
                if connection.execute(
                    "SELECT 1 FROM paper_settlements WHERE paper_trade_id=?", (paper_trade_id,)
                ).fetchone():
                    raise ValueError("Paper position is already settled")
                now = _utcnow()
                kickoff = _time(trade["kickoff"], "kickoff")
                if timestamp < kickoff or observed < kickoff or timestamp > now or observed > now:
                    raise ValueError(
                        "Settlement/evidence must be after kickoff and not in the future"
                    )
                verification = _verify_settlement_evidence(evidence, trade, price)
                payout = _decimal(trade["quantity"], "quantity") * price
                cost = _decimal(trade["total_cost"], "total_cost")
                result = {
                    **request,
                    "settlement_id": "settlement-" + uuid.uuid4().hex,
                    "recorded_at": now.isoformat(),
                    "mode": "paper_only",
                    "evidence_verification": verification,
                    "quantity": trade["quantity"],
                    "notional": trade["notional"],
                    "fee": trade["fee"],
                    "fee_kind": "estimated_upper_bound",
                    "total_cost": trade["total_cost"],
                    "payout": str(payout),
                    "realized_paper_profit": str(payout - cost),
                    "market_id": trade["market_id"],
                    "outcome": trade["outcome"],
                }
                self._insert(
                    connection,
                    "settlement",
                    result,
                    paper_trade_id=paper_trade_id,
                    idempotency_key=idempotency_key,
                )
                connection.commit()
                return result
            except BaseException:
                connection.rollback()
                raise

    def export_events(self) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM ledger_events ORDER BY sequence").fetchall()
        return [{**dict(row), "payload": json.loads(row["payload_json"])} for row in rows]

    def summary(self) -> dict:
        with self._connect() as connection:
            counts = {
                table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table, _ in TABLES.values()
            }
            open_rows = connection.execute(
                "SELECT t.payload_json FROM paper_trades t LEFT JOIN paper_settlements s "
                "ON t.paper_trade_id=s.paper_trade_id WHERE s.paper_trade_id IS NULL"
            ).fetchall()
            settlements = connection.execute(
                "SELECT payload_json FROM paper_settlements"
            ).fetchall()
        return {
            "mode": "paper_only",
            "counts": counts,
            "open_positions": len(open_rows),
            "open_exposure": str(
                sum(
                    (_decimal(json.loads(row[0])["total_cost"], "cost") for row in open_rows),
                    Decimal("0"),
                )
            ),
            "realized_paper_profit": str(
                sum(
                    (
                        _decimal(json.loads(row[0])["realized_paper_profit"], "profit")
                        for row in settlements
                    ),
                    Decimal("0"),
                )
            ),
            "fee_kind": "estimated_upper_bound",
        }

    def audit(
        self, *, verify_artifacts: bool = True, expected_head_hash: str | None = None
    ) -> dict:
        """Check the chain, immutable records, references, calculations, and retained raw files."""
        errors, counts, source_verification = [], {}, {}
        previous, last_time = GENESIS_HASH, None
        with self._connect() as connection:
            if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                errors.append("SQLite integrity check failed")
            if connection.execute("PRAGMA foreign_key_check").fetchall():
                errors.append("Broken foreign-key references")
            rows = connection.execute("SELECT * FROM ledger_events ORDER BY sequence").fetchall()
            seen = set()
            for expected_sequence, row in enumerate(rows, 1):
                try:
                    if row["sequence"] != expected_sequence or row["previous_hash"] != previous:
                        raise ValueError("Event sequence or previous hash mismatch")
                    event = {
                        name: row[name]
                        for name in [
                            "event_id",
                            "event_type",
                            "entity_id",
                            "recorded_at",
                            "previous_hash",
                            "payload_sha256",
                        ]
                    }
                    if _digest(_canonical(event)) != row["event_hash"]:
                        raise ValueError("Event hash mismatch")
                    payload = json.loads(row["payload_json"])
                    if (
                        _canonical(payload) != row["payload_json"]
                        or _digest(row["payload_json"]) != row["payload_sha256"]
                    ):
                        raise ValueError("Event payload hash mismatch")
                    when = _time(row["recorded_at"], "event recorded_at")
                    if last_time is not None and when < last_time:
                        raise ValueError("Event timestamps are not monotonic")
                    kind = row["event_type"]
                    if kind not in TABLES:
                        raise ValueError("Unknown event type")
                    table, key = TABLES[kind]
                    identity = row["entity_id"]
                    if payload.get(key) != identity or (kind, identity) in seen:
                        raise ValueError("Event identity mismatch or duplicate")
                    entity = connection.execute(
                        f"SELECT payload_json,payload_sha256 FROM {table} WHERE {key}=?",
                        (identity,),
                    ).fetchone()
                    if (
                        entity is None
                        or entity["payload_json"] != row["payload_json"]
                        or entity["payload_sha256"] != row["payload_sha256"]
                    ):
                        raise ValueError("Event and entity content differ")
                    if kind == "forecast":
                        _validate_forecast(payload)
                    elif kind == "quote":
                        _validate_quote(payload)
                        source_verification[identity] = (
                            _verify_quote_sources(payload)
                            if verify_artifacts
                            else "artifact_verification_skipped"
                        )
                    elif kind == "evaluation":
                        _reconcile(
                            payload,
                            self._get(connection, "forecast", payload["forecast_id"]),
                            self._get(connection, "quote", payload["quote_id"]),
                        )
                    elif kind == "settlement":
                        trade = self._get(connection, "paper_fill", payload["paper_trade_id"])
                        price = _decimal(payload["settlement_price"], "settlement price")
                        if verify_artifacts:
                            _verify_settlement_evidence(payload["evidence"], trade, price)
                        payout = _decimal(trade["quantity"], "quantity") * price
                        if _decimal(payload["payout"], "payout") != payout or _decimal(
                            payload["realized_paper_profit"], "profit"
                        ) != payout - _decimal(trade["total_cost"], "cost"):
                            raise ValueError("Settlement calculation does not match paper position")
                    seen.add((kind, identity))
                    last_time = when
                except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
                    errors.append(f"Event {row['sequence']}: {exc}")
                previous = row["event_hash"]
            for kind, (table, _) in TABLES.items():
                counts[table] = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                if counts[table] != sum(item[0] == kind for item in seen):
                    errors.append(f"{table} contains records not matched to valid events")
            triggers = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger'"
            ).fetchall()
            names = {row[0] for row in triggers}
            for table in [*[item[0] for item in TABLES.values()], "ledger_events"]:
                if any(
                    f"{table}_no_{action}" not in names
                    for action in ["update", "delete", "replace"]
                ):
                    errors.append(f"Append-only triggers missing for {table}")
        if expected_head_hash is not None and previous != expected_head_hash:
            errors.append("Ledger head differs from external anchor")
        return {
            "valid": not errors,
            "errors": errors,
            "events": len(rows),
            "head_hash": previous,
            "counts": counts,
            "mode": "paper_only",
            "source_verification": source_verification,
        }
