"""Offline checks against the documented public Polymarket US JSON shapes."""

import copy
import hashlib
import json
from pathlib import Path

import pytest
import requests

from predictions.markets import API_BASE, PolymarketUSProvider, normalize_quote


@pytest.fixture
def market():
    return {
        "id": "1458580",
        "slug": "aec-nfl-car-gb-2026-10-29",
        "active": True,
        "closed": False,
        "status": "MARKET_STATUS_OPEN",
        "sportsMarketType": "football_team_full_game_winner",
        "sportsMarketTypeV2": "SPORTS_MARKET_TYPE_MONEYLINE",
        "description": (
            "The market settles to the winner. Overtime is included if played. "
            "If the game ends in a tie, the market will settle to $0.50. "
            "A canceled game settles to the last fair market price."
        ),
        "gameStartTime": "2026-10-30T00:15:00Z",
        "feeCoefficient": 0.0695,
        "minimumTradeQty": 0.01,
        "orderPriceMinTickSize": 0.0025,
        "marketSides": [
            {"long": True, "tradable": True, "team": {"league": "nfl", "abbreviation": "car"}},
            {"long": False, "tradable": True, "team": {"league": "nfl", "abbreviation": "gb"}},
        ],
        # These display values deliberately do NOT describe executable asks.
        "outcomePrices": '["0.99", "0.01"]',
    }


def entry(price, quantity):
    return {"px": {"value": price, "currency": "USD"}, "qty": quantity}


@pytest.fixture
def book(market):
    return {
        "marketSlug": market["slug"],
        "state": "MARKET_STATE_OPEN",
        "transactTime": "2026-10-27T05:45:00Z",
        "offers": [entry("0.36", "5"), entry("0.355", "10")],
        "bids": [entry("0.34", "7"), entry("0.35", "8")],
    }


@pytest.fixture
def source():
    return {
        "market": {"sha256": "a" * 64, "fetched_at": "2026-10-27T05:45:01Z"},
        "book": {"sha256": "b" * 64, "fetched_at": "2026-10-27T05:45:02Z"},
    }


def response(payload=None, status=200):
    value = requests.Response()
    value.status_code = status
    value._content = json.dumps(payload or {}).encode()
    value._content_consumed = True
    value.url = API_BASE + "/synthetic-fixture"
    return value


def test_yes_uses_executable_depth_and_no_complements_bids(market, book, source):
    yes = normalize_quote(market, book, outcome="yes", source=source)
    no = normalize_quote(market, book, outcome="no", source=source)
    assert yes["ask_levels"] == [
        {"price": "0.355", "quantity": "10"},
        {"price": "0.36", "quantity": "5"},
    ]
    assert no["ask_levels"] == [
        {"price": "0.65", "quantity": "8"},
        {"price": "0.66", "quantity": "7"},
    ]
    assert yes["yes_team"] == "CAR" and yes["no_team"] == "GB"
    assert yes["settlement"]["tie_payout"] == "0.5"
    assert yes["settlement"]["cancellation"] == "last_fair_market_price"
    assert (
        yes["settlement"]["terms_sha256"]
        == hashlib.sha256(market["description"].encode()).hexdigest()
    )
    assert yes["fee_schedule"]["theta"] == "0.0695"
    assert yes["quote_id"] != no["quote_id"]


def test_observation_identity_changes_but_liquidity_budget_does_not(market, book, source):
    first = normalize_quote(market, book, outcome="yes", source=source)
    new_source = copy.deepcopy(source)
    new_source["book"]["fetched_at"] = "2026-10-27T05:45:03Z"
    second = normalize_quote(market, book, outcome="yes", source=new_source)
    assert first["quote_id"] != second["quote_id"]
    assert first["liquidity_key"] == second["liquidity_key"]


def test_duplicate_prices_aggregate_depth_and_empty_book_is_explicit(market, book, source):
    book["offers"].append(entry("0.3550", "2.5"))
    quote = normalize_quote(market, book, outcome="yes", source=source)
    assert quote["ask_levels"][0] == {"price": "0.355", "quantity": "12.5"}
    book["offers"] = []
    assert normalize_quote(market, book, outcome="yes", source=source)["ask_levels"] == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("active", False),
        ("closed", True),
        ("status", "MARKET_STATUS_HALTED"),
        ("sportsMarketType", "futures"),
        ("sportsMarketTypeV2", "SPORTS_MARKET_TYPE_SPREAD"),
        ("description", "winner with unknown terms"),
        ("feeCoefficient", None),
        ("feeCoefficient", 0),
        ("feeCoefficient", "NaN"),
        ("minimumTradeQty", 0),
        ("orderPriceMinTickSize", None),
        ("gameStartTime", "2026-10-30T00:15:00"),
    ],
)
def test_unknown_terms_fees_identity_and_closed_markets_fail(market, book, source, field, value):
    market[field] = value
    with pytest.raises(ValueError):
        normalize_quote(market, book, outcome="yes", source=source)


@pytest.mark.parametrize(
    "variant", ["non_nfl", "same_team", "same_side", "not_tradable", "unknown_team"]
)
def test_teams_and_contract_sides_must_be_unambiguous(market, book, source, variant):
    if variant == "non_nfl":
        market["marketSides"][0]["team"]["league"] = "nba"
    elif variant == "same_team":
        market["marketSides"][1]["team"]["abbreviation"] = "car"
    elif variant == "same_side":
        market["marketSides"][1]["long"] = True
    elif variant == "not_tradable":
        market["marketSides"][1]["tradable"] = False
    else:
        market["marketSides"][0]["team"]["abbreviation"] = "unknown"
    with pytest.raises(ValueError):
        normalize_quote(market, book, outcome="yes", source=source)


@pytest.mark.parametrize(
    "variant",
    [
        "slug",
        "closed",
        "missing_time",
        "naive_time",
        "old_fee",
        "negative_qty",
        "currency",
        "crossed",
        "missing_depth",
    ],
)
def test_bad_books_are_not_usable_quotes(market, book, source, variant):
    if variant == "slug":
        book["marketSlug"] = "other-market"
    elif variant == "closed":
        book["state"] = "MARKET_STATE_PREOPEN"
    elif variant == "missing_time":
        book.pop("transactTime")
    elif variant == "naive_time":
        book["transactTime"] = "2026-10-27T05:45:00"
    elif variant == "old_fee":
        book["transactTime"] = "2026-10-18T05:45:00Z"
    elif variant == "negative_qty":
        book["offers"][0]["qty"] = "-1"
    elif variant == "currency":
        book["bids"][0]["px"]["currency"] = "EUR"
    elif variant == "crossed":
        book["bids"][0]["px"]["value"] = "0.9"
    else:
        book.pop("offers")
    with pytest.raises(ValueError):
        normalize_quote(market, book, outcome="yes", source=source)


def test_provider_is_get_only_and_snapshots_exact_raw_responses(
    tmp_path, monkeypatch, market, book
):
    payloads = [{"market": market}, {"marketData": book}]
    requests_seen = []

    def get(url, **kwargs):
        requests_seen.append((url, kwargs))
        result = response(payloads.pop(0))
        result.url = url
        return result

    monkeypatch.setattr("predictions.markets.requests.get", get)
    quote = PolymarketUSProvider(tmp_path).get_quote(market["slug"], "no")
    assert [url for url, _ in requests_seen] == [
        API_BASE + "/v1/market/slug/" + market["slug"],
        API_BASE + "/v1/markets/" + market["slug"] + "/book",
    ]
    for _, kwargs in requests_seen:
        assert kwargs["allow_redirects"] is False
        assert "headers" not in kwargs and "auth" not in kwargs
    for metadata in quote["source"].values():
        content = Path(metadata["path"]).read_bytes()
        assert hashlib.sha256(content).hexdigest() == metadata["sha256"]
        assert json.loads(content)
    assert len(list(tmp_path.glob("*.metadata.json"))) == 2


def test_retry_rate_limit_then_timeout_then_success(tmp_path, monkeypatch):
    limited = response(status=429)
    limited.headers["Retry-After"] = "2"
    actions = [limited, requests.Timeout("temporary"), response({"markets": []})]

    def get(*args, **kwargs):
        action = actions.pop(0)
        if isinstance(action, Exception):
            raise action
        return action

    delays = []
    monkeypatch.setattr("predictions.markets.requests.get", get)
    monkeypatch.setattr("predictions.markets.time.sleep", delays.append)
    assert PolymarketUSProvider(tmp_path, retries=2).list_markets() == []
    assert delays == [2, 2]


def test_permanent_http_error_and_redirect_are_not_retried(tmp_path, monkeypatch):
    calls = []

    def get(*args, **kwargs):
        calls.append(args[0])
        return response(status=404)

    monkeypatch.setattr("predictions.markets.requests.get", get)
    with pytest.raises(requests.HTTPError):
        PolymarketUSProvider(tmp_path).list_markets()
    assert len(calls) == 1
    monkeypatch.setattr("predictions.markets.requests.get", lambda *a, **k: response(status=302))
    with pytest.raises(ValueError, match="Redirect"):
        PolymarketUSProvider(tmp_path).list_markets()


def test_corrupt_immutable_snapshot_is_not_overwritten(tmp_path, monkeypatch):
    result = response({"markets": []})
    path = tmp_path / (hashlib.sha256(result.content).hexdigest() + ".json")
    path.write_text("changed evidence")
    monkeypatch.setattr("predictions.markets.requests.get", lambda *a, **k: result)
    with pytest.raises(ValueError, match="corrupt"):
        PolymarketUSProvider(tmp_path).list_markets()
    assert path.read_text() == "changed evidence"


def test_final_settlement_requires_resolved_market_and_snapshots_evidence(
    tmp_path, monkeypatch, market
):
    market["status"] = "MARKET_STATUS_RESOLVED"
    results = [response({"market": market}), response({"slug": market["slug"], "settlement": 0.5})]
    monkeypatch.setattr("predictions.markets.requests.get", lambda *a, **k: results.pop(0))
    result = PolymarketUSProvider(tmp_path).get_settlement(market["slug"])
    assert result["settlement_price_yes"] == result["settlement_price_no"] == "0.5"
    assert result["market_status"] == "resolved"
    assert result["source"]["settlement"]["sha256"]


def test_provisional_settlement_is_rejected_before_endpoint(tmp_path, monkeypatch, market):
    market["status"] = "MARKET_STATUS_RESOLVING"
    calls = []

    def get(*args, **kwargs):
        calls.append(args[0])
        return response({"market": market})

    monkeypatch.setattr("predictions.markets.requests.get", get)
    with pytest.raises(ValueError, match="finally resolved"):
        PolymarketUSProvider(tmp_path).get_settlement(market["slug"])
    assert len(calls) == 1


def test_nfl_discovery_uses_documented_league_endpoint(tmp_path, monkeypatch, market):
    seen = []

    def get(url, **kwargs):
        seen.append(url)
        return response({"events": [{"markets": [market, {"sportsMarketType": "futures"}]}]})

    monkeypatch.setattr("predictions.markets.requests.get", get)
    assert PolymarketUSProvider(tmp_path).list_nfl_markets() == [market]
    assert seen == [API_BASE + "/v2/leagues/nfl/events"]


@pytest.mark.parametrize(
    "slug", ["https://gamma-api.polymarket.com/markets", "../orders", "foo/bar", "UPPER"]
)
def test_gateway_host_and_paths_cannot_be_overridden(tmp_path, monkeypatch, slug):
    monkeypatch.setattr(
        "predictions.markets.requests.get", lambda *a, **k: pytest.fail("unexpected HTTP")
    )
    with pytest.raises(ValueError):
        PolymarketUSProvider(tmp_path).get_quote(slug)
