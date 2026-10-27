"""Synthetic local paper-journal integration; no exchange executions occur."""

import copy
import hashlib
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from predictions.ev import evaluate_expected_value
from predictions.ledger import PaperLedger


@pytest.fixture
def journal(tmp_path, monkeypatch):
    clock = [datetime(2026, 10, 27, 12, tzinfo=timezone.utc)]
    monkeypatch.setattr("predictions.ledger._utcnow", lambda: clock[0])
    now = clock[0]
    kickoff = now + timedelta(hours=2)
    forecast = {
        "forecast_id": "synthetic-forecast",
        "game_id": "2026_09_CAR_GB",
        "home_team": "CAR",
        "away_team": "GB",
        "kickoff": kickoff.isoformat(),
        "as_of": (now - timedelta(seconds=30)).isoformat(),
        "generated_at": (now - timedelta(seconds=25)).isoformat(),
        "model_id": "synthetic-model",
        "model_version": "v2",
        "model_sha256": "a" * 64,
        "features_sha256": "b" * 64,
        "home_win_probability": 0.75,
        "away_win_probability": 0.25,
        "probability_scope": "conditional_on_decisive_result",
    }
    sources = {}
    for kind in ["market", "book"]:
        path = tmp_path / f"synthetic-{kind}.json"
        path.write_text(json.dumps({"synthetic_fixture": kind}))
        sources[kind] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "url": f"https://gateway.polymarket.us/synthetic-fixture/{kind}",
        }
    quote = {
        "quote_id": "synthetic-quote",
        "market_id": "synthetic-market",
        "market_slug": "synthetic-car-gb",
        "game_id": forecast["game_id"],
        "venue": "polymarket_us",
        "outcome": "yes",
        "yes_team": "CAR",
        "no_team": "GB",
        "game_start_time": kickoff.isoformat(),
        "market_status": "open",
        "quote_at": (now - timedelta(seconds=2)).isoformat(),
        "received_at": (now - timedelta(seconds=1)).isoformat(),
        "ask_levels": [{"price": "0.4", "quantity": "10"}],
        "minimum_trade_quantity": "0.01",
        "price_tick": "0.0025",
        "source": sources,
        "fee_schedule": {
            "type": "polymarket_us_standard_taker",
            "theta": "0.0695",
            "effective_at": "2026-10-19T17:00:00Z",
            "rounding": "ROUND_HALF_EVEN",
            "currency": "USD",
            "source_url": "https://docs.polymarket.us/fees.md",
        },
        "settlement": {
            "rule_type": "nfl_winner_including_overtime",
            "win_payout": "1",
            "loss_payout": "0",
            "tie_payout": "0.5",
            "cancellation": "last_fair_market_price",
            "description": "Synthetic NFL contract includes overtime and half-dollar tie payout.",
            "rules_url": "https://docs.polymarket.us/faqs/sports-faqs.md",
            "terms_sha256": "c" * 64,
        },
    }
    quote["settlement"]["terms_sha256"] = hashlib.sha256(
        quote["settlement"]["description"].encode()
    ).hexdigest()
    ledger = PaperLedger(tmp_path / "paper.sqlite")

    def evaluate(quantity=10, selected_quote=None, **kwargs):
        return evaluate_expected_value(
            forecast,
            selected_quote or quote,
            selected_team="CAR",
            quantity=quantity,
            as_of=clock[0],
            tie_probability="0.02",
            completion_probability="1",
            **kwargs,
        )

    def record(evaluation=None):
        ledger.record_forecast(forecast)
        ledger.record_quote(quote)
        value = evaluation if evaluation is not None else evaluate()
        ledger.record_evaluation(value)
        return value

    return ledger, forecast, quote, evaluate, record, clock


def test_persist_reopen_hash_chain_and_paper_summary(journal):
    ledger, _, _, _, record, _ = journal
    evaluation = record()
    trade = ledger.paper_fill(evaluation["evaluation_id"], idempotency_key="fill-1")
    assert trade["mode"] == "paper_only"
    assert trade["execution"] == "simulated_displayed_liquidity"
    assert trade["fee_kind"] == "estimated_upper_bound"
    assert Decimal(trade["total_cost"]) == Decimal("4.17")
    reopened = PaperLedger(ledger.path)
    audit = reopened.audit()
    assert audit["valid"] and audit["events"] == 4
    assert [event["event_type"] for event in reopened.export_events()] == [
        "forecast",
        "quote",
        "evaluation",
        "paper_fill",
    ]
    assert reopened.summary()["open_exposure"] == "4.17"
    assert reopened.summary()["realized_paper_profit"] == "0"
    assert reopened.audit(expected_head_hash=audit["head_hash"])["valid"]
    assert not reopened.audit(expected_head_hash="0" * 64)["valid"]


def test_records_idempotent_but_conflicting_identity_rejected(journal):
    ledger, forecast, quote, _, record, _ = journal
    evaluation = record()
    assert ledger.record_forecast(forecast) == forecast
    assert ledger.record_quote(quote) == quote
    assert ledger.record_evaluation(evaluation) == evaluation
    assert ledger.audit()["events"] == 3
    changed = copy.deepcopy(forecast)
    changed["home_win_probability"], changed["away_win_probability"] = 0.8, 0.2
    with pytest.raises(ValueError, match="different content"):
        ledger.record_forecast(changed)
    assert ledger.audit()["events"] == 3


def test_evaluation_requires_registered_references_and_rolls_back(journal):
    ledger, _, _, evaluate, _, _ = journal
    with pytest.raises(ValueError, match="Unknown forecast"):
        ledger.record_evaluation(evaluate())
    assert ledger.audit()["valid"] and ledger.audit()["events"] == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "pass"),
        ("mode", "live"),
        ("notional", "0"),
        ("fee", "0"),
        ("total_cost", "0"),
        ("expected_profit", "100000"),
        ("fills", []),
        ("source", {}),
        ("evaluation_id", "forged-id"),
    ],
)
def test_tampered_ev_rejected_before_recording(journal, field, value):
    ledger, forecast, quote, evaluate, _, _ = journal
    ledger.record_forecast(forecast)
    ledger.record_quote(quote)
    evaluation = evaluate()
    evaluation[field] = value
    with pytest.raises(ValueError, match="does not match"):
        ledger.record_evaluation(evaluation)
    assert ledger.audit()["valid"] and ledger.audit()["events"] == 2


def test_custom_forecast_age_reconciles_and_rechecked_when_filling(journal):
    ledger, _, _, evaluate, record, clock = journal
    evaluation = record(evaluate(max_forecast_age_seconds=31))
    clock[0] += timedelta(seconds=2)
    with pytest.raises(ValueError, match="Forecast is stale"):
        ledger.paper_fill(evaluation["evaluation_id"], idempotency_key="expired-forecast")
    assert ledger.audit()["events"] == 3


def test_fill_idempotency_does_not_reserve_again(journal):
    ledger, _, _, _, record, clock = journal
    evaluation = record()
    first = ledger.paper_fill(evaluation["evaluation_id"], idempotency_key="once")
    clock[0] += timedelta(days=1)
    assert ledger.paper_fill(evaluation["evaluation_id"], idempotency_key="once") == first
    assert ledger.audit()["events"] == 4
    with pytest.raises(ValueError, match="another evaluation"):
        ledger.paper_fill("other", idempotency_key="once")


def test_repoll_same_book_cannot_reuse_reserved_liquidity(journal):
    ledger, _, quote, evaluate, record, clock = journal
    first = record()
    ledger.paper_fill(first["evaluation_id"], idempotency_key="first")
    second_quote = copy.deepcopy(quote)
    second_quote["quote_id"] = "new-observation-of-same-book"
    second_quote["received_at"] = clock[0].isoformat()
    ledger.record_quote(second_quote)
    second = evaluate(selected_quote=second_quote)
    ledger.record_evaluation(second)
    with pytest.raises(ValueError, match="already been reserved"):
        ledger.paper_fill(second["evaluation_id"], idempotency_key="second")
    assert ledger.summary()["open_positions"] == 1
    assert ledger.audit()["valid"]


def test_reservations_are_by_price_not_only_total_depth(journal):
    ledger, _, quote, evaluate, record, _ = journal
    quote["ask_levels"] = [{"price": "0.4", "quantity": "10"}, {"price": "0.5", "quantity": "20"}]
    evaluation = record(evaluate())
    ledger.paper_fill(evaluation["evaluation_id"], idempotency_key="cheap")
    with pytest.raises(ValueError, match="reserved"):
        ledger.paper_fill(evaluation["evaluation_id"], idempotency_key="cheap-again")
    assert ledger.summary()["open_positions"] == 1


def test_fractional_depth_reserves_without_float_rounding(journal):
    ledger, _, quote, evaluate, record, _ = journal
    quote["ask_levels"] = [{"price": "0.4", "quantity": "0.6"}, {"price": "0.5", "quantity": "0.4"}]
    evaluation = record(evaluate(quantity=1))
    trade = ledger.paper_fill(evaluation["evaluation_id"], idempotency_key="fractional")
    assert [fill["quantity"] for fill in trade["fills"]] == ["0.6", "0.4"]
    with pytest.raises(ValueError, match="reserved"):
        ledger.paper_fill(evaluation["evaluation_id"], idempotency_key="too-much")


def test_concurrent_writers_cannot_overreserve_snapshot(journal):
    ledger, _, _, _, record, _ = journal
    evaluation = record()
    peers = [PaperLedger(ledger.path), PaperLedger(ledger.path)]

    def fill(index):
        try:
            peers[index].paper_fill(evaluation["evaluation_id"], idempotency_key=f"race-{index}")
            return "filled"
        except ValueError as exc:
            assert "reserved" in str(exc)
            return "rejected"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(fill, [0, 1]))
    assert sorted(results) == ["filled", "rejected"]
    assert ledger.summary()["open_positions"] == 1
    assert ledger.audit()["valid"]


@pytest.mark.parametrize("which", ["stale", "started", "backdated", "future-dated"])
def test_stale_started_and_falsified_time_do_not_fill(journal, which):
    ledger, _, _, _, record, clock = journal
    evaluation = record()
    kwargs = {}
    if which == "stale":
        clock[0] += timedelta(seconds=61)
    elif which == "started":
        clock[0] += timedelta(hours=3)
    else:
        offset = -60 if which == "backdated" else 60
        kwargs["recorded_at"] = clock[0] + timedelta(seconds=offset)
    with pytest.raises(ValueError):
        ledger.paper_fill(evaluation["evaluation_id"], idempotency_key=which, **kwargs)
    assert ledger.audit()["events"] == 3
    assert ledger.summary()["open_positions"] == 0


def test_pass_and_explicit_exposure_limit_block_fill(journal):
    ledger, _, _, evaluate, record, _ = journal
    passed = record(evaluate(min_ev_per_contract="1"))
    with pytest.raises(ValueError, match="paper_candidate"):
        ledger.paper_fill(passed["evaluation_id"], idempotency_key="pass")
    candidate = evaluate()
    ledger.record_evaluation(candidate)
    with pytest.raises(ValueError, match="exposure limit"):
        ledger.paper_fill(candidate["evaluation_id"], idempotency_key="budget", max_exposure="4")
    assert ledger.summary()["open_positions"] == 0


def _settle(ledger, trade, clock, **kwargs):
    clock[0] += timedelta(hours=4)
    evidence = {
        "reference": "synthetic-official-score-fixture",
        "observed_at": clock[0].isoformat(),
        "settlement_type": "tie",
    }
    return ledger.settle(
        trade["paper_trade_id"],
        settlement_price="0.5",
        settled_at=clock[0],
        evidence=evidence,
        idempotency_key="settlement",
        **kwargs,
    )


def test_half_dollar_tie_settlement_and_net_fees_are_appended(journal):
    ledger, _, _, _, record, clock = journal
    evaluation = record()
    trade = ledger.paper_fill(evaluation["evaluation_id"], idempotency_key="trade")
    settled = _settle(ledger, trade, clock)
    assert settled["payout"] == "5.0"
    assert Decimal(settled["realized_paper_profit"]) == Decimal("0.83")
    assert settled["evidence_verification"] == "operator_supplied"
    assert ledger.summary()["open_positions"] == 0
    assert Decimal(ledger.summary()["realized_paper_profit"]) == Decimal("0.83")
    assert ledger.audit()["valid"] and ledger.audit()["events"] == 5
    assert (
        ledger.settle(
            trade["paper_trade_id"],
            settlement_price="0.5",
            settled_at=clock[0],
            evidence=settled["evidence"],
            idempotency_key="settlement",
        )
        == settled
    )
    with pytest.raises(ValueError, match="already settled"):
        ledger.settle(
            trade["paper_trade_id"],
            settlement_price="1",
            settled_at=clock[0],
            evidence={**settled["evidence"], "settlement_type": "win"},
            idempotency_key="again",
        )


def test_provider_evidence_cannot_override_market_or_outcome_price(journal):
    ledger, _, quote, _, record, clock = journal
    evaluation = record()
    trade = ledger.paper_fill(evaluation["evaluation_id"], idempotency_key="trade")
    clock[0] += timedelta(hours=4)
    evidence = {
        "venue": "polymarket_us",
        "market_id": quote["market_id"],
        "market_slug": quote["market_slug"],
        "market_status": "resolved",
        "settlement_price_yes": "0.5",
        "settlement_price_no": "0.5",
        "source": {"settlement": {"sha256": "e" * 64}},
        "reference": "synthetic-provider-result",
        "observed_at": clock[0].isoformat(),
    }
    for price, changed in [("1", evidence), ("0.5", {**evidence, "market_id": "different"})]:
        with pytest.raises(ValueError):
            ledger.settle(
                trade["paper_trade_id"],
                settlement_price=price,
                settled_at=clock[0],
                evidence=changed,
                idempotency_key="invalid",
            )
    result = ledger.settle(
        trade["paper_trade_id"],
        settlement_price="0.5",
        settled_at=clock[0],
        evidence=evidence,
        idempotency_key="valid",
    )
    assert result["evidence_verification"] == "provider_raw_snapshots_not_retained"
    assert ledger.audit()["valid"]


@pytest.mark.parametrize(
    "variant", ["no_reference", "no_timestamp", "before_kickoff", "future", "wrong_tie_price"]
)
def test_invalid_settlement_evidence_rolls_back(journal, variant):
    ledger, _, _, _, record, clock = journal
    evaluation = record()
    trade = ledger.paper_fill(evaluation["evaluation_id"], idempotency_key="trade")
    kickoff = datetime.fromisoformat(trade["kickoff"])
    clock[0] = kickoff + timedelta(hours=4)
    evidence = {
        "reference": "synthetic-source",
        "observed_at": clock[0].isoformat(),
        "settlement_type": "tie",
    }
    timestamp, price = clock[0], "0.5"
    if variant == "no_reference":
        evidence.pop("reference")
    elif variant == "no_timestamp":
        evidence.pop("observed_at")
    elif variant == "before_kickoff":
        timestamp = kickoff - timedelta(seconds=1)
    elif variant == "future":
        timestamp += timedelta(seconds=1)
    else:
        price = "1"
    with pytest.raises(ValueError):
        ledger.settle(
            trade["paper_trade_id"],
            settlement_price=price,
            settled_at=timestamp,
            evidence=evidence,
            idempotency_key="invalid",
        )
    assert ledger.audit()["events"] == 4
    assert ledger.summary()["open_positions"] == 1


def test_update_delete_protection_and_tamper_detection(journal):
    ledger, _, _, _, record, clock = journal
    evaluation = record()
    trade = ledger.paper_fill(evaluation["evaluation_id"], idempotency_key="protected-trade")
    _settle(ledger, trade, clock)
    with sqlite3.connect(ledger.path) as connection:
        for table in [
            "paper_forecasts",
            "paper_quotes",
            "paper_evaluations",
            "paper_trades",
            "paper_settlements",
            "ledger_events",
        ]:
            for operation in [f"UPDATE {table} SET payload_json='{{}}'", f"DELETE FROM {table}"]:
                with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                    connection.execute(operation)
        connection.execute("DROP TRIGGER paper_forecasts_no_update")
        connection.execute("UPDATE paper_forecasts SET payload_json='{}'")
    audit = ledger.audit()
    assert not audit["valid"]
    assert any("Event and entity content differ" in error for error in audit["errors"])
    assert any("triggers missing" in error for error in audit["errors"])


def test_missing_raw_snapshot_fails_artifact_audit_only(journal):
    ledger, _, quote, _, record, _ = journal
    _provider_quote(journal)
    record()
    from pathlib import Path

    Path(quote["source"]["book"]["path"]).unlink()
    assert not ledger.audit()["valid"]
    assert ledger.audit(verify_artifacts=False)["valid"]


def _provider_quote(journal):
    """Retain genuine documented response shapes for raw-binding regression tests."""
    from predictions.markets import normalize_quote

    ledger, forecast, quote, _, _, clock = journal
    slug = "synthetic-car-gb"
    market = {
        "id": "synthetic-market",
        "slug": slug,
        "active": True,
        "closed": False,
        "status": "MARKET_STATUS_OPEN",
        "sportsMarketType": "football_team_full_game_winner",
        "sportsMarketTypeV2": "SPORTS_MARKET_TYPE_MONEYLINE",
        "description": "Overtime is included. A tie will settle to $0.50. Cancellation uses last fair market price.",
        "gameStartTime": forecast["kickoff"],
        "feeCoefficient": "0.0695",
        "minimumTradeQty": "0.01",
        "orderPriceMinTickSize": "0.0025",
        "marketSides": [
            {"long": True, "tradable": True, "team": {"league": "nfl", "abbreviation": "car"}},
            {"long": False, "tradable": True, "team": {"league": "nfl", "abbreviation": "gb"}},
        ],
    }
    book = {
        "marketSlug": slug,
        "state": "MARKET_STATE_OPEN",
        "transactTime": quote["quote_at"],
        "offers": [{"px": {"value": "0.4", "currency": "USD"}, "qty": "10"}],
        "bids": [{"px": {"value": "0.39", "currency": "USD"}, "qty": "10"}],
    }
    sources = {}
    for kind, body in [("market", {"market": market}), ("book", {"marketData": book})]:
        path = ledger.path.parent / f"retained-{kind}.json"
        path.write_text(json.dumps(body))
        sources[kind] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "fetched_at": (clock[0] - timedelta(seconds=1)).isoformat(),
            "url": f"https://gateway.polymarket.us/synthetic/{kind}",
        }
    quote.clear()
    quote.update(normalize_quote(market, book, outcome="yes", source=sources))
    quote["game_id"] = forecast["game_id"]
    return quote


def test_retained_quote_is_reconciled_before_fill_and_audit(journal):
    ledger, _, _, _, record, _ = journal
    quote = _provider_quote(journal)
    evaluation = record()
    trade = ledger.paper_fill(evaluation["evaluation_id"], idempotency_key="verified-book")
    assert trade["quote_verification"] == "raw_snapshots_reconciled"
    assert ledger.audit()["source_verification"][quote["quote_id"]] == "raw_snapshots_reconciled"


@pytest.mark.parametrize(
    "field,value", [("ask_levels", [{"price": "0.1", "quantity": "1000"}]), ("yes_team", "NE")]
)
def test_normalized_quote_cannot_be_edited_while_retaining_raw_proof(journal, field, value):
    ledger, _, _, _, _, _ = journal
    quote = _provider_quote(journal)
    quote[field] = value
    with pytest.raises(ValueError, match="retained raw snapshots"):
        ledger.record_quote(quote)
    assert ledger.audit()["events"] == 0


def test_raw_file_change_after_evaluation_prevents_paper_fill(journal):
    ledger, _, _, _, record, _ = journal
    quote = _provider_quote(journal)
    evaluation = record()
    from pathlib import Path

    Path(quote["source"]["book"]["path"]).write_text("{}")
    with pytest.raises(ValueError, match="checksum differs"):
        ledger.paper_fill(evaluation["evaluation_id"], idempotency_key="corrupt-book")
    assert not ledger.audit()["valid"]


def test_retained_provider_settlement_rejects_edited_normalized_price(journal):
    ledger, _, quote, _, record, clock = journal
    evaluation = record()
    trade = ledger.paper_fill(evaluation["evaluation_id"], idempotency_key="settle-proof")
    clock[0] += timedelta(hours=4)
    market = {
        "id": quote["market_id"],
        "slug": quote["market_slug"],
        "status": "MARKET_STATUS_RESOLVED",
        "sportsMarketType": "football_team_full_game_winner",
        "sportsMarketTypeV2": "SPORTS_MARKET_TYPE_MONEYLINE",
    }
    evidence = {
        "venue": "polymarket_us",
        "market_id": quote["market_id"],
        "market_slug": quote["market_slug"],
        "market_status": "resolved",
        "settlement_price_yes": "0.5",
        "settlement_price_no": "0.5",
        "observed_at": clock[0].isoformat(),
        "reference": "synthetic-public-result",
        "source": {},
    }
    for kind, body in [
        ("market", {"market": market}),
        ("settlement", {"slug": quote["market_slug"], "settlement": "0.5"}),
    ]:
        path = ledger.path.parent / f"settlement-{kind}.json"
        path.write_text(json.dumps(body))
        evidence["source"][kind] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "fetched_at": clock[0].isoformat(),
        }
    tampered = {**evidence, "settlement_price_yes": "1", "settlement_price_no": "0"}
    with pytest.raises(ValueError, match="retained raw snapshots"):
        ledger.settle(
            trade["paper_trade_id"],
            settlement_price="1",
            settled_at=clock[0],
            evidence=tampered,
            idempotency_key="fake-win",
        )
    result = ledger.settle(
        trade["paper_trade_id"],
        settlement_price="0.5",
        settled_at=clock[0],
        evidence=evidence,
        idempotency_key="actual-half",
    )
    assert result["evidence_verification"] == "public_us_settlement_snapshot"
    assert ledger.audit()["valid"]


def test_clock_rollback_is_not_silently_recorded(journal):
    ledger, forecast, _, _, record, clock = journal
    record()
    clock[0] -= timedelta(seconds=1)
    other = {**forecast, "forecast_id": "another-forecast"}
    with pytest.raises(ValueError, match="clock moved backwards"):
        ledger.record_forecast(other)
    assert ledger.audit()["events"] == 3


def test_insert_or_replace_cannot_bypass_append_only_protection(journal):
    ledger, _, _, _, record, _ = journal
    record()
    with sqlite3.connect(ledger.path) as connection:
        connection.execute("PRAGMA recursive_triggers = OFF")
        assert connection.execute("PRAGMA recursive_triggers").fetchone()[0] == 0
        for table in ["paper_forecasts", "paper_quotes", "paper_evaluations", "ledger_events"]:
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                connection.execute(f"INSERT OR REPLACE INTO {table} SELECT * FROM {table}")
    assert ledger.audit()["valid"]


def test_fill_timestamp_cannot_precede_information_even_within_clock_tolerance(journal):
    ledger, _, _, _, record, clock = journal
    evaluation = record()
    with pytest.raises(ValueError, match="information used"):
        ledger.paper_fill(
            evaluation["evaluation_id"],
            idempotency_key="too-early",
            recorded_at=clock[0] - timedelta(seconds=1),
        )
    assert ledger.audit()["events"] == 3
