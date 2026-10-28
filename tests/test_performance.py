"""Prospective accounting on synthetic ledger forecasts and observed scores."""

import hashlib
import json
import sqlite3

import pandas as pd
import pytest

from predictions.ev import evaluate_expected_value
from predictions.ledger import PaperLedger
from predictions.performance import PerformanceTracker

VERSION = "logistic-regression-v2"


@pytest.fixture
def research(tmp_path, monkeypatch):
    clock = [pd.Timestamp("2026-10-20T12:00:00Z")]
    monkeypatch.setattr("predictions.ledger._utcnow", lambda: clock[0].to_pydatetime())
    ledger = PaperLedger(tmp_path / "paper.sqlite")
    tracker = PerformanceTracker(tmp_path / "performance.sqlite")
    games = pd.DataFrame(
        [
            {
                "game_id": f"synthetic-{number}",
                "season": 2026,
                "week": 8 + number,
                "home_team": "CAR",
                "away_team": "GB",
                "kickoff": pd.Timestamp(day),
                "result_available_at": pd.Timestamp(day) + pd.Timedelta(days=1),
                "home_score": home,
                "away_score": away,
            }
            for number, day, home, away in [
                (1, "2026-10-21T18:00:00Z", 28, 17),
                (2, "2026-10-22T18:00:00Z", 24, 24),
                (3, "2026-10-30T18:00:00Z", None, None),
            ]
        ]
    )

    def forecast(number=1, probability=0.2, identity=None, version=VERSION):
        game = games.loc[games["game_id"].eq(f"synthetic-{number}")].iloc[0]
        row = {
            "forecast_id": identity or f"forecast-{number}-{version}-{probability}",
            "game_id": game["game_id"],
            "season": 2026,
            "home_team": "CAR",
            "away_team": "GB",
            "kickoff": game["kickoff"].isoformat(),
            "as_of": (clock[0] - pd.Timedelta(seconds=1)).isoformat(),
            "generated_at": clock[0].isoformat(),
            "model_id": "synthetic-model",
            "model_version": version,
            "model_sha256": "a" * 64,
            "features_sha256": "b" * 64,
            "home_win_probability": probability,
            "away_win_probability": 1 - probability,
            "probability_scope": "conditional_on_decisive_result",
        }
        ledger.record_forecast(row)
        return row

    def source(stamp="2026-10-24T11:00:00Z", digest="c" * 64):
        return {
            "fetched_at": stamp,
            "sha256": digest,
            "source_url": "https://example.test/synthetic-schedule.csv",
        }

    def update(frame=None, as_of="2026-10-24T12:00:00Z", metadata=None):
        return tracker.update(
            ledger.path,
            games if frame is None else frame,
            as_of=as_of,
            source_metadata=metadata or source(),
        )

    return {
        "clock": clock,
        "ledger": ledger,
        "tracker": tracker,
        "games": games,
        "forecast": forecast,
        "source": source,
        "update": update,
    }


def test_first_issued_forecast_not_later_favorable_prediction(research):
    first = research["forecast"]()
    research["clock"][0] += pd.Timedelta(hours=1)
    research["forecast"](probability=0.95, identity="later-good-forecast")
    before = hashlib.sha256(research["ledger"].path.read_bytes()).hexdigest()
    report = research["update"]()
    group = report["models"][VERSION]
    assert group["n"] == 1
    assert group["metrics"]["accuracy"] == 0
    assert group["metrics"]["brier_score"] == pytest.approx(0.64)
    assert group["selected_forecast_ids"] == [first["forecast_id"]]
    assert group["metrics"]["confidence_intervals_95"] is None
    assert group["metrics"]["uncertainty_status"] == "insufficient_observed_blocks"
    assert hashlib.sha256(research["ledger"].path.read_bytes()).hexdigest() == before


def test_polling_does_not_inflate_cohort_and_models_remain_separate(research):
    research["forecast"]()
    research["forecast"](probability=0.8, version="logistic-regression-v1")
    first = research["update"]()
    second = research["update"]()
    assert first == second
    assert len(first["models"]) == 2
    assert first["models"][VERSION]["n"] == 1
    assert first["models"]["logistic-regression-v1"]["metrics"]["accuracy"] == 1
    with sqlite3.connect(research["tracker"].path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM performance_outcomes").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM performance_reports").fetchone()[0] == 1


def test_ties_and_pending_games_have_null_metrics_not_losses(research):
    research["forecast"](2)
    research["forecast"](3)
    report = research["update"]()
    model = report["models"][VERSION]
    assert model["n"] == 0 and model["metrics"] is None
    assert model["ties"] == 1 and model["pending"] == 1
    assert report["status"] == "no_scored_games"


def test_metrics_intervals_require_at_least_two_observed_week_blocks(research):
    research["forecast"](1, probability=0.2)
    research["forecast"](2, probability=0.8)
    games = research["games"].copy()
    games.loc[1, "away_score"] = 20
    metrics = research["update"](games)["models"][VERSION]["metrics"]
    assert metrics["games"] == 2 and metrics["accuracy"] == 0.5
    assert metrics["bootstrap_clusters"] == 2
    assert metrics["confidence_intervals_95"]["accuracy"] == [0, 1]
    assert metrics["uncertainty_status"] == "conditional_week_block_bootstrap"


def test_missing_ledger_creates_no_fabricated_scores_or_ledger(tmp_path, research):
    missing = tmp_path / "missing.sqlite"
    result = research["tracker"].update(
        missing, research["games"], as_of="2026-10-24T12:00Z", source_metadata=research["source"]()
    )
    assert result["models"] == {} and result["status"] == "no_scored_games"
    assert result["paper"]["settled_positions"] == 0
    assert result["paper"]["mark_to_market"] is None
    assert not missing.exists()


def test_future_observed_source_is_rejected(research):
    research["forecast"]()
    with pytest.raises(ValueError, match="observed after"):
        research["update"](metadata=research["source"]("2026-10-25T00:00Z"))


def test_cached_live_scores_cannot_mature_by_waiting(research):
    research["forecast"]()
    report = research["update"](metadata=research["source"]("2026-10-21T19:00Z"))
    assert report["models"][VERSION]["n"] == 0
    assert report["models"][VERSION]["pending"] == 1
    assert report["outcomes"]["observation_count"] == 0


def test_exact_availability_cutoff_is_excluded(research):
    research["forecast"]()
    result = research["update"](
        as_of="2026-10-22T18:00Z", metadata=research["source"]("2026-10-22T18:00Z")
    )
    assert result["models"][VERSION]["n"] == 0


def test_source_identity_mismatch_never_scores_existing_outcome(research):
    research["forecast"]()
    research["update"]()
    changed = research["games"].copy()
    changed.loc[0, "home_team"] = "BUF"
    report = research["update"](
        changed,
        as_of="2026-10-24T14:00Z",
        metadata=research["source"]("2026-10-24T13:00Z", "d" * 64),
    )
    assert report["models"][VERSION]["n"] == 0
    assert report["models"][VERSION]["identity_mismatches"] == 1


def test_score_corrections_append_and_stale_sources_cannot_undo_them(research):
    research["forecast"]()
    old = research["update"]()
    corrected = research["games"].copy()
    corrected.loc[0, ["home_score", "away_score"]] = [10, 17]
    new = research["update"](
        corrected,
        as_of="2026-10-24T14:00Z",
        metadata=research["source"]("2026-10-24T13:00Z", "d" * 64),
    )
    assert old["models"][VERSION]["metrics"]["accuracy"] == 0
    assert new["models"][VERSION]["metrics"]["accuracy"] == 1
    assert new["outcomes"]["correction_count"] == 1
    stale = research["update"](as_of="2026-10-24T15:00Z")
    assert stale["models"][VERSION]["metrics"]["accuracy"] == 1
    with sqlite3.connect(research["tracker"].path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM performance_outcomes").fetchone()[0] == 2
        stored = json.loads(
            connection.execute(
                "SELECT payload_json FROM performance_reports WHERE report_id=?",
                (old["report_id"],),
            ).fetchone()[0]
        )
        assert stored == old


def test_withdrawn_result_is_unresolved_not_a_reused_old_score(research):
    research["forecast"]()
    research["update"]()
    corrected = research["games"].copy()
    corrected.loc[0, ["home_score", "away_score"]] = float("nan")
    report = research["update"](
        corrected,
        as_of="2026-10-24T14:00Z",
        metadata=research["source"]("2026-10-24T13:00Z", "d" * 64),
    )
    assert report["models"][VERSION]["n"] == 0
    assert report["models"][VERSION]["pending"] == 1
    assert report["outcomes"]["correction_count"] == 1


def test_postgame_import_cannot_claim_prospective_forecast(research):
    research["clock"][0] = pd.Timestamp("2026-10-23T12:00Z")
    game = research["games"].iloc[0]
    fake = {
        "forecast_id": "backdated-import",
        "game_id": game["game_id"],
        "season": 2026,
        "home_team": "CAR",
        "away_team": "GB",
        "kickoff": game["kickoff"].isoformat(),
        "as_of": "2026-10-20T12:00Z",
        "generated_at": "2026-10-20T12:00Z",
        "model_id": "synthetic",
        "model_version": VERSION,
        "model_sha256": "a" * 64,
        "features_sha256": "b" * 64,
        "home_win_probability": 0.9,
        "away_win_probability": 0.1,
        "probability_scope": "conditional_on_decisive_result",
    }
    research["ledger"].record_forecast(fake)
    result = research["update"]()
    assert result["models"] == {}
    assert "recorded_pregame" in result["excluded_forecasts"][0]["reason"]


def test_metric_history_update_delete_and_replace_are_blocked(research):
    research["forecast"]()
    report = research["update"]()
    with sqlite3.connect(research["tracker"].path) as connection:
        for command in [
            "UPDATE performance_reports SET as_of='tamper'",
            "DELETE FROM performance_outcomes",
            "INSERT OR REPLACE INTO performance_reports SELECT * FROM performance_reports",
        ]:
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                connection.execute(command)
    assert research["tracker"].latest_report() == report


@pytest.mark.parametrize(
    "price,kind,profit,drawdown",
    [
        ("0.5", "tie", "1.85", "0"),
        ("1", "win", "6.85", "0"),
        ("0", "loss", "-3.15", "3.15"),
    ],
)
def test_paper_results_require_evidenced_settlement_and_keep_fee_label(
    research, price, kind, profit, drawdown
):
    forecast = research["forecast"](probability=0.9)
    now = research["clock"][0]
    description = "Synthetic terms; overtime included and ties pay half."
    quote = {
        "quote_id": "synthetic-quote",
        "market_id": "synthetic-market",
        "market_slug": "synthetic",
        "yes_team": "CAR",
        "no_team": "GB",
        "venue": "polymarket_us",
        "outcome": "yes",
        "market_status": "open",
        "game_start_time": forecast["kickoff"],
        "quote_at": (now - pd.Timedelta(seconds=1)).isoformat(),
        "received_at": now.isoformat(),
        "ask_levels": [{"price": "0.3", "quantity": "10"}],
        "minimum_trade_quantity": "1",
        "price_tick": "0.01",
        "source": {"book": {"sha256": "c" * 64}},
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
            "rules_url": "https://docs.polymarket.us/faqs/sports-faqs.md",
            "description": description,
            "terms_sha256": hashlib.sha256(description.encode()).hexdigest(),
        },
    }
    ledger = research["ledger"]
    ledger.record_quote(quote)
    evaluation = evaluate_expected_value(
        forecast,
        quote,
        selected_team="CAR",
        quantity=10,
        as_of=now,
        tie_probability="0.01",
        completion_probability="1",
    )
    ledger.record_evaluation(evaluation)
    trade = ledger.paper_fill(evaluation["evaluation_id"], idempotency_key="synthetic-fill")
    opened = research["update"]()
    assert opened["paper"]["open_positions"] == 1
    assert opened["paper"]["realized_paper_profit"] == "0"
    research["clock"][0] = pd.Timestamp("2026-10-24T12:00Z")
    ledger.settle(
        trade["paper_trade_id"],
        settlement_price=price,
        settled_at=research["clock"][0],
        idempotency_key="synthetic-settle",
        evidence={
            "reference": "https://example.test/synthetic-settlement",
            "observed_at": research["clock"][0].isoformat(),
            "settlement_type": kind,
        },
    )
    result = research["update"]()
    assert result["paper"]["open_positions"] == 0
    assert result["paper"]["realized_paper_profit"] == profit
    assert result["paper"]["max_drawdown"] == drawdown
    assert result["paper"]["settled_positions"] == 1
    assert result["paper"]["settlement_rows"][0]["evidence_verification"] == "operator_supplied"
    assert result["paper"]["fee_kind"] == "estimated_upper_bound"
    assert opened["report_id"] != result["report_id"]


def test_tampered_ledger_is_not_used_for_metrics(research):
    research["forecast"]()
    with sqlite3.connect(research["ledger"].path) as connection:
        connection.execute("DROP TRIGGER paper_forecasts_no_update")
    with pytest.raises(ValueError, match="invalid paper ledger"):
        research["update"]()
