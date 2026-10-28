"""Operational scheduler tests use synthetic schedules and injected data sources."""

import copy
import fcntl
import hashlib
import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import requests

from predictions.config import default_config, validate_config
from predictions.ev import evaluate_expected_value
from predictions.ledger import PaperLedger
from predictions.research import _PublicServices, run_research_cycle, run_worker


def test_default_public_service_clock_and_fixed_fallback(monkeypatch):
    from predictions.research import _clock

    now = pd.Timestamp("2026-10-28T12:00:00Z")
    monkeypatch.setattr("predictions.research._now", lambda: now)
    assert _clock(SimpleNamespace(), None)() == now
    assert _clock({}, now)() == now
    later = now + pd.Timedelta(seconds=1)
    assert _clock({"clock": lambda: later}, now)() == later


@pytest.fixture
def setup(tmp_path, monkeypatch):
    clock = [pd.Timestamp("2026-10-28T12:01:00Z")]
    value = default_config()
    value["model_version"] = "v1"
    value["bootstrap_samples"] = 20
    config = validate_config(value, base_dir=tmp_path)
    games = pd.DataFrame(
        [
            {
                "game_id": "synthetic-upcoming",
                "season": 2026,
                "kickoff": clock[0] + pd.Timedelta(days=2),
                "kickoff_time_imputed": False,
                "home_team": "BUF",
                "away_team": "NE",
                "home_score": float("nan"),
                "away_score": float("nan"),
                "result_available_at": clock[0] + pd.Timedelta(days=3),
            }
        ]
    )
    calls = {key: 0 for key in ["schedule", "forecast", "markets", "quote", "performance"]}
    monkeypatch.setattr("predictions.ledger._utcnow", lambda: clock[0].to_pydatetime())

    def schedule():
        calls["schedule"] += 1
        return games.copy(), {"sha256": "c" * 64, "fetched_at": clock[0].isoformat()}

    def forecast(ids, supplied):
        calls["forecast"] += 1
        assert ids == ["synthetic-upcoming"]
        assert supplied["bootstrap_samples"] == 20
        return {
            "forecasts": [
                {
                    "forecast_id": f"synthetic-forecast-{calls['forecast']}",
                    "game_id": ids[0],
                    "season": 2026,
                    "home_team": "BUF",
                    "away_team": "NE",
                    "kickoff": games.iloc[0]["kickoff"].isoformat(),
                    "as_of": clock[0].isoformat(),
                    "generated_at": clock[0].isoformat(),
                    "model_id": "synthetic-model",
                    "model_version": "logistic-regression-v1",
                    "model_sha256": "a" * 64,
                    "features_sha256": "b" * 64,
                    "home_win_probability": "0.7",
                    "away_win_probability": "0.3",
                    "probability_scope": "conditional_on_decisive_result",
                }
            ],
            "bundle_path": str(tmp_path / "model"),
        }

    def markets():
        calls["markets"] += 1
        return []

    def performance(games, source, as_of):
        calls["performance"] += 1
        assert source["fetched_at"] <= as_of.isoformat()
        return {"status": "no_scored_games", "models": {}, "paper": {"open_positions": 0}}

    services = {
        "clock": lambda: clock[0],
        "fetch_schedule": schedule,
        "forecast": forecast,
        "list_markets": markets,
        "performance": performance,
        "quote": lambda *args: pytest.fail("No quote exists in this operational fixture"),
        "settlement": lambda *args: pytest.fail("No paper position exists in this fixture"),
    }
    return config, services, clock, calls, games


def test_successful_slot_is_durable_and_idempotent(setup):
    config, services, clock, calls, _ = setup
    first = run_research_cycle(config, services=services)
    second = run_research_cycle(config, services=services)
    assert first["status"] == "completed"
    assert second["status"] == "skipped"
    assert second["reason"] == "slot_already_completed"
    assert calls["forecast"] == 1
    root = Path(config["paths"]["state_dir"])
    assert json.loads((root / "latest.json").read_text()) == first
    assert json.loads((root / "last_success.json").read_text()) == first
    assert json.loads((root / "runs" / first["run_id"] / "run.json").read_text()) == first
    with sqlite3.connect(root / "research.sqlite") as connection:
        assert connection.execute("SELECT status FROM jobs").fetchone()[0] == "completed"
        assert connection.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 1
    assert first["performance"]["status"] == "no_scored_games"
    assert first["evaluations"][0]["reasons"] == ["no_matching_us_market"]


def test_transient_failure_retries_same_slot_using_forecast_checkpoint(setup):
    config, services, _, calls, _ = setup
    successful_markets = services["list_markets"]
    services["list_markets"] = lambda: (_ for _ in ()).throw(requests.Timeout("synthetic timeout"))
    first = run_research_cycle(config, services=services)
    assert first["status"] == "partial"
    assert first["errors"][0]["stage"] == "markets"
    services["list_markets"] = successful_markets
    second = run_research_cycle(config, services=services)
    assert second["status"] == "completed"
    assert first["slot"] == second["slot"] and first["run_id"] != second["run_id"]
    assert calls["forecast"] == 1
    ledger = PaperLedger(config["paths"]["ledger"])
    assert ledger.summary()["counts"]["paper_forecasts"] == 1
    assert ledger.audit()["valid"]


def test_failed_schedule_does_not_claim_completed_or_fabricate_performance(setup):
    config, services, _, calls, _ = setup
    services["fetch_schedule"] = lambda: (_ for _ in ()).throw(
        requests.Timeout("schedule unavailable")
    )
    report = run_research_cycle(config, services=services)
    assert report["status"] == "failed"
    assert report["forecasts"] == [] and report["performance"] is None
    assert calls["forecast"] == calls["performance"] == 0
    root = Path(config["paths"]["state_dir"])
    assert not (root / "last_success.json").exists()
    assert json.loads((root / "latest.json").read_text())["status"] == "failed"


def test_crash_releases_lock_and_next_attempt_recovers_running_state(setup):
    config, services, _, calls, _ = setup
    healthy = services["forecast"]
    services["forecast"] = lambda *args: (_ for _ in ()).throw(KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        run_research_cycle(config, services=services)
    services["forecast"] = healthy
    report = run_research_cycle(config, services=services)
    assert report["status"] == "completed"
    with sqlite3.connect(Path(config["paths"]["state_dir"]) / "research.sqlite") as connection:
        assert [
            row[0] for row in connection.execute("SELECT status FROM attempts ORDER BY rowid")
        ] == ["abandoned", "completed"]
    assert calls["forecast"] == 1


def test_stale_checkpoint_is_recomputed_on_retry(setup):
    config, services, clock, calls, _ = setup
    config["paper"]["max_forecast_age_seconds"] = 60
    healthy = services["list_markets"]
    services["list_markets"] = lambda: (_ for _ in ()).throw(requests.Timeout("retry"))
    first = run_research_cycle(config, services=services)
    clock[0] += pd.Timedelta(seconds=120)
    services["list_markets"] = healthy
    second = run_research_cycle(config, services=services)
    assert first["slot"] == second["slot"]
    assert second["status"] == "completed" and calls["forecast"] == 2


def test_os_lock_prevents_concurrent_cycle(setup):
    config, services, _, calls, _ = setup
    root = Path(config["paths"]["state_dir"])
    root.mkdir(parents=True)
    with (root / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report = run_research_cycle(config, services=services)
    assert report["reason"] == "worker_locked"
    assert calls["schedule"] == 0


def test_two_threads_never_overlap_worker_jobs(setup):
    config, services, _, calls, _ = setup
    entered, release = threading.Event(), threading.Event()
    healthy = services["forecast"]

    def slow_forecast(*args):
        entered.set()
        assert release.wait(5)
        return healthy(*args)

    services["forecast"] = slow_forecast
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(run_research_cycle, config, services=services)
        assert entered.wait(5)
        second = pool.submit(run_research_cycle, config, services=services)
        assert second.result(timeout=5)["reason"] == "worker_locked"
        release.set()
        assert first.result(timeout=5)["status"] == "completed"
    assert calls["forecast"] == 1


def test_worker_waits_for_next_utc_slot_and_stops_after_requested_cycles(setup):
    config, services, clock, calls, _ = setup

    class FastEvent:
        waits = []

        def is_set(self):
            return False

        def set(self):
            raise AssertionError("Unexpected process signal")

        def wait(self, seconds):
            self.waits.append(seconds)
            clock[0] += pd.Timedelta(seconds=seconds)
            return False

    event = FastEvent()
    result = run_worker(config, max_cycles=2, stop_event=event, services=services)
    assert result["cycles"] == 2 and calls["forecast"] == 2
    assert 0 < max(event.waits) <= 60
    slots = [record["slot"] for record in result["recent_runs"]]
    assert slots[0] != slots[1]


def test_worker_retry_backoff_is_bounded_and_respects_stop(setup):
    config, services, clock, _, _ = setup
    services["fetch_schedule"] = lambda: (_ for _ in ()).throw(requests.Timeout("offline"))

    class StopAfterWait(threading.Event):
        def wait(self, seconds):
            assert seconds <= 60
            clock[0] += pd.Timedelta(seconds=seconds)
            self.set()
            return True

    result = run_worker(config, max_cycles=10, services=services, stop_event=StopAfterWait())
    assert result["cycles"] == 1 and result["stopped"]


def test_completed_report_remains_last_success_after_later_failure(setup):
    config, services, clock, _, _ = setup
    success = run_research_cycle(config, services=services)
    clock[0] += pd.Timedelta(seconds=config["interval_seconds"])
    services["fetch_schedule"] = lambda: (_ for _ in ()).throw(requests.Timeout("later failure"))
    failed = run_research_cycle(config, services=services)
    root = Path(config["paths"]["state_dir"])
    assert failed["status"] == "failed"
    assert json.loads((root / "last_success.json").read_text())["run_id"] == success["run_id"]


def test_missing_requested_forecast_game_cannot_claim_forecast_stage_success(setup):
    config, services, _, _, _ = setup
    healthy = services["forecast"]

    def altered(ids, config):
        batch = healthy(ids, config)
        batch["forecasts"][0]["game_id"] = "unrequested-synthetic-game"
        return batch

    services["forecast"] = altered
    report = run_research_cycle(config, services=services)
    assert any(error["stage"] == "forecast" for error in report["errors"])
    assert report["forecasts"] == []


def test_empty_forecast_service_result_is_an_error_not_a_success(setup):
    config, services, _, _, _ = setup
    services["forecast"] = lambda *args: {"forecasts": []}
    report = run_research_cycle(config, services=services)
    assert report["status"] == "partial"
    assert any("omitted" in error["message"] for error in report["errors"])


def synthetic_quote(clock, games, outcome="yes"):
    description = (
        "Overtime is included; a tie pays $0.50, cancellation pays last fair market price."
    )
    return {
        "quote_id": f"synthetic-quote-{outcome}-{clock[0].isoformat()}",
        "venue": "polymarket_us",
        "market_id": "synthetic-market",
        "market_slug": "synthetic-buf-ne",
        "yes_team": "BUF",
        "no_team": "NE",
        "outcome": outcome,
        "market_status": "open",
        "game_start_time": games.iloc[0]["kickoff"].isoformat(),
        "quote_at": clock[0].isoformat(),
        "received_at": clock[0].isoformat(),
        "ask_levels": [{"price": "0.2", "quantity": "100"}],
        "minimum_trade_quantity": "1",
        "price_tick": "0.01",
        "source": {"book": {"sha256": "c" * 64}},
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
        "fee_schedule": {
            "type": "polymarket_us_standard_taker",
            "theta": "0.0695",
            "effective_at": "2026-10-19T17:00:00Z",
            "rounding": "ROUND_HALF_EVEN",
            "currency": "USD",
            "source_url": "https://docs.polymarket.us/fees.md",
        },
    }


def connect_market(services, clock, games):
    services["list_markets"] = lambda: [
        {
            "slug": "synthetic-buf-ne",
            "sportsMarketType": "football_team_full_game_winner",
            "gameStartTime": games.iloc[0]["kickoff"].isoformat(),
            "marketSides": [
                {"team": {"league": "nfl", "abbreviation": code}} for code in ["BUF", "NE"]
            ],
        }
    ]
    services["quote"] = lambda slug, outcome: synthetic_quote(clock, games, outcome)


def test_missing_risk_assumptions_records_both_sides_as_passes(setup):
    config, services, clock, _, games = setup
    connect_market(services, clock, games)
    report = run_research_cycle(config, services=services)
    assert report["status"] == "completed", report["errors"]
    assert len(report["evaluations"]) == 2
    assert all(row["status"] == "pass" for row in report["evaluations"])
    assert all("tie_probability_required" in row["reasons"] for row in report["evaluations"])
    assert report["paper_fills"] == []
    assert report["paper_summary"]["counts"]["paper_evaluations"] == 2


def test_quote_failure_isolated_and_retry_does_not_duplicate_forecast(setup):
    config, services, clock, calls, games = setup
    connect_market(services, clock, games)
    healthy = services["quote"]

    def one_failure(slug, outcome):
        if outcome == "yes":
            raise requests.Timeout("one side temporarily unavailable")
        return healthy(slug, outcome)

    services["quote"] = one_failure
    first = run_research_cycle(config, services=services)
    assert first["status"] == "partial" and len(first["evaluations"]) == 1
    services["quote"] = healthy
    second = run_research_cycle(config, services=services)
    assert second["status"] == "completed" and len(second["evaluations"]) == 2
    assert calls["forecast"] == 1


def test_scheduled_cancellation_range_is_complemented_for_no(setup):
    config, services, clock, _, games = setup
    connect_market(services, clock, games)
    config["paper"]["assumptions"] = {
        "tie_probability_range": ["0", "0.02"],
        "completion_probability_range": ["0.8", "0.9"],
        "cancellation_settlement_price_range": ["0.2", "0.4"],
    }
    report = run_research_cycle(config, services=services)
    assert report["status"] == "completed", report["errors"]
    by_side = {row["outcome"]: row for row in report["evaluations"]}
    assert by_side["yes"]["uncertainty_policy"]["cancellation_settlement_price_range"] == [
        "0.2",
        "0.4",
    ]
    assert by_side["no"]["uncertainty_policy"]["cancellation_settlement_price_range"] == [
        "0.6",
        "0.8",
    ]
    assert by_side["yes"]["assumptions"]["cancellation_settlement_price"] == "0.3"
    assert by_side["no"]["assumptions"]["cancellation_settlement_price"] == "0.7"


def test_same_scheduled_position_is_skipped_even_when_game_cap_exceeds_one(setup, tmp_path):
    config, services, clock, _, games = setup
    connect_market(services, clock, games)
    supplied = copy.deepcopy(config)
    supplied.pop("config_hash")
    supplied["paper"]["max_open_positions_per_game"] = 2
    supplied["paper"]["assumptions"] = {
        "tie_probability_range": ["0", "0.02"],
        "completion_probability_range": ["1", "1"],
        "cancellation_settlement_price_range": None,
    }
    config = validate_config(supplied, base_dir=tmp_path)
    original = services["forecast"]

    def with_interval(ids, config):
        batch = original(ids, config)
        record = batch["forecasts"][0]
        record.update(
            model_training_cutoff_at="2026-09-01T00:00:00Z",
            model_train_max_result_available_at="2026-01-08T00:00:00Z",
            model_training_game_ids_sha256="e" * 64,
        )
        record["probability_interval"] = {
            "home": {"lower": "0.6", "upper": "0.8"},
            "away": {"lower": "0.2", "upper": "0.4"},
            "confidence_level": 0.95,
            "bootstrap_samples": 20,
            "method": "training_week_block_bootstrap",
            "scope": "model_estimation_conditional_on_features",
            "calibration": "not_a_calibrated_predictive_coverage_guarantee",
            "model_id": record["model_id"],
            "model_sha256": record["model_sha256"],
            "features_sha256": record["features_sha256"],
            "game_id": record["game_id"],
            "as_of": record["as_of"],
            "ensemble_sha256": "f" * 64,
            "training_game_ids_sha256": record["model_training_game_ids_sha256"],
            "training_cutoff_at": record["model_training_cutoff_at"],
        }
        return batch

    services["forecast"] = with_interval
    first = run_research_cycle(config, services=services)
    assert first["status"] == "completed", first["errors"]
    assert first["paper_summary"]["counts"]["paper_trades"] == 1
    clock[0] += pd.Timedelta(seconds=config["interval_seconds"])
    second = run_research_cycle(config, services=services)
    assert second["status"] == "completed", second["errors"]
    assert second["paper_summary"]["counts"]["paper_trades"] == 1
    assert second["paper_fills"][0]["reason"] == "existing_scheduled_position"


def existing_position(config, services, clock, games):
    forecast = services["forecast"](["synthetic-upcoming"], config)["forecasts"][0]
    quote = synthetic_quote(clock, games)
    evaluation = evaluate_expected_value(
        forecast,
        quote,
        selected_team="BUF",
        quantity=10,
        as_of=clock[0],
        tie_probability="0",
        completion_probability="1",
    )
    ledger = PaperLedger(config["paths"]["ledger"])
    ledger.record_forecast(forecast)
    ledger.record_quote(quote)
    ledger.record_evaluation(evaluation)
    return ledger.paper_fill(evaluation["evaluation_id"], idempotency_key="synthetic-existing")


def test_unresolved_exchange_settlement_remains_pending(setup):
    config, services, clock, _, games = setup
    trade = existing_position(config, services, clock, games)
    clock[0] = games.iloc[0]["kickoff"] + pd.Timedelta(days=3)
    services["settlement"] = lambda slug: (_ for _ in ()).throw(
        ValueError("Only finally resolved NFL winner markets can be settled")
    )
    report = run_research_cycle(config, services=services)
    assert report["status"] == "completed", report["errors"]
    assert report["settlements"] == [
        {"paper_trade_id": trade["paper_trade_id"], "status": "pending"}
    ]
    assert report["paper_summary"]["open_positions"] == 1
    assert report["paper_summary"]["counts"]["paper_settlements"] == 0


def retained_settlement(config, clock):
    root = Path(config["paths"]["market_cache"])
    root.mkdir(parents=True, exist_ok=True)
    evidence = {
        "venue": "polymarket_us",
        "market_id": "synthetic-market",
        "market_slug": "synthetic-buf-ne",
        "market_status": "resolved",
        "settlement_price_yes": "0.5",
        "settlement_price_no": "0.5",
        "observed_at": clock[0].isoformat(),
        "source": {},
    }
    payloads = {
        "market": {
            "market": {
                "id": "synthetic-market",
                "slug": "synthetic-buf-ne",
                "status": "MARKET_STATUS_RESOLVED",
                "sportsMarketType": "football_team_full_game_winner",
                "sportsMarketTypeV2": "SPORTS_MARKET_TYPE_MONEYLINE",
            }
        },
        "settlement": {"slug": "synthetic-buf-ne", "settlement": "0.5"},
    }
    for name, payload in payloads.items():
        path = root / f"synthetic-{name}.json"
        path.write_text(json.dumps(payload))
        evidence["source"][name] = {
            "url": "https://gateway.polymarket.us/v1/markets/synthetic-buf-ne/settlement",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "path": str(path),
            "fetched_at": clock[0].isoformat(),
        }
    return evidence


def test_resolved_settlement_is_applied_once_from_explicit_provider_price(setup):
    config, services, clock, _, games = setup
    trade = existing_position(config, services, clock, games)
    clock[0] = games.iloc[0]["kickoff"] + pd.Timedelta(days=3)
    evidence = retained_settlement(config, clock)
    settlement_calls = []

    def settled(slug):
        settlement_calls.append(slug)
        return evidence

    services["settlement"] = settled
    first = run_research_cycle(config, services=services)
    assert first["status"] == "completed", first["errors"]
    assert first["settlements"][0]["paper_trade_id"] == trade["paper_trade_id"]
    assert first["settlements"][0]["payout"] == "5.0"
    assert first["settlements"][0]["evidence_verification"] == "public_us_settlement_snapshot"
    clock[0] += pd.Timedelta(seconds=config["interval_seconds"])
    second = run_research_cycle(config, services=services)
    assert second["settlements"] == []
    assert len(settlement_calls) == 1
    assert second["paper_summary"]["counts"]["paper_settlements"] == 1


@pytest.mark.parametrize("missing", ["market", "settlement"])
def test_automatic_settlement_requires_both_retained_raw_snapshots(setup, missing):
    config, services, clock, _, games = setup
    existing_position(config, services, clock, games)
    clock[0] = games.iloc[0]["kickoff"] + pd.Timedelta(days=3)
    evidence = retained_settlement(config, clock)
    evidence["source"][missing].pop("path")
    services["settlement"] = lambda slug: evidence
    report = run_research_cycle(config, services=services)
    assert report["status"] == "partial"
    assert any(
        row["stage"] == "settlement" and "verified raw US provider snapshots" in row["message"]
        for row in report["errors"]
    )
    assert report["paper_summary"]["open_positions"] == 1
    assert report["paper_summary"]["counts"]["paper_settlements"] == 0


def test_settlement_transport_error_is_not_a_fake_refund(setup):
    config, services, clock, _, games = setup
    existing_position(config, services, clock, games)
    clock[0] = games.iloc[0]["kickoff"] + pd.Timedelta(days=3)
    services["settlement"] = lambda slug: (_ for _ in ()).throw(
        requests.Timeout("exchange unavailable")
    )
    report = run_research_cycle(config, services=services)
    assert report["status"] == "partial"
    assert any(row["stage"] == "settlement" for row in report["errors"])
    assert report["paper_summary"]["counts"]["paper_settlements"] == 0


def test_configuration_change_creates_new_slot_job_without_rewriting_history(setup):
    config, services, _, _, _ = setup
    first = run_research_cycle(config, services=services)
    changed = copy.deepcopy(config)
    changed["config_hash"] = "different-policy-hash"
    second = run_research_cycle(changed, services=services)
    assert first["slot"] == second["slot"] and first["run_id"] != second["run_id"]
    assert first["config_hash"] != second["config_hash"]


def test_missing_completed_report_is_recovered_not_falsely_skipped(setup):
    config, services, _, calls, _ = setup
    first = run_research_cycle(config, services=services)
    (Path(config["paths"]["state_dir"]) / "runs" / first["run_id"] / "run.json").unlink()
    retry = run_research_cycle(config, services=services)
    assert retry["status"] == "completed"
    assert retry["run_id"] != first["run_id"]
    assert calls["forecast"] == 1  # The intact forecast checkpoint remains reusable.


def test_public_service_propagates_current_snapshot_observation_metadata(
    setup, monkeypatch, tmp_path
):
    config, _, clock, _, games = setup
    source = {"sha256": "f" * 64, "fetched_at": clock[0].isoformat()}
    path = tmp_path / "current.csv"
    monkeypatch.setattr(
        "predictions.research.NflverseProvider",
        lambda cache: SimpleNamespace(fetch=lambda refresh: (path, source)),
    )
    monkeypatch.setattr("predictions.research._now", lambda: clock[0])
    monkeypatch.setattr("predictions.research.load_games", lambda path, as_of: games)
    captured = []
    monkeypatch.setattr("predictions.live_cli._forecast", lambda args: captured.append(args) or {})
    services = _PublicServices(config)
    assert services.fetch_schedule()[1] == source
    services.forecast(["synthetic-upcoming"], config)
    assert captured[0].schedule_metadata == source
    assert captured[0].input == path
    assert captured[0].bootstrap_samples == config["bootstrap_samples"]


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_worker_cycle_count_rejected(setup, value):
    with pytest.raises(ValueError, match="max_cycles"):
        run_worker(setup[0], max_cycles=value)
