"""Prediction receipt, process isolation, resume, idempotency and source cutoffs."""

import copy
import fcntl
import sqlite3
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from predictions.automation import run_slot, run_worker, status, supervise
from predictions.config import initialize_research, load_config
from predictions.prospective import ForecastJournal, connect, score_evidence

AT = pd.Timestamp("2026-10-30T12:00:00Z")


def forecast(sport="NBA", identity="f1", at=AT):
    value = {
        "sport": sport,
        "forecast_id": identity,
        "game_id": "game1",
        "model_version": sport + "-v1",
        "model_id": "synthetic-model",
        "model_sha256": "a" * 64,
        "features_sha256": "b" * 64,
        "home_team": "BOS" if sport != "NFL" else "BUF",
        "away_team": "NYK" if sport != "NFL" else "MIA",
        "home_name": "Home",
        "away_name": "Away",
        "as_of": (at - pd.Timedelta(minutes=2)).isoformat(),
        "generated_at": (at - pd.Timedelta(minutes=1)).isoformat(),
        "kickoff": (at + pd.Timedelta(days=3)).isoformat(),
        "home_win_probability": 0.6,
        "away_win_probability": 0.4,
        "probability_scope": "conditional_on_decisive_result"
        if sport == "NFL"
        else "full_game_including_overtime_shootout",
    }
    return value


@pytest.fixture
def setup(tmp_path):
    configpath = tmp_path / "research.json"
    initialize_research(configpath)
    return configpath, load_config(configpath)


@pytest.mark.parametrize("sport", ["NFL", "NBA", "NHL"])
def test_capture_idempotent_later_predictions_retained_and_late_import_not_prospective(
    tmp_path, sport
):
    journal = ForecastJournal(tmp_path / "journal.sqlite")
    row = forecast(sport)
    assert journal.capture(sport, [row], received_at=AT)["pregame"] == 1
    assert journal.capture(sport, [row], received_at=AT + pd.Timedelta(days=4))["duplicate"] == 1
    later = forecast(sport, "f2")
    later["home_win_probability"] = 0.4
    later["away_win_probability"] = 0.6
    journal.capture(sport, [later], received_at=AT + pd.Timedelta(minutes=10))
    late = forecast(sport, "postgame-import")
    assert journal.capture(sport, [late], received_at=AT + pd.Timedelta(days=4))["late"] == 1
    records = journal.records(as_of=AT + pd.Timedelta(days=5))
    assert len(records) == 3 and records[0]["original"] == row
    assert records[0]["received_at"] == AT.isoformat()
    assert records[-1]["exclusion"] == "first_received_after_start"
    assert len(journal.records(as_of=AT)) == 1


def test_receipts_are_immutable_and_conflicting_ids_fail(tmp_path):
    journal = ForecastJournal(tmp_path / "journal.sqlite")
    row = forecast()
    journal.capture("NBA", [row], received_at=AT)
    mutated = copy.deepcopy(row)
    mutated["home_team"] = "MIL"
    with pytest.raises(ValueError, match="reused"):
        journal.capture("NBA", [mutated], received_at=AT)
    with connect(journal.path) as db:
        for sql in [
            "DELETE FROM receipts",
            "UPDATE receipts SET payload='{}'",
            "INSERT OR REPLACE INTO receipts SELECT * FROM receipts",
        ]:
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                db.execute(sql)


@pytest.mark.parametrize(
    "field,value",
    [
        ("generated_at", "2026-11-01T00:00:00Z"),
        ("as_of", "2026-11-02T00:00:00Z"),
        ("history_available_through", "2026-10-30T11:58:00Z"),
        ("model_sha256", "bad"),
        ("home_win_probability", float("nan")),
    ],
)
def test_invalid_or_future_information_cannot_be_recorded(tmp_path, field, value):
    row = forecast()
    row[field] = value
    with pytest.raises((ValueError, KeyError)):
        ForecastJournal(tmp_path / "journal.sqlite").capture("NBA", [row], received_at=AT)


def test_score_lines_are_frozen_from_verified_pmf_and_not_claimed_as_market(tmp_path):
    import hashlib

    from predictions.league_model import summarize

    row = forecast()
    mass = np.zeros((5, 5))
    mass[4, 2] = 0.6
    mass[2, 4] = 0.4
    path = tmp_path / "mass.npy"
    np.save(path, mass, allow_pickle=False)
    row.update(
        distribution={"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
        score_summary=summarize(mass),
    )
    journal = ForecastJournal(tmp_path / "journal.sqlite")
    journal.capture("NBA", [row], received_at=AT)
    saved = journal.records(as_of=AT)[0]["score"]
    assert saved["summary"]["expected_total"] == pytest.approx(6)
    assert all(
        x["source"] == "fixed_research_scenario" and x["market_status"].startswith("PASS")
        for x in saved["lines"]
    )
    path.write_bytes(b"corrupt")
    assert journal.records(as_of=AT)[0]["score"] == saved
    with pytest.raises(ValueError, match="checksum"):
        score_evidence(row, "NBA", AT)


def test_completed_slot_is_idempotent_and_sports_fail_independently(setup):
    path, config = setup
    calls = []

    def adapter(c, s, p):
        calls.append(s)
        if s == "NHL":
            raise ConnectionError("provider unavailable")
        return {
            "sport": s,
            "as_of": AT.isoformat(),
            "forecasts": [forecast(s)],
            "status": "completed",
        }

    assert run_slot(config, "NBA", path, adapter=adapter, clock=lambda: AT)["status"] == "completed"
    assert (
        run_slot(config, "NBA", path, adapter=adapter, clock=lambda: AT)["reason"]
        == "slot_completed"
    )
    assert run_slot(config, "NHL", path, adapter=adapter, clock=lambda: AT)["status"] == "failed"
    assert run_slot(config, "NFL", path, adapter=adapter, clock=lambda: AT)["status"] == "completed"
    assert calls == ["NBA", "NHL", "NFL"]
    health = status(config, at=AT)
    assert [x["status"] for x in health["sports"]] == ["completed", "completed", "failed"]
    assert health["sports"][1]["current_predictions"] == 1
    assert status(config, "NBA", at=AT + pd.Timedelta(hours=5))["sports"][0]["stale"]


def test_crash_after_checkpoint_resumes_without_refit_or_duplicate_receipt(setup):
    path, config = setup
    calls = []

    def adapter(*args):
        calls.append(1)
        return {
            "sport": "NBA",
            "as_of": AT.isoformat(),
            "forecasts": [forecast()],
            "status": "completed",
        }

    def crash(*args):
        raise SystemExit("simulated power loss")

    with pytest.raises(SystemExit):
        run_slot(config, "NBA", path, adapter=adapter, clock=lambda: AT, publisher=crash)
    assert run_slot(config, "NBA", path, adapter=adapter, clock=lambda: AT)["status"] == "completed"
    assert len(calls) == 1
    assert (
        len(
            ForecastJournal(Path(config["paths"]["state_dir"]) / "prospective.sqlite").records(
                as_of=AT
            )
        )
        == 1
    )


def test_lock_prevents_overlap_and_watchdog_handles_timeout(setup):
    path, config = setup
    root = Path(config["paths"]["state_dir"])
    with (root / "nba-automation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert run_slot(config, "NBA", path, clock=lambda: AT)["reason"] == "sport_locked"

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 1)

    assert (
        run_worker(config, "NBA", path, max_cycles=1, runner=timeout, clock=lambda: AT)["cycles"]
        == 1
    )
    assert status(config, "NBA", at=AT)["sports"][0]["error_code"] == "cycle_timeout"


def test_supervisor_stops_a_running_child_on_shutdown(tmp_path):
    stop = threading.Event()
    timer = threading.Timer(0.2, stop.set)
    timer.start()
    try:
        with (tmp_path / "child.log").open("w") as output:
            result = supervise(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                stdout=output,
                stderr=subprocess.STDOUT,
                timeout=10,
                stop=stop,
            )
        assert result.returncode != 0 and stop.is_set()
    finally:
        timer.cancel()


def test_status_is_read_only_and_missing_state_has_setup_information(tmp_path):
    from predictions.config import validate_config

    config = validate_config({}, base_dir=tmp_path)
    value = status(config, at=AT)
    assert not value["healthy"] and not Path(config["paths"]["state_dir"]).exists()


def test_parallel_startup_preserves_the_private_key_and_initializes_once(setup, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    from predictions.cli import _parser
    from predictions.platform_cli import dispatch

    path, config = setup
    key = Path(config["dashboard"]["token_file"])
    before = key.read_bytes()
    monkeypatch.setattr("predictions.automation.run_slot", lambda *a, **k: {"status": "completed"})
    args = [
        _parser().parse_args(["platform-run", "--sport", s, "--config", str(path)])
        for s in ["NBA", "NHL", "NFL"]
    ]
    with ThreadPoolExecutor(max_workers=3) as pool:
        assert all(r["status"] == "completed" for r in pool.map(dispatch, args))
    assert key.read_bytes() == before and key.stat().st_mode & 0o077 == 0
