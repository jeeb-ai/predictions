"""Private dashboard tests use local synthetic research reports only."""

import hashlib
import json
from datetime import datetime, timezone

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from predictions import dashboard
from predictions.dashboard_access import (
    ATTEMPTS_KEY,
    AUTH_KEY,
    INPUT_KEY,
    LOCK_KEY,
    attempt_login,
    read_token,
    token_digest,
)

NOW = pd.Timestamp("2026-10-28T12:00:00Z")
TOKEN = "synthetic-test-access-key-0000000000000000"


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    config_path = tmp_path / "research.json"
    config_path.write_text(
        json.dumps(
            {
                "paths": {"state_dir": "state"},
                "dashboard": {"token_file": "private/key", "max_snapshot_age_seconds": 1800},
            }
        )
    )
    token_file = tmp_path / "private" / "key"
    token_file.parent.mkdir()
    token_file.write_text(TOKEN + "\n")
    token_file.chmod(0o600)
    state = tmp_path / "state"
    state.mkdir()
    monkeypatch.setenv("NFL_RESEARCH_CONFIG", str(config_path))
    monkeypatch.setenv("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return NOW.to_pydatetime().astimezone(tz or timezone.utc)

    monkeypatch.setattr(dashboard, "datetime", FrozenDateTime)
    monkeypatch.setattr("predictions.dashboard_access._clock", lambda: NOW.timestamp())
    return {
        "config_path": config_path,
        "config": dashboard.load_dashboard_config(),
        "token_file": token_file,
        "state": state,
        "root": tmp_path,
    }


def empty_report():
    return {
        "run_id": "synthetic-run",
        "as_of": NOW.isoformat(),
        "status": "completed",
        "forecasts": [],
        "evaluations": [],
        "performance": {
            "models": {
                "logistic-regression-v2": {
                    "n": 0,
                    "status": "no_scored_games",
                    "metrics": None,
                    "pending": 1,
                    "ties": 0,
                }
            },
            "paper": {
                "open_positions": 0,
                "open_exposure": "0",
                "realized_paper_profit": "0",
                "max_drawdown": "0",
                "open_position_rows": [],
                "settlement_rows": [],
                "settlement_timeline": [],
            },
            "outcomes": {"correction_count": 0},
        },
        "audit": {"valid": True, "head_hash": "a" * 64},
        "errors": [],
    }


def write_report(workspace, report=None):
    selected = empty_report() if report is None else report
    (workspace["state"] / "latest.json").write_text(json.dumps(selected))
    return selected


def app():
    return AppTest.from_string("from predictions.dashboard import main\nmain()\n").run()


def unlock(application, key=TOKEN):
    application.text_input[0].set_value(key)
    application.button[0].click().run()
    assert not application.exception
    return application


def test_unauthenticated_view_never_reads_private_snapshot(workspace, monkeypatch):
    (workspace["state"] / "latest.json").write_text("PRIVATE SENTINEL INVALID JSON")
    reads = []
    original = dashboard.load_snapshot

    def tracked(config):
        reads.append(True)
        return original(config)

    monkeypatch.setattr(dashboard, "load_snapshot", tracked)
    application = app()
    assert not application.exception
    assert reads == []
    assert not application.dataframe
    assert not application.metric
    assert application.text_input[0].label == "Access key"
    assert "PRIVATE SENTINEL" not in str(application)


def test_wrong_key_remains_generic_and_does_not_read_data(workspace, monkeypatch):
    write_report(workspace)
    monkeypatch.setattr(dashboard, "load_snapshot", lambda config: pytest.fail("Private read"))
    application = unlock(app(), "incorrect synthetic key")
    assert not application.dataframe
    assert application.error[0].value == "We could not verify that access key."
    assert TOKEN not in str(application)
    assert application.session_state[INPUT_KEY] == ""


def test_successful_login_shows_unknown_metrics_without_inventing_zero(workspace):
    write_report(workspace)
    application = unlock(app())
    assert application.title[0].value == "NFL Predictions"
    assert any("No scored games yet" in item.value for item in application.info)
    rows = [frame.value for frame in application.dataframe]
    metrics = next(frame for frame in rows if "Accuracy" in frame)
    assert metrics.iloc[0]["Scored games"] == 0
    assert metrics.iloc[0]["Accuracy"] == "Not yet observed"
    assert metrics.iloc[0]["Calibration error"] == "Not yet observed"
    assert [button.label for button in application.button] == ["Lock dashboard"]


def test_token_rotation_invalidates_session_before_any_more_private_reads(workspace, monkeypatch):
    write_report(workspace)
    reads = []
    original = dashboard.load_snapshot

    def tracked(config):
        reads.append(True)
        return original(config)

    monkeypatch.setattr(dashboard, "load_snapshot", tracked)
    application = unlock(app())
    assert len(reads) == 1
    workspace["token_file"].write_text("rotated-synthetic-token-1111111111111111\n")
    application.run()
    assert not application.exception
    assert len(reads) == 1
    assert not application.dataframe
    assert not application.metric
    assert application.text_input[0].label == "Access key"


def test_login_attempts_lock_and_recover_without_storing_plaintext():
    session = {}
    digest = token_digest(TOKEN)
    for _ in range(3):
        assert not attempt_login(session, "wrong key", digest, 100.0)
    assert session[LOCK_KEY] == 130.0
    assert session[ATTEMPTS_KEY] == 0
    assert not attempt_login(session, TOKEN, digest, 129.0)
    assert AUTH_KEY not in session
    assert attempt_login(session, TOKEN, digest, 130.0)
    assert session[AUTH_KEY] == digest
    assert TOKEN not in session.values()


def test_lockout_is_visible_and_submit_disabled(workspace):
    write_report(workspace)
    application = app()
    for _ in range(3):
        unlock(application, "wrong key")
    assert application.button[0].disabled
    assert "30 seconds" in application.warning[0].value


@pytest.mark.parametrize("mode", [0o644, 0o640, 0o666])
def test_insecure_token_permissions_fail_closed_without_private_data(workspace, mode):
    write_report(workspace)
    workspace["token_file"].chmod(mode)
    application = app()
    assert not application.exception
    assert "Private dashboard setup is required" in application.error[0].value
    assert not application.dataframe
    assert not application.metric


def test_missing_token_or_symlink_is_not_accepted(workspace):
    target = workspace["token_file"]
    target.unlink()
    assert "setup is required" in app().error[0].value
    outside = workspace["root"] / "another-key"
    outside.write_text(TOKEN)
    outside.chmod(0o600)
    target.symlink_to(outside)
    with pytest.raises(ValueError):
        read_token(target)


def test_no_runs_and_stale_or_future_recorded_runs_are_explicit(workspace):
    config = workspace["config"]
    assert dashboard.build_dashboard_view(config, NOW)["report"] is None
    record = empty_report()
    record["as_of"] = (NOW - pd.Timedelta(hours=1)).isoformat()
    write_report(workspace, record)
    assert dashboard.build_dashboard_view(config, NOW)["stale"]
    application = unlock(app())
    assert any("results are stale" in item.value for item in application.warning)
    record["as_of"] = (NOW + pd.Timedelta(seconds=1)).isoformat()
    write_report(workspace, record)
    view = dashboard.build_dashboard_view(config, NOW)
    assert view["stale"] and view["clock_ahead"]


def test_partial_failure_keeps_last_success_visible(workspace):
    good = empty_report()
    good["run_id"] = "previous-success"
    good["as_of"] = (NOW - pd.Timedelta(minutes=15)).isoformat()
    (workspace["state"] / "last_success.json").write_text(json.dumps(good))
    failed = empty_report()
    failed.update(status="partial", errors=[{"stage": "quote", "error": "Synthetic timeout"}])
    write_report(workspace, failed)
    view = dashboard.build_dashboard_view(workspace["config"], NOW)
    assert view["last_successful_run"]["run_id"] == "previous-success"
    application = unlock(app())
    assert any("latest research run is partial" in item.value for item in application.error)
    metric = next(item for item in application.metric if item.label == "Last successful run")
    assert metric.value == "2026-10-28 06:45 CDT"


def test_pointer_checksums_and_state_directory_boundary(workspace):
    root = workspace["state"]
    report = root / "runs" / "one" / "run.json"
    report.parent.mkdir(parents=True)
    payload = json.dumps(empty_report()).encode()
    report.write_bytes(payload)
    latest = root / "latest.json"
    latest.write_text(
        json.dumps({"path": "runs/one/run.json", "sha256": hashlib.sha256(payload).hexdigest()})
    )
    assert dashboard.load_snapshot(workspace["config"])["run_id"] == "synthetic-run"
    latest.write_text(json.dumps({"path": "runs/one/run.json", "sha256": "0" * 64}))
    with pytest.raises(ValueError, match="checksum"):
        dashboard.load_snapshot(workspace["config"])
    outside = workspace["root"] / "outside.json"
    outside.write_bytes(payload)
    latest.write_text(
        json.dumps({"path": str(outside), "sha256": hashlib.sha256(payload).hexdigest()})
    )
    with pytest.raises(ValueError, match="escapes"):
        dashboard.load_snapshot(workspace["config"])
    latest.unlink()
    latest.symlink_to(outside)
    with pytest.raises(ValueError, match="inside"):
        dashboard.load_snapshot(workspace["config"])


def populated_report():
    value = empty_report()
    value["forecasts"] = [
        {
            "game_id": "synthetic-game",
            "home_team": "BUF",
            "away_team": "NE",
            "model_version": "logistic-regression-v2",
            "home_win_probability": 0.65,
            "away_win_probability": 0.35,
            "as_of": NOW.isoformat(),
            "kickoff": (NOW + pd.Timedelta(days=1)).isoformat(),
            "probability_interval": {"home": {"lower": 0.6, "upper": 0.7}},
        }
    ]
    value["evaluations"] = [
        {
            "game_id": "synthetic-game",
            "market_id": "synthetic-market",
            "selected_team": "BUF",
            "outcome": "yes",
            "quantity": 10,
            "status": "pass",
            "ev_per_contract": "0.02",
            "fee_upper_bound": "0.15",
            "quote_at": NOW.isoformat(),
            "uncertainty": {"ev_per_contract_lower": "-0.03", "ev_per_contract_upper": "0.07"},
            "reasons": ["uncertainty_does_not_clear_minimum_ev"],
        }
    ]
    value["performance"]["models"]["logistic-regression-v2"].update(
        n=2,
        status="scored",
        pending=0,
        metrics={
            "accuracy": 0.5,
            "brier_score": 0.25,
            "log_loss": 0.6931,
            "expected_calibration_error": 0.0,
            "confidence_intervals_95": {"accuracy": [0.0, 1.0]},
            "reliability_bins": [
                {"count": 2, "mean_predicted": 0.5, "observed_home_win_rate": 0.5}
            ],
        },
    )
    value["performance"]["paper"].update(
        open_positions=1,
        open_exposure="2.15",
        realized_paper_profit="-1.25",
        max_drawdown="1.25",
        open_position_rows=[
            {"paper_trade_id": "synthetic-paper-position", "quantity": 10, "fee": "0.15"}
        ],
        settlement_rows=[
            {"paper_trade_id": "synthetic-settled-position", "payout": "0", "fee": "0.05"}
        ],
        settlement_timeline=[{"recorded_at": NOW.isoformat(), "cumulative_paper_profit": "-1.25"}],
    )
    return value


def test_populated_report_displays_intervals_fees_real_denominators_and_paper_labels(workspace):
    write_report(workspace, populated_report())
    application = unlock(app())
    rows = [frame.value for frame in application.dataframe]
    forecasts = next(frame for frame in rows if "Home interval" in frame)
    assert forecasts.iloc[0]["Home interval"] == "60.0%–70.0%"
    assert forecasts.iloc[0]["Kickoff (Chicago)"] == "2026-10-29 07:00 CDT"
    evaluations = next(frame for frame in rows if "Lower EV" in frame)
    assert evaluations.iloc[0]["Lower EV"] == "$-0.0300"
    assert evaluations.iloc[0]["Estimated fee cap"] == "$0.1500"
    metrics = next(frame for frame in rows if "Accuracy" in frame)
    assert metrics.iloc[0]["Scored games"] == 2
    assert metrics.iloc[0]["Calibration error"] == "0.0000"
    assert any(
        metric.label == "Realized paper P&L" and metric.value == "$-1.2500"
        for metric in application.metric
    )
    assert any("not calibrated" in item.value for item in application.caption)
    charts = application.get("vega_lite_chart")
    assert len(charts) == 2  # Actual performance schema renders paper P&L and reliability.
    assert any("Cumulative paper P&L" in chart.proto.spec for chart in charts)


def test_legacy_forecast_interval_is_unknown_instead_of_zero_width(workspace):
    report = populated_report()
    report["forecasts"][0].pop("probability_interval")
    write_report(workspace, report)
    application = unlock(app())
    rows = next(frame.value for frame in application.dataframe if "Home interval" in frame.value)
    assert rows.iloc[0]["Home interval"] == "Unavailable"


def test_dashboard_has_no_provider_calls_and_preserves_all_research_files(workspace, monkeypatch):
    write_report(workspace, populated_report())
    before = {path: path.read_bytes() for path in workspace["root"].rglob("*") if path.is_file()}

    def no_network(*args, **kwargs):
        raise AssertionError("A read-only dashboard attempted a network request")

    monkeypatch.setattr("requests.get", no_network)
    monkeypatch.setattr("requests.post", no_network)
    application = unlock(app())
    application.run()
    after = {path: path.read_bytes() for path in workspace["root"].rglob("*") if path.is_file()}
    assert before == after
    assert not application.exception


def test_invalid_snapshot_after_login_is_generic_and_does_not_echo_data(workspace):
    (workspace["state"] / "latest.json").write_text("PRIVATE CORRUPT SOURCE")
    application = unlock(app())
    assert application.error[0].value.startswith("Recorded research data could not be verified")
    assert "PRIVATE CORRUPT SOURCE" not in application.error[0].value
    assert not application.dataframe


def test_partial_metrics_are_unavailable_without_app_failure(workspace):
    report = populated_report()
    report["performance"]["models"]["logistic-regression-v2"]["metrics"] = {"accuracy": 0.5}
    write_report(workspace, report)
    application = unlock(app())
    metrics = next(frame.value for frame in application.dataframe if "Accuracy" in frame.value)
    assert metrics.iloc[0]["Accuracy"] == "50.0%"
    assert metrics.iloc[0]["Brier score"] == "Unavailable"
    assert metrics.iloc[0]["Calibration error"] == "Unavailable"


def test_current_paper_summary_survives_performance_failure(workspace):
    report = empty_report()
    report.update(
        status="partial",
        performance=None,
        paper_summary={
            "open_positions": 1,
            "open_exposure": "2.15",
            "realized_paper_profit": "0",
        },
    )
    write_report(workspace, report)
    application = unlock(app())
    values = {item.label: item.value for item in application.metric}
    assert values["Open paper positions"] == "1"
    assert values["Open paper exposure"] == "$2.1500"
    assert values["Paper drawdown"] == "Unavailable"


def test_last_success_performance_is_labeled_when_current_run_is_pending(workspace):
    previous = populated_report()
    previous["run_id"] = "previous-success"
    (workspace["state"] / "last_success.json").write_text(json.dumps(previous))
    current = empty_report()
    current.update(
        status="running",
        performance=None,
        paper_summary={
            "open_positions": 0,
            "open_exposure": "0",
            "realized_paper_profit": "-1.25",
        },
    )
    write_report(workspace, current)
    application = unlock(app())
    assert any(
        "Performance is from the last successful run" in item.value for item in application.warning
    )
    metrics = next(frame.value for frame in application.dataframe if "Accuracy" in frame.value)
    assert metrics.iloc[0]["Scored games"] == 2
    # Current counts do not display old position rows as if still open.
    assert (
        next(item for item in application.metric if item.label == "Open paper positions").value
        == "0"
    )
    assert not any("paper_trade_id" in frame.value for frame in application.dataframe)


def test_refresh_authorization_rechecks_the_current_file(workspace, monkeypatch):
    session = {AUTH_KEY: token_digest(TOKEN)}
    monkeypatch.setattr(dashboard.st, "session_state", session)
    assert dashboard._current_session_authorized(workspace["config"])
    workspace["token_file"].write_text("rotated-synthetic-token-2222222222222222\n")
    assert not dashboard._current_session_authorized(workspace["config"])
    workspace["token_file"].unlink()
    assert not dashboard._current_session_authorized(workspace["config"])


def test_insufficient_blocks_does_not_imply_narrow_confidence(workspace):
    report = populated_report()
    metrics = report["performance"]["models"]["logistic-regression-v2"]["metrics"]
    metrics.update(confidence_intervals_95=None, uncertainty_status="insufficient_observed_blocks")
    write_report(workspace, report)
    application = unlock(app())
    assert any("not enough observed week blocks" in item.value for item in application.info)
