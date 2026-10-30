"""Private read-only Streamlit views of recorded research and paper results.

Run with ``streamlit run src/predictions/dashboard.py`` and set
``NFL_RESEARCH_CONFIG`` to a validated research JSON configuration. No provider,
worker, model fit, settlement, or ledger mutation is called by this application.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from predictions.config import load_config
from predictions.dashboard_access import AUTH_KEY, authentication_gate, read_token, token_digest

MAX_REPORT_BYTES = 32 * 1024 * 1024
SUCCESS_STATES = {"completed", "success", "succeeded"}
DISPLAY_TIMEZONE = ZoneInfo("America/Chicago")


def load_dashboard_config(path: Path | str | None = None) -> dict:
    selected = path if path is not None else os.environ.get("NFL_RESEARCH_CONFIG")
    if not selected:
        raise ValueError("Private dashboard configuration is required")
    return load_config(selected)


def _read_report(path: Path, root: Path) -> dict:
    resolved = path.resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError("Research snapshot must remain inside the configured state directory")
    if resolved.stat().st_size > MAX_REPORT_BYTES:
        raise ValueError("Research snapshot exceeds the dashboard read limit")
    content = resolved.read_bytes()
    value = json.loads(content)
    if not isinstance(value, dict):
        raise ValueError("Research snapshot must contain a JSON object")
    if "path" in value and "run_id" not in value:
        target = Path(value["path"])
        target = target if target.is_absolute() else root / target
        target = target.resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise ValueError("Research snapshot pointer escapes the configured state directory")
        if target.stat().st_size > MAX_REPORT_BYTES:
            raise ValueError("Research snapshot exceeds the dashboard read limit")
        payload = target.read_bytes()
        if hashlib.sha256(payload).hexdigest() != value.get("sha256"):
            raise ValueError("Research snapshot checksum does not match its pointer")
        value = json.loads(payload)
        if not isinstance(value, dict) or "path" in value and "run_id" not in value:
            raise ValueError("Research snapshot pointer must identify one report")
    return value


def load_snapshot(config: dict) -> dict | None:
    """Read the latest recorded run without initializing a ledger or worker."""
    root = Path(config["paths"]["state_dir"]).resolve()
    latest = root / "latest.json"
    return _read_report(latest, root) if latest.exists() or latest.is_symlink() else None


def _timestamp(value) -> pd.Timestamp | None:
    try:
        result = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    return result.tz_convert("UTC") if pd.notna(result) and result.tzinfo is not None else None


def _last_success(config: dict, report: dict | None) -> dict | None:
    if report and report.get("status") in SUCCESS_STATES:
        return report
    root = Path(config["paths"]["state_dir"]).resolve()
    recorded = root / "last_success.json"
    if recorded.exists() or recorded.is_symlink():
        return _read_report(recorded, root)
    # Support retained histories from a worker that predates last_success.json.
    candidates = []
    for pattern in ["runs/*/run.json", "research_runs/*/run.json", "attempts/*/run.json"]:
        candidates.extend(root.glob(pattern))
    candidates = sorted(candidates, key=lambda path: path.parent.name, reverse=True)[:50]
    successes = []
    for path in candidates:
        candidate = _read_report(path, root)
        when = _timestamp(candidate.get("as_of"))
        if candidate.get("status") in SUCCESS_STATES and when is not None:
            successes.append((when, candidate))
    return max(successes, key=lambda pair: pair[0])[1] if successes else None


def _records(value) -> list[dict]:
    if isinstance(value, dict):
        value = value.get("forecasts", [])
    if not isinstance(value, list):
        return []
    rows = []
    for item in value:
        if isinstance(item, dict) and isinstance(item.get("forecasts"), list):
            rows.extend(row for row in item["forecasts"] if isinstance(row, dict))
        elif isinstance(item, dict):
            rows.append(item)
    return rows


def build_dashboard_view(config: dict, now=None) -> dict:
    """Normalize recorded state for display without generating any new results."""
    observed = _timestamp(now if now is not None else datetime.now(timezone.utc))
    if observed is None:
        raise ValueError("Dashboard time must be timezone aware")
    report = load_snapshot(config)
    when = _timestamp(report.get("as_of")) if report else None
    age = (observed - when).total_seconds() if when is not None else None
    limit = config["dashboard"]["max_snapshot_age_seconds"]
    last_successful = _last_success(config, report)
    performance = report.get("performance", {}) if report else {}
    from_previous = False
    if not isinstance(performance, dict) or not performance:
        previous = last_successful.get("performance") if last_successful else None
        if isinstance(previous, dict) and previous:
            performance, from_previous = previous, True
    prediction_path = Path(config["paths"]["state_dir"]).resolve() / "predictions.json"
    predictions = (
        _read_report(prediction_path, prediction_path.parent) if prediction_path.exists() else None
    )
    forecast_source = (
        predictions or (report if report and report.get("forecasts") else last_successful) or {}
    )
    return {
        "slate": forecast_source.get("slate", []),
        "report": report,
        "last_successful_run": last_successful,
        "snapshot_age_seconds": age,
        "stale": report is not None and (age is None or age < 0 or age > limit),
        "clock_ahead": age is not None and age < 0,
        "forecasts": _records(forecast_source.get("forecasts", [])),
        "evaluations": _records(report.get("evaluations", [])) if report else [],
        "performance": performance if isinstance(performance, dict) else {},
        "performance_from_last_successful_run": from_previous,
    }


def _number(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _percent(value) -> str:
    number = _number(value)
    return f"{number:.1%}" if number is not None else "Unavailable"


def _money(value) -> str:
    number = _number(value)
    return f"${number:,.4f}" if number is not None else "Unavailable"


def _score(value) -> str:
    number = _number(value)
    return f"{number:.4f}" if number is not None else "Unavailable"


def _display_time(value) -> str:
    timestamp = _timestamp(value)
    return (
        timestamp.tz_convert(DISPLAY_TIMEZONE).strftime("%Y-%m-%d %H:%M %Z")
        if timestamp is not None
        else "Unavailable"
    )


def _object(value) -> dict:
    return value if isinstance(value, dict) else {}


def _interval(value) -> str:
    if not isinstance(value, dict):
        return "Unavailable"
    low, high = _number(value.get("lower")), _number(value.get("upper"))
    return (
        f"{low:.1%}–{high:.1%}"
        if low is not None and high is not None and 0 <= low <= high <= 1
        else "Unavailable"
    )


def _display_table(rows: list[dict]) -> None:
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _render_forecasts(rows: list[dict]) -> None:
    st.subheader("Pregame forecasts")
    st.caption(
        "Intervals describe model-estimation uncertainty conditional on features. They are not calibrated future-outcome coverage guarantees."
    )
    if not rows:
        st.info("No pregame forecasts have been recorded for this run.")
        return
    _display_table(
        [
            {
                "Game": f"{row.get('away_team', '?')} at {row.get('home_team', '?')}",
                "Kickoff (Chicago)": _display_time(row.get("kickoff")),
                "Model": row.get("model_version"),
                "Home win, decisive": _percent(row.get("home_win_probability")),
                "Home interval": _interval(_object(row.get("probability_interval")).get("home")),
                "Away win, decisive": _percent(row.get("away_win_probability")),
                "Forecast cutoff (Chicago)": _display_time(row.get("as_of")),
            }
            for row in rows
        ]
    )


def _render_evaluations(rows: list[dict], forecasts=None) -> None:
    by_id = {row["forecast_id"]: row for row in (forecasts or []) if "forecast_id" in row}

    def probability(row):
        if row.get("decisive_win_probability") is not None:
            return row["decisive_win_probability"]
        forecast = by_id.get(row.get("forecast_id"), {})
        if forecast and row.get("selected_team") in {
            forecast.get("home_team"),
            forecast.get("away_team"),
        }:
            return forecast.get(
                "home_win_probability"
                if row["selected_team"] == forecast["home_team"]
                else "away_win_probability"
            )
        return None

    def price(row):
        count = _number(row.get("fill_quantity"))
        notional = _number(row.get("notional"))
        return notional / count if count and notional is not None else None

    st.subheader("Market evaluations")
    st.caption(
        "Research decisions use the recorded observation times, actual US asks, explicit risk ranges, and conservative estimated fees. Quotes shown here can now be stale."
    )
    if not rows:
        st.info("No market evaluations have been recorded for this run.")
        return
    _display_table(
        [
            {
                "Game": row.get("game_id"),
                "Market": row.get("market_slug", row.get("market_id")),
                "Team / side": f"{row.get('selected_team', '?')} / {row.get('outcome', '?')}",
                "Decision": row.get("status"),
                "Model win probability (decisive game)": _percent(probability(row)),
                "Retained executable average ask": _money(price(row)),
                "Unadjusted probability minus price (not EV)": _score(
                    float(probability(row)) - price(row)
                )
                if probability(row) is not None and price(row) is not None
                else "Unavailable",
                "Point EV / contract": _money(row.get("ev_per_contract")),
                "Lower EV": _money(_object(row.get("uncertainty")).get("ev_per_contract_lower")),
                "Upper EV": _money(_object(row.get("uncertainty")).get("ev_per_contract_upper")),
                "Estimated fee cap": _money(row.get("fee_upper_bound", row.get("fee"))),
                "Quantity": row.get("quantity"),
                "Quote time (Chicago)": _display_time(row.get("quote_at")),
                "Reasons": ", ".join(
                    str(reason).replace("_", " ") for reason in row.get("reasons", [])
                ),
            }
            for row in rows
        ]
    )


def _render_paper(performance: dict, report: dict) -> None:
    st.subheader("Paper ledger")
    st.caption(
        "Local simulated positions only. Fees are conservative estimates; paper P&L is separate from model accuracy and real-money returns."
    )
    paper = _object(performance.get("paper"))
    current_summary = _object(report.get("paper_summary"))
    if current_summary:
        # A current ledger summary is more recent than a retained performance
        # report. Do not mix its counts with stale position rows after failure.
        paper = (
            {**paper, **current_summary} if _object(report.get("performance")) else current_summary
        )
    if not isinstance(paper, dict) or not paper:
        st.info("No recorded paper-ledger summary is available yet.")
        return
    columns = st.columns(4)
    columns[0].metric("Open paper positions", str(paper.get("open_positions", "Unavailable")))
    columns[1].metric("Open paper exposure", _money(paper.get("open_exposure")))
    columns[2].metric("Realized paper P&L", _money(paper.get("realized_paper_profit")))
    columns[3].metric("Paper drawdown", _money(paper.get("max_drawdown")))
    open_rows = _records(paper.get("open_position_rows", []))
    settled = _records(paper.get("settlement_rows", paper.get("settlement_timeline", [])))
    st.markdown("**Open paper positions**")
    if open_rows:
        _display_table(open_rows)
    else:
        st.info("No open paper-position rows are recorded.")
    st.markdown("**Evidenced paper settlements**")
    if settled:
        _display_table(settled)
    else:
        st.info("No paper settlements have been recorded.")
    timeline = _records(paper.get("settlement_timeline", []))
    points = []
    for item in timeline:
        when = _timestamp(item.get("recorded_at", item.get("timestamp", item.get("settled_at"))))
        value = _number(item.get("cumulative_profit", item.get("cumulative_paper_profit")))
        if when is not None and value is not None:
            points.append(
                {
                    "Settlement time (Chicago)": when.tz_convert(DISPLAY_TIMEZONE),
                    "Cumulative paper P&L": value,
                }
            )
    if points:
        st.line_chart(pd.DataFrame(points).set_index("Settlement time (Chicago)"))


def _render_performance(performance: dict) -> None:
    st.subheader("Prospective performance")
    st.caption(
        "The first valid pregame forecast per game and model version is scored. Pending games and ties do not become binary losses; repeated polls do not increase the denominator."
    )
    models = performance.get("models", {})
    if not isinstance(models, dict) or not models:
        st.info("No scored games yet. Performance will appear after outcomes are observed.")
        return
    summary, intervals, reliability = [], [], []
    for version, group in models.items():
        if not isinstance(group, dict):
            continue
        metrics = group.get("metrics")
        count = _number(group.get("n"))
        has_metrics = isinstance(metrics, dict) and count is not None and count > 0
        summary.append(
            {
                "Model": version,
                "Scored games": group.get("n") if count is not None else "Unavailable",
                "Pending": group.get("pending", "Unavailable"),
                "Ties excluded": group.get("ties", "Unavailable"),
                "Accuracy": _percent(metrics.get("accuracy"))
                if has_metrics
                else "Not yet observed",
                "Brier score": _score(metrics.get("brier_score"))
                if has_metrics
                else "Not yet observed",
                "Log loss": _score(metrics.get("log_loss")) if has_metrics else "Not yet observed",
                "Calibration error": _score(metrics.get("expected_calibration_error"))
                if has_metrics
                else "Not yet observed",
            }
        )
        if not has_metrics:
            continue
        if metrics.get("uncertainty_status") == "insufficient_observed_blocks":
            st.info(f"{version}: not enough observed week blocks to estimate sampling intervals.")
        for metric, bounds in (metrics.get("confidence_intervals_95") or {}).items():
            if isinstance(bounds, list) and len(bounds) == 2:
                intervals.append(
                    {
                        "Model": version,
                        "Metric": metric,
                        "95% lower": bounds[0],
                        "95% upper": bounds[1],
                    }
                )
        for cell in metrics.get("reliability_bins", []):
            if isinstance(cell, dict) and cell.get("count", 0):
                reliability.append(
                    {
                        "Model": version,
                        "Predicted": cell.get("mean_predicted"),
                        "Observed": cell.get("observed_home_win_rate"),
                        "Games": cell["count"],
                    }
                )
    _display_table(summary)
    if not any(
        isinstance(group, dict) and (_number(group.get("n")) or 0) > 0 for group in models.values()
    ):
        st.info("No scored games yet. Unknown accuracy and calibration are not displayed as zero.")
    if intervals:
        st.markdown("**Sampling uncertainty for fixed recorded forecasts**")
        _display_table(intervals)
    if reliability:
        st.scatter_chart(
            pd.DataFrame(reliability), x="Predicted", y="Observed", color="Model", size="Games"
        )
    corrections = _object(performance.get("outcomes")).get("correction_count", 0)
    if corrections:
        st.warning(
            f"{corrections} outcome corrections are recorded. Earlier metric reports remain retained."
        )


def render_dashboard(view: dict) -> None:
    from predictions.prediction_ui import render_predictions

    st.title("NFL Predictions")
    st.caption("Independent model forecasts · probabilities conditional on a decisive result")
    st.caption("Display times: America/Chicago. Source and audit timestamps remain in UTC.")
    render_predictions(view)
    with st.expander("Research / Administration", expanded=False):
        report = view["report"] or {}
        st.caption(f"Worker status: {report.get('status', 'not started')}")
        if report.get("status") in {"partial", "failed"}:
            st.error(
                f"The latest research run is {report['status']}. Recorded errors are retained below."
            )
        if view["stale"]:
            st.warning(
                "Research results are stale. Check the worker before relying on current quotes."
            )
        if view["clock_ahead"]:
            st.error("The recorded run is ahead of the dashboard clock. Check the host clocks.")
        if view["performance_from_last_successful_run"]:
            st.warning(
                "Performance is from the last successful run; the current run has not published a new report."
            )
        _render_performance(view["performance"])
        _render_paper(view["performance"], report)
        success = view["last_successful_run"]
        st.metric(
            "Last successful run",
            _display_time(success.get("as_of")) if success else "Not yet observed",
        )
        st.subheader("Audit")
        _render_forecasts(view["forecasts"])
        st.json(
            {
                key: report.get(key)
                for key in ["run_id", "as_of", "status", "errors", "ledger_audit"]
            }
        )


def main() -> None:
    st.set_page_config(page_title="NFL Predictions", page_icon="🏈", layout="wide")
    try:
        config = load_dashboard_config()
        token = read_token(config["dashboard"]["token_file"])
    except (OSError, ValueError, TypeError, KeyError):
        st.title("Private NFL research")
        st.error(
            "Private dashboard setup is required. Initialize the host configuration and access key."
        )
        return
    if not authentication_gate(token):
        return
    render_authenticated_dashboard(config)


def _current_session_authorized(config: dict) -> bool:
    try:
        expected = token_digest(read_token(config["dashboard"]["token_file"]))
    except (OSError, ValueError, TypeError, KeyError):
        return False
    actual = st.session_state.get(AUTH_KEY)
    return isinstance(actual, str) and hmac.compare_digest(actual, expected)


@st.fragment(run_every=30)
def render_authenticated_dashboard(config: dict) -> None:
    """Refresh recorded results, revalidating the current key before every read."""
    if not _current_session_authorized(config):
        st.session_state.pop(AUTH_KEY, None)
        st.rerun(scope="app")
        return
    try:
        view = build_dashboard_view(config)
        render_dashboard(view)
    except (OSError, ValueError, TypeError, KeyError):
        st.title("NFL research and paper trading")
        st.error(
            "Recorded research data could not be verified. Check the worker and retained reports."
        )


if __name__ == "__main__":
    main()
