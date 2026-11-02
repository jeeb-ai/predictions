"""Read-only prospective summaries. No provider calls or database writes from the UI."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st


def read_reports(config):
    root = Path(config["paths"]["state_dir"])
    reports = {}
    errors = []
    for sport in ["NFL", "NBA", "NHL"]:
        path = root / (sport.lower() + "-prospective.json")
        try:
            if path.is_file():
                if path.stat().st_size > 32 * 1024 * 1024:
                    raise ValueError("Oversized report")
                value = json.loads(path.read_text())
                if (
                    value.get("type") != "prospective_performance"
                    or value.get("cohort") != "first_valid_receipt_per_game"
                    or value.get("sport") != sport
                ):
                    raise ValueError("Invalid prospective cohort")
                reports[sport] = value
        except (OSError, ValueError, KeyError, TypeError):
            errors.append(sport)
    return {"sports": reports, "errors": errors, "cohort": "prospective_only"}


def render(config):
    from predictions.dashboard import _display_time

    data = read_reports(config)
    with st.expander("Prospective accuracy — future games", expanded=False):
        st.caption(
            "Only the first valid forecast captured before play is graded. Pending games are never losses. Historical backtests are separate."
        )
        if not data["sports"]:
            st.info(
                "Tracking starts with captured pregame predictions. No verified prospective performance report yet; the worker reconciles results automatically."
            )
            return
        selected = st.selectbox(
            "Performance period",
            ["All captured", "Last 7 days", "Last 30 days", "Last 90 days"],
            key="prospective_period",
        )
        columns = st.columns(3)
        for column, sport in zip(columns, ["NFL", "NBA", "NHL"], strict=True):
            with column:
                st.markdown("**" + sport + "**")
                report = data["sports"].get(sport)
                if not report:
                    st.caption("Not reconciled yet")
                    continue
                m = (
                    report["overall"]
                    if selected == "All captured"
                    else report["recent_days"][selected.split()[1]]
                )
                st.metric(
                    "Correct winners", "—" if m["accuracy"] is None else f"{m['accuracy']:.1%}"
                )
                st.caption(
                    f"Record {m['wins']}–{m['losses']} · {m['graded']} graded / {m['predicted']} predicted · {m['pending']} pending · {m['review']} under review"
                )
                st.metric(
                    "Average team-score error",
                    "—" if m["score_mae"] is None else f"{m['score_mae']:.2f}",
                )
                if m["accuracy"] is None:
                    st.caption(
                        "Awaiting verified completed games; no accuracy percentage is claimed."
                    )
                if m["ties"]:
                    st.caption(
                        f"{m['ties']} ties excluded from decisive-winner metrics; score metrics include them."
                    )
                st.caption("Updated " + _display_time(report["as_of"]))
                st.caption(
                    "Results last checked: "
                    + _display_time(report.get("reconciliation", {}).get("last_checked_at"))
                )
                if report.get("reconciliation", {}).get("errors"):
                    st.warning(
                        "Some results could not be verified. Prior verified results remain; pending games are not losses."
                    )
        trend = []
        for sport, report in data["sports"].items():
            for week, m in report["weekly"].items():
                if m["accuracy"] is not None:
                    trend.append({"Week": week, "Sport": sport, "Correct winners": m["accuracy"]})
        if trend:
            st.line_chart(
                pd.DataFrame(trend).pivot(index="Week", columns="Sport", values="Correct winners")
            )
        else:
            st.info(
                "No completed prospective games yet. There is no measured accuracy trend to plot."
            )
        for sport, report in data["sports"].items():
            with st.expander(sport + " calibration and prediction history"):
                m = report["overall"]
                st.write(
                    {
                        "Brier (lower is better)": m["brier"],
                        "Log loss (lower is better)": m["log_loss"],
                        "Margin MAE": m["margin_mae"],
                        "Total MAE": m["total_mae"],
                        "Score games": m["score_games"],
                    }
                )
                bins = [
                    {
                        "Predicted home chance": b["mean_predicted"],
                        "Actual home win rate": b["observed_home_win_rate"],
                        "Games": b["count"],
                    }
                    for b in m["calibration"]
                    if b["count"]
                ]
                if bins:
                    st.scatter_chart(
                        pd.DataFrame(bins),
                        x="Predicted home chance",
                        y="Actual home win rate",
                        size="Games",
                    )
                st.caption(
                    "Small samples do not establish calibration or a betting edge. Confidence groups refer to model probability, not profit."
                )
                history = []
                for row in report["history"]:
                    score = row.get("projected_score") or {}
                    history.append(
                        {
                            "Game": row["away_team"] + " at " + row["home_team"],
                            "Model": row["model_version"],
                            "Captured": _display_time(row["received_at"]),
                            "Start": _display_time(row["kickoff"]),
                            "Home chance": row["home_win_probability"],
                            "Projected home": score.get("home_mean"),
                            "Projected away": score.get("away_mean"),
                            "Actual home": row["actual_home_score"],
                            "Actual away": row["actual_away_score"],
                            "State": row["status"],
                            "Result scope": row.get("result_scope"),
                        }
                    )
                st.dataframe(pd.DataFrame(history), hide_index=True)
                st.caption(
                    "Fixed recorded research lines are not observed market prices. Their calibration and proper losses are available in platform-performance; market value stays PASS."
                )
        for sport in data["errors"]:
            st.warning(
                sport + " prospective report could not be verified; other sports remain available."
            )
