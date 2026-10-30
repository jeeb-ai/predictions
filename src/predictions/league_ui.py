"""Read-only, prediction-first cards for validated sport adapters."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from predictions.league_forecast import league_verdict
from predictions.league_model import line_probabilities


def render_league_card(row):
    from predictions.dashboard import _display_time

    story = league_verdict(row)
    s = row["score_summary"]
    h, a = row["home_name"], row["away_name"]
    with st.container(border=True):
        st.markdown(f"### {row['sport']} · {a} at {h}")
        st.caption(
            f"Kickoff: {_display_time(row['kickoff'])} · Updated: {_display_time(row['generated_at'])} · {row['model_version']}"
        )
        st.markdown(
            "**Predicted winner: " + (h if row["home_win_probability"] >= 0.5 else a) + "**"
        )
        cols = st.columns(2)
        cols[0].metric(h + " win probability", f"{row['home_win_probability']:.1%}")
        cols[1].metric(a + " win probability", f"{row['away_win_probability']:.1%}")
        st.write(story["summary"])
        st.write(
            f"Average projected final score: {h} {s['home_mean']:.1f}, {a} {s['away_mean']:.1f}. Expected home margin {s['expected_margin']:+.1f}; total {s['expected_total']:.1f}."
        )
        st.caption(
            f"Typical-score intervals (90% fitted predictive distribution): {h} {s['home_interval_90']}; {a} {s['away_interval_90']}. These are not guaranteed final scores or win-probability confidence intervals."
        )
        st.caption(
            "Winner probabilities and score projections are separately trained; they may disagree. No per-game parameter-refit interval is claimed. Refer to measured historical calibration; no betting edge is established."
        )
        validation = row.get("historical_validation", {})
        if validation.get("accuracy") is not None:
            ci = (validation.get("intervals_95") or {}).get("accuracy")
            interval = f" (95% week-block interval {ci[0]:.1%}–{ci[1]:.1%})" if ci else ""
            st.caption(
                f"Reserved {validation['season']} season: winner accuracy {validation['accuracy']:.1%}{interval}; Brier {validation['brier']:.4f}; calibration error {validation['ece']:.3f}. This is historical sampling uncertainty, not this game's probability interval."
            )
        if row["sport"] == "NHL":
            st.warning(
                "NHL validation remains limited: recent holdout winner accuracy was close to chance and extra-time probabilities were underestimated. Treat score and line scenarios as research estimates."
            )
        st.caption(row["availability"])
        if row.get("goalie_information"):
            for side in ["home", "away"]:
                goalie = row["goalie_information"][side]
                st.caption(
                    f"{row[side + '_name']} goalie proxy: {goalie['name']} ({goalie['tracked_shots']:.0f} tracked shots in eligible prior appearances). Not a confirmed starter."
                )
        if (pd.Timestamp.now(tz="UTC") - pd.Timestamp(row["as_of"])).total_seconds() > 3600:
            st.warning("Retained forecast is stale; awaiting sport worker refresh.")
        with st.expander("Why this pick?"):
            st.markdown("**Supporting factors**")
            for factor in story["support"]:
                st.write("• " + factor["text"])
            st.markdown("**Risks and opposing factors**")
            for factor in story["risks"]:
                st.write("• " + factor["text"])
            st.write(
                "Player availability is not verified by this historical feed. A lineup change can make this estimate less useful."
            )
        with st.expander("Advanced Analysis"):
            st.caption(
                "Standardization and probability calibration use older data only. Contributions plus the intercept recover the calibrated winner log-odds; they are associations, not causes."
            )
            st.dataframe(pd.DataFrame(row["explanation"]["contributions"]), hide_index=True)
            st.write(
                f"Intercept {row['explanation']['intercept']:+.5f}; final home log-odds {row['explanation']['home_log_odds']:+.5f}."
            )
            st.caption(
                f"Inputs known by {_display_time(row['as_of'])}; latest eligible result {_display_time(row.get('history_available_through'))}."
            )
            st.json(
                {
                    k: row.get(k)
                    for k in [
                        "features",
                        "source_games",
                        "source",
                        "training_seasons",
                        "calibration_season",
                        "model_sha256",
                        "validation_sha256",
                    ]
                }
            )


def render_league_lines(row):
    st.subheader(row["sport"] + " exact-line research scenarios")
    default_spread = -5.5 if row["sport"] == "NBA" else -1.5
    default_total = 220.5 if row["sport"] == "NBA" else 5.5
    spread = st.number_input("Home spread", value=default_spread, step=0.5, key="league_spread")
    total = st.number_input("Combined total", value=default_total, step=0.5, key="league_total")
    try:
        path = Path(row["distribution"]["path"])
        if (
            path.stat().st_size > 1_000_000
            or hashlib.sha256(path.read_bytes()).hexdigest() != row["distribution"]["sha256"]
        ):
            raise ValueError("Invalid league distribution")
        scope = (
            st.selectbox(
                "Outcome scope", ["Full game including OT/shootout", "Regulation 60 minutes"]
            )
            if row["sport"] == "NHL"
            else "Full game including overtime"
        )
        if scope == "Regulation 60 minutes":
            reference = row.get("regulation_distribution", {})
            path = Path(reference.get("path", ""))
            if hashlib.sha256(path.read_bytes()).hexdigest() != reference.get("sha256"):
                raise ValueError("Invalid regulation distribution")
        pmf = np.load(path, allow_pickle=False)
        if (
            pmf.ndim != 2
            or pmf.shape[0] != pmf.shape[1]
            or not np.isfinite(pmf).all()
            or (pmf < 0).any()
            or not np.isclose(pmf.sum(), 1)
        ):
            raise ValueError("Invalid league mass")
        lines = line_probabilities(pmf, spread, total)
    except (OSError, ValueError):
        st.warning("Verified retained score distribution unavailable; line probabilities withheld.")
        return
    st.write(
        f"Home covers {lines['home_cover']['win']:.1%}; away covers {lines['home_cover']['loss']:.1%}; push {lines['home_cover']['push']:.1%}."
    )
    st.write(
        f"Over {lines['over']['win']:.1%}; under {lines['over']['loss']:.1%}; push {lines['over']['push']:.1%}."
    )
    if row.get("score_model_outcomes"):
        v = row["score_model_outcomes"]
        st.write(
            f"Score-model regulation: home win {v['regulation_home_win']:.1%}, draw {v['regulation_draw']:.1%}, away win {v['regulation_away_win']:.1%}."
        )
        st.caption(
            f"Given extra time: shootout share {v['conditional_shootout_share']:.1%}; home OT win {v['conditional_home_ot_win']:.1%}; home shootout win {v['conditional_home_shootout_win']:.1%}. These conditional models use earlier outcomes, not the game's actual future outcome."
        )
        st.caption(
            "Full-game score scenarios award one deciding goal for OT/shootout. Regulation scenarios allow a draw. Match actual market rules before comparing prices; no quote is verified here."
        )
    st.caption(
        f"Selected scope: {scope}. User-entered lines are research scenarios, not observed market prices. These model probabilities are not a validated edge."
    )
    st.info(
        "Market Value: PASS — no verified executable price, liquidity, fee or matching settlement terms."
    )
