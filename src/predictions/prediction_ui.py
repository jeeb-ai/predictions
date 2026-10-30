"""Prediction-first presentation. No market value enters a forecast or explanation."""

import pandas as pd
import streamlit as st

TEAM_NAMES = dict(
    zip(
        "ARI ATL BAL BUF CAR CHI CIN CLE DAL DEN DET GB HOU IND JAX KC LA LAC LV MIA MIN NE NO NYG NYJ PHI PIT SEA SF TB TEN WAS".split(),
        [
            "Arizona Cardinals",
            "Atlanta Falcons",
            "Baltimore Ravens",
            "Buffalo Bills",
            "Carolina Panthers",
            "Chicago Bears",
            "Cincinnati Bengals",
            "Cleveland Browns",
            "Dallas Cowboys",
            "Denver Broncos",
            "Detroit Lions",
            "Green Bay Packers",
            "Houston Texans",
            "Indianapolis Colts",
            "Jacksonville Jaguars",
            "Kansas City Chiefs",
            "Los Angeles Rams",
            "Los Angeles Chargers",
            "Las Vegas Raiders",
            "Miami Dolphins",
            "Minnesota Vikings",
            "New England Patriots",
            "New Orleans Saints",
            "New York Giants",
            "New York Jets",
            "Philadelphia Eagles",
            "Pittsburgh Steelers",
            "Seattle Seahawks",
            "San Francisco 49ers",
            "Tampa Bay Buccaneers",
            "Tennessee Titans",
            "Washington Commanders",
        ],
        strict=True,
    )
)


def team(code):
    return TEAM_NAMES.get(code, code)


def upcoming(rows, now):
    result = []
    for row in rows:
        kickoff = pd.to_datetime(row.get("kickoff"), utc=True, errors="coerce")
        if pd.notna(kickoff) and kickoff > now:
            result.append(row)
    return sorted(result, key=lambda row: (row["kickoff"], row["game_id"]))


def render_predictions(view):
    from predictions.dashboard import (
        _display_time,
        _interval,
        _percent,
        _render_evaluations,
        _timestamp,
        datetime,
        timezone,
    )
    from predictions.verdict import verdict

    now = pd.Timestamp(datetime.now(timezone.utc))
    rows = upcoming(view["forecasts"], now)
    known = {row["game_id"] for row in rows}
    pending = upcoming([row for row in view.get("slate", []) if row["game_id"] not in known], now)
    filters = st.columns(2)
    sport = filters[0].selectbox(
        "Sport", ["All sports", *sorted({r.get("sport", "NFL") for r in rows + pending})]
    )
    days = sorted(
        {
            pd.Timestamp(row["kickoff"]).tz_convert("America/Chicago").strftime("%Y-%m-%d")
            for row in rows + pending
        }
    )
    selected_day = filters[1].selectbox("Game date", ["All upcoming dates", *days])
    search = st.text_input("Find a team or matchup", key="game_search")

    def visible(row):
        date = pd.Timestamp(row["kickoff"]).tz_convert("America/Chicago").strftime("%Y-%m-%d")
        return (
            (sport == "All sports" or row.get("sport", "NFL") == sport)
            and (selected_day == "All upcoming dates" or date == selected_day)
            and search.casefold()
            in (
                row.get("home_name", team(row["home_team"]))
                + " "
                + row.get("away_name", team(row["away_team"]))
            ).casefold()
        )

    rows, pending = [r for r in rows if visible(r)], [r for r in pending if visible(r)]
    predictions, analysis, markets = st.tabs(["Predictions", "Matchup Analysis", "Market Value"])
    with predictions:
        st.subheader("Upcoming matchups")
        st.caption(
            "Our independently trained model. Market prices do not determine these probabilities. The worker automatically refreshes the configured upcoming slate (seven days by default)."
        )
        if not rows and not pending:
            st.info(
                "No upcoming predictions yet. The scheduled worker discovers games from nflverse, independently of market coverage. Check Research / Administration for provider errors."
            )
        for index, row in enumerate(rows):
            if row.get("sport", "NFL") != "NFL":
                from predictions.league_ui import render_league_card

                render_league_card(row)
                continue
            home, away = team(row["home_team"]), team(row["away_team"])
            p = row["home_win_probability"]
            winner = home if p > 0.5 else away if p < 0.5 else "Even matchup"
            with st.container(border=True):
                st.markdown(f"### {away} at {home}")
                st.caption(f"Kickoff: {_display_time(row['kickoff'])}")
                st.markdown(f"**Predicted winner: {winner}**")
                cols = st.columns(2)
                cols[0].metric(f"{home} win probability", _percent(p))
                cols[1].metric(f"{away} win probability", _percent(row["away_win_probability"]))
                st.caption(
                    f"Probability advantage: {abs(2 * p - 1) * 100:.1f} percentage points · not a projected score margin"
                )
                explanation = verdict(row, team)
                st.write(explanation["summary"])
                st.write(explanation["confidence"])
                score = row.get("score_prediction")
                if score:
                    summary = score["summary"]
                    st.write(
                        f"Average projected score: {home} {summary['home_mean']:.1f}, {away} {summary['away_mean']:.1f} · Expected home margin {summary['expected_margin']:+.1f} · Expected total {summary['expected_total']:.1f}"
                    )
                    st.caption(
                        f"Independent tested scoring model: {score['model_kind']}. These are expected points, not a promised exact final score. Typical-score ranges (90% model distribution): {home} {summary['home_interval_90']}, {away} {summary['away_interval_90']}."
                    )
                else:
                    st.caption(
                        row.get(
                            "score_unavailable_reason",
                            "Projected score, spread and total: Not yet available — an approved scoring model must be configured.",
                        )
                    )
                with st.expander("Why this pick?"):
                    st.markdown("**Three strongest supporting inputs**")
                    for factor in explanation["support"]:
                        st.write("• " + factor["text"])
                    if len(explanation["support"]) < 3:
                        st.caption(
                            "Fewer than three supporting contributions are available; none are invented."
                        )
                    st.markdown("**How the opponent could win**")
                    for factor in explanation["risks"]:
                        st.write("• " + factor["text"])
                    st.caption(
                        "A contribution describes how this model uses an input; it is not a causal effect or a guaranteed advantage."
                    )
                interval = row.get("probability_interval", {})
                st.write(
                    f"Model uncertainty — {home}: {_interval(interval.get('home'))}; {away}: {_interval(interval.get('away'))}"
                )
                level = interval.get("confidence_level")
                st.caption(
                    f"{level:.0%} bootstrap model-estimation interval conditional on features; not calibrated predictive coverage."
                    if level is not None
                    else "Uncertainty unavailable for this archived point forecast."
                )
                st.caption(
                    f"Model: {row['model_version']} · Last updated: {_display_time(row.get('generated_at'))} · Input cutoff: {_display_time(row.get('as_of'))}"
                )
                cutoff = _timestamp(row.get("as_of"))
                if cutoff is None or (now - cutoff).total_seconds() > 3600:
                    st.warning(
                        "Forecast is stale: retained pregame prediction awaiting a worker refresh."
                    )
                if st.button("Analyze matchup", key=f"analyze_{index}"):
                    st.session_state["matchup_game"] = row["game_id"]
                    st.info(
                        "Matchup selected. Open Matchup Analysis for statistics and explanation."
                    )
        for row in pending:
            with st.container(border=True):
                st.markdown(f"### {team(row['away_team'])} at {team(row['home_team'])}")
                st.caption(f"Kickoff: {_display_time(row['kickoff'])}")
                st.warning(
                    "Prediction unavailable: model inputs or fitting could not be verified. No substitute market probability is used."
                )
    with analysis:
        if not rows:
            st.info("Matchup analysis appears when an independent forecast is available.")
        else:
            ids = [row["game_id"] for row in rows]
            if st.session_state.get("matchup_game") not in ids:
                st.session_state["matchup_game"] = ids[0]
            selected = st.selectbox(
                "Choose matchup",
                ids,
                key="matchup_game",
                format_func=lambda key: next(
                    f"{r.get('away_name', team(r['away_team']))} at {r.get('home_name', team(r['home_team']))}"
                    for r in rows
                    if r["game_id"] == key
                ),
            )
            row = next(r for r in rows if r["game_id"] == selected)
            if row.get("sport", "NFL") != "NFL":
                from predictions.league_ui import render_league_card, render_league_lines

                render_league_card(row)
                render_league_lines(row)
            else:
                st.write(verdict(row, team)["summary"])
                if row.get("score_prediction"):
                    render_score_lines(row["score_prediction"])
                with st.expander("Advanced Analysis", expanded=False):
                    render_analysis(row)
    with markets:
        st.subheader("Optional Polymarket US comparison")
        st.caption(
            "Executable prices and conditional EV are separate from our predictions. Recorded quotes may now be stale; strict PASS gates remain in force. No real-money execution."
        )
        _render_evaluations(
            [
                evaluation
                for evaluation in view["evaluations"]
                if evaluation.get("game_id") in {r["game_id"] for r in rows}
            ],
            view["forecasts"],
        )
        if any(r.get("sport", "NFL") != "NFL" for r in rows):
            st.info(
                "NBA/NHL market value: PASS — no verified executable quote or matching settlement rules are supported."
            )
        for value in view["evaluations"]:
            if value.get("fills"):
                st.write(
                    f"{value.get('game_id')} · retained executable ask depth used for conditional evaluation"
                )
                st.dataframe(pd.DataFrame(value["fills"]), hide_index=True)


def render_analysis(row):
    from predictions.dashboard import _display_time

    analysis = row.get("matchup_analysis")
    if not analysis:
        st.warning(
            "This archived prediction predates source-value explanations. Refresh through the worker; source values cannot be recovered from differences alone."
        )
        return
    home, away = team(row["home_team"]), team(row["away_team"])
    values = analysis["source_values"]
    st.subheader("Offense, defense and recent form")
    st.caption(
        "EPA is causally reconstructed remaining-half EPA. Defense values are allowed rates (lower is better). Elo supplies opponent-adjusted strength; EPA itself is not opponent-adjusted."
    )
    metrics = [
        "off_epa_per_play",
        "def_epa_per_play",
        "off_success_rate",
        "def_success_rate",
        "off_explosive_rate",
        "def_explosive_rate",
        "win_rate",
        "margin",
        "elo_rating",
        "qb_epa_per_dropback",
        "qb_success_rate",
        "rest_days",
        "games_played",
        "off_history_plays",
        "def_history_plays",
        "qb_history_dropbacks",
    ]
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Statistic": key.replace("_", " "),
                    home: values.get("home_" + key),
                    away: values.get("away_" + key),
                }
                for key in metrics
            ]
        ),
        hide_index=True,
    )
    st.write(
        f"Home field indicator: {row['features'].get('home_advantage')} (0 = neutral, 1 = home field)."
    )
    st.markdown("**Cross-matchup factors**")
    for attacking, defending in [("home", "away"), ("away", "home")]:
        for metric in ["epa_per_play", "success_rate", "explosive_rate"]:
            st.write(
                f"{team(row[attacking + '_team'])} offense {metric.replace('_', ' ')}: {values.get(attacking + '_off_' + metric, 'Unavailable')} · {team(row[defending + '_team'])} defense allowed: {values.get(defending + '_def_' + metric, 'Unavailable')}"
            )
    st.subheader("What drives this forecast?")
    st.caption(
        "Home-minus-away inputs standardized with training-only means/scales. Positive contributions favor the home team; negative contributions favor the away team relative to the fitted intercept. These are model associations, not causal effects."
    )
    contributions = analysis["contributions"]
    for sign, label in [(1, home), (-1, away)]:
        ranked = sorted(
            [r for r in contributions if sign * r["log_odds_contribution"] > 0],
            key=lambda r: abs(r["log_odds_contribution"]),
            reverse=True,
        )[:3]
        st.markdown(f"**Model factors favoring {label}**")
        for factor in ranked:
            st.write(
                f"{factor['feature'].replace('_', ' ')}: {factor['log_odds_contribution']:+.4f} home log-odds"
            )
        if not ranked:
            st.write("No contributions in this direction.")
    st.dataframe(pd.DataFrame(contributions), hide_index=True)
    st.write(
        f"Intercept {analysis['intercept']:+.6f} + contributions {sum(r['log_odds_contribution'] for r in contributions):+.6f} = home log-odds {analysis['home_log_odds']:+.6f}"
    )
    st.write(
        f"Logistic transform recovers home probability: {analysis['reconstructed_home_probability']:.6%}"
    )
    st.subheader("Sources, freshness and missing data")
    st.caption(
        f"Information cutoff: {_display_time(row['as_of'])} · Latest form result: {_display_time(row.get('history_available_through'))} · Latest PBP/Elo result: {_display_time(row.get('advanced_history_available_through'))}"
    )
    for side in ["home", "away"]:
        st.write(
            f"{team(row[side + '_team'])} QB proxy ID: {row.get(side + '_qb_id_proxy') or 'Unavailable'} · prior game: {row.get(side + '_qb_proxy_game_id') or 'Unavailable'} · known at: {_display_time(row.get(side + '_qb_proxy_available_at'))}"
        )
        if not values.get(side + "_games_played") or values.get(side + "_off_history_plays") == 0:
            st.warning(
                f"{team(row[side + '_team'])}: missing observed history; documented neutral defaults enter the model."
            )
        if not values.get(side + "_qb_history_dropbacks"):
            st.warning(
                f"{team(row[side + '_team'])}: QB performance is missing; neutral model defaults are shown."
            )
    for limitation in analysis["limitations"]:
        st.caption(limitation)
    with st.expander("Exact source values and provider provenance"):
        st.json(values)
        st.json(row.get("source_metadata", {}))


def render_score_lines(score):
    from predictions.scoring import outcome_probabilities, read_score_distribution

    st.subheader("Spread and total — choose an exact line")
    st.caption(
        "Research scenarios, not observed market lines. Full-game outcomes include overtime; pushes are separate. No prices or betting edge are invented."
    )
    spread = st.number_input(
        "Home point spread (negative means home favorite)",
        min_value=-100.0,
        max_value=100.0,
        value=-3.5,
        step=0.5,
    )
    total = st.number_input(
        "Combined-points line", min_value=0.0, max_value=200.0, value=44.5, step=0.5
    )
    try:
        probability = outcome_probabilities(
            read_score_distribution(score), home_spread=spread, total_line=total
        )
    except (ValueError, OSError):
        st.warning(
            "Retained scoring distribution is unavailable or fails checksum verification. No line probabilities are shown."
        )
        return
    st.write(
        f"Home covers: {probability['home_cover']['win']:.1%}; away covers: {probability['home_cover']['loss']:.1%}; push: {probability['home_cover']['push']:.1%}."
    )
    st.write(
        f"Over: {probability['over']['win']:.1%}; under: {probability['over']['loss']:.1%}; push: {probability['over']['push']:.1%}."
    )
    st.caption(
        "These estimates come from a tested historical scoring distribution, not from the binary winner probability. They omit parameter-refit uncertainty and require prospective calibration monitoring."
    )
    if float(spread).is_integer() or float(total).is_integer():
        st.warning(
            "Integer-line push estimates are poorly calibrated: for home −3, the recorded Ridge benchmark predicted a 2.8% push rate in 2025 versus 8.5% observed. NFL key scoring numbers are not modeled explicitly. Do not treat these scenario probabilities as a validated betting edge."
        )
    st.caption(
        "Historical line checks cover home −3/−3.5 and totals44/44.5 only. Other exact lines are unvalidated research scenarios."
    )
    st.info(
        "Market value: PASS — no verified executable spread/total quote, fees, liquidity and matching settlement rules are available. Existing verified moneyline comparisons remain separate in Market Value."
    )
