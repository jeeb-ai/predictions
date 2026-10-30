"""Hockey-specific prior xG, tracked-shot goalie proxies and outcome scopes."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from predictions.league_model import fit_league, line_probabilities, pmf_for, summarize

NHL_STATS = [
    "xg_for",
    "xg_against",
    "shot_quality",
    "shots_for",
    "shots_against",
    "pp_xg",
    "pk_xg",
    "form",
    "rest",
    "back_to_back",
    "elo",
    "goalie_save",
    "goalie_shots",
    "history",
]
NHL_COLUMNS = [side + "_" + name for side in ["home", "away"] for name in NHL_STATS]


def nhl_features(games, goalies, as_of=None, window=10):
    games = games.sort_values(["kickoff", "game_id"])
    history = {}
    scheduled = {}
    ratings = {}
    pending = []
    output = []
    by_goalie = {str(k): v.sort_values("kickoff") for k, v in goalies.groupby("goalie")}
    for game in games.to_dict("records"):
        cutoff = min(game["kickoff"], pd.Timestamp(as_of)) if as_of is not None else game["kickoff"]
        ready = [p for p in pending if p["available_at"] < cutoff]
        pending = [p for p in pending if p["available_at"] >= cutoff]
        for past in ready:
            h, a = past["home_team"], past["away_team"]
            rh, ra = ratings.get(h, 1500.0), ratings.get(a, 1500.0)
            expected = 1 / (1 + 10 ** ((ra - rh - 40) / 400))
            change = 16 * (past["home_win"] - expected)
            ratings[h] = rh + change
            ratings[a] = ra - change
        row = {**game}
        used = []
        for side in ["home", "away"]:
            team = game[side + "_team"]
            past = [g for g in history.get(team, []) if g["available_at"] < cutoff][-window:]

            def value(g, key):
                return g[("home" if g["home_team"] == team else "away") + "_" + key]

            values = {
                key: float(np.mean([value(g, key) for g in past])) if past else default
                for key, default in [
                    ("xg_for", 2.5),
                    ("xg_against", 2.5),
                    ("shot_quality", 0.07),
                    ("shots_for", 35),
                    ("shots_against", 35),
                    ("pp_xg", 0.5),
                    ("pk_xg", 0.5),
                ]
            }
            previous = scheduled.get(team)
            day = pd.Timestamp(
                game.get("schedule_day", game["kickoff"].tz_convert("America/New_York").date())
            )
            days = (day - previous).days if previous is not None else 7
            proxy = value(past[-1], "goalie_id") if past else None
            name = value(past[-1], "goalie_name") if past else "Unknown"
            appearances = by_goalie.get(
                str(proxy), pd.DataFrame(columns=["available_at", "shots_on_goal", "goals"])
            )
            appearances = appearances.loc[appearances.available_at.lt(cutoff)].tail(window)
            shots = float(appearances.shots_on_goal.sum())
            goals = float(appearances.goals.sum())
            wins = [g["home_win"] if g["home_team"] == team else 1 - g["home_win"] for g in past]
            values.update(
                form=float(np.mean(wins)) if wins else 0.5,
                rest=min(days, 14),
                back_to_back=float(days == 1),
                elo=ratings.get(team, 1500.0) / 400,
                goalie_save=1 - goals / shots if shots else 0.9,
                goalie_shots=shots,
                history=len(past),
            )
            row.update({side + "_" + key: float(value) for key, value in values.items()})
            row[side + "_source_games"] = [g["game_id"] for g in past]
            row[side + "_goalie_proxy"] = {
                "id": proxy,
                "name": name,
                "status": "prior most-tracked-shots goalie, not confirmed starter",
                "source_game": past[-1]["game_id"] if past else None,
                "tracked_shots": shots,
            }
            used.extend(g["available_at"] for g in past)
            used.extend(appearances.available_at.tolist())
        row["feature_cutoff_at"] = cutoff
        row["history_available_through"] = max(used) if used else None
        output.append(row)
        if pd.notna(game["home_score"]) and (
            as_of is None or game["available_at"] < pd.Timestamp(as_of)
        ):
            history.setdefault(game["home_team"], []).append(game)
            history.setdefault(game["away_team"], []).append(game)
            pending.append(game)
        scheduled[game["home_team"]] = day
        scheduled[game["away_team"]] = day
    return pd.DataFrame(output)


@threadpool_limits.wrap(limits=2)
def fit_nhl(train, cal, columns, sport):
    model = fit_league(train, cal, columns, sport)
    base = np.maximum(
        np.column_stack([m.predict(cal[columns]) for m in model["score_models"]]), 0.05
    )
    actual = cal[["home_score", "away_score"]].to_numpy()
    model["goal_scale"] = actual.sum(axis=0) / base.sum(axis=0)
    scaled = base * model["goal_scale"]
    grid = np.arange(21)
    draw = np.array([np.sum(poisson.pmf(grid, m[0]) * poisson.pmf(grid, m[1])) for m in scaled])
    observed = cal.home_score.eq(cal.away_score).to_numpy()

    def objective(log_weight):
        weight = np.exp(log_weight)
        return (
            np.mean(np.log(1 - draw + weight * draw) - observed * log_weight) + 0.01 * log_weight**2
        )

    model["draw_multiplier"] = float(
        np.exp(minimize_scalar(objective, bounds=(-1.5, 1.5), method="bounded").x)
    )
    baseline_draw = np.sum(
        poisson.pmf(grid, model["baseline_means"][0])
        * poisson.pmf(grid, model["baseline_means"][1])
    )
    rate = float(observed.mean())
    model["baseline_draw_multiplier"] = float(
        np.clip(rate * (1 - baseline_draw) / max(baseline_draw * (1 - rate), 1e-10), 0.2, 5)
    )
    tied = cal.loc[cal.went_extra]
    model["shootout_share"] = (float(tied.outcome_type.eq("SO").sum()) + 1) / (len(tied) + 2)
    model["extra_models"] = {}
    for kind in ["OT", "SO"]:
        subset = train.loc[train.outcome_type.eq(kind)]
        cal_subset = cal.loc[cal.outcome_type.eq(kind)]
        if len(subset) < 30 or subset.home_win.nunique() < 2:
            model["extra_models"][kind] = {
                "constant": (float(subset.home_win.sum()) + 1) / (len(subset) + 2)
            }
            continue
        estimator = make_pipeline(
            StandardScaler(), LogisticRegression(C=0.2, max_iter=2000, random_state=42)
        ).fit(subset[columns], subset.home_win)
        calibrator = None
        if len(cal_subset) >= 20 and cal_subset.home_win.nunique() == 2:
            calibrator = LogisticRegression(C=1, max_iter=2000, random_state=42).fit(
                estimator.decision_function(cal_subset[columns]).reshape(-1, 1), cal_subset.home_win
            )
        model["extra_models"][kind] = {"estimator": estimator, "calibrator": calibrator}
    return model


def nhl_distribution(model, means, row, baseline=False):
    reg = pmf_for(model, means, baseline)
    draw_weight = model["baseline_draw_multiplier"] if baseline else model["draw_multiplier"]
    reg = reg.copy()
    idx = np.arange(len(reg))
    reg[idx, idx] *= draw_weight
    reg /= reg.sum()
    probabilities = {}
    values = pd.DataFrame([{key: float(row[key]) for key in model["columns"]}])
    for kind, item in model["extra_models"].items():
        if "constant" in item:
            probabilities[kind] = item["constant"]
        elif item["calibrator"] is None:
            probabilities[kind] = float(item["estimator"].predict_proba(values)[0, 1])
        else:
            probabilities[kind] = float(
                item["calibrator"].predict_proba(
                    item["estimator"].decision_function(values).reshape(-1, 1)
                )[0, 1]
            )
    extra_home = (1 - model["shootout_share"]) * probabilities["OT"] + model[
        "shootout_share"
    ] * probabilities["SO"]
    if baseline:
        extra_home = 0.5
    full = np.zeros((22, 22))
    full[:21, :21] = reg
    full[idx, idx] = 0
    full[idx + 1, idx] += np.diag(reg) * extra_home
    full[idx, idx + 1] += np.diag(reg) * (1 - extra_home)
    h, a = np.indices(reg.shape)
    draw = float(np.trace(reg))
    hreg = float(reg[h > a].sum())
    areg = float(reg[h < a].sum())
    outcomes = {
        "regulation_home_win": hreg,
        "regulation_draw": draw,
        "regulation_away_win": areg,
        "extra_time_probability": draw,
        "conditional_shootout_share": model["shootout_share"],
        "conditional_home_ot_win": probabilities["OT"],
        "conditional_home_shootout_win": probabilities["SO"],
        "score_model_full_home_win": hreg + draw * extra_home,
        "score_model_full_away_win": areg + draw * (1 - extra_home),
    }
    return full, reg, outcomes


def evaluate_nhl(model, test, spread, total):
    # Winner model is graded against full-game winners, score engine against official final scores.
    pred = np.column_stack([m.predict(test[model["columns"]]) for m in model["score_models"]])
    result = []
    p = model["platt"].predict_proba(
        model["winner"].decision_function(test[model["columns"]]).reshape(-1, 1)
    )[:, 1]
    for (_, row), means, winprob in zip(test.iterrows(), pred, p, strict=True):
        for baseline in [False, True]:
            full, reg, outcomes = nhl_distribution(model, means, row, baseline)
            s = summarize(full)
            h, a = int(row.home_final_score), int(row.away_final_score)
            win = int(row.home_win)
            prob = model["baseline_win"] if baseline else winprob
            lines = line_probabilities(full, spread, total)
            target_reg = (
                0
                if row.home_score > row.away_score
                else 1
                if row.home_score == row.away_score
                else 2
            )
            regprobs = np.array(
                [
                    outcomes["regulation_home_win"],
                    outcomes["regulation_draw"],
                    outcomes["regulation_away_win"],
                ]
            )
            grid = np.arange(len(full))
            record = {
                "game_id": row.game_id,
                "season": int(row.season),
                "date": str(row.kickoff.date()),
                "kind": "baseline" if baseline else "model",
                "probability": float(prob),
                "outcome": win,
                "accuracy": float((prob >= 0.5) == win),
                "brier": float((prob - win) ** 2),
                "log_loss": float(-np.log(max(prob if win else 1 - prob, 1e-15))),
                "score_mae": float((abs(s["home_mean"] - h) + abs(s["away_mean"] - a)) / 2),
                "score_mse": float(((s["home_mean"] - h) ** 2 + (s["away_mean"] - a) ** 2) / 2),
                "joint_log_loss": float(-np.log(max(full[h, a], 1e-300))),
                "margin_mae": float(abs(s["expected_margin"] - h + a)),
                "total_mae": float(abs(s["expected_total"] - h - a)),
                "coverage_90": float(
                    (
                        int(s["home_interval_90"][0] <= h <= s["home_interval_90"][1])
                        + int(s["away_interval_90"][0] <= a <= s["away_interval_90"][1])
                    )
                    / 2
                ),
                "score_crps": float(
                    (
                        np.sum((full.sum(axis=1).cumsum() - (grid >= h)) ** 2)
                        + np.sum((full.sum(axis=0).cumsum() - (grid >= a)) ** 2)
                    )
                    / 2
                ),
                "regulation_brier3": float(np.sum((regprobs - np.eye(3)[target_reg]) ** 2)),
                "regulation_log_loss": float(-np.log(max(regprobs[target_reg], 1e-15))),
                "regulation_draw_probability": outcomes["regulation_draw"],
                "regulation_draw_outcome": int(row.went_extra),
                "shootout_probability": outcomes["regulation_draw"]
                * outcomes["conditional_shootout_share"],
                "shootout_outcome": int(row.outcome_type == "SO"),
            }
            for label, value, events in [
                ("spread", h - a + spread, lines["home_cover"]),
                ("total", h + a - total, lines["over"]),
            ]:
                target = "win" if value > 0 else "loss" if value < 0 else "push"
                record[label + "_brier3"] = sum((events[k] - int(k == target)) ** 2 for k in events)
                record[label + "_probability"] = events["win"]
                record[label + "_outcome"] = int(target == "win")
                record[label + "_push_probability"] = events["push"]
                record[label + "_push_outcome"] = int(target == "push")
            result.append(record)
    return pd.DataFrame(result)


def nhl_score_builder(model, means, row):
    full, reg, outcomes = nhl_distribution(model, means, row)
    return full, {
        "regulation_array": reg,
        "regulation_summary": summarize(reg),
        "score_model_outcomes": outcomes,
        "outcome_rules": {
            "regulation": "60 minutes, ties possible",
            "full_game": "Includes overtime/shootout; official score awards one deciding goal. Market conventions may differ.",
        },
    }


def nhl_slate(root):
    import json

    from predictions.league_data import snapshot

    now = pd.Timestamp.now(tz="UTC")
    day = now.tz_convert("America/Chicago").normalize()
    rows = {}
    sources = []
    for offset in [-7, 0, 7]:
        date = (day + pd.Timedelta(days=offset)).strftime("%Y-%m-%d")
        path, meta = snapshot("https://api-web.nhle.com/v1/schedule/" + date, root, refresh=True)
        sources.append(meta)
        for entry in json.loads(path.read_text())["gameWeek"]:
            for game in entry["games"]:
                if game["gameType"] != 2 or game.get("gameScheduleState", "OK") != "OK":
                    continue
                kickoff = pd.Timestamp(game["startTimeUTC"])
                if kickoff > now + pd.Timedelta(days=7):
                    continue
                h, a = game["homeTeam"], game["awayTeam"]
                rows[str(game["id"])] = {
                    "sport": "NHL",
                    "game_id": str(game["id"]),
                    "season": int(str(game["season"])[:4]),
                    "kickoff": kickoff,
                    "schedule_day": game["gameDate"]
                    if "gameDate" in game
                    else kickoff.tz_convert("America/New_York").date().isoformat(),
                    "available_at": kickoff + pd.Timedelta(hours=24),
                    "home_team": h["abbrev"],
                    "away_team": a["abbrev"],
                    "home_name": h.get("placeName", {}).get("default", h["abbrev"])
                    + " "
                    + h.get("commonName", {}).get("default", ""),
                    "away_name": a.get("placeName", {}).get("default", a["abbrev"])
                    + " "
                    + a.get("commonName", {}).get("default", ""),
                    "home_score": np.nan,
                    "away_score": np.nan,
                    "home_win": np.nan,
                    "outcome_type": "unknown",
                }
    return pd.DataFrame(rows.values()), sources
