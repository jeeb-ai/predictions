"""Deterministic everyday-language summaries of verified logistic contributions."""

import math

LABELS = {
    "win_rate_diff": ("recent results", "win_rate", "share of recent games won"),
    "margin_diff": ("recent scoring balance", "margin", "average points scored minus allowed"),
    "points_for_diff": ("recent scoring", "points_for", "average points scored"),
    "points_against_diff": ("recent defense", "points_against", "average points allowed"),
    "rest_days_diff": ("time between games", "rest_days", "days since the last eligible game"),
    "games_played_diff": ("available history", "games_played", "recent games available"),
    "off_epa_per_play_diff": (
        "offensive efficiency",
        "off_epa_per_play",
        "estimated points gained per play",
    ),
    "def_epa_per_play_diff": (
        "defensive efficiency",
        "def_epa_per_play",
        "estimated points allowed per play",
    ),
    "off_success_rate_diff": (
        "consistent offense",
        "off_success_rate",
        "share of plays improving scoring prospects",
    ),
    "def_success_rate_diff": (
        "consistent defense",
        "def_success_rate",
        "opponent share of plays improving scoring prospects",
    ),
    "off_explosive_rate_diff": (
        "big offensive plays",
        "off_explosive_rate",
        "share of runs gaining 10+ yards or passes gaining 20+",
    ),
    "def_explosive_rate_diff": (
        "big plays allowed",
        "def_explosive_rate",
        "opponent share of runs gaining 10+ yards or passes gaining 20+",
    ),
    "qb_epa_per_dropback_diff": (
        "previous passer performance",
        "qb_epa_per_dropback",
        "estimated points gained per passing play",
    ),
    "qb_success_rate_diff": (
        "previous passer consistency",
        "qb_success_rate",
        "share of passing plays improving scoring prospects",
    ),
    "elo_strength_diff": (
        "opponent-adjusted results",
        "elo_rating",
        "team strength rating from earlier results",
    ),
    "off_history_plays_diff": (
        "offensive sample size",
        "off_history_plays",
        "observed offensive plays",
    ),
    "def_history_plays_diff": (
        "defensive sample size",
        "def_history_plays",
        "observed defensive plays",
    ),
    "qb_history_dropbacks_diff": (
        "passer sample size",
        "qb_history_dropbacks",
        "observed passing plays",
    ),
    "qb_proxy_missing_diff": (
        "missing passer history",
        "qb_proxy_missing",
        "missing-passer indicator (1 means missing)",
    ),
}


def verdict(row, name=lambda value: value):
    p = row["home_win_probability"]
    side = "home" if p >= 0.5 else "away"
    opponent = "away" if side == "home" else "home"
    winner, other = name(row[side + "_team"]), name(row[opponent + "_team"])
    analysis = row.get("matchup_analysis") or {}
    factors = analysis.get("contributions", [])
    if factors:
        z = analysis["intercept"] + sum(f["log_odds_contribution"] for f in factors)
        if not math.isclose(z, analysis["home_log_odds"], abs_tol=1e-9) or not math.isclose(
            1 / (1 + math.exp(-z)), p, abs_tol=1e-9
        ):
            raise ValueError("Recorded explanation does not match this prediction")
    values = analysis.get("source_values", {})
    direction = 1 if side == "home" else -1
    explained = []
    for factor in factors:
        feature = factor["feature"]
        impact = direction * factor["log_odds_contribution"]
        if feature == "home_advantage":
            text = f"Home field: {'the game is at ' + name(row['home_team']) if factor['raw_value'] else 'this is a neutral venue'}. The fitted home-field input favors {winner if impact > 0 else other}."
        elif feature in LABELS:
            label, key, definition = LABELS[feature]
            a, b = values.get(side + "_" + key), values.get(opponent + "_" + key)
            if a is None or b is None:
                text = f"{label.capitalize()}: the recorded model input favors {winner if impact > 0 else other}; individual team values are unavailable."
            else:
                rate = key in {
                    "win_rate",
                    "off_success_rate",
                    "def_success_rate",
                    "off_explosive_rate",
                    "def_explosive_rate",
                    "qb_success_rate",
                }

                def fmt(v):
                    return f"{v:.0%}" if rate else f"{v:.2f}"

                text = f"{label.capitalize()}: {winner} {fmt(a)}, {other} {fmt(b)} ({definition}). The model weights this input toward {winner if impact > 0 else other}."
            if key.startswith("qb_"):
                text += " This is a prior observed passer, not a confirmed starter."
            if key.startswith(("off_", "def_", "qb_")) and any(
                values.get(
                    s
                    + "_"
                    + (
                        "qb_history_dropbacks"
                        if key.startswith("qb_")
                        else key[:3] + "_history_plays"
                    )
                )
                == 0
                for s in ["home", "away"]
            ):
                text += " Missing history uses a neutral default; this is a data limitation."
        else:
            continue
        explained.append({"feature": feature, "winner_log_odds_contribution": impact, "text": text})
    support = sorted(
        [f for f in explained if f["winner_log_odds_contribution"] > 0],
        key=lambda f: -f["winner_log_odds_contribution"],
    )[:3]
    risks = sorted(
        [f for f in explained if f["winner_log_odds_contribution"] < 0],
        key=lambda f: f["winner_log_odds_contribution"],
    )[:2]
    probability = max(p, 1 - p)
    interval = row.get("probability_interval", {}).get(side)
    confidence = f"The model gives {other} a {1 - probability:.0%} chance of winning a decisive game, so this is not a certainty."
    if interval:
        confidence += f" Refitting on resampled historical weeks puts {winner}'s estimate between {interval['lower']:.0%} and {interval['upper']:.0%}."
        if interval["lower"] <= 0.5 <= interval["upper"]:
            confidence += " That range crosses an even chance: the pick is sensitive to which training games are used."
    else:
        confidence += " No model-refit uncertainty range was retained for this forecast."
    confidence += " These ranges do not prove future accuracy; injuries and the upcoming starting lineup are not modeled."
    lead = (
        f"{winner} is the model’s pick at {probability:.0%} for a game that does not finish tied."
        if p != 0.5
        else "The model sees an even matchup; neither team is favored."
    )
    summary = lead + (
        " The largest supporting input is " + support[0]["text"].split(":")[0].lower() + "."
        if support
        else " No supporting feature explanation is available for this record."
    )
    summary += (
        " The strongest counterweight is " + risks[0]["text"].split(":")[0].lower() + "."
        if risks
        else " The opponent can still win; no opposing feature contribution is available in this record."
    )
    summary += " These are patterns in past games, not proof that a statistic causes a win."
    return {"summary": summary, "support": support, "risks": risks, "confidence": confidence}
