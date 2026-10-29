"""Faithful explanations with no invented factors or unearned confidence."""

import math

import pytest

from predictions.verdict import verdict


def forecast(pick_home=True):
    sign = 1 if pick_home else -1
    factors = [
        {"feature": name, "raw_value": 1, "log_odds_contribution": sign * value}
        for name, value in [
            ("elo_strength_diff", 0.7),
            ("off_epa_per_play_diff", 0.4),
            ("qb_epa_per_dropback_diff", 0.2),
            ("rest_days_diff", -0.1),
            ("margin_diff", -0.05),
        ]
    ]
    z = sum(f["log_odds_contribution"] for f in factors)
    return {
        "home_team": "HOME",
        "away_team": "AWAY",
        "home_win_probability": 1 / (1 + math.exp(-z)),
        "matchup_analysis": {
            "intercept": 0,
            "home_log_odds": z,
            "contributions": factors,
            "source_values": {"home_elo_rating": 1550, "away_elo_rating": 1500},
        },
        "probability_interval": {
            "home": {"lower": 0.4, "upper": 0.8},
            "away": {"lower": 0.2, "upper": 0.6},
        },
    }


@pytest.mark.parametrize("home", [True, False])
def test_ranked_support_and_opposing_risks_follow_actual_contributions(home):
    row = forecast(home)
    result = verdict(row)
    assert [f["feature"] for f in result["support"]] == [
        "elo_strength_diff",
        "off_epa_per_play_diff",
        "qb_epa_per_dropback_diff",
    ]
    assert [f["feature"] for f in result["risks"]] == ["rest_days_diff", "margin_diff"]
    assert all(f["winner_log_odds_contribution"] > 0 for f in result["support"])
    assert all(f["winner_log_odds_contribution"] < 0 for f in result["risks"])
    assert ("HOME" if home else "AWAY") + " is the model" in result["summary"]
    assert "not a confirmed starter" in result["support"][2]["text"]
    assert "crosses an even chance" in result["confidence"]
    assert "do not prove future accuracy" in result["confidence"]


def test_legacy_unknown_explanation_does_not_invent_strengths():
    row = forecast()
    row.pop("matchup_analysis")
    row.pop("probability_interval")
    result = verdict(row)
    assert not result["support"] and not result["risks"]
    assert "No supporting feature explanation" in result["summary"]
    assert "No model-refit uncertainty" in result["confidence"]


def test_mismatched_contributions_cannot_be_narrated():
    row = forecast()
    row["matchup_analysis"]["contributions"][0]["log_odds_contribution"] = 3
    with pytest.raises(ValueError, match="does not match"):
        verdict(row)
