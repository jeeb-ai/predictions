"""Synthetic PBP fixtures exercise causal cutoff and side-orientation invariants."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from predictions.features import FEATURE_COLUMNS, build_features
from predictions.features_v2 import (
    ADVANCED_FEATURE_COLUMNS,
    MODEL2_FEATURE_COLUMNS,
    build_features_v2,
)

TEAM_COLUMNS = ["game_id", "team", "opponent", "plays", "epa_sum", "successes", "explosive_plays"]
QB_COLUMNS = ["game_id", "team", "qb_id", "dropbacks", "epa_sum", "successes"]


def game(game_id, date, home="A", away="B", home_score=21, away_score=10, neutral=0):
    kickoff = pd.Timestamp(date, tz="UTC")
    return {
        "game_id": game_id,
        "season": kickoff.year,
        "game_type": "REG",
        "week": 1,
        "kickoff": kickoff,
        "result_available_at": kickoff + pd.Timedelta(hours=24),
        "home_team": home,
        "away_team": away,
        "home_score": home_score,
        "away_score": away_score,
        "neutral": neutral,
    }


def team_row(game_id, team="A", opponent="B", plays=10, epa_sum=2, successes=6, explosive=1):
    return [game_id, team, opponent, plays, epa_sum, successes, explosive]


def qb_row(game_id, team="A", qb="QA", plays=10, epa_sum=3, successes=6):
    return [game_id, team, qb, plays, epa_sum, successes]


def sample():
    games = pd.DataFrame(
        [
            game("past", "2024-09-01 17:00"),
            game("target", "2024-09-08 17:00", home_score=10, away_score=21),
            game("future", "2024-09-15 17:00"),
        ]
    )
    teams = pd.DataFrame(
        [
            row
            for name in games.game_id
            for row in [team_row(name), team_row(name, "B", "A", 20, -4, 8, 2)]
        ],
        columns=TEAM_COLUMNS,
    )
    qbs = pd.DataFrame(
        [
            row
            for name in games.game_id
            for row in [qb_row(name), qb_row(name, "B", "QB", 10, -2, 4)]
        ],
        columns=QB_COLUMNS,
    )
    return games, teams, qbs


def test_v1_inputs_and_cutoffs_remain_unchanged():
    games, teams, qbs = sample()
    original = build_features(games)
    advanced = build_features_v2(games, teams, qbs)
    assert MODEL2_FEATURE_COLUMNS == FEATURE_COLUMNS + ADVANCED_FEATURE_COLUMNS
    assert_frame_equal(advanced[FEATURE_COLUMNS], original[FEATURE_COLUMNS])
    assert advanced.base_feature_version.equals(original.feature_version)
    assert advanced.history_available_through.equals(original.history_available_through)
    assert (advanced.feature_cutoff_at == advanced.kickoff).all()


def test_offense_defense_and_quarterback_rates_have_correct_sides():
    games, teams, qbs = sample()
    target = build_features_v2(games, teams, qbs).set_index("game_id").loc["target"]
    assert target.off_epa_per_play_diff == pytest.approx(0.4)
    assert target.def_epa_per_play_diff == pytest.approx(-0.4)
    assert target.off_success_rate_diff == pytest.approx(0.2)
    assert target.def_success_rate_diff == pytest.approx(-0.2)
    assert target.off_explosive_rate_diff == 0
    assert target.qb_epa_per_dropback_diff == pytest.approx(0.5)
    assert target.home_qb_id_proxy == "QA"
    assert target.away_qb_id_proxy == "QB"
    assert target.home_qb_proxy_game_id == "past"
    assert target.advanced_history_available_through == games.iloc[0].result_available_at


def test_current_and_future_pbp_qb_and_scores_cannot_change_earlier_features():
    games, teams, qbs = sample()
    expected = build_features_v2(games, teams, qbs).set_index("game_id")
    games.loc[games.game_id != "past", ["home_score", "away_score"]] = [99, 0]
    teams.loc[teams.game_id != "past", "epa_sum"] = 10000
    teams.loc[teams.game_id != "past", ["successes", "explosive_plays"]] = 0
    qbs.loc[qbs.game_id != "past", "qb_id"] = "FUTURE_STARTER"
    qbs.loc[qbs.game_id != "past", "epa_sum"] = 10000
    actual = build_features_v2(games, teams, qbs).set_index("game_id")
    columns = MODEL2_FEATURE_COLUMNS + ["home_qb_id_proxy", "away_qb_id_proxy"]
    assert_frame_equal(
        expected.loc[["past", "target"], columns], actual.loc[["past", "target"], columns]
    )
    assert expected.loc["target", "home_win"] == 0
    assert actual.loc["target", "home_win"] == 1


def test_exact_availability_cutoff_is_excluded_for_every_component():
    games, teams, qbs = sample()
    equal = game("equal", "2024-09-02 17:00", home_score=np.nan, away_score=np.nan)
    later = game("later", "2024-09-02 17:00:01", home_score=np.nan, away_score=np.nan)
    games = pd.concat([games.iloc[:1], pd.DataFrame([equal, later])], ignore_index=True)
    result = build_features_v2(games, teams, qbs).set_index("game_id")
    np.testing.assert_array_equal(result.loc["equal", ADVANCED_FEATURE_COLUMNS].to_numpy(float), 0)
    assert pd.isna(result.loc["equal", "advanced_history_available_through"])
    assert result.loc["equal", "home_qb_id_proxy"] is None
    assert result.loc["later", "off_epa_per_play_diff"] == pytest.approx(0.4)
    assert result.loc["later", "elo_strength_diff"] > 0
    assert result.loc["later", "home_qb_id_proxy"] == "QA"
    available = result.advanced_history_available_through.notna()
    assert (
        result.loc[available, "advanced_history_available_through"]
        < result.loc[available, "kickoff"]
    ).all()


def test_swapping_target_teams_reverses_all_advanced_differences():
    games, teams, qbs = sample()
    target = build_features_v2(games, teams, qbs).set_index("game_id").loc["target"]
    games.loc[1, ["home_team", "away_team", "home_score", "away_score"]] = ["B", "A", 21, 10]
    reverse = build_features_v2(games, teams, qbs).set_index("game_id").loc["target"]
    np.testing.assert_allclose(
        reverse[ADVANCED_FEATURE_COLUMNS].to_numpy(float),
        -target[ADVANCED_FEATURE_COLUMNS].to_numpy(float),
    )
    assert reverse.home_qb_id_proxy == target.away_qb_id_proxy
    assert reverse.home_win == 1


def test_rates_weight_plays_instead_of_averaging_game_rates():
    games = pd.DataFrame(
        [
            game("old", "2024-08-01", away="C"),
            game("recent", "2024-08-08", away="C"),
            game("target", "2024-08-15", away="D"),
        ]
    )
    teams = pd.DataFrame(
        [
            team_row("old", opponent="C", plays=10, epa_sum=10, successes=10, explosive=10),
            team_row("recent", opponent="C", plays=90, epa_sum=0, successes=0, explosive=0),
        ],
        columns=TEAM_COLUMNS,
    )
    target = build_features_v2(games, teams, pd.DataFrame(columns=QB_COLUMNS)).iloc[-1]
    assert target.off_epa_per_play_diff == pytest.approx(0.1)
    assert target.off_success_rate_diff == pytest.approx(0.1 - 0.45)
    assert target.off_explosive_rate_diff == 0
    assert target.off_history_plays_diff == 100
    assert target.home_off_history_games == 2
    last = build_features_v2(games, teams, pd.DataFrame(columns=QB_COLUMNS), window=1).iloc[-1]
    assert last.off_epa_per_play_diff == 0
    assert last.home_off_history_games == 1


def test_proxy_uses_last_observed_qb_and_follows_prior_player_history_after_trade():
    games = pd.DataFrame(
        [
            game("before_trade", "2024-08-01", home="C", away="D"),
            game("prior", "2024-08-08", home="A", away="B"),
            game("target", "2024-08-15", home="A", away="D"),
        ]
    )
    qbs = pd.DataFrame(
        [
            qb_row("before_trade", team="C", qb="TRADED", plays=20, epa_sum=10, successes=10),
            qb_row("prior", qb="TRADED", plays=10, epa_sum=0, successes=4),
            qb_row("prior", qb="BACKUP", plays=2, epa_sum=-2, successes=0),
            qb_row("target", qb="UNKNOWN_BEFORE_GAME", plays=30, epa_sum=99, successes=30),
        ],
        columns=QB_COLUMNS,
    )
    target = build_features_v2(games, pd.DataFrame(columns=TEAM_COLUMNS), qbs).iloc[-1]
    assert target.home_qb_id_proxy == "TRADED"
    assert target.home_qb_proxy_game_id == "prior"
    assert target.home_qb_history_games == 2
    assert target.home_qb_history_dropbacks == 30
    assert target.qb_epa_per_dropback_diff == pytest.approx(10 / 30)
    assert target.qb_success_rate_diff == pytest.approx(14 / 30 - 0.45)


def test_qb_usage_ties_are_deterministic_and_never_target_starters():
    games, teams, qbs = sample()
    qbs = pd.concat([qbs, pd.DataFrame([qb_row("past", qb="AA", plays=10)], columns=QB_COLUMNS)])
    result = build_features_v2(games, teams, qbs).set_index("game_id")
    assert result.loc["target", "home_qb_id_proxy"] == "AA"
    shuffled = build_features_v2(
        games.sample(frac=1, random_state=2),
        teams.sample(frac=1, random_state=3),
        qbs.sample(frac=1, random_state=4),
    ).set_index("game_id")
    assert_frame_equal(result, shuffled)


def test_missing_history_has_explicit_finite_defaults_and_missingness():
    games = pd.DataFrame([game("past", "2024-08-01", away="C"), game("target", "2024-08-08")])
    qbs = pd.DataFrame([qb_row("past")], columns=QB_COLUMNS)
    result = build_features_v2(games, pd.DataFrame(columns=TEAM_COLUMNS), qbs)
    assert np.isfinite(result[MODEL2_FEATURE_COLUMNS].to_numpy()).all()
    assert result.iloc[-1].qb_proxy_missing_diff == -1
    assert result.iloc[-1].home_off_history_plays == 0
    assert result.iloc[0].home_qb_id_proxy is None


def test_unfinished_or_unknown_game_pbp_does_not_enter_history():
    games, teams, qbs = sample()
    games.loc[games.game_id == "past", ["home_score", "away_score"]] = np.nan
    teams = pd.concat([teams, pd.DataFrame([team_row("cancelled")], columns=TEAM_COLUMNS)])
    qbs = pd.concat([qbs, pd.DataFrame([qb_row("cancelled")], columns=QB_COLUMNS)])
    target = build_features_v2(games, teams, qbs).set_index("game_id").loc["target"]
    np.testing.assert_array_equal(target[ADVANCED_FEATURE_COLUMNS].to_numpy(float), 0)
    assert target.home_qb_id_proxy is None


def test_elo_strength_depends_on_previously_faced_opponent_strength():
    games = pd.DataFrame(
        [
            game("one", "2024-08-01", home="C", away="D", neutral=1),
            game("two", "2024-08-08", home="A", away="C", neutral=1),
            game("target", "2024-08-15", home="A", away="B", neutral=1),
        ]
    )
    empty_team, empty_qb = pd.DataFrame(columns=TEAM_COLUMNS), pd.DataFrame(columns=QB_COLUMNS)
    strong = build_features_v2(games, empty_team, empty_qb).iloc[-1]
    games.loc[0, ["home_score", "away_score"]] = [10, 21]
    weak = build_features_v2(games, empty_team, empty_qb).iloc[-1]
    assert strong.elo_strength_diff > weak.elo_strength_diff
    assert strong.home_elo_rating > weak.home_elo_rating
    assert strong.advanced_history_available_through < strong.kickoff


def test_neutral_tie_does_not_change_equal_elo_ratings():
    games = pd.DataFrame(
        [
            game("tie", "2024-08-01", home_score=10, away_score=10, neutral=1),
            game("target", "2024-08-08"),
        ]
    )
    result = build_features_v2(
        games, pd.DataFrame(columns=TEAM_COLUMNS), pd.DataFrame(columns=QB_COLUMNS)
    )
    assert result.iloc[-1].home_elo_rating == 1500
    assert result.iloc[-1].away_elo_rating == 1500
    assert result.iloc[-1].elo_strength_diff == 0
    assert pd.isna(result.iloc[0].home_win)


@pytest.mark.parametrize("quarterback", [False, True])
def test_duplicate_aggregate_keys_rejected(quarterback):
    games, teams, qbs = sample()
    if quarterback:
        qbs = pd.concat([qbs, qbs.iloc[[0]]])
    else:
        teams = pd.concat([teams, teams.iloc[[0]]])
    with pytest.raises(ValueError, match="Duplicate"):
        build_features_v2(games, teams, qbs)


@pytest.mark.parametrize(
    "field,value",
    [
        ("plays", 0),
        ("plays", 1.5),
        ("successes", 999),
        ("explosive_plays", -1),
        ("epa_sum", np.inf),
    ],
)
def test_invalid_aggregate_statistics_rejected(field, value):
    games, teams, qbs = sample()
    teams[field] = teams[field].astype(float)
    teams.loc[0, field] = value
    with pytest.raises(ValueError, match="aggregate"):
        build_features_v2(games, teams, qbs)


def test_mismatched_schedule_identity_rejected():
    games, teams, qbs = sample()
    teams.loc[0, "opponent"] = "C"
    with pytest.raises(ValueError, match="opponent"):
        build_features_v2(games, teams, qbs)
    teams.loc[0, "team"] = "C"
    with pytest.raises(ValueError, match="team"):
        build_features_v2(games, teams, qbs)


def test_empty_schedule_preserves_v2_feature_schema():
    games, teams, qbs = sample()
    empty = build_features_v2(games.iloc[:0], teams.iloc[:0], qbs.iloc[:0])
    assert empty.empty
    assert set(MODEL2_FEATURE_COLUMNS + ["advanced_history_available_through"]) <= set(empty)
