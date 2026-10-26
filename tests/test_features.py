"""Synthetic fixtures test information cutoffs and team-side orientation."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from predictions.features import FEATURE_COLUMNS, build_features


def game(
    game_id,
    kickoff,
    home_team="A",
    away_team="B",
    home_score=21,
    away_score=7,
    neutral=0,
):
    kickoff = pd.Timestamp(kickoff, tz="UTC")
    return {
        "game_id": game_id,
        "season": kickoff.year,
        "game_type": "REG",
        "week": 1,
        "kickoff": kickoff,
        "result_available_at": kickoff + pd.Timedelta(hours=24),
        "home_team": home_team,
        "away_team": away_team,
        "home_score": home_score,
        "away_score": away_score,
        "neutral": neutral,
    }


def fixture():
    return pd.DataFrame(
        [
            game("past", "2024-09-01 17:00"),
            game("target", "2024-09-08 17:00", home_score=10, away_score=20),
            game("future", "2024-09-15 17:00", home_score=30, away_score=0),
        ]
    )


def test_current_and_future_scores_cannot_change_pregame_features():
    original = fixture()
    changed = original.copy()
    changed.loc[changed.game_id != "past", ["home_score", "away_score"]] = [99, 0]
    before = build_features(original).set_index("game_id")
    after = build_features(changed).set_index("game_id")
    assert_frame_equal(
        before.loc[["past", "target"], FEATURE_COLUMNS],
        after.loc[["past", "target"], FEATURE_COLUMNS],
    )
    assert before.loc["target", "home_win"] == 0
    assert after.loc["target", "home_win"] == 1


def test_cutoff_is_strict_and_records_used_information():
    games = pd.DataFrame(
        [
            game("past", "2024-09-01 17:00"),
            game("equal", "2024-09-02 17:00", home_score=np.nan, away_score=np.nan),
            game("after", "2024-09-02 17:00:01", home_score=np.nan, away_score=np.nan),
        ]
    )
    result = build_features(games).set_index("game_id")
    assert result.loc["equal", "win_rate_diff"] == 0
    assert pd.isna(result.loc["equal", "history_available_through"])
    assert result.loc["after", "win_rate_diff"] == 1
    assert result.loc["after", "history_available_through"] == games.loc[0, "result_available_at"]
    assert (result.feature_cutoff_at == result.kickoff).all()
    used = result.history_available_through.notna()
    assert (
        result.loc[used, "history_available_through"] < result.loc[used, "feature_cutoff_at"]
    ).all()


def test_shuffled_input_is_deterministic_and_does_not_mutate_input():
    games = fixture()
    copy = games.copy(deep=True)
    expected = build_features(games)
    shuffled = build_features(games.sample(frac=1, random_state=42))
    assert_frame_equal(expected, shuffled)
    assert_frame_equal(games, copy)


def test_simultaneous_games_cannot_use_each_others_results():
    games = pd.DataFrame(
        [
            game("one", "2024-09-01 17:00"),
            game("two", "2024-09-01 17:00", home_team="A", away_team="C"),
        ]
    )
    result = build_features(games)
    assert (result.win_rate_diff == 0).all()
    assert (result.margin_diff == 0).all()
    assert result.history_available_through.isna().all()


def test_home_away_features_and_labels_have_correct_orientation():
    games = pd.DataFrame(
        [
            game("past", "2024-09-01 17:00"),
            game(
                "target",
                "2024-09-08 17:00",
                home_team="B",
                away_team="A",
                home_score=10,
                away_score=20,
            ),
        ]
    )
    result = build_features(games).set_index("game_id")
    target = result.loc["target"]
    assert target.home_win == 0
    assert target.win_rate_diff == -1
    assert target.margin_diff == -28
    assert target.points_for_diff == -14
    assert target.points_against_diff == 14

    swapped = games.copy()
    swapped.loc[1, ["home_team", "away_team", "home_score", "away_score"]] = [
        "A",
        "B",
        20,
        10,
    ]
    reverse = build_features(swapped).set_index("game_id").loc["target"]
    assert reverse.home_win == 1
    np.testing.assert_allclose(
        reverse[FEATURE_COLUMNS[:-1]].to_numpy(dtype=float),
        -target[FEATURE_COLUMNS[:-1]].to_numpy(dtype=float),
    )


def test_ties_have_no_binary_label_but_contribute_half_a_past_win():
    games = pd.DataFrame(
        [
            game("tie", "2024-09-01 17:00", home_score=17, away_score=17),
            game("next", "2024-09-08 17:00", away_team="C"),
        ]
    )
    result = build_features(games).set_index("game_id")
    assert pd.isna(result.loc["tie", "home_win"])
    assert result.loc["next", "win_rate_diff"] == 0
    assert result.loc["next", "games_played_diff"] == 1
    assert result.loc["next", "points_for_diff"] == -4


def test_rolling_window_uses_only_last_eligible_games():
    games = pd.DataFrame(
        [
            game("old", "2023-12-01 17:00", home_score=70, away_score=0),
            game("recent", "2024-09-01 17:00", home_score=7, away_score=21),
            game("target", "2024-09-08 17:00"),
        ]
    )
    target = build_features(games, window=1).set_index("game_id").loc["target"]
    assert target.win_rate_diff == -1
    assert target.margin_diff == -28
    assert target.feature_version.endswith("-w1")


def test_defaults_neutral_venue_and_offseason_rest_cap():
    games = pd.DataFrame(
        [
            game("past", "2023-12-01 17:00", away_team="C", neutral=1),
            game("target", "2024-09-01 17:00"),
        ]
    )
    result = build_features(games).set_index("game_id")
    np.testing.assert_array_equal(
        result.loc["past", FEATURE_COLUMNS].to_numpy(dtype=float), np.zeros(7)
    )
    assert result.loc["target", "rest_days_diff"] == 23
    assert result.loc["target", "home_advantage"] == 1
    assert np.isfinite(result[FEATURE_COLUMNS].to_numpy()).all()


@pytest.mark.parametrize("window", [0, -1, 1.5, True])
def test_invalid_window_rejected(window):
    with pytest.raises(ValueError, match="positive integer"):
        build_features(fixture(), window=window)


def test_invalid_canonical_input_rejected():
    with pytest.raises(ValueError, match="Missing canonical"):
        build_features(pd.DataFrame())
    games = fixture()
    with pytest.raises(ValueError, match="Duplicate"):
        build_features(pd.concat([games, games.iloc[[0]]]))
    games.loc[0, "result_available_at"] = games.loc[0, "kickoff"]
    with pytest.raises(ValueError, match="strictly after"):
        build_features(games)


def test_empty_canonical_frame_has_feature_schema():
    games = fixture().iloc[:0]
    result = build_features(games)
    assert result.empty
    assert set(FEATURE_COLUMNS + ["home_win", "feature_cutoff_at"]) <= set(result)
