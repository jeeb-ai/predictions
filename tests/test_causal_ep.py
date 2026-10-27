"""Chronological EP tests use small explicitly synthetic raw states."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from predictions.causal_ep import (
    EP_FEATURE_COLUMNS,
    _epa_for_states,
    _prepare_states,
    reconstruct_epa,
)


def _raw(season=2019, game="one", home_points=7, away_points=3):
    common = {
        "game_id": f"{season}_{game}",
        "home_team": "BUF",
        "away_team": "NE",
        "game_half": "Half1",
        "qtr": 1,
        "yardline_100": 70.0,
        "down": 1.0,
        "ydstogo": 10.0,
        "posteam_timeouts_remaining": 3.0,
        "defteam_timeouts_remaining": 3.0,
    }
    return pd.DataFrame(
        [
            {
                **common,
                "play_id": 1.0,
                "posteam": "BUF",
                "posteam_score": 0.0,
                "defteam_score": 0.0,
                "total_home_score": 0.0,
                "total_away_score": 0.0,
                "half_seconds_remaining": 1800.0,
            },
            {
                **common,
                "play_id": 2.0,
                "posteam": "BUF",
                "posteam_score": 0.0,
                "defteam_score": 0.0,
                "total_home_score": float(home_points),
                "total_away_score": 0.0,
                "yardline_100": 5.0,
                "half_seconds_remaining": 1750.0,
            },
            {
                **common,
                "play_id": 3.0,
                "posteam": "NE",
                "posteam_score": 0.0,
                "defteam_score": float(home_points),
                "total_home_score": float(home_points),
                "total_away_score": 0.0,
                "half_seconds_remaining": 1700.0,
            },
            {
                **common,
                "play_id": 4.0,
                "posteam": "NE",
                "posteam_score": 0.0,
                "defteam_score": float(home_points),
                "total_home_score": float(home_points),
                "total_away_score": float(away_points),
                "half_seconds_remaining": 1.0,
                "qtr": 2,
                "down": 4.0,
                "yardline_100": 20.0,
            },
            {
                **common,
                "play_id": 5.0,
                "posteam": None,
                "posteam_score": np.nan,
                "defteam_score": np.nan,
                "total_home_score": float(home_points),
                "total_away_score": float(away_points),
                "half_seconds_remaining": 0.0,
                "qtr": 2,
                "down": np.nan,
            },
        ]
    )


class ConstantEP:
    def predict(self, frame):
        return np.full(len(frame), 2.0)


def test_uses_preplay_scores_and_separate_terminal_labels():
    states = _prepare_states(_raw())
    assert states.loc[1, "score_differential"] == 0.0
    assert states.loc[1, "_home_pre"] == 0.0
    assert states.loc[1, "total_home_score"] == 7.0
    assert states.loc[0, "_target"] == 4.0
    assert states.loc[2, "_target"] == 3.0
    mutated = _raw(home_points=28)
    revised = _prepare_states(mutated)
    assert_frame_equal(states.loc[:1, EP_FEATURE_COLUMNS], revised.loc[:1, EP_FEATURE_COLUMNS])
    assert states.loc[0, "_target"] != revised.loc[0, "_target"]


def test_epa_accounts_for_score_and_possession_sign_and_half_terminal():
    values = _epa_for_states(_prepare_states(_raw()), ConstantEP()).set_index("play_id")
    assert values.loc[1, "epa"] == 0.0
    assert values.loc[2, "epa"] == 3.0  # 7 scored - 2 opposing EP - 2 initial EP.
    assert values.loc[3, "epa"] == 0.0
    assert values.loc[4, "epa"] == 1.0  # Terminal FG: 3 - preplay EP2.
    assert pd.isna(values.loc[5, "epa"])


def test_next_half_cannot_supply_next_state_or_terminal_score():
    first = _raw()
    second = _raw(game="one", home_points=35, away_points=21)
    second["game_half"] = "Half2"
    second["qtr"] = second["qtr"] + 2
    second["play_id"] = second["play_id"] + 100
    both = _epa_for_states(_prepare_states(pd.concat([first, second])), ConstantEP())
    alone = _epa_for_states(_prepare_states(first), ConstantEP())
    assert_frame_equal(both.loc[both.play_id.le(5)].reset_index(drop=True), alone)


def test_zero_ep_warmup_and_unknown_state_preserve_all_keys():
    raw = _raw()
    result = _epa_for_states(_prepare_states(raw), None).set_index("play_id")
    assert result.loc[2, "epa"] == 7
    assert result.loc[4, "epa"] == 3
    assert result.index.tolist() == [1, 2, 3, 4, 5]
    assert pd.isna(result.loc[5, "epa"])


def test_shuffling_raw_rows_does_not_change_transitions():
    original = _epa_for_states(_prepare_states(_raw()), ConstantEP())
    shuffled = _epa_for_states(_prepare_states(_raw().sample(frac=1, random_state=4)), ConstantEP())
    assert_frame_equal(original, shuffled)


@pytest.fixture
def seasons(tmp_path):
    result = {}
    for season in [2019, 2020, 2021]:
        path = tmp_path / f"{season}.parquet"
        pd.concat([_raw(season, "one"), _raw(season, "two", 3, 7)]).to_parquet(path, index=False)
        result[season] = path
    return result


def test_annual_estimators_only_use_prior_seasons_and_cache_repeatably(seasons, tmp_path):
    paths, metadata = reconstruct_epa(seasons, tmp_path / "ep")
    assert set(paths) == {2019, 2020, 2021}
    records = {record["season"]: record for record in metadata["models"]}
    assert records[2019]["model_kind"] == "zero-EP-warmup"
    assert records[2020]["training_seasons"] == [2019]
    assert records[2021]["training_seasons"] == [2019, 2020]
    assert records[2020]["training_rows"] == 8
    assert not metadata["raw_upstream_epa_used"]
    repeated_paths, repeated = reconstruct_epa(seasons, tmp_path / "ep")
    assert repeated_paths == paths
    assert repeated == metadata


def test_future_and_current_outcomes_do_not_change_frozen_ep_estimator(seasons, tmp_path):
    paths, metadata = reconstruct_epa(seasons, tmp_path / "first")
    original = {row["season"]: row for row in metadata["models"]}
    # Current2020 outcomes change realized2020 EPA, but cannot change its EP fit.
    pd.concat([_raw(2020, "one", 35, 0), _raw(2020, "two", 35, 0)]).to_parquet(
        seasons[2020], index=False
    )
    pd.concat([_raw(2021, "one", 49, 0), _raw(2021, "two", 49, 0)]).to_parquet(
        seasons[2021], index=False
    )
    new_paths, new_metadata = reconstruct_epa(seasons, tmp_path / "second")
    revised = {row["season"]: row for row in new_metadata["models"]}
    assert_frame_equal(pd.read_parquet(paths[2019]), pd.read_parquet(new_paths[2019]))
    assert original[2020]["model_sha256"] == revised[2020]["model_sha256"]
    model = joblib.load(paths[2020].parent / "model.joblib")
    new_model = joblib.load(new_paths[2020].parent / "model.joblib")
    states = _prepare_states(_raw(2020)).loc[lambda d: d._valid, EP_FEATURE_COLUMNS]
    np.testing.assert_allclose(model.predict(states), new_model.predict(states))


def test_schedule_availability_filters_ep_training(seasons, tmp_path):
    rows = []
    for season in seasons:
        for game in ["one", "two"]:
            kickoff = pd.Timestamp(f"{season}-09-01T17:00:00Z")
            availability = kickoff + pd.Timedelta(days=1)
            if season == 2019 and game == "two":
                availability = pd.Timestamp("2021-01-01T00:00:00Z")
            rows.append(
                {
                    "game_id": f"{season}_{game}",
                    "season": season,
                    "kickoff": kickoff,
                    "result_available_at": availability,
                    "home_score": 7.0,
                    "away_score": 3.0,
                }
            )
    _, metadata = reconstruct_epa(seasons, tmp_path / "ep", games=pd.DataFrame(rows))
    record = metadata["models"][1]
    assert record["training_games"] == 1
    assert record["training_rows"] == 4
    assert pd.Timestamp(record["train_max_result_available_at"]) < pd.Timestamp(
        record["training_cutoff_at"]
    )


def test_published_epa_column_is_ignored(seasons, tmp_path):
    paths, _ = reconstruct_epa(seasons, tmp_path / "original")
    for path in seasons.values():
        raw = pd.read_parquet(path)
        raw["epa"] = 1_000_000.0
        raw.to_parquet(path, index=False)
    revised, _ = reconstruct_epa(seasons, tmp_path / "revised")
    for year in paths:
        assert_frame_equal(pd.read_parquet(paths[year]), pd.read_parquet(revised[year]))


def test_rejects_corrupt_generated_cache(seasons, tmp_path):
    paths, _ = reconstruct_epa(seasons, tmp_path / "ep")
    paths[2020].write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="corrupt"):
        reconstruct_epa(seasons, tmp_path / "ep")


def test_rejects_gaps_duplicates_and_missing_states(seasons, tmp_path):
    with pytest.raises(ValueError, match="consecutive"):
        reconstruct_epa({2019: seasons[2019], 2021: seasons[2021]}, tmp_path / "ep")
    with pytest.raises(ValueError, match="Duplicate"):
        _prepare_states(pd.concat([_raw(), _raw()]))
    with pytest.raises(ValueError, match="Missing"):
        _prepare_states(_raw().drop(columns="posteam_score"))
    with pytest.raises(ValueError, match="At least one"):
        reconstruct_epa({}, tmp_path / "ep")


def test_model_metadata_is_valid_json(seasons, tmp_path):
    paths, _ = reconstruct_epa(seasons, tmp_path / "ep")
    for path in paths.values():
        metadata = json.loads((path.parent / "metadata.json").read_text())
        assert metadata["epa_sha256"]
        assert metadata["feature_columns"] == EP_FEATURE_COLUMNS
