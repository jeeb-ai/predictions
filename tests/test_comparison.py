"""Paired uncertainty and identical-fold checks use synthetic known pregame signals."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from predictions.backtest import run_backtest
from predictions.comparison import compare_probabilities, run_comparison
from predictions.features import FEATURE_COLUMNS


@pytest.fixture
def paired_features():
    rows = []
    for year in range(2019, 2023):
        for week in range(1, 7):
            kickoff = pd.Timestamp(f"{year}-09-01T17:00:00Z") + pd.Timedelta(weeks=week)
            known_signal = 1.0 if week % 2 else -1.0
            rows.append(
                {
                    "game_id": f"synthetic_{year}_{week}",
                    "season": year,
                    "week": week,
                    "game_type": "REG",
                    "kickoff": kickoff,
                    "result_available_at": kickoff + pd.Timedelta(hours=24),
                    "feature_cutoff_at": kickoff,
                    "history_available_through": kickoff - pd.Timedelta(days=6),
                    "home_win": int(known_signal > 0),
                    "home_score": 24 if known_signal > 0 else 10,
                    "away_score": 17,
                    "home_team": "BUF",
                    "away_team": "NE",
                    "feature_version": "synthetic-known-pregame-fixture",
                    **dict.fromkeys(FEATURE_COLUMNS, 0.0),
                    "known_signal": known_signal,
                }
            )
    second = pd.DataFrame(rows)
    first = second.drop(columns="known_signal")
    return first, second


def _run(frames, path, **kwargs):
    return run_comparison(
        *frames,
        path,
        holdout_season=2022,
        min_train_seasons=2,
        bootstrap_samples=50,
        model2_feature_columns=[*FEATURE_COLUMNS, "known_signal"],
        source_metadata={"sha256": "synthetic"},
        **kwargs,
    )


def test_paired_intervals_find_known_improvement_and_recompute_calibration():
    result = compare_probabilities(
        [0, 1] * 3,
        [1, 0] * 3,
        [0, 1] * 3,
        clusters=["a", "a", "b", "b", "c", "c"],
        bootstrap_samples=100,
    )
    delta = result["delta_model_2_minus_model_1"]
    assert delta["accuracy"]["estimate"] == 1
    assert delta["accuracy"]["confidence_interval_95"] == [1, 1]
    assert delta["brier_score"]["estimate"] == -1
    assert delta["expected_calibration_error"]["confidence_interval_95"] == [-1, -1]
    assert all(value["evidence"] == "supports_improvement" for value in delta.values())
    assert result["model_2"]["confidence_intervals_95"]["expected_calibration_error"] == [0, 0]


def test_paired_intervals_do_not_claim_improvement_when_difference_crosses_zero():
    result = compare_probabilities(
        [1, 1], [1, 0], [0, 1], clusters=["a", "b"], bootstrap_samples=1000
    )
    assert result["delta_model_2_minus_model_1"]["accuracy"]["confidence_interval_95"] == [-1, 1]
    assert result["delta_model_2_minus_model_1"]["accuracy"]["evidence"] == "inconclusive"


def test_paired_bootstrap_is_reproducible_and_identical_predictions_have_zero_delta():
    args = ([0, 1, 1, 0], [0.2, 0.7, 0.9, 0.4], [0.2, 0.7, 0.9, 0.4])
    first = compare_probabilities(*args, clusters=[1, 1, 2, 2], bootstrap_samples=50, seed=9)
    assert first == compare_probabilities(
        *args, clusters=[1, 1, 2, 2], bootstrap_samples=50, seed=9
    )
    for value in first["delta_model_2_minus_model_1"].values():
        assert value["estimate"] == 0
        assert value["confidence_interval_95"] == [0, 0]
        assert value["evidence"] == "inconclusive"


def test_no_bootstrap_means_no_uncertainty_evidence():
    result = compare_probabilities(
        [0, 1], [0.3, 0.6], [0.1, 0.9], clusters=[1, 2], bootstrap_samples=0
    )
    for value in result["delta_model_2_minus_model_1"].values():
        assert value["confidence_interval_95"] is None
        assert value["evidence"] == "not_estimated"


@pytest.mark.parametrize("clusters", [[1], [None, 1]])
def test_paired_statistics_require_valid_matching_clusters(clusters):
    with pytest.raises(ValueError, match="cluster labels"):
        compare_probabilities([0, 1], [0.2, 0.7], [0.3, 0.8], clusters=clusters)


def test_two_models_share_exact_folds_and_persist_actual_different_predictions(
    paired_features, tmp_path
):
    report = _run(paired_features, tmp_path)
    assert report["splits"]["validation"]["games"] == 6
    assert report["splits"]["holdout"]["games"] == 6
    root = Path(report["run_dir"])
    assert json.loads((root / "comparison.json").read_text())["splits"] == report["splits"]
    assert (root / "comparison.md").is_file()
    assert (root / "reliability.svg").read_text().startswith("<svg")
    predictions = pd.read_csv(root / "paired_predictions.csv")
    assert len(predictions) == 12
    assert not np.allclose(predictions["model_1_probability"], predictions["model_2_probability"])
    assert report["splits"]["holdout"]["model_2"]["accuracy"] == 1
    for first, second in zip(
        report["models"]["model_1"]["models"], report["models"]["model_2"]["models"], strict=True
    ):
        assert first["training_game_ids_sha256"] == second["training_game_ids_sha256"]
        assert first["training_cutoff_at"] == second["training_cutoff_at"]
        assert 2022 not in first["training_seasons"]
        assert 2022 not in second["training_seasons"]


@pytest.mark.parametrize("column", ["home_win", "season", FEATURE_COLUMNS[0]])
def test_rejects_changed_shared_targets_or_baseline_features(paired_features, tmp_path, column):
    first, second = paired_features
    second = second.copy()
    second.loc[0, column] += 1
    with pytest.raises(ValueError, match="shared inputs or target"):
        _run((first, second), tmp_path)
    assert not list(tmp_path.iterdir())


def test_rejects_intersection_evaluation(paired_features, tmp_path):
    first, second = paired_features
    with pytest.raises(ValueError, match="identical scheduled game IDs"):
        _run((first, second.iloc[1:]), tmp_path)


def test_holdout_labels_do_not_change_either_models_fits(paired_features, tmp_path):
    original = _run(paired_features, tmp_path / "original")
    changed = [frame.copy() for frame in paired_features]
    for frame in changed:
        mask = frame["season"] == 2022
        frame.loc[mask, "home_win"] = 1 - frame.loc[mask, "home_win"]
    altered = _run(changed, tmp_path / "altered")
    a = pd.read_csv(Path(original["run_dir"]) / "paired_predictions.csv")
    b = pd.read_csv(Path(altered["run_dir"]) / "paired_predictions.csv")
    np.testing.assert_allclose(
        a[["model_1_probability", "model_2_probability"]],
        b[["model_1_probability", "model_2_probability"]],
    )
    assert original["splits"]["holdout"]["model_2"]["accuracy"] == 1
    assert altered["splits"]["holdout"]["model_2"]["accuracy"] == 0


def test_advanced_history_cutoff_is_validated(paired_features, tmp_path):
    first, second = paired_features
    second = second.copy()
    second["advanced_history_available_through"] = second["kickoff"]
    with pytest.raises(ValueError, match="strictly before kickoff"):
        _run((first, second), tmp_path)


def test_explicit_v1_configuration_preserves_default_model_predictions(paired_features, tmp_path):
    first, _ = paired_features
    options = dict(holdout_season=2022, min_train_seasons=2, bootstrap_samples=0)
    default = run_backtest(first, tmp_path / "default", **options)
    explicit = run_backtest(
        first,
        tmp_path / "explicit",
        **options,
        feature_columns=FEATURE_COLUMNS,
        model_version="logistic-regression-v1",
    )
    left = pd.read_csv(Path(default["run_dir"]) / "predictions.csv")
    right = pd.read_csv(Path(explicit["run_dir"]) / "predictions.csv")
    np.testing.assert_array_equal(left["home_win_probability"], right["home_win_probability"])
