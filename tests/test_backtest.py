"""Synthetic checks of evaluation boundaries, metrics, and forecast persistence."""

import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from predictions.backtest import evaluate_probabilities, run_backtest
from predictions.features import FEATURE_COLUMNS


@pytest.fixture
def feature_frame():
    records = []
    for year in range(2018, 2022):
        for week in range(1, 5):
            kickoff = pd.Timestamp(f"{year}-09-01T17:00:00Z") + pd.Timedelta(weeks=week)
            home_win = week % 2
            records.append(
                {
                    "game_id": f"{year}_{week}_AWAY_HOME",
                    "season": year,
                    "week": week,
                    "game_type": "REG",
                    "kickoff": kickoff,
                    "result_available_at": kickoff + pd.Timedelta(hours=24),
                    "home_team": "HOME",
                    "away_team": "AWAY",
                    "home_score": 24 if home_win else 17,
                    "away_score": 17 if home_win else 24,
                    "home_win": home_win,
                    "feature_cutoff_at": kickoff,
                    "history_available_through": kickoff - pd.Timedelta(days=6),
                    "feature_version": "synthetic-test-only",
                    **{
                        column: float((week + index) % 3 - 1)
                        for index, column in enumerate(FEATURE_COLUMNS)
                    },
                }
            )
    return pd.DataFrame(records)


def run_fixture(frame, output_dir, **kwargs):
    return run_backtest(
        frame,
        output_dir,
        holdout_season=2021,
        min_train_seasons=2,
        bootstrap_samples=100,
        source_metadata={"sha256": "synthetic-fixture"},
        **kwargs,
    )


def test_metrics_known_values_and_endpoint_bins():
    perfect = evaluate_probabilities([0, 1], [0, 1])
    assert perfect["accuracy"] == 1
    assert perfect["brier_score"] == 0
    assert perfect["log_loss"] == pytest.approx(0, abs=1e-14)
    assert perfect["expected_calibration_error"] == 0
    assert perfect["reliability_bins"][0]["count"] == 1
    assert perfect["reliability_bins"][-1]["count"] == 1
    wrong = evaluate_probabilities([0, 1], [1, 0])
    assert wrong["accuracy"] == 0
    assert wrong["brier_score"] == 1
    assert wrong["log_loss"] > 30
    constant = evaluate_probabilities([0, 1], [0.5, 0.5])
    assert constant["accuracy"] == 0.5
    assert constant["brier_score"] == 0.25
    assert constant["log_loss"] == pytest.approx(np.log(2))


def test_bootstrap_resamples_whole_clusters_and_is_reproducible():
    kwargs = {"clusters": ["week1", "week1", "week2", "week2"], "bootstrap_samples": 200, "seed": 9}
    first = evaluate_probabilities([1, 1, 0, 0], [0.8] * 4, **kwargs)
    assert first == evaluate_probabilities([1, 1, 0, 0], [0.8] * 4, **kwargs)
    assert first["bootstrap_clusters"] == 2
    assert first["confidence_intervals_95"]["accuracy"] == [0, 1]


@pytest.mark.parametrize(
    "labels, probabilities", [([], []), ([0], [1, 0]), ([2], [0.5]), ([1], [np.nan]), ([1], [1.01])]
)
def test_metrics_reject_invalid_inputs(labels, probabilities):
    with pytest.raises(ValueError):
        evaluate_probabilities(labels, probabilities)


def test_chronological_models_holdout_and_audit_outputs(feature_frame, tmp_path):
    report = run_fixture(feature_frame, tmp_path)
    assert report["validation"]["games"] == 4
    assert report["holdout"]["games"] == 4
    assert set(report["per_season"]) == {"2020", "2021"}
    for model in report["models"]:
        assert max(model["training_seasons"]) < model["season"]
        assert 2021 not in model["training_seasons"]
        assert pd.Timestamp(model["train_max_result_available_at"]) < pd.Timestamp(
            model["training_cutoff_at"]
        )
    run_dir = tmp_path / report["run_id"]
    saved = json.loads((run_dir / "report.json").read_text())
    assert saved["holdout"] == report["holdout"]
    predictions = pd.read_csv(run_dir / "predictions.csv")
    assert predictions["source_sha256"].eq("synthetic-fixture").all()
    assert predictions["train_max_season"].lt(predictions["season"]).all()
    assert set(FEATURE_COLUMNS) <= set(predictions.columns)
    assert (run_dir / "reliability.svg").read_text().startswith("<svg")
    assert (run_dir / "metrics.md").is_file()
    assert len(list(run_dir.glob("model-*.joblib"))) == 2
    with sqlite3.connect(tmp_path / "predictions.sqlite") as connection:
        assert connection.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0] == 8
        record = json.loads(
            connection.execute("SELECT prediction_json FROM forecasts LIMIT 1").fetchone()[0]
        )
    assert record["feature_cutoff_at"]
    assert record["model_id"]
    assert record["generated_at"]


def test_holdout_targets_do_not_fit_model_or_preceding_predictions(feature_frame, tmp_path):
    first = run_fixture(feature_frame, tmp_path / "first")
    altered = feature_frame.copy()
    altered.loc[altered["season"] == 2021, "home_win"] = (
        1 - altered.loc[altered["season"] == 2021, "home_win"]
    )
    second = run_fixture(altered, tmp_path / "second")
    first_predictions = pd.read_csv(first["run_dir"] + "/predictions.csv")
    second_predictions = pd.read_csv(second["run_dir"] + "/predictions.csv")
    np.testing.assert_allclose(
        first_predictions["home_win_probability"], second_predictions["home_win_probability"]
    )
    for first_model, second_model in zip(first["models"], second["models"], strict=True):
        assert first_model["scaled_coefficients"] == second_model["scaled_coefficients"]


def test_equal_availability_is_excluded_from_training(feature_frame, tmp_path):
    cutoff = feature_frame.loc[feature_frame["season"] == 2021, "kickoff"].min()
    feature_frame.loc[0, "result_available_at"] = cutoff
    report = run_fixture(feature_frame, tmp_path)
    model = next(item for item in report["models"] if item["season"] == 2021)
    assert model["training_games"] == 11
    assert pd.Timestamp(model["train_max_result_available_at"]) < cutoff


def test_runs_append_without_overwriting_artifacts(feature_frame, tmp_path):
    first = run_fixture(feature_frame, tmp_path)
    first_bytes = (tmp_path / first["run_id"] / "predictions.csv").read_bytes()
    second = run_fixture(feature_frame, tmp_path)
    assert first["run_id"] != second["run_id"]
    assert (tmp_path / first["run_id"] / "predictions.csv").read_bytes() == first_bytes
    with sqlite3.connect(tmp_path / "predictions.sqlite") as connection:
        assert connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0] == 16


def test_rejects_cutoff_violations(feature_frame, tmp_path):
    feature_frame.loc[0, "history_available_through"] = feature_frame.loc[0, "kickoff"]
    with pytest.raises(ValueError, match="strictly before"):
        run_fixture(feature_frame, tmp_path)


def test_rejects_impossible_result_availability(feature_frame, tmp_path):
    feature_frame.loc[0, "result_available_at"] = feature_frame.loc[0, "kickoff"]
    with pytest.raises(ValueError, match="after their own kickoff"):
        run_fixture(feature_frame, tmp_path)


def test_ties_and_postseason_are_excluded_and_season_cutoff_precedes_tie(feature_frame, tmp_path):
    holdout = feature_frame["season"] == 2021
    first = feature_frame.loc[holdout].index[0]
    season_start = feature_frame.loc[first, "kickoff"]
    feature_frame.loc[first, ["home_score", "away_score", "home_win"]] = [17, 17, np.nan]
    feature_frame.loc[feature_frame.index[-1], "game_type"] = "POST"
    report = run_fixture(feature_frame, tmp_path)
    assert report["holdout"]["games"] == 2
    assert report["excluded_tied_regular_season_games"] == 1
    model = next(item for item in report["models"] if item["season"] == 2021)
    assert pd.Timestamp(model["training_cutoff_at"]) == season_start


def test_rejects_insufficient_training_and_empty_holdout(feature_frame, tmp_path):
    with pytest.raises(ValueError, match="Not enough prior seasons"):
        run_backtest(feature_frame, tmp_path, holdout_season=2021, min_train_seasons=4)
    with pytest.raises(ValueError, match="holdout season has no"):
        run_backtest(feature_frame, tmp_path, holdout_season=2022)
