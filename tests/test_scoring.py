"""Independent distributions, line semantics, chronology and timestamp guards."""

import numpy as np
import pandas as pd
import pytest
from test_inference import games  # noqa: F401

from predictions.features import build_features
from predictions.scoring import (
    CANDIDATES,
    H,
    distribution,
    evaluate_rows,
    fit_scoring,
    metrics,
    outcome_probabilities,
    summarize,
)

# Test imports intentionally expose reusable pytest fixtures.
# ruff: noqa: F811


@pytest.fixture(autouse=True)
def frozen_scoring_clock(monkeypatch):
    monkeypatch.setattr("predictions.scoring._now", lambda: pd.Timestamp("2026-10-29T00:00:01Z"))


@pytest.fixture
def history():
    from predictions.scoring import COLUMNS

    rng = np.random.default_rng(44)

    def rows(year, n):
        frame = pd.DataFrame(rng.normal(size=(n, len(COLUMNS))), columns=COLUMNS)
        frame["kickoff"] = pd.date_range(f"{year}-01-01", periods=n, tz="UTC")
        frame["result_available_at"] = frame.kickoff + pd.Timedelta(hours=24)
        frame["home_score"] = rng.poisson(24, size=n)
        frame["away_score"] = rng.poisson(21, size=n)
        frame["game_id"] = [f"{year}-{i}" for i in range(n)]
        frame["season"] = year
        frame["week"] = np.arange(n) // 4
        return frame

    return rows(2023, 40), rows(2024, 20), rows(2025, 12)


@pytest.mark.parametrize("kind", CANDIDATES)
def test_models_generate_coherent_joint_distributions_and_metrics(history, kind):
    train, cal, test = history
    model = fit_scoring(train, cal, kind)
    from predictions.scoring import predict_means

    pmf = distribution(model, predict_means(model, test.iloc[:1])[0])
    assert np.isclose(pmf.sum(), 1) and (pmf >= 0).all()
    summary = summarize(pmf)
    assert summary["home_mean"] + summary["away_mean"] == pytest.approx(summary["expected_total"])
    assert summary["home_mean"] - summary["away_mean"] == pytest.approx(summary["expected_margin"])
    results = evaluate_rows(model, test)
    report = metrics(results)
    assert report["games"] == 12 and report["score_crps"] >= 0
    assert 0 <= report["coverage_90"] <= 1
    assert report["confidence_intervals_95"]


def test_exact_lines_include_pushes_and_half_points_do_not():
    pmf = np.zeros(H.shape)
    pmf[24, 21] = 0.4
    pmf[21, 24] = 0.3
    pmf[20, 20] = 0.3
    integer = outcome_probabilities(pmf, home_spread=-3, total_line=45)
    assert integer["home_cover"] == {"win": 0, "push": 0.4, "loss": 0.6}
    assert integer["over"]["push"] == pytest.approx(0.7)
    half = outcome_probabilities(pmf, home_spread=-2.5, total_line=44.5)
    assert half["home_cover"]["win"] == 0.4 and half["home_cover"]["push"] == 0
    assert half["over"]["win"] == pytest.approx(0.7)
    assert np.isclose(sum(half["over"].values()), 1)


@pytest.mark.parametrize("line", [float("nan"), float("inf"), -200, -3.25])
def test_invalid_lines_fail(line):
    with pytest.raises(ValueError):
        outcome_probabilities(np.ones(H.shape) / H.size, home_spread=line)


def test_train_results_must_precede_calibration(history):
    train, cal, _ = history
    train["result_available_at"] = cal.kickoff.min()
    with pytest.raises(ValueError, match="chronologically"):
        fit_scoring(train, cal, CANDIDATES[0])


def test_scoring_input_does_not_include_current_results_or_market_probabilities(games):
    from predictions.scoring import COLUMNS

    assert not set(COLUMNS) & {
        "home_score",
        "away_score",
        "home_win_probability",
        "away_win_probability",
    }
    features = build_features(games)
    mutated = games.copy()
    mutated.loc[mutated.game_id.eq("synthetic-2026-target"), ["home_score", "away_score"]] = [70, 0]
    changed = build_features(mutated)
    np.testing.assert_equal(
        features.loc[features.game_id.eq("synthetic-2026-target"), COLUMNS].to_numpy(),
        changed.loc[changed.game_id.eq("synthetic-2026-target"), COLUMNS].to_numpy(),
    )


def test_live_score_artifact_and_checksum_are_independent_of_winner(games, tmp_path):
    from predictions.scoring import forecast_scoring, read_score_distribution

    games = games.copy()
    # Expand the synthetic history to provide at least10training and8calibration rows.
    copies = []
    for i in range(3):
        copy = games.copy()
        copy["game_id"] = copy.game_id + "-copy" + str(i)
        copy["kickoff"] += pd.Timedelta(hours=i + 1)
        copy["result_available_at"] += pd.Timedelta(hours=i + 1)
        copies.append(copy.loc[copy.season.lt(2026)])
    frame = pd.concat([games, *copies], ignore_index=True)
    report = {
        "type": "nfl_scoring_validation",
        "status": "evaluated_not_a_profit_claim",
        "holdout_season": 2025,
        "selected_kind": "ridge_joint_gaussian",
    }
    from predictions.scoring import forecast_scoring, read_score_distribution

    report.update(
        feature_columns=__import__("predictions.scoring", fromlist=["COLUMNS"]).COLUMNS,
        grid_max_score=150,
        splits={
            "validation": {
                kind: {"joint_log_loss": i}
                for kind, i in zip(CANDIDATES, [9.0, 7.0, 8.0], strict=True)
            },
            "holdout": {"ridge_joint_gaussian": {"games": 1}},
        },
        folds=[{"train_seasons": [2023], "calibration_season": 2024, "test_season": 2025}],
    )
    score = forecast_scoring(
        frame,
        report,
        ["synthetic-2026-target"],
        tmp_path,
        as_of=pd.Timestamp("2026-10-29T00:00:00Z"),
        source_metadata={},
    )["synthetic-2026-target"]
    pmf = read_score_distribution(score)
    assert np.isclose(pmf.sum(), 1)
    assert max(score["training_seasons"]) < score["calibration_season"] == 2025
    assert score["summary"]["expected_total"] == pytest.approx(
        score["summary"]["home_mean"] + score["summary"]["away_mean"]
    )
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_string(
        "from predictions.prediction_ui import render_score_lines\nrender_score_lines("
        + repr(score)
        + ")"
    ).run()
    assert not app.exception
    assert any("Home covers:" in m.value for m in app.markdown)
    assert any("Market value: PASS" in m.value for m in app.info)
    app.number_input[0].set_value(-3.0).run()
    assert not app.exception
    assert any("poorly calibrated" in m.value for m in app.warning)
    from pathlib import Path

    Path(score["distribution"]["path"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        read_score_distribution(score)


def test_postgame_score_forecasts_fail(games, tmp_path):
    from predictions.scoring import forecast_scoring

    report = {
        "type": "nfl_scoring_validation",
        "status": "evaluated_not_a_profit_claim",
        "holdout_season": 2025,
        "selected_kind": "ridge_joint_gaussian",
    }
    report.update(
        feature_columns=__import__("predictions.scoring", fromlist=["COLUMNS"]).COLUMNS,
        grid_max_score=150,
        splits={
            "validation": {
                kind: {"joint_log_loss": i}
                for kind, i in zip(CANDIDATES, [9.0, 7.0, 8.0], strict=True)
            },
            "holdout": {"ridge_joint_gaussian": {"games": 1}},
        },
        folds=[{"train_seasons": [2023], "calibration_season": 2024, "test_season": 2025}],
    )
    with pytest.raises(ValueError, match="upcoming unfinished"):
        forecast_scoring(
            games,
            report,
            ["synthetic-2026-opener"],
            tmp_path,
            as_of=pd.Timestamp("2026-10-29T00:00:00Z"),
            source_metadata={},
        )


def test_status_only_report_cannot_enable_score_projection():
    from predictions.scoring import validate_report

    with pytest.raises(ValueError):
        validate_report(
            {"type": "nfl_scoring_validation", "status": "evaluated_not_a_profit_claim"}
        )


def test_chronological_selection_ignores_holdout_outcomes(history, tmp_path):
    from predictions.scoring import backtest_scoring, validate_report

    train, cal, test = history
    earlier = train.copy()
    earlier["season"] = 2022
    earlier["game_id"] = "earlier-" + earlier.game_id
    earlier["kickoff"] -= pd.DateOffset(years=1)
    earlier["result_available_at"] -= pd.DateOffset(years=1)
    frame = pd.concat([earlier, train, cal, test], ignore_index=True)
    frame["game_type"] = "REG"
    first, _ = backtest_scoring(
        frame, tmp_path, source_metadata={"mode": "synthetic"}, holdout=2025, min_train_seasons=1
    )
    validate_report(first)
    assert all(
        max(f["train_seasons"]) < f["calibration_season"] < f["test_season"] for f in first["folds"]
    )
    expected = min(CANDIDATES, key=lambda key: first["splits"]["validation"][key]["joint_log_loss"])
    assert first["selected_kind"] == expected
    frame.loc[frame.season.eq(2025), ["home_score", "away_score"]] = [90, 0]
    second, _ = backtest_scoring(
        frame, tmp_path, source_metadata={"mode": "synthetic"}, holdout=2025, min_train_seasons=1
    )
    assert second["selected_kind"] == first["selected_kind"]
    assert second["splits"]["validation"] == first["splits"]["validation"]
