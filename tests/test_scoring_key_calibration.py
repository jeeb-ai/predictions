"""Research-only correction: chronology, coherent probabilities and base reproducibility."""

import json

import numpy as np
import pandas as pd
import pytest

from predictions.scoring import COLUMNS, A, H, evaluate_rows, fit_scoring
from predictions.scoring_key_calibration import (
    GROUPS,
    category,
    correct,
    evaluate,
    fit_correction,
    investigate,
    main,
)


def test_categories_and_correction_preserve_coherence_and_side_symmetry():
    pmf = np.exp(-((H - 24) ** 2 + (A - 21) ** 2) / 200)
    pmf /= pmf.sum()
    np.testing.assert_allclose(correct(pmf, np.zeros(10)), pmf)
    theta = np.linspace(-0.5, 0.5, 10)
    adjusted = correct(pmf, theta)
    assert adjusted.sum() == pytest.approx(1) and (adjusted >= 0).all()
    np.testing.assert_allclose(correct(pmf.T, theta), adjusted.T)
    assert category(np.array([24, 21, 24]), np.array([21, 24, 20])).tolist() == [11, 11, 32]
    # Reweighting changes category mass, not relative odds within a category.
    group = GROUPS == 35
    ratios = adjusted[group] / pmf[group]
    assert np.ptp(ratios) < 1e-12


def test_correction_learns_only_from_supplied_older_calibration_labels():
    masses = np.full((40, 36), 1 / 36)
    labels = np.array([11] * 25 + [35] * 15)
    theta = fit_correction(
        masses, labels, calibration_latest="2024-12-31T00:00:00Z", test_first="2025-09-01T00:00:00Z"
    )
    assert theta.shape == (10,) and theta[1] > 0 and np.abs(theta).max() <= 2
    for last in ["2025-09-01T00:00:00Z", "2026-01-01T00:00:00Z", "NaT", "2024-01-01"]:
        with pytest.raises(ValueError, match="precede"):
            fit_correction(
                masses, labels, calibration_latest=last, test_first="2025-09-01T00:00:00Z"
            )
    with pytest.raises(ValueError, match="Invalid"):
        fit_correction(
            masses * 2,
            labels,
            calibration_latest="2024-12-31T00:00:00Z",
            test_first="2025-09-01T00:00:00Z",
        )


def test_invalid_correction_inputs_and_exact_push_labels():
    pmf = np.full(H.shape, 1 / H.size)
    for bad in [np.full(10, np.nan), np.ones(9), np.full(10, 3)]:
        with pytest.raises(ValueError, match="Invalid"):
            correct(pmf, bad)
    game = pd.Series({"home_score": 24, "away_score": 21})
    result = evaluate(pmf, game)
    assert result["spread_3_push_outcome"] == 1
    assert result["spread_7_push_outcome"] == 0
    assert result["total_44_push_outcome"] == 0


def frames():
    rng = np.random.default_rng(42)
    data = []
    for year, count in [(2021, 40), (2022, 20), (2023, 16), (2024, 16)]:
        frame = pd.DataFrame(rng.normal(size=(count, len(COLUMNS))), columns=COLUMNS)
        frame["season"] = year
        frame["kickoff"] = pd.date_range(f"{year}-01-01", periods=count, tz="UTC")
        frame["result_available_at"] = frame.kickoff + pd.Timedelta(days=1)
        frame["home_score"] = rng.poisson(24, count)
        frame["away_score"] = rng.poisson(21, count)
        frame["game_id"] = [f"{year}-{i}" for i in range(count)]
        frame["game_type"] = "REG"
        frame["week"] = np.arange(count) // 4
        data.append(frame)
    return pd.concat(data, ignore_index=True)


def baseline(frame):
    report = {
        "holdout_season": 2024,
        "selected_kind": "ridge_joint_gaussian",
        "folds": [],
        "splits": {},
        "source_metadata": {"synthetic": True},
    }
    for year, split in [(2023, "validation"), (2024, "holdout")]:
        train = frame.loc[frame.season.lt(year - 1)]
        cal = frame.loc[frame.season.eq(year - 1)]
        test = frame.loc[frame.season.eq(year)]
        model = fit_scoring(train, cal, "ridge_joint_gaussian")
        graded = evaluate_rows(model, test)
        report["folds"].append(
            {
                "test_season": year,
                "calibration_season": year - 1,
                "train_games": len(train),
                "calibration_games": len(cal),
            }
        )
        report["splits"][split] = {
            "ridge_joint_gaussian": {
                k: float(graded[k].mean()) for k in ["joint_log_loss", "score_mae"]
            }
        }
    return report


def test_complete_research_reproduces_base_and_holdout_cannot_change_fitted_weights(tmp_path):
    frame = frames()
    report = baseline(frame)
    first = investigate(frame, report, tmp_path / "first")
    assert first["promoted"] is False
    assert first["splits"]["holdout"]["base"]["games"] == 16
    changed = frame.copy()
    changed.loc[changed.season.eq(2024), ["home_score", "away_score"]] = [55, 0]
    second = investigate(changed, baseline(changed), tmp_path / "second")
    for a, b in zip(first["folds"], second["folds"], strict=True):
        np.testing.assert_allclose(a["weights"], b["weights"])
    assert (
        first["splits"]["holdout"]["candidate"]["joint_log_loss"]
        != second["splits"]["holdout"]["candidate"]["joint_log_loss"]
    )
    assert first["source_code_sha256"]


def test_base_fold_drift_is_rejected(tmp_path):
    frame = frames()
    report = baseline(frame)
    report["folds"][0]["train_games"] -= 1
    with pytest.raises(ValueError, match="folds do not reproduce"):
        investigate(frame, report, tmp_path)


def test_cli_rejects_different_historical_snapshot(tmp_path, monkeypatch):
    from predictions import scoring

    path = tmp_path / "report.json"
    path.write_text(json.dumps({"source_metadata": {"sha256": "wrong"}}))
    source = tmp_path / "source.csv"
    source.write_text("unverified")
    monkeypatch.setattr(scoring, "validate_report", lambda x: x)
    with pytest.raises(ValueError, match="SHA256"):
        main(["--input", str(source), "--validation", str(path)])
