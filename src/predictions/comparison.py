"""Paired, chronology-preserving comparisons of fixed Model 1.0 and Model 2.0."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from predictions.backtest import _losses, evaluate_probabilities, run_backtest
from predictions.features import FEATURE_COLUMNS

MODEL2_VERSION = "logistic-regression-v2"
METRICS = ("accuracy", "brier_score", "log_loss", "expected_calibration_error")


def _cluster_statistics(y: np.ndarray, probability: np.ndarray, inverse, count):
    sums = {
        name: np.bincount(inverse, weights=value, minlength=count)
        for name, value in _losses(y, probability).items()
    }
    bins = np.minimum((probability * 10).astype(int), 9)
    residuals = np.zeros((count, 10))
    np.add.at(residuals, (inverse, bins), probability - y)
    return sums, residuals


def compare_probabilities(
    y_true,
    model1_probability,
    model2_probability,
    *,
    clusters,
    bootstrap_samples: int = 2000,
    seed: int = 42,
) -> dict:
    """Use identical resampled blocks for both models and their metric differences.

    ECE is recomputed in each resample from the ten fixed probability bins;
    it is not approximated as an independent per-game loss. Intervals are
    conditional on the fitted forecasts and unadjusted for multiple metrics.
    """
    first = evaluate_probabilities(y_true, model1_probability, bootstrap_samples=0)
    second = evaluate_probabilities(y_true, model2_probability, bootstrap_samples=0)
    if not isinstance(bootstrap_samples, (int, np.integer)) or bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be a nonnegative integer")
    y = np.asarray(y_true, dtype=float)
    probabilities = [
        np.asarray(value, dtype=float) for value in (model1_probability, model2_probability)
    ]
    labels = np.asarray(clusters)
    if labels.shape != y.shape or pd.isna(labels).any():
        raise ValueError("Nonmissing cluster labels must match the outcome array")
    _, inverse = np.unique(labels.astype(str), return_inverse=True)
    count = int(inverse.max()) + 1
    sizes = np.bincount(inverse, minlength=count)
    sampled = []
    if bootstrap_samples:
        weights = np.random.default_rng(seed).multinomial(
            count, np.full(count, 1 / count), size=bootstrap_samples
        )
        sample_sizes = weights @ sizes
        for probability, metrics in zip(probabilities, [first, second], strict=True):
            sums, residuals = _cluster_statistics(y, probability, inverse, count)
            draws = {name: (weights @ values) / sample_sizes for name, values in sums.items()}
            draws["expected_calibration_error"] = (
                np.abs(weights @ residuals).sum(axis=1) / sample_sizes
            )
            metrics["confidence_intervals_95"] = {
                name: np.quantile(values, [0.025, 0.975]).tolist() for name, values in draws.items()
            }
            metrics["bootstrap_clusters"] = count
            sampled.append(draws)
    differences = {}
    for name in METRICS:
        interval = (
            np.quantile(sampled[1][name] - sampled[0][name], [0.025, 0.975]).tolist()
            if bootstrap_samples
            else None
        )
        evidence = "not_estimated"
        if interval is not None:
            low, high = interval
            if low <= 0 <= high:
                evidence = "inconclusive"
            elif (name == "accuracy" and low > 0) or (name != "accuracy" and high < 0):
                evidence = "supports_improvement"
            else:
                evidence = "supports_deterioration"
        differences[name] = {
            "estimate": float(second[name] - first[name]),
            "confidence_interval_95": interval,
            "better_direction": "higher" if name == "accuracy" else "lower",
            "evidence": evidence,
        }
    return {
        "games": int(len(y)),
        "bootstrap_clusters": count,
        "model_1": first,
        "model_2": second,
        "delta_model_2_minus_model_1": differences,
    }


def _validate_alignment(first: pd.DataFrame, second: pd.DataFrame) -> None:
    required = {
        "game_id",
        "season",
        "game_type",
        "kickoff",
        "result_available_at",
        "home_win",
        "feature_cutoff_at",
        "history_available_through",
        *FEATURE_COLUMNS,
    }
    for frame in (first, second):
        if required - set(frame):
            raise ValueError(f"Missing comparison columns: {sorted(required - set(frame))}")
        if frame["game_id"].isna().any() or frame["game_id"].duplicated().any():
            raise ValueError("Comparison game IDs must be nonmissing and unique")
    left = first.set_index("game_id").sort_index()
    right = second.set_index("game_id").sort_index()
    if not left.index.equals(right.index):
        raise ValueError(
            "Both models must use identical scheduled game IDs; no intersection filtering"
        )
    columns = sorted(required - {"game_id"})
    columns += [
        column
        for column in ["week", "home_team", "away_team", "home_score", "away_score"]
        if column in first or column in second
    ]
    for column in columns:
        if column not in left or column not in right:
            raise ValueError(f"Both models must provide identical {column}")
        a, b = left[column], right[column]
        if column in {
            "kickoff",
            "result_available_at",
            "feature_cutoff_at",
            "history_available_through",
        }:
            a, b = pd.to_datetime(a, utc=True), pd.to_datetime(b, utc=True)
        try:
            pd.testing.assert_series_equal(
                a, b, check_dtype=False, check_names=False, check_exact=True
            )
        except AssertionError as exc:
            raise ValueError(f"Models differ on shared inputs or target column {column}") from exc


def _pair_forecasts(first: dict, second: dict) -> pd.DataFrame:
    checks = (
        "season",
        "training_cutoff_at",
        "train_max_result_available_at",
        "training_seasons",
        "training_games",
        "training_game_ids_sha256",
    )
    if len(first["models"]) != len(second["models"]):
        raise ValueError("Models have different chronological training folds")
    for a, b in zip(first["models"], second["models"], strict=True):
        if any(a[name] != b[name] for name in checks):
            raise ValueError("Models have different chronological training folds")
    frames = [
        pd.read_csv(Path(report["run_dir"]) / "predictions.csv", dtype={"game_id": str})
        .set_index("game_id")
        .sort_index()
        for report in (first, second)
    ]
    left, right = frames
    if not left.index.equals(right.index):
        raise ValueError("Model evaluation targets differ")
    for column in ("season", "week", "home_win", "kickoff", "result_available_at", "is_holdout"):
        if not left[column].equals(right[column]):
            raise ValueError(f"Model evaluation targets differ on {column}")
    keep = [
        "season",
        "week",
        "home_win",
        "kickoff",
        "result_available_at",
        "is_holdout",
        "training_cutoff_at",
        "train_max_result_available_at",
    ]
    paired = left[keep].copy()
    for name, frame in zip(("model_1", "model_2"), frames, strict=True):
        paired[f"{name}_probability"] = frame["home_win_probability"]
        paired[f"{name}_id"] = frame["model_id"]
    return paired.reset_index().sort_values(["kickoff", "game_id"]).reset_index(drop=True)


def _markdown(report: dict) -> str:
    lines = [
        "# NFL Model 1.0 versus Model 2.0",
        "",
        f"Run: `{report['run_id']}`",
        "",
        "| Split | Model | Games | Accuracy | Brier | Log loss | ECE |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split, result in report["splits"].items():
        if result is None:
            continue
        for model in ("model_1", "model_2"):
            metric = result[model]
            lines.append(
                f"| {split} | {model} | {result['games']} | {metric['accuracy']:.2%} | "
                f"{metric['brier_score']:.5f} | {metric['log_loss']:.5f} | "
                f"{metric['expected_calibration_error']:.5f} |"
            )
    lines += [
        "",
        "Differences below are Model 2.0 minus Model 1.0. Accuracy is better when "
        "positive; losses and ECE are better when negative.",
        "",
        "| Split | Metric | Difference | Paired 95% interval | Evidence |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for split, result in report["splits"].items():
        if result is None:
            continue
        for metric, delta in result["delta_model_2_minus_model_1"].items():
            ci = delta["confidence_interval_95"]
            interval = f"[{ci[0]:.5f}, {ci[1]:.5f}]" if ci is not None else "not computed"
            lines.append(
                f"| {split} | {metric} | {delta['estimate']:.5f} | {interval} | "
                f"{delta['evidence']} |"
            )
    lines += [
        "",
        "## Method and limitations",
        "",
        *[f"- {note}" for note in report["methodology"]],
        "",
    ]
    return "\n".join(lines)


def _reliability_svg(report: dict) -> str:
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="940" height="440" viewBox="0 0 940 440">',
        '<rect width="940" height="440" fill="white"/>',
    ]
    for panel, (split, result) in enumerate(report["splits"].items()):
        x0, y0, side = 65 + panel * 460, 345, 280
        parts += [
            f'<text x="{x0 + 140}" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">{split}</text>',
            f'<path d="M{x0} {y0}h{side}M{x0} {y0}v{-side}" fill="none" stroke="black"/>',
            f'<path d="M{x0} {y0}l{side} {-side}" stroke="#999" stroke-dasharray="5 5"/>',
        ]
        for tick in np.linspace(0, 1, 6):
            parts += [
                f'<text x="{x0 + side * tick}" y="365" text-anchor="middle" font-family="sans-serif" font-size="12">{tick:.1f}</text>',
                f'<text x="{x0 - 10}" y="{y0 - side * tick + 4}" text-anchor="end" font-family="sans-serif" font-size="12">{tick:.1f}</text>',
            ]
        if result is None:
            continue
        for model, color in [("model_1", "#2463a5"), ("model_2", "#bd4517")]:
            points = [
                (
                    x0 + side * b["mean_predicted"],
                    y0 - side * b["observed_home_win_rate"],
                    b["count"],
                )
                for b in result[model]["reliability_bins"]
                if b["count"]
            ]
            coords = " ".join(f"{x:.2f},{y:.2f}" for x, y, _ in points)
            parts.append(
                f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>'
            )
            for x, y, count in points:
                parts.append(
                    f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"><title>{model}: {count} games</title></circle>'
                )
        parts.append(
            f'<text x="{x0 + 140}" y="395" text-anchor="middle" font-family="sans-serif" font-size="12">Predicted home-win probability</text>'
        )
    parts += [
        '<text x="15" y="210" transform="rotate(-90 15 210)" text-anchor="middle" font-family="sans-serif" font-size="12">Observed home-win fraction</text>',
        '<text x="340" y="425" fill="#2463a5" font-family="sans-serif" font-size="13">Model 1.0</text>',
        '<text x="490" y="425" fill="#bd4517" font-family="sans-serif" font-size="13">Model 2.0</text>',
        "</svg>",
    ]
    return "\n".join(parts)


def run_comparison(
    v1_features: pd.DataFrame,
    v2_features: pd.DataFrame,
    output_dir: Path,
    *,
    holdout_season: int = 2025,
    min_train_seasons: int = 5,
    bootstrap_samples: int = 2000,
    seed: int = 42,
    source_metadata: dict | None = None,
    model2_feature_columns: list[str] | None = None,
) -> dict:
    """Run both fixed estimators on exactly the same targets and training folds."""
    _validate_alignment(v1_features, v2_features)
    if model2_feature_columns is None:
        from predictions.features_v2 import MODEL2_FEATURE_COLUMNS

        model2_feature_columns = MODEL2_FEATURE_COLUMNS
    timestamp = datetime.now(timezone.utc)
    run_id = timestamp.strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:8]
    root = Path(output_dir) / run_id
    root.mkdir(parents=True, exist_ok=False)
    options = dict(
        holdout_season=holdout_season,
        min_train_seasons=min_train_seasons,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        source_metadata=source_metadata,
    )
    first = run_backtest(v1_features, root / "model_1", **options)
    second = run_backtest(
        v2_features,
        root / "model_2",
        **options,
        feature_columns=model2_feature_columns,
        model_version=MODEL2_VERSION,
        methodology_extra=(
            "Model 2.0 adds strictly pregame play-by-play team and quarterback form plus opponent-adjusted strength.",
        ),
    )
    paired = _pair_forecasts(first, second)

    def summarize(frame):
        if frame.empty:
            return None
        return compare_probabilities(
            frame["home_win"],
            frame["model_1_probability"],
            frame["model_2_probability"],
            clusters=frame["season"].astype(str) + ":" + frame["week"].astype(str),
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )

    holdout_notice = (
        "The 2025 outcomes were previously reported for Model 1.0. It is a reused holdout excluded from Model 2.0 fitting, tuning, selection, and calibration; it cannot be described as a newly unseen test."
        if holdout_season == 2025
        else f"Season {holdout_season} is excluded from fitting, tuning, and calibration; these software checks do not establish that its outcomes were never previously examined."
    )
    report = {
        "schema_version": 1,
        "run_id": run_id,
        "run_dir": str(root),
        "generated_at": timestamp.isoformat(),
        "source_metadata": source_metadata or {},
        "source_code_sha256": first["source_code_sha256"],
        "runtime": first["runtime"],
        "configuration": {
            "holdout_season": holdout_season,
            "min_train_seasons": min_train_seasons,
            "bootstrap_samples": bootstrap_samples,
            "seed": seed,
            "model_1_features": FEATURE_COLUMNS,
            "model_2_features": list(model2_feature_columns),
        },
        "models": {
            "model_1": {key: first[key] for key in ["run_id", "run_dir", "models"]},
            "model_2": {key: second[key] for key in ["run_id", "run_dir", "models"]},
        },
        "splits": {
            "validation": summarize(paired.loc[~paired["is_holdout"]]),
            "holdout": summarize(paired.loc[paired["is_holdout"]]),
        },
        "per_season": {
            str(year): summarize(paired.loc[paired["season"] == year])
            for year in sorted(paired["season"].unique())
        },
        "methodology": [
            "Both versions use identical scheduled games, decisive regular-season targets, expanding prior-season training IDs, season-opening cutoffs, StandardScaler, and fixed C=1 logistic regression. No missing-data intersection or split changes are allowed.",
            "The first five seasons are training warm-up by default. Annual coefficients and scaling remain frozen; earlier same-season outcomes may enter later pregame form only after their strict availability cutoff.",
            holdout_notice,
            "Probabilities are raw logistic outputs. Calibration is measured using ten fixed-width bins, not fitted on evaluation targets.",
            "Individual and paired 95% percentile intervals resample identical season/week blocks with fixed forecasts, recomputing ECE in every sample. They omit model-refit uncertainty, cross-week dependence, and source revisions; per-metric intervals are not adjusted for multiple comparisons.",
            "Paired deltas are Model 2.0 minus Model 1.0. Improvement evidence requires the corresponding interval to exclude zero in the favorable direction; no overall improvement or trading-value claim follows automatically.",
            "Model 2.0 recomputes EPA and success locally using EP models trained only on prior seasons; published retrospective nflfastR EPA is excluded from its predictors. EP training provenance must be retained with the source metadata.",
            "Historical raw source snapshots can be revised retrospectively. Pregame lagging and prior-season EP fitting enforce temporal cutoffs in the supplied data, but do not establish archived historical publication availability.",
        ],
    }
    paired.to_csv(root / "paired_predictions.csv", index=False)
    (root / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (root / "comparison.md").write_text(_markdown(report))
    (root / "reliability.svg").write_text(_reliability_svg(report))
    return report
