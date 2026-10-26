"""Season-frozen walk-forward evaluation and auditable forecast artifacts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from predictions.features import FEATURE_COLUMNS

MODEL_VERSION = "logistic-regression-v1"


def _losses(y: np.ndarray, probability: np.ndarray) -> dict[str, np.ndarray]:
    clipped = np.clip(probability, 1e-15, 1 - 1e-15)
    return {
        "accuracy": ((probability >= 0.5) == y).astype(float),
        "brier_score": (probability - y) ** 2,
        "log_loss": -(y * np.log(clipped) + (1 - y) * np.log1p(-clipped)),
    }


def evaluate_probabilities(
    y_true,
    probabilities,
    *,
    clusters=None,
    bootstrap_samples: int = 1000,
    seed: int = 42,
) -> dict:
    """Measure binary forecasts; optional CIs resample complete supplied clusters."""
    y = np.asarray(y_true, dtype=float)
    probability = np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or probability.shape != y.shape or not len(y):
        raise ValueError("Metrics require equally sized, nonempty one-dimensional arrays")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Metric labels must be binary (ties must be excluded)")
    if not np.isfinite(probability).all() or not ((probability >= 0) & (probability <= 1)).all():
        raise ValueError("Probabilities must be finite numbers between zero and one")
    if bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be nonnegative")
    losses = _losses(y, probability)
    result = {name: float(values.mean()) for name, values in losses.items()}
    result.update(
        games=int(len(y)),
        mean_predicted_home_win=float(probability.mean()),
        observed_home_win_rate=float(y.mean()),
    )
    bins = np.minimum((probability * 10).astype(int), 9)
    reliability = []
    ece = 0.0
    for index in range(10):
        selected = bins == index
        count = int(selected.sum())
        predicted = float(probability[selected].mean()) if count else None
        observed = float(y[selected].mean()) if count else None
        if count:
            ece += count / len(y) * abs(predicted - observed)
        reliability.append(
            {
                "lower": index / 10,
                "upper": (index + 1) / 10,
                "count": count,
                "mean_predicted": predicted,
                "observed_home_win_rate": observed,
            }
        )
    result["expected_calibration_error"] = float(ece)
    result["reliability_bins"] = reliability
    result["confidence_intervals_95"] = None
    if clusters is not None and bootstrap_samples:
        labels = np.asarray(clusters)
        if labels.shape != y.shape:
            raise ValueError("Cluster labels must match the outcome array")
        _, inverse = np.unique(labels.astype(str), return_inverse=True)
        count = int(inverse.max()) + 1
        sizes = np.bincount(inverse)
        draws = np.random.default_rng(seed).integers(0, count, size=(bootstrap_samples, count))
        sampled_sizes = sizes[draws].sum(axis=1)
        intervals = {}
        for name, values in losses.items():
            sums = np.bincount(inverse, weights=values)
            sampled_means = sums[draws].sum(axis=1) / sampled_sizes
            intervals[name] = np.quantile(sampled_means, [0.025, 0.975]).tolist()
        result["confidence_intervals_95"] = intervals
        result["bootstrap_clusters"] = count
    return result


def _summary(frame: pd.DataFrame, bootstrap_samples: int, seed: int) -> dict | None:
    if frame.empty:
        return None
    clusters = frame["season"].astype(str) + ":" + frame["week"].astype(str)
    options = {"clusters": clusters, "bootstrap_samples": bootstrap_samples, "seed": seed}
    y = frame["home_win"].to_numpy()
    return {
        "games": int(len(frame)),
        "model": evaluate_probabilities(y, frame["home_win_probability"], **options),
        "home_rate_baseline": evaluate_probabilities(
            y, frame["baseline_home_win_probability"], **options
        ),
        "always_home_accuracy": float(y.mean()),
    }


def _jsonable_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        if isinstance(result[column].dtype, pd.DatetimeTZDtype):
            result[column] = result[column].map(
                lambda value: value.isoformat() if pd.notna(value) else None
            )
    return result.astype(object).where(pd.notna(result), None)


def _write_forecasts(database: Path, frame: pd.DataFrame, report: dict) -> None:
    """Append a new run transactionally; never replace existing forecasts."""
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(
            "CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, generated_at TEXT NOT NULL, metadata_json TEXT NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS forecasts ("
            "run_id TEXT NOT NULL REFERENCES runs(run_id), game_id TEXT NOT NULL, "
            "model_id TEXT NOT NULL, home_win_probability REAL NOT NULL, "
            "prediction_json TEXT NOT NULL, PRIMARY KEY (run_id, game_id))"
        )
        connection.execute(
            "INSERT INTO runs VALUES (?, ?, ?)",
            (report["run_id"], report["generated_at"], json.dumps(report, allow_nan=False)),
        )
        connection.executemany(
            "INSERT INTO forecasts VALUES (?, ?, ?, ?, ?)",
            [
                (
                    report["run_id"],
                    row["game_id"],
                    row["model_id"],
                    row["home_win_probability"],
                    json.dumps(row, allow_nan=False),
                )
                for row in frame.to_dict(orient="records")
            ],
        )


def _markdown(report: dict) -> str:
    lines = [
        "# NFL pregame logistic-regression backtest",
        "",
        f"Run: `{report['run_id']}`",
        "",
        "| Split | Games | Accuracy | 95% accuracy CI | Brier | Log loss | ECE | Home-rate baseline accuracy |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for name in ["validation", "holdout"]:
        group = report[name]
        if group is None:
            continue
        metric = group["model"]
        interval = (metric["confidence_intervals_95"] or {}).get("accuracy")
        ci = f"{interval[0]:.1%}–{interval[1]:.1%}" if interval else "not computed"
        lines.append(
            f"| {name} | {group['games']} | {metric['accuracy']:.1%} | {ci} | "
            f"{metric['brier_score']:.4f} | {metric['log_loss']:.4f} | "
            f"{metric['expected_calibration_error']:.4f} | {group['home_rate_baseline']['accuracy']:.1%} |"
        )
    lines.extend(
        [
            "",
            "The JSON report contains full per-season metrics, reliability bins, and baseline losses.",
            "",
            "## Method and limits",
            "",
            *[f"- {note}" for note in report["methodology"]],
            "",
            "## Per-season model results",
            "",
            "| Season | Split | Games | Accuracy | Brier | Log loss | ECE |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for season, group in report["per_season"].items():
        metric = group["model"]
        split = (
            "holdout" if int(season) == report["configuration"]["holdout_season"] else "validation"
        )
        lines.append(
            f"| {season} | {split} | {group['games']} | {metric['accuracy']:.1%} | "
            f"{metric['brier_score']:.4f} | {metric['log_loss']:.4f} | {metric['expected_calibration_error']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def _reliability_svg(report: dict) -> str:
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="660" height="490" viewBox="0 0 660 490">',
        '<rect width="660" height="490" fill="white"/>',
        '<text x="330" y="26" text-anchor="middle" font-family="sans-serif" font-size="18">Observed calibration (fixed probability bins)</text>',
        '<path d="M80 410 L480 410 M80 410 L80 50" fill="none" stroke="black"/>',
        '<path d="M80 410 L480 50" stroke="#999" stroke-dasharray="5 5"/>',
    ]
    for tick in np.linspace(0, 1, 6):
        x, y = 80 + 400 * tick, 410 - 360 * tick
        parts.extend(
            [
                f'<text x="{x}" y="431" text-anchor="middle" font-family="sans-serif" font-size="12">{tick:.1f}</text>',
                f'<text x="67" y="{y + 4}" text-anchor="end" font-family="sans-serif" font-size="12">{tick:.1f}</text>',
            ]
        )
    for index, (name, color) in enumerate([("validation", "#2463a5"), ("holdout", "#bd4517")]):
        group = report[name]
        if group is None:
            continue
        points = [
            (
                80 + 400 * cell["mean_predicted"],
                410 - 360 * cell["observed_home_win_rate"],
                cell["count"],
            )
            for cell in group["model"]["reliability_bins"]
            if cell["count"]
        ]
        coords = " ".join(f"{x:.2f},{y:.2f}" for x, y, _ in points)
        parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>')
        for x, y, count in points:
            parts.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"><title>{name}: {count} games</title></circle>'
            )
        parts.append(
            f'<text x="495" y="{80 + 25 * index}" fill="{color}" font-family="sans-serif" font-size="13">{name}</text>'
        )
    parts.extend(
        [
            '<text x="280" y="464" text-anchor="middle" font-family="sans-serif" font-size="14">Mean predicted home-win probability</text>',
            '<text x="22" y="230" transform="rotate(-90 22 230)" text-anchor="middle" font-family="sans-serif" font-size="14">Observed home-win frequency</text>',
            "</svg>",
        ]
    )
    return "\n".join(parts)


def run_backtest(
    features: pd.DataFrame,
    output_dir: Path,
    *,
    holdout_season: int,
    min_train_seasons: int = 5,
    bootstrap_samples: int = 1000,
    seed: int = 42,
    source_metadata: dict | None = None,
) -> dict:
    """Fit once per season using older, already available results; reserve final season."""
    if min_train_seasons < 1:
        raise ValueError("min_train_seasons must be at least one")
    if bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be nonnegative")
    required = set(FEATURE_COLUMNS) | {
        "game_id",
        "season",
        "game_type",
        "kickoff",
        "result_available_at",
        "home_win",
        "feature_cutoff_at",
        "history_available_through",
    }
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"Missing backtest columns: {sorted(missing)}")
    frame = features.copy()
    for column in [
        "kickoff",
        "result_available_at",
        "feature_cutoff_at",
        "history_available_through",
    ]:
        frame[column] = pd.to_datetime(frame[column], utc=True)
    if frame["game_id"].duplicated().any():
        raise ValueError("Duplicate game IDs are not allowed")
    if frame["kickoff"].isna().any() or frame["feature_cutoff_at"].ne(frame["kickoff"]).any():
        raise ValueError("Each feature cutoff must equal its pregame kickoff")
    if (
        frame["history_available_through"].notna()
        & (frame["history_available_through"] >= frame["kickoff"])
    ).any():
        raise ValueError("Historical outcomes must be available strictly before kickoff")
    if "week" not in frame:
        frame["week"] = frame["kickoff"].dt.isocalendar().week.astype(int)
    selected = frame.loc[(frame["game_type"] == "REG") & (frame["season"] <= holdout_season)].copy()
    tie_count = (
        int(
            (
                selected["home_score"].notna() & selected["home_score"].eq(selected["away_score"])
            ).sum()
        )
        if {"home_score", "away_score"} <= set(selected)
        else 0
    )
    selected = selected.loc[selected["home_win"].notna()].sort_values(["kickoff", "game_id"])
    if not np.isin(selected["home_win"], [0, 1]).all():
        raise ValueError("Backtest labels must be binary")
    if not np.isfinite(selected[FEATURE_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError("Features must contain only finite numbers")
    if selected["result_available_at"].isna().any():
        raise ValueError("Completed games require result availability timestamps")
    if (selected["result_available_at"] <= selected["kickoff"]).any():
        raise ValueError("Completed results must become available after their own kickoff")
    seasons = sorted(int(value) for value in selected["season"].unique())
    if holdout_season not in seasons:
        raise ValueError(
            "The requested holdout season has no completed decisive regular-season games"
        )
    evaluation_seasons = [
        year for year in seasons if sum(other < year for other in seasons) >= min_train_seasons
    ]
    if holdout_season not in evaluation_seasons:
        raise ValueError("Not enough prior seasons for the requested holdout")
    timestamp = datetime.now(timezone.utc)
    run_id = timestamp.strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:8]
    generated_at = timestamp.isoformat()
    output_dir = Path(output_dir)
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    code_digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        code_digest.update(path.name.encode())
        code_digest.update(path.read_bytes())
    code_sha = code_digest.hexdigest()
    source = dict(source_metadata or {})
    versions = {
        name: importlib.metadata.version(name)
        for name in ["numpy", "pandas", "scikit-learn", "joblib"]
    }
    model_records, forecasts = [], []
    for year in evaluation_seasons:
        test = selected.loc[selected["season"] == year].copy()
        # Freeze before the season opens, even if its first game is a tie and
        # therefore absent from the binary evaluation targets.
        cutoff = frame.loc[
            (frame["game_type"] == "REG") & (frame["season"] == year), "kickoff"
        ].min()
        train = selected.loc[
            (selected["season"] < year) & (selected["result_available_at"] < cutoff)
        ]
        if train["season"].nunique() < min_train_seasons or train["home_win"].nunique() != 2:
            raise ValueError(
                f"Season {year} lacks sufficient strictly historical training data or both classes"
            )
        model = make_pipeline(
            StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
        )
        model.fit(train[FEATURE_COLUMNS], train["home_win"].astype(int))
        model_id = f"{run_id}-{year}"
        test["home_win_probability"] = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
        test["baseline_home_win_probability"] = float(train["home_win"].mean())
        test["run_id"] = run_id
        test["model_id"] = model_id
        test["model_version"] = MODEL_VERSION
        test["generated_at"] = generated_at
        test["training_cutoff_at"] = cutoff
        test["train_max_result_available_at"] = train["result_available_at"].max()
        test["train_max_season"] = int(train["season"].max())
        test["is_holdout"] = year == holdout_season
        test["source_sha256"] = source.get("sha256")
        test["source_code_sha256"] = code_sha
        forecasts.append(test)
        fitted = model.named_steps["logisticregression"]
        metadata = {
            "model_id": model_id,
            "model_version": MODEL_VERSION,
            "season": year,
            "generated_at": generated_at,
            "training_cutoff_at": cutoff.isoformat(),
            "train_max_result_available_at": train["result_available_at"].max().isoformat(),
            "training_seasons": sorted(int(value) for value in train["season"].unique()),
            "training_games": int(len(train)),
            "training_game_ids_sha256": hashlib.sha256(
                "\n".join(train["game_id"].astype(str)).encode()
            ).hexdigest(),
            "feature_columns": FEATURE_COLUMNS,
            "source_metadata": source,
            "source_code_sha256": code_sha,
            "dependency_versions": versions,
            "scaled_coefficients": dict(
                zip(FEATURE_COLUMNS, fitted.coef_[0].tolist(), strict=True)
            ),
            "intercept": float(fitted.intercept_[0]),
            "hyperparameters": {"C": 1.0, "max_iter": 2000, "random_state": seed},
        }
        joblib.dump(model, run_dir / f"model-{year}.joblib")
        (run_dir / f"model-{year}.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False) + "\n"
        )
        model_records.append(metadata)
    predictions = pd.concat(forecasts, ignore_index=True)
    report = {
        "schema_version": 1,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "generated_at": generated_at,
        "source_metadata": source,
        "source_code_sha256": code_sha,
        "runtime": {"python": platform.python_version(), **versions},
        "configuration": {
            "holdout_season": holdout_season,
            "min_train_seasons": min_train_seasons,
            "bootstrap_samples": bootstrap_samples,
            "seed": seed,
            "feature_columns": FEATURE_COLUMNS,
        },
        "excluded_tied_regular_season_games": tie_count,
        "validation": _summary(
            predictions.loc[~predictions["is_holdout"]], bootstrap_samples, seed
        ),
        "holdout": _summary(predictions.loc[predictions["is_holdout"]], bootstrap_samples, seed),
        "per_season": {
            str(year): _summary(
                predictions.loc[predictions["season"] == year], bootstrap_samples, seed
            )
            for year in evaluation_seasons
        },
        "models": model_records,
        "methodology": [
            "Regular-season decisive games only; tied games are excluded from binary targets (their past form may count half a win).",
            "A StandardScaler and fixed C=1 logistic regression are fitted separately before each evaluation season, using older-season outcomes available strictly before its first kickoff.",
            "The latest selected season is a separate holdout: no fitting, model selection, tuning, or probability recalibration uses its targets. Earlier holdout games can enter later pregame rolling form once available.",
            "Features use completed historical outcomes only, under a conservative kickoff-plus-24-hours availability assumption, not archived publication timestamps; corrected historical source values are not point-in-time snapshots.",
            "Probabilities are raw logistic outputs; calibration is measured, not fitted. Reliability uses ten fixed-width bins, including probability 1 in the final bin.",
            "95% percentile intervals resample season/week blocks with fixed fitted forecasts; they do not include model refitting, source revision uncertainty, or dependence across weeks.",
            "The baseline predicts the training-set home-win rate; always-home accuracy is also reported. These retrospective results do not establish a trading edge.",
        ],
    }
    serializable = _jsonable_frame(predictions)
    serializable.to_csv(run_dir / "predictions.csv", index=False)
    (run_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (run_dir / "metrics.md").write_text(_markdown(report))
    (run_dir / "reliability.svg").write_text(_reliability_svg(report))
    _write_forecasts(output_dir / "predictions.sqlite", serializable, report)
    return report
