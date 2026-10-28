"""Past-only bootstrap uncertainty for fixed chronological live features.

Whole season/week blocks, rather than individual games, are resampled from
already eligible training rows. These intervals describe fitted-model sampling
variation conditional on those features. They do not estimate tie risk, source
revisions, roster changes, or calibrated future-outcome coverage.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

METHOD = "training_week_block_bootstrap"
SCOPE = "model_estimation_conditional_on_features"
CALIBRATION = "not_a_calibrated_predictive_coverage_guarantee"


def validate_bootstrap_settings(samples: int, confidence_level: float, seed: int) -> None:
    if (
        isinstance(samples, bool)
        or not isinstance(samples, int)
        or (samples != 0 and not 20 <= samples <= 2000)
    ):
        raise ValueError("bootstrap_samples must be zero or an integer from 20 to 2000")
    if (
        isinstance(confidence_level, bool)
        or not isinstance(confidence_level, (int, float))
        or not np.isfinite(confidence_level)
        or not 0 < confidence_level < 1
    ):
        raise ValueError("confidence_level must be a finite number strictly between zero and one")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
        raise ValueError("bootstrap_seed must be an integer from zero to 4294967295")


def training_blocks(frame: pd.DataFrame) -> tuple[list[np.ndarray], list[str]]:
    """Return complete, deterministic season/week blocks of training-row positions."""
    if not {"season", "kickoff"} <= set(frame):
        raise ValueError("Bootstrap training requires season and kickoff")
    weeks = (
        frame["week"]
        if "week" in frame
        else pd.to_datetime(frame["kickoff"], utc=True).dt.isocalendar().week
    )
    numeric_weeks = pd.to_numeric(weeks, errors="raise")
    numeric_seasons = pd.to_numeric(frame["season"], errors="raise")
    for name, values in [("week", numeric_weeks), ("season", numeric_seasons)]:
        if (
            values.isna().any()
            or not np.isfinite(values).all()
            or (values <= 0).any()
            or not np.equal(values, np.floor(values)).all()
        ):
            raise ValueError(f"Bootstrap {name} labels must be positive finite integers")
    labels = numeric_seasons.astype(int).astype(str) + ":" + numeric_weeks.astype(int).astype(str)
    names = sorted(labels.unique().tolist())
    if len(names) < 2:
        raise ValueError(
            "Bootstrap uncertainty requires at least two historical season/week blocks"
        )
    array = labels.to_numpy()
    return [np.flatnonzero(array == name) for name in names], names


def fit_ensemble(
    training: pd.DataFrame,
    columns: list[str],
    *,
    samples: int,
    confidence_level: float,
    seed: int,
    training_cutoff_at: str,
    training_game_ids_sha256: str,
) -> tuple[dict, dict]:
    """Fit replicas exclusively from the caller's strictly historical training set."""
    validate_bootstrap_settings(samples, confidence_level, seed)
    if samples == 0:
        raise ValueError("An uncertainty ensemble needs at least 20 bootstrap samples")
    if not columns or len(set(columns)) != len(columns) or not set(columns) <= set(training):
        raise ValueError("Bootstrap feature columns must be present, nonempty, and unique")
    required = {"home_win", "result_available_at", "feature_cutoff_at", "kickoff", "game_id"}
    if not required <= set(training):
        raise ValueError("Bootstrap training lacks outcome or pregame provenance columns")
    cutoff = pd.Timestamp(training_cutoff_at)
    if pd.isna(cutoff) or cutoff.tzinfo is None:
        raise ValueError("Bootstrap training cutoff must be timezone-aware")
    if training.empty or training["game_id"].duplicated().any():
        raise ValueError("Bootstrap training requires unique, nonempty historical game IDs")
    available = pd.to_datetime(training["result_available_at"], utc=True)
    kickoffs = pd.to_datetime(training["kickoff"], utc=True)
    feature_cutoffs = pd.to_datetime(training["feature_cutoff_at"], utc=True)
    if (
        available.isna().any()
        or kickoffs.isna().any()
        or (available >= cutoff).any()
        or (available <= kickoffs).any()
        or feature_cutoffs.ne(kickoffs).any()
    ):
        raise ValueError("Bootstrap rows must use historical pregame features and prior outcomes")
    actual_ids_hash = hashlib.sha256(
        "\n".join(training["game_id"].astype(str)).encode()
    ).hexdigest()
    if actual_ids_hash != training_game_ids_sha256:
        raise ValueError("Bootstrap training IDs do not match the point model")
    y = training["home_win"].to_numpy()
    if not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError("Bootstrap training requires binary outcomes and both classes")
    if not np.isfinite(training[columns].to_numpy(dtype=float)).all():
        raise ValueError("Bootstrap features must be finite")
    blocks, names = training_blocks(training)
    random = np.random.default_rng(seed)
    models, attempts, skipped = [], 0, 0
    with threadpool_limits(limits=2):
        while len(models) < samples and attempts < samples * 10:
            selected = random.integers(0, len(blocks), size=len(blocks))
            positions = np.concatenate([blocks[index] for index in selected])
            attempts += 1
            if len(np.unique(y[positions])) != 2:
                skipped += 1
                continue
            model = make_pipeline(
                StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=42)
            )
            model.fit(training.iloc[positions][columns], y[positions].astype(int))
            models.append(model)
    if len(models) != samples:
        raise ValueError(
            "Insufficient two-class bootstrap draws; no uncertainty interval was saved"
        )
    summary = {
        "schema_version": 1,
        "method": METHOD,
        "scope": SCOPE,
        "calibration": CALIBRATION,
        "bootstrap_samples": samples,
        "confidence_level": float(confidence_level),
        "seed": seed,
        "training_blocks": len(blocks),
        "training_block_labels_sha256": hashlib.sha256("\n".join(names).encode()).hexdigest(),
        "training_cutoff_at": cutoff.isoformat(),
        "training_game_ids_sha256": training_game_ids_sha256,
        "training_games": len(training),
        "attempts": attempts,
        "skipped_single_class_draws": skipped,
        "feature_columns": list(columns),
        "limitations": (
            "Conditional on fixed lagged features and historical training data; does not refit "
            "EPA/feature construction or measure tie/cancellation, roster, source-revision, "
            "or cross-week dependence uncertainty. Not a calibrated predictive interval."
        ),
    }
    return {**summary, "models": models}, summary


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def predict_intervals(directory: Path, metadata: dict, targets: pd.DataFrame) -> list[dict]:
    """Verify a trusted bundle's ensemble before estimating target probability bounds."""
    specification = metadata.get("uncertainty_ensemble")
    if not isinstance(specification, dict) or specification.get("file") != "ensemble.joblib":
        raise ValueError("Live uncertainty artifact metadata is invalid")
    validate_bootstrap_settings(
        specification.get("bootstrap_samples"),
        specification.get("confidence_level"),
        specification.get("seed"),
    )
    if specification["bootstrap_samples"] == 0:
        raise ValueError("A saved uncertainty artifact cannot contain zero samples")
    for key in ["training_cutoff_at", "training_game_ids_sha256", "feature_columns"]:
        if specification.get(key) != metadata.get(key):
            raise ValueError(f"Uncertainty {key} differs from the live point model")
    if (
        specification.get("method") != METHOD
        or specification.get("scope") != SCOPE
        or specification.get("calibration") != CALIBRATION
        or specification.get("model_id") != metadata.get("model_id")
        or specification.get("model_sha256") != metadata.get("model_sha256")
    ):
        raise ValueError("Uncertainty provenance does not match the live point model")
    path = directory / "ensemble.joblib"
    if specification.get("sha256") != file_digest(path):
        raise ValueError("Live uncertainty ensemble checksum mismatch")
    ensemble = joblib.load(path)
    if not isinstance(ensemble, dict) or not isinstance(ensemble.get("models"), list):
        raise ValueError("Live uncertainty ensemble schema is invalid")
    keys = [
        "schema_version",
        "method",
        "scope",
        "calibration",
        "bootstrap_samples",
        "confidence_level",
        "seed",
        "training_blocks",
        "training_block_labels_sha256",
        "training_cutoff_at",
        "training_game_ids_sha256",
        "training_games",
        "attempts",
        "skipped_single_class_draws",
        "feature_columns",
        "limitations",
    ]
    if any(ensemble.get(key) != specification.get(key) for key in keys):
        raise ValueError("Uncertainty artifact content differs from its metadata")
    models = ensemble["models"]
    if len(models) != specification["bootstrap_samples"]:
        raise ValueError("Uncertainty ensemble has an incorrect number of fitted replicas")
    columns = metadata["feature_columns"]
    draws = []
    with threadpool_limits(limits=2):
        for model in models:
            if list(getattr(model, "feature_names_in_", [])) != columns or list(
                getattr(model, "classes_", [])
            ) != [0, 1]:
                raise ValueError("Uncertainty replica feature/class schema differs")
            probability = model.predict_proba(targets[columns])[:, 1]
            if (
                not np.isfinite(probability).all()
                or not ((0 <= probability) & (probability <= 1)).all()
            ):
                raise ValueError("Uncertainty replica returned invalid probabilities")
            draws.append(probability)
    alpha = (1 - specification["confidence_level"]) / 2
    lower, upper = np.quantile(np.asarray(draws), [alpha, 1 - alpha], axis=0)
    return [
        {
            "home": {"lower": float(low), "upper": float(high)},
            "away": {"lower": float(1 - high), "upper": float(1 - low)},
            "model_id": metadata["model_id"],
            "model_sha256": metadata["model_sha256"],
            "ensemble_sha256": specification["sha256"],
            "confidence_level": specification["confidence_level"],
            "method": METHOD,
            "scope": SCOPE,
            "calibration": CALIBRATION,
            "bootstrap_samples": specification["bootstrap_samples"],
            "training_blocks": specification["training_blocks"],
            "seed": specification["seed"],
            "training_cutoff_at": metadata["training_cutoff_at"],
            "training_game_ids_sha256": metadata["training_game_ids_sha256"],
        }
        for low, high in zip(lower, upper, strict=True)
    ]
