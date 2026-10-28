"""Season-frozen models and auditable forecasts made before NFL kickoff.

These helpers do not change the historical feature builders or backtests. Live
features additionally cap information at the forecast's actual ``as_of`` time,
rather than assuming every result available by a future kickoff is known now.
Only trusted local model bundles may be loaded: joblib is not a safe format for
untrusted files, even when an accompanying checksum matches.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from predictions.features import FEATURE_COLUMNS, build_features
from predictions.features_v2 import MODEL2_FEATURE_COLUMNS, build_features_v2

PROBABILITY_SCOPE = "conditional_on_decisive_result"
MODEL_VERSIONS = {"v1": "logistic-regression-v1", "v2": "logistic-regression-v2"}


def _now() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc))


def _timestamp(value, name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError(f"{name} must be a valid timezone-aware timestamp")
    return timestamp.tz_convert("UTC")


def _family(version: str) -> str:
    if version in MODEL_VERSIONS:
        return version
    for family, canonical in MODEL_VERSIONS.items():
        if version == canonical:
            return family
    raise ValueError("model_version must be v1 or v2")


def _columns(family: str) -> list[str]:
    return list(FEATURE_COLUMNS if family == "v1" else MODEL2_FEATURE_COLUMNS)


def _encode(value) -> bytes:
    return (
        json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n"
    ).encode()


def _digest(value) -> str:
    return hashlib.sha256(_encode(value)).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_digest() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _validate_sources(source: dict | None, cutoff: pd.Timestamp) -> dict:
    """Reject later-observed snapshots, including nested seasonal PBP sources."""
    if source is None:
        return {}
    if not isinstance(source, dict):
        raise ValueError("source_metadata must be an object")

    def visit(value):
        if isinstance(value, dict):
            if value.get("fetched_at") is not None:
                fetched = _timestamp(value["fetched_at"], "source fetched_at")
                if fetched > cutoff:
                    raise ValueError("Source snapshot was fetched after as_of")
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    visit(source)
    return dict(source)


def _audit_columns(frame: pd.DataFrame) -> list[str]:
    return [
        name
        for name in frame
        if name.endswith("history_available_through") or name.endswith("qb_proxy_available_at")
    ]


def _validate_features(features: pd.DataFrame, family: str) -> pd.DataFrame:
    required = {
        "game_id",
        "season",
        "game_type",
        "home_team",
        "away_team",
        "kickoff",
        "result_available_at",
        "home_score",
        "away_score",
        "home_win",
        "feature_cutoff_at",
        "feature_version",
        "history_available_through",
        *_columns(family),
    }
    missing = required - set(features)
    if missing:
        raise ValueError(f"Missing live inference columns: {sorted(missing)}")
    frame = features.copy()
    if frame["game_id"].isna().any() or frame["game_id"].duplicated().any():
        raise ValueError("Live inference requires unique nonmissing game IDs")
    for name in ["kickoff", "result_available_at", "feature_cutoff_at", *_audit_columns(frame)]:
        frame[name] = pd.to_datetime(frame[name], utc=True)
    if frame[["kickoff", "result_available_at", "feature_cutoff_at"]].isna().any().any():
        raise ValueError("Live inference requires valid schedule and cutoff timestamps")
    if (frame["result_available_at"] <= frame["kickoff"]).any():
        raise ValueError("Results must become available strictly after kickoff")
    if (frame["feature_cutoff_at"] > frame["kickoff"]).any():
        raise ValueError("Feature cutoffs cannot follow kickoff")
    for name in _audit_columns(frame):
        if (frame[name].notna() & (frame[name] >= frame["feature_cutoff_at"])).any():
            raise ValueError(f"{name} must be strictly earlier than the feature cutoff")
    if not np.isfinite(frame[_columns(family)].to_numpy(dtype=float)).all():
        raise ValueError("Live model features must be finite numbers")
    return frame


def prepare_pregame_features(
    games: pd.DataFrame,
    *,
    as_of,
    model_version: str = "v2",
    team_games: pd.DataFrame | None = None,
    qb_games: pd.DataFrame | None = None,
    window: int = 8,
) -> pd.DataFrame:
    """Build histories known now, preserving historical training-row cutoffs.

    A result available at ``as_of`` is excluded, just like a result available
    later. V2 PBP histories inherit the masked schedule's completion status;
    target-game PBP, QB identities, and future results cannot become features.
    Rest days still refer to the known scheduled kickoff, which is available
    pregame. The existing historical builders are left unchanged.
    """
    cutoff = _timestamp(as_of, "as_of")
    family = _family(model_version)
    frame = games.copy()
    required = {"kickoff", "result_available_at", "home_score", "away_score"}
    if not required <= set(frame):
        raise ValueError(f"Schedule lacks live cutoff columns: {sorted(required - set(frame))}")
    frame["result_available_at"] = pd.to_datetime(frame["result_available_at"], utc=True)
    frame.loc[frame["result_available_at"] >= cutoff, ["home_score", "away_score"]] = np.nan
    if family == "v2":
        if team_games is None or qb_games is None:
            raise ValueError("Model 2.0 live features require causal team and QB PBP aggregates")
        features = build_features_v2(frame, team_games, qb_games, window=window)
    else:
        features = build_features(frame, window=window)
    features.loc[features["kickoff"] > cutoff, "feature_cutoff_at"] = cutoff
    features["inference_as_of"] = cutoff
    return _validate_features(features, family)


def train_live_model(
    features: pd.DataFrame,
    output_dir: Path,
    *,
    model_version: str = "v2",
    as_of,
    season: int,
    source_metadata: dict | None = None,
    bootstrap_samples: int = 0,
    confidence_level: float = 0.95,
    bootstrap_seed: int = 42,
) -> dict:
    """Fit a fresh immutable annual bundle using older-season outcomes only.

    This intentionally does not promote a backtest artifact to production.
    Even midseason, current-season games never enter fitted coefficients or
    scaling. They may enter later live pregame form after they are known.
    """
    cutoff = _timestamp(as_of, "as_of")
    from predictions.uncertainty import fit_ensemble, validate_bootstrap_settings

    validate_bootstrap_settings(bootstrap_samples, confidence_level, bootstrap_seed)
    family = _family(model_version)
    source = _validate_sources(source_metadata, cutoff)
    if not isinstance(season, int) or isinstance(season, bool):
        raise ValueError("season must be an integer")
    frame = _validate_features(features, family)
    current = frame.loc[frame["season"].eq(season)]
    if current.empty:
        raise ValueError("The schedule must include the requested live season")
    opener = current["kickoff"].min()
    training_cutoff = min(cutoff, opener)
    eligible = frame.loc[
        frame["season"].lt(season)
        & frame["game_type"].eq("REG")
        & frame["result_available_at"].lt(training_cutoff)
        & frame["home_score"].notna()
        & frame["away_score"].notna()
        & frame["home_score"].ne(frame["away_score"])
    ].sort_values(["kickoff", "game_id"], kind="stable")
    if eligible.empty or eligible["home_win"].nunique() != 2:
        raise ValueError("Live training requires historical decisive games with both classes")
    expected = (eligible["home_score"] > eligible["away_score"]).astype(int)
    if not np.array_equal(eligible["home_win"].to_numpy(), expected.to_numpy()):
        raise ValueError("Live training labels must agree with completed home/away scores")
    if eligible["feature_cutoff_at"].ne(eligible["kickoff"]).any():
        raise ValueError("Historical training features must retain their own pregame cutoffs")
    versions = eligible["feature_version"].dropna().unique().tolist()
    if len(versions) != 1:
        raise ValueError("Live training requires one consistent feature version")
    columns = _columns(family)
    model = make_pipeline(
        StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=42)
    )
    model.fit(eligible[columns], expected)
    created = _now()
    model_id = (
        f"live-{family}-{season}-{created.strftime('%Y%m%dT%H%M%S%fZ')}-{uuid.uuid4().hex[:8]}"
    )
    directory = Path(output_dir) / model_id
    directory.mkdir(parents=True, exist_ok=False)
    model_path = directory / "model.joblib"
    joblib.dump(model, model_path)
    metadata = {
        "schema_version": 1,
        "artifact_type": "nfl_live_model",
        "model_id": model_id,
        "model_family": family,
        "model_version": MODEL_VERSIONS[family],
        "season": season,
        "created_at": created.isoformat(),
        "training_as_of": cutoff.isoformat(),
        "training_cutoff_at": training_cutoff.isoformat(),
        "season_opening_kickoff": opener.isoformat(),
        "training_seasons": sorted(int(value) for value in eligible["season"].unique()),
        "training_games": len(eligible),
        "training_game_ids_sha256": hashlib.sha256(
            "\n".join(eligible["game_id"].astype(str)).encode()
        ).hexdigest(),
        "train_max_result_available_at": eligible["result_available_at"].max().isoformat(),
        "feature_columns": columns,
        "feature_version": versions[0],
        "model_file": "model.joblib",
        "model_sha256": _file_digest(model_path),
        "source_code_sha256": _code_digest(),
        "source_metadata": source,
        "source_metadata_sha256": _digest(source),
        "dependency_versions": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "scikit-learn", "joblib"]
        },
        "hyperparameters": {"C": 1.0, "max_iter": 2000, "random_state": 42},
        "probability_scope": PROBABILITY_SCOPE,
        "calibration_status": "raw_logistic_not_recalibrated",
        "methodology": "Older-season decisive regular-season training only; coefficients frozen for the live season",
    }
    if bootstrap_samples:
        ensemble, specification = fit_ensemble(
            eligible,
            columns,
            samples=bootstrap_samples,
            confidence_level=confidence_level,
            seed=bootstrap_seed,
            training_cutoff_at=metadata["training_cutoff_at"],
            training_game_ids_sha256=metadata["training_game_ids_sha256"],
        )
        ensemble_path = directory / "ensemble.joblib"
        joblib.dump(ensemble, ensemble_path)
        metadata["uncertainty_ensemble"] = {
            **specification,
            "file": "ensemble.joblib",
            "sha256": _file_digest(ensemble_path),
            "model_id": metadata["model_id"],
            "model_sha256": metadata["model_sha256"],
        }
    (directory / "metadata.json").write_bytes(_encode(metadata))
    return {**metadata, "bundle_path": str(directory)}


def _load_bundle(bundle_path: Path) -> tuple[object, dict]:
    directory = Path(bundle_path)
    if directory.name == "metadata.json":
        directory = directory.parent
    metadata = json.loads((directory / "metadata.json").read_text())
    if (
        not isinstance(metadata, dict)
        or metadata.get("schema_version") != 1
        or metadata.get("artifact_type") != "nfl_live_model"
    ):
        raise ValueError("Not a supported live model bundle")
    family = _family(metadata.get("model_version"))
    if metadata.get("model_file") != "model.joblib" or metadata.get("feature_columns") != _columns(
        family
    ):
        raise ValueError("Live model feature schema or artifact path does not match its version")
    model_path = directory / "model.joblib"
    if metadata.get("model_sha256") != _file_digest(model_path):
        raise ValueError("Live model checksum mismatch")
    model = joblib.load(model_path)
    if list(getattr(model, "feature_names_in_", [])) != metadata["feature_columns"] or list(
        getattr(model, "classes_", [])
    ) != [0, 1]:
        raise ValueError("Live model fitted feature/class schema does not match metadata")
    return model, metadata


def predict_pregame(
    bundle_path: Path,
    features: pd.DataFrame,
    *,
    as_of,
    source_metadata: dict | None = None,
    output_dir: Path | None = None,
    include_analysis: bool = False,
) -> dict:
    """Forecast only known, unfinished games that have not actually started.

    ``features`` may contain the whole schedule; past/finished games are
    explicitly skipped. An explicit selection with no eligible games fails.
    A caller cannot relabel a postgame computation as live by backdating as_of.
    Output probabilities are conditional on a decisive result, not unconditional
    probabilities for contracts that make ties a loss.
    """
    cutoff = _timestamp(as_of, "as_of")
    generated = _now()
    if cutoff > generated:
        raise ValueError("Live inference as_of cannot be in the future")
    model, metadata = _load_bundle(bundle_path)
    family = _family(metadata["model_version"])
    for name in ["training_as_of", "training_cutoff_at"]:
        if _timestamp(metadata[name], name) > cutoff:
            raise ValueError("Live model was trained with an information cutoff after as_of")
    if _timestamp(metadata["train_max_result_available_at"], "training availability") >= cutoff:
        raise ValueError("Live model training outcomes must be strictly earlier than as_of")
    if max(metadata["training_seasons"]) >= metadata["season"]:
        raise ValueError("Live model must not train on its forecast season")
    frame = _validate_features(features, family)
    current = frame.loc[frame["season"].eq(metadata["season"])].copy()
    known_finished = current["home_score"].notna() | current["away_score"].notna()
    started = (current["kickoff"] <= cutoff) | (current["kickoff"] <= generated)
    imputed = current.get("kickoff_time_imputed", pd.Series(False, index=current.index)).astype(
        bool
    )
    skipped = [
        {"game_id": row.game_id, "reason": "started_or_finished"}
        for row in current.loc[started | known_finished].itertuples(index=False)
    ]
    skipped.extend(
        {"game_id": row.game_id, "reason": "unknown_kickoff_time"}
        for row in current.loc[~started & ~known_finished & imputed].itertuples(index=False)
    )
    targets = current.loc[~started & ~known_finished & ~imputed].copy()
    if targets.empty:
        raise ValueError("No upcoming unfinished games with known kickoff remain for this model")
    if targets["feature_cutoff_at"].ne(cutoff).any():
        raise ValueError(
            "Prepare live features at this exact as_of; future-kickoff features are unsafe"
        )
    if targets["feature_version"].ne(metadata["feature_version"]).any():
        raise ValueError("Prediction feature version/window differs from the live model")
    source = _validate_sources(source_metadata, cutoff)
    columns = metadata["feature_columns"]
    probabilities = model.predict_proba(targets[columns])[:, 1]
    intervals = None
    if "uncertainty_ensemble" in metadata:
        from predictions.uncertainty import predict_intervals

        directory = Path(bundle_path)
        if directory.name == "metadata.json":
            directory = directory.parent
        intervals = predict_intervals(directory, metadata, targets)
    forecasts = []
    source_sha = _digest(source)
    for (_, row), probability in zip(targets.iterrows(), probabilities, strict=True):
        values = {name: float(row[name]) for name in columns}
        forecast = {
            "forecast_id": uuid.uuid4().hex,
            "game_id": str(row["game_id"]),
            "season": int(row["season"]),
            "home_team": str(row["home_team"]),
            "away_team": str(row["away_team"]),
            "kickoff": row["kickoff"].isoformat(),
            "as_of": cutoff.isoformat(),
            "feature_cutoff_at": cutoff.isoformat(),
            "generated_at": generated.isoformat(),
            "model_id": metadata["model_id"],
            "model_version": metadata["model_version"],
            "model_sha256": metadata["model_sha256"],
            "model_training_cutoff_at": metadata["training_cutoff_at"],
            "model_train_max_result_available_at": metadata["train_max_result_available_at"],
            "model_source_metadata_sha256": metadata["source_metadata_sha256"],
            "feature_version": row["feature_version"],
            "features": values,
            "features_sha256": _digest(
                {
                    "values": values,
                    "feature_version": row["feature_version"],
                    "as_of": cutoff.isoformat(),
                }
            ),
            "home_win_probability": float(probability),
            "away_win_probability": float(1 - probability),
            "probability_scope": PROBABILITY_SCOPE,
            "tie_probability": None,
            "calibration_status": metadata["calibration_status"],
            "source_metadata": source,
            "source_metadata_sha256": source_sha,
            "source_code_sha256": _code_digest(),
        }
        if include_analysis:
            from predictions.matchup import explain_prediction

            forecast["matchup_analysis"] = explain_prediction(
                model, row, columns, float(probability)
            )
        if intervals is not None:
            forecast["probability_interval"] = {
                **intervals[len(forecasts)],
                "game_id": forecast["game_id"],
                "features_sha256": forecast["features_sha256"],
                "as_of": cutoff.isoformat(),
            }
            forecast["model_training_game_ids_sha256"] = metadata["training_game_ids_sha256"]
        for name in _audit_columns(targets):
            forecast[name] = row[name].isoformat() if pd.notna(row[name]) else None
        for name in [
            "home_qb_id_proxy",
            "away_qb_id_proxy",
            "home_qb_proxy_game_id",
            "away_qb_proxy_game_id",
        ]:
            if name in row:
                forecast[name] = row[name] if pd.notna(row[name]) else None
        forecasts.append(forecast)
    batch = {
        "schema_version": 1,
        "artifact_type": "nfl_pregame_forecasts",
        "batch_id": uuid.uuid4().hex,
        "generated_at": generated.isoformat(),
        "as_of": cutoff.isoformat(),
        "model_id": metadata["model_id"],
        "probability_scope": PROBABILITY_SCOPE,
        "forecasts": forecasts,
        "skipped": skipped,
    }
    if output_dir is not None:
        directory = Path(output_dir) / f"forecasts-{batch['batch_id']}"
        directory.mkdir(parents=True, exist_ok=False)
        path = directory / "forecasts.json"
        encoded = _encode(batch)
        with path.open("xb") as handle:
            handle.write(encoded)
        batch["artifact"] = {"path": str(path), "sha256": hashlib.sha256(encoded).hexdigest()}
    return batch
