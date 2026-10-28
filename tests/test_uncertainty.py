"""Synthetic past-only estimation-uncertainty checks, never performance claims."""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from predictions.inference import predict_pregame, prepare_pregame_features, train_live_model
from predictions.uncertainty import fit_ensemble, training_blocks, validate_bootstrap_settings

AS_OF = pd.Timestamp("2026-10-28T12:00:00Z")
TARGET = "synthetic-2026-target"


@pytest.fixture
def games(monkeypatch):
    monkeypatch.setattr("predictions.inference._now", lambda: AS_OF + pd.Timedelta(seconds=1))
    rows = []
    for season in [2023, 2024, 2025]:
        for week in range(1, 7):
            for pair, (home, away) in enumerate([("BUF", "NE"), ("KC", "NYJ")]):
                rows.append(
                    {
                        "game_id": f"synthetic-{season}-{week}-{pair}",
                        "season": season,
                        "week": week,
                        "kickoff": pd.Timestamp(f"{season}-09-01T17:00:00Z")
                        + pd.Timedelta(weeks=week - 1),
                        "home_team": home,
                        "away_team": away,
                        "home_score": 27 if (week + pair) % 2 else 10,
                        "away_score": 17,
                    }
                )
    for identity, kickoff, score in [
        ("synthetic-2026-opener", "2026-09-01T17:00:00Z", 24),
        ("synthetic-future", "2026-10-29T17:00:00Z", 70),
        (TARGET, "2026-10-31T17:00:00Z", np.nan),
    ]:
        rows.append(
            {
                "game_id": identity,
                "season": 2026,
                "week": 1 if "opener" in identity else 8,
                "kickoff": pd.Timestamp(kickoff),
                "home_team": "BUF",
                "away_team": "NE",
                "home_score": score,
                "away_score": 17 if pd.notna(score) else np.nan,
            }
        )
    frame = pd.DataFrame(rows)
    frame["game_type"] = "REG"
    frame["neutral"] = 0
    frame["kickoff_time_imputed"] = False
    frame["result_available_at"] = frame["kickoff"] + pd.Timedelta(hours=24)
    return frame


def prepared(games):
    return prepare_pregame_features(games, as_of=AS_OF, model_version="v1")


def fit(features, tmp_path, **kwargs):
    return train_live_model(
        features,
        tmp_path / "models",
        model_version="v1",
        as_of=AS_OF,
        season=2026,
        **kwargs,
    )


def forecast(bundle, features):
    return predict_pregame(
        bundle["bundle_path"], features.loc[features["game_id"].eq(TARGET)], as_of=AS_OF
    )["forecasts"][0]


def metadata_path(bundle):
    return Path(bundle["bundle_path"]) / "metadata.json"


def test_disabled_default_keeps_legacy_point_model_and_forecast_schema(games, tmp_path):
    features = prepared(games)
    legacy = fit(features, tmp_path)
    uncertain = fit(features, tmp_path, bootstrap_samples=20)
    assert legacy["model_sha256"] == uncertain["model_sha256"]
    assert "uncertainty_ensemble" not in legacy
    prediction = forecast(legacy, features)
    assert "probability_interval" not in prediction
    assert "model_training_game_ids_sha256" not in prediction


def test_seed_reproduces_intervals_with_complementary_away_bounds(games, tmp_path):
    features = prepared(games)
    first = fit(features, tmp_path, bootstrap_samples=20, bootstrap_seed=123)
    second = fit(features, tmp_path, bootstrap_samples=20, bootstrap_seed=123)
    assert first["uncertainty_ensemble"]["sha256"] == second["uncertainty_ensemble"]["sha256"]
    one, two = forecast(first, features), forecast(second, features)
    interval = one["probability_interval"]
    assert interval["home"] == two["probability_interval"]["home"]
    assert 0 <= interval["home"]["lower"] <= interval["home"]["upper"] <= 1
    assert interval["away"]["lower"] == 1 - interval["home"]["upper"]
    assert interval["away"]["upper"] == 1 - interval["home"]["lower"]
    assert interval["model_id"] == one["model_id"]
    assert interval["model_sha256"] == one["model_sha256"]
    assert interval["game_id"] == one["game_id"]
    assert interval["features_sha256"] == one["features_sha256"]
    assert interval["as_of"] == one["as_of"]
    assert interval["training_game_ids_sha256"] == one["model_training_game_ids_sha256"]
    assert interval["scope"] == "model_estimation_conditional_on_features"
    assert interval["calibration"] == "not_a_calibrated_predictive_coverage_guarantee"
    assert interval["training_blocks"] == 18


def test_current_season_outcomes_do_not_fit_point_or_bootstrap_models(games, tmp_path):
    original = fit(prepared(games), tmp_path, bootstrap_samples=20)
    changed = games.copy()
    changed.loc[changed["season"].eq(2026), ["home_score", "away_score"]] = [0, 99]
    result = fit(prepared(changed), tmp_path, bootstrap_samples=20)
    assert original["training_game_ids_sha256"] == result["training_game_ids_sha256"]
    assert original["model_sha256"] == result["model_sha256"]
    assert original["uncertainty_ensemble"]["sha256"] == result["uncertainty_ensemble"]["sha256"]
    assert original["training_seasons"] == [2023, 2024, 2025]


def test_unknown_future_outcomes_do_not_change_target_interval(games, tmp_path):
    features = prepared(games)
    bundle = fit(features, tmp_path, bootstrap_samples=20)
    first = forecast(bundle, features)
    changed = games.copy()
    changed.loc[changed["result_available_at"].ge(AS_OF), ["home_score", "away_score"]] = [0, 99]
    second = forecast(bundle, prepared(changed))
    assert first["probability_interval"] == second["probability_interval"]
    assert first["home_win_probability"] == second["home_win_probability"]


def test_bootstrap_retains_strict_annual_training_availability(games, tmp_path):
    opener = games.loc[games["game_id"].eq("synthetic-2026-opener"), "kickoff"].iloc[0]
    games.loc[games["game_id"].eq("synthetic-2025-6-1"), "result_available_at"] = opener
    bundle = fit(prepared(games), tmp_path, bootstrap_samples=20)
    info = bundle["uncertainty_ensemble"]
    assert bundle["training_games"] == 35
    assert info["training_games"] == 35
    assert info["training_cutoff_at"] == bundle["training_cutoff_at"]
    assert pd.Timestamp(bundle["train_max_result_available_at"]) < opener


@pytest.mark.parametrize("samples", [True, -1, 1, 19, 2001, 20.0])
def test_invalid_bootstrap_counts_fail_before_artifacts(games, tmp_path, samples):
    with pytest.raises(ValueError, match="bootstrap_samples"):
        fit(prepared(games), tmp_path, bootstrap_samples=samples)
    assert not (tmp_path / "models").exists()


@pytest.mark.parametrize("confidence", [0, 1, -0.1, np.nan, True])
def test_invalid_confidence_levels_fail(games, tmp_path, confidence):
    with pytest.raises(ValueError, match="confidence_level"):
        fit(prepared(games), tmp_path, bootstrap_samples=20, confidence_level=confidence)


def test_single_week_training_cannot_manufacture_an_interval(games, tmp_path):
    features = prepared(games)
    historical = features["season"].lt(2026)
    features.loc[historical, "season"] = 2025
    features.loc[historical, "week"] = 1
    with pytest.raises(ValueError, match="at least two"):
        fit(features, tmp_path, bootstrap_samples=20)


def test_complete_week_blocks_are_resampled_together(games, monkeypatch):
    features = prepared(games)
    training = features.loc[features["season"].lt(2026)].copy()
    columns = ["margin_diff", "rest_days_diff"]
    blocks, names = training_blocks(training)
    assert len(blocks) == len(names) == 18
    assert all(len(block) == 2 for block in blocks)

    class FixedDraw:
        def integers(self, low, high, size):
            return np.zeros(size, dtype=int)  # Repeatedly draw the entire first week.

    observed = []

    class RecordingModel:
        def fit(self, x, y):
            observed.append(x.index.tolist())
            assert len(np.unique(y)) == 2
            return self

    monkeypatch.setattr("predictions.uncertainty.np.random.default_rng", lambda seed: FixedDraw())
    monkeypatch.setattr("predictions.uncertainty.make_pipeline", lambda *args: RecordingModel())
    digest = hashlib.sha256("\n".join(training["game_id"]).encode()).hexdigest()
    _, info = fit_ensemble(
        training,
        columns,
        samples=20,
        confidence_level=0.95,
        seed=42,
        training_cutoff_at="2026-09-01T17:00:00Z",
        training_game_ids_sha256=digest,
    )
    assert info["attempts"] == 20
    expected = training.iloc[np.tile(blocks[0], len(blocks))].index.tolist()
    assert all(indices == expected for indices in observed)


def test_single_class_bootstrap_draws_are_skipped_and_reported(games, tmp_path):
    features = prepared(games)
    historical = features["season"].lt(2026)
    # All home losses share one block and all home wins another. Some block
    # draws contain only one class, and cannot silently yield dummy models.
    features.loc[historical, "season"] = 2025
    features.loc[historical, "week"] = features.loc[historical, "home_win"].astype(int) + 1
    bundle = fit(features, tmp_path, bootstrap_samples=20)
    info = bundle["uncertainty_ensemble"]
    assert info["skipped_single_class_draws"] > 0
    assert info["attempts"] > 20
    assert info["bootstrap_samples"] == 20


def test_corrupt_ensemble_is_rejected_before_its_deserialization(games, tmp_path, monkeypatch):
    features = prepared(games)
    bundle = fit(features, tmp_path, bootstrap_samples=20)
    ensemble = Path(bundle["bundle_path"]) / "ensemble.joblib"
    ensemble.write_bytes(ensemble.read_bytes() + b"corrupt")
    load = joblib.load

    def guarded(path):
        if Path(path).name == "ensemble.joblib":
            pytest.fail("Unverified ensemble was deserialized")
        return load(path)

    monkeypatch.setattr(joblib, "load", guarded)
    with pytest.raises(ValueError, match="checksum"):
        forecast(bundle, features)


@pytest.mark.parametrize("field", ["training_game_ids_sha256", "training_cutoff_at", "model_id"])
def test_mismatched_or_future_ensemble_provenance_is_rejected(games, tmp_path, field):
    features = prepared(games)
    bundle = fit(features, tmp_path, bootstrap_samples=20)
    path = metadata_path(bundle)
    metadata = json.loads(path.read_text())
    metadata["uncertainty_ensemble"][field] = (
        "2027-01-01T00:00:00Z" if field == "training_cutoff_at" else "different"
    )
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="differs|provenance"):
        forecast(bundle, features)


def test_changed_replica_schema_or_missing_replica_fails(games, tmp_path):
    features = prepared(games)
    bundle = fit(features, tmp_path, bootstrap_samples=20)
    directory = Path(bundle["bundle_path"])
    path = directory / "ensemble.joblib"
    artifact = joblib.load(path)
    artifact["models"].pop()
    joblib.dump(artifact, path)
    meta_path = metadata_path(bundle)
    metadata = json.loads(meta_path.read_text())
    metadata["uncertainty_ensemble"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    meta_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="number"):
        forecast(bundle, features)


def test_replica_with_unknown_feature_schema_is_rejected(games, tmp_path):
    features = prepared(games)
    bundle = fit(features, tmp_path, bootstrap_samples=20)
    path = Path(bundle["bundle_path"]) / "ensemble.joblib"
    artifact = joblib.load(path)
    artifact["models"][0].named_steps["standardscaler"].feature_names_in_[0] = "unknown_feature"
    joblib.dump(artifact, path)
    meta_path = metadata_path(bundle)
    metadata = json.loads(meta_path.read_text())
    metadata["uncertainty_ensemble"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    meta_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="feature/class schema"):
        forecast(bundle, features)


def test_single_class_retry_exhaustion_never_saves_dummy_intervals(games, tmp_path, monkeypatch):
    features = prepared(games)
    historical = features["season"].lt(2026)
    features.loc[historical, "season"] = 2025
    features.loc[historical, "week"] = features.loc[historical, "home_win"].astype(int) + 1

    class SingleClassDraw:
        calls = 0

        def integers(self, low, high, size):
            self.calls += 1
            return np.zeros(size, dtype=int)

    random = SingleClassDraw()
    monkeypatch.setattr("predictions.uncertainty.np.random.default_rng", lambda seed: random)
    with pytest.raises(ValueError, match="Insufficient two-class"):
        fit(features, tmp_path, bootstrap_samples=20)
    assert random.calls == 200
    assert not list((tmp_path / "models").glob("*/ensemble.joblib"))
    assert not list((tmp_path / "models").glob("*/metadata.json"))


def test_valid_settings_include_explicit_disabled_mode():
    validate_bootstrap_settings(0, 0.95, 42)
    validate_bootstrap_settings(2000, 0.99, 0)
    with pytest.raises(ValueError, match="seed"):
        validate_bootstrap_settings(20, 0.95, True)
