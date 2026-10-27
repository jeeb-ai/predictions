"""Synthetic live-inference checks; no production performance claims."""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from predictions.features import FEATURE_COLUMNS, build_features
from predictions.features_v2 import MODEL2_FEATURE_COLUMNS
from predictions.inference import (
    PROBABILITY_SCOPE,
    predict_pregame,
    prepare_pregame_features,
    train_live_model,
)

AS_OF = pd.Timestamp("2026-10-27T12:00:00Z")
TARGET = "synthetic-2026-target"


@pytest.fixture
def games(monkeypatch):
    monkeypatch.setattr("predictions.inference._now", lambda: AS_OF + pd.Timedelta(seconds=1))
    rows = []
    for season in [2024, 2025]:
        for week in range(1, 5):
            kickoff = pd.Timestamp(f"{season}-09-01T17:00:00Z") + pd.Timedelta(weeks=week - 1)
            rows.append(
                {
                    "game_id": f"synthetic-{season}-{week}",
                    "season": season,
                    "week": week,
                    "kickoff": kickoff,
                    "home_score": 24 if week % 2 else 10,
                    "away_score": 17,
                }
            )
    for game_id, kickoff, home, away in [
        ("synthetic-2026-opener", "2026-09-01T17:00:00Z", 27, 14),
        ("synthetic-2026-not-yet-known", "2026-10-28T17:00:00Z", 37, 3),
        (TARGET, "2026-10-31T17:00:00Z", np.nan, np.nan),
    ]:
        rows.append(
            {
                "game_id": game_id,
                "season": 2026,
                "week": 1 if "opener" in game_id else 8,
                "kickoff": pd.Timestamp(kickoff),
                "home_score": home,
                "away_score": away,
            }
        )
    frame = pd.DataFrame(rows)
    frame["game_type"] = "REG"
    frame["home_team"] = "BUF"
    frame["away_team"] = "NE"
    frame["neutral"] = 0
    frame["kickoff_time_imputed"] = False
    frame["result_available_at"] = frame["kickoff"] + pd.Timedelta(hours=24)
    return frame


def aggregates(games):
    teams, quarterbacks = [], []
    for game in games.itertuples(index=False):
        for team, opponent, value in [("BUF", "NE", 2.0), ("NE", "BUF", -1.0)]:
            teams.append(
                {
                    "game_id": game.game_id,
                    "team": team,
                    "opponent": opponent,
                    "plays": 10,
                    "epa_sum": value,
                    "successes": 5,
                    "explosive_plays": 2,
                }
            )
            quarterbacks.append(
                {
                    "game_id": game.game_id,
                    "team": team,
                    "qb_id": f"QB_{team}",
                    "dropbacks": 7,
                    "epa_sum": value,
                    "successes": 4,
                }
            )
    return pd.DataFrame(teams), pd.DataFrame(quarterbacks)


def prepared(games, version):
    team, quarterback = aggregates(games)
    return prepare_pregame_features(
        games, as_of=AS_OF, model_version=version, team_games=team, qb_games=quarterback
    )


def fit(features, tmp_path, version="v1", **kwargs):
    return train_live_model(
        features, tmp_path / "models", as_of=AS_OF, season=2026, model_version=version, **kwargs
    )


def target(features):
    return features.loc[features["game_id"].eq(TARGET)].copy()


@pytest.mark.parametrize(
    "version,columns", [("v1", FEATURE_COLUMNS), ("v2", MODEL2_FEATURE_COLUMNS)]
)
def test_live_fit_predict_and_immutable_forecasts(games, tmp_path, version, columns):
    features = prepared(games, version)
    bundle = fit(features, tmp_path, version)
    assert bundle["training_seasons"] == [2024, 2025]
    assert bundle["training_games"] == 8
    assert bundle["feature_columns"] == columns
    assert pd.Timestamp(bundle["train_max_result_available_at"]) < pd.Timestamp(
        bundle["training_cutoff_at"]
    )
    first = predict_pregame(
        bundle["bundle_path"], target(features), as_of=AS_OF, output_dir=tmp_path / "forecasts"
    )
    second = predict_pregame(
        bundle["bundle_path"], target(features), as_of=AS_OF, output_dir=tmp_path / "forecasts"
    )
    forecast = first["forecasts"][0]
    assert len(first["forecasts"]) == 1
    assert forecast["game_id"] == TARGET
    assert 0 < forecast["home_win_probability"] < 1
    assert forecast["home_win_probability"] + forecast["away_win_probability"] == 1
    assert forecast["probability_scope"] == PROBABILITY_SCOPE
    assert forecast["tie_probability"] is None
    assert forecast["calibration_status"] == "raw_logistic_not_recalibrated"
    assert forecast["feature_cutoff_at"] == AS_OF.isoformat()
    assert forecast["model_sha256"] == bundle["model_sha256"]
    assert forecast["features_sha256"] == second["forecasts"][0]["features_sha256"]
    assert first["artifact"]["path"] != second["artifact"]["path"]
    file = Path(first["artifact"]["path"])
    assert hashlib.sha256(file.read_bytes()).hexdigest() == first["artifact"]["sha256"]
    assert json.loads(file.read_text())["forecasts"] == first["forecasts"]


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_current_and_future_outcomes_cannot_enter_as_of_histories(games, version):
    team, quarterback = aggregates(games)
    original = prepare_pregame_features(
        games, as_of=AS_OF, model_version=version, team_games=team, qb_games=quarterback
    )
    future = games["result_available_at"].ge(AS_OF)
    changed = games.copy()
    changed.loc[future, ["home_score", "away_score"]] = [99, 0]
    future_ids = set(games.loc[future, "game_id"])
    team.loc[team["game_id"].isin(future_ids), "epa_sum"] = 9000
    quarterback.loc[quarterback["game_id"].isin(future_ids), "epa_sum"] = 9000
    quarterback.loc[quarterback["game_id"].isin(future_ids), "qb_id"] = "FUTURE_STARTER"
    result = prepare_pregame_features(
        changed, as_of=AS_OF, model_version=version, team_games=team, qb_games=quarterback
    )
    columns = FEATURE_COLUMNS if version == "v1" else MODEL2_FEATURE_COLUMNS
    pd.testing.assert_frame_equal(target(original)[columns], target(result)[columns])
    assert pd.isna(target(result)["home_win"].iloc[0])
    if version == "v2":
        assert target(result)["home_qb_id_proxy"].iloc[0] == "QB_BUF"


def test_history_available_exactly_as_of_is_excluded(games):
    index = games.index[games["game_id"].eq("synthetic-2026-opener")][0]
    games.loc[index, "result_available_at"] = AS_OF
    strict = prepared(games, "v2")
    games.loc[index, "result_available_at"] = AS_OF - pd.Timedelta(seconds=1)
    prior = prepared(games, "v2")
    assert target(strict)["home_qb_proxy_game_id"].iloc[0] == "synthetic-2025-4"
    assert target(prior)["home_qb_proxy_game_id"].iloc[0] == "synthetic-2026-opener"
    assert target(prior)["advanced_history_available_through"].iloc[0] < AS_OF


def test_original_historical_feature_values_remain_unchanged(games):
    original = build_features(games)
    live = prepared(games, "v1")
    historical = original["season"].lt(2026)
    pd.testing.assert_frame_equal(
        original.loc[historical, FEATURE_COLUMNS], live.loc[historical, FEATURE_COLUMNS]
    )
    assert live.loc[historical, "feature_cutoff_at"].equals(live.loc[historical, "kickoff"])


def test_current_season_scores_never_fit_live_coefficients(games, tmp_path):
    original = fit(prepared(games, "v1"), tmp_path)
    games.loc[games["season"].eq(2026), ["home_score", "away_score"]] = [0, 70]
    changed = fit(prepared(games, "v1"), tmp_path)
    assert original["training_game_ids_sha256"] == changed["training_game_ids_sha256"]
    assert original["model_sha256"] == changed["model_sha256"]


def test_training_respects_strict_season_opening_availability(games, tmp_path):
    opener = games.loc[games["game_id"].eq("synthetic-2026-opener"), "kickoff"].iloc[0]
    games.loc[games["game_id"].eq("synthetic-2025-4"), "result_available_at"] = opener
    bundle = fit(prepared(games, "v1"), tmp_path)
    assert bundle["training_games"] == 7
    assert pd.Timestamp(bundle["train_max_result_available_at"]) < opener


def test_kickoff_based_future_features_are_rejected(games, tmp_path):
    live = prepared(games, "v1")
    bundle = fit(live, tmp_path)
    unsafe = target(build_features(games))
    with pytest.raises(ValueError, match="exact as_of"):
        predict_pregame(bundle["bundle_path"], unsafe, as_of=AS_OF)


@pytest.mark.parametrize("invalid", ["started", "imputed", "wrong_season", "finished"])
def test_ineligible_targets_cannot_receive_live_forecast(games, tmp_path, invalid):
    features = prepared(games, "v1")
    bundle = fit(features, tmp_path)
    selected = target(features)
    if invalid == "started":
        selected["kickoff"] = AS_OF
        selected["result_available_at"] = AS_OF + pd.Timedelta(hours=24)
    elif invalid == "imputed":
        selected["kickoff_time_imputed"] = True
    elif invalid == "wrong_season":
        selected["season"] = 2027
    else:
        selected[["home_score", "away_score"]] = [24, 17]
    with pytest.raises(ValueError, match="No upcoming"):
        predict_pregame(bundle["bundle_path"], selected, as_of=AS_OF)


def test_backdating_cannot_turn_postgame_output_into_live_forecast(games, tmp_path, monkeypatch):
    features = prepared(games, "v1")
    bundle = fit(features, tmp_path)
    monkeypatch.setattr("predictions.inference._now", lambda: pd.Timestamp("2027-01-01T00:00:00Z"))
    with pytest.raises(ValueError, match="No upcoming"):
        predict_pregame(bundle["bundle_path"], target(features), as_of=AS_OF)


def test_corrupt_model_fails_before_deserialization(games, tmp_path, monkeypatch):
    features = prepared(games, "v1")
    bundle = fit(features, tmp_path)
    path = Path(bundle["bundle_path"]) / "model.joblib"
    path.write_bytes(path.read_bytes() + b"corrupted")
    monkeypatch.setattr(
        joblib, "load", lambda path: pytest.fail("Unverified model was deserialized")
    )
    with pytest.raises(ValueError, match="checksum"):
        predict_pregame(bundle["bundle_path"], target(features), as_of=AS_OF)


def test_future_model_information_is_rejected(games, tmp_path):
    features = prepared(games, "v1")
    bundle = fit(features, tmp_path)
    path = Path(bundle["bundle_path"]) / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata["training_as_of"] = (AS_OF + pd.Timedelta(seconds=1)).isoformat()
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="after as_of"):
        predict_pregame(bundle["bundle_path"], target(features), as_of=AS_OF)


@pytest.mark.parametrize("nested", [False, True])
def test_training_rejects_future_observed_sources(games, tmp_path, nested):
    source = {"fetched_at": (AS_OF + pd.Timedelta(seconds=1)).isoformat()}
    if nested:
        source = {"play_by_play": {"2025": source}}
    with pytest.raises(ValueError, match="fetched after as_of"):
        fit(prepared(games, "v1"), tmp_path, source_metadata=source)
    assert not (tmp_path / "models").exists()


def test_prediction_rejects_nested_future_observed_source(games, tmp_path):
    features = prepared(games, "v1")
    bundle = fit(features, tmp_path)
    with pytest.raises(ValueError, match="fetched after as_of"):
        predict_pregame(
            bundle["bundle_path"],
            target(features),
            as_of=AS_OF,
            source_metadata={
                "play_by_play": [{"fetched_at": (AS_OF + pd.Timedelta(seconds=1)).isoformat()}]
            },
        )


def test_version_window_and_as_of_must_match(games, tmp_path):
    features = prepared(games, "v1")
    bundle = fit(features, tmp_path)
    wrong_window = prepare_pregame_features(games, as_of=AS_OF, model_version="v1", window=4)
    with pytest.raises(ValueError, match="version/window"):
        predict_pregame(bundle["bundle_path"], target(wrong_window), as_of=AS_OF)
    with pytest.raises(ValueError, match="timezone-aware"):
        prepare_pregame_features(games, as_of="2026-10-27", model_version="v1")
    with pytest.raises(ValueError, match="future"):
        predict_pregame(bundle["bundle_path"], target(features), as_of=AS_OF + pd.Timedelta(days=1))
