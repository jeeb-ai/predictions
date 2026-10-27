"""Offline Model 2 integration checks with explicitly synthetic local PBP."""

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from predictions.cli import main
from predictions.pbp import SOURCE_TEMPLATE


@pytest.fixture
def model2_data(tmp_path, monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("Offline Model 2 test attempted a network request")

    monkeypatch.setattr("requests.get", no_network)
    cache = tmp_path / "pbp"
    cache.mkdir()
    schedule, sources = [], {}
    for season in [2023, 2024, 2025]:
        plays = []
        for week in range(1, 5):
            for pair, (home, away) in enumerate([("BUF", "NE"), ("KC", "NYJ")]):
                game_id = f"synthetic_{season}_{week}_{pair}"
                schedule.append(
                    {
                        "game_id": game_id,
                        "season": season,
                        "game_type": "REG",
                        "week": week,
                        "gameday": f"{season}-09-{1 + 7 * (week - 1):02d}",
                        "gametime": "13:00",
                        "home_team": home,
                        "away_team": away,
                        "home_score": 24 if (week + pair) % 2 else 10,
                        "away_score": 17,
                        "location": "Home",
                    }
                )
                for index in range(8):
                    possession = home if index < 4 else away
                    home_pre = 7 if index >= 4 else 0
                    plays.append(
                        {
                            "game_id": game_id,
                            "play_id": index + 1,
                            "game_half": "Half1",
                            "qtr": 1,
                            "posteam": possession,
                            "defteam": away if possession == home else home,
                            "home_team": home,
                            "away_team": away,
                            "total_home_score": 7 if index >= 3 else 0,
                            "total_away_score": 7 if index == 7 else 0,
                            "posteam_score": home_pre if possession == home else 0,
                            "defteam_score": 0 if possession == home else home_pre,
                            "yardline_100": 70 - 15 * (index % 4),
                            "down": 1 + index % 3,
                            "ydstogo": 10,
                            "half_seconds_remaining": 1800 - 30 * index,
                            "posteam_timeouts_remaining": 3,
                            "defteam_timeouts_remaining": 3,
                            "epa": 9999.0,  # Intentionally unusable published-model output.
                            "yards_gained": 20 if index % 2 else 5,
                            "play_type": "pass",
                            "rush_attempt": 0,
                            "qb_dropback": 1,
                            "qb_scramble": 0,
                            "qb_kneel": 0,
                            "qb_spike": 0,
                            "penalty": 0,
                            "passer_player_id": f"synthetic-qb-{possession}",
                            "rusher_player_id": None,
                        }
                    )
        temporary = tmp_path / f"{season}.parquet"
        pd.DataFrame(plays).to_parquet(temporary, index=False)
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        temporary.rename(cache / f"{digest}.parquet")
        metadata = {
            "season": season,
            "sha256": digest,
            "source_url": SOURCE_TEMPLATE.format(season=season),
            "fetched_at": "2026-02-01T00:00:00Z",
            "fixture": "synthetic-only",
        }
        (cache / f"latest-{season}.json").write_text(json.dumps(metadata))
        sources[season] = metadata
    path = tmp_path / "synthetic-schedule.csv"
    pd.DataFrame(schedule).to_csv(path, index=False)
    return {"schedule": path, "cache": cache, "sources": sources}


def _arguments(data, tmp_path):
    return [
        "compare",
        "--input",
        str(data["schedule"]),
        "--pbp-dir",
        str(data["cache"]),
        "--ep-dir",
        str(tmp_path / "ep"),
        "--output-dir",
        str(tmp_path / "reports"),
        "--through-season",
        "2025",
        "--as-of",
        "2026-03-01T00:00:00Z",
        "--min-train-seasons",
        "1",
        "--bootstrap-samples",
        "20",
        "--workers",
        "2",
        "--offline",
    ]


def _model2_forecasts(report):
    return pd.read_csv(Path(report["models"]["model_2"]["run_dir"]) / "predictions.csv")


def test_offline_comparison_reconstructs_ep_and_keeps_identical_folds(
    model2_data, tmp_path, capsys
):
    assert main(_arguments(model2_data, tmp_path)) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["splits"]["validation"]["games"] == 8
    assert report["splits"]["holdout"]["games"] == 8
    assert report["source_metadata"]["causal_ep"]["raw_upstream_epa_used"] is False
    assert report["source_metadata"]["play_by_play"]["2025"]["fixture"] == "synthetic-only"
    first = report["models"]["model_1"]["models"]
    second = report["models"]["model_2"]["models"]
    assert [m["training_game_ids_sha256"] for m in first] == [
        m["training_game_ids_sha256"] for m in second
    ]
    assert all(2025 not in model["training_seasons"] for model in [*first, *second])
    forecasts = _model2_forecasts(report)
    assert forecasts["home_win_probability"].between(0, 1).all()
    assert forecasts["off_epa_per_play_diff"].abs().max() < 9999
    assert (Path(report["run_dir"]) / "paired_predictions.csv").is_file()
    assert (Path(report["run_dir"]) / "reliability.svg").is_file()


def test_pbp_snapshot_cutoff_masks_history_without_dropping_targets(model2_data, tmp_path, capsys):
    metadata = model2_data["sources"][2025].copy()
    metadata["fetched_at"] = "2025-09-03T00:00:00Z"
    (model2_data["cache"] / "latest-2025.json").write_text(json.dumps(metadata))
    assert main(_arguments(model2_data, tmp_path)) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["splits"]["holdout"]["games"] == 8
    forecasts = _model2_forecasts(report)
    final_week = forecasts.loc[(forecasts["season"] == 2025) & (forecasts["week"] == 4)]
    proxy_cutoffs = pd.to_datetime(final_week["home_qb_proxy_available_at"], utc=True)
    assert proxy_cutoffs.lt(pd.Timestamp(metadata["fetched_at"])).all()
    assert final_week["home_qb_proxy_game_id"].str.contains("synthetic_2025_1_").all()


def test_offline_missing_pbp_cache_fails_without_network(model2_data, tmp_path, capsys):
    (model2_data["cache"] / "latest-2024.json").unlink()
    assert main(_arguments(model2_data, tmp_path)) == 1
    assert "offline" in capsys.readouterr().err.lower()
    assert not (tmp_path / "reports").exists()


def test_compare_rejects_incomplete_holdout_before_fetching_pbp(model2_data, tmp_path, capsys):
    games = pd.read_csv(model2_data["schedule"])
    games.loc[games.index[-1], ["home_score", "away_score"]] = float("nan")
    games.to_csv(model2_data["schedule"], index=False)
    assert main(_arguments(model2_data, tmp_path)) == 1
    assert "incomplete" in capsys.readouterr().err
    assert not (tmp_path / "ep").exists()


def test_fetch_pbp_offline_verifies_each_local_snapshot(model2_data, capsys):
    assert (
        main(
            [
                "fetch-pbp",
                "--start-season",
                "2023",
                "--through-season",
                "2025",
                "--cache-dir",
                str(model2_data["cache"]),
                "--offline",
            ]
        )
        == 0
    )
    sources = json.loads(capsys.readouterr().out)
    assert set(sources) == {"2023", "2024", "2025"}
    assert all(Path(record["path"]).is_file() for record in sources.values())


def test_compare_rejects_invalid_workers_before_network(model2_data, tmp_path, capsys):
    args = _arguments(model2_data, tmp_path)
    args[args.index("--workers") + 1] = "0"
    assert main(args) == 1
    assert "--workers" in capsys.readouterr().err
