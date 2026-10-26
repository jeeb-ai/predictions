"""Offline integration checks use explicitly synthetic schedule fixtures."""

import hashlib
import json
import sqlite3

import pandas as pd
import pytest

from predictions.cli import main
from predictions.data import SOURCE_URL


@pytest.fixture
def schedule_csv(tmp_path, monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("Offline CLI test attempted a network request")

    monkeypatch.setattr("requests.get", no_network)
    rows = []
    for season in [2019, 2020, 2021]:
        for week in range(1, 5):
            for pair, (home, away) in enumerate([("BUF", "NE"), ("KC", "NYJ")]):
                rows.append(
                    {
                        "game_id": f"synthetic_{season}_{week}_{pair}",
                        "season": season,
                        "game_type": "REG",
                        "week": week,
                        "gameday": f"{season}-09-{1 + (week - 1) * 7:02d}",
                        "gametime": "13:00",
                        "home_team": home,
                        "away_team": away,
                        "home_score": 24 if (week + pair) % 2 else 10,
                        "away_score": 17,
                        "location": "Home",
                    }
                )
    path = tmp_path / "synthetic-games.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _arguments(path, tmp_path):
    return [
        "backtest",
        "--input",
        str(path),
        "--output-dir",
        str(tmp_path / "reports"),
        "--as-of",
        "2022-03-01T00:00:00Z",
        "--min-train-seasons",
        "1",
        "--bootstrap-samples",
        "10",
    ]


def test_offline_cli_produces_auditable_backtest(schedule_csv, tmp_path, capsys):
    assert main(_arguments(schedule_csv, tmp_path)) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["configuration"]["holdout_season"] == 2021
    assert report["validation"]["games"] == 8
    assert report["holdout"]["games"] == 8
    assert report["source_metadata"]["source_url"] is None
    assert (
        report["source_metadata"]["sha256"] == hashlib.sha256(schedule_csv.read_bytes()).hexdigest()
    )
    run_dir = tmp_path / "reports" / report["run_id"]
    forecasts = pd.read_csv(run_dir / "predictions.csv")
    assert len(forecasts) == 16
    assert forecasts["home_win_probability"].between(0, 1).all()
    assert forecasts.loc[forecasts["is_holdout"], "train_max_season"].eq(2020).all()
    assert (run_dir / "metrics.md").exists()
    assert (run_dir / "reliability.svg").exists()
    assert (run_dir / "model-2021.joblib").exists()
    with sqlite3.connect(tmp_path / "reports" / "predictions.sqlite") as connection:
        assert connection.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0] == 16


def test_cli_rejects_incomplete_holdout(schedule_csv, tmp_path, capsys):
    games = pd.read_csv(schedule_csv)
    games.loc[games.index[-1], ["home_score", "away_score"]] = float("nan")
    games.to_csv(schedule_csv, index=False)
    assert main(_arguments(schedule_csv, tmp_path) + ["--through-season", "2021"]) == 1
    assert "incomplete" in capsys.readouterr().err
    assert not (tmp_path / "reports").exists()


def test_cli_cannot_use_scores_not_available_as_of(schedule_csv, tmp_path, capsys):
    args = _arguments(schedule_csv, tmp_path)
    args[args.index("--as-of") + 1] = "2021-09-10T00:00:00Z"
    assert main(args + ["--through-season", "2021"]) == 1
    assert "incomplete" in capsys.readouterr().err


def test_cli_reports_malformed_metadata_without_traceback(schedule_csv, tmp_path, capsys):
    schedule_csv.with_suffix(".json").write_text("[]")
    assert main(_arguments(schedule_csv, tmp_path)) == 1
    assert "metadata" in capsys.readouterr().err


def test_cli_preserves_timestamped_snapshot_provenance(schedule_csv, tmp_path, capsys):
    checksum = hashlib.sha256(schedule_csv.read_bytes()).hexdigest()
    metadata = {
        "sha256": checksum,
        "source_url": SOURCE_URL,
        "fetched_at": "2022-01-01T00:00:00Z",
    }
    (tmp_path / f"20220101T000000Z_{checksum}.json").write_text(json.dumps(metadata))
    assert main(_arguments(schedule_csv, tmp_path)) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["source_metadata"]["source_url"] == SOURCE_URL
    assert report["source_metadata"]["fetched_at"] == metadata["fetched_at"]
    assert pd.Timestamp(report["source_metadata"]["as_of"]) == pd.Timestamp(metadata["fetched_at"])


def test_cached_live_scores_do_not_become_final_by_waiting(schedule_csv, tmp_path, capsys):
    checksum = hashlib.sha256(schedule_csv.read_bytes()).hexdigest()
    metadata = {
        "sha256": checksum,
        "source_url": SOURCE_URL,
        "fetched_at": "2021-09-10T00:00:00Z",
    }
    (tmp_path / f"20210910T000000Z_{checksum}.json").write_text(json.dumps(metadata))
    assert main(_arguments(schedule_csv, tmp_path) + ["--through-season", "2021"]) == 1
    assert "incomplete" in capsys.readouterr().err


def test_cli_refuses_local_refresh(schedule_csv, tmp_path, capsys):
    assert main(_arguments(schedule_csv, tmp_path) + ["--refresh"]) == 1
    assert "--refresh cannot be combined with --input" in capsys.readouterr().err
