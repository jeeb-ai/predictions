"""Basketball-specific cutoffs, possessions, rest, calibrated probabilities and UI."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from predictions.league_forecast import issue_league, publish_league
from predictions.league_model import fit_league, line_probabilities, pmf_for, winner_explanation
from predictions.nba import NBA_COLUMNS, canonical_nba, nba_features


def sample_games():
    rows = []
    for year in range(2015, 2024):
        for i in range(16):
            day = pd.Timestamp(f"{year}-10-01", tz="UTC") + pd.Timedelta(days=i * 3)
            rows.append(
                {
                    "sport": "NBA",
                    "game_id": f"{year}-{i}",
                    "season": year,
                    "kickoff": day,
                    "schedule_day": str(day.date()),
                    "available_at": day + pd.Timedelta(days=2),
                    "home_team": "BOS",
                    "away_team": "NYK",
                    "home_name": "Boston Celtics",
                    "away_name": "New York Knicks",
                    "home_score": 115 if i % 3 else 90,
                    "away_score": 105,
                    "home_efficiency": 115 if i % 3 else 90,
                    "away_efficiency": 105,
                    "pace": 100.0,
                    "neutral": False,
                }
            )
    return pd.DataFrame(rows)


def test_target_and_future_results_cannot_change_inputs():
    frame = sample_games()
    original = nba_features(frame)
    target = frame.game_id.iloc[-4]
    cutoff = frame.loc[frame.game_id.eq(target), "kickoff"].iloc[0]
    changed = frame.copy()
    changed.loc[
        changed.available_at.ge(cutoff),
        ["home_score", "away_score", "home_efficiency", "away_efficiency"],
    ] = [500, 1, 500, 1]
    mutated = nba_features(changed)
    np.testing.assert_allclose(
        original.loc[original.game_id.eq(target), NBA_COLUMNS],
        mutated.loc[mutated.game_id.eq(target), NBA_COLUMNS],
    )
    history = original.loc[original.game_id.eq(target), "history_available_through"].iloc[0]
    assert history < cutoff


def test_back_to_back_uses_schedule_even_when_previous_result_unavailable():
    f = sample_games().iloc[:2].copy()
    f.loc[f.index[1], "kickoff"] = f.iloc[0].kickoff + pd.Timedelta(days=1, hours=4)
    f.loc[f.index[1], "schedule_day"] = str((f.iloc[0].kickoff + pd.Timedelta(days=1)).date())
    out = nba_features(f).iloc[1]
    assert out.home_back_to_back == 1 and out.home_rest == 1
    assert out.home_history == 0


def test_box_score_possessions_and_nominal_home_are_verified():
    frame = pd.DataFrame(
        [
            {
                "GAME_ID": "g",
                "TEAM_ID": 1,
                "TEAM_ABBREVIATION": "BOS",
                "TEAM_NAME": "Boston",
                "MATCHUP": "BOS @ NYK",
                "GAME_DATE": "2025-01-01",
                "season": 2024,
                "PTS": 100,
                "FGA": 90,
                "OREB": 10,
                "TOV": 15,
                "FTA": 20,
                "MIN": 240,
            },
            {
                "GAME_ID": "g",
                "TEAM_ID": 2,
                "TEAM_ABBREVIATION": "NYK",
                "TEAM_NAME": "New York",
                "MATCHUP": "BOS @ NYK",
                "GAME_DATE": "2025-01-01",
                "season": 2024,
                "PTS": 110,
                "FGA": 90,
                "OREB": 10,
                "TOV": 15,
                "FTA": 20,
                "MIN": 240,
            },
        ]
    )
    canonical = canonical_nba(frame).iloc[0]
    assert canonical.home_team == "NYK"
    assert canonical.possessions == pytest.approx(103.8)
    assert canonical.home_efficiency == pytest.approx(110 / 103.8 * 100)
    frame.loc[1, "MATCHUP"] = "NYK @ BOS"
    with pytest.raises(ValueError, match="Contradictory"):
        canonical_nba(frame)


def test_league_training_calibration_and_explanation_are_independent():
    f = nba_features(sample_games())
    train = f.loc[f.season.lt(2022)]
    cal = f.loc[f.season.eq(2022)]
    target = f.loc[f.season.eq(2023)].iloc[-1]
    model = fit_league(train, cal, NBA_COLUMNS, "NBA")
    p, e = winner_explanation(model, target)
    assert p == pytest.approx(
        1
        / (
            1
            + np.exp(
                -(e["intercept"] + sum(x["log_odds_contribution"] for x in e["contributions"]))
            )
        )
    )
    means = np.array([110.0, 108.0])
    pmf = pmf_for(model, means)
    assert pmf.sum() == pytest.approx(1) and np.trace(pmf) == 0
    q = line_probabilities(pmf, -2, 220)
    assert sum(q["home_cover"].values()) == pytest.approx(1) and q["home_cover"]["push"] > 0
    assert line_probabilities(pmf, -2.5, 220.5)["over"]["push"] == 0
    with pytest.raises(ValueError):
        fit_league(cal, train, NBA_COLUMNS, "NBA")


def test_timestamped_nba_forecast_renders_behind_existing_auth(tmp_path, monkeypatch):
    from predictions import dashboard
    from predictions.config import initialize_research, load_config

    past = sample_games()
    now = pd.Timestamp.now(tz="UTC")
    future = past.iloc[-1:].copy()
    future["game_id"] = "upcoming"
    future["season"] = 2024
    future["kickoff"] = now + pd.Timedelta(days=3)
    future["schedule_day"] = str(future.kickoff.iloc[0].date())
    future["available_at"] = future.kickoff + pd.Timedelta(days=2)
    future[["home_score", "away_score", "home_efficiency", "away_efficiency"]] = np.nan
    features = nba_features(pd.concat([past, future], ignore_index=True), as_of=now)
    report = {
        "sport": "NBA",
        "model_version": "nba-v1",
        "columns": NBA_COLUMNS,
        "holdout_season": 2023,
        "folds": [{"test_season": 2023}],
    }
    batch = issue_league(
        features, report, tmp_path / "models", as_of=now, source={"synthetic": True}
    )
    configpath = tmp_path / "research.json"
    initialize_research(configpath)
    c = load_config(configpath)
    publish_league(batch, c["paths"]["state_dir"])
    monkeypatch.setenv("NFL_RESEARCH_CONFIG", str(configpath))
    app = AppTest.from_string("from predictions.dashboard import main\nmain()").run()
    assert not app.metric
    app.text_input[0].set_value(Path(c["dashboard"]["token_file"]).read_text().strip())
    app.button[0].click().run(timeout=20)
    assert not app.exception
    assert any("NBA ·" in v.value for v in app.markdown)
    assert any("Average projected final score" in v.value for v in app.markdown)
    assert dashboard.build_dashboard_view(c)["forecasts"][0]["sport"] == "NBA"


def test_corrupt_league_snapshot_does_not_hide_nfl_or_other_valid_data(tmp_path):
    from predictions.config import initialize_research, load_config
    from predictions.dashboard import build_dashboard_view

    config = tmp_path / "research.json"
    initialize_research(config)
    c = load_config(config)
    root = Path(c["paths"]["state_dir"])
    (root / "nba-predictions.json").write_text("{broken private content")
    view = build_dashboard_view(c)
    assert view["league_errors"] == ["NBA"]
    assert view["forecasts"] == []


def test_cached_response_fetched_before_availability_stays_masked(monkeypatch, tmp_path):
    from predictions.nba import fetch_nba

    columns = [
        "GAME_ID",
        "TEAM_ID",
        "TEAM_ABBREVIATION",
        "TEAM_NAME",
        "MATCHUP",
        "GAME_DATE",
        "SEASON_ID",
        "PTS",
        "FGA",
        "OREB",
        "TOV",
        "FTA",
        "MIN",
    ]
    data = {
        "resultSets": [
            {
                "headers": columns,
                "rowSet": [
                    [
                        "g",
                        1,
                        "BOS",
                        "Boston",
                        "BOS vs. NYK",
                        "2025-01-01",
                        "22024",
                        110,
                        90,
                        10,
                        15,
                        20,
                        240,
                    ],
                    [
                        "g",
                        2,
                        "NYK",
                        "New York",
                        "NYK @ BOS",
                        "2025-01-01",
                        "22024",
                        100,
                        90,
                        10,
                        15,
                        20,
                        240,
                    ],
                ],
            }
        ]
    }
    path = tmp_path / "raw.json"
    path.write_text(json.dumps(data))

    def cached(year, *args, **kwargs):
        part = pd.DataFrame(data["resultSets"][0]["rowSet"], columns=columns)
        part["season"] = year
        return canonical_nba(part), {
            "provider": "test fixture",
            "fetched_at": "2025-01-02T00:00:00Z",
        }

    monkeypatch.setattr("predictions.nba_history.load_season", cached)
    frame, _ = fetch_nba(tmp_path, 2024, 2024)
    assert frame.home_score.isna().all()
    assert nba_features(frame, as_of=pd.Timestamp("2025-02-01T00:00:00Z")).home_history.eq(0).all()
