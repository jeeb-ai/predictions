"""Hockey cutoffs, goalie proxies and regulation/extra-time settlement semantics."""

import json

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from predictions.league_forecast import issue_league, league_verdict
from predictions.league_model import line_probabilities, winner_explanation, winner_target
from predictions.nhl import NHL_COLUMNS, fit_nhl, nhl_distribution, nhl_features, nhl_score_builder
from predictions.nhl_data import FEATURES, release_frame


def hockey_games():
    rows, keepers = [], []
    for year in range(2015, 2024):
        for i in range(18):
            day = pd.Timestamp(f"{year}-10-01", tz="UTC") + pd.Timedelta(days=i * 3)
            extra = i % 3 == 0
            win = i % 2
            row = {
                "sport": "NHL",
                "game_id": f"{year}-{i}",
                "season": year,
                "kickoff": day,
                "schedule_day": str(day.date()),
                "available_at": day + pd.Timedelta(hours=24),
                "home_team": "BOS",
                "away_team": "NYR",
                "home_name": "Boston Bruins",
                "away_name": "New York Rangers",
                "home_score": 2 if extra else 2 + win,
                "away_score": 2 if extra else 3 - win,
                "home_final_score": 2 + win,
                "away_final_score": 3 - win,
                "home_win": win,
                "went_extra": extra,
                "outcome_type": ("OT" if i % 2 else "SO") if extra else "REG",
            }
            for side in ["home", "away"]:
                row.update(
                    {
                        side + "_" + k: v
                        for k, v in {
                            "xg_for": 2.2 + win / 2,
                            "xg_against": 2.7 - win / 2,
                            "shots_for": 34 + win,
                            "shots_against": 35 - win,
                            "shot_quality": 0.07,
                            "pp_xg": 0.5,
                            "pk_xg": 0.6,
                            "goalie_id": side,
                            "goalie_name": side + " keeper",
                        }.items()
                    }
                )
                keepers.append(
                    {
                        "game_id": row["game_id"],
                        "goalie": side,
                        "kickoff": day,
                        "available_at": row["available_at"],
                        "shots_on_goal": 30,
                        "goals": 2 + win,
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(keepers)


def trained():
    games, goalies = hockey_games()
    frame = nhl_features(games, goalies)
    model = fit_nhl(
        frame.loc[frame.season.lt(2022)], frame.loc[frame.season.eq(2022)], NHL_COLUMNS, "NHL"
    )
    return frame, model


def test_hockey_target_results_goalie_and_future_events_cannot_change_forecast_inputs():
    games, goalies = hockey_games()
    original = nhl_features(games, goalies)
    target = games.iloc[-3]
    games.loc[
        games.available_at.ge(target.kickoff),
        ["home_score", "home_win", "home_xg_for", "home_goalie_id"],
    ] = [10, 1, 99, "future"]
    goalies.loc[goalies.available_at.ge(target.kickoff), ["shots_on_goal", "goals"]] = [1, 100]
    changed = nhl_features(games, goalies)
    before = original.loc[original.game_id.eq(target.game_id)].iloc[0]
    after = changed.loc[changed.game_id.eq(target.game_id)].iloc[0]
    np.testing.assert_allclose(before[NHL_COLUMNS].astype(float), after[NHL_COLUMNS].astype(float))
    assert before.home_goalie_proxy == after.home_goalie_proxy
    assert before.home_goalie_proxy["source_game"] != target.game_id
    assert "not confirmed" in before.home_goalie_proxy["status"]
    assert before.history_available_through < target.kickoff


def test_hockey_back_to_back_uses_schedule_not_unavailable_result():
    games, goalies = hockey_games()
    games = games.iloc[:2].copy()
    games.loc[games.index[1], "kickoff"] = games.iloc[0].kickoff + pd.Timedelta(days=1)
    games.loc[games.index[1], "schedule_day"] = str(games.iloc[1].kickoff.date())
    row = nhl_features(games, goalies).iloc[1]
    assert row.home_back_to_back == 1 and row.home_rest == 1
    assert row.home_history == 0  # Equal availability is explicitly excluded.
    assert row.home_goalie_proxy["id"] is None


def test_hockey_winner_labels_use_full_results_even_when_regulation_tied():
    games, _ = hockey_games()
    assert winner_target(games).tolist() == games.home_win.tolist()
    tied_home_winners = games.went_extra & games.home_win.eq(1)
    assert tied_home_winners.any()
    assert winner_target(games.loc[tied_home_winners]).eq(1).all()


def test_hockey_mass_outcome_scopes_pushes_and_explanation_reconcile():
    frame, model = trained()
    row = frame.iloc[-1]
    means = np.array([m.predict(frame.iloc[-1:][NHL_COLUMNS])[0] for m in model["score_models"]])
    full, reg, outcomes = nhl_distribution(model, means, row)
    assert full.sum() == pytest.approx(1) and reg.sum() == pytest.approx(1)
    assert np.trace(full) == 0 and np.trace(reg) > 0
    assert sum(
        outcomes[k] for k in ["regulation_home_win", "regulation_draw", "regulation_away_win"]
    ) == pytest.approx(1)
    assert outcomes["score_model_full_home_win"] + outcomes[
        "score_model_full_away_win"
    ] == pytest.approx(1)
    assert line_probabilities(reg, 0, 6)["home_cover"]["push"] == pytest.approx(np.trace(reg))
    assert line_probabilities(full, 0, 6)["home_cover"]["push"] == 0
    # An extra-time result adds exactly one official deciding goal to a tied regulation game.
    _, extra = nhl_score_builder(model, means, row)
    expected_reg_total = np.sum(reg * np.indices(reg.shape).sum(axis=0))
    expected_full_total = np.sum(full * np.indices(full.shape).sum(axis=0))
    assert expected_full_total - expected_reg_total == pytest.approx(np.trace(reg))
    assert extra["outcome_rules"]["regulation"] == "60 minutes, ties possible"
    p, e = winner_explanation(model, row)
    assert p == pytest.approx(
        expit(e["intercept"] + sum(c["log_odds_contribution"] for c in e["contributions"]))
    )
    story = league_verdict(
        {
            "home_name": "Boston",
            "away_name": "New York",
            "home_win_probability": p,
            "explanation": e,
        }
    )
    assert story["support"] and story["risks"]


def test_release_rejects_published_checksum_mismatch(monkeypatch, tmp_path):
    from predictions import nhl_data

    index = tmp_path / "index.json"
    index.write_text(
        json.dumps(
            {
                "assets": [
                    {
                        "name": "shots",
                        "digest": "sha256:" + "a" * 64,
                        "browser_download_url": "https://example.org/shots",
                    }
                ]
            }
        )
    )
    raw = tmp_path / "shots"
    raw.write_bytes(b"bad")

    def fake(url, *args, **kwargs):
        return (
            (index, {"sha256": "catalog"}) if "api.github" in url else (raw, {"sha256": "b" * 64})
        )

    monkeypatch.setattr(nhl_data, "snapshot", fake)
    with pytest.raises(ValueError, match="checksum mismatch"):
        release_frame("tag", "shots", tmp_path)
    assert not any("goal" in k or "xg" in k or "score" in k for k in FEATURES)


def test_league_rejects_wrong_feature_cutoff_and_future_source(tmp_path):
    games, goalies = hockey_games()
    now = pd.Timestamp.now(tz="UTC")
    future = games.iloc[-1:].copy()
    future["game_id"] = "future"
    future["season"] = 2024
    future["kickoff"] = now + pd.Timedelta(days=3)
    future["available_at"] = future.kickoff + pd.Timedelta(hours=24)
    future[["home_score", "away_score"]] = np.nan
    features = nhl_features(pd.concat([games, future], ignore_index=True), goalies, as_of=now)
    report = {
        "sport": "NHL",
        "columns": NHL_COLUMNS,
        "model_version": "nhl-v1",
        "holdout_season": 2023,
        "folds": [1],
    }
    with pytest.raises(ValueError, match="exact as_of"):
        issue_league(features, report, tmp_path, as_of=now - pd.Timedelta(seconds=1), source={})
    with pytest.raises(ValueError, match="source snapshot observed after"):
        issue_league(
            features,
            report,
            tmp_path,
            as_of=now,
            source={"nested": [{"fetched_at": (now + pd.Timedelta(seconds=1)).isoformat()}]},
        )
    batch = issue_league(
        features,
        report,
        tmp_path,
        as_of=now,
        source={"synthetic": True},
        fitter=fit_nhl,
        score_builder=nhl_score_builder,
    )
    assert len(batch["forecasts"]) == 1
    forecast = batch["forecasts"][0]
    assert forecast["regulation_distribution"]["sha256"]
    assert forecast["goalie_information"]["home"]["id"] == "home"
    assert "unknown" in forecast["availability"]


def test_authenticated_hockey_ui_regulation_scope_and_quote_independence(tmp_path, monkeypatch):
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    from predictions.config import initialize_research, load_config
    from predictions.league_forecast import publish_league

    games, goalies = hockey_games()
    now = pd.Timestamp.now(tz="UTC")
    future = games.iloc[-1:].copy()
    future["game_id"] = "nhl-future"
    future["season"] = 2024
    future["kickoff"] = now + pd.Timedelta(days=2)
    future["available_at"] = future.kickoff + pd.Timedelta(hours=24)
    future[["home_score", "away_score"]] = np.nan
    report = {
        "sport": "NHL",
        "columns": NHL_COLUMNS,
        "model_version": "nhl-v1",
        "holdout_season": 2023,
        "folds": [1],
    }
    batch = issue_league(
        nhl_features(pd.concat([games, future], ignore_index=True), goalies, as_of=now),
        report,
        tmp_path / "models",
        as_of=now,
        source={},
        fitter=fit_nhl,
        score_builder=nhl_score_builder,
    )
    config = tmp_path / "research.json"
    initialize_research(config)
    c = load_config(config)
    publish_league(batch, c["paths"]["state_dir"])
    monkeypatch.setenv("NFL_RESEARCH_CONFIG", str(config))
    app = AppTest.from_string("from predictions.dashboard import main\nmain()").run()
    assert not app.metric
    app.text_input[0].set_value(Path(c["dashboard"]["token_file"]).read_text().strip())
    app.button[0].click().run(timeout=20)
    assert not app.exception
    expected = f"{batch['forecasts'][0]['home_win_probability']:.1%}"
    assert any(m.value == expected for m in app.metric)
    scope = next(x for x in app.selectbox if x.label == "Outcome scope")
    scope.select("Regulation 60 minutes").run()
    assert not app.exception
    assert any("Score-model regulation" in m.value for m in app.markdown)
    assert any("not a confirmed starter" in m.value.lower() for m in app.caption)
    before = [m.value for m in app.metric]
    # Even an arbitrary quote supplied as secondary metadata cannot change the independently
    # issued forecast or the selected regulation calculation.
    batch["forecasts"][0]["market_comparison"] = {"ask": 0.01, "status": "pass"}
    (Path(c["paths"]["state_dir"]) / "nhl-predictions.json").write_text(json.dumps(batch))
    app.run()
    assert not app.exception and [m.value for m in app.metric] == before
    assert any("PASS" in i.value for i in app.info)


def test_shot_xg_fit_uses_only_prior_seasons_not_vendor_or_target_goals(tmp_path, monkeypatch):
    import joblib

    from predictions import nhl_data

    monkeypatch.setattr(nhl_data, "release_frame", lambda *a, **k: (None, {}))
    mutated = False

    def year_data(year, *args):
        day = pd.Timestamp(f"{year}-10-01", tz="UTC")
        game = pd.DataFrame(
            [
                {
                    "game_id": year,
                    "kickoff": day,
                    "available_at": day + pd.Timedelta(days=1),
                    "home_team_abbr": "BOS",
                    "away_team_abbr": "NYR",
                    "home_team_name": "Boston",
                    "away_team_name": "New York",
                    "home_score": 3,
                    "away_score": 2,
                    "outcome_type": "REG",
                }
            ]
        )
        shots = []
        for i in range(12):
            goal = int(i % 3 == 0)
            if mutated and year == 2022:
                goal = 1 - goal
            shots.append(
                {
                    **{k: float(i + 1) for k in FEATURES},
                    "game_id": year,
                    "home_abbr": "BOS",
                    "away_abbr": "NYR",
                    "is_home": i % 2 == 0,
                    "goal": goal,
                    "season": year,
                    "event_type": "GOAL" if goal else "SHOT",
                    "opponent_skaters": 4,
                    "skater_advantage": 1,
                    "empty": 0,
                    "event_goalie_id": "keeper",
                    "home_goalie_id": "keeper",
                    "away_goalie_id": "keeper",
                    "event_goalie_name": "Prior keeper",
                    "xg": 0.99,  # A vendor value must have no effect on our fitted model.
                }
            )
        return game, pd.DataFrame(shots), {}

    monkeypatch.setattr(nhl_data, "get_year", year_data)
    _, meta = nhl_data.prepare(tmp_path / "first", 2019, 2022)
    assert meta["vendor_xg_used"] is False
    for season, info in meta["causal_xg"].items():
        assert max(info["training_seasons"]) < int(season)
    assert meta["causal_xg"]["2022"]["training_seasons"] == [2019, 2020, 2021]
    first = joblib.load(tmp_path / "first" / "causal-xg-2022.joblib")
    mutated = True
    nhl_data.prepare(tmp_path / "changed", 2019, 2022)
    second = joblib.load(tmp_path / "changed" / "causal-xg-2022.joblib")
    inputs = year_data(2022)[1][FEATURES]
    np.testing.assert_allclose(first.predict_proba(inputs), second.predict_proba(inputs))
