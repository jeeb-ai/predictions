# ruff: noqa: F811
"""End-to-end trained forecasts -> persisted snapshot -> authenticated Streamlit UI."""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest
from test_inference import AS_OF, TARGET, aggregates, games  # noqa: F401
from test_research_integration import system  # noqa: F401

from predictions import dashboard
from predictions.config import initialize_research
from predictions.inference import (
    _load_bundle,
    predict_pregame,
    prepare_pregame_features,
    train_live_model,
)
from predictions.research import run_research_cycle


def test_trained_model_to_prediction_home_and_matchup_ui(games, tmp_path, monkeypatch):
    teams, qbs = aggregates(games)
    features = prepare_pregame_features(games, as_of=AS_OF, team_games=teams, qb_games=qbs)
    bundle = train_live_model(
        features, tmp_path / "models", as_of=AS_OF, season=2026, bootstrap_samples=20
    )
    row = predict_pregame(
        Path(bundle["bundle_path"]),
        features.loc[features.game_id.eq(TARGET)],
        as_of=AS_OF,
        include_analysis=True,
    )["forecasts"][0]
    model, _ = _load_bundle(Path(bundle["bundle_path"]))
    explanation = row["matchup_analysis"]
    total = explanation["intercept"] + sum(
        r["log_odds_contribution"] for r in explanation["contributions"]
    )
    assert np.isclose(
        total,
        model.decision_function(features.loc[features.game_id.eq(TARGET), model.feature_names_in_])[
            0
        ],
    )
    assert np.isclose(1 / (1 + np.exp(-total)), row["home_win_probability"])
    for value in explanation["contributions"]:
        assert np.isclose(
            value["standardized_value"],
            (value["raw_value"] - value["training_mean"]) / value["training_scale"],
        )
        assert np.isclose(
            value["log_odds_contribution"], value["standardized_value"] * value["coefficient"]
        )
    assert TARGET not in explanation["source_values"]["home_off_source_game_ids"]
    assert explanation["source_values"]["home_off_epa_per_play"] == 0.2
    assert explanation["source_values"]["home_qb_id_proxy"] == "QB_BUF"
    config_path = tmp_path / "research.json"
    init = initialize_research(config_path)
    config = dashboard.load_config(config_path)
    root = Path(config["paths"]["state_dir"])
    snapshot = {
        "run_id": "synthetic-v4-ui",
        "as_of": AS_OF.isoformat(),
        "forecasts": [row],
        "slate": [],
    }
    (root / "predictions.json").write_text(json.dumps(snapshot))
    (root / "latest.json").write_text(
        json.dumps(
            {
                "run_id": "synthetic-market-outage",
                "as_of": AS_OF.isoformat(),
                "status": "failed",
                "forecasts": [],
                "evaluations": [],
            }
        )
    )
    monkeypatch.setenv("NFL_RESEARCH_CONFIG", str(config_path))

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return AS_OF.to_pydatetime()

    monkeypatch.setattr(dashboard, "datetime", Clock)
    app = AppTest.from_string("from predictions.dashboard import main\nmain()").run()
    assert not app.dataframe and not app.metric
    app.text_input[0].set_value(Path(init["token_file"]).read_text().strip())
    app.button[0].click().run()
    assert not app.exception
    assert app.title[0].value == "NFL Predictions"
    assert any(e.label == "Why this pick?" for e in app.expander)
    assert any(e.label == "Advanced Analysis" and not e.proto.expanded for e in app.expander)
    assert any("model’s pick" in m.value for m in app.markdown)
    assert [t.label for t in app.tabs] == ["Predictions", "Matchup Analysis", "Market Value"]
    assert any("Predicted winner:" in item.value for item in app.markdown)
    displayed = {m.label: m.value for m in app.metric}
    assert displayed["Buffalo Bills win probability"] == f"{row['home_win_probability']:.1%}"
    assert displayed["New England Patriots win probability"] == f"{row['away_win_probability']:.1%}"
    table = next(f.value for f in app.dataframe if "log_odds_contribution" in f.value)
    assert np.isclose(table.log_odds_contribution.sum() + explanation["intercept"], total)
    assert any("not a confirmed starter" in c.value for c in app.caption)
    assert any(e.label == "Research / Administration" for e in app.expander)
    # Markets can change arbitrarily; the independent snapshot and UI probabilities persist.
    before = (root / "predictions.json").read_bytes()
    report = json.loads((root / "latest.json").read_text())
    report["evaluations"] = [
        {"game_id": TARGET, "status": "pass", "fills": [{"price": ".99", "quantity": 1}]}
    ]
    (root / "latest.json").write_text(json.dumps(report))
    app.run()
    assert not app.exception
    assert (root / "predictions.json").read_bytes() == before
    assert {m.label: m.value for m in app.metric}["Buffalo Bills win probability"] == displayed[
        "Buffalo Bills win probability"
    ]


def test_worker_publishes_predictions_before_market_outage(system):
    def outage():
        snapshot = json.loads(
            (Path(system["config"]["paths"]["state_dir"]) / "predictions.json").read_text()
        )
        assert snapshot["forecasts"] and snapshot["slate"]
        raise TimeoutError("Synthetic market outage")

    system["services"]["list_markets"] = outage
    result = run_research_cycle(system["config"], services=system["services"])
    assert result["status"] == "partial"
    assert result["forecasts"]
    view = dashboard.build_dashboard_view(system["config"], system["clock"][0])
    assert view["forecasts"] == result["forecasts"]
    assert not any(r.get("status") == "paper_candidate" for r in result["evaluations"])


def test_quote_changes_cannot_change_independent_worker_probabilities(system):
    first = run_research_cycle(system["config"], services=system["services"])

    def stale_quote(*args):
        raise TimeoutError("Synthetic stale/unavailable quote")

    system["services"]["quote"] = stale_quote
    system["clock"][0] += pd.Timedelta(minutes=1)
    second = run_research_cycle(system["config"], services=system["services"])
    for key in ["features", "home_win_probability", "away_win_probability"]:
        assert first["forecasts"][0][key] == second["forecasts"][0][key]


def test_failed_refresh_retains_valid_pregame_prediction_and_slate(system):
    first = run_research_cycle(system["config"], services=system["services"])

    def failed_forecast(*args):
        raise ValueError("Synthetic missing data")

    system["services"]["forecast"] = failed_forecast
    system["clock"][0] += pd.Timedelta(minutes=1)
    result = run_research_cycle(system["config"], services=system["services"])
    assert result["errors"]
    view = dashboard.build_dashboard_view(system["config"], system["clock"][0])
    assert view["forecasts"] == first["forecasts"]
    assert view["slate"][0]["game_id"] == system["target"]


def test_changed_schedule_does_not_reuse_incompatible_forecast(system):
    run_research_cycle(system["config"], services=system["services"])
    system["games"].loc[system["games"].game_id.eq(system["target"]), "kickoff"] += pd.Timedelta(
        hours=1
    )

    def failed_forecast(*args):
        raise ValueError("Synthetic missing data")

    system["services"]["forecast"] = failed_forecast
    system["clock"][0] += pd.Timedelta(minutes=1)
    run_research_cycle(system["config"], services=system["services"])
    view = dashboard.build_dashboard_view(system["config"], system["clock"][0])
    assert not view["forecasts"]
    assert view["slate"]
