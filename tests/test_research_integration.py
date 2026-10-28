"""Actual scheduled modules on synthetic data; only public providers are mocked."""

import hashlib
import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests

from predictions.config import validate_config
from predictions.inference import predict_pregame, prepare_pregame_features, train_live_model
from predictions.ledger import PaperLedger
from predictions.markets import API_BASE, PolymarketUSProvider
from predictions.performance import PerformanceTracker
from predictions.research import _PublicServices, run_research_cycle


@pytest.fixture
def system(tmp_path, monkeypatch):
    clock = [pd.Timestamp("2026-10-28T12:00:00Z")]
    kickoff = clock[0] + pd.Timedelta(days=2)
    target = "synthetic-research-target"
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
                        "home_score": 27 if (week + pair) % 3 else 10,
                        "away_score": 17,
                    }
                )
    rows.extend(
        [
            {
                "game_id": "synthetic-2026-opener",
                "season": 2026,
                "week": 1,
                "kickoff": pd.Timestamp("2026-09-01T17:00:00Z"),
                "home_team": "BUF",
                "away_team": "NE",
                "home_score": 24,
                "away_score": 17,
            },
            {
                "game_id": target,
                "season": 2026,
                "week": 8,
                "kickoff": kickoff,
                "home_team": "BUF",
                "away_team": "NE",
                "home_score": np.nan,
                "away_score": np.nan,
            },
        ]
    )
    games = pd.DataFrame(rows)
    games["game_type"] = "REG"
    games["neutral"] = 0
    games["kickoff_time_imputed"] = False
    games["result_available_at"] = games["kickoff"] + pd.Timedelta(hours=24)
    config = validate_config(
        {
            "interval_seconds": 60,
            "model_version": "v1",
            "bootstrap_samples": 20,
            "paper": {
                "quantity": 2,
                "assumptions": {
                    "tie_probability_range": ["0", "0.02"],
                    "completion_probability_range": ["1", "1"],
                },
            },
        },
        base_dir=tmp_path,
    )
    monkeypatch.setattr("predictions.inference._now", lambda: clock[0])
    monkeypatch.setattr("predictions.ledger._utcnow", lambda: clock[0].to_pydatetime())

    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock[0].to_pydatetime().astimezone(tz or timezone.utc)

    monkeypatch.setattr("predictions.markets.datetime", FrozenDateTime)
    slug = "synthetic-nfl-buf-ne-2026-10-30"
    market = {
        "id": "synthetic-research-market",
        "slug": slug,
        "active": True,
        "closed": False,
        "status": "MARKET_STATUS_OPEN",
        "sportsMarketType": "football_team_full_game_winner",
        "sportsMarketTypeV2": "SPORTS_MARKET_TYPE_MONEYLINE",
        "gameStartTime": kickoff.isoformat(),
        "description": (
            "Overtime is included. If the game ends in a tie, settle to $0.50. "
            "Cancellations settle at last fair market price."
        ),
        "feeCoefficient": "0.0695",
        "minimumTradeQty": "1",
        "orderPriceMinTickSize": "0.005",
        "marketSides": [
            {"long": True, "tradable": True, "team": {"league": "nfl", "abbreviation": "BUF"}},
            {"long": False, "tradable": True, "team": {"league": "nfl", "abbreviation": "NE"}},
        ],
    }
    state = {"preferred": "yes", "calls": [], "source": None}

    def read_public(url, **kwargs):
        assert url.startswith(API_BASE + "/")
        assert not kwargs.get("headers")
        state["calls"].append(url)
        if url.endswith(f"/v1/market/slug/{slug}"):
            payload = {"market": market}
        elif url.endswith(f"/v1/markets/{slug}/book"):
            yes = state["preferred"] == "yes"
            payload = {
                "marketData": {
                    "marketSlug": slug,
                    "state": "MARKET_STATE_OPEN",
                    "transactTime": (clock[0] - pd.Timedelta(seconds=1)).isoformat(),
                    "offers": [
                        {
                            "px": {"value": "0.01" if yes else "0.995", "currency": "USD"},
                            "qty": "20",
                        }
                    ],
                    "bids": [
                        {
                            "px": {"value": "0.005" if yes else "0.99", "currency": "USD"},
                            "qty": "20",
                        }
                    ],
                }
            }
        elif url.endswith(f"/v1/markets/{slug}/settlement"):
            payload = {"slug": slug, "settlement": "1"}
        else:
            raise AssertionError(f"Unexpected public read: {url}")
        response = requests.Response()
        response.status_code = 200
        response.url = url
        response._content = json.dumps(payload).encode()
        response._content_consumed = True
        return response

    def no_execution(*args, **kwargs):
        raise AssertionError("Scheduled paper research attempted a network write")

    monkeypatch.setattr(requests, "get", read_public)
    for method in ["post", "put", "patch", "delete"]:
        monkeypatch.setattr(requests, method, no_execution)
    provider = PolymarketUSProvider(Path(config["paths"]["market_cache"]))
    tracker = PerformanceTracker(Path(config["paths"]["performance_db"]))

    def fetch_schedule():
        source = {
            "source_url": "https://example.invalid/synthetic-schedule.csv",
            "sha256": hashlib.sha256(games.to_json(date_format="iso").encode()).hexdigest(),
            "fetched_at": clock[0].isoformat(),
        }
        state["source"] = source
        return games.copy(), source

    def forecast(game_ids, policy):
        features = prepare_pregame_features(games, as_of=clock[0], model_version="v1")
        metadata = {"schedule": state["source"]}
        bundle = train_live_model(
            features,
            Path(policy["paths"]["model_dir"]),
            model_version="v1",
            as_of=clock[0],
            season=2026,
            source_metadata=metadata,
            bootstrap_samples=policy["bootstrap_samples"],
            confidence_level=policy["confidence_level"],
            bootstrap_seed=policy["bootstrap_seed"],
        )
        batch = predict_pregame(
            Path(bundle["bundle_path"]),
            features.loc[features["game_id"].isin(game_ids)],
            as_of=clock[0],
            source_metadata=metadata,
        )
        row = batch["forecasts"][0]
        state["preferred"] = "yes" if row["home_win_probability"] >= 0.5 else "no"
        return {**batch, "bundle_path": bundle["bundle_path"]}

    services = {
        "clock": lambda: clock[0],
        "fetch_schedule": fetch_schedule,
        "forecast": forecast,
        "list_markets": lambda: [market.copy()],
        "quote": provider.get_quote,
        "settlement": provider.get_settlement,
        "performance": lambda frame, source, as_of: tracker.update(
            Path(config["paths"]["ledger"]), frame, as_of=as_of, source_metadata=source
        ),
    }
    return {
        "clock": clock,
        "kickoff": kickoff,
        "target": target,
        "games": games,
        "market": market,
        "state": state,
        "services": services,
        "config": config,
    }


@pytest.mark.parametrize("position_limit", [1, 2])
def test_scheduled_uncertainty_to_paper_settlement_and_prospective_scores(system, position_limit):
    raw = {key: value for key, value in system["config"].items() if key != "config_hash"}
    raw["paper"] = {**raw["paper"], "max_open_positions_per_game": position_limit}
    config = validate_config(raw, base_dir=Path.cwd())
    services = system["services"]
    first = run_research_cycle(config, services=services)
    assert first["status"] == "completed", first["errors"]
    assert first["ledger_audit"]["valid"]
    assert first["forecasts"][0]["probability_interval"]["bootstrap_samples"] == 20
    assert first["forecasts"][0]["probability_interval"]["training_cutoff_at"] < first["as_of"]
    filled = [row for row in first["paper_fills"] if row["status"] == "filled"]
    assert len(filled) == 1
    assert Decimal(filled[0]["uncertainty"]["ev_per_contract_lower"]) > Decimal("0.02")
    group = first["performance"]["models"]["logistic-regression-v1"]
    assert group["n"] == 0 and group["metrics"] is None and group["pending"] == 1
    first_id = first["forecasts"][0]["forecast_id"]
    first_file = Path(config["paths"]["state_dir"]) / "runs" / first["run_id"] / "run.json"

    calls = len(system["state"]["calls"])
    duplicate = run_research_cycle(config, services=services)
    assert duplicate["status"] == "skipped"
    assert duplicate["reason"] == "slot_already_completed"
    assert len(system["state"]["calls"]) == calls

    # A new quote/book timestamp in a later slot must not open a second
    # position for the same game or inflate the prospective score cohort.
    system["clock"][0] += pd.Timedelta(minutes=2)
    second = run_research_cycle(config, services=services)
    assert second["status"] == "completed", second["errors"]
    assert second["paper_summary"]["counts"]["paper_trades"] == 1
    expected_skip = (
        "existing_open_position" if position_limit == 1 else "existing_scheduled_position"
    )
    assert any(row.get("reason") == expected_skip for row in second["paper_fills"])
    assert second["performance"]["models"]["logistic-regression-v1"]["selected_forecast_ids"] == [
        first_id
    ]

    system["clock"][0] = system["kickoff"] + pd.Timedelta(days=2)
    system["games"].loc[
        system["games"]["game_id"].eq(system["target"]), ["home_score", "away_score"]
    ] = [24, 10]
    system["market"]["status"] = "MARKET_STATUS_RESOLVED"
    system["market"]["active"], system["market"]["closed"] = False, True
    final = run_research_cycle(config, services=services)
    assert final["status"] == "completed", final["errors"]
    assert len([row for row in final["settlements"] if row["status"] == "settled"]) == 1
    assert final["paper_summary"]["counts"]["paper_trades"] == 1
    assert final["paper_summary"]["counts"]["paper_settlements"] == 1
    assert final["paper_summary"]["open_positions"] == 0
    model = final["performance"]["models"]["logistic-regression-v1"]
    assert model["n"] == 1 and model["pending"] == 0
    assert model["selected_forecast_ids"] == [first_id]
    probability = first["forecasts"][0]["home_win_probability"]
    assert model["metrics"]["brier_score"] == pytest.approx((probability - 1) ** 2)
    assert final["ledger_audit"]["valid"]
    assert PaperLedger(Path(config["paths"]["ledger"])).audit()["valid"]
    # Later labels must not rewrite the original empty-outcome snapshot.
    assert (
        json.loads(first_file.read_text())["performance"]["models"]["logistic-regression-v1"][
            "metrics"
        ]
        is None
    )


def test_missing_assumption_ranges_cannot_automatically_paper_fill(system):
    config = system["config"]
    config["paper"]["assumptions"] = {
        "tie_probability_range": None,
        "completion_probability_range": None,
        "cancellation_settlement_price_range": None,
    }
    report = run_research_cycle(config, services=system["services"])
    assert report["status"] == "completed", report["errors"]
    assert report["evaluations"]
    assert all(item["status"] == "pass" for item in report["evaluations"])
    assert all(
        "tie_probability_range_required" in item["reasons"] for item in report["evaluations"]
    )
    assert report["paper_summary"]["counts"]["paper_trades"] == 0
    assert report["performance"]["models"]["logistic-regression-v1"]["metrics"] is None


def test_transactional_position_limit_race_is_an_ordinary_policy_skip(system, monkeypatch):
    def cap_reached(*args, **kwargs):
        # Another ledger writer can consume the cap after the scheduler's
        # preliminary read. The transactional guard remains authoritative.
        raise ValueError("Paper game position limit reached")

    monkeypatch.setattr(PaperLedger, "paper_fill", cap_reached)
    report = run_research_cycle(system["config"], services=system["services"])
    assert report["status"] == "completed", report["errors"]
    assert report["errors"] == []
    assert any(item["status"] == "paper_candidate" for item in report["evaluations"])
    assert report["paper_fills"] == [
        {
            "game_id": system["target"],
            "status": "not_filled",
            "reason": "Paper game position limit reached",
        }
    ]
    assert report["paper_summary"]["counts"]["paper_trades"] == 0


def test_discovery_continues_after_page_with_no_winner_markets(tmp_path, monkeypatch):
    winner = {"id": "synthetic-later-winner", "sportsMarketType": "football_team_full_game_winner"}
    pages = {
        0: [{"markets": [{"sportsMarketType": "futures"}]}],
        100: [{"markets": [winner]}],
        200: [],
    }
    offsets = []

    def get(url, **kwargs):
        assert url == API_BASE + "/v2/leagues/nfl/events"
        assert kwargs["params"]["limit"] == 100
        offset = kwargs["params"]["offset"]
        offsets.append(offset)
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps({"events": pages[offset]}).encode()
        response._content_consumed = True
        response.url = url
        return response

    monkeypatch.setattr("predictions.markets.requests.get", get)
    services = _PublicServices(validate_config({}, base_dir=tmp_path))
    assert services.list_markets() == [winner]
    assert offsets == [0, 100, 200]


def test_discovery_has_bounded_pagination_even_without_winner_markets(tmp_path, monkeypatch):
    services = _PublicServices(validate_config({}, base_dir=tmp_path))
    offsets = []

    def page(*, limit, offset):
        assert limit == 100
        offsets.append(offset)
        return [], 100

    monkeypatch.setattr(services.provider, "list_nfl_market_page", page)
    with pytest.raises(ValueError, match="pagination bound"):
        services.list_markets()
    assert offsets == list(range(0, 1000, 100))
