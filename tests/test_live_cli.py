"""Offline live-to-paper workflows on synthetic games and mocked public GETs."""

import copy
import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest
import requests

from predictions.cli import main
from predictions.markets import API_BASE


@pytest.fixture
def workflow(tmp_path, monkeypatch):
    clock = [pd.Timestamp("2026-10-27T12:00:00Z")]
    kickoff = clock[0] + pd.Timedelta(days=2)
    ledger = tmp_path / "paper.sqlite"
    schedule = tmp_path / "synthetic-games.csv"
    market_slug = "synthetic-nfl-buf-ne-2026-10-29"
    rows = []
    for season in [2024, 2025]:
        for week in range(1, 7):
            played = pd.Timestamp(f"{season}-09-01T17:00:00Z") + pd.Timedelta(weeks=week - 1)
            eastern = played.tz_convert("America/New_York")
            rows.append(
                {
                    "game_id": f"synthetic-{season}-{week}",
                    "season": season,
                    "week": week,
                    "gameday": eastern.strftime("%Y-%m-%d"),
                    "gametime": eastern.strftime("%H:%M"),
                    "home_score": 27 if week % 3 else 10,
                    "away_score": 17,
                }
            )
    for identity, played, home, away in [
        ("synthetic-2026-opener", pd.Timestamp("2026-09-01T17:00:00Z"), 24, 17),
        ("synthetic-future-source", clock[0] + pd.Timedelta(days=1), 70, 0),
        ("synthetic-target", kickoff, None, None),
    ]:
        eastern = played.tz_convert("America/New_York")
        rows.append(
            {
                "game_id": identity,
                "season": 2026,
                "week": 1 if "opener" in identity else 8,
                "gameday": eastern.strftime("%Y-%m-%d"),
                "gametime": eastern.strftime("%H:%M"),
                "home_score": home,
                "away_score": away,
            }
        )
    games = pd.DataFrame(rows)
    games["game_type"] = "REG"
    games["home_team"] = "BUF"
    games["away_team"] = "NE"
    games["location"] = "Home"
    games.to_csv(schedule, index=False)

    monkeypatch.setattr("predictions.live_cli._now", lambda: clock[0])
    monkeypatch.setattr("predictions.inference._now", lambda: clock[0])
    monkeypatch.setattr("predictions.ledger._utcnow", lambda: clock[0].to_pydatetime())

    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            value = clock[0].to_pydatetime()
            return value.astimezone(tz or timezone.utc)

    monkeypatch.setattr("predictions.markets.datetime", FrozenDateTime)
    market = {
        "id": "synthetic-market-1",
        "slug": market_slug,
        "active": True,
        "closed": False,
        "status": "MARKET_STATUS_OPEN",
        "sportsMarketType": "football_team_full_game_winner",
        "sportsMarketTypeV2": "SPORTS_MARKET_TYPE_MONEYLINE",
        "description": (
            "The market settles to the winner. Overtime is included if played. "
            "If the game ends in a tie, the market will settle to $0.50. "
            "A canceled game settles to the last fair market price."
        ),
        "gameStartTime": kickoff.isoformat(),
        "feeCoefficient": 0.0695,
        "minimumTradeQty": 0.01,
        "orderPriceMinTickSize": 0.0025,
        "marketSides": [
            {"long": True, "tradable": True, "team": {"league": "nfl", "abbreviation": "buf"}},
            {"long": False, "tradable": True, "team": {"league": "nfl", "abbreviation": "ne"}},
        ],
    }
    book = {
        "marketSlug": market_slug,
        "state": "MARKET_STATE_OPEN",
        "transactTime": (clock[0] - pd.Timedelta(seconds=1)).isoformat(),
        "offers": [{"px": {"value": "0.20", "currency": "USD"}, "qty": "10.00"}],
        "bids": [{"px": {"value": "0.10", "currency": "USD"}, "qty": "10.00"}],
    }
    gets = []

    def public_get(url, **kwargs):
        assert url.startswith(API_BASE + "/"), "Only the documented public US gateway is allowed"
        assert not kwargs.get("headers"), "Public reads must not carry authentication"
        gets.append(url)
        if url.endswith(f"/v1/market/slug/{market_slug}"):
            payload = {"market": market}
        elif url.endswith(f"/v1/markets/{market_slug}/book"):
            payload = {"marketData": book}
        elif url.endswith(f"/v1/markets/{market_slug}/settlement"):
            payload = {"slug": market_slug, "settlement": "0.50"}
        elif url.endswith("/v2/leagues/nfl/events"):
            payload = {"events": [{"markets": [market]}]}
        else:
            raise AssertionError(f"Unexpected read-only endpoint: {url}")
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps(payload).encode()
        response._content_consumed = True
        response.url = url
        return response

    def never_trade(*args, **kwargs):
        raise AssertionError("The paper workflow attempted a write request")

    monkeypatch.setattr("requests.get", public_get)
    for method in ["post", "put", "patch", "delete"]:
        monkeypatch.setattr(requests, method, never_trade)
    return {
        "clock": clock,
        "kickoff": kickoff,
        "schedule": schedule,
        "ledger": ledger,
        "market": market,
        "book": book,
        "slug": market_slug,
        "gets": gets,
        "root": tmp_path,
    }


def invoke(args, capsys):
    code = main([str(value) for value in args])
    captured = capsys.readouterr()
    return code, json.loads(captured.out) if captured.out else None, captured.err


def forecast(workflow, capsys):
    root = workflow["root"]
    code, batch, error = invoke(
        [
            "forecast",
            "--input",
            workflow["schedule"],
            "--game-id",
            "synthetic-target",
            "--model-version",
            "v1",
            "--offline",
            "--as-of",
            workflow["clock"][0].isoformat(),
            "--model-dir",
            root / "models",
            "--output-dir",
            root / "forecasts",
            "--ledger",
            workflow["ledger"],
        ],
        capsys,
    )
    assert code == 0, error
    return batch, batch["forecasts"][0]


def quote(workflow, forecast_record, capsys):
    # Make the better-probability side cheap without altering model outputs.
    if forecast_record["home_win_probability"] >= 0.5:
        outcome, team = "yes", "BUF"
    else:
        outcome, team = "no", "NE"
        workflow["book"]["offers"][0]["px"]["value"] = "0.90"
        workflow["book"]["bids"][0]["px"]["value"] = "0.80"
    path = workflow["root"] / "quote.json"
    code, result, error = invoke(
        [
            "quote",
            "--market",
            workflow["slug"],
            "--outcome",
            outcome,
            "--cache-dir",
            workflow["root"] / "market-cache",
            "--output",
            path,
            "--ledger",
            workflow["ledger"],
        ],
        capsys,
    )
    assert code == 0, error
    return path, team, result


def evaluate(workflow, batch, quote_path, team, capsys, *, assumptions=True):
    arguments = [
        "ev",
        "--forecast",
        Path(batch["artifact"]["path"]),
        "--quote",
        quote_path,
        "--team",
        team,
        "--quantity",
        "5",
        "--ledger",
        workflow["ledger"],
    ]
    if assumptions:
        arguments += ["--tie-probability", "0.01", "--completion-probability", "1"]
    return invoke(arguments, capsys)


def test_offline_forecast_quote_ev_fill_settlement_and_audit(workflow, capsys):
    batch, prediction = forecast(workflow, capsys)
    quote_path, team, observed_quote = quote(workflow, prediction, capsys)
    code, evaluation, error = evaluate(workflow, batch, quote_path, team, capsys)
    assert code == 0, error
    assert evaluation["status"] == "paper_candidate"
    assert Decimal(evaluation["notional"]) == Decimal("1")
    assert Decimal(evaluation["fee_upper_bound"]) == Decimal("0.06")
    assert evaluation["forecast_id"] == prediction["forecast_id"]
    assert evaluation["quote_id"] == observed_quote["quote_id"]
    assert evaluation["fee_kind"] == "estimated_upper_bound"
    fill_args = [
        "paper-fill",
        "--evaluation-id",
        evaluation["evaluation_id"],
        "--idempotency-key",
        "synthetic-paper-fill",
        "--max-exposure",
        "10",
        "--ledger",
        workflow["ledger"],
    ]
    code, trade, error = invoke(fill_args, capsys)
    assert code == 0, error
    assert trade["execution"] == "simulated_displayed_liquidity"
    assert trade["mode"] == "paper_only"
    assert invoke(fill_args, capsys)[1]["paper_trade_id"] == trade["paper_trade_id"]

    # Actual settlement snapshots are read only after the synthetic game starts.
    workflow["clock"][0] = workflow["kickoff"] + pd.Timedelta(hours=3)
    workflow["market"]["status"] = "MARKET_STATUS_RESOLVED"
    evidence_path = workflow["root"] / "settlement.json"
    code, _, error = invoke(
        [
            "market-settlement",
            "--market",
            workflow["slug"],
            "--cache-dir",
            workflow["root"] / "market-cache",
            "--output",
            evidence_path,
        ],
        capsys,
    )
    assert code == 0, error
    settle_args = [
        "paper-settle",
        "--trade-id",
        trade["paper_trade_id"],
        "--price",
        "0.50",
        "--evidence",
        evidence_path,
        "--idempotency-key",
        "synthetic-settlement",
        "--ledger",
        workflow["ledger"],
    ]
    code, settlement, error = invoke(settle_args, capsys)
    assert code == 0, error
    assert Decimal(settlement["payout"]) == Decimal("2.50")
    assert Decimal(settlement["realized_paper_profit"]) == Decimal("1.44")
    assert settlement["evidence_verification"] == "public_us_settlement_snapshot"
    code, audit, error = invoke(["paper-audit", "--ledger", workflow["ledger"]], capsys)
    assert code == 0, error
    assert audit["valid"] and audit["events"] == 5
    assert audit["summary"]["open_positions"] == 0
    assert audit["summary"]["counts"]["paper_trades"] == 1
    assert len(workflow["gets"]) == 4


def test_missing_risk_assumptions_record_pass_and_cannot_fill(workflow, capsys):
    batch, prediction = forecast(workflow, capsys)
    path, team, _ = quote(workflow, prediction, capsys)
    code, evaluation, error = evaluate(workflow, batch, path, team, capsys, assumptions=False)
    assert code == 0, error
    assert evaluation["status"] == "pass"
    assert {"tie_probability_required", "completion_probability_required"} <= set(
        evaluation["reasons"]
    )
    code, _, error = invoke(
        [
            "paper-fill",
            "--evaluation-id",
            evaluation["evaluation_id"],
            "--idempotency-key",
            "must-not-fill",
            "--ledger",
            workflow["ledger"],
        ],
        capsys,
    )
    assert code == 1 and "paper_candidate" in error


def test_stale_public_book_cannot_become_paper_candidate(workflow, capsys):
    batch, prediction = forecast(workflow, capsys)
    workflow["book"]["transactTime"] = (workflow["clock"][0] - pd.Timedelta(minutes=5)).isoformat()
    path, team, _ = quote(workflow, prediction, capsys)
    code, evaluation, error = evaluate(workflow, batch, path, team, capsys)
    assert code == 0, error
    assert evaluation["status"] == "pass" and "quote_stale" in evaluation["reasons"]


def test_future_results_cannot_change_cli_forecast(workflow, capsys):
    _, first = forecast(workflow, capsys)
    games = pd.read_csv(workflow["schedule"])
    future = games["game_id"].eq("synthetic-future-source")
    games.loc[future, ["home_score", "away_score"]] = [0, 99]
    games.to_csv(workflow["schedule"], index=False)
    _, second = forecast(workflow, capsys)
    assert first["features"] == second["features"]
    assert first["home_win_probability"] == second["home_win_probability"]
    assert pd.Timestamp(first["feature_cutoff_at"]) == workflow["clock"][0]


def test_started_game_cannot_be_relabelled_live_in_cli(workflow, capsys):
    workflow["clock"][0] = workflow["kickoff"] + pd.Timedelta(seconds=1)
    code, _, error = invoke(
        [
            "forecast",
            "--input",
            workflow["schedule"],
            "--game-id",
            "synthetic-target",
            "--model-version",
            "v1",
            "--offline",
            "--as-of",
            "2026-10-27T12:00:00Z",
            "--model-dir",
            workflow["root"] / "models",
            "--output-dir",
            workflow["root"] / "out",
            "--ledger",
            workflow["ledger"],
        ],
        capsys,
    )
    assert code == 1 and "No upcoming" in error


def test_offline_model2_fails_missing_pbp_without_network(workflow, capsys):
    code, _, error = invoke(
        [
            "forecast",
            "--input",
            workflow["schedule"],
            "--game-id",
            "synthetic-target",
            "--model-version",
            "v2",
            "--offline",
            "--pbp-dir",
            workflow["root"] / "no-pbp",
            "--model-dir",
            workflow["root"] / "models",
            "--ledger",
            workflow["ledger"],
        ],
        capsys,
    )
    assert code == 1 and "offline" in error.casefold()
    assert workflow["gets"] == []


def test_discovery_uses_public_us_nfl_endpoint_only(workflow, capsys):
    code, result, error = invoke(
        ["us-markets", "--cache-dir", workflow["root"] / "discovery"], capsys
    )
    assert code == 0, error
    assert result["venue"] == "polymarket_us"
    assert result["markets"][0]["slug"] == workflow["slug"]
    assert workflow["gets"] == [API_BASE + "/v2/leagues/nfl/events"]


def test_mismatched_market_cannot_be_evaluated_as_matching_game(workflow, capsys):
    batch, prediction = forecast(workflow, capsys)
    workflow["market"]["gameStartTime"] = (workflow["kickoff"] + pd.Timedelta(days=1)).isoformat()
    path, team, _ = quote(workflow, prediction, capsys)
    code, result, error = evaluate(workflow, batch, path, team, capsys)
    assert code == 0, error
    assert result["status"] == "pass" and "market_kickoff_mismatch" in result["reasons"]


def test_forged_evaluation_is_not_an_execution_command(workflow, capsys):
    batch, prediction = forecast(workflow, capsys)
    path, team, _ = quote(workflow, prediction, capsys)
    code, evaluation, error = evaluate(workflow, batch, path, team, capsys)
    assert code == 0, error
    altered = copy.deepcopy(evaluation)
    altered["total_cost"] = "0.01"
    from predictions.ledger import PaperLedger

    with pytest.raises(ValueError, match="total_cost"):
        PaperLedger(workflow["ledger"]).record_evaluation(altered)
