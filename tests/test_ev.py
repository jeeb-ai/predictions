"""Financial examples and rejection cases use explicitly synthetic observations."""

import copy
import hashlib
from decimal import Decimal

import pandas as pd
import pytest

from predictions.ev import evaluate_expected_value

NOW = pd.Timestamp("2026-10-27T06:00:00Z")


@pytest.fixture
def observations():
    forecast = {
        "forecast_id": "synthetic-forecast",
        "game_id": "2026_08_CAR_GB",
        "home_team": "GB",
        "away_team": "CAR",
        "kickoff": "2026-10-30T00:15:00+00:00",
        "as_of": "2026-10-27T05:59:30+00:00",
        "feature_cutoff_at": "2026-10-27T05:59:30+00:00",
        "home_win_probability": "0.7",
        "away_win_probability": "0.3",
        "probability_scope": "conditional_on_decisive_result",
        "model_id": "synthetic-model",
        "model_version": "logistic-regression-v2",
        "model_sha256": "a" * 64,
        "features_sha256": "b" * 64,
    }
    description = "Overtime is included. Ties pay $0.50. Cancellations use last fair market price."
    quote = {
        "quote_id": "synthetic-quote",
        "liquidity_key": "synthetic-book",
        "venue": "polymarket_us",
        "market_id": "synthetic-market",
        "market_slug": "aec-nfl-car-gb-2026-10-29",
        "yes_team": "CAR",
        "no_team": "GB",
        "outcome": "yes",
        "market_status": "open",
        "game_start_time": forecast["kickoff"],
        "quote_at": "2026-10-27T05:59:58+00:00",
        "received_at": "2026-10-27T05:59:59+00:00",
        "ask_levels": [{"price": "0.2", "quantity": "100000"}],
        "minimum_trade_quantity": "1",
        "price_tick": "0.0025",
        "source": {"book": {"sha256": "c" * 64}},
        "settlement": {
            "rule_type": "nfl_winner_including_overtime",
            "win_payout": "1",
            "loss_payout": "0",
            "tie_payout": "0.5",
            "cancellation": "last_fair_market_price",
            "rules_url": "https://docs.polymarket.us/faqs/sports-faqs.md",
            "description": description,
            "terms_sha256": hashlib.sha256(description.encode()).hexdigest(),
        },
        "fee_schedule": {
            "type": "polymarket_us_standard_taker",
            "theta": "0.0695",
            "effective_at": "2026-10-19T17:00:00Z",
            "rounding": "ROUND_HALF_EVEN",
            "currency": "USD",
            "source_url": "https://docs.polymarket.us/fees.md",
        },
    }
    return forecast, quote


def evaluate(observations, **kwargs):
    options = {
        "selected_team": "CAR",
        "quantity": 10,
        "as_of": NOW,
        "tie_probability": "0.02",
        "completion_probability": "1",
    }
    options.update(kwargs)
    return evaluate_expected_value(*observations, **options)


def test_exact_fee_and_conditional_decisive_to_tie_adjusted_ev(observations):
    result = evaluate(observations)
    assert result["status"] == "paper_candidate"
    assert Decimal(result["fee"]) == Decimal("0.11")
    assert Decimal(result["total_cost"]) == Decimal("2.11")
    assert Decimal(result["expected_payout"]) == Decimal("3.04")
    assert Decimal(result["expected_profit"]) == Decimal("0.93")
    assert Decimal(result["ev_per_contract"]) == Decimal("0.093")
    assert result["fee_kind"] == "estimated_upper_bound"
    assert result["match_evidence"]["yes_team"] == "CAR"  # YES is the away team.


def test_no_outcome_uses_home_probability_not_yes_probability(observations):
    observations[1]["outcome"] = "no"
    observations[1]["ask_levels"] = [{"price": "0.6", "quantity": "10"}]
    result = evaluate(observations, selected_team="GB")
    assert result["status"] == "paper_candidate"
    assert Decimal(result["expected_payout"]) == Decimal("6.96")
    assert Decimal(result["fee"]) == Decimal("0.17")
    assert Decimal(result["expected_profit"]) == Decimal("0.79")


def test_cancellation_uses_explicit_lfmp_assumption_not_refund(observations):
    result = evaluate(
        observations, completion_probability="0.8", cancellation_settlement_price="0.4"
    )
    assert Decimal(result["expected_payout"]) == Decimal("3.232")
    assert Decimal(result["expected_profit"]) == Decimal("1.122")


@pytest.mark.parametrize("missing", ["tie_probability", "completion_probability"])
def test_missing_probability_assumptions_are_passes(observations, missing):
    result = evaluate(observations, **{missing: None})
    assert result["status"] == "pass"
    assert f"{missing}_required" in result["reasons"]
    assert result["expected_profit"] is None


def test_cancellation_price_required_when_completion_not_certain(observations):
    result = evaluate(observations, completion_probability="0.9")
    assert result["status"] == "pass"
    assert "cancellation_settlement_price_required" in result["reasons"]


def test_multilevel_depth_and_conservative_cumulative_fee_cap(observations):
    observations[1]["ask_levels"] = [
        {"price": "0.3", "quantity": "1"},
        {"price": "0.2", "quantity": "1"},
    ]
    result = evaluate(observations, quantity=2)
    assert [fill["price"] for fill in result["fills"]] == ["0.2", "0.3"]
    assert Decimal(result["notional"]) == Decimal("0.5")
    assert Decimal(result["fee_upper_bound"]) == Decimal("0.03")
    assert Decimal(result["fee_estimate_per_displayed_level"]) == Decimal("0.02")
    assert Decimal(result["total_cost"]) == Decimal("0.53")


def test_fractional_displayed_quantities_are_consumed_without_rounding_depth(observations):
    observations[1]["ask_levels"] = [
        {"price": "0.2", "quantity": "0.75"},
        {"price": "0.25", "quantity": "1.25"},
    ]
    result = evaluate(observations, quantity=2)
    assert [fill["quantity"] for fill in result["fills"]] == ["0.75", "1.25"]
    assert Decimal(result["notional"]) == Decimal("0.4625")


@pytest.mark.parametrize("price,fee", [("0.10", "6.26"), ("0.50", "17.38")])
def test_documented_us_standard_fee_examples(observations, price, fee):
    observations[1]["ask_levels"] = [{"price": price, "quantity": "1000"}]
    result = evaluate(observations, quantity=1000)
    assert Decimal(result["fee"]) == Decimal(fee)


def test_insufficient_depth_never_candidates_partial_fill(observations):
    observations[1]["ask_levels"] = [{"price": "0.2", "quantity": "9.99"}]
    result = evaluate(observations)
    assert result["status"] == "pass"
    assert "insufficient_depth" in result["reasons"]
    assert result["expected_profit"] is None


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("venue", "polymarket_international", "unsupported_venue"),
        ("yes_team", "BUF", "market_team_mismatch"),
        ("game_id", "different_game", "market_game_mismatch"),
        ("game_start_time", "2026-10-31T00:15:00Z", "market_kickoff_mismatch"),
        ("market_status", "closed", "market_not_open"),
        ("quote_at", "2026-10-27T05:58:00Z", "quote_stale"),
        ("received_at", "2026-10-27T06:00:01Z", "quote_from_future"),
    ],
)
def test_quote_identity_and_time_disqualifiers(observations, field, value, reason):
    observations[1][field] = value
    result = evaluate(observations)
    assert result["status"] == "pass"
    assert reason in result["reasons"]


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("as_of", "2026-10-27T04:00:00Z", "forecast_stale"),
        ("as_of", "2026-10-27T06:00:01Z", "forecast_from_future"),
        (
            "advanced_history_available_through",
            "2026-10-27T05:59:30Z",
            "forecast_history_not_strictly_before_cutoff",
        ),
        ("probability_scope", "unconditional", "unsupported_probability_scope"),
    ],
)
def test_forecast_time_scope_disqualifiers(observations, field, value, reason):
    observations[0][field] = value
    result = evaluate(observations)
    assert result["status"] == "pass"
    assert reason in result["reasons"]


def test_started_game_is_rejected_even_with_a_fresh_quote(observations):
    result = evaluate(observations, as_of="2026-10-30T00:15:00Z")
    assert "game_started_or_nonpregame_forecast" in result["reasons"]


@pytest.mark.parametrize("field", ["fee_schedule", "settlement"])
def test_unknown_fees_or_terms_pass(observations, field):
    observations[1][field] = {}
    result = evaluate(observations)
    assert result["status"] == "pass"
    assert result["expected_profit"] is None


@pytest.mark.parametrize("invalid", ["NaN", "Infinity", "-0.1", "1.1", True])
def test_invalid_probabilities_raise(observations, invalid):
    with pytest.raises(ValueError):
        evaluate(observations, tie_probability=invalid)


def test_ev_threshold_and_fee_deduction_can_force_a_pass(observations):
    result = evaluate(observations, min_ev_per_contract="0.1")
    assert result["status"] == "pass"
    assert "not_above_minimum_ev_after_fees" in result["reasons"]


def test_evaluations_are_deterministic_and_do_not_mutate_inputs(observations):
    before = copy.deepcopy(observations)
    first, second = evaluate(observations), evaluate(observations)
    assert first == second
    assert observations == before


def test_changed_terms_text_cannot_keep_original_terms_hash(observations):
    observations[1]["settlement"]["description"] = "Different settlement agreement"
    result = evaluate(observations)
    assert result["status"] == "pass"
    assert "unsupported_settlement_terms" in result["reasons"]


def test_below_minimum_and_off_tick_prices_cannot_fill(observations):
    observations[1]["minimum_trade_quantity"] = "20"
    assert "below_minimum_quantity" in evaluate(observations)["reasons"]
    observations[1]["minimum_trade_quantity"] = "1"
    observations[1]["ask_levels"][0]["price"] = "0.2001"
    result = evaluate(observations)
    assert result["status"] == "pass"
    assert "price_tick_mismatch" in result["reasons"]
