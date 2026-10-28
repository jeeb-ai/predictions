"""Synthetic uncertainty scenarios test financial bounds and audit compatibility."""

import copy
import hashlib
import json
from decimal import Decimal

import pandas as pd
import pytest

from predictions.ev import evaluate_expected_value, validate_probability_range

NOW = pd.Timestamp("2026-10-28T12:00:00Z")


def _observations():
    description = "Overtime is included. Ties pay $0.50. Cancellations use last fair market price."
    forecast = {
        "forecast_id": "synthetic-forecast",
        "game_id": "2026_08_CAR_GB",
        "home_team": "GB",
        "away_team": "CAR",
        "kickoff": "2026-10-30T00:15:00Z",
        "as_of": "2026-10-28T11:59:30Z",
        "feature_cutoff_at": "2026-10-28T11:59:30Z",
        "home_win_probability": "0.7",
        "away_win_probability": "0.3",
        "probability_scope": "conditional_on_decisive_result",
        "model_id": "synthetic-model",
        "model_version": "logistic-regression-v2",
        "model_sha256": "a" * 64,
        "features_sha256": "b" * 64,
        "model_training_cutoff_at": "2026-09-01T00:00:00Z",
        "model_train_max_result_available_at": "2026-01-06T00:00:00Z",
        "model_training_game_ids_sha256": "d" * 64,
        "probability_interval": {
            "home": {"lower": "0.65", "upper": "0.75"},
            "away": {"lower": "0.25", "upper": "0.35"},
            "confidence_level": 0.95,
            "bootstrap_samples": 200,
            "method": "training_week_block_bootstrap",
            "scope": "model_estimation_conditional_on_features",
            "calibration": "not_a_calibrated_predictive_coverage_guarantee",
            "model_id": "synthetic-model",
            "model_sha256": "a" * 64,
            "ensemble_sha256": "e" * 64,
            "training_game_ids_sha256": "d" * 64,
            "training_cutoff_at": "2026-09-01T00:00:00Z",
            "game_id": "2026_08_CAR_GB",
            "features_sha256": "b" * 64,
            "as_of": "2026-10-28T11:59:30Z",
        },
    }
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
        "quote_at": "2026-10-28T11:59:58Z",
        "received_at": "2026-10-28T11:59:59Z",
        "ask_levels": [{"price": "0.2", "quantity": "1000"}],
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


@pytest.fixture
def observations():
    return _observations()


def evaluate(observations, **kwargs):
    options = {
        "selected_team": "CAR",
        "quantity": 10,
        "as_of": NOW,
        "tie_probability": "0.02",
        "completion_probability": "1",
        "require_uncertainty": True,
        "tie_probability_range": ["0", "0.04"],
        "completion_probability_range": ["1", "1"],
    }
    options.update(kwargs)
    return evaluate_expected_value(*observations, **options)


def test_legacy_evaluation_remains_byte_exact(observations):
    result = evaluate(
        observations,
        require_uncertainty=False,
        tie_probability_range=None,
        completion_probability_range=None,
    )
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    assert (
        hashlib.sha256(encoded).hexdigest()
        == "1fb2c4992ef302773a56cef5a82129c79b7e1174a2d22fc58ed509e8e5e54704"
    )
    assert "uncertainty" not in result and "uncertainty_policy" not in result


def test_closed_form_lower_and_upper_ev_after_fees(observations):
    result = evaluate(observations)
    bounds = result["uncertainty"]
    assert result["status"] == "paper_candidate"
    assert Decimal(result["fee"]) == Decimal("0.11")
    # CAR's probability is below .5: ties help its contract. The minimum is
    # p=.25,tie=0 and maximum p=.35,tie=.04, not a symmetric EV error bar.
    assert Decimal(bounds["expected_payout_lower"]) == Decimal("2.5")
    assert Decimal(bounds["expected_payout_upper"]) == Decimal("3.56")
    assert Decimal(bounds["expected_profit_lower"]) == Decimal("0.39")
    assert Decimal(bounds["ev_per_contract_lower"]) == Decimal("0.039")
    assert Decimal(bounds["ev_per_contract_upper"]) == Decimal("0.145")
    assert bounds["lower_bound_assumptions"]["tie_probability"] == "0"
    assert bounds["calibration"] == "not_a_calibrated_predictive_coverage_guarantee"


def test_positive_point_ev_with_negative_lower_bound_is_pass(observations):
    forecast = observations[0]
    forecast["probability_interval"]["away"] = {"lower": "0.1", "upper": "0.4"}
    forecast["probability_interval"]["home"] = {"lower": "0.6", "upper": "0.9"}
    result = evaluate(observations)
    assert Decimal(result["expected_profit"]) > 0
    assert Decimal(result["uncertainty"]["expected_profit_lower"]) < 0
    assert result["status"] == "pass"
    assert "uncertainty_does_not_clear_minimum_ev" in result["reasons"]


def test_no_contract_uses_home_interval_and_adverse_tie_bound(observations):
    observations[1]["outcome"] = "no"
    observations[1]["ask_levels"] = [{"price": "0.6", "quantity": "10"}]
    result = evaluate(observations, selected_team="GB")
    assert result["status"] == "paper_candidate"
    assert result["uncertainty"]["selected_team_probability"] == {"lower": "0.65", "upper": "0.75"}
    assert result["uncertainty"]["lower_bound_assumptions"]["tie_probability"] == "0.04"
    assert Decimal(result["uncertainty"]["expected_payout_lower"]) == Decimal("6.44")
    assert Decimal(result["uncertainty"]["expected_profit_lower"]) == Decimal("0.27")


def test_completion_and_lfmp_extremes_are_jointly_evaluated(observations):
    result = evaluate(
        observations,
        completion_probability="0.9",
        cancellation_settlement_price="0.25",
        completion_probability_range=["0.8", "1"],
        cancellation_settlement_price_range=["0", "0.5"],
    )
    bounds = result["uncertainty"]
    assert Decimal(bounds["expected_payout_lower"]) == Decimal("2.0")
    assert Decimal(bounds["expected_payout_upper"]) == Decimal("3.848")
    assert result["status"] == "pass"  # Fees make the worst case negative.
    assert bounds["lower_bound_assumptions"]["completion_probability"] == "0.8"
    assert bounds["upper_bound_assumptions"]["cancellation_settlement_price"] == "0.5"


def test_point_estimate_outside_bootstrap_interval_is_not_malformed(observations):
    observations[0]["probability_interval"]["away"] = {"lower": "0.31", "upper": "0.4"}
    observations[0]["probability_interval"]["home"] = {"lower": "0.6", "upper": "0.69"}
    result = evaluate(observations)
    assert result["status"] == "paper_candidate"
    assert Decimal(result["uncertainty"]["selected_team_probability"]["lower"]) > Decimal("0.3")


@pytest.mark.parametrize(
    "keyword,reason",
    [
        ("tie_probability_range", "tie_probability_range_required"),
        ("completion_probability_range", "completion_probability_range_required"),
    ],
)
def test_missing_required_ranges_never_candidate(observations, keyword, reason):
    result = evaluate(observations, **{keyword: None})
    assert result["status"] == "pass"
    assert reason in result["reasons"]


def test_missing_model_interval_is_auditable_pass(observations):
    observations[0].pop("probability_interval")
    result = evaluate(observations)
    assert result["status"] == "pass"
    assert "probability_interval_required" in result["reasons"]
    assert result["uncertainty"]["ev_per_contract_lower"] is None


def test_cancellation_range_required_if_completion_can_be_below_one(observations):
    result = evaluate(
        observations, completion_probability="1", completion_probability_range=["0.99", "1"]
    )
    assert result["status"] == "pass"
    assert "cancellation_settlement_price_range_required" in result["reasons"]


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("model_id", "another-model", "probability_interval_model_mismatch"),
        ("model_sha256", "b" * 64, "probability_interval_model_mismatch"),
        ("training_game_ids_sha256", "b" * 64, "probability_interval_training_data_mismatch"),
        ("ensemble_sha256", "invalid", "probability_interval_provenance_required"),
        ("method", "row_bootstrap", "unsupported_probability_interval_method"),
        ("scope", "outcome_probability", "unsupported_probability_interval_method"),
        ("game_id", "another-game", "probability_interval_forecast_mismatch"),
        ("features_sha256", "f" * 64, "probability_interval_forecast_mismatch"),
        ("as_of", "2026-10-28T11:59:29Z", "probability_interval_forecast_mismatch"),
        ("training_cutoff_at", "2026-10-28T12:01:00Z", "probability_interval_training_from_future"),
    ],
)
def test_mismatched_or_future_interval_provenance_passes(observations, field, value, reason):
    observations[0]["probability_interval"][field] = value
    result = evaluate(observations)
    assert result["status"] == "pass"
    assert reason in result["reasons"]


@pytest.mark.parametrize(
    "invalid", [["0.7", "0.6"], ["-0.1", "0.2"], ["0", "1.1"], ["NaN", "1"], [True, 1], [0], "0,1"]
)
def test_malformed_ranges_raise_clear_errors(observations, invalid):
    with pytest.raises(ValueError):
        evaluate(observations, tie_probability_range=invalid)


def test_malformed_probability_interval_cannot_drive_entry(observations):
    observations[0]["probability_interval"]["home"] = {"lower": ".8", "upper": ".7"}
    with pytest.raises(ValueError, match="lower bound"):
        evaluate(observations)


def test_home_away_interval_complements_are_required(observations):
    observations[0]["probability_interval"]["home"]["lower"] = "0.6"
    with pytest.raises(ValueError, match="complementary"):
        evaluate(observations)


def test_point_assumptions_must_stay_inside_explicit_ranges(observations):
    result = evaluate(observations, tie_probability="0.1")
    assert result["status"] == "pass"
    assert "point_assumption_outside_uncertainty_range" in result["reasons"]


def test_policy_roundtrip_reproduces_full_evaluation_and_identity(observations):
    first = evaluate(observations)
    second = evaluate_expected_value(
        *observations,
        selected_team="CAR",
        quantity=10,
        as_of=NOW,
        tie_probability="0.02",
        completion_probability="1",
        **first["uncertainty_policy"],
    )
    assert first == second
    before = first["evaluation_id"]
    changed = evaluate(observations, tie_probability_range=["0.01", "0.04"])
    assert changed["evaluation_id"] != before


def test_stale_quotes_and_partial_depth_remain_passes(observations):
    observations[1]["quote_at"] = "2026-10-28T11:58:00Z"
    result = evaluate(observations)
    assert result["status"] == "pass"
    assert "quote_stale" in result["reasons"]
    observations[1]["quote_at"] = "2026-10-28T11:59:58Z"
    observations[1]["ask_levels"][0]["quantity"] = "9.99"
    assert "insufficient_depth" in evaluate(observations)["reasons"]


def test_fractional_depth_reaching_full_quantity_has_robust_bounds(observations):
    observations[1]["ask_levels"] = [
        {"price": "0.2", "quantity": "0.75"},
        {"price": "0.25", "quantity": "1.25"},
    ]
    result = evaluate(observations, quantity=2)
    assert result["uncertainty"]["ev_per_contract_lower"] is not None


def test_custom_threshold_is_applied_to_worst_case_after_fees(observations):
    result = evaluate(observations, min_ev_per_contract="0.04")
    assert Decimal(result["ev_per_contract"]) > Decimal("0.04")
    assert result["status"] == "pass"
    assert "uncertainty_does_not_clear_minimum_ev" in result["reasons"]


def test_ranges_activate_uncertainty_even_without_require_flag(observations):
    result = evaluate(observations, require_uncertainty=False)
    assert "uncertainty" in result
    assert result["uncertainty_policy"]["require_uncertainty"] is False


def test_inputs_are_preserved_and_range_helper_is_exact(observations):
    before = copy.deepcopy(observations)
    evaluate(observations)
    assert observations == before
    assert validate_probability_range(["0.001", "0.1"]) == (Decimal("0.001"), Decimal("0.1"))
