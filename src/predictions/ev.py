"""Conservative, fee-aware paper EV for verified Polymarket US NFL contracts.

Forecast probabilities are conditional on a decisive NFL result. Ties and
completion/cancellation assumptions must be supplied explicitly. This module
never submits orders and never interprets a positive point estimate as proof
of a profitable strategy.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation, localcontext

import pandas as pd

from predictions.data import TEAM_ALIASES, TEAMS

CENT = Decimal("0.01")
ONE = Decimal(1)
ZERO = Decimal(0)
STANDARD_THETA = Decimal("0.0695")
FEE_SOURCE = "https://docs.polymarket.us/fees.md"
RULES_SOURCE = "https://docs.polymarket.us/faqs/sports-faqs.md"
FEE_EFFECTIVE_AT = pd.Timestamp("2026-10-19T17:00:00Z")


def _decimal(value, name: str) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{name} must be a finite decimal")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{name} must be a finite decimal") from exc
    if not result.is_finite():
        raise ValueError(f"{name} must be a finite decimal")
    return result


def _probability(value, name: str) -> Decimal:
    result = _decimal(value, name)
    if not ZERO <= result <= ONE:
        raise ValueError(f"{name} must be between zero and one")
    return result


def _time(value, name: str) -> pd.Timestamp:
    try:
        result = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a timezone-aware timestamp") from exc
    if pd.isna(result) or result.tzinfo is None:
        raise ValueError(f"{name} must be a timezone-aware timestamp")
    return result.tz_convert("UTC")


def _team(value, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a canonical NFL team")
    result = TEAM_ALIASES.get(value.strip().upper(), value.strip().upper())
    if result not in TEAMS:
        raise ValueError(f"{name} must be a canonical NFL team")
    return result


def _identity(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _number(value: Decimal) -> str:
    return format(value, "f")


def _money(value: Decimal) -> str:
    return format(value.quantize(CENT, rounding=ROUND_HALF_EVEN), ".2f")


def _finish(result: dict) -> dict:
    result["reasons"] = list(dict.fromkeys(result["reasons"]))
    content = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
    result["evaluation_id"] = "ev-" + hashlib.sha256(content.encode()).hexdigest()[:24]
    return result


def _supported_terms(terms) -> bool:
    if not isinstance(terms, dict):
        return False
    try:
        return (
            terms.get("rule_type") == "nfl_winner_including_overtime"
            and _decimal(terms.get("win_payout"), "win_payout") == ONE
            and _decimal(terms.get("loss_payout"), "loss_payout") == ZERO
            and _decimal(terms.get("tie_payout"), "tie_payout") == Decimal("0.5")
            and terms.get("cancellation") == "last_fair_market_price"
            and terms.get("rules_url") == RULES_SOURCE
            and isinstance(terms.get("description"), str)
            and bool(terms["description"].strip())
            and isinstance(terms.get("terms_sha256"), str)
            and re.fullmatch(r"[a-f0-9]{64}", terms["terms_sha256"]) is not None
            and hashlib.sha256(terms["description"].encode()).hexdigest() == terms["terms_sha256"]
        )
    except ValueError:
        return False


def _supported_fees(schedule) -> bool:
    if not isinstance(schedule, dict):
        return False
    try:
        return (
            schedule.get("type") == "polymarket_us_standard_taker"
            and _decimal(schedule.get("theta"), "theta") == STANDARD_THETA
            and schedule.get("rounding") == "ROUND_HALF_EVEN"
            and schedule.get("currency") == "USD"
            and schedule.get("source_url") == FEE_SOURCE
            and _time(schedule.get("effective_at"), "fee effective_at") == FEE_EFFECTIVE_AT
        )
    except ValueError:
        return False


def _order_cost(levels, quantity: int, theta: Decimal) -> dict:
    """Consume all required depth; distinguish a conservative cap from a fill estimate.

    The exchange caps cumulative collected commission at the rounded cumulative
    exact fee. Per-fill rounding can reduce it further. Aggregated L2 depth does
    not reveal the resting-order partition, so the cap is the fee used for EV.
    """
    if not isinstance(levels, list):
        raise ValueError("ask_levels must be a list")
    parsed = []
    for level in levels:
        if not isinstance(level, dict):
            raise ValueError("Each ask level must be an object")
        price = _decimal(level.get("price"), "ask price")
        available = _decimal(level.get("quantity"), "ask quantity")
        if not ZERO < price < ONE or available <= ZERO:
            raise ValueError(
                "Ask prices must be strictly between zero and one and quantities positive"
            )
        parsed.append((price, available))
    parsed.sort(key=lambda level: level[0])
    remaining = Decimal(quantity)
    exact_total, collected_estimate, notional = ZERO, ZERO, ZERO
    fills = []
    for price, available in parsed:
        if remaining <= ZERO:
            break
        used = min(remaining, available)
        value = used * price
        exact = theta * used * price * (ONE - price)
        exact_total += exact
        cumulative_cap = exact_total.quantize(CENT, rounding=ROUND_HALF_EVEN)
        # The cap can only reduce this hypothetical per-displayed-level fill.
        charged = min(
            exact.quantize(CENT, rounding=ROUND_HALF_EVEN),
            max(ZERO, cumulative_cap - collected_estimate),
        )
        collected_estimate += charged
        notional += value
        fills.append(
            {
                "price": _number(price),
                "quantity": _number(used),
                "notional": _number(value),
                "exact_fee": _number(exact),
                "estimated_fee": _money(charged),
            }
        )
        remaining -= used
    fee_cap = exact_total.quantize(CENT, rounding=ROUND_HALF_EVEN)
    return {
        "fills": fills,
        "fill_quantity": _number(Decimal(quantity) - remaining),
        "notional": _number(notional),
        "fee": _money(fee_cap),
        "fee_upper_bound": _money(fee_cap),
        "fee_estimate_per_displayed_level": _money(collected_estimate),
        "exact_fee_before_rounding": _number(exact_total),
        "total_cost": _number(notional + fee_cap),
        "fee_kind": "estimated_upper_bound",
        "fee_note": (
            "L2 depth hides resting-order splits. EV uses the rounded cumulative exact-fee "
            "cap; actual per-fill collected fees can be lower. Rebates are not assumed."
        ),
    }


def evaluate_expected_value(
    forecast: dict,
    quote: dict,
    *,
    selected_team: str,
    quantity: int,
    as_of,
    tie_probability=None,
    completion_probability=None,
    cancellation_settlement_price=None,
    max_quote_age_seconds: int = 60,
    max_forecast_age_seconds: int = 3600,
    min_ev_per_contract="0",
) -> dict:
    """Evaluate a buy of YES or NO; disqualifications are auditable paper passes.

    Malformed inputs raise ValueError. Missing risk assumptions, stale inputs,
    unsupported contracts, insufficient liquidity, or nonpositive EV return
    ``status='pass'``. A candidate is only a paper candidate, never an order.
    ``tie_probability`` is conditional on a completed official game.
    """
    if not isinstance(forecast, dict) or not isinstance(quote, dict):
        raise ValueError("forecast and quote must be objects")
    if (
        isinstance(quantity, bool)
        or not isinstance(quantity, int)
        or not 1 <= quantity <= 1_000_000
    ):
        raise ValueError("quantity must be an integer from 1 to 1000000")
    for name, value in [
        ("max_quote_age_seconds", max_quote_age_seconds),
        ("max_forecast_age_seconds", max_forecast_age_seconds),
    ]:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    minimum_ev = _decimal(min_ev_per_contract, "min_ev_per_contract")
    if minimum_ev < ZERO:
        raise ValueError("min_ev_per_contract must be nonnegative")
    now = _time(as_of, "as_of")
    kickoff = _time(forecast.get("kickoff"), "forecast kickoff")
    forecast_at = _time(forecast.get("as_of"), "forecast as_of")
    quoted_at = _time(quote.get("quote_at"), "quote_at")
    received_at = _time(quote.get("received_at"), "received_at")
    market_start = _time(quote.get("game_start_time"), "game_start_time")
    home = _team(forecast.get("home_team"), "home_team")
    away = _team(forecast.get("away_team"), "away_team")
    yes_team = _team(quote.get("yes_team"), "yes_team")
    no_team = _team(quote.get("no_team"), "no_team")
    selected = _team(selected_team, "selected_team")
    home_probability = _probability(forecast.get("home_win_probability"), "home_win_probability")
    away_probability = _probability(forecast.get("away_win_probability"), "away_win_probability")
    if abs(home_probability + away_probability - ONE) > Decimal("0.000000000001"):
        raise ValueError("Decisive home and away probabilities must sum to one")
    outcome = quote.get("outcome")
    result = {
        "schema_version": 1,
        "mode": "paper_only",
        "forecast_id": _identity(forecast.get("forecast_id"), "forecast_id"),
        "quote_id": _identity(quote.get("quote_id"), "quote_id"),
        "liquidity_key": quote.get("liquidity_key"),
        "market_id": _identity(quote.get("market_id"), "market_id"),
        "game_id": _identity(forecast.get("game_id"), "game_id"),
        "selected_team": selected,
        "outcome": outcome,
        "as_of": now.isoformat(),
        "kickoff": kickoff.isoformat(),
        "quote_at": quoted_at.isoformat(),
        "received_at": received_at.isoformat(),
        "forecast_as_of": forecast_at.isoformat(),
        "max_quote_age_seconds": max_quote_age_seconds,
        "max_forecast_age_seconds": max_forecast_age_seconds,
        "min_ev_per_contract": _number(minimum_ev),
        "quantity": quantity,
        "fill_quantity": "0",
        "fills": [],
        "notional": None,
        "fee": None,
        "fee_upper_bound": None,
        "fee_estimate_per_displayed_level": None,
        "total_cost": None,
        "expected_payout": None,
        "expected_profit": None,
        "ev_per_contract": None,
        "completed_game_expected_payout": None,
        "completed_game_ev_per_contract": None,
        "status": "pass",
        "reasons": [],
        "probability_scope": forecast.get("probability_scope"),
        "fee_schedule": copy.deepcopy(quote.get("fee_schedule")),
        "settlement": copy.deepcopy(quote.get("settlement")),
        "source": copy.deepcopy(quote.get("source", {})),
        "match_evidence": {
            "method": "exact_canonical_teams_and_kickoff",
            "game_id": forecast["game_id"],
            "quote_game_id": quote.get("game_id"),
            "home_team": home,
            "away_team": away,
            "yes_team": yes_team,
            "no_team": no_team,
            "forecast_kickoff": kickoff.isoformat(),
            "market_game_start_time": market_start.isoformat(),
        },
        "forecast_provenance": {
            key: copy.deepcopy(forecast[key])
            for key in ["model_id", "model_version", "model_sha256", "features_sha256", "source"]
            if key in forecast
        },
        "assumptions": {},
        "ev_scope": "conditional_on_explicit_tie_completion_and_cancellation_assumptions",
    }
    reasons = result["reasons"]
    checks = [
        (quote.get("venue") != "polymarket_us", "unsupported_venue"),
        (quote.get("market_status") != "open", "market_not_open"),
        (
            quote.get("game_id") is not None and quote["game_id"] != forecast["game_id"],
            "market_game_mismatch",
        ),
        (
            home == away or yes_team == no_team or {home, away} != {yes_team, no_team},
            "market_team_mismatch",
        ),
        (market_start != kickoff, "market_kickoff_mismatch"),
        (outcome not in {"yes", "no"}, "unsupported_outcome"),
        (selected != (yes_team if outcome == "yes" else no_team), "selected_team_outcome_mismatch"),
        (selected not in {home, away}, "selected_team_not_in_forecast"),
        (
            forecast.get("probability_scope") != "conditional_on_decisive_result",
            "unsupported_probability_scope",
        ),
        (now >= kickoff or forecast_at >= kickoff, "game_started_or_nonpregame_forecast"),
        (forecast_at > now, "forecast_from_future"),
        ((now - forecast_at).total_seconds() > max_forecast_age_seconds, "forecast_stale"),
        (quoted_at > now or received_at > now, "quote_from_future"),
        (received_at < quoted_at, "quote_timestamp_inversion"),
        ((now - quoted_at).total_seconds() > max_quote_age_seconds, "quote_stale"),
        ((now - received_at).total_seconds() > max_quote_age_seconds, "quote_receipt_stale"),
        (not _supported_terms(quote.get("settlement")), "unsupported_settlement_terms"),
        (not _supported_fees(quote.get("fee_schedule")), "unsupported_fee_schedule"),
        (now < FEE_EFFECTIVE_AT, "fee_schedule_not_effective"),
    ]
    reasons.extend(reason for condition, reason in checks if condition)
    for name in [
        "history_available_through",
        "advanced_history_available_through",
        "home_qb_proxy_available_at",
        "away_qb_proxy_available_at",
        "model_train_max_result_available_at",
    ]:
        if forecast.get(name) is not None and _time(forecast[name], name) >= forecast_at:
            reasons.append("forecast_history_not_strictly_before_cutoff")
    if (
        forecast.get("model_training_cutoff_at") is not None
        and _time(forecast["model_training_cutoff_at"], "model_training_cutoff_at") > forecast_at
    ):
        reasons.append("model_training_cutoff_from_future")
    if (
        forecast.get("feature_cutoff_at") is not None
        and _time(forecast["feature_cutoff_at"], "feature_cutoff_at") != forecast_at
    ):
        reasons.append("forecast_feature_cutoff_mismatch")
    for name, value in [
        ("tie_probability", tie_probability),
        ("completion_probability", completion_probability),
        ("cancellation_settlement_price", cancellation_settlement_price),
    ]:
        result["assumptions"][name] = (
            _number(_probability(value, name)) if value is not None else None
        )
    if tie_probability is None:
        reasons.append("tie_probability_required")
    if completion_probability is None:
        reasons.append("completion_probability_required")
    elif _probability(completion_probability, "completion_probability") < ONE and (
        cancellation_settlement_price is None
    ):
        reasons.append("cancellation_settlement_price_required")
    structural_reasons = set(reasons) - {
        "tie_probability_required",
        "completion_probability_required",
        "cancellation_settlement_price_required",
    }
    if structural_reasons:
        return _finish(result)
    minimum_quantity = quote.get("minimum_trade_quantity")
    if minimum_quantity is not None and Decimal(quantity) < _decimal(
        minimum_quantity, "minimum_trade_quantity"
    ):
        reasons.append("below_minimum_quantity")
        return _finish(result)
    with localcontext() as context:
        context.prec = 40
        cost = _order_cost(quote.get("ask_levels"), quantity, STANDARD_THETA)
        result.update(cost)
        tick_value = quote.get("price_tick")
        if tick_value is not None:
            tick = _decimal(tick_value, "price_tick")
            if tick <= ZERO:
                raise ValueError("price_tick must be positive")
            if any(_decimal(fill["price"], "price") % tick for fill in result["fills"]):
                reasons.append("price_tick_mismatch")
        if _decimal(result["fill_quantity"], "fill_quantity") != Decimal(quantity):
            reasons.append("insufficient_depth")
            return _finish(result)
        if tie_probability is not None:
            tie = _probability(tie_probability, "tie_probability")
            decisive = home_probability if selected == home else away_probability
            completed_per_contract = (ONE - tie) * decisive + tie * Decimal("0.5")
            completed_payout = Decimal(quantity) * completed_per_contract
            total_cost = _decimal(result["total_cost"], "total_cost")
            result["decisive_win_probability"] = _number(decisive)
            result["completed_game_expected_payout"] = _number(completed_payout)
            result["completed_game_ev_per_contract"] = _number(
                (completed_payout - total_cost) / Decimal(quantity)
            )
            if not reasons:
                completion = _probability(completion_probability, "completion_probability")
                cancellation = (
                    _probability(cancellation_settlement_price, "cancellation_settlement_price")
                    if cancellation_settlement_price is not None
                    else ZERO
                )
                payout = Decimal(quantity) * (
                    completion * completed_per_contract + (ONE - completion) * cancellation
                )
                profit = payout - total_cost
                per_contract = profit / Decimal(quantity)
                result["expected_payout"] = _number(payout)
                result["expected_profit"] = _number(profit)
                result["ev_per_contract"] = _number(per_contract)
                if per_contract <= minimum_ev:
                    reasons.append("not_above_minimum_ev_after_fees")
                else:
                    result["status"] = "paper_candidate"
    return _finish(result)
