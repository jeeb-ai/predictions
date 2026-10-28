"""Exact fitted logistic decomposition and pregame source values, without narratives."""

import math

import numpy as np
import pandas as pd


def explain_prediction(model, row, columns, probability):
    inputs = pd.DataFrame([{name: float(row[name]) for name in columns}])
    scaler = model.named_steps["standardscaler"]
    logistic = model.named_steps["logisticregression"]
    standardized = scaler.transform(inputs)[0]
    coefficients = logistic.coef_[0]
    contributions = standardized * coefficients
    intercept = float(logistic.intercept_[0])
    log_odds = intercept + float(contributions.sum())
    if not np.isclose(log_odds, model.decision_function(inputs)[0], atol=1e-10):
        raise ValueError("Feature contributions do not reconcile to the fitted model")
    if not np.isclose(1 / (1 + math.exp(-log_odds)), probability, atol=1e-12):
        raise ValueError("Explanation does not reconcile to the forecast probability")

    def clean(value):
        if isinstance(value, list):
            return value
        if value is None or pd.isna(value):
            return None
        if hasattr(value, "isoformat"):
            return value.isoformat()
        if isinstance(value, (float, int, np.number)):
            return float(value)
        return str(value)

    sources = {
        name: clean(row[name])
        for name in row.index
        if name.startswith(("home_", "away_"))
        and name not in {"home_score", "away_score", "home_win", "home_team", "away_team"}
    }
    return {
        "method": "exact_standardized_logistic_decomposition",
        "intercept": intercept,
        "home_log_odds": log_odds,
        "reconstructed_home_probability": probability,
        "contributions": [
            {
                "feature": name,
                "raw_value": float(row[name]),
                "training_mean": float(scaler.mean_[i]),
                "training_scale": float(scaler.scale_[i]),
                "standardized_value": float(standardized[i]),
                "coefficient": float(coefficients[i]),
                "log_odds_contribution": float(contributions[i]),
            }
            for i, name in enumerate(columns)
        ],
        "source_values": sources,
        "limitations": [
            "Associations in a fitted model, not causal effects.",
            "QB is the most-used passer in the last observed game, not a confirmed starter.",
            "EPA uses an older-season remaining-half EP estimator; opponent adjustment is Elo, not opponent-adjusted EPA.",
            "Empty histories use documented neutral model defaults; sample counts identify missing observations.",
        ],
    }
