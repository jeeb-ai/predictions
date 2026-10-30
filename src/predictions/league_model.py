"""Sport-neutral annual fitting, proper scores and exact-line research utilities.

Each adapter supplies its own features, outcome rules, score support and folds.
No NFL probability, feature definition or market observation enters this module.
"""

import hashlib
import json
import uuid
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression, PoissonRegressor, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


@threadpool_limits.wrap(limits=2)
def fit_league(train, cal, columns, sport):
    if train.available_at.max() >= cal.kickoff.min() or len(train) < 20 or len(cal) < 10:
        raise ValueError("Invalid chronological league fit/calibration")
    logistic = make_pipeline(
        StandardScaler(), LogisticRegression(C=1, max_iter=2000, random_state=42)
    )
    logistic.fit(train[columns], (train.home_score > train.away_score).astype(int))
    platt = LogisticRegression(C=1, max_iter=2000, random_state=42).fit(
        logistic.decision_function(cal[columns]).reshape(-1, 1),
        (cal.home_score > cal.away_score).astype(int),
    )
    models = []
    for side in ["home", "away"]:
        estimator = (
            Ridge(alpha=10) if sport == "NBA" else PoissonRegressor(alpha=0.1, max_iter=1000)
        )
        model = make_pipeline(StandardScaler(), estimator).fit(
            train[columns], train[side + "_score"]
        )
        models.append(model)
    pred = np.column_stack([m.predict(cal[columns]) for m in models])
    actual = cal[["home_score", "away_score"]].to_numpy()
    residual = actual - pred
    return {
        "sport": sport,
        "columns": columns,
        "winner": logistic,
        "platt": platt,
        "score_models": models,
        "bias": residual.mean(axis=0),
        "covariance": np.cov(residual.T) + np.eye(2) * 0.1,
        "baseline_means": actual.mean(axis=0),
        "baseline_covariance": np.cov(actual.T) + np.eye(2) * 0.1,
        "baseline_win": float((train.home_score > train.away_score).mean()),
        "train_ids_sha256": hashlib.sha256(
            "\n".join(train.game_id.astype(str)).encode()
        ).hexdigest(),
        "calibration_ids_sha256": hashlib.sha256(
            "\n".join(cal.game_id.astype(str)).encode()
        ).hexdigest(),
    }


def pmf_for(model, means, baseline=False):
    limit = 250 if model["sport"] == "NBA" else 20
    grid = np.arange(limit + 1)
    h, a = np.meshgrid(grid, grid, indexing="ij")
    if model["sport"] == "NBA":
        center = model["baseline_means"] if baseline else means + model["bias"]
        delta = np.stack([h - center[0], a - center[1]], axis=-1)
        pmf = np.exp(
            -0.5
            * np.einsum(
                "...i,ij,...j->...",
                delta,
                np.linalg.inv(model["baseline_covariance"] if baseline else model["covariance"]),
                delta,
            )
        )
        # NBA final scores include overtime and cannot be tied.
        np.fill_diagonal(pmf, 0)
    else:
        center = model["baseline_means"] if baseline else np.maximum(means, 0.05)
        pmf = np.outer(poisson.pmf(grid, center[0]), poisson.pmf(grid, center[1]))
    return pmf / pmf.sum()


def line_probabilities(pmf, spread, total):
    if (
        not np.isfinite([spread, total]).all()
        or not float(spread * 2).is_integer()
        or not float(total * 2).is_integer()
    ):
        raise ValueError("Only finite integer/half-point lines supported")
    h, a = np.indices(pmf.shape)

    def events(value):
        return {
            "win": float(pmf[value > 0].sum()),
            "push": float(pmf[value == 0].sum()),
            "loss": float(pmf[value < 0].sum()),
        }

    return {
        "home_cover": events(h - a + spread),
        "over": events(h + a - total),
        "market_status": "PASS: user-entered scenario, no verified executable quote/settlement",
    }


def summarize(pmf):
    grid = np.arange(pmf.shape[0])
    home = pmf.sum(axis=1)
    away = pmf.sum(axis=0)

    def q(p, level):
        return int(np.searchsorted(p.cumsum(), level))

    return {
        "home_mean": float(home @ grid),
        "away_mean": float(away @ grid),
        "expected_margin": float(home @ grid - away @ grid),
        "expected_total": float(home @ grid + away @ grid),
        "home_interval_90": [q(home, 0.05), q(home, 0.95)],
        "away_interval_90": [q(away, 0.05), q(away, 0.95)],
    }


def winner_explanation(model, row):
    columns = model["columns"]
    input = pd.DataFrame([{c: float(row[c]) for c in columns}])
    pipeline = model["winner"]
    z = pipeline.named_steps["standardscaler"].transform(input)[0]
    platt = float(model["platt"].coef_[0, 0])
    coef = pipeline.named_steps["logisticregression"].coef_[0] * platt
    intercept = float(
        pipeline.named_steps["logisticregression"].intercept_[0] * platt
        + model["platt"].intercept_[0]
    )
    contributions = z * coef
    logit = intercept + contributions.sum()
    p = float(expit(logit))
    expected = model["platt"].predict_proba(pipeline.decision_function(input).reshape(-1, 1))[0, 1]
    if not np.isclose(p, expected, atol=1e-12):
        raise ValueError("League explanation does not reconcile")
    return p, {
        "intercept": intercept,
        "home_log_odds": float(logit),
        "contributions": [
            {
                "feature": c,
                "raw_value": float(row[c]),
                "standardized_value": float(z[i]),
                "coefficient": float(coef[i]),
                "log_odds_contribution": float(contributions[i]),
            }
            for i, c in enumerate(columns)
        ],
    }


def evaluate_league(model, test, spread, total):
    result = []
    pred = np.column_stack([m.predict(test[model["columns"]]) for m in model["score_models"]])
    prob = model["platt"].predict_proba(
        model["winner"].decision_function(test[model["columns"]]).reshape(-1, 1)
    )[:, 1]
    for (_, row), means, p in zip(test.iterrows(), pred, prob, strict=True):
        for baseline in [False, True]:
            pmf = pmf_for(model, means, baseline)
            s = summarize(pmf)
            h, a = int(row.home_score), int(row.away_score)
            win = int(h > a)
            pwin = model["baseline_win"] if baseline else p
            if max(h, a) >= pmf.shape[0]:
                raise ValueError("Observed score beyond configured support")
            lines = line_probabilities(pmf, spread, total)
            rec = {
                "game_id": row.game_id,
                "season": int(row.season),
                "date": str(row.kickoff.date()),
                "kind": "baseline" if baseline else "model",
                "probability": float(pwin),
                "outcome": win,
                "accuracy": float((pwin >= 0.5) == win),
                "brier": float((pwin - win) ** 2),
                "log_loss": float(-np.log(max(pwin if win else 1 - pwin, 1e-15))),
                "score_mae": float((abs(s["home_mean"] - h) + abs(s["away_mean"] - a)) / 2),
                "score_mse": float(((s["home_mean"] - h) ** 2 + (s["away_mean"] - a) ** 2) / 2),
                "joint_log_loss": float(-np.log(max(pmf[h, a], 1e-300))),
                "margin_mae": float(abs(s["expected_margin"] - h + a)),
                "total_mae": float(abs(s["expected_total"] - h - a)),
                "coverage_90": float(
                    (
                        int(s["home_interval_90"][0] <= h <= s["home_interval_90"][1])
                        + int(s["away_interval_90"][0] <= a <= s["away_interval_90"][1])
                    )
                    / 2
                ),
            }
            grid = np.arange(pmf.shape[0])
            rec["score_crps"] = float(
                (
                    np.sum((pmf.sum(axis=1).cumsum() - (grid >= h)) ** 2)
                    + np.sum((pmf.sum(axis=0).cumsum() - (grid >= a)) ** 2)
                )
                / 2
            )
            for label, value, events in [
                ("spread", h - a + spread, lines["home_cover"]),
                ("total", h + a - total, lines["over"]),
            ]:
                target = "win" if value > 0 else "loss" if value < 0 else "push"
                rec[label + "_brier3"] = sum((events[k] - int(k == target)) ** 2 for k in events)
                rec[label + "_probability"] = events["win"]
                rec[label + "_outcome"] = int(target == "win")
                rec[label + "_push_probability"] = events["push"]
                rec[label + "_push_outcome"] = int(target == "push")
            result.append(rec)
    return pd.DataFrame(result)


def league_metrics(frame):
    keys = [
        "accuracy",
        "brier",
        "log_loss",
        "score_mae",
        "joint_log_loss",
        "margin_mae",
        "total_mae",
        "coverage_90",
        "score_crps",
        "spread_brier3",
        "total_brier3",
    ]
    result = {k: float(frame[k].mean()) for k in keys}
    result.update(games=len(frame), score_rmse=float(np.sqrt(frame.score_mse.mean())))
    bins = {}
    for name, pcol, ycol in [
        ("winner", "probability", "outcome"),
        ("spread", "spread_probability", "spread_outcome"),
        ("total", "total_probability", "total_outcome"),
    ]:
        bins[name] = []
        for i in range(10):
            group = frame.loc[
                frame[pcol].ge(i / 10)
                & (frame[pcol].lt((i + 1) / 10) if i < 9 else frame[pcol].le(1))
            ]
            bins[name].append(
                {
                    "count": len(group),
                    "predicted": float(group[pcol].mean()) if len(group) else None,
                    "observed": float(group[ycol].mean()) if len(group) else None,
                }
            )
    result["calibration"] = bins
    result["ece"] = sum(
        b["count"] * abs(b["predicted"] - b["observed"]) for b in bins["winner"] if b["count"]
    ) / len(frame)
    groups = frame.assign(block=pd.to_datetime(frame.date).dt.strftime("%Y-%U")).groupby("block")
    counts = groups.size().to_numpy()
    sums = groups[keys].sum().to_numpy()
    rng = np.random.default_rng(42)
    samples = []
    for _ in range(500):
        index = rng.integers(0, len(counts), len(counts))
        samples.append(sums[index].sum(axis=0) / counts[index].sum())
    result["intervals_95"] = (
        {
            key: np.quantile(np.asarray(samples)[:, i], [0.025, 0.975]).tolist()
            for i, key in enumerate(keys)
        }
        if len(counts) > 1
        else None
    )
    return result


def backtest_league(features, columns, sport, root, holdout, source, spread, total):
    path = Path(root) / (sport.lower() + "-" + uuid.uuid4().hex)
    path.mkdir(parents=True, exist_ok=False)
    seasons = sorted(int(y) for y in features.season.unique() if y <= holdout)
    results = []
    folds = []
    for season in seasons[6:]:
        test = features.loc[features.season.eq(season)]
        cal = features.loc[
            features.season.eq(season - 1) & features.available_at.lt(test.kickoff.min())
        ]
        train = features.loc[
            features.season.lt(season - 1) & features.available_at.lt(cal.kickoff.min())
        ]
        model = fit_league(train, cal, columns, sport)
        joblib.dump(model, path / f"model-{season}.joblib")
        results.append(evaluate_league(model, test, spread, total))
        folds.append(
            {
                "test_season": season,
                "calibration_season": season - 1,
                "training_seasons": sorted(int(y) for y in train.season.unique()),
                "training_games": len(train),
                "calibration_games": len(cal),
                "train_ids_sha256": model["train_ids_sha256"],
                "calibration_ids_sha256": model["calibration_ids_sha256"],
                "latest_calibration_result": cal.available_at.max().isoformat(),
                "first_test_cutoff": test.kickoff.min().isoformat(),
            }
        )
    predictions = pd.concat(results, ignore_index=True)
    predictions.to_json(path / "predictions.json", orient="records")
    report = {
        "type": "league_validation",
        "sport": sport,
        "model_version": sport.lower() + "-v1",
        "run_dir": str(path),
        "source": source,
        "columns": columns,
        "holdout_season": holdout,
        "folds": folds,
        "splits": {
            split: {
                kind: league_metrics(predictions.loc[mask & predictions.kind.eq(kind)])
                for kind in ["model", "baseline"]
            }
            for split, mask in [
                ("validation", predictions.season.lt(holdout)),
                ("holdout", predictions.season.eq(holdout)),
            ]
        },
        "scenarios": {
            "home_spread": spread,
            "total": total,
            "kind": "fixed research lines, not historical market prices",
        },
        "limitations": [
            "Retrospective provider snapshots; original publication vintages not proven.",
            "No verified historical player availability supplied; unknown, not assumed healthy.",
            "Week-block fixed-prediction intervals omit refitting and selection uncertainty.",
            "No verified market quote or reliable betting-edge claim.",
        ],
    }
    (path / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
