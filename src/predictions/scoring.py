"""Independent NFL joint score distributions with chronological calibration.

No binary winner-model probabilities or market observations enter this module.
Scores include overtime as in nflverse. The integer grid is a predictive
approximation, not a simulation or a guarantee of football scoring structure.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson
from sklearn.linear_model import PoissonRegressor, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from predictions.features import FEATURE_COLUMNS

COLUMNS = [
    *FEATURE_COLUMNS,
    *[
        f"{side}_{name}"
        for side in ["home", "away"]
        for name in [
            "win_rate",
            "margin",
            "points_for",
            "points_against",
            "rest_days",
            "games_played",
        ]
    ],
]
CANDIDATES = ["constant_poisson", "ridge_joint_gaussian", "poisson_joint_nb_mixture"]
GRID = np.arange(151)
H, A = np.meshgrid(GRID, GRID, indexing="ij")


def _now():
    return pd.Timestamp.now(tz="UTC")


@threadpool_limits.wrap(limits=2)
def fit_scoring(train, calibration, kind):
    if kind not in CANDIDATES or len(train) < 10 or len(calibration) < 8:
        raise ValueError("Unsupported model or insufficient scoring history")
    if train.result_available_at.max() >= calibration.kickoff.min():
        raise ValueError("Score calibration must follow training chronologically")
    models = []
    for side in ["home", "away"]:
        if kind == "constant_poisson":
            models.append(float(train[side + "_score"].mean()))
        else:
            estimator = (
                Ridge(alpha=10)
                if kind == "ridge_joint_gaussian"
                else PoissonRegressor(alpha=0.1, max_iter=1000)
            )
            model = make_pipeline(StandardScaler(), estimator)
            model.fit(train[COLUMNS], train[side + "_score"])
            models.append(model)
    result = {"kind": kind, "models": models, "feature_columns": COLUMNS}
    means = predict_means(result, calibration)
    actual = calibration[["home_score", "away_score"]].to_numpy()
    residual = actual - means
    if kind == "ridge_joint_gaussian":
        result.update(bias=residual.mean(axis=0), covariance=np.cov(residual.T) + np.eye(2) * 0.1)
    if kind == "poisson_joint_nb_mixture":
        components = []
        for x, y in [(False, False), (False, True), (True, False), (True, True)]:
            mask = (residual[:, 0] >= 0) == x
            mask &= (residual[:, 1] >= 0) == y
            if not mask.any():
                continue
            ratio = np.maximum(actual[mask] / means[mask], 0.01).mean(axis=0)
            group = actual[mask]
            variance = group.var(axis=0) if len(group) > 1 else actual.var(axis=0)
            alpha = np.clip(
                (variance - group.mean(axis=0)) / np.maximum(group.mean(axis=0), 1) ** 2, 0.005, 2
            )
            components.append({"weight": float(mask.mean()), "ratio": ratio, "alpha": alpha})
        result["components"] = components
    return result


def predict_means(model, frame):
    return np.column_stack(
        [
            np.full(len(frame), item)
            if isinstance(item, float)
            else np.maximum(item.predict(frame[COLUMNS]), 0.1)
            for item in model["models"]
        ]
    )


def distribution(model, mean):
    if model["kind"] == "constant_poisson":
        pmf = np.outer(poisson.pmf(GRID, mean[0]), poisson.pmf(GRID, mean[1]))
    elif model["kind"] == "ridge_joint_gaussian":
        delta = np.stack([H - mean[0] - model["bias"][0], A - mean[1] - model["bias"][1]], axis=-1)
        inv = np.linalg.inv(model["covariance"])
        quadratic = np.einsum("...i,ij,...j->...", delta, inv, delta)
        pmf = np.exp(-0.5 * quadratic) / (2 * np.pi * np.sqrt(np.linalg.det(model["covariance"])))
    else:
        pmf = np.zeros(H.shape)
        for component in model["components"]:
            mu = mean * component["ratio"]
            shape = 1 / component["alpha"]
            pmf += component["weight"] * np.outer(
                nbinom.pmf(GRID, shape[0], shape[0] / (shape[0] + mu[0])),
                nbinom.pmf(GRID, shape[1], shape[1] / (shape[1] + mu[1])),
            )
    mass = pmf.sum()
    if (
        not np.isfinite(mass)
        or mass <= 0
        or mass < (0.001 if model["kind"] == "ridge_joint_gaussian" else 0.99)
    ):
        raise ValueError("Score distribution has excessive truncated/invalid mass")
    return pmf / mass


def outcome_probabilities(pmf, *, home_spread=0, total_line=44.5):
    if (
        not np.isfinite([home_spread, total_line]).all()
        or abs(home_spread) > 150
        or not 0 <= total_line <= 300
    ):
        raise ValueError("Invalid exact spread/total line")
    if (
        pmf.shape != H.shape
        or not np.isfinite(pmf).all()
        or (pmf < 0).any()
        or not np.isclose(pmf.sum(), 1)
    ):
        raise ValueError("Invalid joint probability distribution")
    if not float(2 * home_spread).is_integer() or not float(2 * total_line).is_integer():
        raise ValueError(
            "Only integer and half-point lines are supported; quarter-line settlement is unknown"
        )
    margin = H - A + home_spread
    total = H + A - total_line

    def events(value):
        return {
            "win": float(pmf[value > 0].sum()),
            "push": float(pmf[value == 0].sum()),
            "loss": float(pmf[value < 0].sum()),
        }

    return {
        "home_cover": events(margin),
        "over": events(total),
        "rules": "Full-game scores including overtime; push is separate. No market settlement or quote is assumed.",
    }


def summarize(pmf):
    home = pmf.sum(axis=1)
    away = pmf.sum(axis=0)

    def quantile(p, q):
        return min(150, int(np.searchsorted(p.cumsum(), q)))

    return {
        "home_mean": float(home @ GRID),
        "away_mean": float(away @ GRID),
        "home_median": quantile(home, 0.5),
        "away_median": quantile(away, 0.5),
        "expected_margin": float((pmf * (H - A)).sum()),
        "expected_total": float((pmf * (H + A)).sum()),
        "home_interval_90": [quantile(home, 0.05), quantile(home, 0.95)],
        "away_interval_90": [quantile(away, 0.05), quantile(away, 0.95)],
        "home_win_probability": float(pmf[H > A].sum()),
        "tie_probability": float(np.trace(pmf)),
        "uncertainty": "Predictive score distribution conditional on fitted inputs; coverage must be assessed out of sample. Not model-refit uncertainty.",
    }


def _crps(probabilities, observed):
    return float(np.sum((probabilities.cumsum() - (GRID >= observed)) ** 2))


def evaluate_rows(model, frame):
    rows = []
    for (_, row), mean in zip(frame.iterrows(), predict_means(model, frame), strict=True):
        pmf = distribution(model, mean)
        summary = summarize(pmf)
        h, a = int(row.home_score), int(row.away_score)
        if max(h, a) > 150:
            raise ValueError("Observed score exceeds declared scoring grid")
        record = {
            "game_id": row.game_id,
            "season": int(row.season),
            "week": int(row.week),
            "kind": model["kind"],
            **summary,
            "home_score": h,
            "away_score": a,
            "joint_log_loss": float(-np.log(max(pmf[h, a], 1e-300))),
            "score_mae": (abs(summary["home_mean"] - h) + abs(summary["away_mean"] - a)) / 2,
            "score_mse": ((summary["home_mean"] - h) ** 2 + (summary["away_mean"] - a) ** 2) / 2,
            "margin_mae": abs(summary["expected_margin"] - h + a),
            "total_mae": abs(summary["expected_total"] - h - a),
            "score_crps": (_crps(pmf.sum(axis=1), h) + _crps(pmf.sum(axis=0), a)) / 2,
            "coverage_90": (
                int(summary["home_interval_90"][0] <= h <= summary["home_interval_90"][1])
                + int(summary["away_interval_90"][0] <= a <= summary["away_interval_90"][1])
            )
            / 2,
        }
        decisive = h != a
        decisive_probability = summary["home_win_probability"] / max(
            1 - summary["tie_probability"], 1e-12
        )
        record["moneyline_probability"] = decisive_probability
        record["moneyline_outcome"] = float(h > a) if decisive else np.nan
        record["moneyline_accuracy"] = (
            float((decisive_probability >= 0.5) == (h > a)) if decisive else np.nan
        )
        record["moneyline_brier"] = (decisive_probability - int(h > a)) ** 2 if decisive else np.nan
        record["moneyline_log_loss"] = (
            -float(np.log(max(decisive_probability if h > a else 1 - decisive_probability, 1e-300)))
            if decisive
            else np.nan
        )
        margin_probability = np.bincount((H - A + 150).ravel(), weights=pmf.ravel(), minlength=301)
        total_probability = np.bincount((H + A).ravel(), weights=pmf.ravel(), minlength=301)
        record["margin_crps"] = float(
            np.sum((margin_probability.cumsum() - (np.arange(301) >= h - a + 150)) ** 2)
        )
        record["total_crps"] = float(
            np.sum((total_probability.cumsum() - (np.arange(301) >= h + a)) ** 2)
        )
        for name, spread, line in [
            ("spread_integer", -3, 44.5),
            ("spread_half", -3.5, 44.5),
            ("total_integer", 0, 44),
            ("total_half", 0, 44.5),
        ]:
            prob = outcome_probabilities(pmf, home_spread=spread, total_line=line)[
                "over" if name.startswith("total") else "home_cover"
            ]
            value = h + a - line if name.startswith("total") else h - a + spread
            target = "win" if value > 0 else "loss" if value < 0 else "push"
            record[name + "_probability"] = prob["win"]
            record[name + "_outcome"] = int(target == "win")
            record[name + "_push_probability"] = prob["push"]
            record[name + "_push_outcome"] = int(target == "push")
            record[name + "_brier"] = sum((prob[key] - int(key == target)) ** 2 for key in prob)
            record[name + "_log_loss"] = -float(np.log(max(prob[target], 1e-300)))
        rows.append(record)
    return pd.DataFrame(rows)


def metrics(frame):
    names = [
        "joint_log_loss",
        "score_mae",
        "margin_mae",
        "total_mae",
        "score_crps",
        "coverage_90",
        *[
            name + suffix
            for name in ["spread_integer", "spread_half", "total_integer", "total_half"]
            for suffix in ["_brier", "_log_loss"]
        ],
    ]
    values = {key: float(frame[key].mean()) for key in names}
    values["score_rmse"] = float(np.sqrt(frame.score_mse.mean()))
    for key in [
        "moneyline_accuracy",
        "moneyline_brier",
        "moneyline_log_loss",
        "margin_crps",
        "total_crps",
    ]:
        values[key] = float(frame[key].mean())
    values["games"] = len(frame)
    values["confidence_intervals_95"] = sampling_intervals(frame)
    values["reliability"] = {}
    for name in ["spread_integer", "spread_half", "total_integer", "total_half"]:
        bins = []
        for i in range(10):
            selected = frame.loc[
                (frame[name + "_probability"] >= i / 10)
                & (
                    frame[name + "_probability"] < (i + 1) / 10
                    if i < 9
                    else frame[name + "_probability"] <= 1
                )
            ]
            bins.append(
                {
                    "lower": i / 10,
                    "upper": (i + 1) / 10,
                    "count": len(selected),
                    "predicted": float(selected[name + "_probability"].mean())
                    if len(selected)
                    else None,
                    "observed": float(selected[name + "_outcome"].mean())
                    if len(selected)
                    else None,
                }
            )
        values["reliability"][name] = {
            "bins": bins,
            "predicted_push_rate": float(frame[name + "_push_probability"].mean()),
            "observed_push_rate": float(frame[name + "_push_outcome"].mean()),
        }
    return values


def backtest_scoring(features, root, *, source_metadata, holdout=2025, min_train_seasons=5):
    """Compare predeclared models; last older season calibrates, never test rows."""
    output = Path(root) / ("scoring-" + uuid.uuid4().hex)
    output.mkdir(parents=True, exist_ok=False)
    regular = features.loc[features.game_type.eq("REG") & features.home_score.notna()].copy()
    seasons = sorted(int(s) for s in regular.season.unique() if s <= holdout)
    results = []
    audit = []
    for season in seasons[min_train_seasons + 1 :]:
        test = regular.loc[regular.season.eq(season)]
        older = seasons[: seasons.index(season)]
        cal_season = older[-1]
        calibration = regular.loc[
            regular.season.eq(cal_season) & regular.result_available_at.lt(test.kickoff.min())
        ]
        train = regular.loc[
            regular.season.lt(cal_season)
            & regular.result_available_at.lt(calibration.kickoff.min())
        ]
        if test.empty or calibration.empty:
            continue
        audit.append(
            {
                "test_season": season,
                "calibration_season": cal_season,
                "train_seasons": sorted(train.season.unique().tolist()),
                "train_games": len(train),
                "calibration_games": len(calibration),
                "train_latest_result": train.result_available_at.max().isoformat(),
                "calibration_latest_result": calibration.result_available_at.max().isoformat(),
                "test_first_kickoff": test.kickoff.min().isoformat(),
            }
        )
        for kind in CANDIDATES:
            model = fit_scoring(train, calibration, kind)
            joblib.dump(model, output / f"{kind}-{season}.joblib")
            results.append(evaluate_rows(model, test))
    if not results:
        raise ValueError("No chronological scoring folds")
    predictions = pd.concat(results, ignore_index=True)
    predictions.to_json(output / "predictions.json", orient="records")
    if not predictions.season.lt(holdout).any() or not predictions.season.eq(holdout).any():
        raise ValueError("Need nonempty chronological validation and holdout scoring folds")
    split = {}
    for name, mask in [
        ("validation", predictions.season.lt(holdout)),
        ("holdout", predictions.season.eq(holdout)),
    ]:
        split[name] = {
            kind: metrics(predictions.loc[mask & predictions.kind.eq(kind)]) for kind in CANDIDATES
        }
    selected = min(CANDIDATES, key=lambda kind: split["validation"][kind]["joint_log_loss"])
    report = {
        "schema_version": 1,
        "type": "nfl_scoring_validation",
        "run_dir": str(output),
        "source_metadata": source_metadata,
        "feature_columns": COLUMNS,
        "splits": split,
        "holdout_season": holdout,
        "selected_kind": selected,
        "selection_rule": "Lowest validation joint log loss; holdout never selects the model. Fixed hyperparameters.",
        "folds": audit,
        "limitations": [
            "Current historical snapshots, not original publication vintages.",
            "2025 has been examined by earlier winner-model work; cannot claim it was truly unseen.",
            "No injuries, confirmed starters, market lines or binary win probabilities are model inputs.",
            "Gaussian discrete-score and joint negative-binomial mixture are approximations; football scoring lattice and overtime mechanisms are not explicitly modeled.",
            "Predictive probabilities require prospective calibration monitoring; retrospective results do not establish trading value.",
        ],
        "status": "evaluated_not_a_profit_claim",
        "grid_max_score": 150,
    }
    report["paired_comparisons"] = {
        name: {
            other: paired_intervals(predictions.loc[mask], selected, other)
            for other in CANDIDATES
            if other != selected
        }
        for name, mask in [
            ("validation", predictions.season.lt(holdout)),
            ("holdout", predictions.season.eq(holdout)),
        ]
    }
    report["per_season"] = {
        str(year): {
            kind: {
                key: value
                for key, value in metrics(
                    predictions.loc[predictions.season.eq(year) & predictions.kind.eq(kind)]
                ).items()
                if key != "reliability"
            }
            for kind in CANDIDATES
        }
        for year in sorted(predictions.season.unique())
    }
    report["sampling_uncertainty"] = (
        "500 fixed-prediction season/week block resamples, seed42; excludes model refitting, selection uncertainty and dependence across weeks."
    )
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report, predictions


def sampling_intervals(frame, draws=500):
    """Week-block fixed-forecast sampling uncertainty, not model refitting."""
    names = [
        "joint_log_loss",
        "score_mae",
        "score_mse",
        "margin_mae",
        "total_mae",
        "score_crps",
        "coverage_90",
    ]
    grouped = frame.groupby(["season", "week"])[names].sum()
    counts = frame.groupby(["season", "week"]).size().to_numpy()
    if len(grouped) < 2:
        return None
    values = grouped.to_numpy()
    rng = np.random.default_rng(42)
    samples = []
    for _ in range(draws):
        picks = rng.integers(0, len(grouped), size=len(grouped))
        samples.append(values[picks].sum(axis=0) / counts[picks].sum())
    samples = np.array(samples)
    samples[:, 2] = np.sqrt(samples[:, 2])
    names[2] = "score_rmse"
    return {key: np.quantile(samples[:, i], [0.025, 0.975]).tolist() for i, key in enumerate(names)}


def forecast_scoring(games, report, game_ids, root, *, as_of, source_metadata):
    """Fit an approved scoring family on older seasons and issue immutable forecasts."""
    import hashlib

    from predictions.features import build_features

    cutoff = pd.Timestamp(as_of)
    generated = _now()
    if cutoff.tzinfo is None or cutoff > generated:
        raise ValueError("Scoring as_of must be timezone-aware and not future")
    if (
        report.get("type") != "nfl_scoring_validation"
        or report.get("status") != "evaluated_not_a_profit_claim"
    ):
        raise ValueError("Completed scoring validation required")
    from predictions.inference import _validate_sources

    _validate_sources(source_metadata, cutoff)
    frame = games.copy()
    frame.loc[frame.result_available_at >= cutoff, ["home_score", "away_score"]] = np.nan
    features = build_features(frame)
    validate_report(report)
    targets = features.loc[features.game_id.isin(game_ids)]
    if len(targets) != len(set(game_ids)) or targets.season.nunique() != 1:
        raise ValueError("Scoring batch needs known IDs from one season")
    if not targets.game_type.eq("REG").all():
        raise ValueError("Only NFL regular-season scoring is historically evaluated")
    season = int(targets.iloc[0].season)
    if report["holdout_season"] >= season:
        raise ValueError("Live season must follow historical evaluation")
    if (
        targets.home_score.notna().any()
        or (targets.kickoff <= generated).any()
        or targets.get("kickoff_time_imputed", pd.Series(False, index=targets.index)).any()
    ):
        raise ValueError("Scoring requires upcoming unfinished games with known kickoff")
    older = features.loc[
        features.season.lt(season) & features.home_score.notna() & features.game_type.eq("REG")
    ]
    calibration = older.loc[
        older.season.eq(season - 1)
        & older.result_available_at.lt(
            min(cutoff, features.loc[features.season.eq(season), "kickoff"].min())
        )
    ]
    train = older.loc[
        older.season.lt(season - 1) & older.result_available_at.lt(calibration.kickoff.min())
    ]
    model = fit_scoring(train, calibration, report["selected_kind"])
    directory = Path(root) / ("score-" + uuid.uuid4().hex)
    directory.mkdir(parents=True, exist_ok=False)
    joblib.dump(model, directory / "model.joblib")

    def sha(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()

    records = {}
    for (_, row), mean in zip(targets.iterrows(), predict_means(model, targets), strict=True):
        pmf = distribution(model, mean)
        identifier = uuid.uuid4().hex
        path = directory / (identifier + ".npy")
        np.save(path, pmf, allow_pickle=False)
        generated = _now()
        if generated >= row.kickoff:
            raise ValueError("Game started during scoring computation")
        output = {
            "artifact_type": "nfl_score_forecast",
            "schema_version": 1,
            "prediction_id": identifier,
            "game_id": row.game_id,
            "home_team": row.home_team,
            "away_team": row.away_team,
            "kickoff": row.kickoff.isoformat(),
            "as_of": cutoff.isoformat(),
            "generated_at": generated.isoformat(),
            "source_metadata": source_metadata,
            "model_kind": model["kind"],
            "validation_report_sha256": hashlib.sha256(
                json.dumps(report, sort_keys=True).encode()
            ).hexdigest(),
            "training_game_ids_sha256": hashlib.sha256(
                "\n".join(train.game_id).encode()
            ).hexdigest(),
            "calibration_game_ids_sha256": hashlib.sha256(
                "\n".join(calibration.game_id).encode()
            ).hexdigest(),
            "training_seasons": sorted(int(s) for s in train.season.unique()),
            "calibration_season": season - 1,
            "train_latest_result": train.result_available_at.max().isoformat(),
            "calibration_latest_result": calibration.result_available_at.max().isoformat(),
            "features": {key: float(row[key]) for key in COLUMNS},
            "summary": summarize(pmf),
            "distribution": {"path": str(path.resolve()), "sha256": sha(path)},
            "model_sha256": sha(directory / "model.joblib"),
            "market_comparison": {
                "status": "pass",
                "reason": "No verified executable quote or settlement rules supplied.",
            },
        }
        (directory / (identifier + ".json")).write_text(
            json.dumps(output, indent=2, allow_nan=False) + "\n"
        )
        records[row.game_id] = output
    return records


def read_score_distribution(score):
    """Only trusted local artifact paths are supported; never network requests."""
    import hashlib

    path = Path(score["distribution"]["path"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != score["distribution"]["sha256"]:
        raise ValueError("Scoring distribution checksum mismatch")
    if path.stat().st_size > 1_000_000:
        raise ValueError("Scoring artifact exceeds read limit")
    values = np.load(path, allow_pickle=False)
    if (
        values.shape != H.shape
        or not np.isfinite(values).all()
        or (values < 0).any()
        or not np.isclose(values.sum(), 1)
    ):
        raise ValueError("Invalid retained scoring distribution")
    return values


def validate_report(report):
    """An operator may approve a complete local evaluation, never a status-only stub."""
    try:
        if (
            report["type"] != "nfl_scoring_validation"
            or report["status"] != "evaluated_not_a_profit_claim"
            or report["feature_columns"] != COLUMNS
            or report["grid_max_score"] != 150
        ):
            raise ValueError("Scoring validation schema mismatch")
        scores = {
            kind: report["splits"]["validation"][kind]["joint_log_loss"] for kind in CANDIDATES
        }
        if not all(np.isfinite(value) for value in scores.values()) or report[
            "selected_kind"
        ] != min(CANDIDATES, key=scores.get):
            raise ValueError("Model selection must use validation loss only")
        if not report["folds"] or any(
            max(f["train_seasons"]) >= f["calibration_season"]
            or f["calibration_season"] >= f["test_season"]
            for f in report["folds"]
        ):
            raise ValueError("Invalid chronological scoring folds")
        if report["splits"]["holdout"][report["selected_kind"]]["games"] <= 0:
            raise ValueError("No held-out scoring results")
    except (KeyError, TypeError) as exc:
        raise ValueError("Completed chronological scoring validation is required") from exc
    return report


def paired_intervals(frame, selected, other, draws=500):
    left = frame.loc[frame.kind.eq(selected)].set_index("game_id")
    right = frame.loc[frame.kind.eq(other)].set_index("game_id").loc[left.index]
    delta = left[["season", "week"]].copy()
    keys = ["joint_log_loss", "score_mae", "score_crps"]
    for key in keys:
        delta[key] = left[key] - right[key]
    groups = delta.groupby(["season", "week"])
    sums = groups[keys].sum().to_numpy()
    counts = groups.size().to_numpy()
    rng = np.random.default_rng(42)
    samples = []
    for _ in range(draws):
        index = rng.integers(0, len(sums), len(sums))
        samples.append(sums[index].sum(axis=0) / counts[index].sum())
    samples = np.asarray(samples)
    return {
        key: {
            "selected_minus_comparator": float(delta[key].mean()),
            "confidence_interval_95": np.quantile(samples[:, i], [0.025, 0.975]).tolist(),
        }
        for i, key in enumerate(keys)
    }
