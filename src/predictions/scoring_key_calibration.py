"""Isolated NFL key-number research; never changes production score forecasts."""

import hashlib
import json
import uuid
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from predictions.scoring import (
    A,
    H,
    distribution,
    fit_scoring,
    outcome_probabilities,
    predict_means,
)

MARGINS = [0, 3, 7, 10, 14]
TOTALS = [37, 41, 44, 47, 51]
SCENARIOS = {
    "spread_3": (-3, "home_cover"),
    "spread_7": (-7, "home_cover"),
    "total_44": (44, "over"),
    "total_47": (47, "over"),
}


def category(h, a):
    margin = np.full(np.shape(h), 5, dtype=int)
    total = np.full(np.shape(h), 5, dtype=int)
    for index, value in enumerate(MARGINS):
        margin[np.abs(h - a) == value] = index
    for index, value in enumerate(TOTALS):
        total[h + a == value] = index
    return margin * 6 + total


GROUPS = category(H, A)
DESIGN = np.zeros((36, 10))
for group in range(36):
    margin, total = divmod(group, 6)
    if margin < 5:
        DESIGN[group, margin] = 1
    if total < 5:
        DESIGN[group, 5 + total] = 1


def fit_correction(probability_groups, observed_groups, *, calibration_latest, test_first):
    latest, first = pd.Timestamp(calibration_latest), pd.Timestamp(test_first)
    if (
        pd.isna(latest)
        or pd.isna(first)
        or latest.tzinfo is None
        or first.tzinfo is None
        or latest >= first
    ):
        raise ValueError("Key-number calibration must precede the test season")
    masses = np.asarray(probability_groups, dtype=float)
    observed = np.asarray(observed_groups, dtype=int)
    if (
        masses.ndim != 2
        or masses.shape[1] != 36
        or len(masses) != len(observed)
        or len(observed) < 8
        or not np.isfinite(masses).all()
        or (masses < 0).any()
        or not np.allclose(masses.sum(axis=1), 1)
        or (observed < 0).any()
        or (observed >= 36).any()
    ):
        raise ValueError("Invalid key-number calibration data")
    target = DESIGN[observed].mean(axis=0)

    def objective(theta):
        weights = np.exp(DESIGN @ theta)
        z = masses @ weights
        loss = np.mean(np.log(z)) - target @ theta + 0.02 * np.sum(theta**2)
        expected = ((masses * weights) / z[:, None]) @ DESIGN
        gradient = expected.mean(axis=0) - target + 0.04 * theta
        return loss, gradient

    optimum = minimize(objective, np.zeros(10), jac=True, method="L-BFGS-B", bounds=[(-2, 2)] * 10)
    if not optimum.success:
        raise ValueError("Key-number calibration failed: " + optimum.message)
    return optimum.x


def correct(pmf, theta):
    if (
        pmf.shape != H.shape
        or not np.isfinite(pmf).all()
        or (pmf < 0).any()
        or not np.isclose(pmf.sum(), 1)
        or np.shape(theta) != (10,)
        or not np.isfinite(theta).all()
        or np.abs(theta).max() > 2.000001
    ):
        raise ValueError("Invalid key-number correction")
    adjusted = pmf * np.exp(DESIGN @ theta)[GROUPS]
    return adjusted / adjusted.sum()


def evaluate(pmf, game):
    h, a = int(game.home_score), int(game.away_score)
    home_mean = float(np.sum(pmf * H))
    away_mean = float(np.sum(pmf * A))
    result = {
        "joint_log_loss": float(-np.log(max(pmf[h, a], 1e-300))),
        "score_mae": (abs(home_mean - h) + abs(away_mean - a)) / 2,
    }
    for name, (line, kind) in SCENARIOS.items():
        events = outcome_probabilities(
            pmf, **({"home_spread": line} if kind == "home_cover" else {"total_line": line})
        )[kind]
        value = h - a + line if kind == "home_cover" else h + a - line
        label = "win" if value > 0 else "loss" if value < 0 else "push"
        result[name + "_brier3"] = sum((p - int(k == label)) ** 2 for k, p in events.items())
        result[name + "_log_loss"] = float(-np.log(max(events[label], 1e-300)))
        result[name + "_push_probability"] = events["push"]
        result[name + "_push_outcome"] = int(label == "push")
    return result


def intervals(frame, keys, draws=500):
    grouped = frame.groupby(["season", "week"])
    sums, counts = grouped[keys].sum().to_numpy(), grouped.size().to_numpy()
    if len(counts) < 2:
        return None
    rng = np.random.default_rng(42)
    samples = []
    for _ in range(draws):
        index = rng.integers(0, len(counts), len(counts))
        samples.append(sums[index].sum(axis=0) / counts[index].sum())
    return {
        key: np.quantile(np.array(samples)[:, i], [0.025, 0.975]).tolist()
        for i, key in enumerate(keys)
    }


def investigate(features, base_report, root):
    holdout = int(base_report["holdout_season"])
    if base_report["selected_kind"] != "ridge_joint_gaussian":
        raise ValueError("Experiment predeclared for the existing ridge_joint_gaussian baseline")
    regular = features.loc[features.game_type.eq("REG") & features.home_score.notna()]
    output = Path(root) / ("key-calibration-" + uuid.uuid4().hex)
    output.mkdir(parents=True, exist_ok=False)
    records, folds = [], []
    for original in base_report["folds"]:
        season, cal_year = int(original["test_season"]), int(original["calibration_season"])
        test = regular.loc[regular.season.eq(season)]
        cal = regular.loc[
            regular.season.eq(cal_year) & regular.result_available_at.lt(test.kickoff.min())
        ]
        train = regular.loc[
            regular.season.lt(cal_year) & regular.result_available_at.lt(cal.kickoff.min())
        ]
        if len(train) != original["train_games"] or len(cal) != original["calibration_games"]:
            raise ValueError("Historical folds do not reproduce the approved baseline")
        model = fit_scoring(train, cal, "ridge_joint_gaussian")
        groups = [
            np.bincount(GROUPS.ravel(), weights=distribution(model, mean).ravel(), minlength=36)
            for mean in predict_means(model, cal)
        ]
        theta = fit_correction(
            groups,
            category(cal.home_score.to_numpy(), cal.away_score.to_numpy()),
            calibration_latest=cal.result_available_at.max(),
            test_first=test.kickoff.min(),
        )
        joblib.dump(
            {"base_model": model, "correction": theta}, output / f"research-{season}.joblib"
        )
        folds.append(
            {
                **original,
                "weights": theta.tolist(),
                "research_model_sha256": hashlib.sha256(
                    (output / f"research-{season}.joblib").read_bytes()
                ).hexdigest(),
                "training_game_ids_sha256": hashlib.sha256(
                    "\n".join(train.game_id).encode()
                ).hexdigest(),
                "calibration_game_ids_sha256": hashlib.sha256(
                    "\n".join(cal.game_id).encode()
                ).hexdigest(),
            }
        )
        for (_, game), mean in zip(test.iterrows(), predict_means(model, test), strict=True):
            base = distribution(model, mean)
            for name, pmf in [("base", base), ("candidate", correct(base, theta))]:
                records.append(
                    {
                        "game_id": game.game_id,
                        "season": season,
                        "week": int(game.week),
                        "kind": name,
                        **evaluate(pmf, game),
                    }
                )
    predictions = pd.DataFrame(records)
    predictions.to_json(output / "predictions.json", orient="records")
    keys = [
        "joint_log_loss",
        "score_mae",
        *[
            name + suffix
            for name in SCENARIOS
            for suffix in ["_brier3", "_log_loss", "_push_probability", "_push_outcome"]
        ],
    ]
    comparison = {}
    for split, mask in [
        ("validation", predictions.season.lt(holdout)),
        ("holdout", predictions.season.eq(holdout)),
    ]:
        frame = predictions.loc[mask]
        summary = {}
        for kind in ["base", "candidate"]:
            f = frame.loc[frame.kind.eq(kind)]
            summary[kind] = {
                "games": len(f),
                **{key: float(f[key].mean()) for key in keys},
                "intervals_95": intervals(f, keys),
            }
        base = frame.loc[frame.kind.eq("base")].set_index("game_id")
        candidate = frame.loc[frame.kind.eq("candidate")].set_index("game_id")
        differences = candidate[keys] - base[keys]
        differences[["season", "week"]] = candidate[["season", "week"]]
        summary["paired_candidate_minus_base_95"] = intervals(differences, keys)
        reference = base_report["splits"][split]["ridge_joint_gaussian"]
        for metric in ["joint_log_loss", "score_mae"]:
            if not np.isclose(summary["base"][metric], reference[metric], atol=1e-10):
                raise ValueError("Baseline metric reproduction failed: " + metric)
        comparison[split] = summary
    report = {
        "type": "nfl_key_number_research",
        "run_dir": str(output),
        "holdout_season": holdout,
        "base_model": "ridge_joint_gaussian",
        "source": base_report["source_metadata"],
        "source_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "base_report_sha256": hashlib.sha256(
            json.dumps(base_report, sort_keys=True).encode()
        ).hexdigest(),
        "method": {
            "absolute_margin_categories": MARGINS,
            "total_categories": TOTALS,
            "reference": "all other values",
            "penalty": 0.02,
            "weight_bounds": [-2, 2],
            "fit": "previous calibration season only; base models refitted identically",
        },
        "scenarios": SCENARIOS,
        "folds": folds,
        "splits": comparison,
        "promoted": False,
        "limitations": [
            "2025 outcomes and original key-number diagnostics were already examined; this is retrospective research, not a newly untouched holdout.",
            "Fixed-prediction paired week-block intervals omit refitting, selection and source-revision uncertainty.",
            "Predeclared category weights are an approximation to football scoring, not a possession/drive model.",
            "No historical executable market prices or betting-edge validation. Production scoring and winner models unchanged.",
        ],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main(argv=None):
    import argparse

    from predictions.data import load_games
    from predictions.features import build_features
    from predictions.scoring import validate_report

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/scoring-calibration"))
    args = parser.parse_args(argv)
    report = validate_report(json.loads(args.validation.read_text()))
    digest = hashlib.sha256(args.input.read_bytes()).hexdigest()
    if digest != report["source_metadata"].get("sha256"):
        raise ValueError("Input SHA256 must match the original validated schedule snapshot")
    cutoff = pd.Timestamp(report["source_metadata"]["as_of"])
    if pd.isna(cutoff) or cutoff.tzinfo is None:
        raise ValueError("Validated source cutoff must be timezone-aware")
    features = build_features(load_games(args.input, as_of=cutoff))
    print(json.dumps(investigate(features, report, args.output_dir), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
