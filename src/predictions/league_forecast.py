"""Immutable forecasts and sport-aware ordinary-language interpretation."""

import hashlib
import json
import uuid
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from predictions.data import _atomic_write
from predictions.league_model import fit_league, pmf_for, summarize, winner_explanation


def _now():
    return pd.Timestamp.now(tz="UTC")


def issue_league(
    features, report, root, *, as_of, source, days=7, fitter=fit_league, score_builder=None
):
    now = _now()
    as_of = pd.Timestamp(as_of)
    if as_of.tzinfo is None or as_of > now:
        raise ValueError("League forecast cutoff must be observed, not future")
    sport = report["sport"]
    columns = report["columns"]
    targets = features.loc[
        features.kickoff.gt(now)
        & features.kickoff.le(now + pd.Timedelta(days=days))
        & features.home_score.isna()
    ]
    if not targets.empty:
        if pd.to_datetime(targets.feature_cutoff_at, utc=True).ne(as_of).any():
            raise ValueError("Prepare league features at this exact as_of")
        used = pd.to_datetime(targets.history_available_through, utc=True)
        if (used.notna() & used.ge(as_of)).any():
            raise ValueError("League feature history must precede as_of")

    def check_source(value):
        if isinstance(value, dict):
            if value.get("fetched_at") is not None and pd.Timestamp(value["fetched_at"]) > as_of:
                raise ValueError("League source snapshot observed after as_of")
            for child in value.values():
                check_source(child)
        elif isinstance(value, list):
            for child in value:
                check_source(child)

    check_source(source)
    if targets.empty:
        return {
            "sport": sport,
            "as_of": as_of.isoformat(),
            "forecasts": [],
            "status": "no_upcoming_games",
        }
    if targets.season.nunique() != 1:
        raise ValueError("League batch needs one season")
    season = int(targets.iloc[0].season)
    if season <= report["holdout_season"] or not report["folds"]:
        raise ValueError("Approved historical evaluation must precede live season")
    older = features.loc[features.home_score.notna() & features.available_at.lt(as_of)]
    cal = older.loc[older.season.eq(season - 1)]
    train = older.loc[older.season.lt(season - 1) & older.available_at.lt(cal.kickoff.min())]
    model = fitter(train, cal, columns, sport)
    path = Path(root) / (sport.lower() + "-" + uuid.uuid4().hex)
    path.mkdir(parents=True, exist_ok=False)
    joblib.dump(model, path / "model.joblib")
    manifest = path / "source.json"
    manifest.write_text(json.dumps(source, indent=2, allow_nan=False) + "\n")
    source_ref = {
        "manifest_path": str(manifest.resolve()),
        "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "provider": source.get("provider", sport + " verified provider snapshots"),
    }
    rows = []
    skipped = []
    means = np.column_stack([m.predict(targets[columns]) for m in model["score_models"]])
    for (_, row), mean in zip(targets.iterrows(), means, strict=True):
        p, explanation = winner_explanation(model, row)
        extra = {}
        if score_builder is None:
            pmf = pmf_for(model, mean)
        else:
            pmf, extra = score_builder(model, mean, row)
        identifier = uuid.uuid4().hex
        dist = path / (identifier + ".npy")
        np.save(dist, pmf, allow_pickle=False)
        generated = _now()
        if generated >= row.kickoff:
            skipped.append({"game_id": str(row.game_id), "reason": "started_during_computation"})
            continue
        record = {
            "sport": sport,
            "forecast_id": identifier,
            "game_id": str(row.game_id),
            "home_team": row.home_team,
            "away_team": row.away_team,
            "home_name": row.home_name,
            "away_name": row.away_name,
            "kickoff": row.kickoff.isoformat(),
            "as_of": as_of.isoformat(),
            "generated_at": generated.isoformat(),
            "model_version": report["model_version"],
            "model_sha256": hashlib.sha256((path / "model.joblib").read_bytes()).hexdigest(),
            "home_win_probability": p,
            "away_win_probability": 1 - p,
            "probability_scope": "full_game_including_overtime_no_ties"
            if sport == "NBA"
            else "full_game_including_overtime_shootout",
            "features": {c: float(row[c]) for c in columns},
            "explanation": explanation,
            "score_summary": summarize(pmf),
            "distribution": {
                "path": str(dist.resolve()),
                "sha256": hashlib.sha256(dist.read_bytes()).hexdigest(),
            },
            "history_available_through": row.history_available_through.isoformat()
            if pd.notna(row.history_available_through)
            else None,
            "source_games": {side: row[side + "_source_games"] for side in ["home", "away"]},
            "availability": "Unknown: no verified point-in-time player availability in the historical source; not modeled.",
            "source": source_ref,
            "training_seasons": sorted(int(y) for y in train.season.unique()),
            "calibration_season": season - 1,
            "train_ids_sha256": model["train_ids_sha256"],
            "calibration_ids_sha256": model["calibration_ids_sha256"],
            "historical_validation": {
                "season": report["holdout_season"],
                **{
                    k: report.get("splits", {}).get("holdout", {}).get("model", {}).get(k)
                    for k in [
                        "games",
                        "accuracy",
                        "brier",
                        "log_loss",
                        "ece",
                        "intervals_95",
                        "regulation_draw_probability",
                        "regulation_draw_outcome",
                    ]
                },
            },
            "validation_sha256": hashlib.sha256(
                json.dumps(report, sort_keys=True).encode()
            ).hexdigest(),
            "market_comparison": {
                "status": "pass",
                "reason": "No verified executable market quote or matching settlement rules.",
            },
        }
        if "regulation_array" in extra:
            reg_path = path / (identifier + "-regulation.npy")
            np.save(reg_path, extra.pop("regulation_array"), allow_pickle=False)
            extra["regulation_distribution"] = {
                "path": str(reg_path.resolve()),
                "sha256": hashlib.sha256(reg_path.read_bytes()).hexdigest(),
            }
        record.update(extra)
        if sport == "NHL":
            record["goalie_information"] = {
                side: row[side + "_goalie_proxy"] for side in ["home", "away"]
            }
            record["availability"] = (
                "Confirmed starting goalies and player availability are unknown. Prior tracked-shot goalie proxies are used, not actual upcoming starters."
            )
        (path / (identifier + ".json")).write_text(
            json.dumps(record, indent=2, allow_nan=False) + "\n"
        )
        rows.append(record)
    return {
        "sport": sport,
        "as_of": as_of.isoformat(),
        "forecasts": rows,
        "skipped": skipped,
        "status": "completed",
        "model_path": str(path),
    }


def publish_league(batch, state):
    state = Path(state)
    state.mkdir(parents=True, exist_ok=True)
    from predictions.prospective import ForecastJournal

    ForecastJournal(state / "prospective.sqlite").capture(batch["sport"], batch["forecasts"])
    run = state / "league-runs" / (batch["sport"].lower() + "-" + uuid.uuid4().hex)
    run.mkdir(parents=True, exist_ok=False)
    payload = json.dumps(batch, indent=2, allow_nan=False).encode()
    (run / "predictions.json").write_bytes(payload)
    import fcntl

    with (state / (batch["sport"].lower() + ".publish.lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        latest = state / (batch["sport"].lower() + "-predictions.json")
        if latest.exists() and pd.Timestamp(json.loads(latest.read_text())["as_of"]) > pd.Timestamp(
            batch["as_of"]
        ):
            return
        _atomic_write(latest, payload)


def league_verdict(row):
    home = row["home_name"]
    away = row["away_name"]
    p = row["home_win_probability"]
    winner = home if p >= 0.5 else away
    other = away if p >= 0.5 else home
    sign = 1 if p >= 0.5 else -1
    explanation = row["explanation"]
    z = explanation["intercept"] + sum(
        v["log_odds_contribution"] for v in explanation["contributions"]
    )
    from scipy.special import expit

    if not np.isclose(expit(z), p, atol=1e-12):
        raise ValueError("League explanation probability mismatch")
    labels = {
        "off_rating": "points scored per 100 estimated possessions",
        "def_rating": "points allowed per 100 estimated possessions",
        "pace": "possessions per 48 minutes",
        "form": "share of recent games won",
        "rest": "days since previous scheduled game",
        "back_to_back": "back-to-back indicator (1 means yes)",
        "elo": "opponent-adjusted rating divided by 400",
        "history": "recent observed games available",
        "xg_for": "quality-adjusted scoring chances per earlier 60-minute game (expected goals)",
        "xg_against": "quality-adjusted chances allowed per earlier 60-minute game",
        "shot_quality": "estimated goal chance per earlier unblocked shot",
        "shots_for": "tracked unblocked attempts per earlier regulation game",
        "shots_against": "opponent tracked attempts per earlier regulation game",
        "pp_xg": "earlier man-advantage scoring chances per game (expected goals)",
        "pk_xg": "earlier opponent man-advantage chances allowed per game",
        "goalie_save": "tracked-shot save fraction of the prior observed goalie",
        "goalie_shots": "tracked shots in that goalie proxy's eligible prior appearances",
    }
    factors = []
    for value in explanation["contributions"]:
        side, key = value["feature"].split("_", 1)
        label = labels.get(key, key.replace("_", " "))
        impact = sign * value["log_odds_contribution"]
        team = home if side == "home" else away
        factors.append(
            {
                "impact": impact,
                "text": f"{team}: {value['raw_value']:.2f} {label}. This input moves the model toward {winner if impact > 0 else other}, relative to its training average.",
            }
        )
    support = sorted([v for v in factors if v["impact"] > 0], key=lambda v: -v["impact"])[:3]
    risks = sorted([v for v in factors if v["impact"] < 0], key=lambda v: v["impact"])[:2]
    return {
        "summary": f"{winner} is the model's pick at {max(p, 1 - p):.0%}. {other} still has a {min(p, 1 - p):.0%} estimated chance. These are historical patterns, not proof that a statistic causes a win.",
        "support": support,
        "risks": risks,
    }
