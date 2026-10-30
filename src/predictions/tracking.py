"""Prospective-only evaluation from first admitted receipts and evidenced league outcomes."""

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from predictions.data import _atomic_write
from predictions.prospective import (
    SPORTS,
    ForecastJournal,
    connect,
    digest,
    encode,
    immutable,
    now,
    stamp,
)

LOG = logging.getLogger(__name__)


def normalize_result(row):
    if row["sport"] not in SPORTS or not isinstance(row["game_id"], str):
        raise ValueError("Unsupported result identity")
    state = row["status"]
    if state not in ["final", "scheduled", "postponed", "cancelled", "incomplete", "disputed"]:
        raise ValueError("Unsupported result state")
    observed, kickoff = stamp(row["observed_at"]), stamp(row["kickoff"])
    source = row["source"]
    if stamp(source["fetched_at"]) > observed:
        raise ValueError("Result source was observed after observation timestamp")
    if source.get("classification") not in ["official_league_data", "official_stats_archive"]:
        raise ValueError("Result requires official league provenance")

    def source_times(value):
        if isinstance(value, dict):
            if value.get("fetched_at") and stamp(value["fetched_at"]) > observed:
                raise ValueError("Result evidence contains a future observation")
            for child in value.values():
                source_times(child)
        elif isinstance(value, list):
            for child in value:
                source_times(child)

    source_times(source)
    value = source.get("sha256", "")
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Result requires a source digest")
    if source.get("path"):
        if hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest() != value:
            raise ValueError("Result raw source checksum mismatch")
    if not row.get("home_team") or not row.get("away_team") or row["home_team"] == row["away_team"]:
        raise ValueError("Result opponents are invalid")
    result = {
        k: row.get(k)
        for k in [
            "sport",
            "game_id",
            "home_team",
            "away_team",
            "kickoff",
            "status",
            "home_score",
            "away_score",
            "outcome_type",
        ]
    }
    result["kickoff"] = kickoff.isoformat()
    if state == "final":
        if kickoff + pd.Timedelta(hours=24) > observed:
            raise ValueError("Final result is not yet conservatively available")
        for side in ["home", "away"]:
            number = row[side + "_score"]
            if (
                isinstance(number, bool)
                or not isinstance(number, (int, float))
                or not np.isfinite(number)
                or number < 0
                or int(number) != number
            ):
                raise ValueError("Final scores must be nonnegative integers")
            result[side + "_score"] = int(number)
        h, a = result["home_score"], result["away_score"]
        if row["sport"] != "NFL" and h == a:
            raise ValueError("Final NBA/NHL game cannot be tied")
        if row["sport"] == "NHL":
            if row["outcome_type"] not in ["REG", "OT", "SO"]:
                raise ValueError("Unknown NHL result scope")
            if row["outcome_type"] in ["OT", "SO"] and abs(h - a) != 1:
                raise ValueError("NHL deciding goal scope is contradictory")
            result["regulation_home_score"] = h - int(h > a) if row["outcome_type"] != "REG" else h
            result["regulation_away_score"] = a - int(a > h) if row["outcome_type"] != "REG" else a
    return {
        "content": result,
        "content_sha256": digest(result),
        "observed_at": observed.isoformat(),
        "source": source,
    }


class OutcomeJournal:
    def __init__(self, path):
        self.path = Path(path)
        ForecastJournal(path)
        with connect(path) as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS result_observations (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT, observation_id TEXT UNIQUE NOT NULL,
                    sport TEXT NOT NULL, game_id TEXT NOT NULL, payload TEXT NOT NULL, sha256 TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS result_game ON result_observations(sport,game_id,sequence);
                CREATE TABLE IF NOT EXISTS result_checks(sport TEXT,game_id TEXT,checked_at TEXT,error_code TEXT,PRIMARY KEY(sport,game_id));
                CREATE TABLE IF NOT EXISTS prospective_metrics(snapshot_id TEXT PRIMARY KEY,sport TEXT NOT NULL,as_of TEXT NOT NULL,payload TEXT NOT NULL,sha256 TEXT NOT NULL);
            """)
            for table, key in [
                ("result_observations", "observation_id"),
                ("prospective_metrics", "snapshot_id"),
            ]:
                immutable(db, table)
                db.execute(
                    f"CREATE TRIGGER IF NOT EXISTS {table}_no_replace BEFORE INSERT ON {table} WHEN EXISTS(SELECT 1 FROM {table} WHERE {key}=NEW.{key}) BEGIN SELECT RAISE(ABORT,'append-only history'); END"
                )
            db.commit()

    def observations(self, as_of):
        result = []
        with connect(self.path, readonly=True) as db:
            for row in db.execute(
                "SELECT payload,sha256 FROM result_observations ORDER BY sequence"
            ):
                item = json.loads(row["payload"])
                if digest(item) != row["sha256"]:
                    raise ValueError("Result observation checksum mismatch")
                if stamp(item["observed_at"]) <= stamp(as_of):
                    result.append(item)
        return result

    def append(self, raw, *, as_of):
        item = normalize_result(raw)
        if stamp(item["observed_at"]) > stamp(as_of):
            raise ValueError("Cannot use future results")
        sport, identity = item["content"]["sport"], item["content"]["game_id"]
        with connect(self.path) as db:
            db.execute("BEGIN IMMEDIATE")
            prior = db.execute(
                "SELECT payload,sha256 FROM result_observations WHERE sport=? AND game_id=? ORDER BY sequence DESC LIMIT 1",
                (sport, identity),
            ).fetchone()
            previous = json.loads(prior["payload"]) if prior else None
            if previous and digest(previous) != prior["sha256"]:
                raise ValueError("Result history checksum mismatch")
            if previous and stamp(item["observed_at"]) < stamp(previous["observed_at"]):
                return "stale_source"
            content = item["content"]
            last_final = previous.get("last_confirmed_final") if previous else None
            candidate = previous.get("candidate") if previous else None
            last_class = previous.get("last_confirmed_source_class") if previous else None
            incoming_class = item["source"]["classification"]
            weaker = (
                incoming_class == "official_stats_archive" and last_class == "official_league_data"
            )
            if weaker and content["status"] != "final":
                return "stale_archive_nonfinal"
            if (
                candidate
                and previous.get("candidate_source_class") == "official_league_data"
                and incoming_class == "official_stats_archive"
                and content == last_final
            ):
                return "archive_cannot_resolve_official_dispute"
            if previous and content == previous["content"] and candidate is None:
                return "unchanged"
            if last_final and content != last_final:
                confirmed = (
                    not weaker
                    and stamp(item["source"]["fetched_at"])
                    > stamp(previous["source"]["fetched_at"])
                    and candidate == content
                    and stamp(item["observed_at"])
                    >= stamp(previous["observed_at"]) + pd.Timedelta(minutes=5)
                )
                if candidate == content and not confirmed:
                    return "awaiting_confirmation"
                if not confirmed:
                    item["candidate"] = content
                    item["candidate_source_class"] = incoming_class
                    item["content"] = {**content, "status": "disputed"}
                    item["reason"] = "changed_result_requires_independent_later_confirmation"
                else:
                    item["reason"] = "correction_confirmed_by_later_fetch"
            elif content["status"] == "final":
                item["reason"] = "official_final"
            if item["content"]["status"] == "final":
                if content != last_final or incoming_class == "official_league_data":
                    last_class = incoming_class
                last_final = content
            item["last_confirmed_source_class"] = last_class
            item["last_confirmed_final"] = last_final
            item["previous_observation_id"] = previous["observation_id"] if previous else None
            item["reported_content_sha256"] = item["content_sha256"]
            item["content_sha256"] = digest(item["content"])
            item["observation_id"] = digest(item)
            db.execute(
                "INSERT INTO result_observations(observation_id,sport,game_id,payload,sha256) VALUES(?,?,?,?,?)",
                (item["observation_id"], sport, identity, encode(item), digest(item)),
            )
            db.commit()
        return item["content"]["status"]


def primary_receipts(records):
    chosen = {}
    for record in records:
        if record["eligible"]:
            chosen.setdefault((record["sport"], record["game_id"]), record)
    return list(chosen.values())


def _metrics(rows):
    graded = [r for r in rows if r["status"] == "graded"]
    binary = [r for r in graded if r["winner_outcome"] is not None]
    picks = [r for r in binary if r["picked_team"] is not None]
    scores = [r for r in graded if r.get("score_errors") is not None]
    result = {
        "predicted": len(rows),
        "graded": len(graded),
        "probability_games": len(binary),
        "picked_games": len(picks),
        "pending": sum(r["status"] == "pending" for r in rows),
        "cancelled": sum(r["status"] == "cancelled" for r in rows),
        "review": sum(r["status"] == "review" for r in rows),
        "ties": sum(r.get("tie", False) for r in graded),
        "wins": sum(r["picked_team"] == r["actual_winner"] for r in picks),
        "losses": sum(r["picked_team"] != r["actual_winner"] for r in picks),
        "accuracy": None,
        "brier": None,
        "log_loss": None,
        "ece": None,
        "calibration": [],
        "score_games": len(scores),
        "score_mae": None,
        "margin_mae": None,
        "total_mae": None,
        "confidence_intervals_95": None,
    }
    if picks:
        result["accuracy"] = result["wins"] / len(picks)
    if binary:
        from predictions.backtest import evaluate_probabilities

        p = np.array([r["home_win_probability"] for r in binary])
        y = np.array([r["winner_outcome"] for r in binary])
        clusters = [stamp(r["kickoff"]).strftime("%Y-%U") for r in binary]
        values = evaluate_probabilities(
            y,
            p,
            clusters=clusters if len(binary) >= 20 and len(set(clusters)) >= 2 else None,
            bootstrap_samples=500,
        )
        result["confidence_intervals_95"] = values["confidence_intervals_95"]
        if result["confidence_intervals_95"] and len(picks) != len(binary):
            result["confidence_intervals_95"]["accuracy"] = None
        result.update({k: values[k] for k in ["brier_score", "log_loss"]})
        result["brier"] = result.pop("brier_score")
        result["calibration"] = values["reliability_bins"]
        result["ece"] = values.get("expected_calibration_error", values.get("ece"))
    if scores:
        for metric in ["score_mae", "margin_mae", "total_mae"]:
            result[metric] = float(np.mean([r["score_errors"][metric] for r in scores]))
    return result


def report_for(journal, sport, *, as_of):
    at = stamp(as_of)
    receipts = ForecastJournal(journal.path).records(as_of=at)
    primary = [r for r in primary_receipts(receipts) if r["sport"] == sport]
    latest = {(r["content"]["sport"], r["content"]["game_id"]): r for r in journal.observations(at)}
    rows = []
    line_rows = []
    for receipt in primary:
        row = {
            k: receipt[k]
            for k in [
                "sport",
                "game_id",
                "forecast_id",
                "model_version",
                "received_at",
                "generated_at",
                "as_of",
                "kickoff",
                "home_team",
                "away_team",
                "home_win_probability",
            ]
        }
        p = receipt["home_win_probability"]
        row["picked_team"] = (
            receipt["home_team"] if p > 0.5 else receipt["away_team"] if p < 0.5 else None
        )
        row.update(
            status="pending",
            actual_winner=None,
            winner_outcome=None,
            score_errors=None,
            actual_home_score=None,
            actual_away_score=None,
        )
        row["projected_score"] = receipt["score"]["summary"]
        row["recorded_line_scenarios"] = receipt["score"]["lines"]
        row["score_evidence_status"] = receipt["score"]["reason"]
        observation = latest.get((sport, receipt["game_id"]))
        if observation:
            outcome = observation["content"]
            row["result_scope"] = outcome.get("outcome_type")
            row["result_observation_id"] = observation["observation_id"]
            row["result_observed_at"] = observation["observed_at"]
            matches = (
                outcome["home_team"] == receipt["home_team"]
                and outcome["away_team"] == receipt["away_team"]
                and stamp(outcome["kickoff"]) == stamp(receipt["kickoff"])
                and stamp(receipt["received_at"]) < stamp(outcome["kickoff"])
            )
            teams_match = (
                outcome["home_team"] == receipt["home_team"]
                and outcome["away_team"] == receipt["away_team"]
            )
            if teams_match and outcome["status"] == "cancelled":
                row["status"] = "cancelled"
            elif teams_match and outcome["status"] == "postponed":
                row["reason"] = "postponed_not_graded"
            elif not matches or outcome["status"] == "disputed":
                row["status"] = "review"
                row["reason"] = "schedule_identity_or_result_requires_review"
            elif outcome["status"] == "cancelled":
                row["status"] = "cancelled"
            elif outcome["status"] == "final":
                h, a = outcome["home_score"], outcome["away_score"]
                tie = h == a
                row.update(
                    status="graded",
                    actual_home_score=h,
                    actual_away_score=a,
                    tie=tie,
                    actual_winner=None
                    if tie
                    else outcome["home_team"]
                    if h > a
                    else outcome["away_team"],
                    winner_outcome=None if tie else int(h > a),
                    actual_margin=h - a,
                    actual_total=h + a,
                )
                summary = receipt["score"]["summary"]
                if summary:
                    row["score_errors"] = {
                        "score_mae": (abs(summary["home_mean"] - h) + abs(summary["away_mean"] - a))
                        / 2,
                        "margin_mae": abs(summary["expected_margin"] - h + a),
                        "total_mae": abs(summary["expected_total"] - h - a),
                    }
                for line in receipt["score"]["lines"]:
                    lh, la = (
                        (outcome["regulation_home_score"], outcome["regulation_away_score"])
                        if line["scope"] == "regulation"
                        else (h, a)
                    )
                    value = (
                        lh - la + line["line"]
                        if line["kind"] == "home_cover"
                        else lh + la - line["line"]
                    )
                    actual = "win" if value > 0 else "loss" if value < 0 else "push"
                    probs = line["probabilities"]
                    line_rows.append(
                        {
                            "kind": line["kind"],
                            "line": line["line"],
                            "scope": line["scope"],
                            "source": line["source"],
                            "game_id": receipt["game_id"],
                            "brier3": sum((v - int(k == actual)) ** 2 for k, v in probs.items()),
                            "log_loss": float(-np.log(max(probs[actual], 1e-15))),
                            "predicted_push": probs["push"],
                            "actual_push": int(actual == "push"),
                            "predicted_win": probs["win"],
                            "actual_win": int(actual == "win"),
                        }
                    )
        rows.append(row)
    models = {
        version: _metrics([r for r in rows if r["model_version"] == version])
        for version in sorted({r["model_version"] for r in rows})
    }
    bands = {}
    for lo in [50, 60, 70, 80, 90]:
        group = [
            r
            for r in rows
            if lo
            <= max(r["home_win_probability"], 1 - r["home_win_probability"]) * 100
            < (lo + 10 if lo < 90 else 101)
        ]
        bands[f"{lo}–{lo + 9 if lo < 90 else 100}%"] = _metrics(group)
    periods = {
        str(days): _metrics(
            [r for r in rows if at - pd.Timedelta(days=days) <= stamp(r["kickoff"]) <= at]
        )
        for days in [7, 30, 90]
    }
    weeks = {}
    for row in rows:
        if row["status"] == "graded":
            week = stamp(row["kickoff"]).strftime("%Y-%U")
            weeks.setdefault(week, []).append(row)
    lines = []
    if line_rows:
        frame = pd.DataFrame(line_rows)
        for key, group in frame.groupby(["kind", "line", "scope", "source"]):
            lines.append(
                {
                    "kind": str(key[0]),
                    "line": float(key[1]),
                    "scope": str(key[2]),
                    "source": str(key[3]),
                }
                | {
                    "games": len(group),
                    **{
                        k: float(group[k].mean())
                        for k in [
                            "brier3",
                            "log_loss",
                            "predicted_push",
                            "actual_push",
                            "predicted_win",
                            "actual_win",
                        ]
                    },
                    "market_status": "PASS; research scenarios, no executable-price/profitability claim",
                }
            )
    for value in lines:
        from predictions.backtest import evaluate_probabilities

        samples = [
            r
            for r in line_rows
            if all(r[k] == value[k] for k in ["kind", "line", "scope", "source"])
        ]
        calibration = evaluate_probabilities(
            [r["actual_win"] for r in samples],
            [r["predicted_win"] for r in samples],
            bootstrap_samples=0,
        )
        value["calibration"] = calibration["reliability_bins"]
        value["ece_win_event"] = calibration["expected_calibration_error"]
    return {
        "type": "prospective_performance",
        "sport": sport,
        "as_of": at.isoformat(),
        "cohort": "first_valid_receipt_per_game",
        "overall": _metrics(rows),
        "by_model_version": models,
        "by_confidence": bands,
        "recent_days": periods,
        "weekly": {week: _metrics(group) for week, group in weeks.items()},
        "recorded_lines": lines,
        "history": rows,
        "retained_receipts": sum(r["sport"] == sport for r in receipts),
        "late_imports": sum(r["sport"] == sport and not r["eligible"] for r in receipts),
        "limitations": [
            "Prospective receipts only; retrospective backtests are separate.",
            "NFL ties excluded from decisive-winner metrics but included in score metrics.",
            "A changed kickoff/opponent or disputed result is not graded automatically.",
            "Research lines are not historical market prices. No betting edge or execution.",
            "Pending/missing outcomes never count as losses.",
        ],
    }


def reconcile(config, sport, *, fetcher=None, clock=now, force=False):
    from predictions.official_results import fetch_results

    root = Path(config["paths"]["state_dir"])
    status_path = root / (sport.lower() + "-result-status.json")
    try:
        previous_status = json.loads(status_path.read_text()) if status_path.is_file() else {}
        if previous_status.get("checked_at"):
            stamp(previous_status["checked_at"])
    except (OSError, ValueError, TypeError):
        previous_status = {}
    journal = OutcomeJournal(root / "prospective.sqlite")
    at = stamp(clock())
    records = primary_receipts(ForecastJournal(journal.path).records(as_of=at))
    observations = journal.observations(at)
    latest = {(r["content"]["sport"], r["content"]["game_id"]): r for r in observations}
    targets = []
    for r in records:
        if r["sport"] != sport or stamp(r["kickoff"]) + pd.Timedelta(hours=24) > at:
            continue
        prior = latest.get((sport, r["game_id"]))
        if (
            force
            or not prior
            or prior["content"]["status"] != "final"
            or stamp(r["kickoff"]) >= at - pd.Timedelta(days=7)
        ):
            targets.append(r)
    with connect(journal.path, readonly=True) as db:
        checks = {
            (r["sport"], r["game_id"]): r["checked_at"]
            for r in db.execute("SELECT * FROM result_checks")
        }
    targets.sort(
        key=lambda r: (
            latest.get((sport, r["game_id"]), {}).get("content", {}).get("status") == "final",
            checks.get((sport, r["game_id"]), ""),
            r["kickoff"],
        )
    )
    due = (
        force
        or not previous_status.get("checked_at")
        or (at - stamp(previous_status["checked_at"])).total_seconds()
        >= (
            config["automation"]["retry_seconds"]
            if previous_status.get("errors")
            else config["automation"]["results_interval_seconds"]
        )
    )
    targets = targets[:100] if due else []
    errors = [] if due else previous_status.get("errors", [])
    statuses = []
    try:
        values = (
            (fetcher or fetch_results)(sport, targets, root / "outcome-sources") if targets else []
        )
        end = stamp(clock())
        requested = {r["game_id"] for r in targets}
        returned = set()
        for value in values:
            if not isinstance(value, dict):
                errors.append("malformed_result")
                continue
            if value.get("sport") != sport or value.get("game_id") not in requested:
                errors.append("unrequested_result")
                continue
            returned.add(value["game_id"])
            try:
                statuses.append(journal.append(value, as_of=end))
            except (ValueError, KeyError, TypeError, OSError):
                errors.append("result_evidence_invalid")
        if requested - returned:
            errors.append("some_official_results_unavailable")
    except Exception:
        LOG.exception("%s official results unavailable; verified prior outcomes retained", sport)
        errors.append("official_provider_unavailable")
        end = stamp(clock())
    with connect(journal.path) as db:
        for target in targets:
            db.execute(
                "INSERT INTO result_checks VALUES(?,?,?,?) ON CONFLICT(sport,game_id) DO UPDATE SET checked_at=excluded.checked_at,error_code=excluded.error_code",
                (sport, target["game_id"], end.isoformat(), errors[0] if errors else None),
            )
        db.commit()
    if due:
        _atomic_write(
            status_path,
            (
                encode(
                    {
                        "sport": sport,
                        "checked_at": end.isoformat(),
                        "errors": errors,
                        "last_success_at": previous_status.get("last_success_at")
                        if errors
                        else end.isoformat(),
                    }
                )
                + "\n"
            ).encode(),
        )
    report = report_for(journal, sport, as_of=end)
    summary = {k: v for k, v in report.items() if k != "history"}
    identifier = digest(summary)
    with connect(journal.path) as db:
        if not db.execute(
            "SELECT 1 FROM prospective_metrics WHERE snapshot_id=?", (identifier,)
        ).fetchone():
            db.execute(
                "INSERT INTO prospective_metrics VALUES(?,?,?,?,?)",
                (identifier, sport, end.isoformat(), encode(summary), digest(summary)),
            )
        db.commit()
    report["reconciliation"] = {
        "status": "partial" if errors else "completed",
        "errors": errors,
        "requested": len(targets),
        "observations": len(statuses),
        "last_checked_at": end.isoformat() if due else previous_status.get("checked_at"),
        "official_fetch_requested": bool(targets),
        "action": "Retained verified outcomes remain. Automatic retry will check missing/disputed games."
        if errors
        else None,
    }
    _atomic_write(root / (sport.lower() + "-prospective.json"), (encode(report) + "\n").encode())
    return report
