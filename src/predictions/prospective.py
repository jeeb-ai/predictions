"""Append-only receipt of real forecasts. A claimed generation time is not a receipt."""

import hashlib
import io
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

SPORTS = ("NFL", "NBA", "NHL")


def now():
    return pd.Timestamp.now(tz="UTC")


def stamp(value):
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is None:
        raise ValueError("Timestamp must include a timezone")
    return result.tz_convert("UTC")


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(encode(value).encode()).hexdigest()


@contextmanager
def connect(path, readonly=False):
    path = Path(path).resolve()
    if readonly:
        db = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=30)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(path, timeout=30)
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA synchronous=FULL")
        db.execute("PRAGMA journal_mode=WAL")
    db.row_factory = sqlite3.Row
    try:
        yield db
    finally:
        db.close()


def immutable(db, table):
    for action in ["UPDATE", "DELETE"]:
        db.execute(
            f"CREATE TRIGGER IF NOT EXISTS {table}_no_{action} BEFORE {action} ON {table} BEGIN SELECT RAISE(ABORT,'append-only history'); END"
        )


def score_evidence(row, sport, received):
    """Freeze exact research lines before play; they are not observed market odds."""
    score = row.get("score_prediction", {}) if sport == "NFL" else row
    ref = score.get("distribution")
    if not ref:
        return {"summary": None, "lines": [], "reason": "score_distribution_unavailable"}
    if sport == "NFL" and max(stamp(score["generated_at"]), stamp(score["as_of"])) > received:
        raise ValueError("Score forecast was not yet observed")

    def read(reference):
        path = Path(reference["path"])
        if path.stat().st_size > 1_000_000:
            raise ValueError("Oversized score distribution")
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != reference["sha256"]:
            raise ValueError("Score distribution checksum mismatch")
        pmf = np.load(io.BytesIO(content), allow_pickle=False)
        if (
            pmf.ndim != 2
            or pmf.shape[0] != pmf.shape[1]
            or len(pmf) > 251
            or not np.isfinite(pmf).all()
            or (pmf < 0).any()
            or not np.isclose(pmf.sum(), 1)
        ):
            raise ValueError("Invalid score distribution")
        return pmf

    from predictions.league_model import line_probabilities, summarize

    pmf = read(ref)
    summary = summarize(pmf)
    original = score.get("summary", score.get("score_summary", {}))
    if any(
        not np.isclose(summary[k], original[k], atol=1e-9)
        for k in ["home_mean", "away_mean", "expected_margin", "expected_total"]
    ):
        raise ValueError("Score summary does not reconcile to retained distribution")
    spread, total = {"NFL": (-3, 44), "NBA": (-5.5, 220.5), "NHL": (-1.5, 5.5)}[sport]
    scopes = [("full_game", pmf, ref["sha256"])]
    if sport == "NHL" and row.get("regulation_distribution"):
        scopes.append(
            (
                "regulation",
                read(row["regulation_distribution"]),
                row["regulation_distribution"]["sha256"],
            )
        )
    lines = []
    for scope, mass, sha in scopes:
        values = line_probabilities(mass, spread, total)
        for kind, line, probs in [
            ("home_cover", spread, values["home_cover"]),
            ("over", total, values["over"]),
        ]:
            lines.append(
                {
                    "kind": kind,
                    "line": line,
                    "scope": scope,
                    "probabilities": probs,
                    "observed_at": received.isoformat(),
                    "source": "fixed_research_scenario",
                    "distribution_sha256": sha,
                    "market_status": "PASS; no observed executable market line",
                }
            )
    return {"summary": summary, "lines": lines, "reason": None}


def normalize(sport, row, received):
    if sport not in SPORTS or row.get("sport", sport) != sport:
        raise ValueError("Unsupported or contradictory sport")
    for key in ["forecast_id", "game_id", "model_version", "home_team", "away_team"]:
        if not isinstance(row.get(key), str) or not row[key].strip():
            raise ValueError("Forecast missing " + key)
    if row["home_team"] == row["away_team"]:
        raise ValueError("Forecast must have distinct opponents")
    if sport == "NFL":
        from predictions.ledger import _validate_forecast

        _validate_forecast(row)
    for field in ["model_sha256"]:
        value = row.get(field)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ValueError("Invalid forecast model digest")
    if sport != "NFL" and row.get("explanation"):
        from predictions.league_forecast import league_verdict

        league_verdict(row)
    ref = row.get("source", {})
    if ref.get("manifest_path"):
        content = Path(ref["manifest_path"]).read_bytes()
        if hashlib.sha256(content).hexdigest() != ref.get("sha256"):
            raise ValueError("Forecast source manifest checksum mismatch")
    received, generated, cutoff, kickoff = map(
        stamp, [received, row["generated_at"], row["as_of"], row["kickoff"]]
    )
    if not cutoff <= generated <= received:
        raise ValueError("Forecast source/generation time is after its receipt")
    if row.get("history_available_through") and stamp(row["history_available_through"]) >= cutoff:
        raise ValueError("Forecast history was not strictly pregame")
    p, q = float(row["home_win_probability"]), float(row["away_win_probability"])
    if (
        not np.isfinite([p, q]).all()
        or not 0 <= p <= 1
        or not np.isclose(p + q, 1, atol=1e-12, rtol=0)
    ):
        raise ValueError("Invalid forecast probabilities")
    scope = row.get("probability_scope")
    if (
        not isinstance(scope, str)
        or (sport == "NFL" and "decisive" not in scope)
        or (sport != "NFL" and "full_game" not in scope)
    ):
        raise ValueError("Unsupported winner probability scope")
    eligible = max(received, generated, cutoff) < kickoff
    score = {"summary": None, "lines": [], "reason": "not_recorded_pregame"}
    if eligible:
        try:
            score = score_evidence(row, sport, received)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            score = {
                "summary": None,
                "lines": [],
                "reason": type(exc).__name__ + ": score evidence could not be verified",
            }
    return {
        "sport": sport,
        "game_id": row["game_id"],
        "forecast_id": row["forecast_id"],
        "model_version": row["model_version"],
        "received_at": received.isoformat(),
        "generated_at": generated.isoformat(),
        "as_of": cutoff.isoformat(),
        "kickoff": kickoff.isoformat(),
        "home_team": row["home_team"],
        "away_team": row["away_team"],
        "probability_scope": scope,
        "home_win_probability": p,
        "eligible": eligible,
        "exclusion": None if eligible else "first_received_after_start",
        "score": score,
        "original": row,
    }


class ForecastJournal:
    def __init__(self, path):
        self.path = Path(path)
        with connect(self.path) as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS receipts (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT, sport TEXT NOT NULL,
                    forecast_id TEXT NOT NULL, game_id TEXT NOT NULL, received_at TEXT NOT NULL,
                    original_sha256 TEXT NOT NULL, payload TEXT NOT NULL, sha256 TEXT NOT NULL,
                    UNIQUE(sport, forecast_id));
                CREATE INDEX IF NOT EXISTS receipts_game ON receipts(sport,game_id,sequence);
                CREATE TABLE IF NOT EXISTS platform_jobs (
                    job_id TEXT PRIMARY KEY, sport TEXT NOT NULL, slot TEXT NOT NULL,
                    status TEXT NOT NULL, started_at TEXT NOT NULL, finished_at TEXT,
                    checkpoint TEXT, checkpoint_sha256 TEXT, error_code TEXT);
            """)
            immutable(db, "receipts")
            db.execute(
                "CREATE TRIGGER IF NOT EXISTS receipts_no_replace BEFORE INSERT ON receipts WHEN EXISTS(SELECT 1 FROM receipts WHERE sport=NEW.sport AND forecast_id=NEW.forecast_id) BEGIN SELECT RAISE(ABORT,'append-only history'); END"
            )
            db.commit()

    def capture(self, sport, forecasts, *, received_at=None, db=None):
        received = stamp(received_at) if received_at is not None else now()
        if db is None:
            with connect(self.path) as connection:
                connection.execute("BEGIN IMMEDIATE")
                result = self.capture(sport, forecasts, received_at=received, db=connection)
                connection.commit()
                return result
        counts = {"inserted": 0, "duplicate": 0, "pregame": 0, "late": 0}
        for row in forecasts:
            original_sha = digest(row)
            previous = db.execute(
                "SELECT original_sha256 FROM receipts WHERE sport=? AND forecast_id=?",
                (sport, row["forecast_id"]),
            ).fetchone()
            if previous:
                if previous[0] != original_sha:
                    raise ValueError("Forecast ID was reused with changed content")
                counts["duplicate"] += 1
                continue
            item = normalize(sport, row, received)
            db.execute(
                "INSERT INTO receipts(sport,forecast_id,game_id,received_at,original_sha256,payload,sha256) VALUES(?,?,?,?,?,?,?)",
                (
                    sport,
                    row["forecast_id"],
                    row["game_id"],
                    received.isoformat(),
                    original_sha,
                    encode(item),
                    digest(item),
                ),
            )
            counts["inserted"] += 1
            counts["pregame" if item["eligible"] else "late"] += 1
        return counts

    def records(self, *, as_of=None):
        cutoff = stamp(as_of) if as_of is not None else now()
        with connect(self.path, readonly=True) as db:
            output = []
            for row in db.execute("SELECT * FROM receipts ORDER BY sequence"):
                item = json.loads(row["payload"])
                if (
                    digest(item) != row["sha256"]
                    or digest(item["original"]) != row["original_sha256"]
                ):
                    raise ValueError("Forecast journal checksum mismatch")
                if stamp(item["received_at"]) <= cutoff:
                    output.append(item)
            return output


def capture_snapshots(config):
    root = Path(config["paths"]["state_dir"])
    journal = ForecastJournal(root / "prospective.sqlite")
    results = {}
    for sport in SPORTS:
        path = root / (
            "predictions.json" if sport == "NFL" else sport.lower() + "-predictions.json"
        )
        try:
            if path.is_file():
                if path.stat().st_size > 32 * 1024 * 1024:
                    raise ValueError("Oversized forecast snapshot")
                results[sport] = journal.capture(
                    sport, json.loads(path.read_text()).get("forecasts", [])
                )
            else:
                results[sport] = {"status": "no_snapshot"}
        except (OSError, ValueError, KeyError, TypeError):
            results[sport] = {
                "status": "invalid_snapshot",
                "action": "Refresh this sport; other sports are retained.",
            }
    return results
