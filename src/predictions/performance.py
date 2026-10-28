"""Prospective forecast scores and evidenced paper results, never invented returns.

The first issued, ledger-recorded pregame forecast per game/model version defines
the cohort. Outcomes and metric snapshots append rather than replace history.
All paper-ledger access is read-only, including its existing integrity audit.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing, contextmanager
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from predictions.backtest import evaluate_probabilities
from predictions.ledger import PaperLedger, _validate_forecast


def _time(value, field: str) -> pd.Timestamp:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a timezone-aware timestamp") from exc
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError(f"{field} must be a timezone-aware timestamp")
    return stamp.tz_convert("UTC")


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


class _ReadOnlyLedger(PaperLedger):
    """Reuse the journal's audit without creating tables, triggers, or files."""

    def __init__(self, path: Path):
        self.path = path.resolve()

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
        finally:
            connection.close()


def _ledger_records(path: Path, as_of: pd.Timestamp) -> tuple[dict, dict]:
    empty = {"forecasts": [], "trades": [], "settlements": []}
    if not path.exists():
        return empty, {"valid": True, "status": "ledger_not_created", "events": 0}
    audit = _ReadOnlyLedger(path).audit()
    if not audit["valid"]:
        raise ValueError("Cannot track an invalid paper ledger: " + "; ".join(audit["errors"]))
    records = {}
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        for kind, table in [
            ("forecasts", "paper_forecasts"),
            ("trades", "paper_trades"),
            ("settlements", "paper_settlements"),
        ]:
            records[kind] = [
                json.loads(row[0])
                for row in connection.execute(f"SELECT payload_json FROM {table} ORDER BY rowid")
            ]
        recorded = {
            row[0]: _time(row[1], "forecast recorded_at")
            for row in connection.execute(
                "SELECT entity_id, recorded_at FROM ledger_events WHERE event_type='forecast'"
            )
        }
    for forecast in records["forecasts"]:
        stamp = recorded.get(forecast["forecast_id"])
        forecast["_ledger_recorded_at"] = stamp.isoformat() if stamp is not None else None
    records["trades"] = [
        row
        for row in records["trades"]
        if _time(row["recorded_at"], "paper fill recorded_at") <= as_of
    ]
    records["settlements"] = [
        row
        for row in records["settlements"]
        if _time(row["recorded_at"], "settlement recorded_at") <= as_of
        and _time(row["settled_at"], "settled_at") <= as_of
        and _time(row["evidence"]["observed_at"], "settlement evidence observed_at") <= as_of
    ]
    return records, {
        "valid": True,
        "status": "audited",
        "head_hash": audit["head_hash"],
        "events": audit["events"],
    }


def _select_forecasts(forecasts: list[dict], as_of: pd.Timestamp) -> tuple[list[dict], list[dict]]:
    valid, rejected = [], []
    for row in forecasts:
        try:
            _validate_forecast(row)
            if not isinstance(row.get("season"), int) or isinstance(row["season"], bool):
                raise ValueError("forecast_requires_integer_season")
            cutoff = _time(row["as_of"], "forecast as_of")
            generated = _time(row.get("generated_at"), "forecast generated_at")
            recorded = _time(row.get("_ledger_recorded_at"), "forecast ledger recorded_at")
            kickoff = _time(row["kickoff"], "forecast kickoff")
            if max(cutoff, generated, recorded) > as_of:
                raise ValueError("forecast_not_yet_observed")
            if generated >= kickoff or recorded >= kickoff:
                raise ValueError("forecast_not_issued_and_recorded_pregame")
            valid.append(row)
        except (ValueError, KeyError) as exc:
            rejected.append({"forecast_id": row.get("forecast_id"), "reason": str(exc)})
    valid.sort(
        key=lambda row: (
            _time(row["generated_at"], "generated_at"),
            _time(row["as_of"], "as_of"),
            row["forecast_id"],
        )
    )
    chosen = {}
    for row in valid:
        chosen.setdefault((row["game_id"], row["model_version"]), row)
    return list(chosen.values()), rejected


def _source(metadata: dict, as_of: pd.Timestamp) -> dict:
    if not isinstance(metadata, dict):
        raise ValueError("Outcome source_metadata must be an object")
    source = metadata.get("schedule", metadata)
    if not isinstance(source, dict):
        raise ValueError("Outcome schedule source must be an object")
    digest = source.get("sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(char not in "0123456789abcdef" for char in digest)
    ):
        raise ValueError("Outcome source requires a SHA-256 digest")
    fetched = _time(source.get("fetched_at"), "outcome source fetched_at")
    if fetched > as_of:
        raise ValueError("Outcome source was observed after performance as_of")
    return {**source, "fetched_at": fetched.isoformat()}


def _games(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "game_id",
        "season",
        "week",
        "kickoff",
        "result_available_at",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
    }
    if required - set(frame):
        raise ValueError(f"Performance schedule lacks columns: {sorted(required - set(frame))}")
    result = frame.copy()
    if result["game_id"].isna().any() or result["game_id"].duplicated().any():
        raise ValueError("Performance schedule requires unique nonmissing game IDs")
    for column in ["kickoff", "result_available_at"]:
        result[column] = pd.to_datetime(result[column], utc=True)
        if result[column].isna().any():
            raise ValueError(f"Performance schedule has missing {column}")
    if (result["result_available_at"] <= result["kickoff"]).any():
        raise ValueError("Outcome availability must follow kickoff")
    for column in ["home_score", "away_score"]:
        result[column] = pd.to_numeric(result[column], errors="raise")
        known = result[column].dropna()
        if not np.isfinite(known).all() or (known < 0).any() or (known != np.floor(known)).any():
            raise ValueError("Performance outcomes require nonnegative integer scores")
    if result["home_score"].isna().ne(result["away_score"].isna()).any():
        raise ValueError("Performance outcomes require both scores or neither")
    return result


def _matches(forecast: dict, game: dict) -> bool:
    return (
        forecast["home_team"] == game["home_team"]
        and forecast["away_team"] == game["away_team"]
        and _time(forecast["kickoff"], "forecast kickoff") == _time(game["kickoff"], "kickoff")
        and int(forecast["season"]) == int(game["season"])
    )


def _paper(records: dict) -> dict:
    trades = {row["paper_trade_id"]: row for row in records["trades"]}
    settlements = sorted(
        [row for row in records["settlements"] if row["paper_trade_id"] in trades],
        key=lambda row: _time(row["recorded_at"], "settlement recorded_at"),
    )
    settled = {row["paper_trade_id"] for row in settlements}
    cumulative = peak = maximum_drawdown = Decimal("0")
    timeline = []
    for row in settlements:
        trade = trades[row["paper_trade_id"]]
        profit = Decimal(row["realized_paper_profit"])
        cumulative += profit
        peak = max(peak, cumulative)
        drawdown = peak - cumulative
        maximum_drawdown = max(maximum_drawdown, drawdown)
        timeline.append(
            {
                "settlement_id": row["settlement_id"],
                "paper_trade_id": row["paper_trade_id"],
                "game_id": trade["game_id"],
                "market_id": trade["market_id"],
                "outcome": trade["outcome"],
                "quantity": trade["quantity"],
                "payout": row["payout"],
                "fee": row["fee"],
                "realized_paper_profit": str(profit),
                "recorded_at": row["recorded_at"],
                "evidence_verification": row.get("evidence_verification"),
                "cumulative_paper_profit": str(cumulative),
                "drawdown": str(drawdown),
            }
        )
    fields = [
        "paper_trade_id",
        "game_id",
        "market_id",
        "outcome",
        "selected_team",
        "quantity",
        "notional",
        "fee",
        "total_cost",
        "recorded_at",
    ]
    open_rows = [
        {field: row.get(field) for field in fields}
        for key, row in trades.items()
        if key not in settled
    ]
    return {
        "mode": "paper_only",
        "open_positions": len(open_rows),
        "open_exposure": str(sum((Decimal(row["total_cost"]) for row in open_rows), Decimal(0))),
        "realized_paper_profit": str(cumulative),
        "max_drawdown": str(maximum_drawdown),
        "open_position_rows": open_rows,
        "settlement_rows": timeline,
        "settlement_timeline": timeline,
        "settled_positions": len(settlements),
        "fee_kind": "estimated_upper_bound",
        "mark_to_market": None,
    }


class PerformanceTracker:
    """Append observed outcomes and immutable reports in a separate local database."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS performance_outcomes (
                    observation_id TEXT PRIMARY KEY, game_id TEXT NOT NULL,
                    observed_at TEXT NOT NULL, source_fetched_at TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL, payload_json TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS performance_reports (
                    report_id TEXT PRIMARY KEY, as_of TEXT NOT NULL,
                    payload_json TEXT NOT NULL, payload_sha256 TEXT NOT NULL);
            """)
            for table, identity in [
                ("performance_outcomes", "observation_id"),
                ("performance_reports", "report_id"),
            ]:
                for action in ["UPDATE", "DELETE"]:
                    connection.execute(
                        f"CREATE TRIGGER IF NOT EXISTS {table}_no_{action.lower()} BEFORE {action} "
                        f"ON {table} BEGIN SELECT RAISE(ABORT, 'performance history is append-only'); END"
                    )
                connection.execute(
                    f"CREATE TRIGGER IF NOT EXISTS {table}_no_replace BEFORE INSERT ON {table} "
                    f"WHEN EXISTS (SELECT 1 FROM {table} WHERE {identity}=NEW.{identity}) "
                    "BEGIN SELECT RAISE(ABORT, 'performance history is append-only'); END"
                )
            connection.commit()

    @staticmethod
    def _observations(connection, as_of: pd.Timestamp) -> list[dict]:
        rows = []
        for text, digest in connection.execute(
            "SELECT payload_json,payload_sha256 FROM performance_outcomes ORDER BY rowid"
        ):
            payload = json.loads(text)
            if _hash(payload) != digest:
                raise ValueError("Performance outcome checksum mismatch")
            if _time(payload["observed_at"], "outcome observed_at") <= as_of:
                rows.append(payload)
        # Preserve append order when observations share an as-of timestamp.
        return sorted(rows, key=lambda row: row["observed_at"])

    def update(
        self, ledger_path: Path | str, games: pd.DataFrame, *, as_of, source_metadata: dict
    ) -> dict:
        cutoff = _time(as_of, "performance as_of")
        source = _source(source_metadata, cutoff)
        schedule = _games(games)
        records, audit = _ledger_records(Path(ledger_path), cutoff)
        selected, excluded = _select_forecasts(records["forecasts"], cutoff)
        lookup = {row["game_id"]: row for row in schedule.to_dict(orient="records")}
        referenced = {row["game_id"] for row in selected}
        fetched = _time(source["fetched_at"], "source fetched_at")
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                observations = self._observations(connection, cutoff)
                latest = {row["game_id"]: row for row in observations}
                for identity in sorted(referenced):
                    game = lookup.get(identity)
                    if game is None or not any(
                        _matches(row, game) for row in selected if row["game_id"] == identity
                    ):
                        continue
                    if game["kickoff"] >= cutoff or game["result_available_at"] >= min(
                        cutoff, fetched
                    ):
                        continue
                    previous = latest.get(identity)
                    if (
                        previous is not None
                        and _time(previous["source_fetched_at"], "previous source") > fetched
                    ):
                        continue  # A stale cached source cannot undo a later observed correction.
                    completed = pd.notna(game["home_score"]) and pd.notna(game["away_score"])
                    content = {
                        "game_id": identity,
                        "season": int(game["season"]),
                        "week": int(game["week"]),
                        "home_team": game["home_team"],
                        "away_team": game["away_team"],
                        "kickoff": game["kickoff"].isoformat(),
                        "result_available_at": game["result_available_at"].isoformat(),
                        "home_score": int(game["home_score"]) if completed else None,
                        "away_score": int(game["away_score"]) if completed else None,
                        "status": "completed" if completed else "unresolved",
                    }
                    digest = _hash(content)
                    if previous is not None and previous["content_sha256"] == digest:
                        continue
                    payload = {
                        **content,
                        "content_sha256": digest,
                        "observed_at": cutoff.isoformat(),
                        "source_sha256": source["sha256"],
                        "source_fetched_at": source["fetched_at"],
                        "source_url": source.get("source_url"),
                    }
                    payload["observation_id"] = "outcome-" + _hash(payload)
                    connection.execute(
                        "INSERT INTO performance_outcomes VALUES (?,?,?,?,?,?,?)",
                        (
                            payload["observation_id"],
                            identity,
                            payload["observed_at"],
                            fetched.isoformat(),
                            digest,
                            _json(payload),
                            _hash(payload),
                        ),
                    )
                    observations.append(payload)
                    latest[identity] = payload
                report = self._report(
                    selected, excluded, latest, observations, records, audit, cutoff, source, lookup
                )
                text = _json(report)
                existing = connection.execute(
                    "SELECT payload_json FROM performance_reports WHERE report_id=?",
                    (report["report_id"],),
                ).fetchone()
                if existing is None:
                    connection.execute(
                        "INSERT INTO performance_reports VALUES (?,?,?,?)",
                        (report["report_id"], cutoff.isoformat(), text, _hash(report)),
                    )
                elif existing[0] != text:
                    raise ValueError("Performance report ID has conflicting content")
                connection.commit()
                return report
            except BaseException:
                connection.rollback()
                raise

    @staticmethod
    def _report(
        selected, excluded, latest, observations, records, audit, cutoff, source, schedule
    ) -> dict:
        groups = {}
        for forecast in selected:
            groups.setdefault(forecast["model_version"], []).append(forecast)
        models = {}
        for version, forecasts in groups.items():
            scored, pending, tied, mismatched = [], 0, 0, 0
            for forecast in forecasts:
                current = schedule.get(forecast["game_id"])
                if current is None or not _matches(forecast, current):
                    pending += 1
                    mismatched += 1
                    continue
                outcome = latest.get(forecast["game_id"])
                if outcome is None or not _matches(forecast, outcome):
                    pending += 1
                    mismatched += int(outcome is not None)
                    continue
                if outcome["status"] != "completed":
                    pending += 1
                elif outcome["home_score"] == outcome["away_score"]:
                    tied += 1
                else:
                    scored.append(
                        {
                            "forecast_id": forecast["forecast_id"],
                            "model_id": forecast["model_id"],
                            "game_id": forecast["game_id"],
                            "season": outcome["season"],
                            "week": outcome["week"],
                            "kickoff": outcome["kickoff"],
                            "forecast_as_of": forecast["as_of"],
                            "generated_at": forecast["generated_at"],
                            "home_win_probability": forecast["home_win_probability"],
                            "home_win": int(outcome["home_score"] > outcome["away_score"]),
                            "outcome_observation_id": outcome["observation_id"],
                        }
                    )
            metrics = None
            if scored:
                clusters = [f"{row['season']}:{row['week']}" for row in scored]
                block_count = len(set(clusters))
                metrics = evaluate_probabilities(
                    [row["home_win"] for row in scored],
                    [row["home_win_probability"] for row in scored],
                    clusters=clusters,
                    bootstrap_samples=1000 if block_count >= 2 else 0,
                    seed=42,
                )
                metrics["bootstrap_clusters"] = block_count
                metrics["uncertainty_status"] = (
                    "conditional_week_block_bootstrap"
                    if block_count >= 2
                    else "insufficient_observed_blocks"
                )
            models[version] = {
                "n": len(scored),
                "status": "ok" if scored else "no_scored_games",
                "metrics": metrics,
                "pending": pending,
                "ties": tied,
                "identity_mismatches": mismatched,
                "selected_forecast_ids": [row["forecast_id"] for row in forecasts],
                "selected_games": len(forecasts),
                "scored_games": scored,
            }
        per_game_counts = {}
        for row in observations:
            per_game_counts[row["game_id"]] = per_game_counts.get(row["game_id"], 0) + 1
        report = {
            "schema_version": 1,
            "kind": "prospective_performance",
            "as_of": cutoff.isoformat(),
            "status": "ok" if any(group["n"] for group in models.values()) else "no_scored_games",
            "models": models,
            "outcomes": {
                "known_games": sum(row["status"] == "completed" for row in latest.values()),
                "observation_count": len(observations),
                "correction_count": sum(max(0, count - 1) for count in per_game_counts.values()),
                "source_sha256": source["sha256"],
                "source_fetched_at": source["fetched_at"],
            },
            "paper": _paper(records),
            "excluded_forecasts": excluded,
            "ledger_audit": audit,
            "selection_policy": "First generated_at, then as_of and forecast_id per game/model version; generation, cutoff, and ledger recording must precede kickoff",
            "methodology": [
                "Prospective ledger forecasts only; repeated polling cannot increase the selected game count.",
                "Binary metrics exclude tied and unresolved games; probabilities concern decisive results.",
                "Outcome source observations must be known by as_of, with completed-result availability strictly earlier than both as_of and source fetch time.",
                "Source corrections append observations and new immutable metric snapshots; old reports are retained.",
                "95% percentile intervals resample season/week blocks with fixed forecasts (1000 draws, seed 42); they omit refitting, source revisions, and cross-week dependence.",
                "Fewer than two observed season/week blocks have no bootstrap interval; a single block cannot establish statistical uncertainty from resampling itself.",
                "Paper PnL uses evidenced settlements and estimated fee upper bounds; it is not real-money performance or marked-to-market value.",
            ],
        }
        report["report_id"] = "performance-" + _hash(report)
        return report

    def latest_report(self) -> dict | None:
        with closing(
            sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True)
        ) as connection:
            row = connection.execute(
                "SELECT payload_json,payload_sha256 FROM performance_reports ORDER BY as_of DESC,rowid DESC LIMIT 1"
            ).fetchone()
        if row is None:
            return None
        payload = json.loads(row[0])
        if _hash(payload) != row[1]:
            raise ValueError("Performance report checksum mismatch")
        return payload
