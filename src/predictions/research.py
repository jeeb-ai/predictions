"""Durable scheduled research using public reads and the local paper ledger.

A process lock prevents overlapping cycles. UTC slots are idempotent after a
successful completion; failed/partial attempts may retry, reusing still-fresh
forecast checkpoints while rereading quotes. Committed ledger facts survive a
crash independently of operational job state. No real execution API is used.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import logging
import signal
import sqlite3
import threading
import uuid
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from predictions.data import NflverseProvider, _atomic_write, load_games
from predictions.ev import evaluate_expected_value
from predictions.ledger import PaperLedger, _verify_settlement_evidence
from predictions.markets import PolymarketUSProvider, _team

LOGGER = logging.getLogger(__name__)


def _now() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc))


def _timestamp(value) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is None:
        raise ValueError("Research timestamps must include a timezone")
    return result.tz_convert("UTC")


def _encode(value) -> bytes:
    return (json.dumps(value, sort_keys=True, allow_nan=False, indent=2) + "\n").encode()


def _read_json(path: Path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


@contextmanager
def _connection(path: Path):
    connection = sqlite3.connect(path, timeout=30)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
    finally:
        connection.close()


def _initialize_state(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "runs").mkdir(exist_ok=True)
    (root / "checkpoints").mkdir(exist_ok=True)
    with _connection(root / "research.sqlite") as connection:
        connection.executescript(
            "CREATE TABLE IF NOT EXISTS jobs (job_key TEXT PRIMARY KEY, config_hash TEXT NOT NULL, "
            "slot TEXT NOT NULL, status TEXT NOT NULL, last_run_id TEXT, report_path TEXT);"
            "CREATE TABLE IF NOT EXISTS attempts (run_id TEXT PRIMARY KEY, job_key TEXT NOT NULL, "
            "started_at TEXT NOT NULL, finished_at TEXT, status TEXT NOT NULL, report_path TEXT);"
        )


def _config_hash(config: dict) -> str:
    if config.get("config_hash"):
        return str(config["config_hash"])
    return hashlib.sha256(_encode(config)).hexdigest()


def _open_positions(ledger: PaperLedger) -> list[dict]:
    with closing(
        sqlite3.connect(ledger.path.resolve().as_uri() + "?mode=ro", uri=True)
    ) as connection:
        rows = connection.execute(
            "SELECT t.payload_json FROM paper_trades t LEFT JOIN paper_settlements s "
            "ON t.paper_trade_id=s.paper_trade_id WHERE s.paper_trade_id IS NULL"
        ).fetchall()
    return [json.loads(row[0]) for row in rows]


class _PublicServices:
    """Adapters are separate so tests can substitute public data boundaries."""

    def __init__(self, config: dict):
        self.config = config
        self.provider = PolymarketUSProvider(Path(config["paths"]["market_cache"]))
        self.schedule_path = None
        self.schedule_metadata = None

    def fetch_schedule(self):
        path, metadata = NflverseProvider(Path(self.config["paths"]["schedule_cache"])).fetch(
            refresh=True
        )
        self.schedule_path = path
        self.schedule_metadata = metadata
        cutoff = min(_now(), _timestamp(metadata["fetched_at"]))
        return load_games(path, as_of=cutoff), metadata

    def forecast(self, game_ids: list[str], config: dict):
        from predictions.live_cli import _forecast

        paths = config["paths"]
        return _forecast(
            SimpleNamespace(
                game_id=game_ids,
                model_version=config["model_version"],
                bundle=None,
                include_analysis=True,
                input=self.schedule_path,
                schedule_metadata=self.schedule_metadata,
                cache_dir=Path(paths["schedule_cache"]),
                pbp_dir=Path(paths["pbp_dir"]),
                ep_dir=Path(paths["ep_dir"]),
                model_dir=Path(paths["model_dir"]),
                output_dir=Path(paths["reports_dir"]) / "forecasts",
                window=config.get("window", 8),
                workers=config["workers"],
                as_of=None,
                offline=False,
                ledger=Path(paths["ledger"]),
                bootstrap_samples=config["bootstrap_samples"],
                confidence_level=config["confidence_level"],
                bootstrap_seed=config["bootstrap_seed"],
            )
        )

    def list_markets(self):
        markets = []
        for offset in range(0, 1000, 100):
            page, event_count = self.provider.list_nfl_market_page(limit=100, offset=offset)
            markets.extend(page)
            # The endpoint paginates events, while this helper returns winner
            # markets. Only an empty raw event page proves traversal is done;
            # a page with no matching winner markets must not hide later games.
            if event_count == 0:
                return markets
        raise ValueError("NFL market discovery exceeded the supported pagination bound")

    def quote(self, slug: str, outcome: str):
        return self.provider.get_quote(slug, outcome)

    def settlement(self, slug: str):
        return self.provider.get_settlement(slug)

    def performance(self, games, source, as_of):
        from predictions.performance import PerformanceTracker

        return PerformanceTracker(Path(self.config["paths"]["performance_db"])).update(
            Path(self.config["paths"]["ledger"]),
            games,
            as_of=as_of,
            source_metadata=source,
        )


def _service(services, name):
    return services[name] if isinstance(services, dict) else getattr(services, name)


def _clock(services, fixed):
    try:
        provider = _service(services, "clock")
    except (KeyError, AttributeError):
        return (lambda: fixed) if fixed is not None else _now
    return lambda: _timestamp(provider())


def _market_matches(market: dict, forecast: dict) -> bool:
    if market.get("sportsMarketType") != "football_team_full_game_winner":
        return False
    sides = market.get("marketSides")
    if not isinstance(sides, list) or len(sides) != 2:
        return False
    try:
        return {_team(side) for side in sides} == {
            forecast["home_team"],
            forecast["away_team"],
        } and _timestamp(market.get("gameStartTime")) == _timestamp(forecast["kickoff"])
    except (ValueError, KeyError, TypeError):
        return False


def _midpoint(value):
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("Probability assumption ranges need lower and upper bounds")
    lower, upper = (Decimal(str(item)) for item in value)
    if not lower.is_finite() or not upper.is_finite() or not 0 <= lower <= upper <= 1:
        raise ValueError("Probability assumption bounds must satisfy 0 <= lower <= upper <= 1")
    return str((lower + upper) / 2)


def _fresh_checkpoint(batch, game_ids, clock, max_age):
    if not isinstance(batch, dict) or not isinstance(batch.get("forecasts"), list):
        return None
    forecasts = [record for record in batch["forecasts"] if record.get("game_id") in game_ids]
    if {record["game_id"] for record in forecasts} != set(game_ids):
        return None
    try:
        now = clock()
        if any(
            _timestamp(record["kickoff"]) <= now
            or _timestamp(record["as_of"]) > now
            or (now - _timestamp(record["as_of"])).total_seconds() > max_age
            for record in forecasts
        ):
            return None
    except (ValueError, KeyError, TypeError):
        return None
    return {**batch, "forecasts": forecasts}


def _execute(config, services, clock, report, checkpoint_path):
    paper = config["paper"]
    ledger = PaperLedger(Path(config["paths"]["ledger"]))
    audit = ledger.audit()
    if not audit["valid"]:
        raise ValueError("Existing paper ledger audit failed; research writes are paused")
    report["stages"]["initial_audit"] = "ok"

    def error(stage, exc, **context):
        report["errors"].append(
            {"stage": stage, "type": type(exc).__name__, "message": str(exc), **context}
        )

    games = source = None
    try:
        games, source = _service(services, "fetch_schedule")()
        report["stages"]["schedule"] = "ok"
    except Exception as exc:
        error("schedule", exc)

    checkpoint = _read_json(checkpoint_path, {"batches": {}})
    if games is not None:
        start = clock()
        end = start + pd.Timedelta(hours=config["lookahead_hours"])
        eligible = games.loc[(games["kickoff"] > start) & (games["kickoff"] <= end)].copy()
        if "kickoff_time_imputed" in eligible:
            eligible = eligible.loc[~eligible["kickoff_time_imputed"].astype(bool)]
        report["slate"] = [
            {
                "game_id": str(row.game_id),
                "home_team": str(row.home_team),
                "away_team": str(row.away_team),
                "kickoff": row.kickoff.isoformat(),
            }
            for row in eligible.itertuples()
        ]
        root = Path(config["paths"]["state_dir"])
        prior = _read_json(root / "predictions.json", {}).get("forecasts", [])
        slate_by_id = {row["game_id"]: row for row in report["slate"]}
        retained = {
            row["game_id"]: row
            for row in prior
            if row["game_id"] in slate_by_id
            and all(
                row.get(key) == slate_by_id[row["game_id"]][key]
                for key in ["home_team", "away_team", "kickoff"]
            )
        }

        def publish_predictions():
            retained.update({row["game_id"]: row for row in report["forecasts"]})
            snapshot = {
                "run_id": report["run_id"],
                "as_of": clock().isoformat(),
                "forecasts": list(retained.values()),
                "slate": report["slate"],
                "errors": [item for item in report["errors"] if item["stage"] == "forecast"],
            }
            _atomic_write(root / "predictions.json", _encode(snapshot))

        publish_predictions()
        for season in sorted(eligible["season"].unique()):
            ids = eligible.loc[eligible["season"].eq(season), "game_id"].astype(str).tolist()
            try:
                batch = _fresh_checkpoint(
                    checkpoint.get("batches", {}).get(str(season)),
                    ids,
                    clock,
                    paper["max_forecast_age_seconds"],
                )
                if batch is None:
                    batch = _service(services, "forecast")(ids, config)
                    returned = [item["game_id"] for item in batch["forecasts"]]
                    skipped = [item["game_id"] for item in batch.get("skipped", [])]
                    if (
                        len(set(returned)) != len(returned)
                        or set(returned) & set(skipped)
                        or set(returned) | set(skipped) != set(ids)
                    ):
                        raise ValueError("Forecast service omitted or changed requested game IDs")
                    checkpoint.setdefault("batches", {})[str(season)] = batch
                    _atomic_write(checkpoint_path, _encode(checkpoint))
                for forecast in batch["forecasts"]:
                    if forecast["game_id"] not in ids:
                        raise ValueError("Forecast service returned an unrequested game")
                    ledger.record_forecast(forecast)
                    report["forecasts"].append(forecast)
                if batch.get("bundle_path"):
                    report.setdefault("model_bundles", {})[str(season)] = batch["bundle_path"]
            except Exception as exc:
                error("forecast", exc, season=int(season))
        report["stages"]["forecasts"] = "checked"
        # Publish independent predictions before any market request can block/fail.
        publish_predictions()

    # Market settlement reads follow independent prediction publication.
    for trade in _open_positions(ledger):
        identity = trade["paper_trade_id"]
        if _timestamp(trade["kickoff"]) > clock():
            report["settlements"].append({"paper_trade_id": identity, "status": "pending"})
            continue
        try:
            evidence = _service(services, "settlement")(trade["quote"]["market_slug"])
            if evidence.get("market_status") != "resolved":
                report["settlements"].append({"paper_trade_id": identity, "status": "pending"})
                continue
            evidence = copy.deepcopy(evidence)
            evidence["reference"] = evidence["source"]["settlement"]["url"]
            price = evidence[f"settlement_price_{trade['outcome']}"]
            verification = _verify_settlement_evidence(evidence, trade, Decimal(str(price)))
            if verification != "public_us_settlement_snapshot":
                raise ValueError("Automatic settlement requires verified raw US provider snapshots")
            settled = ledger.settle(
                identity,
                settlement_price=price,
                settled_at=evidence["observed_at"],
                evidence=evidence,
                idempotency_key=f"auto-settle:{identity}",
            )
            report["settlements"].append({**settled, "status": "settled"})
        except ValueError as exc:
            if "finally resolved" in str(exc):
                report["settlements"].append({"paper_trade_id": identity, "status": "pending"})
            else:
                error("settlement", exc, paper_trade_id=identity)
        except Exception as exc:
            error("settlement", exc, paper_trade_id=identity)
    report["stages"]["settlements"] = "checked"

    markets = []
    if report["forecasts"]:
        try:
            markets = _service(services, "list_markets")()
            report["stages"]["markets"] = "ok"
        except Exception as exc:
            error("markets", exc)
    assumptions = paper["assumptions"]
    for forecast in report["forecasts"]:
        matching = [market for market in markets if _market_matches(market, forecast)]
        if not matching:
            report["evaluations"].append(
                {
                    "game_id": forecast["game_id"],
                    "status": "pass",
                    "reasons": ["no_matching_us_market"],
                }
            )
        candidates = []
        for market in matching:
            for outcome in ["yes", "no"]:
                try:
                    quote = _service(services, "quote")(market["slug"], outcome)
                    ledger.record_quote(quote)
                    selected = quote["yes_team"] if outcome == "yes" else quote["no_team"]
                    cancellation_range = assumptions.get("cancellation_settlement_price_range")
                    if outcome == "no" and cancellation_range is not None:
                        # Scheduled policy defines the YES LFMP. The binary NO
                        # contract pays its complement under that same event.
                        low, high = (Decimal(str(value)) for value in cancellation_range)
                        cancellation_range = [str(1 - high), str(1 - low)]
                    evaluation = evaluate_expected_value(
                        forecast,
                        quote,
                        selected_team=selected,
                        quantity=paper["quantity"],
                        as_of=clock(),
                        tie_probability=_midpoint(assumptions.get("tie_probability_range")),
                        completion_probability=_midpoint(
                            assumptions.get("completion_probability_range")
                        ),
                        cancellation_settlement_price=_midpoint(cancellation_range),
                        max_quote_age_seconds=paper["max_quote_age_seconds"],
                        max_forecast_age_seconds=paper["max_forecast_age_seconds"],
                        min_ev_per_contract=paper["min_ev_per_contract"],
                        require_uncertainty=True,
                        tie_probability_range=assumptions.get("tie_probability_range"),
                        completion_probability_range=assumptions.get(
                            "completion_probability_range"
                        ),
                        cancellation_settlement_price_range=cancellation_range,
                    )
                    ledger.record_evaluation(evaluation)
                    report["evaluations"].append(evaluation)
                    if evaluation["status"] == "paper_candidate":
                        candidates.append(evaluation)
                except Exception as exc:
                    error("market_evaluation", exc, game_id=forecast["game_id"], outcome=outcome)
        if paper["enabled"] and candidates:
            existing = [
                position
                for position in _open_positions(ledger)
                if position["game_id"] == forecast["game_id"]
            ]
            if len(existing) >= paper["max_open_positions_per_game"]:
                report["paper_fills"].append(
                    {
                        "game_id": forecast["game_id"],
                        "status": "not_filled",
                        "reason": "existing_open_position",
                    }
                )
                continue
            prior_keys = {position["idempotency_key"] for position in existing}

            def candidate_key(item):
                return f"scheduled:{item['game_id']}:{forecast['model_version']}:{item['market_id']}:{item['outcome']}"

            candidates = [item for item in candidates if candidate_key(item) not in prior_keys]
            if not candidates:
                report["paper_fills"].append(
                    {
                        "game_id": forecast["game_id"],
                        "status": "not_filled",
                        "reason": "existing_scheduled_position",
                    }
                )
                continue
            best = max(
                candidates, key=lambda item: Decimal(item["uncertainty"]["ev_per_contract_lower"])
            )
            try:
                filled = ledger.paper_fill(
                    best["evaluation_id"],
                    idempotency_key=candidate_key(best),
                    max_exposure=paper["max_open_exposure"],
                    max_open_positions_per_game=paper["max_open_positions_per_game"],
                )
                report["paper_fills"].append({**filled, "status": "filled"})
            except ValueError as exc:
                policy = [
                    "already",
                    "stale",
                    "exposure",
                    "started",
                    "open position",
                    "position limit",
                ]
                if any(term in str(exc).lower() for term in policy):
                    report["paper_fills"].append(
                        {"game_id": forecast["game_id"], "status": "not_filled", "reason": str(exc)}
                    )
                else:
                    error("paper_fill", exc, game_id=forecast["game_id"])
            except Exception as exc:
                error("paper_fill", exc, game_id=forecast["game_id"])
    report["stages"]["evaluations"] = "checked"
    if games is not None:
        try:
            report["performance"] = _service(services, "performance")(games, source, clock())
            report["stages"]["performance"] = "ok"
        except Exception as exc:
            error("performance", exc)
    report["paper_summary"] = ledger.summary()
    report["ledger_audit"] = ledger.audit()
    if not report["ledger_audit"]["valid"]:
        error("final_audit", ValueError("Paper ledger audit failed after the cycle"))


def run_research_cycle(config: dict, *, now=None, services=None) -> dict:
    """Run one UTC slot, or retry its incomplete attempt under a process lock."""
    root = Path(config["paths"]["state_dir"])
    _initialize_state(root)
    with (root / "worker.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"status": "skipped", "reason": "worker_locked", "mode": "paper_only"}
        services = services if services is not None else _PublicServices(config)
        clock = _clock(services, _timestamp(now) if now is not None else None)
        started = clock()
        interval = config["interval_seconds"]
        slot = pd.Timestamp(
            int(started.timestamp()) // interval * interval, unit="s", tz="UTC"
        ).isoformat()
        fingerprint = _config_hash(config)
        job_key = hashlib.sha256(f"{fingerprint}:{slot}".encode()).hexdigest()
        state = root / "research.sqlite"
        with _connection(state) as connection:
            connection.execute("BEGIN IMMEDIATE")
            previous = connection.execute(
                "SELECT * FROM jobs WHERE job_key=?", (job_key,)
            ).fetchone()
            if previous is not None and previous["status"] == "completed":
                try:
                    saved = _read_json(Path(previous["report_path"]))
                    complete = (
                        isinstance(saved, dict)
                        and saved.get("run_id") == previous["last_run_id"]
                        and saved.get("status") == "completed"
                        and saved.get("slot") == slot
                        and saved.get("config_hash") == fingerprint
                    )
                except (OSError, ValueError, TypeError):
                    complete = False
                if complete:
                    return {
                        "status": "skipped",
                        "reason": "slot_already_completed",
                        "slot": slot,
                        "run_id": previous["last_run_id"],
                        "report_path": previous["report_path"],
                        "mode": "paper_only",
                    }
            # Acquiring the global OS lock proves earlier running attempts no
            # longer own this worker, including a crashed process's last slot.
            connection.execute(
                "UPDATE attempts SET status='abandoned',finished_at=? WHERE status='running'",
                (started.isoformat(),),
            )
            run_id = started.strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:8]
            connection.execute(
                "INSERT OR REPLACE INTO jobs VALUES (?,?,?,?,?,?)",
                (job_key, fingerprint, slot, "running", run_id, None),
            )
            connection.execute(
                "INSERT INTO attempts VALUES (?,?,?,?,?,?)",
                (run_id, job_key, started.isoformat(), None, "running", None),
            )
            connection.commit()
        report = {
            "schema_version": 1,
            "run_id": run_id,
            "slot": slot,
            "config_hash": fingerprint,
            "as_of": started.isoformat(),
            "started_at": started.isoformat(),
            "configuration": copy.deepcopy(config),
            "status": "running",
            "mode": "paper_only",
            "forecasts": [],
            "evaluations": [],
            "paper_fills": [],
            "settlements": [],
            "performance": None,
            "errors": [],
            "stages": {},
        }
        _atomic_write(root / "latest.json", _encode(report))
        try:
            _execute(config, services, clock, report, root / "checkpoints" / f"{job_key}.json")
        except Exception as exc:
            report["errors"].append(
                {"stage": "cycle", "type": type(exc).__name__, "message": str(exc)}
            )
        success = bool(report["forecasts"] or report["settlements"] or report["performance"])
        report["status"] = (
            "completed" if not report["errors"] else ("partial" if success else "failed")
        )
        report["finished_at"] = clock().isoformat()
        directory = root / "runs" / run_id
        directory.mkdir(exist_ok=False)
        encoded = _encode(report)
        _atomic_write(directory / "run.json", encoded)
        _atomic_write(root / "latest.json", encoded)
        if report["status"] == "completed":
            _atomic_write(root / "last_success.json", encoded)
        with _connection(state) as connection:
            connection.execute(
                "UPDATE attempts SET finished_at=?,status=?,report_path=? WHERE run_id=?",
                (report["finished_at"], report["status"], str(directory / "run.json"), run_id),
            )
            connection.execute(
                "UPDATE jobs SET status=?,report_path=? WHERE job_key=?",
                (report["status"], str(directory / "run.json"), job_key),
            )
            connection.commit()
        return report


def run_worker(config: dict, *, max_cycles=None, stop_event=None, services=None) -> dict:
    """Poll fixed UTC slots, with bounded interruptible waits and retry backoff."""
    if max_cycles is not None and (type(max_cycles) is not int or max_cycles < 1):
        raise ValueError("max_cycles must be a positive integer")
    stop = stop_event if stop_event is not None else threading.Event()
    handlers = {}
    if threading.current_thread() is threading.main_thread():
        for signum in [signal.SIGINT, signal.SIGTERM]:
            handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, lambda *_: stop.set())
    count, recent = 0, []
    try:
        while not stop.is_set() and (max_cycles is None or count < max_cycles):
            report = run_research_cycle(config, services=services)
            count += 1
            recent.append({key: report.get(key) for key in ["run_id", "status", "slot", "reason"]})
            recent = recent[-10:]
            LOGGER.info("Research cycle %s: %s", report.get("run_id"), report["status"])
            if max_cycles is not None and count >= max_cycles:
                break
            clock = _clock(services or {}, None)
            remainder = (
                config["interval_seconds"] - clock().timestamp() % config["interval_seconds"]
            )
            delay = (
                min(remainder, config.get("retry_seconds", 60))
                if report["status"] in {"failed", "partial"}
                else remainder
            )
            while delay > 0 and not stop.is_set():
                chunk = min(60, delay)
                if stop.wait(chunk):
                    break
                delay -= chunk
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
    return {"cycles": count, "recent_runs": recent, "stopped": stop.is_set(), "mode": "paper_only"}
