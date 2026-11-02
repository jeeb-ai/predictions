"""Independent sport cycles, durable checkpoints and an interruptible process watchdog."""

import fcntl
import hashlib
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from predictions.data import _atomic_write
from predictions.prospective import (
    SPORTS,
    ForecastJournal,
    capture_snapshots,
    connect,
    digest,
    encode,
    now,
    stamp,
)

LOG = logging.getLogger(__name__)


class SetupRequired(ValueError):
    pass


def approved(config, sport):
    names = {
        "NFL": ("scoring-approved.json", "v5-scoring-validation.json"),
        "NBA": ("nba-approved.json", "nba-validation.json"),
        "NHL": ("nhl-approved.json", "nhl-validation.json"),
    }
    local, bundled = names[sport]
    path = Path(config["paths"]["reports_dir"]) / local
    if not path.is_file():
        path = Path(config["automation"]["approvals_dir"]) / bundled
    if not path.is_file():
        raise SetupRequired(f"Install the completed {sport} validation report as {local}")
    return path


def forecast_sport(config, sport, config_path):
    """Call the existing, independently validated forecasters; no market requests."""
    if sport in ["NBA", "NHL"]:
        from predictions.league_cli import dispatch

        at = now()
        year = at.year if at.month >= 7 else at.year - 1
        return dispatch(
            SimpleNamespace(
                command=sport.lower() + "-run",
                config=Path(config_path),
                start_season=2010 if sport == "NBA" else 2009,
                through_season=year,
                cache_dir=Path(config["paths"]["data_dir"])
                / "leagues"
                / ("nba" if sport == "NBA" else "nhl-official"),
                validation=approved(config, sport),
                nba_provider=config["automation"]["nba_provider"],
                output_dir=Path(config["paths"]["reports_dir"]) / sport.lower(),
                defer_publish=True,
                lookahead_hours=config["lookahead_hours"],
            )
        )
    from predictions.cli import _parser
    from predictions.data import NflverseProvider, load_games
    from predictions.live_cli import _forecast

    schedule_path, source = NflverseProvider(Path(config["paths"]["schedule_cache"])).fetch(
        refresh=True
    )
    games = load_games(schedule_path, as_of=min(now(), stamp(source["fetched_at"])))
    at = now()
    selected = games.loc[
        games.kickoff.gt(at) & games.kickoff.le(at + pd.Timedelta(hours=config["lookahead_hours"]))
    ]
    if "kickoff_time_imputed" in selected:
        selected = selected.loc[~selected.kickoff_time_imputed]
    forecasts, skipped = [], []
    for season in sorted(selected.season.unique()):
        ids = selected.loc[selected.season.eq(season), "game_id"].tolist()
        paths = config["paths"]
        argv = [
            "forecast",
            "--model-version",
            config["model_version"],
            "--include-analysis",
            "--input",
            str(schedule_path),
            "--cache-dir",
            paths["schedule_cache"],
            "--pbp-dir",
            paths["pbp_dir"],
            "--ep-dir",
            paths["ep_dir"],
            "--model-dir",
            paths["model_dir"],
            "--output-dir",
            str(Path(paths["reports_dir"]) / "forecasts"),
            "--ledger",
            paths["ledger"],
            "--bootstrap-samples",
            str(config["bootstrap_samples"]),
            "--confidence-level",
            str(config["confidence_level"]),
            "--bootstrap-seed",
            str(config["bootstrap_seed"]),
            "--workers",
            str(config["workers"]),
        ]
        try:
            argv += ["--score-validation", str(approved(config, sport))]
        except SetupRequired:
            LOG.warning("NFL scoring approval unavailable; preserved winner model remains active")
        for identity in ids:
            argv += ["--game-id", identity]
        args = _parser().parse_args(argv)
        args.schedule_metadata = source
        args.skip_ledger = True
        batch = _forecast(args)
        forecasts.extend(batch["forecasts"])
        skipped.extend(batch.get("skipped", []))
    return {
        "sport": sport,
        "as_of": now().isoformat(),
        "forecasts": forecasts,
        "skipped": skipped,
        "slate": [
            {
                "game_id": r.game_id,
                "home_team": r.home_team,
                "away_team": r.away_team,
                "kickoff": r.kickoff.isoformat(),
            }
            for r in selected.itertuples()
        ],
        "status": "completed" if len(selected) else "no_upcoming_games",
        "source": source,
    }


def _publish(root, sport, batch):
    from predictions.league_forecast import publish_league

    if sport != "NFL":
        publish_league(batch, root)
    else:
        _atomic_write(root / "predictions.json", (encode(batch) + "\n").encode())


def _read_status(root, sport):
    path = root / (sport.lower() + "-automation.json")
    if not path.exists():
        return {"sport": sport, "status": "not_started"}
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {"sport": sport, "status": "status_unavailable"}


def _save_status(root, sport, **updates):
    value = {**_read_status(root, sport), **updates, "sport": sport}
    _atomic_write(root / (sport.lower() + "-automation.json"), (encode(value) + "\n").encode())


def source_refresh(batch):
    stamps = []

    def visit(value):
        if isinstance(value, dict):
            if value.get("fetched_at"):
                stamps.append(stamp(value["fetched_at"]))
            if value.get("manifest_path"):
                path = Path(value["manifest_path"])
                content = path.read_bytes()
                if hashlib.sha256(content).hexdigest() != value.get("sha256"):
                    raise ValueError("Source manifest checksum mismatch")
                visit(json.loads(content))
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(batch)
    return max(stamps).isoformat() if stamps else None


def run_slot(config, sport, config_path, *, adapter=None, clock=now, publisher=None):
    if sport not in SPORTS:
        raise ValueError("Sport must be NFL, NBA or NHL")
    root = Path(config["paths"]["state_dir"])
    journal = ForecastJournal(root / "prospective.sqlite")
    with (root / (sport.lower() + "-automation.lock")).open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"sport": sport, "status": "skipped", "reason": "sport_locked"}
        start = stamp(clock())
        interval = config["automation"]["intervals"][sport]
        slot = pd.Timestamp(
            int(start.timestamp()) // interval * interval, unit="s", tz="UTC"
        ).isoformat()
        key = digest([sport, config["config_hash"], slot])
        with connect(journal.path) as db:
            db.execute("BEGIN IMMEDIATE")
            previous = db.execute("SELECT * FROM platform_jobs WHERE job_id=?", (key,)).fetchone()
            if previous is not None and previous["status"] == "completed":
                return {
                    "sport": sport,
                    "status": "skipped",
                    "reason": "slot_completed",
                    "slot": slot,
                }
            # Owning the OS lock proves any previous running process is gone.
            db.execute(
                "UPDATE platform_jobs SET status='abandoned',finished_at=? WHERE sport=? AND status='running'",
                (start.isoformat(), sport),
            )
            checkpoint = previous["checkpoint"] if previous is not None else None
            checkpoint_sha = previous["checkpoint_sha256"] if previous is not None else None
            db.execute(
                "INSERT INTO platform_jobs VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(job_id) DO UPDATE SET status='running',started_at=excluded.started_at,finished_at=NULL,error_code=NULL",
                (key, sport, slot, "running", start.isoformat(), None, None, None, None),
            )
            db.commit()
        _save_status(
            root,
            sport,
            status="running",
            started_at=start.isoformat(),
            last_attempt_at=start.isoformat(),
            error_code=None,
        )
        try:
            if checkpoint is None:
                batch = (adapter or forecast_sport)(config, sport, config_path)
                if batch.get("sport", sport) != sport or not isinstance(
                    batch.get("forecasts"), list
                ):
                    raise ValueError("Invalid forecast batch")
                received = stamp(clock())
                with connect(journal.path) as db:
                    db.execute("BEGIN IMMEDIATE")
                    captured = journal.capture(
                        sport, batch["forecasts"], received_at=received, db=db
                    )
                    db.execute(
                        "UPDATE platform_jobs SET checkpoint=?,checkpoint_sha256=? WHERE job_id=?",
                        (encode(batch), digest(batch), key),
                    )
                    db.commit()
            else:
                batch = json.loads(checkpoint)
                if digest(batch) != checkpoint_sha:
                    raise ValueError("Cycle checkpoint checksum mismatch")
                captured = journal.capture(sport, batch["forecasts"], received_at=clock())
            # A crash here can reuse the committed receipts/checkpoint on restart.
            (publisher or _publish)(root, sport, batch)
            refreshed = source_refresh(batch)
            finished = stamp(clock()).isoformat()
            with connect(journal.path) as db:
                db.execute(
                    "UPDATE platform_jobs SET status='completed',finished_at=? WHERE job_id=?",
                    (finished, key),
                )
                db.commit()
            _save_status(
                root,
                sport,
                status="completed",
                finished_at=finished,
                last_prediction_cycle_at=finished,
                last_data_refresh_at=refreshed,
                error_code=None,
                setup_action=None,
            )
            return {
                "sport": sport,
                "status": "completed",
                "slot": slot,
                "forecasts": len(batch["forecasts"]),
                "capture": captured,
            }
        except Exception as exc:
            code = "setup_required" if isinstance(exc, SetupRequired) else "cycle_failed"
            finished = stamp(clock()).isoformat()
            with connect(journal.path) as db:
                db.execute(
                    "UPDATE platform_jobs SET status='failed',finished_at=?,error_code=? WHERE job_id=?",
                    (finished, code, key),
                )
                db.commit()
            _save_status(
                root,
                sport,
                status="failed",
                finished_at=finished,
                error_code=code,
                setup_action="Install the completed sport validation report."
                if code == "setup_required"
                else "Earlier forecasts are retained. Check the private cycle log and provider access; the worker retries automatically.",
            )
            LOG.exception("%s forecast cycle failed; earlier predictions retained", sport)
            return {"sport": sport, "status": "failed", "slot": slot, "error_code": code}


def status(config, sport=None, *, at=None):
    at = stamp(at) if at is not None else now()
    root = Path(config["paths"]["state_dir"])
    items = []
    for name in [sport] if sport else SPORTS:
        value = _read_status(root, name)
        for key in ["last_prediction_cycle_at", "last_data_refresh_at"]:
            value.setdefault(key, None)
        started = stamp(value["started_at"]) if value.get("started_at") else None
        success = (
            stamp(value["last_prediction_cycle_at"]) if value["last_prediction_cycle_at"] else None
        )
        running = (
            value["status"] == "running"
            and started is not None
            and 0 <= (at - started).total_seconds() <= config["automation"]["max_cycle_seconds"]
        )
        fresh = success is not None and 0 <= (at - success).total_seconds() <= max(
            config["dashboard"]["max_snapshot_age_seconds"],
            2 * config["automation"]["intervals"][name],
        )
        value.update(
            healthy=running or (fresh and value["status"] in ["completed", "failed"]),
            stale=not fresh,
            interval_seconds=config["automation"]["intervals"][name],
        )
        path = root / ("predictions.json" if name == "NFL" else name.lower() + "-predictions.json")
        value["current_predictions"] = 0
        try:
            if path.is_file() and path.stat().st_size <= 32 * 1024 * 1024:
                rows = json.loads(path.read_text()).get("forecasts", [])
                value["current_predictions"] = sum(
                    stamp(r["kickoff"]) > at and stamp(r["generated_at"]) < stamp(r["kickoff"])
                    for r in rows
                )
        except (OSError, ValueError, KeyError, TypeError):
            value["prediction_snapshot_error"] = (
                "Refresh could not be verified; other sports remain available."
            )
        items.append(value)
    return {
        "mode": "predictions_only",
        "sports": items,
        "healthy": all(x["healthy"] for x in items),
        "as_of": at.isoformat(),
    }


def _signal_process(process, sig):
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        pass


def supervise(argv, *, stdout, stderr, timeout, check=False, stop):
    process = subprocess.Popen(argv, stdout=stdout, stderr=stderr, start_new_session=True)
    deadline = time.monotonic() + timeout
    timed_out = False
    try:
        while process.poll() is None:
            if stop.is_set() or time.monotonic() >= deadline:
                timed_out = not stop.is_set()
                _signal_process(process, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    _signal_process(process, signal.SIGKILL)
                    process.wait()
                break
            try:
                process.wait(timeout=min(1, max(0.01, deadline - time.monotonic())))
            except subprocess.TimeoutExpired:
                pass
    finally:
        if process.poll() is None:
            _signal_process(process, signal.SIGKILL)
            process.wait()
    if timed_out:
        raise subprocess.TimeoutExpired(argv, timeout)
    return subprocess.CompletedProcess(argv, process.returncode)


def run_worker(config, sport, config_path, *, max_cycles=None, stop=None, runner=None, clock=now):
    """One process per sport; Compose starts all three independently."""
    if sport not in SPORTS or (
        max_cycles is not None and (type(max_cycles) is not int or max_cycles < 1)
    ):
        raise ValueError("Invalid worker sport or max_cycles")
    root = Path(config["paths"]["state_dir"])
    root.mkdir(parents=True, exist_ok=True)
    stop = stop or threading.Event()
    handlers = {}
    if threading.current_thread() is threading.main_thread():
        for sig in [signal.SIGINT, signal.SIGTERM]:
            handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, lambda *_: stop.set())
    cycles = 0
    last_status = "stopped"
    try:
        with (root / (sport.lower() + "-daemon.lock")).open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return {"sport": sport, "status": "skipped", "reason": "daemon_locked"}
            capture_snapshots(config)
            while not stop.is_set() and (max_cycles is None or cycles < max_cycles):
                logs = root / "automation-logs" / sport.lower()
                logs.mkdir(parents=True, exist_ok=True)
                log = logs / (uuid.uuid4().hex + ".log")
                try:
                    with log.open("x") as output:
                        log.chmod(0o600)
                        execute = runner or (lambda *a, **k: supervise(*a, stop=stop, **k))
                        result = execute(
                            [
                                sys.executable,
                                "-m",
                                "predictions",
                                "platform-run",
                                "--sport",
                                sport,
                                "--config",
                                str(Path(config_path).resolve()),
                            ],
                            stdout=output,
                            stderr=subprocess.STDOUT,
                            timeout=config["automation"]["max_cycle_seconds"],
                            check=False,
                        )
                        failed = result.returncode != 0
                except (subprocess.TimeoutExpired, OSError) as exc:
                    failed = True
                    _save_status(
                        root,
                        sport,
                        status="failed",
                        finished_at=stamp(clock()).isoformat(),
                        error_code="cycle_timeout"
                        if isinstance(exc, subprocess.TimeoutExpired)
                        else "worker_spawn_failed",
                        setup_action="Cycle exceeded its time limit. Retained forecasts remain; automatic retry follows.",
                    )
                last_status = "failed" if failed else "completed"
                cycles += 1
                if max_cycles is not None and cycles >= max_cycles:
                    break
                interval = config["automation"]["intervals"][sport]
                delay = (
                    min(config["automation"]["retry_seconds"], interval)
                    if failed
                    else interval - clock().timestamp() % interval
                )
                while delay > 0 and not stop.is_set():
                    chunk = min(30, delay)
                    stop.wait(chunk)
                    delay -= chunk
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    return {
        "sport": sport,
        "cycles": cycles,
        "mode": "predictions_only",
        "status": "stopped" if stop.is_set() else last_status,
    }
