"""Commands for a private, self-hosted paper research service."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path

import pandas as pd

from predictions.config import initialize_research, load_config


def add_commands(sub) -> None:
    for name, help_text in [
        ("research-init", "Create research configuration and a private dashboard token"),
        ("research-run", "Run one idempotent scheduled research/paper cycle"),
        ("research-worker", "Run recurring research cycles with durable state"),
        ("research-status", "Read the latest research cycle without modifying state"),
        ("dashboard", "Serve the authenticated dashboard on localhost"),
    ]:
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--config", type=Path, default=Path("research.local.json"))
        if name == "research-worker":
            command.add_argument("--max-cycles", type=int)
        if name == "research-status":
            command.add_argument("--healthcheck", action="store_true")
        if name == "dashboard":
            command.add_argument("--port", type=int, default=8501)


def research_status(config: dict) -> dict:
    latest = Path(config["paths"]["state_dir"]) / "latest.json"
    if not latest.exists():
        return {"status": "not_started", "healthy": False, "mode": "paper_only"}
    result = json.loads(latest.read_text())
    if not isinstance(result, dict):
        raise ValueError("Latest research report must be an object")
    at = pd.Timestamp(result.get("finished_at", result.get("completed_at", result.get("as_of"))))
    if pd.isna(at) or at.tzinfo is None:
        raise ValueError("Latest research report has no valid timestamp")
    age = (pd.Timestamp.now(tz="UTC") - at).total_seconds()
    fresh = 0 <= age <= config["dashboard"]["max_snapshot_age_seconds"]
    return {
        "mode": "paper_only",
        "status": result.get("status"),
        "run_id": result.get("run_id"),
        "as_of": result.get("as_of"),
        "age_seconds": age,
        "healthy": fresh and result.get("status") in {"completed", "partial", "success"},
        "stale": not fresh,
        "errors": result.get("errors", []),
        "latest_path": str(latest),
    }


def dispatch(args) -> dict | None:
    if args.command not in {
        "research-init",
        "research-run",
        "research-worker",
        "research-status",
        "dashboard",
    }:
        return None
    if args.command == "research-init":
        return initialize_research(args.config)
    config = load_config(args.config)
    if args.command == "research-status":
        return research_status(config)
    if args.command == "dashboard":
        if not 1024 <= args.port <= 65535:
            raise ValueError("Dashboard port must be between 1024 and 65535")
        if not Path(config["dashboard"]["token_file"]).is_file():
            raise ValueError("Initialize the private dashboard token with research-init first")
        import predictions.dashboard

        environment = dict(os.environ)
        environment["NFL_RESEARCH_CONFIG"] = str(args.config.resolve())
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                predictions.dashboard.__file__,
                "--server.address=127.0.0.1",
                f"--server.port={args.port}",
                "--server.headless=true",
                "--browser.gatherUsageStats=false",
                "--server.enableCORS=true",
                "--server.enableXsrfProtection=true",
            ],
            env=environment,
            check=False,
        )
        if result.returncode:
            raise ValueError(f"Dashboard process exited with status {result.returncode}")
        return {"dashboard": "stopped", "mode": "read_only"}
    initialize_research(args.config)
    from predictions.research import run_research_cycle, run_worker

    if args.command == "research-run":
        return run_research_cycle(config)
    if args.max_cycles is not None and args.max_cycles < 1:
        raise ValueError("--max-cycles must be positive")
    stop = threading.Event()
    previous = {name: signal.getsignal(name) for name in (signal.SIGINT, signal.SIGTERM)}
    try:
        for name in previous:
            signal.signal(name, lambda *_: stop.set())
        return run_worker(config, max_cycles=args.max_cycles, stop_event=stop)
    finally:
        for name, handler in previous.items():
            signal.signal(name, handler)
