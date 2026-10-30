"""Prediction-only operational commands. No exchange or paper-fill calls."""

from pathlib import Path


def add_commands(sub):
    for name in ["platform-capture", "platform-run", "platform-worker", "platform-status"]:
        parser = sub.add_parser(name)
        parser.add_argument("--config", type=Path, default=Path("research.local.json"))
        if name in ["platform-run", "platform-worker"]:
            parser.add_argument("--sport", choices=["NFL", "NBA", "NHL"], required=True)
        if name == "platform-status":
            parser.add_argument("--sport", choices=["NFL", "NBA", "NHL"])
            parser.add_argument("--healthcheck", action="store_true")
        if name == "platform-worker":
            parser.add_argument("--max-cycles", type=int)


def dispatch(args):
    if not args.command.startswith("platform-"):
        return None
    from predictions import automation
    from predictions.config import load_config
    from predictions.prospective import capture_snapshots

    config = load_config(args.config)
    if args.command in ["platform-run", "platform-worker"]:
        import fcntl

        from predictions.config import initialize_research

        root = Path(config["paths"]["state_dir"])
        root.mkdir(parents=True, exist_ok=True)
        with (root / "platform-initialize.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            initialize_research(args.config)
    if args.command == "platform-capture":
        return capture_snapshots(config)
    if args.command == "platform-status":
        return automation.status(config, args.sport)
    if args.command == "platform-run":
        return automation.run_slot(config, args.sport, args.config)
    if args.command == "platform-worker":
        return automation.run_worker(config, args.sport, args.config, max_cycles=args.max_cycles)
    return None
