"""Explicit multi-sport fetch/backtest/run commands; no execution APIs."""

import json
from pathlib import Path

import pandas as pd


def add_commands(sub):
    for name in ["nba-fetch", "nba-backtest", "nba-run", "nhl-backtest", "nhl-run"]:
        command = sub.add_parser(name)
        command.add_argument(
            "--cache-dir",
            type=Path,
            default=Path(
                "data/leagues/nhl-official" if name.startswith("nhl") else "data/leagues/nba"
            ),
        )
        command.add_argument(
            "--start-season", type=int, default=2009 if name.startswith("nhl") else 2010
        )
        command.add_argument("--through-season", type=int, default=2025)
        command.add_argument("--config", type=Path, default=Path("research.local.json"))
        command.add_argument("--validation", type=Path)
        if name.startswith("nba"):
            command.add_argument(
                "--nba-provider",
                choices=["auto", "archive", "nba-stats"],
                default="auto",
                help="auto reuses verified NBA caches then release archives; nba-stats opts into the live API",
            )
            command.add_argument(
                "--refresh",
                action="store_true",
                help="Refresh only the last requested season; retain earlier verified files",
            )
        command.add_argument(
            "--output-dir",
            type=Path,
            default=Path("reports/nhl" if name.startswith("nhl") else "reports/nba"),
        )

    worker = sub.add_parser(
        "leagues-worker", help="Refresh validated sport predictions independently of markets"
    )
    worker.add_argument("--config", type=Path, default=Path("research.local.json"))
    worker.add_argument("--sport", choices=["NBA", "NHL"], action="append", required=True)
    worker.add_argument("--max-cycles", type=int)
    worker.add_argument("--nba-provider", choices=["auto", "archive", "nba-stats"], default="auto")


def dispatch(args):
    if args.command == "leagues-worker":
        if args.max_cycles is not None and args.max_cycles < 1:
            raise ValueError("--max-cycles must be positive")
        return run_league_worker(args)
    if args.command in {"nhl-backtest", "nhl-run"}:
        return dispatch_nhl(args)
    if args.command not in {"nba-fetch", "nba-backtest", "nba-run"}:
        return None
    from predictions.league_model import backtest_league
    from predictions.nba import NBA_COLUMNS, fetch_nba, nba_features, nba_slate

    games, source = fetch_nba(
        args.cache_dir,
        args.start_season,
        args.through_season,
        refresh=args.command == "nba-run" or getattr(args, "refresh", False),
        provider=getattr(args, "nba_provider", "auto"),
    )
    if args.command == "nba-fetch":
        return {
            "sport": "NBA",
            "status": "verified",
            "games": len(games),
            "seasons": {str(y): int(n) for y, n in games.groupby("season").size().items()},
            "cache_dir": str(args.cache_dir),
            "source": source,
        }
    if args.command == "nba-backtest":
        holdout = games.loc[games.season.eq(args.through_season)]
        if (
            pd.Timestamp.now(tz="UTC") < pd.Timestamp(f"{args.through_season + 1}-07-01", tz="UTC")
            or len(holdout) < 1000
            or holdout.kickoff.max() < pd.Timestamp(f"{args.through_season + 1}-04-01", tz="UTC")
        ):
            raise ValueError("NBA holdout season is incomplete")
        return backtest_league(
            nba_features(games),
            NBA_COLUMNS,
            "NBA",
            args.output_dir,
            args.through_season,
            source,
            -5.5,
            220.5,
        )
    if args.validation is None:
        raise ValueError("NBA run requires --validation from completed backtest")
    report = json.loads(args.validation.read_text())
    if (
        report.get("sport") != "NBA"
        or report.get("type") != "league_validation"
        or report.get("columns") != NBA_COLUMNS
    ):
        raise ValueError("Invalid NBA validation report")
    slate, meta = nba_slate(args.cache_dir)
    now = pd.Timestamp.now(tz="UTC")
    slate = slate.loc[
        slate.kickoff.ge(now - pd.Timedelta(days=14))
        & slate.kickoff.le(now + pd.Timedelta(days=7))
        & slate.game_id.str.startswith("002")
    ]
    # A cached historical box-score feed may overlap the schedule; final games win.
    frames = pd.concat([games, slate], ignore_index=True).drop_duplicates("game_id", keep="first")
    from predictions.config import load_config
    from predictions.league_forecast import issue_league, publish_league

    config = load_config(args.config)
    batch = issue_league(
        nba_features(frames, as_of=now),
        report,
        args.output_dir,
        as_of=now,
        source={"historical": source, "schedule": meta},
    )
    publish_league(batch, config["paths"]["state_dir"])
    return batch


def run_league_worker(args):
    """Sequential sport adapters; independently persisted successes survive errors."""
    import fcntl
    import signal
    import threading
    from types import SimpleNamespace

    from predictions.config import load_config
    from predictions.data import _atomic_write

    config = load_config(args.config)
    state = Path(config["paths"]["state_dir"])
    state.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    previous = {s: signal.getsignal(s) for s in [signal.SIGINT, signal.SIGTERM]}
    for s in previous:
        signal.signal(s, lambda *_: stop.set())
    cycles = 0
    outcomes = []
    try:
        with (state / "league-worker.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return {"status": "skipped", "reason": "league_worker_locked"}
            while not stop.is_set() and (args.max_cycles is None or cycles < args.max_cycles):
                for sport in args.sport:
                    now = pd.Timestamp.now(tz="UTC")
                    season = now.year if now.month >= 7 else now.year - 1
                    validation = Path(config["paths"]["reports_dir"]) / (
                        sport.lower() + "-approved.json"
                    )
                    try:
                        if not validation.is_file():
                            raise ValueError("Completed sport validation report is not installed")
                        result = dispatch(
                            SimpleNamespace(
                                command=sport.lower() + "-run",
                                cache_dir=Path(config["paths"]["data_dir"])
                                / "leagues"
                                / ("nhl-official" if sport == "NHL" else "nba"),
                                start_season=2010 if sport == "NBA" else 2009,
                                through_season=season,
                                config=args.config,
                                validation=validation,
                                nba_provider=getattr(args, "nba_provider", "auto"),
                                output_dir=Path(config["paths"]["reports_dir"]) / sport.lower(),
                            )
                        )
                        outcome = {
                            "sport": sport,
                            "as_of": now.isoformat(),
                            "status": result["status"],
                            "forecasts": len(result["forecasts"]),
                        }
                    except Exception as exc:
                        outcome = {
                            "sport": sport,
                            "as_of": now.isoformat(),
                            "status": "failed",
                            "error": str(exc),
                        }
                    _atomic_write(
                        state / (sport.lower() + "-status.json"), json.dumps(outcome).encode()
                    )
                    outcomes.append(outcome)
                cycles += 1
                if args.max_cycles is not None and cycles >= args.max_cycles:
                    break
                stop.wait(min(config["interval_seconds"], 60))
                # Interruptible remaining interval; no blocking wait exceeds60seconds.
                remaining = max(0, config["interval_seconds"] - 60)
                while remaining and not stop.is_set():
                    chunk = min(remaining, 60)
                    stop.wait(chunk)
                    remaining -= chunk
    finally:
        for s, handler in previous.items():
            signal.signal(s, handler)
    return {"cycles": cycles, "recent": outcomes[-10:], "mode": "predictions_only"}


def dispatch_nhl(args):
    import fcntl

    from predictions.league_model import backtest_league
    from predictions.nhl import (
        NHL_COLUMNS,
        evaluate_nhl,
        fit_nhl,
        nhl_features,
        nhl_score_builder,
        nhl_slate,
    )
    from predictions.nhl_data import prepare

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    with (args.cache_dir / "build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        games, source = prepare(
            args.cache_dir,
            args.start_season,
            args.through_season,
            refresh_current=args.command == "nhl-run",
        )
        goalies = pd.read_parquet(args.cache_dir / "goalies.parquet")
    if args.command == "nhl-backtest":
        if (
            pd.Timestamp.now(tz="UTC") < pd.Timestamp(f"{args.through_season + 1}-07-01", tz="UTC")
            or len(games.loc[games.season.eq(args.through_season)]) < 800
        ):
            raise ValueError("NHL holdout incomplete")
        features = nhl_features(games, goalies)
        return backtest_league(
            features,
            NHL_COLUMNS,
            "NHL",
            args.output_dir,
            args.through_season,
            source,
            -1.5,
            5.5,
            fitter=fit_nhl,
            evaluator=evaluate_nhl,
        )
    if args.validation is None:
        raise ValueError("NHL run requires completed historical validation")
    report = json.loads(args.validation.read_text())
    if (
        report.get("sport") != "NHL"
        or report.get("type") != "league_validation"
        or report.get("columns") != NHL_COLUMNS
    ):
        raise ValueError("Invalid NHL validation report")
    slate, metadata = nhl_slate(args.cache_dir)
    now = pd.Timestamp.now(tz="UTC")
    frames = pd.concat([games, slate], ignore_index=True).drop_duplicates("game_id", keep="first")
    from predictions.config import load_config
    from predictions.league_forecast import issue_league, publish_league

    config = load_config(args.config)
    batch = issue_league(
        nhl_features(frames, goalies, as_of=now),
        report,
        args.output_dir,
        as_of=now,
        source={
            "provider": "NHL official events via fastRhockey; own older-season xG",
            "historical": source,
            "schedule": metadata,
        },
        fitter=fit_nhl,
        score_builder=nhl_score_builder,
    )
    publish_league(batch, config["paths"]["state_dir"])
    return batch
