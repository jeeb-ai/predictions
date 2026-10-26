"""Reproducible data fetch and historical evaluation commands."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import requests

from predictions.backtest import run_backtest
from predictions.data import NflverseProvider, load_games
from predictions.features import build_features


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Research-only NFL win-probability backtests")
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch", help="Snapshot the public nflverse schedule CSV")
    fetch.add_argument("--cache-dir", type=Path, default=Path("data/raw"))
    fetch.add_argument("--refresh", action="store_true")
    backtest = sub.add_parser("backtest", help="Run annual walk-forward logistic regression")
    backtest.add_argument("--input", type=Path, help="Use a local nflverse-format CSV offline")
    backtest.add_argument("--cache-dir", type=Path, default=Path("data/raw"))
    backtest.add_argument("--refresh", action="store_true")
    backtest.add_argument("--output-dir", type=Path, default=Path("reports"))
    backtest.add_argument("--through-season", type=int, help="Final complete season; held out")
    backtest.add_argument("--as-of", help="UTC outcome availability cutoff (default: now)")
    backtest.add_argument("--window", type=int, default=8)
    backtest.add_argument("--min-train-seasons", type=int, default=5)
    backtest.add_argument("--bootstrap-samples", type=int, default=1000)
    backtest.add_argument("--seed", type=int, default=42)
    return parser


def _local_metadata(path: Path) -> dict:
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    candidates = [path.with_suffix(".json"), *sorted(path.parent.glob(f"*_{checksum}.json"))]
    for sidecar in candidates:
        if not sidecar.exists():
            continue
        metadata = json.loads(sidecar.read_text())
        if not isinstance(metadata, dict) or metadata.get("sha256") != checksum:
            raise ValueError("Local CSV checksum does not match its metadata sidecar")
        return metadata
    return {
        "source_url": None,
        "local_path": str(path.resolve()),
        "sha256": checksum,
        "fetched_at": None,
        "provenance_note": "User-supplied local CSV; original source and fetch time unknown",
    }


def _complete_seasons(games: pd.DataFrame) -> list[int]:
    regular = games.loc[games["game_type"].eq("REG")]
    complete = (
        regular.assign(complete=regular["home_score"].notna() & regular["away_score"].notna())
        .groupby("season")["complete"]
        .all()
    )
    return sorted(int(season) for season in complete.index[complete])


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "fetch":
            path, metadata = NflverseProvider(args.cache_dir).fetch(refresh=args.refresh)
            print(json.dumps({"path": str(path), "metadata": metadata}, indent=2))
            return 0
        if args.input is None:
            path, metadata = NflverseProvider(args.cache_dir).fetch(refresh=args.refresh)
        else:
            if args.refresh:
                raise ValueError("--refresh cannot be combined with --input")
            path, metadata = args.input, _local_metadata(args.input)
        as_of = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp.now(tz="UTC")
        as_of = as_of.tz_localize("UTC") if as_of.tzinfo is None else as_of.tz_convert("UTC")
        requested_as_of = as_of
        if metadata.get("fetched_at"):
            fetched_at = pd.Timestamp(metadata["fetched_at"])
            if fetched_at.tzinfo is None or pd.isna(fetched_at):
                raise ValueError("Snapshot fetched_at must be a valid timezone-aware timestamp")
            as_of = min(as_of, fetched_at.tz_convert("UTC"))
        games = load_games(path, as_of=as_of)
        complete = _complete_seasons(games)
        if not complete:
            raise ValueError("No complete regular season is available at the requested --as-of")
        holdout = args.through_season if args.through_season is not None else max(complete)
        if holdout not in complete:
            raise ValueError(f"Season {holdout} is absent or incomplete at {as_of.isoformat()}")
        games = games.loc[games["season"].le(holdout)].copy()
        features = build_features(games, window=args.window)
        metadata = {
            **metadata,
            "as_of": as_of.isoformat(),
            "requested_as_of": requested_as_of.isoformat(),
            "feature_window": args.window,
        }
        summary = run_backtest(
            features,
            args.output_dir,
            holdout_season=holdout,
            min_train_seasons=args.min_train_seasons,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
            source_metadata=metadata,
        )
        print(json.dumps(summary, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, requests.RequestException) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
