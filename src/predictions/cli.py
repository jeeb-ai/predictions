"""Reproducible data fetch and historical evaluation commands."""

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

from predictions.backtest import run_backtest
from predictions.data import NflverseProvider, load_games
from predictions.features import build_features
from predictions.live_cli import add_commands, dispatch
from predictions.research_cli import add_commands as add_research_commands
from predictions.research_cli import dispatch as dispatch_research


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
    pbp = sub.add_parser("fetch-pbp", help="Snapshot nflverse play-by-play Parquet seasons")
    pbp.add_argument("--start-season", type=int, default=1999)
    pbp.add_argument("--through-season", type=int, required=True)
    pbp.add_argument("--cache-dir", type=Path, default=Path("data/pbp"))
    pbp.add_argument("--workers", type=int, default=4)
    pbp.add_argument("--refresh", action="store_true")
    pbp.add_argument("--offline", action="store_true")
    compare = sub.add_parser("compare", help="Compare fixed Model 1.0 and causal-PBP Model 2.0")
    compare.add_argument("--input", type=Path, help="Use a local nflverse schedule CSV")
    compare.add_argument("--cache-dir", type=Path, default=Path("data/raw"))
    compare.add_argument("--refresh", action="store_true")
    compare.add_argument("--output-dir", type=Path, default=Path("reports/model2"))
    compare.add_argument("--through-season", type=int, default=2025)
    compare.add_argument("--as-of", help="UTC outcome cutoff; source fetch times also apply")
    compare.add_argument("--window", type=int, default=8)
    compare.add_argument("--min-train-seasons", type=int, default=5)
    compare.add_argument("--bootstrap-samples", type=int, default=2000)
    compare.add_argument("--seed", type=int, default=42)
    compare.add_argument("--pbp-dir", type=Path, default=Path("data/pbp"))
    compare.add_argument("--ep-dir", type=Path, default=Path("data/causal_ep"))
    compare.add_argument("--workers", type=int, default=4)
    compare.add_argument("--offline", action="store_true")
    add_commands(sub)
    add_research_commands(sub)
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


def _prepare_games(args) -> tuple[pd.DataFrame, dict, int]:
    if getattr(args, "offline", False):
        if args.refresh:
            raise ValueError("--offline cannot be combined with --refresh")
        if args.input is None and not (args.cache_dir / "latest.json").exists():
            raise FileNotFoundError("Offline comparison needs --input or a cached schedule")
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
    metadata = {
        **metadata,
        "as_of": as_of.isoformat(),
        "requested_as_of": requested_as_of.isoformat(),
        "feature_window": args.window,
    }
    return games.loc[games["season"].le(holdout)].copy(), metadata, holdout


def _fetch_pbp(seasons, cache_dir, workers, refresh=False, offline=False) -> dict:
    from predictions.pbp import PbpProvider

    if not 1 <= workers <= 8:
        raise ValueError("--workers must be between 1 and 8")
    provider = PbpProvider(cache_dir)

    def fetch(season):
        path, metadata = provider.fetch(season, refresh=refresh, offline=offline)
        print(f"PBP {season}: verified {metadata['sha256'][:12]}", file=sys.stderr, flush=True)
        return season, (path, metadata)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(pool.map(fetch, seasons))


def _compare(args) -> dict:
    from predictions.causal_ep import reconstruct_epa
    from predictions.comparison import run_comparison
    from predictions.features_v2 import build_features_v2
    from predictions.pbp import aggregate_pbp

    games, metadata, holdout = _prepare_games(args)
    seasons = sorted(int(season) for season in games["season"].unique())
    snapshots = _fetch_pbp(seasons, args.pbp_dir, args.workers, args.refresh, args.offline)
    paths = {season: item[0] for season, item in snapshots.items()}
    # A cached PBP snapshot can contain a then-live partial game. Waiting must
    # not make that raw record eligible for EP training or pregame histories.
    pbp_games = games.copy()
    for season, (_, source) in snapshots.items():
        fetched = pd.Timestamp(source["fetched_at"])
        if fetched.tzinfo is None or pd.isna(fetched):
            raise ValueError("PBP fetched_at must be a valid timezone-aware timestamp")
        invalid = pbp_games["season"].eq(season) & pbp_games["result_available_at"].ge(fetched)
        pbp_games.loc[invalid, ["home_score", "away_score"]] = float("nan")
    overrides, ep_metadata = reconstruct_epa(paths, args.ep_dir, games=pbp_games)
    eligible_ids = set(pbp_games.loc[pbp_games["home_score"].notna(), "game_id"])
    teams, quarterbacks, coverage = [], [], []
    for season, path in paths.items():
        team, qb = aggregate_pbp(path, epa_override=overrides[season])
        teams.append(team.loc[team["game_id"].isin(eligible_ids)])
        quarterbacks.append(qb.loc[qb["game_id"].isin(eligible_ids)])
        coverage.append(
            {
                "season": season,
                "team_games": len(team),
                "eligible_plays": int(team["eligible_plays"].sum()),
                "missing_epa_plays": int(team["missing_epa_plays"].sum()),
            }
        )
    first = build_features(games, window=args.window)
    second = build_features_v2(
        games,
        pd.concat(teams, ignore_index=True),
        pd.concat(quarterbacks, ignore_index=True),
        window=args.window,
    )
    metadata = {
        **metadata,
        "play_by_play": {str(season): item[1] for season, item in snapshots.items()},
        "causal_ep": ep_metadata,
        "pbp_coverage": coverage,
    }
    print("Running paired chronological comparison", file=sys.stderr, flush=True)
    return run_comparison(
        first,
        second,
        args.output_dir,
        holdout_season=holdout,
        min_train_seasons=args.min_train_seasons,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        source_metadata=metadata,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        research_result = dispatch_research(args)
        if research_result is not None:
            print(json.dumps(research_result, indent=2, allow_nan=False))
            if (
                args.command == "research-status"
                and args.healthcheck
                and not research_result["healthy"]
            ):
                return 1
            return (
                1
                if args.command == "research-run" and research_result.get("status") == "failed"
                else 0
            )
        result = dispatch(args)
        if result is not None:
            print(json.dumps(result, indent=2, allow_nan=False))
            return 1 if args.command == "paper-audit" and not result["valid"] else 0
        if args.command == "fetch":
            path, metadata = NflverseProvider(args.cache_dir).fetch(refresh=args.refresh)
            print(json.dumps({"path": str(path), "metadata": metadata}, indent=2))
            return 0
        if args.command == "fetch-pbp":
            if args.through_season < args.start_season:
                raise ValueError("--through-season must be at least --start-season")
            result = _fetch_pbp(
                range(args.start_season, args.through_season + 1),
                args.cache_dir,
                args.workers,
                args.refresh,
                args.offline,
            )
            print(
                json.dumps(
                    {
                        str(year): {"path": str(path), "metadata": source}
                        for year, (path, source) in result.items()
                    },
                    indent=2,
                )
            )
            return 0
        if args.command == "compare":
            print(json.dumps(_compare(args), indent=2, allow_nan=False))
            return 0
        games, metadata, holdout = _prepare_games(args)
        features = build_features(games, window=args.window)
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
