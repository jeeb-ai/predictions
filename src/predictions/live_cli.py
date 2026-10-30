"""Live research and local paper-ledger commands; no execution API exists."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd


def _now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def add_commands(sub) -> None:
    forecast = sub.add_parser("forecast", help="Train or load an annual model and forecast pregame")
    forecast.add_argument("--game-id", action="append", required=True)
    forecast.add_argument("--model-version", choices=["v1", "v2"], default="v2")
    forecast.add_argument("--include-analysis", action="store_true")
    forecast.add_argument(
        "--score-validation",
        type=Path,
        help="Trusted completed scoring validation report; optional",
    )
    forecast.add_argument("--bundle", type=Path, help="Trusted local live-model bundle directory")
    forecast.add_argument("--input", type=Path, help="Local nflverse schedule snapshot")
    forecast.add_argument("--cache-dir", type=Path, default=Path("data/raw"))
    forecast.add_argument("--pbp-dir", type=Path, default=Path("data/pbp"))
    forecast.add_argument("--ep-dir", type=Path, default=Path("data/causal_ep"))
    forecast.add_argument("--model-dir", type=Path, default=Path("models/live"))
    forecast.add_argument("--output-dir", type=Path, default=Path("reports/live"))
    forecast.add_argument("--window", type=int, default=8)
    forecast.add_argument("--workers", type=int, default=4)
    forecast.add_argument("--bootstrap-samples", type=int, default=0)
    forecast.add_argument("--confidence-level", type=float, default=0.95)
    forecast.add_argument("--bootstrap-seed", type=int, default=42)
    forecast.add_argument("--as-of", help="Information cutoff; default is after source fetching")
    forecast.add_argument("--offline", action="store_true")
    forecast.add_argument("--ledger", type=Path, default=Path("data/paper/ledger.sqlite"))

    markets = sub.add_parser("us-markets", help="Discover public Polymarket US NFL winner markets")
    markets.add_argument("--limit", type=int, default=20)
    markets.add_argument("--offset", type=int, default=0)
    markets.add_argument("--cache-dir", type=Path, default=Path("data/polymarket_us"))
    markets.add_argument("--output", type=Path)

    quote = sub.add_parser("quote", help="Read an executable US YES or NO ask book")
    quote.add_argument("--market", required=True, help="Verified Polymarket US NFL market slug")
    quote.add_argument("--outcome", choices=["yes", "no"], default="yes")
    quote.add_argument("--cache-dir", type=Path, default=Path("data/polymarket_us"))
    quote.add_argument("--output", type=Path)
    quote.add_argument("--ledger", type=Path, default=Path("data/paper/ledger.sqlite"))

    settlement = sub.add_parser("market-settlement", help="Read a final US settlement for audit")
    settlement.add_argument("--market", required=True)
    settlement.add_argument("--cache-dir", type=Path, default=Path("data/polymarket_us"))
    settlement.add_argument("--output", type=Path)

    ev = sub.add_parser("ev", help="Evaluate depth and fees using explicit outcome assumptions")
    ev.add_argument("--forecast", type=Path, required=True)
    ev.add_argument("--quote", type=Path, required=True)
    ev.add_argument("--game-id", help="Required when the forecast batch contains multiple games")
    ev.add_argument("--team", required=True)
    ev.add_argument("--quantity", type=int, required=True)
    ev.add_argument("--tie-probability")
    ev.add_argument("--completion-probability")
    ev.add_argument("--cancellation-settlement-price")
    ev.add_argument("--max-quote-age-seconds", type=int, default=60)
    ev.add_argument("--max-forecast-age-seconds", type=int, default=3600)
    ev.add_argument("--min-ev-per-contract", default="0")
    ev.add_argument("--output", type=Path)
    ev.add_argument("--ledger", type=Path, default=Path("data/paper/ledger.sqlite"))

    fill = sub.add_parser("paper-fill", help="Record a local simulated fill; never sends an order")
    fill.add_argument("--evaluation-id", required=True)
    fill.add_argument("--idempotency-key", required=True)
    fill.add_argument("--max-exposure", help="Optional maximum total unsettled paper cost in USD")
    fill.add_argument("--ledger", type=Path, default=Path("data/paper/ledger.sqlite"))

    settle = sub.add_parser("paper-settle", help="Append an evidenced local paper settlement")
    settle.add_argument("--trade-id", required=True)
    settle.add_argument("--price", required=True, help="Actual per-contract settlement, USD 0 to 1")
    settle.add_argument("--evidence", type=Path, required=True)
    settle.add_argument("--idempotency-key", required=True)
    settle.add_argument("--ledger", type=Path, default=Path("data/paper/ledger.sqlite"))

    audit = sub.add_parser("paper-audit", help="Verify append-only paper records and hash chain")
    audit.add_argument("--ledger", type=Path, default=Path("data/paper/ledger.sqlite"))
    audit.add_argument(
        "--expected-head-hash", help="Compare with a separately retained audit anchor"
    )


def _read(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _write(path: Path | None, value: dict) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as handle:
            handle.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _forecast(args) -> dict:
    from predictions.cli import _local_metadata
    from predictions.data import NflverseProvider, load_games
    from predictions.inference import (
        predict_pregame,
        prepare_pregame_features,
        train_live_model,
    )
    from predictions.ledger import PaperLedger

    if args.input is not None:
        path = args.input
        supplied_source = getattr(args, "schedule_metadata", None)
        if supplied_source is None:
            schedule_source = _local_metadata(path)
        else:
            if (
                not isinstance(supplied_source, dict)
                or supplied_source.get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest()
            ):
                raise ValueError("Observed schedule metadata does not match its snapshot")
            schedule_source = supplied_source
    else:
        if args.offline and not (args.cache_dir / "latest.json").exists():
            raise FileNotFoundError("Offline forecasts need --input or a cached schedule")
        path, schedule_source = NflverseProvider(args.cache_dir).fetch(refresh=not args.offline)
    raw_games = load_games(path)
    targets = raw_games.loc[raw_games["game_id"].isin(args.game_id)]
    if set(targets["game_id"]) != set(args.game_id):
        raise ValueError("Every requested --game-id must exist in the schedule")
    if targets["season"].nunique() != 1:
        raise ValueError("A forecast batch must belong to one NFL season")
    season = int(targets["season"].iloc[0])
    raw_games = raw_games.loc[raw_games["season"].le(season)].copy()
    snapshots = {}
    if args.model_version == "v2":
        from predictions.pbp import PbpProvider

        if not 1 <= args.workers <= 8:
            raise ValueError("--workers must be between 1 and 8")
        provider = PbpProvider(args.pbp_dir)

        def fetch(year):
            return year, provider.fetch(
                year, refresh=not args.offline and year == season, offline=args.offline
            )

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            snapshots = dict(pool.map(fetch, sorted(int(y) for y in raw_games["season"].unique())))
    as_of = pd.Timestamp(args.as_of) if args.as_of else _now()
    if pd.isna(as_of) or as_of.tzinfo is None:
        raise ValueError("Live --as-of must include a timezone")
    as_of = as_of.tz_convert("UTC")
    observed = (
        pd.Timestamp(schedule_source["fetched_at"]) if schedule_source.get("fetched_at") else as_of
    )
    if pd.isna(observed) or observed.tzinfo is None:
        raise ValueError("Schedule fetched_at must include a timezone")
    games = load_games(path, as_of=min(as_of, observed))
    games = games.loc[games["season"].le(season)].copy()
    metadata = {
        "schedule": schedule_source,
        "data_mode": "cached_snapshots" if args.offline else "fresh_public_sources",
        "feature_window": args.window,
    }
    team_games = qb_games = None
    if snapshots:
        from predictions.causal_ep import reconstruct_epa
        from predictions.pbp import aggregate_pbp

        pbp_games = games.copy()
        for year, (_, source) in snapshots.items():
            observed = pd.Timestamp(source["fetched_at"])
            if pd.isna(observed) or observed.tzinfo is None:
                raise ValueError("PBP fetched_at must include a timezone")
            invalid = pbp_games["season"].eq(year) & pbp_games["result_available_at"].ge(observed)
            pbp_games.loc[invalid, ["home_score", "away_score"]] = float("nan")
        paths = {year: item[0] for year, item in snapshots.items()}
        overrides, ep_metadata = reconstruct_epa(paths, args.ep_dir, games=pbp_games)
        eligible_ids = set(pbp_games.loc[pbp_games["home_score"].notna(), "game_id"])
        teams, quarterbacks = [], []
        for year, pbp_path in paths.items():
            team, qb = aggregate_pbp(pbp_path, epa_override=overrides[year])
            teams.append(team.loc[team["game_id"].isin(eligible_ids)])
            quarterbacks.append(qb.loc[qb["game_id"].isin(eligible_ids)])
        team_games = pd.concat(teams, ignore_index=True)
        qb_games = pd.concat(quarterbacks, ignore_index=True)
        metadata.update(
            play_by_play={str(year): item[1] for year, item in snapshots.items()},
            causal_ep=ep_metadata,
        )
    features = prepare_pregame_features(
        games,
        as_of=as_of,
        model_version=args.model_version,
        team_games=team_games,
        qb_games=qb_games,
        window=args.window,
    )
    bundle = args.bundle
    requested_samples = getattr(args, "bootstrap_samples", 0)
    if bundle is not None and requested_samples:
        specification = _read(Path(bundle) / "metadata.json").get("uncertainty_ensemble", {})
        if not isinstance(specification, dict) or any(
            specification.get(key) != value
            for key, value in [
                ("bootstrap_samples", requested_samples),
                ("confidence_level", getattr(args, "confidence_level", 0.95)),
                ("seed", getattr(args, "bootstrap_seed", 42)),
            ]
        ):
            raise ValueError("Reused bundle does not match the requested bootstrap uncertainty")
    if bundle is None:
        trained = train_live_model(
            features,
            args.model_dir,
            model_version=args.model_version,
            as_of=as_of,
            season=season,
            source_metadata=metadata,
            bootstrap_samples=getattr(args, "bootstrap_samples", 0),
            confidence_level=getattr(args, "confidence_level", 0.95),
            bootstrap_seed=getattr(args, "bootstrap_seed", 42),
        )
        bundle = Path(trained["bundle_path"])
    batch = predict_pregame(
        bundle,
        features.loc[features["game_id"].isin(args.game_id)].copy(),
        as_of=as_of,
        source_metadata=metadata,
        output_dir=None if getattr(args, "score_validation", None) else args.output_dir,
        include_analysis=getattr(args, "include_analysis", False),
    )
    scoring_validation = getattr(args, "score_validation", None)
    if scoring_validation is not None:
        from predictions.scoring import forecast_scoring

        try:
            scores = forecast_scoring(
                games,
                _read(scoring_validation),
                [r["game_id"] for r in batch["forecasts"]],
                args.model_dir / "scoring",
                as_of=as_of,
                source_metadata=metadata,
            )
            for record in batch["forecasts"]:
                record["score_prediction"] = scores[record["game_id"]]
        except (ValueError, OSError) as exc:
            batch["scoring_error"] = str(exc)
            for record in batch["forecasts"]:
                record["score_unavailable_reason"] = (
                    "Scoring validation or fitting could not be verified; independent winner forecast preserved."
                )
        import uuid

        directory = args.output_dir / ("forecasts-" + uuid.uuid4().hex)
        directory.mkdir(parents=True, exist_ok=False)
        path = directory / "forecasts.json"
        _write(path, batch)
        batch["artifact"] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    ledger = PaperLedger(args.ledger)
    for forecast in batch["forecasts"]:
        ledger.record_forecast(forecast)
    batch["bundle_path"] = str(bundle)
    batch["ledger_path"] = str(args.ledger)
    return batch


def _pick_forecast(payload: dict, game_id: str | None) -> dict:
    records = payload.get("forecasts", [payload])
    if not isinstance(records, list) or any(not isinstance(item, dict) for item in records):
        raise ValueError("Forecast file must contain a forecast object or batch")
    if game_id is not None:
        records = [item for item in records if item.get("game_id") == game_id]
    if len(records) != 1:
        raise ValueError("Select exactly one forecast with --game-id")
    return records[0]


def dispatch(args) -> dict | None:
    """Return None for an existing historical command, otherwise its JSON result."""
    commands = {
        "forecast",
        "us-markets",
        "quote",
        "market-settlement",
        "ev",
        "paper-fill",
        "paper-settle",
        "paper-audit",
    }
    if args.command not in commands:
        return None
    from predictions.ledger import PaperLedger
    from predictions.markets import PolymarketUSProvider

    try:
        if args.command == "forecast":
            return _forecast(args)
        if args.command == "us-markets":
            value = {
                "venue": "polymarket_us",
                "markets": PolymarketUSProvider(args.cache_dir).list_nfl_markets(
                    args.limit, args.offset
                ),
            }
        elif args.command == "quote":
            value = PolymarketUSProvider(args.cache_dir).get_quote(args.market, args.outcome)
            PaperLedger(args.ledger).record_quote(value)
        elif args.command == "market-settlement":
            value = PolymarketUSProvider(args.cache_dir).get_settlement(args.market)
        elif args.command == "ev":
            from predictions.ev import evaluate_expected_value

            forecast, quote = _pick_forecast(_read(args.forecast), args.game_id), _read(args.quote)
            value = evaluate_expected_value(
                forecast,
                quote,
                selected_team=args.team,
                quantity=args.quantity,
                as_of=_now(),
                tie_probability=args.tie_probability,
                completion_probability=args.completion_probability,
                cancellation_settlement_price=args.cancellation_settlement_price,
                max_quote_age_seconds=args.max_quote_age_seconds,
                max_forecast_age_seconds=args.max_forecast_age_seconds,
                min_ev_per_contract=args.min_ev_per_contract,
            )
            ledger = PaperLedger(args.ledger)
            ledger.record_forecast(forecast)
            ledger.record_quote(quote)
            ledger.record_evaluation(value)
        elif args.command == "paper-fill":
            return PaperLedger(args.ledger).paper_fill(
                args.evaluation_id,
                idempotency_key=args.idempotency_key,
                max_exposure=args.max_exposure,
            )
        elif args.command == "paper-settle":
            evidence = _read(args.evidence)
            if "source" in evidence and "settlement" in evidence["source"]:
                evidence = {**evidence, "reference": evidence["source"]["settlement"]["url"]}
            return PaperLedger(args.ledger).settle(
                args.trade_id,
                settlement_price=args.price,
                settled_at=evidence.get("observed_at", _now()),
                evidence=evidence,
                idempotency_key=args.idempotency_key,
            )
        else:
            ledger = PaperLedger(args.ledger)
            value = ledger.audit(expected_head_hash=args.expected_head_hash)
            value["summary"] = ledger.summary()
        _write(getattr(args, "output", None), value)
        return value
    except sqlite3.Error as exc:
        raise ValueError(f"Paper ledger rejected the operation: {exc}") from exc
