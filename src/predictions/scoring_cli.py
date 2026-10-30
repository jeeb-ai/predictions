"""Scoring evaluation and explicit-line research; independent of the winner engine."""

import json
from pathlib import Path

import pandas as pd

from predictions.features import build_features
from predictions.scoring import (
    backtest_scoring,
    outcome_probabilities,
)


def add_commands(sub):
    evaluate = sub.add_parser(
        "score-backtest", help="Compare independent NFL scoring distributions chronologically"
    )
    evaluate.add_argument("--input", type=Path, required=True)
    evaluate.add_argument("--as-of", required=True)
    evaluate.add_argument("--through-season", type=int, default=2025)
    evaluate.add_argument("--output-dir", type=Path, default=Path("reports/scoring"))
    forecast = sub.add_parser(
        "score-forecast", help="Produce a timestamped independently modeled score forecast"
    )
    forecast.add_argument("--input", type=Path, required=True)
    forecast.add_argument("--validation", type=Path, required=True)
    forecast.add_argument("--game-id", required=True)
    forecast.add_argument("--home-spread", type=float, default=0)
    forecast.add_argument("--total-line", type=float, default=44.5)
    forecast.add_argument("--output-dir", type=Path, default=Path("reports/score-forecasts"))


def dispatch(args):
    if args.command not in {"score-backtest", "score-forecast"}:
        return None

    from predictions.cli import _local_metadata
    from predictions.data import load_games

    source = _local_metadata(args.input)
    cutoff = (
        pd.Timestamp(args.as_of) if args.command == "score-backtest" else pd.Timestamp.now(tz="UTC")
    )
    if pd.isna(cutoff) or cutoff.tzinfo is None:
        raise ValueError("Scoring cutoff must include a timezone")
    observed = pd.Timestamp(source["fetched_at"]) if source.get("fetched_at") else cutoff
    games = load_games(args.input, as_of=min(cutoff, observed))
    features = build_features(games)
    source = {
        **source,
        "requested_as_of": cutoff.isoformat(),
        "as_of": min(cutoff, observed).isoformat(),
    }
    if args.command == "score-backtest":
        complete = games.loc[games.season.eq(args.through_season) & games.game_type.eq("REG")]
        if complete.empty or complete.home_score.isna().any():
            raise ValueError("Scoring holdout season is missing or incomplete")
        report, _ = backtest_scoring(
            features.loc[features.season.le(args.through_season)],
            args.output_dir,
            source_metadata=source,
            holdout=args.through_season,
        )
        return report
    report = json.loads(args.validation.read_text())
    if (
        report.get("type") != "nfl_scoring_validation"
        or report.get("status") != "evaluated_not_a_profit_claim"
    ):
        raise ValueError("A completed chronological scoring validation is required")
    from predictions.scoring import forecast_scoring, read_score_distribution

    score = forecast_scoring(
        games, report, [args.game_id], args.output_dir, as_of=cutoff, source_metadata=source
    )[args.game_id]
    score["exact_lines"] = {
        "home_spread": args.home_spread,
        "total_line": args.total_line,
        **outcome_probabilities(
            read_score_distribution(score), home_spread=args.home_spread, total_line=args.total_line
        ),
    }
    return score
