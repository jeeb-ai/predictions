"""Pregame PBP form, observed-quarterback proxies, and online team strength.

Every source game must have a completed schedule result with an availability
strictly earlier than the target kickoff. The caller must also use causal EPA
estimates: the strict Model 2 pipeline computes EPA with older-season EP fits,
rather than assuming retrospectively published nflverse EPA was known then.
"""

from bisect import bisect_left
from dataclasses import dataclass

import numpy as np
import pandas as pd

from predictions.features import FEATURE_COLUMNS, build_features

ADVANCED_FEATURE_COLUMNS = [
    "off_epa_per_play_diff",
    "def_epa_per_play_diff",
    "off_success_rate_diff",
    "def_success_rate_diff",
    "off_explosive_rate_diff",
    "def_explosive_rate_diff",
    "qb_epa_per_dropback_diff",
    "qb_success_rate_diff",
    "elo_strength_diff",
    "off_history_plays_diff",
    "def_history_plays_diff",
    "qb_history_dropbacks_diff",
    "qb_proxy_missing_diff",
]
MODEL2_FEATURE_COLUMNS = [*FEATURE_COLUMNS, *ADVANCED_FEATURE_COLUMNS]
FEATURE_VERSION = "pregame-pbp-v2"


@dataclass(frozen=True)
class _PlayGame:
    game_id: str
    kickoff: pd.Timestamp
    available_at: pd.Timestamp
    samples: int
    epa_sum: float
    successes: int
    explosive_plays: int = 0
    qb_id: str | None = None


def _clean_aggregates(frame: pd.DataFrame, *, quarterback: bool) -> pd.DataFrame:
    sample = "dropbacks" if quarterback else "plays"
    keys = ["game_id", "team", "qb_id" if quarterback else "opponent"]
    required = {*keys, sample, "epa_sum", "successes"}
    if not quarterback:
        required.add("explosive_plays")
    missing = required - set(frame)
    if missing:
        raise ValueError(
            f"Missing {'QB' if quarterback else 'team'} aggregate columns: {sorted(missing)}"
        )
    result = frame[list(sorted(required))].copy()
    for field in keys:
        if result[field].isna().any() or result[field].astype(str).str.strip().eq("").any():
            raise ValueError(f"PBP aggregates contain missing {field}")
        result[field] = result[field].astype(str)
    unique = ["game_id", "team", "qb_id"] if quarterback else ["game_id", "team"]
    if result.duplicated(unique).any():
        raise ValueError("Duplicate PBP aggregate identities are not allowed")
    for field in [sample, "successes", *([] if quarterback else ["explosive_plays"])]:
        values = pd.to_numeric(result[field], errors="raise")
        if (
            not np.isfinite(values).all()
            or (values < 0).any()
            or (values != np.floor(values)).any()
        ):
            raise ValueError(f"PBP aggregate {field} must be finite nonnegative integers")
        result[field] = values.astype(int)
    if (result[sample] <= 0).any():
        raise ValueError(f"PBP aggregate {sample} must be positive; omit empty groups")
    for field in ["successes", *([] if quarterback else ["explosive_plays"])]:
        if (result[field] > result[sample]).any():
            raise ValueError(f"PBP aggregate {field} exceeds {sample}")
    result["epa_sum"] = pd.to_numeric(result["epa_sum"], errors="raise")
    if not np.isfinite(result["epa_sum"]).all():
        raise ValueError("PBP aggregate epa_sum must be finite")
    return result


def _join_schedule(aggregates: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    # Out-of-scope/cancelled games, such as 2022 BUF-CIN, never become history.
    joined = aggregates.merge(
        schedule[
            ["game_id", "home_team", "away_team", "kickoff", "result_available_at", "home_score"]
        ],
        on="game_id",
        how="inner",
        validate="many_to_one",
    )
    home = joined["team"].eq(joined["home_team"])
    away = joined["team"].eq(joined["away_team"])
    if not (home | away).all():
        raise ValueError("PBP aggregate team does not match the scheduled game")
    if "opponent" in joined:
        expected = joined["away_team"].where(home, joined["home_team"])
        if joined["opponent"].ne(expected).any():
            raise ValueError("PBP aggregate opponent does not match the scheduled game")
    return joined.loc[joined["home_score"].notna()].copy()


def _sorted_histories(histories: dict[str, list[_PlayGame]]) -> dict[str, list[pd.Timestamp]]:
    for history in histories.values():
        history.sort(key=lambda row: (row.available_at, row.kickoff, row.game_id))
    return {key: [row.available_at for row in history] for key, history in histories.items()}


def _eligible(
    histories: dict[str, list[_PlayGame]],
    clocks: dict[str, list[pd.Timestamp]],
    key: str,
    cutoff: pd.Timestamp,
    window: int,
) -> list[_PlayGame]:
    stop = bisect_left(clocks.get(key, []), cutoff)
    eligible = histories.get(key, [])[:stop]
    return sorted(eligible, key=lambda row: (row.kickoff, row.game_id))[-window:]


def _rates(history: list[_PlayGame]) -> dict:
    samples = sum(row.samples for row in history)
    return {
        "epa": sum(row.epa_sum for row in history) / samples if samples else 0.0,
        "success": sum(row.successes for row in history) / samples if samples else 0.45,
        "explosive": sum(row.explosive_plays for row in history) / samples if samples else 0.10,
        "samples": samples,
        "games": len(history),
    }


def build_features_v2(
    games: pd.DataFrame,
    team_games: pd.DataFrame,
    qb_games: pd.DataFrame,
    window: int = 8,
) -> pd.DataFrame:
    """Add strictly lagged, play-weighted features while retaining V1 inputs.

    PBP windows contain the last ``window`` available games with eligible PBP.
    The QB proxy is the most-used passer in a team's last *observed* QB game;
    ties in dropbacks are resolved by QB ID. Its form follows that player across
    teams over the last ``window`` available appearances. Actual target-game
    starters, depth charts, and target-game PBP coverage are never inspected.

    Opponent adjustment is online Elo (1500 initial, K=20, 50 home points,
    400-point scale), updated only with available completed results. No season
    outcome totals or full-season opponent averages enter team strength.
    """
    frame = build_features(games, window=window)
    frame["base_feature_version"] = frame["feature_version"]
    teams = _clean_aggregates(team_games, quarterback=False)
    quarterbacks = _clean_aggregates(qb_games, quarterback=True)
    source_team_rows = len(teams)
    source_qb_rows = len(quarterbacks)
    teams = _join_schedule(teams, frame)
    quarterbacks = _join_schedule(quarterbacks, frame)

    offense: dict[str, list[_PlayGame]] = {}
    defense: dict[str, list[_PlayGame]] = {}
    qb_history: dict[str, list[_PlayGame]] = {}
    team_qbs: dict[str, list[_PlayGame]] = {}
    for row in teams.itertuples(index=False):
        record = _PlayGame(
            row.game_id,
            row.kickoff,
            row.result_available_at,
            row.plays,
            row.epa_sum,
            row.successes,
            row.explosive_plays,
        )
        offense.setdefault(row.team, []).append(record)
        defense.setdefault(row.opponent, []).append(record)
    for row in quarterbacks.itertuples(index=False):
        record = _PlayGame(
            row.game_id,
            row.kickoff,
            row.result_available_at,
            row.dropbacks,
            row.epa_sum,
            row.successes,
            qb_id=row.qb_id,
        )
        qb_history.setdefault(row.qb_id, []).append(record)
    dominant = quarterbacks.sort_values(
        ["game_id", "team", "dropbacks", "qb_id"],
        ascending=[True, True, False, True],
        kind="stable",
    ).drop_duplicates(["game_id", "team"])
    for row in dominant.itertuples(index=False):
        team_qbs.setdefault(row.team, []).append(
            _PlayGame(
                row.game_id,
                row.kickoff,
                row.result_available_at,
                row.dropbacks,
                row.epa_sum,
                row.successes,
                qb_id=row.qb_id,
            )
        )
    off_clocks = _sorted_histories(offense)
    def_clocks = _sorted_histories(defense)
    qb_clocks = _sorted_histories(qb_history)
    proxy_clocks = _sorted_histories(team_qbs)
    completed = frame.loc[frame["home_score"].notna()].sort_values(
        ["result_available_at", "game_id"], kind="stable"
    )
    events = list(completed.itertuples(index=False))
    next_event = 0
    ratings: dict[str, float] = {}
    rating_clocks: dict[str, pd.Timestamp] = {}
    derived = []
    for game in frame.itertuples(index=False):
        while next_event < len(events) and events[next_event].result_available_at < game.kickoff:
            previous = events[next_event]
            home_rating = ratings.get(previous.home_team, 1500.0)
            away_rating = ratings.get(previous.away_team, 1500.0)
            home_field = 0.0 if previous.neutral else 50.0
            expected = 1 / (1 + 10 ** ((away_rating - home_rating - home_field) / 400))
            observed = (
                0.5
                if previous.home_score == previous.away_score
                else float(previous.home_score > previous.away_score)
            )
            change = 20.0 * (observed - expected)
            ratings[previous.home_team] = home_rating + change
            ratings[previous.away_team] = away_rating - change
            rating_clocks[previous.home_team] = previous.result_available_at
            rating_clocks[previous.away_team] = previous.result_available_at
            next_event += 1
        sides = []
        metadata = {}
        dependencies = []
        for side, team in [("home", game.home_team), ("away", game.away_team)]:
            off = _eligible(offense, off_clocks, team, game.kickoff, window)
            defense_history = _eligible(defense, def_clocks, team, game.kickoff, window)
            proxy = _eligible(team_qbs, proxy_clocks, team, game.kickoff, 1)
            quarterback = proxy[0].qb_id if proxy else None
            qb = (
                _eligible(qb_history, qb_clocks, quarterback, game.kickoff, window) if proxy else []
            )
            off_stats, def_stats, qb_stats = _rates(off), _rates(defense_history), _rates(qb)
            sides.append(
                {
                    "off_epa_per_play": off_stats["epa"],
                    "def_epa_per_play": def_stats["epa"],
                    "off_success_rate": off_stats["success"],
                    "def_success_rate": def_stats["success"],
                    "off_explosive_rate": off_stats["explosive"],
                    "def_explosive_rate": def_stats["explosive"],
                    "qb_epa_per_dropback": qb_stats["epa"],
                    "qb_success_rate": qb_stats["success"],
                    "elo_strength": ratings.get(team, 1500.0) / 400,
                    "off_history_plays": off_stats["samples"],
                    "def_history_plays": def_stats["samples"],
                    "qb_history_dropbacks": qb_stats["samples"],
                    "qb_proxy_missing": float(not proxy),
                }
            )
            metadata.update(
                {
                    f"{side}_qb_id_proxy": quarterback,
                    f"{side}_qb_proxy_game_id": proxy[0].game_id if proxy else None,
                    f"{side}_qb_proxy_available_at": proxy[0].available_at if proxy else pd.NaT,
                    f"{side}_off_history_games": off_stats["games"],
                    f"{side}_def_history_games": def_stats["games"],
                    f"{side}_qb_history_games": qb_stats["games"],
                    f"{side}_off_history_plays": off_stats["samples"],
                    f"{side}_def_history_plays": def_stats["samples"],
                    f"{side}_qb_history_dropbacks": qb_stats["samples"],
                    f"{side}_elo_rating": ratings.get(team, 1500.0),
                }
            )
            dependencies.extend(row.available_at for row in [*off, *defense_history, *qb, *proxy])
            if team in rating_clocks:
                dependencies.append(rating_clocks[team])
        home, away = sides
        derived.append(
            {
                **{f"{key}_diff": home[key] - away[key] for key in home},
                **metadata,
                "advanced_history_available_through": max(dependencies) if dependencies else pd.NaT,
            }
        )
    additions = pd.DataFrame(derived, index=frame.index)
    for name in ADVANCED_FEATURE_COLUMNS:
        frame[name] = additions[name] if name in additions else pd.Series(dtype=float)
    audit_columns = [
        "advanced_history_available_through",
        "home_qb_proxy_available_at",
        "away_qb_proxy_available_at",
    ]
    for name in additions:
        if name not in ADVANCED_FEATURE_COLUMNS:
            frame[name] = additions[name]
    for name in audit_columns:
        frame[name] = pd.to_datetime(
            frame.get(name, pd.Series(dtype="datetime64[ns, UTC]")), utc=True
        )
    frame["feature_version"] = f"{FEATURE_VERSION}-w{window}"
    frame.attrs["pbp_join_diagnostics"] = {
        "source_team_game_rows": source_team_rows,
        "source_qb_game_rows": source_qb_rows,
        "eligible_completed_team_game_rows": len(teams),
        "eligible_completed_qb_game_rows": len(quarterbacks),
    }
    return frame
