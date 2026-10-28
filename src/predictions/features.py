"""Transparent pregame form features with explicit information cutoffs.

Only a game's identity, schedule, and *previously available* scores are used.
The nflverse snapshot is retrospective rather than a point-in-time archive;
``result_available_at`` is a conservative ingestion assumption, not a claim
that a historical response was observed at that timestamp.
"""

from bisect import bisect_left
from dataclasses import dataclass

import numpy as np
import pandas as pd

FEATURE_VERSION = "rolling-form-v1"
FEATURE_COLUMNS = [
    "win_rate_diff",
    "margin_diff",
    "points_for_diff",
    "points_against_diff",
    "rest_days_diff",
    "games_played_diff",
    "home_advantage",
]


@dataclass(frozen=True)
class _PastGame:
    available_at: pd.Timestamp
    kickoff: pd.Timestamp
    game_id: str
    points_for: float
    points_against: float

    @property
    def win_points(self) -> float:
        if self.points_for == self.points_against:
            return 0.5
        return float(self.points_for > self.points_against)


def _form(history: list[_PastGame], cutoff: pd.Timestamp) -> dict:
    if not history:
        return {
            "win_rate": 0.5,
            "margin": 0.0,
            "points_for": 21.0,
            "points_against": 21.0,
            "rest_days": 7.0,
            "games_played": 0.0,
        }
    return {
        "win_rate": float(np.mean([game.win_points for game in history])),
        "margin": float(np.mean([game.points_for - game.points_against for game in history])),
        "points_for": float(np.mean([game.points_for for game in history])),
        "points_against": float(np.mean([game.points_against for game in history])),
        "rest_days": min(
            30.0, (cutoff - max(game.kickoff for game in history)).total_seconds() / 86400
        ),
        "games_played": float(len(history)),
    }


def build_features(games: pd.DataFrame, window: int = 8) -> pd.DataFrame:
    """Return deterministic home-minus-away rolling form for each scheduled game.

    A prior result is eligible only when ``result_available_at < kickoff``.
    Histories include completed regular-season and postseason games, span
    season boundaries, and retain only the last ``window`` eligible games.
    Ties contribute half a historical win but have no binary target label.
    No current scores, market observations, injuries, or postgame statistics
    enter ``FEATURE_COLUMNS``.
    """
    if not isinstance(window, int) or isinstance(window, bool) or window < 1:
        raise ValueError("window must be a positive integer")
    required = {
        "game_id",
        "kickoff",
        "result_available_at",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "neutral",
    }
    missing = required - set(games.columns)
    if missing:
        raise ValueError(f"Missing canonical game columns: {sorted(missing)}")
    frame = games.copy()
    if frame["game_id"].duplicated().any():
        raise ValueError("Duplicate game_id values are not allowed")
    for column in ("kickoff", "result_available_at"):
        if frame[column].isna().any():
            raise ValueError(f"Missing {column} values are not allowed")
        frame[column] = pd.to_datetime(frame[column], utc=True)
    if (frame["result_available_at"] <= frame["kickoff"]).any():
        raise ValueError("Results must become available strictly after their own kickoff")
    if (frame["home_team"] == frame["away_team"]).any():
        raise ValueError("A team cannot play itself")
    partial = frame["home_score"].isna() != frame["away_score"].isna()
    if partial.any():
        raise ValueError("A game must have both scores or neither score")
    frame = frame.sort_values(["kickoff", "game_id"], kind="stable").reset_index(drop=True)
    histories: dict[str, list[_PastGame]] = {}
    for game in frame.itertuples(index=False):
        if pd.isna(game.home_score):
            continue
        for team, points_for, points_against in (
            (game.home_team, game.home_score, game.away_score),
            (game.away_team, game.away_score, game.home_score),
        ):
            histories.setdefault(team, []).append(
                _PastGame(
                    game.result_available_at,
                    game.kickoff,
                    game.game_id,
                    float(points_for),
                    float(points_against),
                )
            )
    availability: dict[str, list[pd.Timestamp]] = {}
    for team, history in histories.items():
        history.sort(key=lambda game: (game.available_at, game.kickoff, game.game_id))
        availability[team] = [game.available_at for game in history]

    rows = []
    for game in frame.itertuples(index=False):
        sides = []
        used_history = []
        source_details = {}
        for team in (game.home_team, game.away_team):
            history = histories.get(team, [])
            # bisect_left deliberately excludes results available at the exact cutoff.
            stop = bisect_left(availability.get(team, []), game.kickoff)
            eligible = history[:stop]
            # Availability order can differ from game order for imputed kickoff times.
            selected = sorted(eligible, key=lambda previous: (previous.kickoff, previous.game_id))[
                -window:
            ]
            side = "home" if team == game.home_team else "away"
            source_details[f"{side}_form_source_game_ids"] = [item.game_id for item in selected]
            source_details[f"{side}_form_available_through"] = max(
                (item.available_at for item in selected), default=None
            )
            sides.append(_form(selected, game.kickoff))
            used_history.extend(selected)
        home, away = sides
        row = {f"{name}_diff": home[name] - away[name] for name in home}
        row.update(
            {
                f"{side}_{name}": value
                for side, stats in [("home", home), ("away", away)]
                for name, value in stats.items()
            }
        )
        row.update(source_details)
        row["home_advantage"] = float(not bool(game.neutral))
        row["history_available_through"] = (
            max(previous.available_at for previous in used_history) if used_history else pd.NaT
        )
        rows.append(row)

    derived = pd.DataFrame(rows, index=frame.index)
    for column in FEATURE_COLUMNS:
        frame[column] = derived[column] if column in derived else pd.Series(dtype=float)
    for name in derived:
        if name.startswith(("home_", "away_")):
            frame[name] = derived[name]
    frame["history_available_through"] = pd.to_datetime(
        derived.get("history_available_through", pd.Series(dtype="datetime64[ns, UTC]")),
        utc=True,
    )
    frame["feature_cutoff_at"] = frame["kickoff"]
    frame["feature_version"] = f"{FEATURE_VERSION}-w{window}"
    completed = frame["home_score"].notna()
    decisive = completed & (frame["home_score"] != frame["away_score"])
    frame["home_win"] = np.nan
    frame.loc[decisive, "home_win"] = (
        frame.loc[decisive, "home_score"] > frame.loc[decisive, "away_score"]
    ).astype(float)
    return frame
