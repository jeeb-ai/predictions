"""NBA-specific possessions, efficiency, schedule rest and lagged form."""

import json

import numpy as np
import pandas as pd

from predictions.league_data import snapshot

NBA_COLUMNS = [
    f"{side}_{name}"
    for side in ["home", "away"]
    for name in [
        "off_rating",
        "def_rating",
        "pace",
        "form",
        "rest",
        "back_to_back",
        "elo",
        "history",
    ]
]


def fetch_nba(root, start=2010, through=2025, refresh=False):
    frames = []
    sources = {}
    for year in range(start, through + 1):
        path, meta = snapshot(
            "https://stats.nba.com/stats/leaguegamefinder",
            root,
            params={
                "LeagueID": "00",
                "Season": f"{year}-{str(year + 1)[-2:]}",
                "SeasonType": "Regular Season",
                "PlayerOrTeam": "T",
            },
            refresh=refresh and year == through,
        )
        data = json.loads(path.read_text())["resultSets"][0]
        frame = pd.DataFrame(data["rowSet"], columns=data["headers"])
        if frame.empty or not frame.SEASON_ID.astype(str).eq("2" + str(year)).all():
            raise ValueError("NBA response season mismatch or empty")
        frame["season"] = year
        frames.append(frame)
        sources[str(year)] = meta
    result = canonical_nba(pd.concat(frames, ignore_index=True))
    result["source_observed_at"] = pd.to_datetime(
        result.season.map(lambda y: sources[str(y)]["fetched_at"]), utc=True
    )
    unsafe = result.available_at.ge(result.source_observed_at)
    result.loc[unsafe, ["home_score", "away_score", "home_efficiency", "away_efficiency"]] = np.nan
    return result, sources


def canonical_nba(frame):
    rows = []
    if frame.duplicated(["GAME_ID", "TEAM_ID"]).any():
        raise ValueError("Duplicate NBA team/game rows")
    for identity, group in frame.groupby("GAME_ID", sort=True):
        if len(group) != 2:
            raise ValueError("NBA game requires both team box scores")
        proposals = []
        for matchup in group.MATCHUP:
            if " vs. " in matchup:
                proposals.append(matchup.split(" vs. ")[0])
            elif " @ " in matchup:
                proposals.append(matchup.split(" @ ")[1])
            else:
                raise ValueError("Unknown NBA matchup syntax")
        if len(set(proposals)) != 1:
            raise ValueError("Contradictory NBA home assignment")
        home = group.loc[group.TEAM_ABBREVIATION.eq(proposals[0])]
        away = group.loc[~group.TEAM_ABBREVIATION.eq(proposals[0])]
        if len(home) != 1 or len(away) != 1:
            raise ValueError("Unverified NBA home/away pairing")
        h, a = home.iloc[0], away.iloc[0]
        completed = "WL" not in group or set(group.WL) == {"W", "L"}
        if (
            h.GAME_DATE != a.GAME_DATE
            or h.season != a.season
            or h.TEAM_ID == a.TEAM_ID
            or (completed and h.PTS == a.PTS)
        ):
            raise ValueError("Invalid NBA matchup/result")
        possession = np.mean([r.FGA - r.OREB + r.TOV + 0.44 * r.FTA for r in [h, a]])
        if completed and (not np.isfinite(possession) or possession <= 0):
            raise ValueError("Invalid estimated NBA possessions")
        if completed and "WL" in group and (h.WL != ("W" if h.PTS > a.PTS else "L")):
            raise ValueError("NBA result contradicts win/loss labels")
        if not completed:
            possession = np.nan
        minutes = 48 + 5 * max(0, round((np.mean([h.MIN, a.MIN]) - 240) / 25))
        day = pd.Timestamp(h.GAME_DATE, tz="UTC")
        rows.append(
            {
                "sport": "NBA",
                "schedule_day": day.date().isoformat(),
                "game_id": str(identity),
                "season": int(h.season),
                "kickoff": day,
                "available_at": day + pd.Timedelta(days=2),
                "home_team": h.TEAM_ABBREVIATION,
                "away_team": a.TEAM_ABBREVIATION,
                "home_name": h.TEAM_NAME,
                "away_name": a.TEAM_NAME,
                "home_score": float(h.PTS) if completed else np.nan,
                "away_score": float(a.PTS) if completed else np.nan,
                "home_efficiency": 100 * h.PTS / possession,
                "away_efficiency": 100 * a.PTS / possession,
                "pace": 48 * possession / minutes,
                "possessions": possession,
                "neutral": False,
                "minutes": minutes,
            }
        )
    return pd.DataFrame(rows).sort_values(["kickoff", "game_id"]).reset_index(drop=True)


def nba_features(games, as_of=None, window=10):
    games = games.sort_values(["kickoff", "game_id"]).copy()
    history = {}
    scheduled = {}
    ratings = {}
    pending = []
    rows = []
    for game in games.to_dict("records"):
        cutoff = min(game["kickoff"], pd.Timestamp(as_of)) if as_of is not None else game["kickoff"]
        # Result observations enter ratings only when available strictly before cutoff.
        ready = [g for g in pending if g["available_at"] < cutoff]
        pending = [g for g in pending if g["available_at"] >= cutoff]
        for past in ready:
            h, a = past["home_team"], past["away_team"]
            rh, ra = ratings.get(h, 1500.0), ratings.get(a, 1500.0)
            expected = 1 / (1 + 10 ** ((ra - rh - 60) / 400))
            change = 20 * (float(past["home_score"] > past["away_score"]) - expected)
            ratings[h] = rh + change
            ratings[a] = ra - change
        row = {**game}
        used = []
        for side, other in [("home", "away"), ("away", "home")]:
            team = game[side + "_team"]
            eligible = [g for g in history.get(team, []) if g["available_at"] < cutoff][-window:]
            own = [
                g[("home" if g["home_team"] == team else "away") + "_efficiency"] for g in eligible
            ]
            against = [
                g[("away" if g["home_team"] == team else "home") + "_efficiency"] for g in eligible
            ]
            wins = [
                float((g["home_score"] > g["away_score"]) == (g["home_team"] == team))
                for g in eligible
            ]
            last = scheduled.get(team)
            day = pd.Timestamp(game.get("schedule_day", game["kickoff"].date()))
            days = (day - last).days if last is not None else 7.0
            values = {
                "off_rating": np.mean(own) if own else 100.0,
                "def_rating": np.mean(against) if against else 100.0,
                "pace": np.mean([g["pace"] for g in eligible]) if eligible else 100.0,
                "form": np.mean(wins) if wins else 0.5,
                "rest": min(14.0, days),
                "back_to_back": float(days < 2),
                "elo": ratings.get(team, 1500.0) / 400,
                "history": len(eligible),
            }
            row.update({side + "_" + key: float(value) for key, value in values.items()})
            row[side + "_source_games"] = [g["game_id"] for g in eligible]
            used.extend(g["available_at"] for g in eligible)
        row["feature_cutoff_at"] = cutoff
        row["history_available_through"] = max(used) if used else None
        rows.append(row)
        if pd.notna(game["home_score"]) and (
            as_of is None or game["available_at"] < pd.Timestamp(as_of)
        ):
            history.setdefault(game["home_team"], []).append(game)
            history.setdefault(game["away_team"], []).append(game)
            pending.append(game)
        scheduled[game["home_team"]] = pd.Timestamp(
            game.get("schedule_day", game["kickoff"].date())
        )
        scheduled[game["away_team"]] = pd.Timestamp(
            game.get("schedule_day", game["kickoff"].date())
        )
    return pd.DataFrame(rows)


def nba_slate(root):
    path, meta = snapshot(
        "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2_1.json", root, refresh=True
    )
    data = json.loads(path.read_text())["leagueSchedule"]
    rows = []
    for date in data["gameDates"]:
        for game in date["games"]:
            if game["gameStatus"] not in [1, 2, 3] or game.get("postponedStatus", "A") == "P":
                continue
            h, a = game["homeTeam"], game["awayTeam"]
            kickoff = pd.Timestamp(game["gameDateTimeUTC"])
            rows.append(
                {
                    "sport": "NBA",
                    "schedule_day": kickoff.tz_convert("America/New_York").date().isoformat(),
                    "game_id": str(game["gameId"]),
                    "season": int(data["seasonYear"][:4]),
                    "kickoff": kickoff,
                    "available_at": kickoff + pd.Timedelta(days=2),
                    "home_team": h["teamTricode"],
                    "away_team": a["teamTricode"],
                    "home_name": h["teamCity"] + " " + h["teamName"],
                    "away_name": a["teamCity"] + " " + a["teamName"],
                    "home_score": np.nan,
                    "away_score": np.nan,
                    "home_efficiency": np.nan,
                    "away_efficiency": np.nan,
                    "pace": np.nan,
                    "neutral": bool(game.get("isNeutral", False)),
                }
            )
    return pd.DataFrame(rows), meta
