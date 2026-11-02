"""Read-only official result adapters. Outcome observations never feed prediction features."""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd

from predictions.data import TEAM_ALIASES
from predictions.league_data import DownloadError, snapshot
from predictions.prospective import stamp


def _json(content):
    value = json.loads(content)
    if not isinstance(value, dict):
        raise ValueError("Official feed must be a JSON object")
    return value


def _source(path, meta, provider):
    return {
        **meta,
        "path": str(path.resolve()),
        "provider": provider,
        "classification": "official_league_data",
    }


def _event(sport, identity, home, away, kickoff, status, source, hs=None, aws=None, kind=None):
    return {
        "sport": sport,
        "game_id": str(identity),
        "home_team": home,
        "away_team": away,
        "kickoff": stamp(kickoff).isoformat(),
        "status": status,
        "home_score": hs,
        "away_score": aws,
        "outcome_type": kind,
        "observed_at": source["fetched_at"],
        "source": source,
    }


class _FlightHTML(HTMLParser):
    def __init__(self):
        super().__init__()
        self.inside = False
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag == "script":
            self.inside = True

    def handle_endtag(self, tag):
        if tag == "script":
            self.inside = False

    def handle_data(self, text):
        if self.inside and "self.__next_f.push(" in text:
            try:
                data = json.loads(text.split("self.__next_f.push(", 1)[1].rsplit(")", 1)[0])
                if data[0] == 1 and isinstance(data[1], str):
                    self.parts.append(data[1])
            except (ValueError, IndexError, TypeError):
                pass


def nfl_cards(content):
    """Decode publicly embedded React data, never execute JavaScript or private APIs."""
    parser = _FlightHTML()
    parser.feed(content.decode("utf-8"))
    frames = {}
    for line in "".join(parser.parts).split("\n"):
        key, sep, body = line.partition(":")
        if sep:
            try:
                frames[key] = json.loads(body)
            except ValueError:
                pass

    def resolve(value):
        seen = set()
        while isinstance(value, str) and value.startswith("$"):
            if value == "$undefined":
                return None
            if value in seen or len(seen) > 20:
                raise ValueError("Cyclic NFL score reference")
            seen.add(value)
            key, *path = value[1:].split(":")
            if key not in frames:
                raise ValueError("Unsupported NFL score reference")
            value = frames[key]
            for part in path:
                if part == "props" and isinstance(value, list) and len(value) == 4:
                    value = value[3]
                elif isinstance(value, list):
                    value = value[int(part)]
                else:
                    value = value[part]
        return value

    cards = []

    def walk(value):
        if isinstance(value, dict):
            if {"gameId", "gameTime", "homeTeam", "awayTeam", "href", "gameSummary"}.issubset(
                value
            ):
                cards.append(value)
            if {"id", "time", "season", "homeTeam", "awayTeam", "externalIds", "summary"}.issubset(
                value
            ):
                slug = next(
                    (v["id"] for v in value["externalIds"] if v.get("source") == "slug"), None
                )
                home, away = resolve(value["homeTeam"]), resolve(value["awayTeam"])

                def abbreviation(team):
                    code = team.get("abbreviation")
                    if not code:
                        code = str(team.get("currentLogo", "")).rsplit("/", 1)[-1]
                    return code

                if slug and abbreviation(home) and abbreviation(away):
                    cards.append(
                        {
                            "gameId": value["id"],
                            "gameTime": value["time"],
                            "href": "/games/" + slug,
                            "homeTeam": {"abbreviation": abbreviation(home)},
                            "awayTeam": {"abbreviation": abbreviation(away)},
                            "gameSummary": value["summary"],
                        }
                    )
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(frames)
    output = {}
    for card in cards:
        match = re.search(r"-(\d{4})-(reg|post)-(\d+)$", str(card["href"]))
        if not match:
            continue
        season, kind, week = match.groups()
        week = int(week) + (18 if kind == "post" else 0)
        h, a = resolve(card["homeTeam"]), resolve(card["awayTeam"])
        if not isinstance(h, dict) or not isinstance(a, dict):
            continue
        home = {**TEAM_ALIASES, "AZ": "ARI"}.get(h["abbreviation"], h["abbreviation"])
        away = {**TEAM_ALIASES, "AZ": "ARI"}.get(a["abbreviation"], a["abbreviation"])
        identity = f"{season}_{week:02}_{away}_{home}"
        summary = resolve(card["gameSummary"])
        phase = summary.get("phase") if isinstance(summary, dict) else None
        status = {
            "FINAL": "final",
            "CANCELLED": "cancelled",
            "CANCELED": "cancelled",
            "POSTPONED": "postponed",
        }.get(phase, "scheduled")
        scores = [None, None]
        if status == "final":
            if summary.get("gameId", card["gameId"]) != card["gameId"]:
                raise ValueError("NFL summary belongs to another game")
            for index, side in enumerate(["homeTeam", "awayTeam"]):
                team = resolve(summary[side])
                score = resolve(team["score"])
                total = score["total"]
                if (
                    not isinstance(total, int)
                    or isinstance(total, bool)
                    or total < 0
                    or sum(score.get(p, 0) or 0 for p in ["q1", "q2", "q3", "q4", "ot"]) != total
                ):
                    raise ValueError("NFL official score total does not reconcile")
                scores[index] = total
        value = {
            "game_id": identity,
            "home_team": home,
            "away_team": away,
            "kickoff": stamp(card["gameTime"]).isoformat(),
            "status": status,
            "home_score": scores[0],
            "away_score": scores[1],
            "outcome_type": "REG_OR_OT",
            "official_game_id": card["gameId"],
        }
        if identity in output and output[identity] != value:
            raise ValueError("Contradictory NFL official scoreboard entries")
        output[identity] = value
    if not output:
        raise ValueError("NFL official scoreboard format unavailable; no results inferred")
    return output


def nfl_results(targets, root):
    output = []
    weeks = sorted({tuple(str(g["game_id"]).split("_")[:2]) for g in targets})
    for season, week in weeks:
        number = int(week)
        route = (
            f"{season}/week-{number}"
            if number <= 18
            else f"{season}/post-season/week-{number - 18}"
        )
        path, meta = snapshot(
            "https://www.nfl.com/scores/" + route,
            Path(root) / "nfl",
            refresh=True,
            attempts=3,
            timeout=(8, 20),
            validator=nfl_cards,
            label=f"NFL official results {season} week{week}",
        )
        source = _source(path, meta, "NFL.com public official scoreboard")
        cards = nfl_cards(path.read_bytes())
        for game in targets:
            if tuple(str(game["game_id"]).split("_")[:2]) != (season, week):
                continue
            value = cards.get(game["game_id"])
            if value:
                output.append(
                    {**value, "sport": "NFL", "source": source, "observed_at": source["fetched_at"]}
                )
    return output


def nba_cdn_results(content, source):
    payload = _json(content)
    if not isinstance(payload.get("leagueSchedule"), dict):
        raise ValueError("NBA official result schema unavailable")
    data = payload["leagueSchedule"]
    output = {}
    for day in data["gameDates"]:
        for g in day["games"]:
            if not str(g["gameId"]).startswith("002"):
                continue
            h, a = g["homeTeam"], g["awayTeam"]
            state = "final" if g["gameStatus"] == 3 else "scheduled"
            text = str(g.get("gameStatusText", "")).lower()
            if g.get("postponedStatus") == "P" or "postponed" in text:
                state = "postponed"
            if "cancel" in text:
                state = "cancelled"
            row = _event(
                "NBA",
                g["gameId"],
                h["teamTricode"],
                a["teamTricode"],
                g["gameDateTimeUTC"],
                state,
                source,
                h.get("score") if state == "final" else None,
                a.get("score") if state == "final" else None,
                "FULL_OT",
            )
            if row["game_id"] in output:
                raise ValueError("Duplicate NBA official game ID")
            output[row["game_id"]] = row
    if not output:
        raise ValueError("Empty NBA official schedule")
    return output


def nba_results(targets, root):
    from predictions.nba_history import CDN_SCHEDULE, schedule_table

    output = {}
    try:
        path, meta = snapshot(
            CDN_SCHEDULE,
            Path(root) / "nba",
            refresh=True,
            validator=_json,
            attempts=3,
            timeout=(8, 20),
            label="NBA official results",
        )
        source = _source(path, meta, "NBA official CDN schedule")
        output.update(nba_cdn_results(path.read_bytes(), source))
    except (DownloadError, ValueError, KeyError, TypeError):
        pass
    for year in sorted(
        {2000 + int(g["game_id"][3:5]) for g in targets if g["game_id"] not in output}
    ):
        from predictions.nba_history import _espn_day, _match_key, release_table

        table, meta = schedule_table(year, Path(root) / "nba", refresh=True)
        espn, em = release_table(
            "espn_nba_schedules",
            f"nba_schedule_{year + 1}.parquet",
            Path(root) / "nba",
            refresh=True,
        )
        known = {}
        for value in espn.loc[pd.to_numeric(espn.season_type).eq(2)].to_dict("records"):
            key = _match_key(
                _espn_day(value), value["home_abbreviation"], value["away_abbreviation"]
            )
            if key[1] == "TBD" or key[2] == "TBD":
                continue
            if key in known:
                raise ValueError("Ambiguous NBA result schedule match")
            known[key] = value
        source = {
            **meta,
            "provider": "NBA Stats official-origin schedule plus ESPN confirmed tipoff",
            "classification": "official_stats_archive",
            "components": {"official": meta, "tipoff": em},
            "path": str(
                (Path(root) / "nba" / "archive" / "nba_stats_schedules" / meta["file"]).resolve()
            ),
        }
        source["fetched_at"] = max(stamp(meta["fetched_at"]), stamp(em["fetched_at"])).isoformat()
        wanted = {g["game_id"] for g in targets}
        for game in table.to_dict("records"):
            if game["game_id"] not in wanted:
                continue
            other = known.get(
                _match_key(
                    game["game_date"],
                    game["home_team_abbreviation"],
                    game["away_team_abbreviation"],
                )
            )
            if other is None:
                continue
            state = (
                "final" if game["finished"] and other.get("status_type_completed") else "scheduled"
            )
            if state == "final" and (
                game["home_pts"] != other["home_score"] or game["away_pts"] != other["away_score"]
            ):
                state = "disputed"
            output[game["game_id"]] = _event(
                "NBA",
                game["game_id"],
                game["home_team_abbreviation"],
                game["away_team_abbreviation"],
                other["date"],
                state,
                source,
                int(game["home_pts"]) if state == "final" else None,
                int(game["away_pts"]) if state == "final" else None,
                "FULL_OT",
            )
    return [output[g["game_id"]] for g in targets if g["game_id"] in output]


def nhl_landing(content, source):
    g = _json(content)
    if g.get("gameType") != 2:
        raise ValueError("Unsupported NHL result game type")
    h, a = g["homeTeam"], g["awayTeam"]
    declared = (g.get("gameOutcome") or {}).get("lastPeriodType")
    descriptor = (g.get("periodDescriptor") or {}).get("periodType")
    kind = declared or descriptor
    state = "final" if g["gameState"] in ["OFF", "FINAL"] else "scheduled"
    if g.get("gameScheduleState") == "PPD":
        state = "postponed"
    if g.get("gameScheduleState") in ["CNCL", "CANCELLED"]:
        state = "cancelled"
    if state == "final" and declared and descriptor and declared != descriptor:
        state = "disputed"
    elif state == "final" and kind not in ["REG", "OT", "SO"]:
        state = "incomplete"
    return _event(
        "NHL",
        g["id"],
        h["abbrev"],
        a["abbrev"],
        g["startTimeUTC"],
        state,
        source,
        h.get("score") if state == "final" else None,
        a.get("score") if state == "final" else None,
        kind,
    )


def nhl_results(targets, root):
    def one(game):
        path, meta = snapshot(
            "https://api-web.nhle.com/v1/gamecenter/" + game["game_id"] + "/landing",
            Path(root) / "nhl",
            refresh=True,
            validator=_json,
            attempts=3,
            timeout=(8, 20),
            label="NHL official result " + game["game_id"],
        )
        return nhl_landing(path.read_bytes(), _source(path, meta, "NHL official gamecenter"))

    def safe(game):
        try:
            return one(game)
        except (DownloadError, ValueError, KeyError, TypeError):
            return {"sport": "NHL", "game_id": game["game_id"], "status": "provider_error"}

    with ThreadPoolExecutor(max_workers=4) as pool:
        return list(pool.map(safe, targets))


def fetch_results(sport, targets, root):
    return {"NFL": nfl_results, "NBA": nba_results, "NHL": nhl_results}[sport](targets, root)
