"""Validated NBA Stats/ESPN release archives; no model or feature substitutions."""

import io
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from predictions.league_data import DownloadError, cached_snapshot, snapshot

LOG = logging.getLogger(__name__)
STATS_URL = "https://stats.nba.com/stats/leaguegamefinder"
RELEASE_URL = "https://api.github.com/repos/sportsdataverse/sportsdataverse-data/releases/tags/"
CDN_SCHEDULE = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2_1.json"
ALIASES = {
    "GS": "GSW",
    "NY": "NYK",
    "NO": "NOP",
    "SA": "SAS",
    "UTAH": "UTA",
    "WSH": "WAS",
    "PHO": "PHX",
}
BOX_COLUMNS = {
    "GAME_ID",
    "TEAM_ID",
    "TEAM_ABBREVIATION",
    "TEAM_NAME",
    "MATCHUP",
    "GAME_DATE",
    "SEASON_ID",
    "PTS",
    "FGA",
    "OREB",
    "TOV",
    "FTA",
    "MIN",
    "WL",
}


def season_name(year):
    return f"{year}-{str(year + 1)[-2:]}"


def stats_params(year):
    return {
        "LeagueID": "00",
        "Season": season_name(year),
        "SeasonType": "Regular Season",
        "PlayerOrTeam": "T",
    }


def _json(content):
    try:
        value = json.loads(content)
        if not isinstance(value, dict):
            raise ValueError("expected a JSON object")
        return value
    except (ValueError, UnicodeError) as exc:
        raise ValueError("expected valid JSON, not an HTML/error page") from exc


def stats_frame(content, year):
    try:
        data = _json(content)["resultSets"][0]
        frame = pd.DataFrame(data["rowSet"], columns=data["headers"])
        if not BOX_COLUMNS.issubset(frame) or frame.empty:
            raise ValueError("empty or missing required team box-score columns")
        if not frame.SEASON_ID.astype(str).eq("2" + str(year)).all():
            raise ValueError("response season differs from requested season")
        frame["season"] = year
        return frame
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("invalid leaguegamefinder response schema") from exc


def _catalog(content):
    data = _json(content)
    if not isinstance(data.get("assets"), list):
        raise ValueError("GitHub release has no assets array")


def release_table(tag, name, root, refresh=False):
    """End-year assets, pinned to the publisher's SHA256, cached independently."""
    root = Path(root) / "archive"
    index, publication = snapshot(
        RELEASE_URL + tag,
        root / "catalog",
        refresh=refresh,
        validator=_catalog,
        timeout=(8, 25),
        attempts=3,
        label="NBA release catalog " + tag,
    )
    asset = next((a for a in json.loads(index.read_text())["assets"] if a["name"] == name), None)
    if asset is None:
        raise ValueError("Archive has not published " + tag + "/" + name)
    expected = asset.get("digest", "")
    if not isinstance(expected, str) or not expected.startswith("sha256:"):
        raise ValueError("Archive asset has no published SHA256: " + name)

    def parquet(content):
        try:
            pd.read_parquet(io.BytesIO(content))
        except Exception as exc:
            raise ValueError("invalid parquet response for " + name) from exc

    path, meta = snapshot(
        asset["browser_download_url"],
        root / tag,
        refresh=refresh,
        expected_sha256=expected[7:],
        validator=parquet,
        timeout=(8, 30),
        attempts=3,
        label="NBA archive " + name,
    )
    return pd.read_parquet(path), {
        **meta,
        "asset": name,
        "release": tag,
        "release_catalog_sha256": publication["sha256"],
        "source_repository": "https://github.com/sportsdataverse/hoopR-nba-stats-data"
        if tag.startswith("nba_stats")
        else "https://github.com/sportsdataverse/hoopR-nba-data",
    }


def _regular(frame, year, id_column="game_id"):
    if "season" not in frame or not pd.to_numeric(frame.season).eq(year + 1).all():
        raise ValueError("NBA archive end-year differs from requested season")
    if id_column not in frame or frame[id_column].isna().any():
        raise ValueError("Missing NBA game identity")
    ids = frame[id_column].astype(str)
    return frame.loc[ids.str.startswith(f"002{str(year)[-2:]}")].copy()


def schedule_table(year, root, refresh=False):
    frame, meta = release_table(
        "nba_stats_schedules", f"nba_schedule_{year + 1}.parquet", root, refresh
    )
    required = {
        "game_id",
        "season",
        "game_date",
        "home_team_id",
        "away_team_id",
        "home_team_abbreviation",
        "away_team_abbreviation",
        "home_team_name",
        "away_team_name",
        "home_pts",
        "away_pts",
        "home_wl",
        "away_wl",
    }
    if not required.issubset(frame):
        raise ValueError("NBA schedule archive schema mismatch")
    frame = _regular(frame, year)
    if frame.game_id.duplicated().any() or frame.empty:
        raise ValueError("NBA schedule has duplicate IDs or no regular games")
    frame["game_date"] = pd.to_datetime(frame.game_date).dt.strftime("%Y-%m-%d")
    for side in ["home", "away"]:
        frame[side + "_team_abbreviation"] = frame[side + "_team_abbreviation"].replace(ALIASES)
    frame["finished"] = (frame.home_wl.eq("W") & frame.away_wl.eq("L")) | (
        frame.home_wl.eq("L") & frame.away_wl.eq("W")
    )
    closed = pd.Timestamp.now(tz="UTC") >= pd.Timestamp(f"{year + 1}-07-01", tz="UTC")
    if closed:
        # Known regular-season totals include lockout/COVID/cancelled-game exceptions.
        expected = {2011: 990, 2012: 1229, 2019: 1059, 2020: 1080}.get(year, 1230)
        actual = int(frame.finished.sum())
        if actual != expected:
            raise ValueError(
                f"Incomplete {season_name(year)} schedule: {actual} final games; expected {expected}"
            )
    return frame, meta


def _rows_for_box(box, schedule, year):
    """Map v3 box scores to the unchanged leaguegamefinder column semantics."""
    required = {
        "game_id",
        "team_id",
        "team_tricode",
        "side",
        "minutes",
        "points",
        "field_goals_attempted",
        "rebounds_offensive",
        "turnovers",
        "free_throws_attempted",
    }
    if not required.issubset(box):
        raise ValueError("NBA team box archive schema mismatch")
    box = _regular(box, year)
    box = box.loc[box.game_id.isin(schedule.loc[schedule.finished, "game_id"])]
    by_id = schedule.set_index("game_id", drop=False)
    rows = []
    for row in box.to_dict("records"):
        identity = str(row["game_id"])
        if identity not in by_id.index:
            raise ValueError("Box score game absent from official schedule: " + identity)
        game = by_id.loc[identity]
        side = str(row["side"]).lower()
        if side not in ["home", "away"] or int(row["team_id"]) != int(game[side + "_team_id"]):
            raise ValueError("NBA box/schedule team-side mismatch: " + identity)
        code = ALIASES.get(row["team_tricode"], row["team_tricode"])
        if code != game[side + "_team_abbreviation"]:
            raise ValueError("NBA box/schedule team-code mismatch: " + identity)
        pieces = str(row["minutes"]).split(":")
        if len(pieces) != 2 or not 0 <= float(pieces[1]) < 60:
            raise ValueError("Invalid NBA team minutes: " + identity)
        minutes = float(pieces[0]) + float(pieces[1]) / 60
        rows.append(
            _team_row(
                game,
                side,
                year,
                row["points"],
                row["field_goals_attempted"],
                row["rebounds_offensive"],
                row["turnovers"],
                row["free_throws_attempted"],
                minutes,
            )
        )
    return pd.DataFrame(rows)


def _team_row(game, side, year, points, fga, oreb, tov, fta, minutes):
    other = "away" if side == "home" else "home"
    team = game[side + "_team_abbreviation"]
    matchup = team + (" vs. " if side == "home" else " @ ") + game[other + "_team_abbreviation"]
    return {
        "GAME_ID": str(game["game_id"]),
        "TEAM_ID": int(game[side + "_team_id"]),
        "TEAM_ABBREVIATION": team,
        "TEAM_NAME": game[side + "_team_name"],
        "MATCHUP": matchup,
        "GAME_DATE": game["game_date"],
        "SEASON_ID": "2" + str(year),
        "season": year,
        "PTS": points,
        "FGA": fga,
        "OREB": oreb,
        "TOV": tov,
        "FTA": fta,
        "MIN": minutes,
        "WL": game[side + "_wl"],
    }


def espn_tables(year, root, refresh=False):
    scores, sm = release_table(
        "espn_nba_schedules", f"nba_schedule_{year + 1}.parquet", root, refresh
    )
    boxes, bm = release_table(
        "espn_nba_team_boxscores", f"team_box_{year + 1}.parquet", root, refresh
    )
    if "season" not in scores or "season" not in boxes:
        raise ValueError("ESPN archive is missing season provenance")
    if (
        not pd.to_numeric(scores.season).eq(year + 1).all()
        or not pd.to_numeric(boxes.season).eq(year + 1).all()
    ):
        raise ValueError("ESPN archive season mismatch")
    return scores, boxes, [sm, bm]


def _espn_day(row):
    timestamp = pd.Timestamp(row["date"])
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError("ESPN event must have an observed timezone-aware start time")
    return timestamp.tz_convert("America/New_York").date().isoformat()


def _match_key(day, home, away):
    return str(day)[:10], ALIASES.get(home, home), ALIASES.get(away, away)


def espn_rows(scores, boxes, schedule, year):
    required = {
        "game_id",
        "status_type_completed",
        "status_period",
        "game_date",
        "date",
        "season_type",
        "home_abbreviation",
        "away_abbreviation",
        "home_score",
        "away_score",
    }
    if not required.issubset(scores) or not {
        "game_id",
        "team_home_away",
        "team_abbreviation",
        "team_score",
        "field_goals_attempted",
        "offensive_rebounds",
        "turnovers",
        "free_throws_attempted",
    }.issubset(boxes):
        raise ValueError("ESPN box/schedule schema mismatch")
    selected = scores.loc[
        pd.to_numeric(scores.season_type).eq(2) & scores.status_type_completed.eq(True)
    ]
    mapping = {}
    for row in selected.to_dict("records"):
        key = _match_key(_espn_day(row), row["home_abbreviation"], row["away_abbreviation"])
        if key in mapping:
            raise ValueError("Ambiguous ESPN date/team match")
        mapping[key] = row
    by_id = {str(k): v for k, v in boxes.groupby("game_id")}
    output = []
    for game in schedule.loc[schedule.finished].to_dict("records"):
        key = _match_key(
            game["game_date"], game["home_team_abbreviation"], game["away_team_abbreviation"]
        )
        other = mapping.get(key)
        if other is None or str(other["game_id"]) not in by_id:
            continue  # Coverage validation below rejects any missing eligible completed game.
        periods = float(other["status_period"])
        if not np.isfinite(periods) or not periods.is_integer() or not 4 <= periods <= 12:
            raise ValueError("Unknown ESPN regulation/overtime duration")
        pair = by_id[str(other["game_id"])]
        if len(pair) != 2 or set(pair.team_home_away) != {"home", "away"}:
            raise ValueError("ESPN game does not have exactly two team sides")
        for side in ["home", "away"]:
            row = pair.loc[pair.team_home_away.eq(side)].iloc[0]
            if (
                ALIASES.get(row.team_abbreviation, row.team_abbreviation)
                != game[side + "_team_abbreviation"]
                or float(row.team_score) != float(game[side + "_pts"])
                or float(other[side + "_score"]) != float(game[side + "_pts"])
            ):
                raise ValueError(
                    "ESPN and official schedule identity/score disagreement: " + game["game_id"]
                )
            # ESPN 'turnovers' matches NBA player turnovers; 'total_turnovers' includes
            # additional team turnovers and would silently change existing possessions.
            output.append(
                _team_row(
                    game,
                    side,
                    year,
                    row.team_score,
                    row.field_goals_attempted,
                    row.offensive_rebounds,
                    row.turnovers,
                    row.free_throws_attempted,
                    240 + 25 * (periods - 4),
                )
            )
    return pd.DataFrame(output)


def validate_season(frame, schedule, year, observed):
    from predictions.nba import canonical_nba

    if frame.empty or not BOX_COLUMNS.issubset(frame):
        raise ValueError("No complete NBA team box scores")
    if not frame.GAME_ID.astype(str).str.fullmatch(f"002{str(year)[-2:]}[0-9]{{5}}").all():
        raise ValueError("Invalid NBA regular-season game IDs")
    for field in ["PTS", "FGA", "OREB", "TOV", "FTA", "MIN"]:
        values = pd.to_numeric(frame[field], errors="coerce")
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Invalid NBA numeric box field " + field)
    finished = schedule.loc[schedule.finished].set_index("game_id")
    canonical = canonical_nba(frame)
    complete = canonical.loc[canonical.home_score.notna()]
    if not set(complete.game_id).issubset(set(finished.index)):
        raise ValueError("NBA result lacks matching final official schedule entry")
    for game in complete.itertuples():
        official = finished.loc[game.game_id]
        if (
            game.home_team != official.home_team_abbreviation
            or game.away_team != official.away_team_abbreviation
            or game.home_score != float(official.home_pts)
            or game.away_score != float(official.away_pts)
            or game.schedule_day != official.game_date
        ):
            raise ValueError(
                "NBA box and schedule identity/date/score disagreement: " + game.game_id
            )
    deadline = pd.to_datetime(finished.game_date, utc=True) + pd.Timedelta(days=2)
    needed = set(finished.index[deadline.lt(observed)])
    closed = pd.Timestamp.now(tz="UTC") >= pd.Timestamp(f"{year + 1}-07-01", tz="UTC")
    if closed:
        needed = set(finished.index)
    else:
        all_deadlines = pd.to_datetime(schedule.game_date, utc=True) + pd.Timedelta(days=2)
        unresolved = schedule.loc[~schedule.finished & all_deadlines.lt(observed), "game_id"]
        if len(unresolved):
            raise ValueError(
                f"Unresolved past NBA schedule entries in {season_name(year)}: {unresolved.head(5).tolist()}; "
                "refresh the provider data rather than assume missing results or cancellations"
            )
    missing = sorted(needed - set(complete.game_id))
    if missing:
        raise ValueError(
            f"Incomplete {season_name(year)} box scores: {len(missing)} eligible games missing; examples {missing[:5]}"
        )
    return canonical, {
        "final_schedule_games": len(finished),
        "box_games": len(complete),
        "eligible_games_required": len(needed),
        "closed_season": closed,
    }


def _source(provider, parts, **extra):
    observed = min(pd.Timestamp(p["fetched_at"]) for p in parts)
    if pd.isna(observed) or observed.tzinfo is None:
        raise ValueError("NBA source requires timezone-aware observation timestamps")
    return {
        "adapter_version": "nba-history-v2",
        "provider": provider,
        "fetched_at": observed.isoformat(),
        "components": parts,
        **extra,
    }


def load_season(year, root, *, provider="auto", refresh=False):
    """Auto: existing verified NBA cache, then archives. Live stats API is opt-in."""
    label = "NBA " + season_name(year)
    LOG.warning("%s: checking schedule and resumable %s downloads", label, provider)
    schedule, sm = schedule_table(year, root, refresh)
    errors = []
    if provider == "nba-stats":
        path, meta = snapshot(
            STATS_URL,
            root,
            params=stats_params(year),
            refresh=refresh,
            validator=lambda c: stats_frame(c, year),
            attempts=3,
            timeout=(8, 20),
            label=label + " leaguegamefinder",
        )
        frame = stats_frame(path.read_bytes(), year)
        source = _source("NBA Stats leaguegamefinder", [meta, sm])
        canonical, coverage = validate_season(
            frame, schedule, year, pd.Timestamp(source["fetched_at"])
        )
        return canonical, {**source, "coverage": coverage}
    if provider == "auto" and not refresh:
        try:
            old = cached_snapshot(
                STATS_URL, root, params=stats_params(year), validator=lambda c: stats_frame(c, year)
            )
            if old is not None:
                path, meta = old
                source = _source("NBA Stats verified existing cache", [meta, sm])
                canonical, coverage = validate_season(
                    stats_frame(path.read_bytes(), year),
                    schedule,
                    year,
                    pd.Timestamp(source["fetched_at"]),
                )
                return canonical, {**source, "coverage": coverage}
        except ValueError as exc:
            LOG.warning("%s: rejecting old stats cache: %s", label, exc)
            errors.append(str(exc))
    try:
        box, bm = release_table(
            "nba_stats_team_boxscores", f"team_boxscores_{year + 1}.parquet", root, refresh
        )
        source = _source("NBA Stats via sportsdataverse release", [sm, bm])
        canonical, coverage = validate_season(
            _rows_for_box(box, schedule, year), schedule, year, pd.Timestamp(source["fetched_at"])
        )
        return canonical, {**source, "coverage": coverage, "fallback_reasons": errors}
    except (ValueError, DownloadError) as exc:
        errors.append(str(exc))
        LOG.warning("%s: NBA box archive unavailable; checking ESPN archive: %s", label, exc)
    try:
        scores, boxes, parts = espn_tables(year, root, refresh)
        source = _source(
            "ESPN via sportsdataverse; identities/scores checked against NBA schedule", [sm, *parts]
        )
        canonical, coverage = validate_season(
            espn_rows(scores, boxes, schedule, year),
            schedule,
            year,
            pd.Timestamp(source["fetched_at"]),
        )
        return canonical, {**source, "coverage": coverage, "fallback_reasons": errors}
    except (ValueError, DownloadError) as exc:
        raise ValueError(
            f"{label}: no complete validated archive. NBA: {errors}; ESPN: {exc}. No games substituted; verified earlier seasons remain cached."
        ) from exc


def fetch_history(root, start, through, *, provider="auto", refresh=False):
    from predictions.data import _atomic_write

    if provider not in ["auto", "nba-stats", "archive"] or start > through or start < 2010:
        raise ValueError(
            "NBA provider must be auto/nba-stats/archive and seasons must start at 2010 or later"
        )
    root = Path(root)
    frames, sources = [], {}
    manifest = root / "seasons"
    manifest.mkdir(parents=True, exist_ok=True)
    for year in range(start, through + 1):
        try:
            frame, source = load_season(
                year, root, provider=provider, refresh=refresh and year == through
            )
        except (ValueError, DownloadError) as exc:
            raise ValueError(f"NBA download stopped at {season_name(year)}. {exc}") from exc
        frames.append(frame)
        sources[str(year)] = source
        _atomic_write(
            manifest / f"{year}.json",
            (
                json.dumps(
                    {
                        "season": year,
                        "verified_at": datetime.now(timezone.utc).isoformat(),
                        **source,
                    },
                    indent=2,
                )
                + "\n"
            ).encode(),
        )
        LOG.warning(
            "NBA %s: validated %d games using %s", season_name(year), len(frame), source["provider"]
        )
    result = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["kickoff", "game_id"])
        .reset_index(drop=True)
    )
    result["source_observed_at"] = pd.to_datetime(
        result.season.map(lambda y: sources[str(y)]["fetched_at"]), utc=True
    )
    unsafe = result.available_at.ge(result.source_observed_at)
    result.loc[unsafe, ["home_score", "away_score", "home_efficiency", "away_efficiency"]] = np.nan
    return result, sources


def archive_slate(root):
    """No guessed tipoffs: join actual ESPN timestamps to official NBA game identities."""
    now = pd.Timestamp.now(tz="UTC")
    year = now.year if now.month >= 7 else now.year - 1
    schedule, sm = schedule_table(year, root, refresh=False)
    scores, em = release_table(
        "espn_nba_schedules", f"nba_schedule_{year + 1}.parquet", root, refresh=False
    )
    if not {"season", "season_type", "date", "home_abbreviation", "away_abbreviation"}.issubset(
        scores
    ):
        raise ValueError("NBA live schedule fallback schema mismatch")
    if not pd.to_numeric(scores.season).eq(year + 1).all():
        raise ValueError("NBA live schedule fallback has wrong season")
    by_key = {}
    known_teams = (set(schedule.home_team_abbreviation) | set(schedule.away_team_abbreviation)) - {
        None,
        "",
        "TBD",
    }
    for row in scores.loc[pd.to_numeric(scores.season_type).eq(2)].to_dict("records"):
        key = _match_key(_espn_day(row), row["home_abbreviation"], row["away_abbreviation"])
        if key[1] not in known_teams or key[2] not in known_teams:
            continue  # Unassigned NBA Cup placeholders are not identified matchups.
        if key in by_key:
            raise ValueError("Ambiguous ESPN live schedule identity")
        by_key[key] = row
    output = []
    missing = []
    for game in schedule.to_dict("records"):
        day = pd.Timestamp(game["game_date"], tz="America/New_York")
        if day < now - pd.Timedelta(days=14) or day > now + pd.Timedelta(days=8):
            continue
        key = _match_key(
            game["game_date"], game["home_team_abbreviation"], game["away_team_abbreviation"]
        )
        other = by_key.get(key)
        if other is None:
            missing.append(game["game_id"])
            continue
        if other.get("status_type_name") in [
            "STATUS_POSTPONED",
            "STATUS_CANCELED",
            "STATUS_CANCELLED",
        ]:
            continue
        kickoff = pd.Timestamp(other["date"])
        if (
            pd.isna(kickoff)
            or kickoff.tzinfo is None
            or kickoff.tz_convert("America/New_York").date() != day.date()
        ):
            raise ValueError("NBA schedule fallback has missing or contradictory tipoff")
        output.append(
            {
                "sport": "NBA",
                "game_id": str(game["game_id"]),
                "season": year,
                "kickoff": kickoff,
                "schedule_day": game["game_date"],
                "available_at": kickoff + pd.Timedelta(days=2),
                "home_team": game["home_team_abbreviation"],
                "away_team": game["away_team_abbreviation"],
                "home_name": game["home_team_name"],
                "away_name": game["away_team_name"],
                "home_score": np.nan,
                "away_score": np.nan,
                "home_efficiency": np.nan,
                "away_efficiency": np.nan,
                "pace": np.nan,
                "neutral": bool(other.get("neutral_site", False)),
            }
        )
    columns = [
        "sport",
        "game_id",
        "season",
        "kickoff",
        "schedule_day",
        "available_at",
        "home_team",
        "away_team",
        "home_name",
        "away_name",
        "home_score",
        "away_score",
        "home_efficiency",
        "away_efficiency",
        "pace",
        "neutral",
    ]
    if missing:
        LOG.warning(
            "NBA schedule fallback has no verified tipoff for %d games: %s",
            len(missing),
            missing[:5],
        )
    return pd.DataFrame(output, columns=columns), _source(
        "NBA schedule archive + ESPN verified tipoffs", [sm, em], unavailable_tipoffs=missing
    )
