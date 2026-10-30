"""Official NHL events distributed by fastRhockey; no vendor xG is consumed."""

import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from predictions.league_data import snapshot

FEATURES = ["distance", "angle", "distance2", "angle2", "empty", "skater_advantage"]
CODES = {
    "PHX": "ARI",
    "ATL": "ATL",
    "T.B": "TBL",
    "N.J": "NJD",
    "S.J": "SJS",
    "L.A": "LAK",
    "UTH": "UTA",
    "WIN": "WPG",
    "LA": "LAK",
}


def code(value):
    return CODES.get(value, value)


def release_frame(tag, name, root, refresh=False):
    index, publication = snapshot(
        "https://api.github.com/repos/sportsdataverse/sportsdataverse-data/releases/tags/" + tag,
        Path(root) / "catalog",
        refresh=refresh,
    )
    asset = next((a for a in json.loads(index.read_text())["assets"] if a["name"] == name), None)
    if asset is None:
        raise ValueError("Published NHL season asset unavailable: " + name)
    expected = asset.get("digest")
    if not expected or not expected.startswith("sha256:"):
        raise ValueError("NHL asset has no published SHA256")
    path, meta = snapshot(asset["browser_download_url"], root, refresh=refresh, timeout=90)
    if meta["sha256"] != expected[7:]:
        path, meta = snapshot(asset["browser_download_url"], root, refresh=True, timeout=90)
        if meta["sha256"] != expected[7:]:
            raise ValueError("NHL release checksum mismatch")
    return path, {
        **meta,
        "published_sha256": expected[7:],
        "provider": "NHL via sportsdataverse/fastRhockey",
        "release_catalog_sha256": publication["sha256"],
    }


def official_season(year, root, refresh=False):
    sources = {}
    games = {}
    pending = {"BOS", "LAK", "TOR", "TBL"}
    done = set()

    def fetch(club):
        path, meta = snapshot(
            f"https://api-web.nhle.com/v1/club-schedule-season/{club}/{year}{year + 1}",
            Path(root) / "official-schedules",
            refresh=refresh,
        )
        return club, json.loads(path.read_text()).get("games", []), meta

    with ThreadPoolExecutor(max_workers=4) as pool:
        while pending:
            results = list(pool.map(fetch, sorted(pending)))
            pending = set()
            for club, items, meta in results:
                done.add(club)
                sources[club] = meta
                for game in items:
                    if game.get("gameType") != 2:
                        continue
                    for side in ["homeTeam", "awayTeam"]:
                        candidate = game[side]["abbrev"]
                        if candidate not in done:
                            pending.add(candidate)
                    if game.get("gameState") not in ["OFF", "FINAL"]:
                        continue
                    h, a = game["homeTeam"], game["awayTeam"]
                    kind = game.get("gameOutcome", {}).get("lastPeriodType")
                    if kind not in ["REG", "OT", "SO"]:
                        continue
                    kickoff = pd.Timestamp(game["startTimeUTC"])
                    available = kickoff + pd.Timedelta(hours=24)
                    if available >= pd.Timestamp(meta["fetched_at"]):
                        continue
                    identity = int(game["id"])
                    entry = {
                        "game_id": identity,
                        "home_team_abbr": h["abbrev"],
                        "away_team_abbr": a["abbrev"],
                        "home_team_name": h.get("placeName", {}).get("default", h["abbrev"])
                        + " "
                        + h.get("commonName", {}).get("default", ""),
                        "away_team_name": a.get("placeName", {}).get("default", a["abbrev"])
                        + " "
                        + a.get("commonName", {}).get("default", ""),
                        "home_score": int(h["score"]),
                        "away_score": int(a["score"]),
                        "kickoff": kickoff,
                        "available_at": available,
                        "outcome_type": kind,
                    }
                    if identity in games and any(
                        games[identity][k] != entry[k]
                        for k in [
                            "home_score",
                            "away_score",
                            "home_team_abbr",
                            "away_team_abbr",
                            "outcome_type",
                        ]
                    ):
                        raise ValueError("Official NHL club sources disagree")
                    games[identity] = entry
            pending -= done
    return pd.DataFrame(games.values()), sources


def get_year(year, root, refresh=False):
    games, sources = official_season(year, root, refresh)
    pbp, pm = release_frame("nhl_pbp_lite", f"play_by_play_lite_{year + 1}.parquet", root, refresh)
    games = games.loc[games.available_at.lt(pd.Timestamp(pm["fetched_at"]))]
    use = [
        "game_id",
        "season_type",
        "event_type",
        "event_idx",
        "period",
        "period_type",
        "period_seconds",
        "event_team_abbr",
        "event_team_type",
        "home_abbr",
        "away_abbr",
        "x_fixed",
        "y_fixed",
        "x",
        "y",
        "empty_net",
        "home_skaters",
        "away_skaters",
        "event_goalie_id",
        "event_goalie_name",
        "home_goalie_id",
        "away_goalie_id",
        "home_goalie",
        "away_goalie",
    ]
    available = pq.ParquetFile(pbp).schema_arrow.names
    # Published xg columns and all score/win-probability fields are deliberately excluded.
    shots = pd.read_parquet(pbp, columns=[c for c in use if c in available])
    shots = shots.loc[
        shots.game_id.isin(games.game_id)
        & shots.event_type.isin(["SHOT", "GOAL", "MISSED_SHOT"])
        & shots.period.le(3)
    ].copy()
    for column in use:
        if column not in shots:
            shots[column] = np.nan
    shots["goal"] = shots.event_type.eq("GOAL").astype(int)
    dx = 89 - pd.to_numeric(shots.x_fixed, errors="coerce")
    yy = pd.to_numeric(shots.y_fixed, errors="coerce")
    shots["distance"] = np.sqrt(dx**2 + yy**2)
    shots["angle"] = np.degrees(np.arctan2(yy.abs(), dx)).abs()
    shots["distance2"] = shots.distance**2 / 2500
    shots["angle2"] = shots.angle**2 / 8100
    shots["empty"] = shots.empty_net.astype(float)
    shots["is_home"] = shots.event_team_type.str.lower().eq("home")
    diff = shots.home_skaters - shots.away_skaters
    shots["skater_advantage"] = diff.where(shots.is_home, -diff)
    shots["opponent_skaters"] = shots.away_skaters.where(shots.is_home, shots.home_skaters)
    shots["season"] = year
    return games, shots, {"official_clubs": sources, "pbp": pm}


def prepare(root, start=2009, through=2025, refresh_current=False):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    # Raw downloads are bounded independently; fitting is deliberately sequential.
    with ThreadPoolExecutor(max_workers=4) as pool:
        for _ in pool.map(
            lambda y: release_frame(
                "nhl_pbp_lite",
                f"play_by_play_lite_{y + 1}.parquet",
                root,
                refresh_current and y == through,
            ),
            range(start, through + 1),
        ):
            pass
    code_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    existing = None
    old_meta_path = root / "source.json"
    if (
        old_meta_path.exists()
        and (root / "games.parquet").exists()
        and (root / "goalies.parquet").exists()
    ):
        old = json.loads(old_meta_path.read_text())
        if (
            old.get("code_sha256") == code_sha
            and old.get("games_sha256")
            == hashlib.sha256((root / "games.parquet").read_bytes()).hexdigest()
            and old.get("goalies_sha256")
            == hashlib.sha256((root / "goalies.parquet").read_bytes()).hexdigest()
        ):
            existing = pd.read_parquet(root / "games.parquet")
            if through <= old["through_season"] and not refresh_current:
                filtered = {**old, "through_season": through}
                for key in ["sources", "causal_xg", "coverage"]:
                    filtered[key] = {
                        year: value for year, value in old[key].items() if int(year) <= through
                    }
                return existing.loc[existing.season.le(through)].copy(), filtered
    history = []
    rows = []
    goalies = []
    sources = {}
    xg = {}
    coverage = {}
    build_from = start
    if existing is not None:
        build_from = min(through, old["through_season"] + 1)
        keep = existing.loc[existing.season.lt(build_from)]
        rows = keep.to_dict("records")
        old_goalies = pd.read_parquet(root / "goalies.parquet")
        goalies = old_goalies.loc[old_goalies.game_id.isin(keep.game_id)].to_dict("records")
        sources = {k: v for k, v in old["sources"].items() if int(k) < build_from}
        xg = {k: v for k, v in old["causal_xg"].items() if int(k) < build_from}
        coverage = {k: v for k, v in old["coverage"].items() if int(k) < build_from}
        for prior in range(max(start, build_from - 3), build_from):
            _, prior_shots, _ = get_year(prior, root)
            history.append(prior_shots[FEATURES + ["goal", "season"]].copy())
    for year in range(build_from, through + 1):
        games, shots, source = get_year(year, root, refresh_current and year == through)
        sources[str(year)] = source
        if history:
            train = pd.concat(history[-3:], ignore_index=True)
            with threadpool_limits(limits=2):
                estimator = make_pipeline(
                    SimpleImputer(strategy="median", keep_empty_features=True),
                    StandardScaler(),
                    LogisticRegression(C=1, max_iter=1000, random_state=42),
                ).fit(train[FEATURES], train.goal)
            path = root / f"causal-xg-{year}.joblib"
            joblib.dump(estimator, path)
            shots["causal_xg"] = estimator.predict_proba(shots[FEATURES])[:, 1]
            xg[str(year)] = {
                "training_seasons": sorted(int(y) for y in train.season.unique()),
                "training_shots": len(train),
                "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "test_log_loss": float(log_loss(shots.goal, shots.causal_xg)),
                "test_brier": float(brier_score_loss(shots.goal, shots.causal_xg)),
                "features": FEATURES,
            }
            by_game = {int(k): v for k, v in shots.groupby("game_id")}
            count = 0
            for game in games.itertuples():
                identity = int(game.game_id)
                if identity not in by_game:
                    continue
                g = by_game[identity]
                home, away = code(game.home_team_abbr), code(game.away_team_abbr)
                fh, fa = int(game.home_score), int(game.away_score)
                if fh == fa:
                    raise ValueError("NHL final game cannot be tied")
                kind = game.outcome_type
                if set(g.home_abbr.dropna().map(code)) != {home} or set(
                    g.away_abbr.dropna().map(code)
                ) != {away}:
                    raise ValueError(f"Official NHL/PBP team identity mismatch {identity}")
                hg, ag = (fh, fa) if kind == "REG" else (fh - int(fh > fa), fa - int(fa > fh))
                if kind != "REG" and hg != ag:
                    raise ValueError(f"NHL final/period scopes disagree {identity} {fh, fa}")
                row = {
                    "sport": "NHL",
                    "game_id": str(identity),
                    "season": year,
                    "kickoff": game.kickoff,
                    "schedule_day": game.kickoff.tz_convert("America/New_York").date().isoformat(),
                    "available_at": game.available_at,
                    "home_team": home,
                    "away_team": away,
                    "home_name": str(game.home_team_name),
                    "away_name": str(game.away_team_name),
                    "home_score": hg,
                    "away_score": ag,
                    "home_final_score": fh,
                    "away_final_score": fa,
                    "home_win": int(fh > fa),
                    "outcome_type": kind,
                    "went_extra": kind != "REG",
                }
                for side, flag in [("home", True), ("away", False)]:
                    attack = g.loc[g.is_home.eq(flag)]
                    defense = g.loc[g.is_home.ne(flag)]
                    if attack.empty or defense.empty:
                        break
                    for key, value in {
                        "xg_for": attack.causal_xg.sum(),
                        "xg_against": defense.causal_xg.sum(),
                        "shots_for": len(attack),
                        "shots_against": len(defense),
                        "shot_quality": attack.causal_xg.mean(),
                        "pp_xg": attack.loc[
                            attack.skater_advantage.gt(0)
                            & attack.opponent_skaters.lt(5)
                            & attack["empty"].eq(0),
                            "causal_xg",
                        ].sum(),
                        "pk_xg": defense.loc[
                            defense.skater_advantage.gt(0)
                            & defense.opponent_skaters.lt(5)
                            & defense["empty"].eq(0),
                            "causal_xg",
                        ].sum(),
                    }.items():
                        row[side + "_" + key] = float(value)
                    keeper = defense.event_goalie_id.copy()
                    fallback = defense["home_goalie_id" if flag else "away_goalie_id"]
                    keeper = keeper.fillna(fallback)
                    faced = defense.loc[
                        defense.event_type.isin(["SHOT", "GOAL"])
                        & keeper.notna()
                        & keeper.ne(0)
                        & defense["empty"].eq(0)
                    ].copy()
                    faced["keeper"] = keeper.loc[faced.index]
                    sizes = faced.groupby("keeper").size()
                    primary = sizes.idxmax() if len(sizes) else None
                    row[side + "_goalie_id"] = str(primary) if primary is not None else None
                    names = faced.loc[faced.keeper.eq(primary), "event_goalie_name"].dropna()
                    row[side + "_goalie_name"] = str(names.iloc[0]) if len(names) else "Unknown"
                    for goalie, events in faced.groupby("keeper"):
                        goalies.append(
                            {
                                "game_id": str(identity),
                                "goalie": str(goalie),
                                "kickoff": game.kickoff,
                                "available_at": game.available_at,
                                "shots_on_goal": len(events),
                                "goals": int(events.goal.sum()),
                            }
                        )
                else:
                    rows.append(row)
                    count += 1
            coverage[str(year)] = {
                "finished_regular_schedule_games": len(games),
                "games_with_usable_regulation_pbp": count,
            }
        history.append(shots[FEATURES + ["goal", "season"]].copy())
        history = history[-3:]
        print("NHL official prepared", year, len(rows), file=sys.stderr, flush=True)
    data = pd.DataFrame(rows)
    data.to_parquet(root / "games.parquet", index=False)
    pd.DataFrame(goalies).to_parquet(root / "goalies.parquet", index=False)
    metadata = {
        "provider": "NHL official events via sportsdataverse fastRhockey releases",
        "sources": sources,
        "causal_xg": xg,
        "coverage": coverage,
        "vendor_xg_used": False,
        "code_sha256": code_sha,
        "through_season": through,
        "games_sha256": hashlib.sha256((root / "games.parquet").read_bytes()).hexdigest(),
        "goalies_sha256": hashlib.sha256((root / "goalies.parquet").read_bytes()).hexdigest(),
    }
    (root / "source.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return data, metadata
