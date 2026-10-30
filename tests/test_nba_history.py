"""Archive normalization, source coverage, end-year mapping and pregame acquisition."""

import json

import numpy as np
import pandas as pd
import pytest

from predictions import nba_history as history
from predictions.league_data import DownloadError
from predictions.nba_history import (
    _rows_for_box,
    archive_slate,
    espn_rows,
    fetch_history,
    load_season,
    schedule_table,
    stats_frame,
    stats_params,
    validate_season,
)

META = {
    "fetched_at": "2026-10-30T00:00:00Z",
    "sha256": "a" * 64,
    "url": "https://example.org/archived-data",
}


def schedule(year=2024, day="2024-10-22"):
    return pd.DataFrame(
        [
            {
                "game_id": f"002{str(year)[-2:]}00001",
                "season": year + 1,
                "game_date": day,
                "home_team_id": 1,
                "away_team_id": 2,
                "home_team_abbreviation": "BOS",
                "away_team_abbreviation": "NYK",
                "home_team_name": "Boston Celtics",
                "away_team_name": "New York Knicks",
                "home_pts": 110,
                "away_pts": 100,
                "home_wl": "W",
                "away_wl": "L",
                "finished": True,
            }
        ]
    )


def boxes(year=2024):
    return pd.DataFrame(
        [
            {
                "game_id": f"002{str(year)[-2:]}00001",
                "season": year + 1,
                "team_id": i,
                "team_tricode": team,
                "side": side,
                "minutes": "265:00",
                "points": score,
                "field_goals_attempted": 90,
                "rebounds_offensive": 10,
                "turnovers": 15,
                "free_throws_attempted": 20,
            }
            for i, team, side, score in [(1, "BOS", "home", 110), (2, "NYK", "away", 100)]
        ]
    )


def espn():
    scores = pd.DataFrame(
        [
            {
                "game_id": "espn-1",
                "season": 2025,
                "season_type": 2,
                "date": "2024-10-23T02:00:00Z",
                "game_date": "2024-10-23",
                "status_type_completed": True,
                "status_period": 5,
                "home_abbreviation": "BOS",
                "away_abbreviation": "NY",
                "home_score": 110,
                "away_score": 100,
            }
        ]
    )
    box = pd.DataFrame(
        [
            {
                "game_id": "espn-1",
                "season": 2025,
                "team_home_away": side,
                "team_abbreviation": team,
                "team_score": score,
                "field_goals_attempted": 90,
                "offensive_rebounds": 10,
                "turnovers": 15,
                "total_turnovers": 99,
                "free_throws_attempted": 20,
            }
            for team, side, score in [("BOS", "home", 110), ("NY", "away", 100)]
        ]
    )
    return scores, box


def test_archive_maps_end_year_and_retains_exact_possessions_and_overtime():
    raw = _rows_for_box(boxes(), schedule(), 2024)
    frame, coverage = validate_season(raw, schedule(), 2024, pd.Timestamp(META["fetched_at"]))
    game = frame.iloc[0]
    assert game.game_id == "0022400001" and game.season == 2024
    assert game.possessions == pytest.approx(103.8)
    assert game.pace == pytest.approx(103.8 * 48 / 53)
    assert game.available_at == pd.Timestamp("2024-10-24T00:00:00Z")
    assert coverage["eligible_games_required"] == 1


def test_espn_preserves_nba_ids_eastern_days_player_turnovers_and_periods():
    scores, box = espn()
    raw = espn_rows(scores, box, schedule(), 2024)
    np.testing.assert_allclose(raw.TOV, [15, 15])  # Not ESPN total_turnovers=99.
    first, _ = validate_season(raw, schedule(), 2024, pd.Timestamp(META["fetched_at"]))
    second, _ = validate_season(
        _rows_for_box(boxes(), schedule(), 2024), schedule(), 2024, pd.Timestamp(META["fetched_at"])
    )
    pd.testing.assert_frame_equal(first, second, check_dtype=False)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "score", "team", "nan", "negative"])
def test_incomplete_or_contradictory_box_season_is_rejected(mutation):
    official = schedule()
    raw = _rows_for_box(boxes(), official, 2024)
    if mutation == "missing":
        extra = official.copy()
        extra["game_id"] = "0022400002"
        official = pd.concat([official, extra])
    elif mutation == "duplicate":
        raw = pd.concat([raw, raw.iloc[:1]])
    elif mutation == "score":
        raw.loc[0, "PTS"] = 999
    elif mutation == "team":
        raw.loc[0, "TEAM_ABBREVIATION"] = "MIL"
    elif mutation == "nan":
        raw.loc[0, "FGA"] = np.nan
    else:
        raw.loc[0, "TOV"] = -1
    with pytest.raises(ValueError):
        validate_season(raw, official, 2024, pd.Timestamp(META["fetched_at"]))


def test_canceled_placeholders_are_excluded_only_when_schedule_is_not_final():
    normal = schedule()
    canceled = normal.copy()
    canceled["game_id"] = "0022400002"
    canceled["finished"] = False
    canceled[["home_wl", "away_wl"]] = None
    official = pd.concat([normal, canceled])
    bad = boxes()
    bad["game_id"] = "0022400002"
    bad["team_id"] = 0
    bad["minutes"] = None
    raw = _rows_for_box(pd.concat([boxes(), bad]), official, 2024)
    assert len(raw) == 2 and set(raw.GAME_ID) == {"0022400001"}


def test_schedule_denominator_cannot_silently_shrink(monkeypatch, tmp_path):
    monkeypatch.setattr(history, "release_table", lambda *a, **k: (schedule(), META))
    with pytest.raises(ValueError, match="expected 1230"):
        schedule_table(2024, tmp_path)


def test_wrong_end_year_and_unknown_overtime_are_rejected():
    bad = boxes()
    bad["season"] = 2024
    with pytest.raises(ValueError, match="end-year"):
        _rows_for_box(bad, schedule(), 2024)
    scores, box = espn()
    scores["status_period"] = np.nan
    with pytest.raises(ValueError, match="duration"):
        espn_rows(scores, box, schedule(), 2024)
    scores, box = espn()
    box.loc[0, "team_score"] = 999
    with pytest.raises(ValueError, match="disagreement"):
        espn_rows(scores, box, schedule(), 2024)


def test_auto_uses_verified_cache_without_calling_stats_api(monkeypatch, tmp_path):
    raw = _rows_for_box(boxes(), schedule(), 2024)
    data = {"resultSets": [{"headers": raw.columns.tolist(), "rowSet": raw.values.tolist()}]}
    path = tmp_path / "cached.raw"
    path.write_text(json.dumps(data))
    monkeypatch.setattr(history, "schedule_table", lambda *a, **k: (schedule(), META))
    monkeypatch.setattr(history, "cached_snapshot", lambda *a, **k: (path, META))
    monkeypatch.setattr(
        history, "snapshot", lambda *a, **k: pytest.fail("Unexpected network request")
    )
    frame, source = load_season(2024, tmp_path)
    assert len(frame) == 1 and "existing cache" in source["provider"]


def test_archive_falls_back_to_validated_espn_without_live_stats(monkeypatch, tmp_path):
    monkeypatch.setattr(history, "schedule_table", lambda *a, **k: (schedule(), META))
    monkeypatch.setattr(
        history,
        "release_table",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("asset unavailable")),
    )
    scores, box = espn()
    monkeypatch.setattr(history, "espn_tables", lambda *a, **k: (scores, box, [META, META]))
    frame, source = load_season(2024, tmp_path, provider="archive")
    assert len(frame) == 1 and "ESPN" in source["provider"]
    assert source["fallback_reasons"] == ["asset unavailable"]


def test_stats_schema_and_requested_season_are_validated():
    raw = _rows_for_box(boxes(), schedule(), 2024)
    data = {"resultSets": [{"headers": raw.columns.tolist(), "rowSet": raw.values.tolist()}]}
    assert len(stats_frame(json.dumps(data), 2024)) == 2
    with pytest.raises(ValueError, match="season"):
        stats_frame(json.dumps(data), 2023)
    with pytest.raises(ValueError):
        stats_frame("<html>blocked</html>", 2024)
    assert stats_params(2011)["Season"] == "2011-12"


def test_acquisition_failure_retains_completed_season_manifest(monkeypatch, tmp_path):
    from predictions.nba import canonical_nba

    canonical = canonical_nba(_rows_for_box(boxes(), schedule(), 2024))

    def load(year, *args, **kwargs):
        if year == 2025:
            raise DownloadError("https://example.org/2025", "NBA2025-26", ["RemoteDisconnected"])
        return canonical, {**META, "provider": "fixture"}

    monkeypatch.setattr(history, "load_season", load)
    with pytest.raises(ValueError, match="stopped at 2025-26"):
        fetch_history(tmp_path, 2024, 2025)
    assert (tmp_path / "seasons/2024.json").is_file() and not (
        tmp_path / "seasons/2025.json"
    ).exists()


def test_current_slate_fallback_uses_real_tipoff_not_midnight(monkeypatch, tmp_path):
    now = pd.Timestamp.now(tz="UTC")
    year = now.year if now.month >= 7 else now.year - 1
    tip = (now + pd.Timedelta(days=2)).normalize() + pd.Timedelta(hours=23)
    day = tip.tz_convert("America/New_York").date().isoformat()
    official = schedule(year, day)
    scores = pd.DataFrame(
        [
            {
                "season": year + 1,
                "season_type": 2,
                "date": tip.isoformat(),
                "home_abbreviation": "BOS",
                "away_abbreviation": "NY",
                "status_type_name": "STATUS_SCHEDULED",
            }
        ]
    )
    unknown = scores.iloc[[0, 0]].copy()
    unknown[["home_abbreviation", "away_abbreviation"]] = "TBD"
    scores = pd.concat([scores, unknown], ignore_index=True)
    monkeypatch.setattr(history, "schedule_table", lambda *a, **k: (official, META))
    monkeypatch.setattr(history, "release_table", lambda *a, **k: (scores, META))
    slate, source = archive_slate(tmp_path)
    assert slate.iloc[0].kickoff == tip and pd.isna(slate.iloc[0].home_score)
    assert "ESPN verified tipoffs" in source["provider"]
    from predictions import nba

    monkeypatch.setattr(
        nba,
        "_cdn_slate",
        lambda *a: (_ for _ in ()).throw(DownloadError("https://cdn.nba.com", "CDN", ["HTTP 403"])),
    )
    fallback, _ = nba.nba_slate(tmp_path)
    pd.testing.assert_frame_equal(slate, fallback)


def test_current_season_cannot_hide_old_unresolved_schedule_games():
    now = pd.Timestamp.now(tz="UTC")
    year = now.year if now.month >= 7 else now.year - 1
    day = (now - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    official = schedule(year, day)
    unresolved = official.copy()
    unresolved["game_id"] = f"002{str(year)[-2:]}00002"
    unresolved["finished"] = False
    unresolved[["home_wl", "away_wl"]] = None
    raw = _rows_for_box(boxes(year), official, year)
    with pytest.raises(ValueError, match="Unresolved past"):
        validate_season(raw, pd.concat([official, unresolved]), year, now)


@pytest.mark.parametrize("column", ["date", "season_type", "status_period"])
def test_incomplete_espn_response_schema_has_useful_error(column):
    scores, box = espn()
    with pytest.raises(ValueError, match="schema mismatch"):
        espn_rows(scores.drop(columns=[column]), box, schedule(), 2024)
