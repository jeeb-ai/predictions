"""Recorded source schemas: public NFL hydration, NBA final states and NHL scopes."""

import json

import pandas as pd
import pytest

from predictions import official_results as official
from predictions.league_data import DownloadError

SOURCE = {
    "fetched_at": "2026-10-30T12:00:00Z",
    "sha256": "a" * 64,
    "classification": "official_league_data",
    "provider": "Explicit synthetic fixture",
}


def nfl_html():
    away = {
        "abbreviation": "CAR",
        "summary": {"score": {"q1": 3, "q2": 7, "q3": 3, "q4": 6, "ot": 0, "total": 19}},
    }
    card = {
        "gameId": "official-fixture",
        "gameTime": "2026-10-30T00:15:00Z",
        "homeTeam": {"abbreviation": "GB"},
        "awayTeam": away,
        "href": "/games/panthers-at-packers-2026-reg-8",
        "gameSummary": {
            "phase": "FINAL",
            "homeTeam": {"score": {"q1": 6, "q2": 14, "q3": 7, "q4": 0, "ot": 0, "total": 27}},
            "awayTeam": "$r:props:cards:0:awayTeam:summary",
        },
    }
    frame = ["$", None, None, {"cards": [card]}]
    value = "r:" + json.dumps(frame) + "\n"
    return (
        "<html><script>self.__next_f.push(" + json.dumps([1, value]) + ")</script></html>"
    ).encode()


def nba_body(status=3, home=110, away=100):
    return json.dumps(
        {
            "leagueSchedule": {
                "gameDates": [
                    {
                        "games": [
                            {
                                "gameId": "0022600080",
                                "gameStatus": status,
                                "gameStatusText": "Final" if status == 3 else "Q3",
                                "gameDateTimeUTC": "2026-10-26T23:00:00Z",
                                "homeTeam": {"teamTricode": "BOS", "score": home},
                                "awayTeam": {"teamTricode": "NYK", "score": away},
                            }
                        ]
                    }
                ]
            }
        }
    ).encode()


def nhl_body(kind="SO", state="OFF"):
    return json.dumps(
        {
            "id": 2026020100,
            "gameType": 2,
            "gameState": state,
            "startTimeUTC": "2026-10-26T23:00:00Z",
            "gameOutcome": {"lastPeriodType": kind},
            "homeTeam": {"abbrev": "BOS", "score": 3},
            "awayTeam": {"abbrev": "NYR", "score": 2},
        }
    ).encode()


def test_nfl_reads_public_scores_references_and_scheduled_time_without_executing_js():
    result = official.nfl_cards(nfl_html())["2026_08_CAR_GB"]
    assert result["home_score"] == 27 and result["away_score"] == 19 and result["status"] == "final"
    assert result["kickoff"] == "2026-10-30T00:15:00+00:00"
    with pytest.raises(ValueError, match="format unavailable"):
        official.nfl_cards(b"<html>Homepage redirect without verified scores</html>")
    with pytest.raises(ValueError, match="reconcile"):
        official.nfl_cards(nfl_html().replace(b"27}", b"28}"))


def test_nfl_unsupported_reference_is_not_guessed():
    with pytest.raises(ValueError, match="reference"):
        official.nfl_cards(nfl_html().replace(b"$r:props", b"$unknown:props"))


def test_nba_live_scores_are_not_final_outcomes():
    live = official.nba_cdn_results(nba_body(2), SOURCE)["0022600080"]
    final = official.nba_cdn_results(nba_body(), SOURCE)["0022600080"]
    assert live["status"] == "scheduled" and live["home_score"] is None
    assert final["status"] == "final" and final["home_score"] == 110
    with pytest.raises(ValueError):
        official.nba_cdn_results(b'{"error":"unavailable"}', SOURCE)


@pytest.mark.parametrize(
    "kind,expected", [("REG", "final"), ("OT", "final"), ("SO", "final"), ("UNKNOWN", "incomplete")]
)
def test_nhl_result_scope_requires_explicit_reg_ot_so(kind, expected):
    value = official.nhl_landing(nhl_body(kind), SOURCE)
    assert value["status"] == expected and value["outcome_type"] == kind
    if kind == "UNKNOWN":
        assert value["home_score"] is None


def test_one_nhl_endpoint_failure_does_not_discard_other_results(monkeypatch, tmp_path):
    raw = tmp_path / "result.json"
    raw.write_bytes(nhl_body())

    def fetch(url, *args, **kwargs):
        if "/bad/" in url:
            raise DownloadError(url, "NHL", ["HTTP 503"])
        return raw, SOURCE

    monkeypatch.setattr(official, "snapshot", fetch)
    values = official.nhl_results([{"game_id": "bad"}, {"game_id": "2026020100"}], tmp_path)
    assert values[0]["status"] == "provider_error" and values[1]["status"] == "final"


def test_nba_archive_fallback_uses_independently_observed_tipoff(monkeypatch, tmp_path):
    from test_nba_history import schedule

    from predictions import nba_history

    table = schedule()
    scores = pd.DataFrame(
        [
            {
                "season_type": 2,
                "date": "2024-10-23T02:00:00Z",
                "home_abbreviation": "BOS",
                "away_abbreviation": "NY",
                "status_type_completed": True,
                "home_score": 110,
                "away_score": 100,
            }
        ]
    )
    meta = {**SOURCE, "file": "sample.raw"}
    monkeypatch.setattr(
        official,
        "snapshot",
        lambda *a, **k: (_ for _ in ()).throw(
            DownloadError("https://cdn.nba.com", "NBA", ["HTTP 403"])
        ),
    )
    monkeypatch.setattr(nba_history, "schedule_table", lambda *a, **k: (table, meta))
    monkeypatch.setattr(nba_history, "release_table", lambda *a, **k: (scores, meta))
    targets = [{"game_id": "0022400001", "kickoff": "2024-10-23T03:00:00Z"}]
    values = official.nba_results(targets, tmp_path)
    assert values[0]["kickoff"] == "2024-10-23T02:00:00+00:00"
    assert values[0]["kickoff"] != targets[0]["kickoff"] and values[0]["home_score"] == 110


def test_nfl_week_game_objects_are_not_confused_with_the_current_score_strip():
    game = {
        "id": "week-specific",
        "time": "2026-10-25T17:00:00Z",
        "season": 2026,
        "homeTeam": {"currentLogo": "https://static.www.nfl.com/logos/AZ"},
        "awayTeam": {"currentLogo": "https://static.www.nfl.com/logos/BUF"},
        "externalIds": [{"source": "slug", "id": "bills-at-cardinals-2026-reg-7"}],
        "summary": {
            "gameId": "week-specific",
            "phase": "FINAL",
            "homeTeam": {"score": {"q1": 7, "q2": 0, "q3": 0, "q4": 0, "total": 7}},
            "awayTeam": {"score": {"q1": 0, "q2": 0, "q3": 0, "q4": 3, "total": 3}},
        },
    }
    payload = "r:" + json.dumps(["$", None, None, {"games": [game]}]) + "\n"
    html = ("<script>self.__next_f.push(" + json.dumps([1, payload]) + ")</script>").encode()
    value = official.nfl_cards(html)["2026_07_BUF_ARI"]
    assert value["home_score"] == 7 and value["away_score"] == 3


def test_nhl_landing_uses_explicit_period_descriptor_when_gameoutcome_absent():
    body = json.loads(nhl_body())
    body.pop("gameOutcome")
    body["periodDescriptor"] = {"number": 5, "periodType": "SO", "maxRegulationPeriods": 3}
    value = official.nhl_landing(json.dumps(body).encode(), SOURCE)
    assert value["status"] == "final" and value["outcome_type"] == "SO"
    body["gameOutcome"] = {"lastPeriodType": "REG"}
    assert official.nhl_landing(json.dumps(body).encode(), SOURCE)["status"] == "disputed"
