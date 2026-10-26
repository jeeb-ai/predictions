"""Offline source-adapter tests; no access to live endpoints is needed."""

from unittest.mock import Mock

import pandas as pd
import pytest
import requests

from predictions.data import SOURCE_URL, NflverseProvider, load_games


@pytest.fixture
def schedule():
    return pd.DataFrame(
        [
            {
                "game_id": "2020_01_KC_HOU",
                "season": 2020,
                "game_type": "REG",
                "week": 1,
                "gameday": "2020-09-10",
                "gametime": "20:20",
                "away_team": "HOU",
                "home_team": "KC",
                "away_score": 20,
                "home_score": 34,
                "location": "Home",
                "spread_line": 9.5,
            }
        ]
    )


def write_csv(tmp_path, schedule):
    path = tmp_path / "input.csv"
    schedule.to_csv(path, index=False)
    return path


def response(content, status=200):
    result = Mock(status_code=status, content=content)
    if status >= 400:
        result.raise_for_status.side_effect = requests.HTTPError(f"HTTP {status}")
    return result


def test_canonical_schedule_timezone_and_excluded_columns(tmp_path, schedule):
    games = load_games(write_csv(tmp_path, schedule))
    assert games.loc[0, "kickoff"] == pd.Timestamp("2020-09-11T00:20:00Z")
    assert games.loc[0, "result_available_at"] == pd.Timestamp("2020-09-12T00:20:00Z")
    assert games.loc[0, "home_team"] == "KC"
    assert games.loc[0, "week"] == 1
    assert games.loc[0, "neutral"] == 0
    assert not games.loc[0, "kickoff_time_imputed"]
    assert "spread_line" not in games
    assert "gameday" not in games


def test_aliases_neutral_and_missing_time_conservative_bounds(tmp_path, schedule):
    schedule.loc[0, ["home_team", "away_team", "location", "gametime"]] = [
        "OAK",
        "SD",
        "Neutral",
        None,
    ]
    games = load_games(write_csv(tmp_path, schedule))
    assert games.loc[0, "home_team"] == "LV"
    assert games.loc[0, "away_team"] == "LAC"
    assert games.loc[0, "neutral"] == 1
    assert games.loc[0, "kickoff_time_imputed"]
    assert games.loc[0, "kickoff"] == pd.Timestamp("2020-09-10T04:00:00Z")
    assert games.loc[0, "result_available_at"] == pd.Timestamp("2020-09-12T03:59:00Z")


def test_as_of_masks_outcomes_at_exact_availability(tmp_path, schedule):
    path = write_csv(tmp_path, schedule)
    at = pd.Timestamp("2020-09-12T00:20:00Z")
    masked = load_games(path, as_of=at)
    assert len(masked) == 1
    assert masked[["home_score", "away_score"]].isna().all().all()
    later = load_games(path, as_of=at + pd.Timedelta(seconds=1))
    assert later.loc[0, "home_score"] == 34
    with pytest.raises(ValueError, match="timezone-aware"):
        load_games(path, as_of="2020-09-13")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("home_team", "MISSING", "unknown"),
        ("home_team", "HOU", "itself"),
        ("home_team", None, "missing"),
        ("home_score", -1, "invalid"),
        ("home_score", 1.5, "integers"),
        ("home_score", None, "Both scores"),
        ("home_score", float("inf"), "invalid"),
        ("season", 2020.5, "integers"),
        ("week", 0, "invalid"),
        ("location", "Away", "location"),
    ],
)
def test_invalid_schedule_rejected(tmp_path, schedule, field, value, message):
    schedule[field] = value
    with pytest.raises(ValueError, match=message):
        load_games(write_csv(tmp_path, schedule))


def test_missing_schema_duplicate_and_empty_rejected(tmp_path, schedule):
    with pytest.raises(ValueError, match="required columns"):
        load_games(write_csv(tmp_path, schedule.drop(columns="week")))
    with pytest.raises(ValueError, match="duplicate"):
        load_games(write_csv(tmp_path, pd.concat([schedule, schedule])))
    with pytest.raises(ValueError, match="no games"):
        load_games(write_csv(tmp_path, schedule.iloc[0:0]))


def test_schedule_with_unplayed_game_keeps_both_missing_scores(tmp_path, schedule):
    schedule[["home_score", "away_score"]] = float("nan")
    games = load_games(write_csv(tmp_path, schedule))
    assert games[["home_score", "away_score"]].isna().all().all()


def test_download_snapshots_cache_and_checksum(tmp_path, monkeypatch, schedule):
    raw = schedule.to_csv(index=False).encode()
    get = Mock(return_value=response(raw))
    monkeypatch.setattr("predictions.data.requests.get", get)
    provider = NflverseProvider(tmp_path / "cache")
    path, metadata = provider.fetch()
    assert path.read_bytes() == raw
    assert metadata["source_url"] == SOURCE_URL
    assert metadata["sha256"] == path.stem
    assert pd.Timestamp(metadata["fetched_at"]).tzinfo is not None
    assert "redistribution" in metadata["license_note"]
    assert provider.fetch() == (path, metadata)
    assert get.call_count == 1
    get.assert_called_once_with(SOURCE_URL, timeout=30)
    path.write_text("corrupt")
    with pytest.raises(ValueError, match="Cached nflverse snapshot"):
        provider.fetch()
    repaired, refreshed = provider.fetch(refresh=True)
    assert repaired.read_bytes() == raw
    assert refreshed["sha256"] == metadata["sha256"]
    assert get.call_count == 2


def test_retry_transient_failures_with_bounded_backoff(tmp_path, monkeypatch, schedule):
    get = Mock(
        side_effect=[
            requests.Timeout("timeout"),
            response(b"rate limited", 429),
            response(b"service unavailable", 503),
            response(schedule.to_csv(index=False).encode()),
        ]
    )
    sleep = Mock()
    monkeypatch.setattr("predictions.data.requests.get", get)
    monkeypatch.setattr("predictions.data.time.sleep", sleep)
    path, _ = NflverseProvider(tmp_path, retries=3).fetch()
    assert path.exists()
    assert get.call_count == 4
    assert [call.args[0] for call in sleep.call_args_list] == [1, 2, 4]


def test_permanent_failure_not_retried_or_published(tmp_path, monkeypatch):
    get = Mock(return_value=response(b"not found", 404))
    monkeypatch.setattr("predictions.data.requests.get", get)
    with pytest.raises(requests.HTTPError):
        NflverseProvider(tmp_path).fetch()
    assert get.call_count == 1
    assert not (tmp_path / "latest.json").exists()


def test_exhausted_timeout_does_not_publish_snapshot(tmp_path, monkeypatch):
    get = Mock(side_effect=requests.Timeout("timeout"))
    monkeypatch.setattr("predictions.data.requests.get", get)
    monkeypatch.setattr("predictions.data.time.sleep", Mock())
    with pytest.raises(requests.Timeout):
        NflverseProvider(tmp_path, retries=2).fetch()
    assert get.call_count == 3
    assert not (tmp_path / "latest.json").exists()


def test_bad_success_response_does_not_publish_snapshot(tmp_path, monkeypatch):
    get = Mock(return_value=response(b"<html>not a schedule</html>"))
    monkeypatch.setattr("predictions.data.requests.get", get)
    with pytest.raises(ValueError, match="required columns"):
        NflverseProvider(tmp_path).fetch()
    assert not (tmp_path / "latest.json").exists()


def test_bad_cache_metadata_rejected(tmp_path):
    (tmp_path / "latest.json").write_text('{"sha256":"../escape"}')
    with pytest.raises(ValueError, match="Cached nflverse snapshot"):
        NflverseProvider(tmp_path).fetch()


def test_unavailable_scores_can_still_be_loaded_as_future_schedule(tmp_path, schedule):
    path = write_csv(tmp_path, schedule)
    future = load_games(path, as_of="2010-01-01T00:00:00Z")
    assert len(future) == 1
    assert future["home_score"].isna().all()


def test_input_rows_are_sorted_by_true_kickoff(tmp_path, schedule):
    later = schedule.copy()
    later["game_id"] = "later"
    later["gameday"] = "2020-09-11"
    combined = pd.concat([later, schedule], ignore_index=True)
    games = load_games(write_csv(tmp_path, combined))
    assert games["game_id"].tolist() == ["2020_01_KC_HOU", "later"]
