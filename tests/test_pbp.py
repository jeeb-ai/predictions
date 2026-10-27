"""Local play-by-play fixtures and mocked HTTPS exercise the real adapter."""

import hashlib
import io
import json

import pandas as pd
import pytest
import requests

from predictions.pbp import PbpProvider, aggregate_pbp, read_pbp


@pytest.fixture
def plays():
    base = {
        "game_id": "2024_01_BUF_NE",
        "posteam": "BUF",
        "defteam": "NE",
        "play_type": "pass",
        "rush_attempt": 0,
        "qb_dropback": 1,
        "qb_scramble": 0,
        "qb_kneel": 0,
        "qb_spike": 0,
        "penalty": 0,
        "passer_player_id": "qb-one",
        "rusher_player_id": None,
        "epa": 0.4,
        "yards_gained": 20,
    }
    changes = [
        {},
        {
            "play_type": "run",
            "rush_attempt": 1,
            "qb_dropback": 0,
            "passer_player_id": None,
            "rusher_player_id": "rb-one",
            "epa": 0.2,
            "yards_gained": 10,
        },
        {"epa": -0.8, "yards_gained": -5},  # Sack remains a dropback.
        {
            "play_type": "run",
            "rush_attempt": 1,
            "qb_scramble": 1,
            "passer_player_id": None,
            "rusher_player_id": "qb-one",
            "epa": 0.3,
            "yards_gained": 12,
        },
        {"epa": 0.0, "yards_gained": 0},
    ]
    return pd.DataFrame([{**base, **change, "play_id": n} for n, change in enumerate(changes)])


def _parquet_bytes(frame):
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    return buffer.getvalue()


class _Response:
    def __init__(self, content=b"", status=200, fail_during_stream=False):
        self.content = content
        self.status_code = status
        self.fail_during_stream = fail_during_stream

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size):
        yield self.content[:7]
        if self.fail_during_stream:
            raise requests.ConnectionError("interrupted stream")
        yield self.content[7:]


def test_scrimmage_epa_success_explosives_and_qb_denominators(plays):
    team, qb = aggregate_pbp(plays)
    assert len(team) == len(qb) == 1
    assert team.iloc[0]["plays"] == 5
    assert team.iloc[0]["epa_sum"] == pytest.approx(0.1)
    assert team.iloc[0]["successes"] == 3
    assert team.iloc[0]["explosive_plays"] == 2
    assert qb.iloc[0]["qb_id"] == "qb-one"
    assert qb.iloc[0]["dropbacks"] == 4
    assert qb.iloc[0]["epa_sum"] == pytest.approx(-0.1)
    assert qb.iloc[0]["successes"] == 2


@pytest.mark.parametrize(
    "change",
    [
        {"play_type": "no_play"},
        {"penalty": 1},
        {"qb_kneel": 1},
        {"qb_spike": 1},
        {"special_teams_play": 1},
        {"no_play": 1},
        {"play_type": "kickoff"},
    ],
)
def test_excludes_administrative_penalty_and_non_scrimmage_plays(plays, change):
    frame = plays.iloc[:1].copy()
    for key, value in change.items():
        frame[key] = value
    team, qb = aggregate_pbp(frame)
    assert team.empty
    assert qb.empty


def test_missing_epa_is_not_fabricated_and_coverage_is_retained(plays):
    frame = plays.copy()
    frame.loc[0, "epa"] = float("nan")
    team, _ = aggregate_pbp(frame)
    assert team.iloc[0]["plays"] == 4
    assert team.iloc[0]["eligible_plays"] == 5
    assert team.iloc[0]["missing_epa_plays"] == 1
    assert team.iloc[0]["epa_sum"] == pytest.approx(-0.3)


def test_unknown_qb_not_assigned_to_another_player(plays):
    frame = plays.iloc[:1].copy()
    frame["passer_player_id"] = None
    frame["rusher_player_id"] = "not-a-qb"
    team, qb = aggregate_pbp(frame)
    assert len(team) == 1
    assert qb.empty


def test_aliases_are_canonical(plays):
    frame = plays.copy()
    frame["posteam"] = "OAK"
    frame["defteam"] = "SD"
    team, qb = aggregate_pbp(frame)
    assert team.iloc[0]["team"] == "LV"
    assert team.iloc[0]["opponent"] == "LAC"
    assert qb.iloc[0]["team"] == "LV"


@pytest.mark.parametrize(
    "column,value,match",
    [
        ("posteam", "BAD", "unknown"),
        ("defteam", "BUF", "itself"),
        ("epa", float("inf"), "EPA"),
        ("rush_attempt", 2, "binary"),
        ("yards_gained", float("nan"), "yards gained"),
    ],
)
def test_rejects_malformed_eligible_plays(plays, column, value, match):
    frame = plays.copy()
    frame[column] = value
    with pytest.raises(ValueError, match=match):
        aggregate_pbp(frame)


def test_rejects_duplicate_play_identity(plays):
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate_pbp(pd.concat([plays, plays.iloc[:1]]))


def test_override_replaces_upstream_epa_by_identity_not_row_order(plays, tmp_path):
    override = plays[["game_id", "play_id"]].iloc[::-1].copy()
    override["epa"] = -1.0
    path = tmp_path / "causal.parquet"
    override.to_parquet(path, index=False)
    team, qb = aggregate_pbp(plays, epa_override=path)
    assert team.iloc[0]["epa_sum"] == -5
    assert team.iloc[0]["successes"] == 0
    assert qb.iloc[0]["epa_sum"] == -4


def test_override_must_cover_raw_plays_and_be_unique(plays):
    override = plays[["game_id", "play_id", "epa"]].copy()
    with pytest.raises(ValueError, match="cover every"):
        aggregate_pbp(plays, epa_override=override.iloc[1:])
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_pbp(plays, epa_override=pd.concat([override, override.iloc[:1]]))


def test_fetch_preserves_snapshot_provenance_and_reuses_verified_cache(
    plays, tmp_path, monkeypatch
):
    content = _parquet_bytes(plays)
    calls = []

    def fetch(url, **kwargs):
        calls.append((url, kwargs))
        return _Response(content)

    monkeypatch.setattr("predictions.pbp.requests.get", fetch)
    provider = PbpProvider(tmp_path)
    path, metadata = provider.fetch(2024)
    assert path.read_bytes() == content
    assert metadata["sha256"] == hashlib.sha256(content).hexdigest()
    assert metadata["license"] == "CC-BY-4.0"
    assert metadata["season"] == 2024
    assert pd.Timestamp(metadata["fetched_at"]).tzinfo is not None
    assert "retrospective" in metadata["epa_provenance_note"]
    assert provider.fetch(2024) == (path, metadata)
    assert provider.fetch(2024, offline=True) == (path, metadata)
    assert len(calls) == 1
    assert calls[0][1]["stream"] is True
    assert "verify" not in calls[0][1]
    assert list(tmp_path.glob("2024_*_*.json"))
    pd.testing.assert_frame_equal(read_pbp(path), plays, check_like=True)


def test_corrupt_cache_requires_explicit_refresh(plays, tmp_path, monkeypatch):
    content = _parquet_bytes(plays)
    monkeypatch.setattr("predictions.pbp.requests.get", lambda *a, **kw: _Response(content))
    provider = PbpProvider(tmp_path)
    path, _ = provider.fetch(2024)
    path.write_bytes(b"broken")
    with pytest.raises(ValueError, match="Cached PBP"):
        provider.fetch(2024)
    restored, _ = provider.fetch(2024, refresh=True)
    assert restored.read_bytes() == content


def test_cache_cannot_redirect_to_another_season(plays, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "predictions.pbp.requests.get", lambda *a, **kw: _Response(_parquet_bytes(plays))
    )
    provider = PbpProvider(tmp_path)
    _, metadata = provider.fetch(2024)
    metadata["season"] = 2023
    (tmp_path / "latest-2024.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Cached PBP"):
        provider.fetch(2024)


@pytest.mark.parametrize("failure", ["timeout", "stream", "429", "503"])
def test_transient_fetch_retries_without_publishing_partial_snapshot(
    plays, tmp_path, monkeypatch, failure
):
    content = _parquet_bytes(plays)
    calls, sleeps = [], []

    def fetch(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            if failure == "timeout":
                raise requests.Timeout("fixture timeout")
            if failure == "stream":
                return _Response(content, fail_during_stream=True)
            return _Response(status=int(failure))
        return _Response(content)

    monkeypatch.setattr("predictions.pbp.requests.get", fetch)
    monkeypatch.setattr("predictions.pbp.time.sleep", sleeps.append)
    path, _ = PbpProvider(tmp_path, retries=1).fetch(2024)
    assert path.read_bytes() == content
    assert len(calls) == 2
    assert sleeps == [1]
    assert all(p.suffix in [".json", ".parquet"] for p in tmp_path.iterdir())


def test_permanent_http_error_is_not_retried(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "predictions.pbp.requests.get", lambda *a, **kw: calls.append(1) or _Response(status=404)
    )
    with pytest.raises(requests.HTTPError):
        PbpProvider(tmp_path).fetch(2024)
    assert len(calls) == 1
    assert not list(tmp_path.iterdir())


def test_invalid_parquet_does_not_publish_cache_pointer(tmp_path, monkeypatch):
    monkeypatch.setattr("predictions.pbp.requests.get", lambda *a, **kw: _Response(b"not parquet"))
    with pytest.raises(ValueError, match="Parquet"):
        PbpProvider(tmp_path).fetch(2024)
    assert not list(tmp_path.iterdir())


def test_selective_read_checks_missing_requested_columns(plays, tmp_path):
    path = tmp_path / "fixture.parquet"
    plays.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="requested columns"):
        read_pbp(path, extra_columns=("missing-context",))


@pytest.mark.parametrize("season", [1998, "2024", True])
def test_invalid_seasons_rejected(tmp_path, season):
    with pytest.raises(ValueError, match="season"):
        PbpProvider(tmp_path).fetch(season)


def test_uncached_offline_fetch_never_attempts_http(tmp_path, monkeypatch):
    def no_http(*args, **kwargs):
        raise AssertionError("Offline replay must never request HTTP")

    monkeypatch.setattr("predictions.pbp.requests.get", no_http)
    with pytest.raises(FileNotFoundError, match="offline"):
        PbpProvider(tmp_path).fetch(2024, offline=True)


def test_offline_and_refresh_conflict(tmp_path):
    with pytest.raises(ValueError, match="offline and refresh"):
        PbpProvider(tmp_path).fetch(2024, offline=True, refresh=True)
