"""Bounded transport failures, response validation and immutable resume semantics."""

import hashlib
import json
from http.client import RemoteDisconnected

import pytest
import requests

from predictions import league_data
from predictions.league_data import DownloadError, cached_snapshot, snapshot

URL = "https://example.org/public-data"


def response(status=200, body=b'{"ok":true}', headers=None):
    r = requests.Response()
    r.status_code = status
    r._content = body
    r._content_consumed = True
    r.url = "https://example.org/temporary?secret=must-not-be-logged"
    r.headers.update(headers or {})
    return r


def get_sequence(monkeypatch, values):
    values = iter(values)
    calls, delays = [], []

    def get(url, **kwargs):
        calls.append((url, kwargs))
        value = next(values)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(league_data.requests, "get", get)
    monkeypatch.setattr(league_data.time, "sleep", delays.append)
    return calls, delays


def test_remote_disconnect_timeout_and_success_have_bounded_safe_diagnostics(
    monkeypatch, tmp_path, caplog
):
    calls, delays = get_sequence(
        monkeypatch,
        [
            requests.ConnectionError(RemoteDisconnected("closed")),
            requests.ReadTimeout("read timeout"),
            response(),
        ],
    )
    path, meta = snapshot(URL, tmp_path, attempts=3, timeout=(8, 20), label="NBA 2010-11")
    assert len(calls) == 3 and delays == [1, 2]
    assert meta["url"] == URL and meta["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert "NBA 2010-11" in caplog.text and "RemoteDisconnected" in caplog.text
    assert "must-not-be-logged" not in caplog.text
    assert all(c[1]["timeout"] == (8, 20) and c[1].get("verify", True) for c in calls)


def test_rate_limit_honors_bounded_retry_after_and_transient_server_error(monkeypatch, tmp_path):
    calls, delays = get_sequence(
        monkeypatch, [response(429, headers={"Retry-After": "5000"}), response(503), response()]
    )
    snapshot(URL, tmp_path, attempts=3)
    assert len(calls) == 3 and delays == [30, 2]


def test_http_403_is_not_retried(monkeypatch, tmp_path):
    calls, delays = get_sequence(monkeypatch, [response(403)])
    with pytest.raises(DownloadError, match="HTTP 403"):
        snapshot(URL, tmp_path, label="NBA 2010-11")
    assert len(calls) == 1 and delays == [] and not list(tmp_path.glob("*.metadata.json"))


def test_invalid_response_is_not_published_and_valid_retry_is_checksummed(monkeypatch, tmp_path):
    calls, _ = get_sequence(monkeypatch, [response(body=b"<html>blocked</html>"), response()])
    path, meta = snapshot(
        URL,
        tmp_path,
        validator=json.loads,
        expected_sha256=hashlib.sha256(b'{"ok":true}').hexdigest(),
    )
    assert len(calls) == 2 and json.loads(path.read_bytes())["ok"]
    assert meta["published_sha256"] == meta["sha256"]
    assert len(list(tmp_path.glob("*.raw"))) == 1


def test_partial_transfer_does_not_replace_previous_verified_snapshot(monkeypatch, tmp_path):
    get_sequence(monkeypatch, [response()])
    path, meta = snapshot(URL, tmp_path)
    pointer = tmp_path / (league_data.cache_key(URL) + ".metadata.json")
    previous = pointer.read_bytes()
    calls, _ = get_sequence(
        monkeypatch, [requests.exceptions.ChunkedEncodingError("incomplete")] * 3
    )
    with pytest.raises(DownloadError, match="3 attempts"):
        snapshot(URL, tmp_path, refresh=True, attempts=3)
    assert len(calls) == 3 and pointer.read_bytes() == previous
    assert cached_snapshot(URL, tmp_path) == (path, meta)


def test_resume_requires_no_network_and_detects_corruption(monkeypatch, tmp_path):
    calls, _ = get_sequence(monkeypatch, [response()])
    first = snapshot(URL, tmp_path)
    assert snapshot(URL, tmp_path) == first and len(calls) == 1
    first[0].write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        snapshot(URL, tmp_path)
    assert len(calls) == 1


def test_published_checksum_and_cache_path_validation_fail_closed(monkeypatch, tmp_path):
    get_sequence(monkeypatch, [response()] * 3)
    with pytest.raises(DownloadError, match="published SHA256"):
        snapshot(URL, tmp_path, attempts=3, expected_sha256="a" * 64)
    assert not list(tmp_path.glob("*.metadata.json"))
    pointer = tmp_path / (league_data.cache_key(URL) + ".metadata.json")
    pointer.write_text(json.dumps({"file": "../outside", "sha256": "a" * 64}))
    with pytest.raises(ValueError, match="filename"):
        cached_snapshot(URL, tmp_path)


@pytest.mark.parametrize(
    "options",
    [
        {"attempts": 0},
        {"attempts": 6},
        {"attempts": True},
        {"timeout": 0},
        {"timeout": float("inf")},
        {"expected_sha256": "not-a-digest"},
    ],
)
def test_invalid_download_configuration_is_rejected_without_network(tmp_path, options):
    with pytest.raises(ValueError):
        snapshot(URL, tmp_path, **options)
