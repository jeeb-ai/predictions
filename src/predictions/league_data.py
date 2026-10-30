"""Public read-only snapshots with bounded retries, atomic writes and checksums."""

import hashlib
import json
import logging
import math
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from http.client import RemoteDisconnected
from pathlib import Path

import requests

from predictions.data import _atomic_write

LOG = logging.getLogger(__name__)


class DownloadError(requests.RequestException):
    """One bounded request failed; contains canonical URLs, never signed redirects."""

    def __init__(self, url, label, failures):
        self.url = url
        self.failures = failures
        super().__init__(
            f"{label or 'League data'}: GET {url} failed after {len(failures)} attempts; "
            f"{failures[-1]}. Verified earlier downloads are retained."
        )


def cache_key(url, params=None):
    return hashlib.sha256(json.dumps([url, params], sort_keys=True).encode()).hexdigest()


def cached_snapshot(url, root, *, params=None, validator=None, expected_sha256=None):
    """Read an existing snapshot without network access; corruption fails explicitly."""
    root = Path(root)
    pointer = root / (cache_key(url, params) + ".metadata.json")
    if not pointer.exists():
        return None
    try:
        meta = json.loads(pointer.read_text())
        name = meta["file"]
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("Invalid snapshot filename")
        path = root / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != meta["sha256"] or (expected_sha256 and digest != expected_sha256):
            raise ValueError("League snapshot checksum mismatch")
        if not path.is_file() or path.resolve().parent != root.resolve():
            raise ValueError("Snapshot escapes cache")
        if validator is not None:
            validator(path.read_bytes())
        return path, meta
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid cached league snapshot {pointer.name}: {exc}") from exc


def _transport_reason(error):
    stack, seen = [error], set()
    while stack:
        item = stack.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        if isinstance(item, RemoteDisconnected):
            return "RemoteDisconnected"
        if isinstance(item, BaseException):
            stack.extend(item.args)
            stack.extend([item.__cause__, item.__context__])
        elif isinstance(item, (tuple, list)):
            stack.extend(item)
    return type(error).__name__


def _retry_delay(response, attempt):
    delay = min(2**attempt, 8)
    header = response.headers.get("Retry-After") if response is not None else None
    if header:
        try:
            requested = float(header)
        except ValueError:
            try:
                requested = (
                    parsedate_to_datetime(header) - datetime.now(timezone.utc)
                ).total_seconds()
            except (ValueError, TypeError, OverflowError):
                requested = delay
        if math.isfinite(requested):
            delay = max(delay, min(30, requested))
    return delay


def _snapshot(
    url,
    root,
    *,
    params=None,
    refresh=False,
    timeout=40,
    attempts=4,
    validator=None,
    expected_sha256=None,
    label=None,
):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    key = cache_key(url, params)
    pointer = root / (key + ".metadata.json")
    if not refresh:
        existing = cached_snapshot(
            url, root, params=params, validator=validator, expected_sha256=expected_sha256
        )
        if existing is not None:
            return existing
    canonical_url = requests.Request("GET", url, params=params).prepare().url
    failures = []
    for attempt in range(attempts):
        response = None
        try:
            response = requests.get(
                url,
                params=params,
                headers={"User-Agent": "Mozilla/5.0", "Referer": "https://www.nba.com/"},
                timeout=timeout,
                allow_redirects=True,
            )
            response.raise_for_status()
            content = response.content
            if not content:
                raise ValueError("empty response")
            sha = hashlib.sha256(content).hexdigest()
            if expected_sha256 and sha != expected_sha256:
                raise ValueError("published SHA256 mismatch")
            if validator is not None:
                validator(content)
            fetched = datetime.now(timezone.utc)
            path = root / (sha + ".raw")
            # Never expose a partial body or metadata pointer after an interrupted transfer.
            if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != sha:
                _atomic_write(path, content)
            meta = {
                "url": canonical_url,
                "sha256": sha,
                "fetched_at": fetched.isoformat(),
                "file": path.name,
                "bytes": len(content),
                "rights": "Public access does not grant redistribution or commercial rights; review provider terms.",
            }
            if expected_sha256:
                meta["published_sha256"] = expected_sha256
            evidence = root / (fetched.strftime("%Y%m%dT%H%M%S%f") + "-" + sha + ".json")
            payload = (json.dumps(meta, indent=2) + "\n").encode()
            _atomic_write(evidence, payload)
            _atomic_write(pointer, payload)
            return path, meta
        except requests.HTTPError as exc:
            status = response.status_code
            failures.append(f"HTTP {status}")
            if status not in [408, 425, 429] and status < 500:
                raise DownloadError(canonical_url, label, failures) from exc
        except (
            requests.Timeout,
            requests.ConnectionError,
            requests.exceptions.ChunkedEncodingError,
        ) as exc:
            # Avoid printing temporary signed URLs/proxy credentials from nested exceptions.
            reason = _transport_reason(exc)
            failures.append(reason)
        except ValueError as exc:
            failures.append(f"response validation: {exc}")
        finally:
            if response is not None:
                response.close()
        LOG.warning(
            "%s GET %s attempt %d/%d: %s",
            label or "League data",
            canonical_url,
            attempt + 1,
            attempts,
            failures[-1],
        )
        if attempt + 1 < attempts:
            time.sleep(_retry_delay(response, attempt))
    raise DownloadError(canonical_url, label, failures)


def snapshot(
    url,
    root,
    *,
    params=None,
    refresh=False,
    timeout=40,
    attempts=4,
    validator=None,
    expected_sha256=None,
    label=None,
):
    """Serialize cache keys; retries are bounded, including 429 Retry-After (max 30s)."""
    import fcntl

    if not isinstance(attempts, int) or isinstance(attempts, bool) or not 1 <= attempts <= 5:
        raise ValueError("Download attempts must be between 1 and 5")
    if expected_sha256 is not None and (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(c not in "0123456789abcdef" for c in expected_sha256)
    ):
        raise ValueError("Invalid expected SHA256")
    limits = timeout if isinstance(timeout, tuple) else (timeout,)
    if any(not math.isfinite(float(t)) or t <= 0 for t in limits):
        raise ValueError("Download timeouts must be finite and positive")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / (cache_key(url, params) + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _snapshot(
            url,
            root,
            params=params,
            refresh=refresh,
            timeout=timeout,
            attempts=attempts,
            validator=validator,
            expected_sha256=expected_sha256,
            label=label,
        )
