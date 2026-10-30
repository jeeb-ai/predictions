"""Public read-only snapshots with retry, provenance, TLS and checksum checks."""

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests


def _snapshot(url, root, *, params=None, refresh=False, timeout=40):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(json.dumps([url, params], sort_keys=True).encode()).hexdigest()
    pointer = root / (key + ".metadata.json")
    if pointer.exists() and not refresh:
        meta = json.loads(pointer.read_text())
        path = root / meta["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != meta["sha256"]:
            raise ValueError("League snapshot checksum mismatch")
        return path, meta
    for attempt in range(4):
        try:
            response = requests.get(
                url,
                params=params,
                headers={"User-Agent": "Mozilla/5.0", "Referer": "https://www.nba.com/"},
                timeout=timeout,
                allow_redirects=True,
            )
            if response.status_code == 429 or response.status_code >= 500:
                if attempt < 3:
                    time.sleep(min(2**attempt, 8))
                    continue
            response.raise_for_status()
            break
        except (requests.Timeout, requests.ConnectionError):
            if attempt == 3:
                raise
            time.sleep(2**attempt)
    content = response.content
    sha = hashlib.sha256(content).hexdigest()
    path = root / (sha + ".raw")
    if not path.exists():
        path.write_bytes(content)
    meta = {
        "url": requests.Request("GET", url, params=params).prepare().url,
        "sha256": sha,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "file": path.name,
        "bytes": len(content),
        "rights": "Public access does not grant redistribution or commercial rights; review provider terms.",
    }
    evidence = root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + "-" + sha + ".json")
    evidence.write_text(json.dumps(meta, indent=2) + "\n")
    pointer.write_text(json.dumps(meta, indent=2) + "\n")
    return path, meta


def snapshot(url, root, *, params=None, refresh=False, timeout=40):
    """Serialize identical cache keys, including concurrent release-catalog reads."""
    import fcntl

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(json.dumps([url, params], sort_keys=True).encode()).hexdigest()
    with (root / (key + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _snapshot(url, root, params=params, refresh=refresh, timeout=timeout)
