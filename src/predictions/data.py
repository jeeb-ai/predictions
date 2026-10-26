"""Read-only nflverse schedule adapter and conservative historical timestamps.

The schedule is a present-day snapshot, not a point-in-time archive. Final scores
are assumed usable 24 hours after kickoff; actual historical publication times
are unavailable. Missing kickoff times use the beginning of the day for feature
cutoffs and the end of the day plus 24 hours for outcome availability.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

SOURCE_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
LICENSE_URL = "https://github.com/nflverse/nfldata"
LICENSE_NOTE = (
    "No explicit license file was found in the source repository during implementation. "
    "Public availability is not a redistribution license; review upstream and "
    "underlying-source terms before redistribution or commercial use."
)
REQUIRED_COLUMNS = {
    "game_id",
    "season",
    "game_type",
    "week",
    "gameday",
    "gametime",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "location",
}
TEAM_ALIASES = {
    "SD": "LAC",
    "OAK": "LV",
    "STL": "LA",
    "LAR": "LA",
    "JAC": "JAX",
    "WSH": "WAS",
}
TEAMS = set(
    "ARI ATL BAL BUF CAR CHI CIN CLE DAL DEN DET GB HOU IND JAX KC LA LAC LV "
    "MIA MIN NE NO NYG NYJ PHI PIT SEA SF TB TEN WAS".split()
)


def _atomic_write(path: Path, content: bytes) -> None:
    """Replace one generated cache file without exposing a partial write."""
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class NflverseProvider:
    """Fetch and checksum raw CSV snapshots; no credentials or market access."""

    def __init__(self, cache_dir: Path, timeout: float = 30, retries: int = 3):
        if timeout <= 0 or retries < 0:
            raise ValueError("timeout must be positive and retries nonnegative")
        self.cache_dir = Path(cache_dir)
        self.timeout = timeout
        self.retries = retries

    def fetch(self, refresh: bool = False) -> tuple[Path, dict]:
        """Return a verified cached snapshot, or fetch and record a new one.

        ``retries`` counts additional attempts after the first request. Transient
        connection failures, HTTP 429, and server failures receive bounded
        exponential backoff. Permanent HTTP errors are not retried.
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        latest = self.cache_dir / "latest.json"
        if latest.exists() and not refresh:
            try:
                metadata = json.loads(latest.read_text())
                digest = metadata["sha256"]
                if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                    raise ValueError("invalid snapshot checksum")
                if metadata["source_url"] != SOURCE_URL:
                    raise ValueError("unexpected snapshot source")
                fetched_at = pd.Timestamp(metadata["fetched_at"])
                if fetched_at.tzinfo is None:
                    raise ValueError("snapshot timestamp must include a timezone")
                path = self.cache_dir / f"{digest}.csv"
                if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    raise ValueError("snapshot checksum mismatch")
                return path, metadata
            except (OSError, ValueError, KeyError, TypeError) as error:
                raise ValueError(
                    "Cached nflverse snapshot is invalid; fetch with refresh=True "
                    "to replace it from the verified HTTPS source"
                ) from error

        for attempt in range(self.retries + 1):
            try:
                response = requests.get(SOURCE_URL, timeout=self.timeout)
                transient_status = response.status_code == 429 or response.status_code >= 500
                if transient_status and attempt < self.retries:
                    time.sleep(min(2**attempt, 8))
                    continue
                response.raise_for_status()
                content = response.content
                # Reject an HTTP-200 HTML error page or broken upstream schema
                # before publishing a new cache pointer.
                _normalize_games(pd.read_csv(io.BytesIO(content)), as_of=None)
                break
            except (requests.ConnectionError, requests.Timeout):
                if attempt == self.retries:
                    raise
                time.sleep(min(2**attempt, 8))
        digest = hashlib.sha256(content).hexdigest()
        path = self.cache_dir / f"{digest}.csv"
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            _atomic_write(path, content)
        metadata = {
            "sha256": digest,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "source_url": SOURCE_URL,
            "license_url": LICENSE_URL,
            "license_note": LICENSE_NOTE,
            "bytes": len(content),
        }
        encoded = json.dumps(metadata, indent=2, sort_keys=True).encode() + b"\n"
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        _atomic_write(self.cache_dir / f"{stamp}_{digest}.json", encoded)
        _atomic_write(latest, encoded)
        return path, metadata


def _normalize_games(frame: pd.DataFrame, as_of=None) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"NFL schedule is missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("NFL schedule has no games")
    games = frame[list(REQUIRED_COLUMNS)].copy()
    for field in ("game_id", "game_type", "home_team", "away_team", "gameday", "location"):
        if games[field].isna().any():
            raise ValueError(f"NFL schedule has missing {field}")
        games[field] = games[field].astype(str).str.strip()
        if games[field].eq("").any():
            raise ValueError(f"NFL schedule has blank {field}")
    if games["game_id"].duplicated().any():
        raise ValueError("NFL schedule contains duplicate game_id values")
    for field in ("season", "week"):
        values = pd.to_numeric(games[field], errors="raise")
        if values.isna().any() or not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError(f"NFL schedule has invalid {field}")
        if not np.equal(values, np.floor(values)).all():
            raise ValueError(f"NFL schedule {field} must contain integers")
        games[field] = values.astype(int)
    for field in ("home_team", "away_team"):
        games[field] = games[field].str.upper().replace(TEAM_ALIASES)
        if not games[field].isin(TEAMS).all():
            raise ValueError(f"NFL schedule has unknown {field} codes")
    if games["home_team"].eq(games["away_team"]).any():
        raise ValueError("A team cannot play itself")
    for field in ("home_score", "away_score"):
        values = pd.to_numeric(games[field], errors="raise")
        known = values.dropna()
        if not np.isfinite(known).all() or (known < 0).any():
            raise ValueError(f"NFL schedule has invalid {field}")
        if not np.equal(known, np.floor(known)).all():
            raise ValueError("NFL scores must be nonnegative integers")
        games[field] = values.astype(float)
    if games["home_score"].isna().ne(games["away_score"].isna()).any():
        raise ValueError("Both scores must be present or both absent")
    locations = games["location"].str.casefold()
    if not locations.isin(["home", "neutral"]).all():
        raise ValueError("NFL schedule location must be Home or Neutral")
    games["neutral"] = locations.eq("neutral").astype(int)

    raw_time = games["gametime"].fillna("").astype(str).str.strip()
    imputed = raw_time.eq("")
    earliest = pd.to_datetime(
        games["gameday"] + " " + raw_time.mask(imputed, "00:00"),
        format="%Y-%m-%d %H:%M",
        errors="raise",
    )
    latest = pd.to_datetime(
        games["gameday"] + " " + raw_time.mask(imputed, "23:59"),
        format="%Y-%m-%d %H:%M",
        errors="raise",
    )
    games["kickoff"] = earliest.dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    games["result_available_at"] = latest.dt.tz_localize("America/New_York").dt.tz_convert(
        "UTC"
    ) + pd.Timedelta(hours=24)
    games["kickoff_time_imputed"] = imputed
    if as_of is not None:
        cutoff = pd.Timestamp(as_of)
        if cutoff.tzinfo is None or pd.isna(cutoff):
            raise ValueError("as_of must be a valid timezone-aware timestamp")
        games.loc[games["result_available_at"] >= cutoff, ["home_score", "away_score"]] = np.nan
    columns = [
        "game_id",
        "season",
        "game_type",
        "week",
        "kickoff",
        "result_available_at",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "neutral",
        "kickoff_time_imputed",
    ]
    return games[columns].sort_values(["kickoff", "game_id"]).reset_index(drop=True)


def load_games(path: Path, *, as_of=None) -> pd.DataFrame:
    """Load validated canonical schedule fields, dropping odds and result columns.

    ``as_of`` must be timezone-aware. Scores not conservatively available strictly
    before that instant are masked, while their schedule rows remain available.
    """
    return _normalize_games(pd.read_csv(path), as_of=as_of)
