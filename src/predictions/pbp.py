"""Read-only nflverse play-by-play snapshots and transparent game aggregates.

The supplied nflfastR EPA is retrospective, learned-model output. Its training
vintage is not an as-of guarantee. Historical forecasting must use a causally
fitted EPA replacement or explicitly label raw-EPA analyses retrospective.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests

from predictions.data import TEAM_ALIASES, TEAMS, _atomic_write

SOURCE_TEMPLATE = (
    "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet"
)
LICENSE_URL = "https://github.com/nflverse/nflverse-data/blob/main/LICENSE.md"
EPA_PROVENANCE_NOTE = (
    "Raw nflfastR EPA is retrospective learned-model output, not an archived "
    "historical-model vintage. Its original training data may postdate an early "
    "evaluation season. Use the causally fitted EPA pipeline for historical prediction."
)
REQUIRED_COLUMNS = [
    "game_id",
    "play_id",
    "posteam",
    "defteam",
    "epa",
    "yards_gained",
    "play_type",
    "rush_attempt",
    "qb_dropback",
    "qb_scramble",
    "qb_kneel",
    "qb_spike",
    "penalty",
    "passer_player_id",
    "rusher_player_id",
]
OPTIONAL_COLUMNS = ["special_teams_play", "no_play"]
TEAM_COLUMNS = [
    "game_id",
    "team",
    "opponent",
    "plays",
    "epa_sum",
    "successes",
    "explosive_plays",
    "eligible_plays",
    "missing_epa_plays",
]
QB_COLUMNS = ["game_id", "team", "qb_id", "dropbacks", "epa_sum", "successes"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _schema(path: Path) -> list[str]:
    try:
        schema = pq.ParquetFile(path).schema_arrow.names
    except Exception as exc:
        raise ValueError("PBP snapshot is not readable Parquet") from exc
    missing = set(REQUIRED_COLUMNS) - set(schema)
    if missing:
        raise ValueError(f"PBP snapshot is missing required columns: {sorted(missing)}")
    return schema


class PbpProvider:
    """Stream and verify immutable season snapshots without loading full files in RAM."""

    def __init__(self, cache_dir: Path, timeout: float = 30, retries: int = 3):
        if (
            timeout <= 0
            or not math.isfinite(timeout)
            or not isinstance(retries, int)
            or isinstance(retries, bool)
            or retries < 0
        ):
            raise ValueError("timeout must be positive/finite and retries nonnegative")
        self.cache_dir = Path(cache_dir)
        self.timeout = timeout
        self.retries = retries

    def fetch(self, season: int, refresh: bool = False, offline: bool = False) -> tuple[Path, dict]:
        if not isinstance(season, int) or isinstance(season, bool) or season < 1999:
            raise ValueError("PBP season must be an integer at least 1999")
        if offline and refresh:
            raise ValueError("offline and refresh cannot be combined")
        source = SOURCE_TEMPLATE.format(season=season)
        pointer = self.cache_dir / f"latest-{season}.json"
        if offline and not pointer.exists():
            raise FileNotFoundError(
                f"No cached PBP snapshot for {season}; offline mode forbids HTTP"
            )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if pointer.exists() and not refresh:
            try:
                metadata = json.loads(pointer.read_text())
                digest = metadata["sha256"]
                if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                    raise ValueError("Invalid PBP checksum")
                if metadata["source_url"] != source or metadata["season"] != season:
                    raise ValueError("PBP metadata source or season mismatch")
                fetched = pd.Timestamp(metadata["fetched_at"])
                if fetched.tzinfo is None or pd.isna(fetched):
                    raise ValueError("PBP fetched_at must be timezone-aware")
                path = self.cache_dir / f"{digest}.parquet"
                if _sha256(path) != digest:
                    raise ValueError("PBP checksum mismatch")
                _schema(path)
                return path, metadata
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise ValueError(
                    f"Cached PBP snapshot for {season} is invalid; use refresh=True to refetch"
                ) from exc

        temporary = None
        for attempt in range(self.retries + 1):
            try:
                with requests.get(source, stream=True, timeout=self.timeout) as response:
                    transient = response.status_code == 429 or response.status_code >= 500
                    if transient and attempt < self.retries:
                        time.sleep(min(2**attempt, 8))
                        continue
                    response.raise_for_status()
                    with tempfile.NamedTemporaryFile(dir=self.cache_dir, delete=False) as handle:
                        temporary = Path(handle.name)
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                handle.write(chunk)
                        handle.flush()
                        os.fsync(handle.fileno())
                _schema(temporary)
                break
            except (requests.ConnectionError, requests.Timeout):
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
                    temporary = None
                if attempt == self.retries:
                    raise
                time.sleep(min(2**attempt, 8))
            except BaseException:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
                raise
        assert temporary is not None
        try:
            digest = _sha256(temporary)
            path = self.cache_dir / f"{digest}.parquet"
            if path.exists() and _sha256(path) == digest:
                temporary.unlink()
            else:
                os.replace(temporary, path)
            now = datetime.now(timezone.utc)
            metadata = {
                "season": season,
                "sha256": digest,
                "source_url": source,
                "fetched_at": now.isoformat(),
                "bytes": path.stat().st_size,
                "rows": pq.ParquetFile(path).metadata.num_rows,
                "license": "CC-BY-4.0",
                "license_url": LICENSE_URL,
                "attribution": "nflverse/nflverse-data and nflfastR contributors",
                "epa_provenance_note": EPA_PROVENANCE_NOTE,
            }
            encoded = json.dumps(metadata, indent=2, sort_keys=True).encode() + b"\n"
            stamp = now.strftime("%Y%m%dT%H%M%S%fZ")
            _atomic_write(self.cache_dir / f"{season}_{stamp}_{digest}.json", encoded)
            _atomic_write(pointer, encoded)
            return path, metadata
        finally:
            temporary.unlink(missing_ok=True)


def read_pbp(path: Path, extra_columns: tuple[str, ...] = ()) -> pd.DataFrame:
    """Read only aggregation columns plus explicitly requested causal-EP inputs."""
    path = Path(path)
    schema = _schema(path)
    missing = set(extra_columns) - set(schema)
    if missing:
        raise ValueError(f"PBP snapshot is missing requested columns: {sorted(missing)}")
    columns = list(dict.fromkeys(REQUIRED_COLUMNS + list(extra_columns)))
    columns.extend(column for column in OPTIONAL_COLUMNS if column in schema)
    return pd.read_parquet(path, columns=columns)


def aggregate_pbp(
    path: Path | pd.DataFrame,
    *,
    epa_override: Path | pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate eligible scrimmage plays with play-weighted sufficient statistics.

    ``epa_override`` replaces upstream EPA through a validated play-identity join;
    the Model 2 pipeline supplies its causally fitted reconstruction here.
    EPA-null plays are omitted, never assigned fabricated zero EPA;
    denominator coverage is retained. Success means EPA > 0. Explosive runs gain
    at least ten yards; explosive dropbacks gain at least twenty. Penalties,
    no-plays, kneels, spikes and special teams are excluded. QB scrambles use the
    rusher ID only when the passer ID is unavailable.
    """
    frame = path.copy() if isinstance(path, pd.DataFrame) else read_pbp(path)
    missing = set(REQUIRED_COLUMNS) - set(frame)
    if missing:
        raise ValueError(f"PBP data are missing required columns: {sorted(missing)}")
    if frame[["game_id", "play_id"]].isna().any().any():
        raise ValueError("PBP play identity cannot be missing")
    if frame.duplicated(["game_id", "play_id"]).any():
        raise ValueError("Duplicate (game_id, play_id) records are not allowed")
    if epa_override is not None:
        replacement = (
            epa_override.copy()
            if isinstance(epa_override, pd.DataFrame)
            else pd.read_parquet(epa_override, columns=["game_id", "play_id", "epa"])
        )
        if not {"game_id", "play_id", "epa"}.issubset(replacement):
            raise ValueError("EPA override requires game_id, play_id, and epa")
        if replacement[["game_id", "play_id"]].isna().any().any():
            raise ValueError("EPA override play identity cannot be missing")
        if replacement.duplicated(["game_id", "play_id"]).any():
            raise ValueError("EPA override contains duplicate play identities")
        frame = frame.drop(columns="epa").merge(
            replacement[["game_id", "play_id", "epa"]],
            on=["game_id", "play_id"],
            how="left",
            validate="one_to_one",
            indicator=True,
        )
        if frame["_merge"].ne("both").any():
            raise ValueError("EPA override does not cover every raw play identity")
        frame = frame.drop(columns="_merge")
    flags = ["rush_attempt", "qb_dropback", "qb_scramble", "qb_kneel", "qb_spike", "penalty"]
    flags += [column for column in OPTIONAL_COLUMNS if column in frame]
    for column in flags:
        values = pd.to_numeric(frame[column], errors="raise").fillna(0)
        if not values.isin([0, 1]).all():
            raise ValueError(f"PBP {column} must contain binary flags")
        frame[column] = values.astype(bool)
    eligible = (
        (frame["rush_attempt"] | frame["qb_dropback"])
        & frame["play_type"].isin(["run", "pass"])
        & ~frame["qb_kneel"]
        & ~frame["qb_spike"]
        & ~frame["penalty"]
    )
    for column in OPTIONAL_COLUMNS:
        if column in frame:
            eligible &= ~frame[column]
    frame = frame.loc[eligible].copy()
    for column in ["posteam", "defteam"]:
        frame[column] = frame[column].astype("string").str.upper().replace(TEAM_ALIASES)
        if not frame[column].isin(TEAMS).all():
            raise ValueError(f"Eligible PBP has an unknown or missing {column}")
    if frame["posteam"].eq(frame["defteam"]).any():
        raise ValueError("A PBP offense cannot play itself")
    if (frame.groupby(["game_id", "posteam"])["defteam"].nunique() > 1).any():
        raise ValueError("PBP offense has multiple opponents within a game")
    epa = pd.to_numeric(frame["epa"], errors="raise")
    if not np.isfinite(epa.dropna()).all():
        raise ValueError("PBP EPA must be finite when present")
    frame["epa"] = epa
    frame["yards_gained"] = pd.to_numeric(frame["yards_gained"], errors="raise")
    if not np.isfinite(frame["yards_gained"].dropna()).all():
        raise ValueError("PBP yards gained must be finite")
    if frame["yards_gained"].isna().any():
        raise ValueError("Eligible PBP yards gained cannot be missing")
    coverage = frame.groupby(["game_id", "posteam", "defteam"], as_index=False).agg(
        eligible_plays=("play_id", "size"), missing_epa_plays=("epa", lambda x: x.isna().sum())
    )
    frame = frame.loc[frame["epa"].notna()].copy()
    if frame.empty:
        return pd.DataFrame(columns=TEAM_COLUMNS), pd.DataFrame(columns=QB_COLUMNS)
    frame["success"] = frame["epa"] > 0
    frame["explosive"] = np.where(
        frame["qb_dropback"], frame["yards_gained"] >= 20, frame["yards_gained"] >= 10
    )
    keys = ["game_id", "posteam", "defteam"]
    teams = frame.groupby(keys, as_index=False).agg(
        plays=("play_id", "size"),
        epa_sum=("epa", "sum"),
        successes=("success", "sum"),
        explosive_plays=("explosive", "sum"),
    )
    teams = teams.merge(coverage, on=keys, how="left", validate="one_to_one")
    teams = teams.rename(columns={"posteam": "team", "defteam": "opponent"})[TEAM_COLUMNS]
    frame["qb_id"] = frame["passer_player_id"].astype("string")
    missing_qb = frame["qb_id"].isna() | frame["qb_id"].str.strip().eq("")
    frame.loc[missing_qb & frame["qb_scramble"], "qb_id"] = frame.loc[
        missing_qb & frame["qb_scramble"], "rusher_player_id"
    ].astype("string")
    assigned = frame["qb_dropback"] & frame["qb_id"].notna() & frame["qb_id"].str.strip().ne("")
    quarterbacks = (
        frame.loc[assigned]
        .groupby(["game_id", "posteam", "qb_id"], as_index=False)
        .agg(dropbacks=("play_id", "size"), epa_sum=("epa", "sum"), successes=("success", "sum"))
    )
    quarterbacks = quarterbacks.rename(columns={"posteam": "team"})[QB_COLUMNS]
    return teams, quarterbacks
