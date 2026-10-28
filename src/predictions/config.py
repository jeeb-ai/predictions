"""Validated, secret-free configuration for a self-hosted research worker."""

from __future__ import annotations

import hashlib
import json
import math
import os
import secrets
import stat
from decimal import Decimal, InvalidOperation
from pathlib import Path


def default_config() -> dict:
    return {
        "schema_version": 1,
        "interval_seconds": 900,
        "lookahead_hours": 168,
        "model_version": "v2",
        "bootstrap_samples": 100,
        "confidence_level": 0.95,
        "bootstrap_seed": 42,
        "workers": 4,
        "paths": {
            "state_dir": "data/research",
            "data_dir": "data",
            "ledger": "data/paper/research.sqlite",
            "performance_db": "data/research/performance.sqlite",
            "model_dir": "models/research",
            "reports_dir": "reports/research",
            "schedule_cache": "data/raw",
            "pbp_dir": "data/pbp",
            "ep_dir": "data/causal_ep",
            "market_cache": "data/polymarket_us",
        },
        "paper": {
            "enabled": True,
            "quantity": 10,
            "max_open_exposure": "100",
            "max_open_positions_per_game": 1,
            "min_ev_per_contract": "0.02",
            "max_quote_age_seconds": 60,
            "max_forecast_age_seconds": 3600,
            "assumptions": {
                "tie_probability_range": None,
                "completion_probability_range": None,
                "cancellation_settlement_price_range": None,
            },
        },
        "dashboard": {
            "token_file": "data/private/dashboard.token",
            "max_snapshot_age_seconds": 1800,
        },
    }


def _integer(value, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer between {minimum} and {maximum}")
    return value


def _decimal(value, name: str, *, positive: bool = False) -> str:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{name} must be a finite decimal")
    try:
        number = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"{name} must be a finite decimal") from exc
    if not number.is_finite() or number < 0 or (positive and number == 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'} and finite")
    return str(number)


def _range(value, name: str):
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must be null or a two-element probability range")
    bounds = [Decimal(_decimal(item, name)) for item in value]
    if not 0 <= bounds[0] <= bounds[1] <= 1:
        raise ValueError(f"{name} must satisfy 0 <= lower <= upper <= 1")
    return [str(item) for item in bounds]


def _merge(default: dict, supplied: dict, location: str = "config") -> dict:
    if not isinstance(supplied, dict):
        raise ValueError(f"{location} must be an object")
    if supplied.keys() - default.keys():
        raise ValueError(f"Unknown {location} fields: {sorted(supplied.keys() - default.keys())}")
    result = {}
    for key, value in default.items():
        candidate = supplied.get(key, value)
        result[key] = (
            _merge(value, candidate, f"{location}.{key}") if isinstance(value, dict) else candidate
        )
    return result


def validate_config(value: dict, *, base_dir: Path) -> dict:
    result = _merge(default_config(), value)
    if type(result["schema_version"]) is not int or result["schema_version"] != 1:
        raise ValueError("Unsupported research configuration schema")
    for key, minimum, maximum in [
        ("interval_seconds", 60, 86400),
        ("lookahead_hours", 1, 336),
        ("bootstrap_samples", 20, 2000),
        ("bootstrap_seed", 0, 2**32 - 1),
        ("workers", 1, 8),
    ]:
        result[key] = _integer(result[key], key, minimum, maximum)
    if not isinstance(result["model_version"], str) or result["model_version"] not in {"v1", "v2"}:
        raise ValueError("model_version must be v1 or v2")
    confidence = result["confidence_level"]
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(confidence)
        or not 0.5 < confidence < 1
    ):
        raise ValueError("confidence_level must be between 0.5 and 1")
    for key, path in result["paths"].items():
        if not isinstance(path, str) or not path.strip():
            raise ValueError(f"paths.{key} must be a nonempty path")
        result["paths"][key] = str((base_dir / path).resolve())
    paper = result["paper"]
    if type(paper["enabled"]) is not bool:
        raise ValueError("paper.enabled must be a boolean")
    for key, minimum, maximum in [
        ("quantity", 1, 1000000),
        ("max_open_positions_per_game", 1, 10),
        ("max_quote_age_seconds", 1, 3600),
        ("max_forecast_age_seconds", 1, 86400),
    ]:
        paper[key] = _integer(paper[key], f"paper.{key}", minimum, maximum)
    paper["max_open_exposure"] = _decimal(
        paper["max_open_exposure"], "max_open_exposure", positive=True
    )
    paper["min_ev_per_contract"] = _decimal(paper["min_ev_per_contract"], "min_ev_per_contract")
    paper["assumptions"] = {
        key: _range(bounds, key) for key, bounds in paper["assumptions"].items()
    }
    dashboard = result["dashboard"]
    token = dashboard["token_file"]
    if not isinstance(token, str) or not token.strip():
        raise ValueError("dashboard.token_file must be a nonempty path")
    dashboard["token_file"] = os.path.abspath(base_dir / token)
    dashboard["max_snapshot_age_seconds"] = _integer(
        dashboard["max_snapshot_age_seconds"], "max_snapshot_age_seconds", 60, 172800
    )
    # Include normalized policy and paths, never a token or secret value.
    result["config_hash"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    return result


def load_config(path: Path | str) -> dict:
    path = Path(path).resolve()
    return validate_config(json.loads(path.read_text()), base_dir=path.parent)


def initialize_research(path: Path | str) -> dict:
    path = Path(path).resolve()
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as handle:
            handle.write(json.dumps(default_config(), indent=2) + "\n")
        # The nonroot container must read this policy file. Secret values live
        # only in the separate mode-0600 token, never in configuration JSON.
        path.chmod(0o644)
    config = load_config(path)
    for key, value in config["paths"].items():
        (Path(value).parent if key in {"ledger", "performance_db"} else Path(value)).mkdir(
            parents=True, exist_ok=True
        )
    token = Path(config["dashboard"]["token_file"])
    token.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not token.exists():
        descriptor = os.open(token, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as handle:
            handle.write(secrets.token_urlsafe(32) + "\n")
    properties = token.lstat()
    if not stat.S_ISREG(properties.st_mode) or properties.st_mode & 0o077:
        raise ValueError("Dashboard token must be a regular private file with mode 0600")
    if len(token.read_text().strip()) < 32:
        raise ValueError("Dashboard token must contain at least 32 characters")
    return {
        "config_path": str(path),
        "config_hash": config["config_hash"],
        "state_dir": config["paths"]["state_dir"],
        "token_file": str(token),
        "mode": "paper_only",
        "secret_value_printed": False,
    }
