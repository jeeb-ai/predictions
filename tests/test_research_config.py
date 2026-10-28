"""Local configuration and secret-file setup, without publishing credentials."""

import copy
import json
import stat
from pathlib import Path

import pytest

from predictions.config import default_config, initialize_research, load_config, validate_config


def test_initialization_is_private_idempotent_and_does_not_print_secret(tmp_path):
    path = tmp_path / "research.local.json"
    first = initialize_research(path)
    token = Path(first["token_file"])
    secret = token.read_text().strip()
    assert len(secret) >= 32
    assert stat.S_IMODE(token.stat().st_mode) == 0o600
    assert secret not in json.dumps(first)
    original = path.read_bytes()
    assert stat.S_IMODE(path.stat().st_mode) == 0o644
    assert initialize_research(path) == first
    assert path.read_bytes() == original
    assert token.read_text().strip() == secret
    assert first["secret_value_printed"] is False


def test_edited_policy_is_preserved_and_changes_config_hash(tmp_path):
    path = tmp_path / "policy.json"
    initialize_research(path)
    old = load_config(path)
    value = json.loads(path.read_text())
    value["paper"]["enabled"] = False
    value["paper"]["assumptions"]["tie_probability_range"] = ["0.005", "0.02"]
    path.write_text(json.dumps(value))
    initialize_research(path)
    current = load_config(path)
    assert not current["paper"]["enabled"]
    assert current["paper"]["assumptions"]["tie_probability_range"] == ["0.005", "0.02"]
    assert current["config_hash"] != old["config_hash"]
    assert Path(current["paths"]["state_dir"]).is_absolute()


@pytest.mark.parametrize(
    "key,value",
    [
        ("interval_seconds", 1),
        ("bootstrap_samples", 0),
        ("workers", True),
        ("lookahead_hours", -1),
        ("confidence_level", float("nan")),
        ("model_version", []),
    ],
)
def test_invalid_worker_settings_fail(tmp_path, key, value):
    config = default_config()
    config[key] = value
    with pytest.raises(ValueError):
        validate_config(config, base_dir=tmp_path)


def test_unknown_policy_and_bad_risk_ranges_fail(tmp_path):
    config = default_config()
    config["paper"]["assumptions"]["tie_probability_range"] = ["0.2", "0.1"]
    with pytest.raises(ValueError, match="lower"):
        validate_config(config, base_dir=tmp_path)
    config = default_config()
    config["paper"]["real_money"] = True
    with pytest.raises(ValueError, match="Unknown"):
        validate_config(config, base_dir=tmp_path)


def test_world_readable_token_and_symlink_are_rejected(tmp_path):
    path = tmp_path / "config.json"
    result = initialize_research(path)
    token = Path(result["token_file"])
    token.chmod(0o644)
    with pytest.raises(ValueError, match="0600"):
        initialize_research(path)
    token.chmod(0o600)
    original = token.with_suffix(".private")
    token.rename(original)
    token.symlink_to(original)
    with pytest.raises(ValueError, match="regular"):
        initialize_research(path)


def test_configuration_defaults_never_invent_outcome_assumptions(tmp_path):
    config = validate_config(copy.deepcopy(default_config()), base_dir=tmp_path)
    assert config["paper"]["enabled"] is True
    assert all(value is None for value in config["paper"]["assumptions"].values())
    assert config["paper"]["max_open_positions_per_game"] == 1
    assert config["bootstrap_samples"] >= 20
