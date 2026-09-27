import pytest
import yaml

from racing.config import DEFAULTS, ConfigError, deep_merge, load_config, save_config, validate


def test_defaults_are_valid():
    assert validate(DEFAULTS) == DEFAULTS


def test_deep_merge_does_not_mutate_inputs():
    base = {"a": {"b": 1, "c": 2}}
    merged = deep_merge(base, {"a": {"b": 5}})
    assert merged == {"a": {"b": 5, "c": 2}}
    assert base == {"a": {"b": 1, "c": 2}}


def test_values_are_clamped_and_coerced():
    cfg = validate(deep_merge(DEFAULTS, {"simulation": {"num_cars": 999}, "car": {"max_speed": "3"}}))
    assert cfg["simulation"]["num_cars"] == 64
    assert cfg["car"]["max_speed"] == 3.0


def test_unknown_keys_are_dropped():
    cfg = validate(deep_merge(DEFAULTS, {"car": {"eski_alan": 1}, "yabanci": {}}))
    assert "eski_alan" not in cfg["car"]
    assert "yabanci" not in cfg


def test_invalid_type_raises():
    with pytest.raises(ConfigError, match=r"car\.max_speed"):
        validate(deep_merge(DEFAULTS, {"car": {"max_speed": "hızlı"}}))


@pytest.mark.parametrize("seed", ["abc", [1, 2], True])
def test_invalid_seed_raises(seed):
    with pytest.raises(ConfigError, match="seed"):
        validate(deep_merge(DEFAULTS, {"simulation": {"seed": seed}}))


def test_epsilon_end_never_exceeds_start():
    cfg = validate(deep_merge(DEFAULTS, {"learning": {"epsilon_start": 0.3, "epsilon_end": 0.9}}))
    assert cfg["learning"]["epsilon_end"] == 0.3


def test_load_creates_missing_file(tmp_path):
    path = tmp_path / "config.yaml"
    assert load_config(path) == DEFAULTS
    assert path.exists()


def test_save_then_load_roundtrip(tmp_path):
    path = tmp_path / "config.yaml"
    cfg = deep_merge(DEFAULTS, {"car": {"max_speed": 9.5}})
    save_config(cfg, path)
    assert load_config(path)["car"]["max_speed"] == 9.5


def test_load_rejects_non_mapping(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump([1, 2, 3]), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


def test_load_rejects_broken_yaml(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("car: [unterminated", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)
