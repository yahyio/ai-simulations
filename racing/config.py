"""Yapılandırma: varsayılan değerler, yükleme, doğrulama ve kaydetme."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

CONFIG_PATH = Path("config.yaml")

DEFAULTS: dict[str, dict[str, Any]] = {
    "display": {
        "width": 1440,
        "height": 810,
        "fps": 60,
        "show_sensors": False,
        "show_trail": True,
    },
    "simulation": {
        "num_cars": 16,
        "steps_per_frame": 4,
        "max_steps_per_episode": 3000,
        "idle_timeout_steps": 120,
        "sim_fps": 60,
        "autosave_every": 10,
        "seed": None,
    },
    "car": {
        "length": 26,
        "width": 12,
        "max_speed": 7.0,
        "acceleration": 0.18,
        "brake": 0.35,
        "drag": 0.012,
        "turn_rate_deg": 4.5,
        "understeer": 0.35,
        "sensor_count": 7,
        "sensor_range": 220,
        "sensor_fov_deg": 180,
    },
    "learning": {
        "device": "auto",
        "learning_rate": 0.0003,
        "discount_factor": 0.99,
        "batch_size": 128,
        "buffer_size": 200000,
        "n_step": 3,
        "hidden_size": 256,
        "target_tau": 0.005,
        "learning_starts": 2000,
        "train_every": 2,
        "gradient_steps": 1,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay_steps": 100000,
    },
    "rewards": {
        "progress": 1.0,
        "time_penalty": -0.01,
        "crash": -5.0,
        "lap_complete": 20.0,
        "wrong_way": -5.0,
        "idle": -2.0,
    },
    "track": {
        "file": "track.png",
        "reverse": False,
        "track_color": [0, 0, 0],
        "start_line_color": [255, 0, 0],
        "color_tolerance": 100,
    },
}

# (min, max) sınırları — sayısal alanlar bu aralığa kırpılır.
RANGES: dict[tuple[str, str], tuple[float, float]] = {
    ("display", "width"): (640, 3840),
    ("display", "height"): (480, 2160),
    ("display", "fps"): (10, 240),
    ("simulation", "num_cars"): (1, 64),
    ("simulation", "steps_per_frame"): (1, 64),
    ("simulation", "max_steps_per_episode"): (100, 100000),
    ("simulation", "idle_timeout_steps"): (10, 10000),
    ("simulation", "sim_fps"): (10, 240),
    ("simulation", "autosave_every"): (1, 10000),
    ("car", "length"): (4, 200),
    ("car", "width"): (2, 100),
    ("car", "max_speed"): (0.5, 50.0),
    ("car", "acceleration"): (0.01, 5.0),
    ("car", "brake"): (0.01, 10.0),
    ("car", "drag"): (0.0, 0.5),
    ("car", "turn_rate_deg"): (0.5, 30.0),
    ("car", "understeer"): (0.0, 0.9),
    ("car", "sensor_count"): (3, 31),
    ("car", "sensor_range"): (20, 2000),
    ("car", "sensor_fov_deg"): (30, 360),
    ("learning", "learning_rate"): (1e-6, 0.1),
    ("learning", "discount_factor"): (0.5, 0.9999),
    ("learning", "batch_size"): (8, 4096),
    ("learning", "buffer_size"): (1000, 5_000_000),
    ("learning", "n_step"): (1, 20),
    ("learning", "hidden_size"): (16, 2048),
    ("learning", "target_tau"): (0.0001, 1.0),
    ("learning", "learning_starts"): (0, 1_000_000),
    ("learning", "train_every"): (1, 64),
    ("learning", "gradient_steps"): (1, 32),
    ("learning", "epsilon_start"): (0.0, 1.0),
    ("learning", "epsilon_end"): (0.0, 1.0),
    ("learning", "epsilon_decay_steps"): (1, 10_000_000),
    ("track", "color_tolerance"): (1, 441),
}


class ConfigError(ValueError):
    """Yapılandırma dosyası okunamadığında veya geçersiz olduğunda."""


def deep_merge(base: dict, override: dict) -> dict:
    """`override` değerlerini `base` üzerine yazan yeni bir sözlük döndürür."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _coerce(value: Any, default: Any, where: str) -> Any:
    """Değeri varsayılanın tipine dönüştürür; olmazsa ConfigError fırlatır."""
    if default is None:
        return value
    try:
        if isinstance(default, bool):
            if isinstance(value, str):
                return value.strip().lower() in ("1", "true", "yes", "evet", "on")
            return bool(value)
        if isinstance(default, int):
            return int(round(float(value)))
        if isinstance(default, float):
            return float(value)
        if isinstance(default, list):
            if not isinstance(value, (list, tuple)) or len(value) != len(default):
                raise TypeError
            return [int(v) for v in value]
        if isinstance(default, str):
            return str(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"'{where}' değeri geçersiz: {value!r}") from exc
    return value


def validate(config: dict) -> dict:
    """Bilinmeyen anahtarları atar, tipleri düzeltir, sınırları uygular."""
    clean: dict[str, dict[str, Any]] = {}
    for section, defaults in DEFAULTS.items():
        given = config.get(section) or {}
        if not isinstance(given, dict):
            raise ConfigError(f"'{section}' bölümü bir sözlük olmalı")
        clean[section] = {}
        for key, default in defaults.items():
            value = _coerce(given.get(key, default), default, f"{section}.{key}")
            bounds = RANGES.get((section, key))
            if bounds is not None:
                value = type(value)(min(max(value, bounds[0]), bounds[1]))
            clean[section][key] = value

    seed = clean["simulation"]["seed"]
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, (int, float, str)):
            raise ConfigError(f"'simulation.seed' bir tam sayı ya da null olmalı: {seed!r}")
        try:
            clean["simulation"]["seed"] = int(seed)
        except ValueError as exc:
            raise ConfigError(f"'simulation.seed' bir tam sayı ya da null olmalı: {seed!r}") from exc

    learning = clean["learning"]
    learning["epsilon_end"] = min(learning["epsilon_end"], learning["epsilon_start"])
    if learning["device"] not in ("auto", "cpu", "cuda"):
        learning["device"] = "auto"
    return clean


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Yapılandırmayı yükler; dosya yoksa varsayılanlarla oluşturur."""
    if not path.exists():
        config = copy.deepcopy(DEFAULTS)
        save_config(config, path)
        return config
    try:
        with path.open("r", encoding="utf-8") as fh:
            user = yaml.safe_load(fh) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path} okunamadı: {exc}") from exc
    if not isinstance(user, dict):
        raise ConfigError(f"{path} geçerli bir YAML sözlüğü değil")
    return validate(deep_merge(DEFAULTS, user))


def save_config(config: dict, path: Path = CONFIG_PATH) -> None:
    """Yapılandırmayı YAML olarak (bölüm sırası korunarak) kaydeder."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(validate(config), fh, sort_keys=False, allow_unicode=True)
    tmp.replace(path)
