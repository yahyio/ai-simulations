"""Ortak test verileri: sentetik halka pist ve küçük bir yapılandırma."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from racing.config import DEFAULTS, deep_merge, validate
from racing.track import Track

RING_SIZE = 300
RING_CENTER = 150
RING_INNER = 70
RING_OUTER = 120


def make_ring_rgb(start_line: bool = True, closed: bool = True) -> np.ndarray:
    """Beyaz zemin üstünde siyah halka; tepede dikey kırmızı başlangıç çizgisi."""
    yy, xx = np.mgrid[0:RING_SIZE, 0:RING_SIZE]
    r = np.hypot(xx - RING_CENTER, yy - RING_CENTER)
    rgb = np.full((RING_SIZE, RING_SIZE, 3), 255, dtype=np.uint8)
    rgb[(r >= RING_INNER) & (r <= RING_OUTER)] = 0
    if not closed:
        # Alt tarafı kes: halka artık döngü değil.
        rgb[(yy > RING_CENTER + 40) & (np.abs(xx - RING_CENTER) < 12)] = 255
    if start_line:
        top = RING_CENTER - RING_OUTER
        rgb[top:top + (RING_OUTER - RING_INNER) + 1, RING_CENTER - 2:RING_CENTER + 3] = (255, 0, 0)
    return rgb


@pytest.fixture
def config() -> dict:
    return validate(deep_merge(DEFAULTS, {
        "simulation": {"num_cars": 4, "max_steps_per_episode": 400, "seed": 1},
        "car": {"length": 10, "width": 6, "max_speed": 4.0, "sensor_range": 60},
        "learning": {"batch_size": 16, "buffer_size": 5000, "learning_starts": 32, "hidden_size": 32,
                     "device": "cpu"},
    }))


@pytest.fixture
def ring_track(config) -> Track:
    return Track(make_ring_rgb(), config["track"], name="ring")


@pytest.fixture
def defaults() -> dict:
    return copy.deepcopy(DEFAULTS)
