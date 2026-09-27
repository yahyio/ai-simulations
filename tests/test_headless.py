"""Uçtan uca: PNG pist dosyasından başsız eğitim, kayıt ve devam."""
import pygame
import pytest

from main import run_headless
from racing import storage
from racing.config import deep_merge, validate
from tests.conftest import make_ring_rgb


@pytest.fixture
def headless_config(tmp_path, monkeypatch, config):
    monkeypatch.chdir(tmp_path)
    surface = pygame.surfarray.make_surface(make_ring_rgb().transpose(1, 0, 2))
    pygame.image.save(surface, str(tmp_path / "ring.png"))
    return validate(deep_merge(config, {"track": {"file": "ring.png"}}))


def test_headless_training_saves_model_and_resumes(headless_config):
    assert run_headless(headless_config, episodes=2, resume=False) == 0
    assert storage.MODEL_PATH.exists()
    assert storage.load_stats().episodes == 2

    assert run_headless(headless_config, episodes=1, resume=True) == 0
    assert storage.load_stats().episodes == 3


def test_headless_reports_missing_track(headless_config):
    broken = deep_merge(headless_config, {"track": {"file": "yok.png"}})
    assert run_headless(broken, episodes=1, resume=False) == 1
