import csv
import json

import pytest

from racing import storage
from racing.storage import EpisodeSummary, TrainingStats, append_episode_log, load_stats, reset_all, save_stats


def summary(episode: int, laps: int = 0, lap_time: float | None = None, progress: float = 0.5) -> EpisodeSummary:
    return EpisodeSummary(episode, laps, lap_time, progress, 1.0, 0.5, 0.1, 100)


@pytest.fixture(autouse=True)
def in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def test_record_tracks_best_lap_and_first_lap():
    stats = TrainingStats()
    assert stats.record(summary(1)) is False
    assert stats.record(summary(2, laps=3, lap_time=9.0)) is True
    assert stats.record(summary(3, laps=1, lap_time=9.5)) is False
    assert stats.record(summary(4, laps=2, lap_time=8.0)) is True
    assert (stats.episodes, stats.total_laps, stats.best_lap_time) == (4, 6, 8.0)
    assert stats.first_lap_episode == 2 and stats.best_lap_episode == 4
    assert stats.history_lap_time == [None, 9.0, 9.5, 8.0]


def test_history_is_bounded():
    stats = TrainingStats()
    for ep in range(storage.HISTORY_LIMIT + 50):
        stats.record(summary(ep + 1))
    assert len(stats.history_progress) == storage.HISTORY_LIMIT


def test_recent_lap_rate():
    stats = TrainingStats()
    for ep, laps in enumerate([0, 1, 0, 2], start=1):
        stats.record(summary(ep, laps=laps, lap_time=10.0 if laps else None))
    assert stats.recent_lap_rate(window=4) == 0.5


def test_save_and_load_roundtrip():
    stats = TrainingStats()
    stats.record(summary(1, laps=1, lap_time=7.5))
    save_stats(stats)
    assert load_stats() == stats


def test_load_ignores_legacy_or_corrupt_files():
    storage.STATS_PATH.parent.mkdir(parents=True)
    storage.STATS_PATH.write_text(json.dumps({"total_episodes": 3092, "best_lap_time": None}))
    assert load_stats() == TrainingStats()
    storage.STATS_PATH.write_text("{bozuk")
    assert load_stats() == TrainingStats()


def test_episode_log_writes_header_once():
    append_episode_log(summary(1))
    append_episode_log(summary(2, laps=1, lap_time=6.25))
    with storage.EPISODE_LOG_PATH.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["episode"] for r in rows] == ["1", "2"]
    assert rows[0]["best_lap_time"] == "" and rows[1]["best_lap_time"] == "6.250"


def test_reset_all_removes_only_existing_files():
    save_stats(TrainingStats())
    append_episode_log(summary(1))
    removed = reset_all()
    assert set(removed) == {storage.STATS_PATH, storage.EPISODE_LOG_PATH}
    assert reset_all() == []
