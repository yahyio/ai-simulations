import pytest

from racing import storage
from racing.trainer import Mode, Trainer


@pytest.fixture(autouse=True)
def in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def run_until_episode(trainer: Trainer, limit: int = 2000):
    for _ in range(limit):
        events = trainer.step()
        if events.episode is not None:
            return events
    raise AssertionError("bölüm tamamlanmadı")


def test_training_episode_updates_stats_and_log(ring_track, config):
    trainer = Trainer(config, ring_track)
    events = run_until_episode(trainer)
    assert events.episode.episode == 1
    assert trainer.stats.episodes == 1
    assert storage.EPISODE_LOG_PATH.exists()
    assert trainer.env.running.all()  # yeni bölüm başladı


def test_epsilons_are_spread_across_cars(ring_track, config):
    trainer = Trainer(config, ring_track)
    eps = trainer.epsilons()
    assert eps.shape == (config["simulation"]["num_cars"],)
    assert eps[0] < eps[-1] == pytest.approx(trainer.agent.epsilon)


def test_save_then_resume_restores_progress(ring_track, config):
    trainer = Trainer(config, ring_track)
    run_until_episode(trainer)
    trainer.save()
    assert storage.MODEL_PATH.exists() and storage.STATS_PATH.exists()

    resumed = Trainer(config, ring_track, resume=True)
    assert resumed.episode == 1
    assert resumed.agent.env_steps == trainer.agent.env_steps


def test_resume_on_other_track_keeps_model_but_resets_records(ring_track, config):
    trainer = Trainer(config, ring_track)
    run_until_episode(trainer)
    trainer.stats.best_lap_time = 5.0
    trainer.save()

    ring_track.name = "baska_pist"
    resumed = Trainer(config, ring_track, resume=True)
    assert resumed.agent.env_steps == trainer.agent.env_steps
    assert resumed.stats.best_lap_time is None
    assert resumed.stats.episodes == 1 and resumed.stats.track == "baska_pist"


def test_watch_without_model_raises(ring_track, config):
    with pytest.raises(FileNotFoundError):
        Trainer(config, ring_track, mode=Mode.WATCH)


def test_watch_mode_is_greedy_single_car(ring_track, config):
    Trainer(config, ring_track).save()
    watcher = Trainer(config, ring_track, mode=Mode.WATCH)
    assert watcher.env.num_cars == 1
    assert watcher.epsilons().tolist() == [0.0]
    events = run_until_episode(watcher)
    assert watcher.stats.episodes == 0  # izleme istatistikleri değiştirmez
    assert events.episode.epsilon == 0.0


def test_restart_episode_resets_env(ring_track, config):
    trainer = Trainer(config, ring_track)
    for _ in range(5):
        trainer.step()
    trainer.restart_episode()
    assert trainer.env.steps == 0
