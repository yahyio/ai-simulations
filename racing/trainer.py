"""Eğitim / izleme döngüsünün çekirdeği — arayüzden bağımsızdır, başsız da çalışır."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .agent import DQNAgent
from .env import NUM_ACTIONS, RaceEnv, Status
from .storage import (
    BEST_MODEL_PATH,
    MODEL_PATH,
    EpisodeSummary,
    TrainingStats,
    append_episode_log,
    load_stats,
    save_stats,
)
from .track import Track

# Arabalar farklı keşif oranlarıyla sürer: ilk araba neredeyse açgözlü (öğrendiğini
# gösterir), sonuncusu tam epsilon ile keşfeder. Bu çeşitlilik öğrenmeyi hızlandırır.
EPSILON_SPREAD_MIN = 0.05


class Mode(Enum):
    TRAIN = "train"
    WATCH = "watch"


@dataclass(frozen=True)
class StepEvents:
    episode: EpisodeSummary | None = None
    new_record: bool = False
    laps_this_step: int = 0


class Trainer:
    """Ortamı, ajanı ve istatistikleri bir araya getirir."""

    def __init__(self, config: dict, track: Track, mode: Mode = Mode.TRAIN, resume: bool = False):
        self.config = config
        self.mode = mode
        self.training = mode is Mode.TRAIN
        seed = config["simulation"]["seed"]
        num_cars = config["simulation"]["num_cars"] if self.training else 1
        self.autosave_every = config["simulation"]["autosave_every"]

        self.env = RaceEnv(track, config, num_cars, np.random.default_rng(seed))
        self.agent = DQNAgent(self.env.obs_dim, NUM_ACTIONS, config["learning"], num_cars, seed)
        stats = load_stats() if (resume or not self.training) else TrainingStats()
        if self.training and stats.track and stats.track != track.name:
            # Pist değişti: model kalır, tur rekorları ve grafikler yeni piste göre sıfırlanır.
            stats = TrainingStats(episodes=stats.episodes)
        stats.track = track.name
        self.stats = stats

        if self.training and resume and MODEL_PATH.exists():
            self.agent.load(MODEL_PATH)
        elif not self.training:
            path = BEST_MODEL_PATH if BEST_MODEL_PATH.exists() else MODEL_PATH
            if not path.exists():
                raise FileNotFoundError("İzlenecek kayıtlı model yok. Önce eğitim başlatın.")
            self.agent.load(path)

        self.eps_scale = np.linspace(EPSILON_SPREAD_MIN, 1.0, num_cars) if num_cars > 1 else np.ones(1)
        self.episode = self.stats.episodes
        self.watch_laps: list[float] = []
        self.obs = self.env.reset()

    # ------------------------------------------------------------------ #
    def epsilons(self) -> np.ndarray:
        if not self.training:
            return np.zeros(self.env.num_cars)
        return self.agent.epsilon * self.eps_scale

    def step(self) -> StepEvents:
        """Simülasyonu bir adım ilerletir (gerekirse öğrenir ve bölümü kapatır)."""
        actions = self.agent.act(self.obs, self.epsilons())
        result = self.env.step(actions)
        if self.training:
            self.agent.observe(self.obs, actions, result)
            self.agent.train()
        self.obs = result.obs

        laps = int(result.laps.sum())
        if not self.env.episode_over:
            return StepEvents(laps_this_step=laps)
        summary, new_record = self._finish_episode()
        self.obs = self.env.reset()
        return StepEvents(episode=summary, new_record=new_record, laps_this_step=laps)

    def _summarize(self, episode: int) -> EpisodeSummary:
        env = self.env
        lapped = env.status == Status.LAP
        return EpisodeSummary(
            episode=episode,
            laps=int(lapped.sum()),
            best_lap_time=float(np.nanmin(env.lap_times)) if lapped.any() else None,
            best_progress=float(env.progress_fraction().max()),
            mean_reward=float(env.episode_rewards.mean()),
            epsilon=float(self.agent.epsilon) if self.training else 0.0,
            loss=None if np.isnan(self.agent.loss) else float(self.agent.loss),
            sim_steps=env.steps,
        )

    def _finish_episode(self) -> tuple[EpisodeSummary, bool]:
        self.agent.end_episode()
        if not self.training:
            summary = self._summarize(len(self.watch_laps) + 1)
            if summary.best_lap_time is not None:
                self.watch_laps.append(summary.best_lap_time)
            return summary, False

        self.episode += 1
        summary = self._summarize(self.episode)
        new_record = self.stats.record(summary)
        append_episode_log(summary)
        if new_record:
            self.agent.save(BEST_MODEL_PATH)
        if self.episode % self.autosave_every == 0:
            self.save()
        return summary, new_record

    def save(self) -> None:
        if self.training:
            self.agent.save(MODEL_PATH)
            save_stats(self.stats)

    def restart_episode(self) -> None:
        """Bölümü kaydetmeden baştan başlatır."""
        self.agent.end_episode()
        self.obs = self.env.reset()
