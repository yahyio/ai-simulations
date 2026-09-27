"""Yürüme eğitimi/izleme çekirdeği — arayüzden bağımsız, başsız da çalışır."""
from __future__ import annotations

import csv
import json
import logging
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from enum import Enum
from pathlib import Path

import numpy as np

from .env import ACT_DIM, CONTACT_OBS, CONTROL_DT, MAX_EPISODE_STEPS, OBS_DIM, TASKS, Episode, Task
from .ppo import PPO, PPOConfig, expected_header, read_checkpoint_header
from .vecenv import make_vec_env

log = logging.getLogger(__name__)

DATA_DIR = Path("data") / "walker"
BUNDLED_DIR = Path("models") / "walker"     # depoyla gelen hazır modeller (yalnızca izleme)
LOG_DIR = Path("logs")
STATS_VERSION = 1
HISTORY_LIMIT = 400
EPISODE_TAIL = 100          # "en iyi model" için son kaç denemenin ortalaması
BEST_MIN_EPISODES = 50
AUTOSAVE_EVERY_UPDATES = 10
DISPLAY_LANES = 8           # ekranda kulvarlarda gösterilen karakter sayısı
WATCH_MAX_STEPS = 3 * MAX_EPISODE_STEPS
SPEED_SMOOTHING = 0.2
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


class Mode(Enum):
    TRAIN = "train"
    WATCH = "watch"


def paths(task: str) -> dict[str, Path]:
    base = DATA_DIR / task
    return {
        "model": base / "model.pt",
        "best": base / "best_model.pt",
        "stats": base / "stats.json",
        "log": LOG_DIR / f"walker_{task}.csv",
        "bundled": BUNDLED_DIR / f"{task}.pt",
    }


class IncompatibleModelError(ValueError):
    """Kayıtlı model bu sürümle uyumsuz (ör. eski 2B sürümden kalma)."""


def checkpoint_compatible(path: Path, cfg: PPOConfig = PPOConfig()) -> bool:
    """Model dosyası bu ağ ve gözlem boyutlarıyla yüklenebilir mi?"""
    return path.exists() and read_checkpoint_header(path) == expected_header(OBS_DIM, ACT_DIM, cfg)


def has_model(task: str, cfg: PPOConfig = PPOConfig()) -> bool:
    """Bu bilgisayarda eğitimi sürdürülebilecek bir model var mı?"""
    return checkpoint_compatible(paths(task)["model"], cfg)


def has_legacy_model(task: str, cfg: PPOConfig = PPOConfig()) -> bool:
    """Yerel model dosyası var ama bu sürümle kullanılamıyor mu?"""
    path = paths(task)["model"]
    return path.exists() and not checkpoint_compatible(path, cfg)


def watch_source(task: str, cfg: PPOConfig = PPOConfig()) -> Path | None:
    """İzlemek için en iyi uyumlu model: yerel en iyi → yerel son → depoyla gelen hazır model."""
    p = paths(task)
    return next((path for path in (p["best"], p["model"], p["bundled"]) if checkpoint_compatible(path, cfg)), None)


def reset_task(task: str) -> list[Path]:
    removed = []
    for key in ("model", "best", "stats", "log"):
        path = paths(task)[key]
        if path.exists():
            path.unlink()
            removed.append(path)
    return removed


@dataclass
class WalkStats:
    task: str = ""
    episodes: int = 0
    best_distance: float = 0.0
    best_distance_episode: int | None = None
    best_speed: float = 0.0
    best_mean_distance: float = 0.0
    recent_distances: list[float] = field(default_factory=list)
    history_mean_distance: list[float] = field(default_factory=list)
    history_best_distance: list[float] = field(default_factory=list)
    history_mean_seconds: list[float] = field(default_factory=list)

    def record(self, ep: Episode) -> bool:
        """Denemeyi işler; yeni mesafe rekoruysa True."""
        self.episodes += 1
        self.recent_distances = (self.recent_distances + [ep.distance])[-EPISODE_TAIL:]
        if ep.seconds >= 3.0:  # kısa bir sıçramayı "hız rekoru" sayma
            self.best_speed = max(self.best_speed, ep.speed)
        if ep.distance > self.best_distance:
            self.best_distance = ep.distance
            self.best_distance_episode = self.episodes
            return True
        return False

    def close_update(self, episodes: list[Episode]) -> None:
        if not episodes:
            return
        mean_distance = float(np.mean([e.distance for e in episodes]))
        mean_seconds = float(np.mean([e.seconds for e in episodes]))
        self.history_mean_distance = (self.history_mean_distance + [mean_distance])[-HISTORY_LIMIT:]
        self.history_best_distance = (self.history_best_distance + [max(e.distance for e in episodes)])[-HISTORY_LIMIT:]
        self.history_mean_seconds = (self.history_mean_seconds + [mean_seconds])[-HISTORY_LIMIT:]

    @property
    def recent_mean_distance(self) -> float:
        return float(np.mean(self.recent_distances)) if self.recent_distances else 0.0

    def to_dict(self) -> dict:
        return {"version": STATS_VERSION, **asdict(self)}

    @classmethod
    def from_dict(cls, data: dict) -> WalkStats:
        if data.get("version") != STATS_VERSION:
            raise ValueError("tanınmayan istatistik biçimi")
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})


def load_stats(task: str) -> WalkStats:
    path = paths(task)["stats"]
    if not path.exists():
        return WalkStats(task=task)
    try:
        with path.open("r", encoding="utf-8") as fh:
            return WalkStats.from_dict(json.load(fh))
    except (OSError, ValueError, TypeError) as exc:
        log.warning("%s okunamadı (%s); istatistikler sıfırdan başlıyor.", path, exc)
        return WalkStats(task=task)


def save_stats(stats: WalkStats) -> None:
    path = paths(stats.task)["stats"]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(stats.to_dict(), fh, indent=2)
    tmp.replace(path)


def append_log(task: str, episode_number: int, ep: Episode) -> None:
    path = paths(task)["log"]
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if is_new:
            writer.writerow(["timestamp", "episode", "distance_m", "seconds", "speed_mps", "reward", "fell"])
        writer.writerow([datetime.now().strftime(TIMESTAMP_FORMAT), episode_number, f"{ep.distance:.3f}",
                         f"{ep.seconds:.2f}", f"{ep.speed:.3f}", f"{ep.reward:.2f}", int(ep.fell)])


@dataclass(frozen=True)
class Tick:
    finished: tuple[Episode, ...] = ()
    new_record: bool = False
    updating: bool = False
    update_done: bool = False


@dataclass(frozen=True)
class RunnerInfo:
    distance: float
    speed: float
    seconds: float
    contacts: tuple[bool, bool]


class WalkTrainer:
    def __init__(self, task_key: str, mode: Mode = Mode.TRAIN, resume: bool = False,
                 cfg: PPOConfig = PPOConfig(), seed: int | None = None, workers: int | None = None):
        if task_key not in TASKS:
            raise ValueError(f"Bilinmeyen görev: {task_key!r}")
        self.task: Task = TASKS[task_key]
        self.mode = mode
        self.training = mode is Mode.TRAIN
        self.paths = paths(task_key)
        self.model_source: Path | None = None
        if self.training:
            self.env = make_vec_env(cfg.num_envs, task_key, seed=seed, workers=workers)
        else:
            cfg = replace(cfg, num_envs=DISPLAY_LANES)
            self.env = make_vec_env(DISPLAY_LANES, task_key, seed=seed, workers=0, max_steps=WATCH_MAX_STEPS)
        try:
            self.ppo = PPO(self.env.obs_dim, self.env.act_dim, cfg, seed=seed)
            self.stats = load_stats(task_key) if (resume or not self.training) else WalkStats(task=task_key)
            if self.training and resume and self.paths["model"].exists():
                if not checkpoint_compatible(self.paths["model"], cfg):
                    raise IncompatibleModelError(
                        f"{self.paths['model']} bu sürümle uyumsuz (büyük olasılıkla eski 2B sürümden kalma).")
                self.ppo.load(self.paths["model"])
            elif not self.training:
                self.model_source = watch_source(task_key, cfg)
                if self.model_source is None:
                    raise FileNotFoundError(f"İzlenecek {self.task.label.lower()} modeli yok. Önce eğitin.")
                self.ppo.load(self.model_source)
        except Exception:
            self.env.close()
            raise

        self.obs = self.env.reset()
        self.qpos = self.env.qpos
        self.steps = self.env.steps
        self.speeds = np.zeros(self.env.num_envs)
        self._update: Iterator[None] | None = None
        self._since_update: list[Episode] = []
        self.watch_episodes: list[Episode] = []

    # ------------------------------------------------------------------ #
    @property
    def updating(self) -> bool:
        return self._update is not None

    @property
    def display_count(self) -> int:
        return min(self.env.num_envs, DISPLAY_LANES)

    def tick(self) -> Tick:
        """Bir iş birimi: öğrenme sürüyorsa bir mini-parti, değilse bir ortam adımı."""
        if self._update is not None:
            try:
                next(self._update)
                return Tick(updating=True)
            except StopIteration:
                self._update = None
                self._finish_update()
                return Tick(update_done=True)

        actions, norm, logp, values = self.ppo.act(self.obs, deterministic=not self.training, learn=self.training)
        prev_x = self.qpos[:, 0].copy()
        batch = self.env.step(actions)
        if self.training and self.ppo.record(norm, actions, logp, values, batch.rewards,
                                             batch.terminated, batch.truncated, batch.final_obs):
            self._update = self.ppo.update(batch.obs)
        self.obs, self.qpos, self.steps = batch.obs, batch.qpos, batch.steps
        speed = np.where(self.steps > 0, (self.qpos[:, 0] - prev_x) / CONTROL_DT, 0.0)
        self.speeds = (1 - SPEED_SMOOTHING) * self.speeds + SPEED_SMOOTHING * speed
        return Tick(finished=batch.finished, new_record=self._record(batch.finished))

    def _record(self, finished: tuple[Episode, ...]) -> bool:
        if not self.training:
            self.watch_episodes.extend(finished)
            return False
        new_record = False
        for ep in finished:
            new_record |= self.stats.record(ep)
            append_log(self.task.key, self.stats.episodes, ep)
        self._since_update.extend(finished)
        return new_record

    def _finish_update(self) -> None:
        self.stats.close_update(self._since_update)
        self._since_update = []
        tail = self.stats.recent_mean_distance
        if len(self.stats.recent_distances) >= BEST_MIN_EPISODES and tail > self.stats.best_mean_distance:
            self.stats.best_mean_distance = tail
            self.ppo.save(self.paths["best"])
        if self.ppo.updates % AUTOSAVE_EVERY_UPDATES == 0:
            self.save()

    # ------------------------------------------------------------------ #
    def leader(self) -> int:
        """Ekrandaki kulvarlarda en önde giden."""
        return int(np.argmax(self.qpos[: self.display_count, 0]))

    def runner(self, i: int) -> RunnerInfo:
        contacts = self.obs[i, CONTACT_OBS] > 0.5
        return RunnerInfo(float(self.qpos[i, 0]), float(self.speeds[i]), float(self.steps[i] * CONTROL_DT),
                          (bool(contacts[1]), bool(contacts[0])))   # (sol, sağ)

    def save(self) -> None:
        if self.training:
            self.ppo.save(self.paths["model"])
            save_stats(self.stats)

    def close(self) -> None:
        self.env.close()
