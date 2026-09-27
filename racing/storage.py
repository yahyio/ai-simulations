"""Kalıcı veriler: eğitim istatistikleri (JSON), bölüm günlüğü (CSV) ve dosya yolları."""
from __future__ import annotations

import csv
import json
import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

DATA_DIR = Path("data")
LOG_DIR = Path("logs")
MODEL_PATH = DATA_DIR / "model.pt"
BEST_MODEL_PATH = DATA_DIR / "best_model.pt"
STATS_PATH = DATA_DIR / "training_stats.json"
EPISODE_LOG_PATH = LOG_DIR / "episodes.csv"

STATS_VERSION = 2
HISTORY_LIMIT = 500
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_FIELDS = [
    "timestamp", "episode", "laps", "best_lap_time", "best_progress",
    "mean_reward", "epsilon", "loss", "sim_steps",
]


@dataclass(frozen=True)
class EpisodeSummary:
    """Bir bölümün (tüm arabaların bitirdiği bir yarışın) özeti."""
    episode: int
    laps: int
    best_lap_time: float | None
    best_progress: float
    mean_reward: float
    epsilon: float
    loss: float | None
    sim_steps: int


@dataclass
class TrainingStats:
    track: str = ""                 # rekorların ait olduğu pist
    episodes: int = 0
    total_laps: int = 0
    best_lap_time: float | None = None
    best_lap_episode: int | None = None
    first_lap_episode: int | None = None
    best_progress: float = 0.0
    history_progress: list[float] = field(default_factory=list)
    history_laps: list[int] = field(default_factory=list)
    history_lap_time: list[float | None] = field(default_factory=list)

    def record(self, summary: EpisodeSummary) -> bool:
        """Özeti istatistiklere işler. Yeni tur rekoru kırıldıysa True döner."""
        self.episodes = summary.episode
        self.total_laps += summary.laps
        self.best_progress = max(self.best_progress, summary.best_progress)
        if summary.laps and self.first_lap_episode is None:
            self.first_lap_episode = summary.episode

        new_record = summary.best_lap_time is not None and (
            self.best_lap_time is None or summary.best_lap_time < self.best_lap_time
        )
        if new_record:
            self.best_lap_time = summary.best_lap_time
            self.best_lap_episode = summary.episode

        self.history_progress = (self.history_progress + [summary.best_progress])[-HISTORY_LIMIT:]
        self.history_laps = (self.history_laps + [summary.laps])[-HISTORY_LIMIT:]
        self.history_lap_time = (self.history_lap_time + [summary.best_lap_time])[-HISTORY_LIMIT:]
        return new_record

    def recent_lap_rate(self, window: int = 20) -> float:
        """Son bölümlerde en az bir tur tamamlanan bölüm oranı."""
        recent = self.history_laps[-window:]
        return sum(1 for laps in recent if laps) / len(recent) if recent else 0.0

    def to_dict(self) -> dict:
        return {"version": STATS_VERSION, **asdict(self)}

    @classmethod
    def from_dict(cls, data: dict) -> TrainingStats:
        if data.get("version") != STATS_VERSION:
            raise ValueError("eski veya tanınmayan istatistik biçimi")
        known = {k: data[k] for k in cls.__dataclass_fields__ if k in data}
        return cls(**known)


def load_stats(path: Path = STATS_PATH) -> TrainingStats:
    """İstatistikleri yükler; dosya yoksa ya da bozuksa sıfırdan başlar."""
    if not path.exists():
        return TrainingStats()
    try:
        with path.open("r", encoding="utf-8") as fh:
            return TrainingStats.from_dict(json.load(fh))
    except (OSError, ValueError, TypeError) as exc:
        log.warning("%s okunamadı (%s); istatistikler sıfırdan başlıyor.", path, exc)
        return TrainingStats()


def save_stats(stats: TrainingStats, path: Path = STATS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(stats.to_dict(), fh, indent=2)
    tmp.replace(path)


def _fmt(value: float | None, digits: int) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return f"{value:.{digits}f}"


def append_episode_log(summary: EpisodeSummary, path: Path = EPISODE_LOG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if is_new:
            writer.writerow(LOG_FIELDS)
        writer.writerow([
            datetime.now().strftime(TIMESTAMP_FORMAT),
            summary.episode,
            summary.laps,
            _fmt(summary.best_lap_time, 3),
            _fmt(summary.best_progress, 4),
            _fmt(summary.mean_reward, 3),
            _fmt(summary.epsilon, 4),
            _fmt(summary.loss, 5),
            summary.sim_steps,
        ])


def has_model() -> bool:
    return MODEL_PATH.exists()


def reset_all() -> list[Path]:
    """Model, istatistik ve günlük dosyalarını siler; silinenleri döndürür."""
    removed = []
    for path in (MODEL_PATH, BEST_MODEL_PATH, STATS_PATH, EPISODE_LOG_PATH):
        if path.exists():
            path.unlink()
            removed.append(path)
    return removed
