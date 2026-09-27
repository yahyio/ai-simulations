"""Vektörize yarış ortamı: tüm arabaların fiziği, sensörleri ve ödülleri tek adımda.

Zaman sabit adımlıdır (1 adım = 1/sim_fps saniye); duvar saatine bağlı hiçbir
hesap yoktur, bu yüzden simülasyon hızı öğrenmeyi ve tur sürelerini etkilemez.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .track import Track

# Eylem tablosu: (direksiyon, gaz). Direksiyon -1 sol / 0 düz / +1 sağ,
# gaz +1 hızlan / 0 bırak / -1 fren.
ACTIONS = np.array([(s, t) for t in (1, 0, -1) for s in (-1, 0, 1)], dtype=np.float32)
NUM_ACTIONS = len(ACTIONS)

RAY_STEP = 4.0               # Sensör ışını örnekleme aralığı (px)
PROGRESS_SCALE = 10.0        # Her 10 px ilerleme = 1 × rewards.progress
LAP_ENTER_FRACTION = 0.75    # Tur sayılması için çizgiden önce bu orana ulaşılmalı
LAP_EXIT_FRACTION = 0.25
FULL_STEER_SPEED_RATIO = 0.2  # Bu hız oranının altında direksiyon etkisi azalır
SPAWN_JITTER_PX = 2.0
SPAWN_JITTER_RAD = 0.03
IDLE_EPSILON_PX = 0.5


class Status:
    RUNNING = 0
    CRASHED = 1
    LAP = 2
    IDLE = 3
    WRONG_WAY = 4
    TIMEOUT = 5


STATUS_LABELS = {
    Status.RUNNING: "yarışıyor",
    Status.CRASHED: "pist dışı",
    Status.LAP: "tur tamam",
    Status.IDLE: "ilerlemiyor",
    Status.WRONG_WAY: "ters yön",
    Status.TIMEOUT: "süre doldu",
}


@dataclass(frozen=True)
class StepResult:
    obs: np.ndarray         # (N, obs_dim) yeni gözlemler
    rewards: np.ndarray     # (N,)
    terminated: np.ndarray  # (N,) gerçek bitiş (çarpma, tur, ters yön, hareketsizlik)
    truncated: np.ndarray   # (N,) zaman sınırı — değer tahmini sürdürülür
    active: np.ndarray      # (N,) bu adımdan önce yarışan arabalar
    laps: np.ndarray        # (N,) bu adımda tur tamamlayanlar


class RaceEnv:
    """N arabanın aynı pistte paralel sürdüğü ortam (arabalar birbirine çarpmaz)."""

    def __init__(self, track: Track, config: dict, num_cars: int, rng: np.random.Generator | None = None):
        car = config["car"]
        sim = config["simulation"]
        self.track = track
        self.rewards_cfg = dict(config["rewards"])
        self.num_cars = int(num_cars)
        self.rng = rng if rng is not None else np.random.default_rng()

        self.car_length = float(car["length"])
        self.car_width = float(car["width"])
        self.max_speed = float(car["max_speed"])
        self.acceleration = float(car["acceleration"])
        self.brake = float(car["brake"])
        self.drag = float(car["drag"])
        self.turn_rate = np.radians(float(car["turn_rate_deg"]))
        self.understeer = float(car["understeer"])
        self.sensor_range = float(car["sensor_range"])
        fov = np.radians(float(car["sensor_fov_deg"]))
        self.sensor_angles = np.linspace(-fov / 2, fov / 2, int(car["sensor_count"]))
        self.ray_steps = np.minimum(np.arange(RAY_STEP, self.sensor_range + RAY_STEP, RAY_STEP), self.sensor_range)

        self.max_steps = int(sim["max_steps_per_episode"])
        self.idle_timeout = int(sim["idle_timeout_steps"])
        self.sim_fps = float(sim["sim_fps"])
        self.obs_dim = len(self.sensor_angles) + 4

        n = self.num_cars
        self.x = np.zeros(n)
        self.y = np.zeros(n)
        self.heading = np.zeros(n)
        self.speed = np.zeros(n)
        self.steer = np.zeros(n)
        self.status = np.zeros(n, dtype=np.int8)
        self.progress = np.zeros(n)
        self.distance = np.zeros(n)
        self.best_distance = np.zeros(n)
        self.idle_steps = np.zeros(n, dtype=np.int32)
        self.episode_rewards = np.zeros(n)
        self.lap_times = np.full(n, np.nan)
        self.sensors = np.ones((n, len(self.sensor_angles)))
        self.steps = 0
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self) -> np.ndarray:
        n = self.num_cars
        x0, y0, h0 = self.track.spawn_pose(self.car_length)
        lateral = np.array([-np.sin(h0), np.cos(h0)])
        spread = n > 1
        jitter = self.rng.uniform(-SPAWN_JITTER_PX, SPAWN_JITTER_PX, n) if spread else np.zeros(n)
        turn = self.rng.uniform(-SPAWN_JITTER_RAD, SPAWN_JITTER_RAD, n) if spread else np.zeros(n)
        self.x[:] = x0 + lateral[0] * jitter
        self.y[:] = y0 + lateral[1] * jitter
        self.heading[:] = h0 + turn
        self.speed[:] = 0.0
        self.steer[:] = 0.0
        self.status[:] = Status.RUNNING
        self.progress[:] = np.maximum(self.track.progress_at(self.x, self.y), 0.0)
        self.distance[:] = 0.0
        self.best_distance[:] = 0.0
        self.idle_steps[:] = 0
        self.episode_rewards[:] = 0.0
        self.lap_times[:] = np.nan
        self.steps = 0
        self._update_sensors(np.arange(n))
        return self.observe()

    @property
    def running(self) -> np.ndarray:
        return self.status == Status.RUNNING

    @property
    def episode_over(self) -> bool:
        return not self.running.any()

    # ------------------------------------------------------------------ #
    def step(self, actions: np.ndarray) -> StepResult:
        n = self.num_cars
        active = self.running.copy()
        rewards = np.zeros(n)
        terminated = np.zeros(n, dtype=bool)
        truncated = np.zeros(n, dtype=bool)
        laps = np.zeros(n, dtype=bool)
        idx = np.flatnonzero(active)
        if idx.size == 0:
            return StepResult(self.observe(), rewards, terminated, truncated, active, laps)

        self.steps += 1
        self._integrate(idx, np.asarray(actions)[idx])
        delta, lap, wrong_way = self._advance_progress(idx)

        r = self.rewards_cfg
        reward = r["progress"] * delta / PROGRESS_SCALE + r["time_penalty"]

        improved = self.distance[idx] > self.best_distance[idx] + IDLE_EPSILON_PX
        self.best_distance[idx] = np.maximum(self.best_distance[idx], self.distance[idx])
        self.idle_steps[idx] = np.where(improved, 0, self.idle_steps[idx] + 1)

        crashed = ~self._body_on_track(idx)
        idle = self.idle_steps[idx] >= self.idle_timeout
        # Öncelik: pist dışı > ters yön > tur > hareketsizlik
        wrong_way &= ~crashed
        lap &= ~crashed & ~wrong_way
        idle &= ~crashed & ~wrong_way & ~lap

        reward += np.where(crashed, r["crash"], 0.0)
        reward += np.where(wrong_way, r["wrong_way"], 0.0)
        reward += np.where(lap, r["lap_complete"], 0.0)
        reward += np.where(idle, r["idle"], 0.0)

        status = np.full(idx.size, Status.RUNNING, dtype=np.int8)
        status[idle] = Status.IDLE
        status[lap] = Status.LAP
        status[wrong_way] = Status.WRONG_WAY
        status[crashed] = Status.CRASHED
        ended = status != Status.RUNNING
        timeout = ~ended & (self.steps >= self.max_steps)
        status[timeout] = Status.TIMEOUT

        self.status[idx] = status
        self.lap_times[idx[lap]] = self.steps / self.sim_fps
        self.episode_rewards[idx] += reward
        rewards[idx] = reward
        terminated[idx] = ended
        truncated[idx] = timeout
        laps[idx] = lap

        self._update_sensors(idx)
        return StepResult(self.observe(), rewards, terminated, truncated, active, laps)

    def _integrate(self, idx: np.ndarray, actions: np.ndarray) -> None:
        """Basit ama sürüş hissi veren fizik: ivme, fren, sürtünme, savrulma."""
        steer, throttle = ACTIONS[actions].T
        speed = self.speed[idx]
        speed = speed + np.where(throttle > 0, self.acceleration, 0.0)
        speed = speed - np.where(throttle < 0, self.brake, 0.0)
        speed = np.clip(speed - self.drag * speed, 0.0, self.max_speed)

        ratio = speed / self.max_speed
        grip = 1.0 - self.understeer * ratio**2  # yüksek hızda daha geniş viraj
        low_speed = np.clip(speed / (FULL_STEER_SPEED_RATIO * self.max_speed), 0.0, 1.0)
        heading = self.heading[idx] + steer * self.turn_rate * grip * low_speed

        self.speed[idx] = speed
        self.heading[idx] = np.mod(heading + np.pi, 2 * np.pi) - np.pi
        self.steer[idx] = steer
        self.x[idx] += speed * np.cos(heading)
        self.y[idx] += speed * np.sin(heading)

    def _advance_progress(self, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pist boyunca ilerlemeyi günceller; tur ve ters yön geçişlerini tespit eder."""
        length = self.track.length
        old = self.progress[idx]
        new = self.track.progress_at(self.x[idx], self.y[idx]).astype(np.float64)
        new = np.where(new < 0, old, new)
        lap = (old > LAP_ENTER_FRACTION * length) & (new < LAP_EXIT_FRACTION * length)
        wrong_way = (old < LAP_EXIT_FRACTION * length) & (new > LAP_ENTER_FRACTION * length)
        delta = new - old + np.where(lap, length, 0.0) - np.where(wrong_way, length, 0.0)
        self.progress[idx] = new
        self.distance[idx] += delta
        return delta, lap, wrong_way

    def _body_on_track(self, idx: np.ndarray) -> np.ndarray:
        """Arabanın dört köşesi de pistte mi?"""
        c, s = np.cos(self.heading[idx]), np.sin(self.heading[idx])
        hl, hw = self.car_length / 2, self.car_width / 2
        ok = np.ones(idx.size, dtype=bool)
        for fl, fw in ((hl, hw), (hl, -hw), (-hl, hw), (-hl, -hw)):
            px = self.x[idx] + c * fl - s * fw
            py = self.y[idx] + s * fl + c * fw
            ok &= self.track.on_track(px, py)
        return ok

    def _update_sensors(self, idx: np.ndarray) -> None:
        """Her araba için yelpaze şeklinde ışın gönderip pist kenarına mesafeyi ölçer."""
        angles = self.heading[idx, None] + self.sensor_angles[None, :]
        px = self.x[idx, None, None] + np.cos(angles)[..., None] * self.ray_steps
        py = self.y[idx, None, None] + np.sin(angles)[..., None] * self.ray_steps
        hit = ~self.track.on_track(px, py)
        first = hit.argmax(axis=2)
        dist = np.where(hit.any(axis=2), self.ray_steps[first], self.sensor_range)
        self.sensors[idx] = dist / self.sensor_range

    # ------------------------------------------------------------------ #
    def observe(self) -> np.ndarray:
        """Gözlem: sensörler + hız + piste göre yön hatası (sin, cos) + son direksiyon."""
        direction = self.track.direction_at(self.x, self.y)
        error = self.heading - np.arctan2(direction[:, 1], direction[:, 0])
        extras = np.stack([self.speed / self.max_speed, np.sin(error), np.cos(error), self.steer], axis=1)
        return np.concatenate([self.sensors, extras], axis=1).astype(np.float32)

    def sensor_endpoints(self) -> np.ndarray:
        """Işınların uç noktaları (N, S, 2) — çizim için."""
        angles = self.heading[:, None] + self.sensor_angles[None, :]
        dist = self.sensors * self.sensor_range
        return np.stack(
            [self.x[:, None] + np.cos(angles) * dist, self.y[:, None] + np.sin(angles) * dist], axis=-1
        )

    def progress_fraction(self) -> np.ndarray:
        """Tur yüzdesi (0–1); turu tamamlayanlar tam 1."""
        fraction = np.clip(self.distance / self.track.length, 0.0, 1.0)
        return np.where(self.status == Status.LAP, 1.0, fraction)

    def leader(self) -> int:
        """En uzağa gitmiş araba (yarışanlar öncelikli)."""
        return int(np.argmax(self.distance + np.where(self.running, 1e6, 0.0)))
