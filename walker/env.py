"""Vektörize hümanoid ortamı: N karakter aynı anda dener; düşen hemen baştan başlar.

Karakter hiçbir hareket bilgisiyle başlamaz: 11 eklem motoruna [-1, 1] arası sinyal
gönderir, karşılığında yalnızca bir sayı (ödül) alır. Gerisi deneme yanılmadır.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .model import FOOT_GEOMS, PELVIS_HEIGHT, TIMESTEP, load_model

FRAME_SKIP = 4                    # kontrol adımı başına fizik adımı
CONTROL_DT = TIMESTEP * FRAME_SKIP  # 16 ms
MAX_EPISODE_STEPS = 1000          # 16 saniye
MIN_PELVIS_HEIGHT = 0.62          # bunun altı: düştü
MAX_TILT = 0.9                    # gövde eğimi (radyan) bunun üstü: düştü
INIT_NOISE = 0.03
VEL_CLIP = 10.0
HEALTHY_REWARD = 1.0
CTRL_COST = 0.005
OVERSPEED_PENALTY = 0.75          # yürümede hedef hızın üstündeki her m/s için


@dataclass(frozen=True)
class Task:
    key: str
    label: str
    target_speed: float | None    # None: olabildiğince hızlı
    speed_weight: float
    description: str
    posture_weight: float         # gövde eğikliği cezası: w × (1 − cos eğim)
    step_air_target: float        # s — ayak bundan uzun havada kaldıysa iniş ödüllenir
    step_air_weight: float


# Duruş ve adım terimleri hareketi öğretmez; yalnızca "güzel" hareketi daha çok ödüllendirir.
# Onlarsız ajan belden 45° geriye yatıp minik adımlarla sürünmeyi keşfediyordu.
TASKS = {
    "yurume": Task("yurume", "Yürüme", 1.5, 1.25, "hedef 1.5 m/s — dik, dengeli yürüyüş", 3.0, 0.45, 2.0),
    "kosu": Task("kosu", "Koşu", None, 1.25, "olabildiğince hızlı ileri", 0.5, 0.25, 1.0),
}


@dataclass(frozen=True)
class Episode:
    """Biten bir denemenin özeti."""
    env: int
    reward: float
    distance: float     # m
    steps: int
    fell: bool

    @property
    def seconds(self) -> float:
        return self.steps * CONTROL_DT

    @property
    def speed(self) -> float:
        return self.distance / max(self.seconds, 1e-9)


def forward_reward(task: Task, speed):
    """Hız ödülü: koşuda doğrusal; yürümede hedefe kadar artar, üstünde azalır."""
    if task.target_speed is None:
        return task.speed_weight * speed
    below = np.minimum(speed, task.target_speed)
    above = np.maximum(speed - task.target_speed, 0.0)
    return task.speed_weight * below - OVERSPEED_PENALTY * above


class WalkerEnv:
    def __init__(self, num_envs: int, task: Task, seed: int | None = None, max_steps: int = MAX_EPISODE_STEPS):
        self.model = load_model()
        self.num_envs = num_envs
        self.task = task
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.datas = [mujoco.MjData(self.model) for _ in range(num_envs)]
        self.foot_order = [self.model.geom(name).id for name in FOOT_GEOMS]
        self.act_dim = self.model.nu
        self.obs_dim = (self.model.nq - 1) + self.model.nv + len(FOOT_GEOMS)
        self.chest_body = self.model.body("chest").id
        self.returns = np.zeros(num_envs)
        self.steps = np.zeros(num_envs, dtype=np.int64)
        self.air_time = np.zeros((num_envs, len(FOOT_GEOMS)))
        self.last_foot = np.full(num_envs, -1)   # en son yere inen ayak (sıra kuralı için)

    # ------------------------------------------------------------------ #
    def reset(self) -> np.ndarray:
        for i in range(self.num_envs):
            self._reset_one(i)
        return self.observe_all()

    def _reset_one(self, i: int) -> None:
        d = self.datas[i]
        mujoco.mj_resetData(self.model, d)
        d.qpos[:] += self.rng.uniform(-INIT_NOISE, INIT_NOISE, self.model.nq)
        d.qvel[:] += self.rng.uniform(-INIT_NOISE, INIT_NOISE, self.model.nv)
        d.qpos[0] = 0.0
        mujoco.mj_forward(self.model, d)
        self.returns[i] = 0.0
        self.steps[i] = 0
        self.air_time[i] = 0.0
        self.last_foot[i] = -1

    def step(self, actions: np.ndarray):
        """Tüm karakterleri bir kontrol adımı ilerletir.

        Döner: obs, ödüller, terminated (düştü), truncated (süre doldu), final_obs
        (bitenlerin sıfırlanmadan önceki son gözlemi — değer tahmini için) ve biten denemeler.
        """
        actions = np.clip(actions, -1.0, 1.0)
        n = self.num_envs
        obs = np.zeros((n, self.obs_dim), dtype=np.float32)
        final_obs = np.zeros_like(obs)
        rewards = np.zeros(n)
        terminated = np.zeros(n, dtype=bool)
        truncated = np.zeros(n, dtype=bool)
        finished: list[Episode] = []

        for i, d in enumerate(self.datas):
            x_before = d.qpos[0]
            d.ctrl[:] = actions[i]
            mujoco.mj_step(self.model, d, nstep=FRAME_SKIP)
            speed = (d.qpos[0] - x_before) / CONTROL_DT
            fell = self.fallen(d)
            contacts = self.foot_contacts(d)
            reward = (forward_reward(self.task, speed) + HEALTHY_REWARD
                      - CTRL_COST * float(np.square(actions[i]).sum())
                      - self.task.posture_weight * (1.0 - self.upright(d))
                      + self._step_reward(i, contacts))
            rewards[i] = reward
            self.returns[i] += reward
            self.steps[i] += 1
            obs[i] = self._observe(d, contacts)

            timeout = not fell and self.steps[i] >= self.max_steps
            if fell or timeout:
                terminated[i], truncated[i] = fell, timeout
                final_obs[i] = obs[i]
                finished.append(Episode(i, float(self.returns[i]), float(d.qpos[0]), int(self.steps[i]), fell))
                self._reset_one(i)
                obs[i] = self._observe(d)
        return obs, rewards, terminated, truncated, final_obs, finished

    # ------------------------------------------------------------------ #
    @staticmethod
    def fallen(d: mujoco.MjData) -> bool:
        pelvis_z = PELVIS_HEIGHT + d.qpos[1]
        return bool(pelvis_z < MIN_PELVIS_HEIGHT or abs(d.qpos[2]) > MAX_TILT)

    def foot_contacts(self, d: mujoco.MjData) -> np.ndarray:
        """Her ayak için 1: yere değiyor, 0: havada."""
        if d.ncon == 0:
            return np.zeros(len(self.foot_order), dtype=np.float32)
        touching = set(d.contact.geom[: d.ncon].ravel().tolist())  # küçük dizide np.isin'den ~10x hızlı
        return np.array([gid in touching for gid in self.foot_order], dtype=np.float32)

    def upright(self, d: mujoco.MjData) -> float:
        """Göğsün dikliği: 1 = tam dik, 0 = yatay (göğüs gövdesinin z ekseninin dikey bileşeni)."""
        return float(d.xmat[self.chest_body][8])

    def _step_reward(self, i: int, contacts: np.ndarray) -> float:
        """Yere inen ayak için: havada kalma süresi hedefin üstündeyse ödül, altındaysa ceza.

        Aynı ayak üst üste inerse (tek bacakla sekme) hiç havada kalmamış gibi cezalandırılır;
        böylece sağ-sol sırayla adım atmak kazançlı olur.
        """
        target = self.task.step_air_target
        reward = 0.0
        for k in np.flatnonzero((contacts > 0) & (self.air_time[i] > 0)):
            reward += -target if self.last_foot[i] == k else self.air_time[i, k] - target
            self.last_foot[i] = k
        self.air_time[i] = np.where(contacts > 0, 0.0, self.air_time[i] + CONTROL_DT)
        return self.task.step_air_weight * reward

    def _observe(self, d: mujoco.MjData, contacts: np.ndarray | None = None) -> np.ndarray:
        return np.concatenate([
            d.qpos[1:],                               # yükseklik, eğim, eklem açıları (x hariç)
            np.clip(d.qvel, -VEL_CLIP, VEL_CLIP),     # hızlar
            self.foot_contacts(d) if contacts is None else contacts,  # ayaklar yere değiyor mu
        ]).astype(np.float32)

    def observe_all(self) -> np.ndarray:
        return np.stack([self._observe(d) for d in self.datas])

    def positions(self) -> np.ndarray:
        """Her karakterin kat ettiği mesafe (m)."""
        return np.array([d.qpos[0] for d in self.datas])
