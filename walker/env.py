"""3B hümanoid ortamı: N karakter aynı anda dener; düşen hemen baştan başlar.

Karakter hiçbir hareket bilgisiyle başlamaz. Her 20 ms'de 21 eklemine [-1, 1] arası bir
hedef gönderir (eklem, PD kontrolüyle o açıya çekilir) ve karşılığında yalnızca bir sayı
(ödül) alır. Dengede durmak, adım atmak, yürümek ve koşmak deneme yanılmayla ortaya çıkar.

Tek ipucu bir metronomdur: gözlemdeki adım saati. Ayak temasları bu ritme uyduğunda ödül
biraz artar; bacakların nasıl hareket edeceğini yine ajan kendisi bulur. Saat olmadan ajan
iki ayakla birlikte zıplayarak (kanguru gibi) ilerlemeyi keşfediyordu (ölçüldü).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from .model import (
    DEFAULT_POSE,
    FOOT_GEOMS,
    JOINTS,
    TIMESTEP,
    joint_addresses,
    joint_ranges,
    load_model,
    standing_qpos,
    yaw_of,
)

FRAME_SKIP = 5
CONTROL_DT = TIMESTEP * FRAME_SKIP      # 20 ms (50 Hz kontrol)
MAX_EPISODE_STEPS = 1000                # 20 saniye
MIN_PELVIS_HEIGHT = 0.55                # altı: düştü
MIN_UPRIGHT = 0.3                       # gövde dikliği (cos eğim) altı: düştü
INIT_JOINT_NOISE = 0.05
INIT_VEL_NOISE = 0.1
VEL_CLIP = 20.0

HEALTHY_REWARD = 2.0
FALL_PENALTY = 10.0
UPRIGHT_GATE = (0.7, 0.95)              # gövde dikliği: bunun altında hız ödülü sıfır, üstünde tam
OVERSPEED_PENALTY = 0.75
LATERAL_SPEED_WEIGHT = 0.5
YAW_RATE_WEIGHT = 0.05
HEADING_WEIGHT = 1.0
DRIFT_WEIGHT = 0.1
FEET_MIN_GAP = 0.18                     # m — ayaklar bundan yakınsa (ya da çaprazsa) ceza
FEET_GAP_WEIGHT = 5.0
FOOT_HEIGHT_WEIGHT = 2.0                # salınan ayak gereğinden yükseğe kalkarsa (m başına)
SOLE_OFFSET = 0.025                     # ayak kutusunun yarı yüksekliği: merkezden tabana
JOINT_DEVIATION_WEIGHT = 0.5            # bel, kalça yan/dönme ve omuz eklemlerinin doğal duruştan sapması (rad²)
REGULARIZED_JOINTS = ("abdomen_z", "abdomen_y", "abdomen_x", "hip_x_r", "hip_z_r", "hip_x_l", "hip_z_l",
                      "shoulder_x_r", "shoulder_x_l")
ACTION_RATE_WEIGHT = 0.01
TORQUE_WEIGHT = 2e-6
CONTACT_WEIGHT = 0.4                    # ayak başına: yere değme durumu adım saatine uyuyorsa
SLIP_WEIGHT = 1.0                       # yere basan ayağın kayma hızı² (m²/s²) cezası
FOOT_PHASE_OFFSETS = (0.0, 0.5)         # sağ, sol ayak: yarım tur arayla


ACT_DIM = len(JOINTS)
CLOCK_DIMS = 2                                                                      # adım saati: sin, cos
OBS_DIM = 1 + 3 + 3 + 3 + 2 + 1 + 3 * ACT_DIM + len(FOOT_GEOMS) + CLOCK_DIMS        # _observe() ile aynı düzen
CONTACT_OBS = slice(OBS_DIM - CLOCK_DIMS - len(FOOT_GEOMS), OBS_DIM - CLOCK_DIMS)   # gözlemde ayak temasları


@dataclass(frozen=True)
class Task:
    key: str
    label: str
    target_speed: float | None      # None: olabildiğince hızlı (max_speed'e kadar)
    max_speed: float
    speed_weight: float
    description: str
    posture_weight: float           # gövde eğikliği cezası: w × (1 − cos eğim)
    step_air_target: float          # s — ayak bundan uzun havada kaldıysa iniş ödüllenir
    step_air_cap: float             # s — bundan uzun havada kalmanın ek ödülü yok
    step_air_weight: float
    foot_clearance: float           # m — salınan ayağın cezasız kalkabileceği yükseklik
    gait_period: float              # s — adım saatinin bir turu (sağ + sol adım)
    stance_fraction: float          # turun ne kadarında ayak yerde olmalı (yürüme > 0.5 > koşu)
    flight_penalty: float           # iki ayak birden havadaysa (yürümenin tanımı: bir ayak hep yerde)


# Duruş, adım ve yön terimleri hareketi öğretmez; yalnızca "güzel" hareketi daha çok
# ödüllendirir. Onlarsız ajan sürünme, tek ayakla sekme, yan yan kayma ya da bacağını
# yatay tekmeleyip havada tutma gibi "hileler" keşfeder (hepsi denemelerde görüldü).
# Yürüme: 1 s'lik tur, ayak turun %60'ında yerde (insan yürüyüşü: 120 adım/dk, %20 çift destek).
# Koşu: 0.7 s'lik tur, ayak turun %35'inde yerde — iki ayağın da havada olduğu uçuş anları var.
TASKS = {
    "yurume": Task("yurume", "Yürüme", target_speed=1.4, max_speed=1.4, speed_weight=1.5,
                   description="hedef 1.4 m/s — dik, dengeli yürüyüş", posture_weight=2.0,
                   step_air_target=0.40, step_air_cap=0.55, step_air_weight=2.0, foot_clearance=0.18,
                   gait_period=1.0, stance_fraction=0.6, flight_penalty=1.0),
    "kosu": Task("kosu", "Koşu", target_speed=None, max_speed=8.0, speed_weight=1.25,
                 description="olabildiğince hızlı ileri", posture_weight=1.0,
                 step_air_target=0.30, step_air_cap=0.45, step_air_weight=1.0, foot_clearance=0.45,
                 gait_period=0.7, stance_fraction=0.35, flight_penalty=0.0),
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
    """Hız ödülü: koşuda üst sınıra kadar doğrusal; yürümede hedefe kadar artar, üstünde azalır."""
    capped = np.minimum(speed, task.max_speed)
    if task.target_speed is None:
        return task.speed_weight * capped
    over = np.maximum(speed - task.target_speed, 0.0)
    return task.speed_weight * np.minimum(speed, task.target_speed) - OVERSPEED_PENALTY * over


class WalkerEnv:
    def __init__(self, num_envs: int, task: Task, seed: int | None = None, max_steps: int = MAX_EPISODE_STEPS):
        self.model = load_model()
        self.num_envs = num_envs
        self.task = task
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.datas = [mujoco.MjData(self.model) for _ in range(num_envs)]

        self.pelvis = self.model.body("pelvis").id
        self.torso = self.model.body("torso").id
        self.feet = [self.model.geom(name).id for name in FOOT_GEOMS]
        ranges = joint_ranges(self.model)
        self.lo, self.hi = ranges[:, 0], ranges[:, 1]
        self.qadr, self.dadr = joint_addresses(self.model)   # eylem sırası → qpos/qvel adresleri
        self.center = DEFAULT_POSE.copy()
        self.scale = 0.5 * (self.hi - self.lo)      # eylem ±1 → eklem aralığının yarısı kadar sapma
        names = [j.name for j in JOINTS]
        self.regularized = np.array([names.index(n) for n in REGULARIZED_JOINTS])

        self.act_dim = self.model.nu
        self.obs_dim = OBS_DIM
        self.returns = np.zeros(num_envs)
        self.steps = np.zeros(num_envs, dtype=np.int64)
        self.prev_actions = np.zeros((num_envs, self.act_dim))
        self.air_time = np.zeros((num_envs, len(self.feet)))
        self.last_foot = np.full(num_envs, -1)
        self.prev_feet = np.zeros((num_envs, len(self.feet), 2))    # ayakların önceki adımdaki (x, y) konumu
        self.phase_offsets = np.array(FOOT_PHASE_OFFSETS)

    # ------------------------------------------------------------------ #
    def reset(self) -> np.ndarray:
        for i in range(self.num_envs):
            self._reset_one(i)
        return self.observe_all()

    def _reset_one(self, i: int) -> None:
        d = self.datas[i]
        mujoco.mj_resetData(self.model, d)
        d.qpos[:] = standing_qpos()
        d.qpos[self.qadr] += self.rng.uniform(-INIT_JOINT_NOISE, INIT_JOINT_NOISE, self.act_dim)
        d.qvel[:] = self.rng.uniform(-INIT_VEL_NOISE, INIT_VEL_NOISE, self.model.nv)
        d.ctrl[:] = self.center
        mujoco.mj_forward(self.model, d)
        self.returns[i] = 0.0
        self.steps[i] = 0
        self.prev_actions[i] = 0.0
        self.air_time[i] = 0.0
        self.last_foot[i] = -1
        self.prev_feet[i] = d.geom_xpos[self.feet][:, :2]

    def targets(self, actions: np.ndarray) -> np.ndarray:
        """Eylem [-1, 1] → eklem hedef açısı (varsayılan duruş etrafında, sınırlar içinde)."""
        return np.clip(self.center + self.scale * actions, self.lo, self.hi)

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
            x0, y0 = d.qpos[0], d.qpos[1]
            d.ctrl[:] = self.targets(actions[i])
            mujoco.mj_step(self.model, d, nstep=FRAME_SKIP)
            vx = (d.qpos[0] - x0) / CONTROL_DT
            vy = (d.qpos[1] - y0) / CONTROL_DT
            contacts = self.foot_contacts(d)
            reward = self._reward(i, d, actions[i], vx, vy, contacts)
            self.prev_actions[i] = actions[i]
            self.prev_feet[i] = d.geom_xpos[self.feet][:, :2]
            rewards[i] = reward
            self.returns[i] += reward
            self.steps[i] += 1
            obs[i] = self._observe(d, i, contacts)

            fell = self.fallen(d)
            if fell:
                rewards[i] -= FALL_PENALTY
                self.returns[i] -= FALL_PENALTY
            timeout = not fell and self.steps[i] >= self.max_steps
            if fell or timeout:
                terminated[i], truncated[i] = fell, timeout
                final_obs[i] = obs[i]
                finished.append(Episode(i, float(self.returns[i]), float(d.qpos[0]), int(self.steps[i]), fell))
                self._reset_one(i)
                obs[i] = self._observe(d, i)
        return obs, rewards, terminated, truncated, final_obs, finished

    # ------------------------------------------------------------------ #
    def _reward(self, i: int, d: mujoco.MjData, action: np.ndarray, vx: float, vy: float,
                contacts: np.ndarray) -> float:
        task = self.task
        yaw = yaw_of(d.qpos[3:7])
        upright = self.upright(d)
        feet = d.geom_xpos[self.feet]
        gap = feet[1, 1] - feet[0, 1] if feet[1, 1] > feet[0, 1] else 0.0   # sol ayak solda olmalı
        lift = np.maximum(feet[:, 2] - SOLE_OFFSET - task.foot_clearance, 0.0).sum()
        deviation = d.qpos[self.qadr[self.regularized]] - self.center[self.regularized]
        down = contacts > 0
        in_step = float(np.count_nonzero(down == self.stance(i)))           # ritme uyan ayak sayısı
        planted = down & (self.air_time[i] == 0)                            # bir önceki adımda da yerdeydi
        foot_speed = (feet[:, :2] - self.prev_feet[i]) / CONTROL_DT
        slip = float(np.square(foot_speed[planted]).sum())
        # Hız ödülü yalnızca dik dururken: öne doğru düşerek (balıklama) hız toplamak kârsız.
        gate = float(np.clip((upright - UPRIGHT_GATE[0]) / (UPRIGHT_GATE[1] - UPRIGHT_GATE[0]), 0.0, 1.0))
        return (
            gate * float(forward_reward(task, vx))
            + HEALTHY_REWARD
            - task.posture_weight * (1.0 - upright)
            - LATERAL_SPEED_WEIGHT * vy * vy
            - YAW_RATE_WEIGHT * float(d.qvel[5]) ** 2
            - HEADING_WEIGHT * (1.0 - math.cos(yaw))
            - DRIFT_WEIGHT * float(d.qpos[1]) ** 2
            - FEET_GAP_WEIGHT * max(0.0, FEET_MIN_GAP - gap)
            - FOOT_HEIGHT_WEIGHT * float(lift)
            - JOINT_DEVIATION_WEIGHT * float(np.square(deviation).sum())
            - ACTION_RATE_WEIGHT * float(np.square(action - self.prev_actions[i]).sum())
            - TORQUE_WEIGHT * float(np.square(d.actuator_force).sum())
            + CONTACT_WEIGHT * in_step
            - SLIP_WEIGHT * slip
            - task.flight_penalty * float(not down.any())
            + self._step_reward(i, contacts)
        )

    def _step_reward(self, i: int, contacts: np.ndarray) -> float:
        """Yere inen ayak için: havada kalma süresi hedefin üstündeyse ödül, altındaysa ceza.

        Aynı ayak üst üste inerse (tek bacakla sekme) ya da iki ayak aynı anda inerse (iki ayakla
        zıplama) hiç havada kalmamış gibi cezalandırılır; böylece sağ-sol sırayla adım atmak kazançlı olur.
        """
        target, cap = self.task.step_air_target, self.task.step_air_cap
        landed = np.flatnonzero((contacts > 0) & (self.air_time[i] > 0))
        hop = len(landed) > 1
        reward = 0.0
        for k in landed:
            airborne = min(self.air_time[i, k], cap)   # bacağı havada tutmak ek kazanç getirmez
            reward += -target if hop or self.last_foot[i] == k else airborne - target
            self.last_foot[i] = k
        self.air_time[i] = np.where(contacts > 0, 0.0, self.air_time[i] + CONTROL_DT)
        return self.task.step_air_weight * reward

    def clock(self, i: int) -> float:
        """Adım saati: denemenin başından beri geçen süre, tur kesri olarak [0, 1)."""
        return float(self.steps[i] * CONTROL_DT / self.task.gait_period) % 1.0

    def stance(self, i: int) -> np.ndarray:
        """Saate göre hangi ayak yerde olmalı (sağ, sol)."""
        leg_phase = (self.clock(i) + self.phase_offsets) % 1.0
        return leg_phase < self.task.stance_fraction

    def upright(self, d: mujoco.MjData) -> float:
        """Göğsün dikliği: 1 = tam dik, 0 = yatay (gövde z ekseninin dikey bileşeni)."""
        return float(d.xmat[self.torso][8])

    def fallen(self, d: mujoco.MjData) -> bool:
        return bool(d.qpos[2] < MIN_PELVIS_HEIGHT or self.upright(d) < MIN_UPRIGHT)

    def foot_contacts(self, d: mujoco.MjData) -> np.ndarray:
        """Her ayak için 1: yere değiyor, 0: havada."""
        if d.ncon == 0:
            return np.zeros(len(self.feet), dtype=np.float32)
        touching = set(d.contact.geom[: d.ncon].ravel().tolist())   # küçük dizide np.isin'den hızlı
        return np.array([gid in touching for gid in self.feet], dtype=np.float32)

    def _observe(self, d: mujoco.MjData, i: int, contacts: np.ndarray | None = None) -> np.ndarray:
        rot = d.xmat[self.pelvis].reshape(3, 3)
        yaw = yaw_of(d.qpos[3:7])
        angle = 2 * math.pi * self.clock(i)
        return np.concatenate([
            [d.qpos[2]],                                        # leğen yüksekliği
            -rot[2, :],                                         # gövde çerçevesinde yerçekimi yönü (eğim)
            rot.T @ d.qvel[0:3],                                # gövde çerçevesinde doğrusal hız
            d.qvel[3:6],                                        # açısal hız
            [math.cos(yaw), math.sin(yaw), d.qpos[1]],          # yön ve kulvardan sapma
            d.qpos[self.qadr] - self.center,                    # eklem açıları (varsayılana göre)
            np.clip(d.qvel[self.dadr], -VEL_CLIP, VEL_CLIP),    # eklem hızları
            self.prev_actions[i],                               # son eylem
            self.foot_contacts(d) if contacts is None else contacts,
            [math.sin(angle), math.cos(angle)],                 # adım saati
        ]).astype(np.float32)

    def observe_all(self) -> np.ndarray:
        return np.stack([self._observe(d, i) for i, d in enumerate(self.datas)])

    def snapshot(self) -> np.ndarray:
        """Çizim için tüm karakterlerin konumları (N, nq)."""
        return np.stack([d.qpos.copy() for d in self.datas])

    def positions(self) -> np.ndarray:
        return np.array([d.qpos[0] for d in self.datas])
