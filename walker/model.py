"""3B hümanoid (MuJoCo MJCF): 1.75 m, ~72 kg, 21 eklem, PD kontrollü eklem motorları.

İki model üretilir:
- fizik modeli: tek hümanoid (eğitim ortamlarında kullanılır, hızlıdır),
- görüntü modeli: atletizm pistinin kulvarlarında N hümanoid, gökyüzü ve gölgeler.
Her ikisi de aynı gövde tanımından üretildiği için eklem sırası birebir aynıdır;
fizik modelinin durumu görüntü modelindeki kulvara doğrudan kopyalanabilir.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import mujoco
import numpy as np

TIMESTEP = 0.004            # fizik adımı (s)
PELVIS_HEIGHT = 0.955       # ayakta dururken leğen kemiği yüksekliği (m)
LANE_WIDTH = 1.22           # atletizm kulvarı genişliği (m)
MAX_LANES = 8
TRACK_LENGTH = 400.0        # görsel pist uzunluğu (m)
FREE_QPOS = 7               # serbest kök ekleminin konum boyutu (x y z + dörtlü)
FREE_QVEL = 6

# Kulvar tişört renkleri (görüntü modeli)
SHIRT_COLORS = (
    (0.95, 0.55, 0.15), (0.20, 0.55, 0.95), (0.90, 0.25, 0.30), (0.25, 0.75, 0.45),
    (0.65, 0.40, 0.90), (0.95, 0.80, 0.20), (0.15, 0.75, 0.80), (0.95, 0.45, 0.70),
)


@dataclass(frozen=True)
class Joint:
    name: str
    axis: tuple[int, int, int]
    range_deg: tuple[float, float]
    kp: float        # PD yay sabiti (N·m/rad)
    kv: float        # PD sönüm (N·m·s/rad)
    force: float     # en büyük tork (N·m)
    default_deg: float = 0.0


def _leg(side: str) -> tuple[Joint, ...]:
    outward = (-35, 15) if side == "r" else (-15, 35)   # +x ekseni: sağ bacak için dışa = negatif
    return (
        Joint(f"hip_x_{side}", (1, 0, 0), outward, 300, 12, 200),
        Joint(f"hip_z_{side}", (0, 0, 1), (-30, 30), 150, 6, 120),
        Joint(f"hip_y_{side}", (0, 1, 0), (-120, 30), 350, 15, 250, -4.6),
        Joint(f"knee_{side}", (0, 1, 0), (-2, 150), 350, 12, 250, 8.6),
        # İnsan ayak bileği ~150-200 N·m üretebilir; 70 kg'lık gövdeyi dengelemek buna muhtaç.
        Joint(f"ankle_y_{side}", (0, 1, 0), (-40, 50), 150, 4, 150, -4.0),
        Joint(f"ankle_x_{side}", (1, 0, 0), (-25, 25), 80, 3, 80),
    )


def _arm(side: str) -> tuple[Joint, ...]:
    outward = (-120, 10) if side == "r" else (-10, 120)
    lean = -6.0 if side == "r" else 6.0
    return (
        Joint(f"shoulder_y_{side}", (0, 1, 0), (-150, 60), 60, 2, 60),
        Joint(f"shoulder_x_{side}", (1, 0, 0), outward, 60, 2, 60, lean),
        Joint(f"elbow_{side}", (0, 1, 0), (-150, 0), 40, 1, 40, -12.0),
    )


# Eylem sırası: bel (3) · sağ bacak (6) · sol bacak (6) · sağ kol (3) · sol kol (3) = 21
JOINTS: tuple[Joint, ...] = (
    Joint("abdomen_z", (0, 0, 1), (-45, 45), 150, 6, 120),
    Joint("abdomen_y", (0, 1, 0), (-30, 60), 300, 10, 200),
    Joint("abdomen_x", (1, 0, 0), (-35, 35), 300, 10, 200),
    *_leg("r"), *_leg("l"), *_arm("r"), *_arm("l"),
)
NUM_JOINTS = len(JOINTS)
DEFAULT_POSE = np.radians([j.default_deg for j in JOINTS])
FOOT_GEOMS = ("foot_r", "foot_l")


def _j(prefix: str, name: str) -> str:
    spec = next(j for j in JOINTS if j.name == name)
    axis = " ".join(str(a) for a in spec.axis)
    return f'<joint name="{prefix}{name}" axis="{axis}" range="{spec.range_deg[0]} {spec.range_deg[1]}"/>'


def humanoid_xml(prefix: str = "", pos: tuple[float, float] = (0.0, 0.0), shirt: str = "shirt0") -> str:
    """Tek bir hümanoidin gövde ağacı. `prefix` çoklu kopyalarda isim çakışmasını önler."""
    p = prefix

    def arm(side: str, sign: int) -> str:
        return f"""
        <body name="{p}upper_arm_{side}" pos="0 {sign * 0.23:.3f} 0.39">
          {_j(p, f"shoulder_y_{side}")}
          {_j(p, f"shoulder_x_{side}")}
          <geom name="{p}upper_arm_{side}" class="upper" type="capsule" fromto="0 0 0 0 0 -0.26" size="0.045" material="{shirt}"/>
          <body name="{p}forearm_{side}" pos="0 0 -0.28">
            {_j(p, f"elbow_{side}")}
            <geom name="{p}forearm_{side}" class="upper" type="capsule" fromto="0 0 0 0 0 -0.24" size="0.037" material="skin"/>
            <geom name="{p}hand_{side}" class="upper" type="sphere" pos="0 0 -0.28" size="0.045" material="skin"/>
          </body>
        </body>"""

    def leg(side: str, sign: int) -> str:
        return f"""
      <body name="{p}thigh_{side}" pos="0 {sign * 0.09:.3f} -0.04">
        {_j(p, f"hip_x_{side}")}
        {_j(p, f"hip_z_{side}")}
        {_j(p, f"hip_y_{side}")}
        <geom name="{p}thigh_{side}" class="leg" type="capsule" fromto="0 0 0 0 0 -0.38" size="0.065" material="skin"/>
        <geom name="{p}shorts_{side}" class="visual" type="capsule" fromto="0 0 0.02 0 0 -0.15" size="0.073" material="shorts"/>
        <body name="{p}shin_{side}" pos="0 0 -0.42">
          {_j(p, f"knee_{side}")}
          <geom name="{p}shin_{side}" class="leg" type="capsule" fromto="0 0 0 0 0 -0.37" size="0.05" material="skin"/>
          <geom name="{p}sock_{side}" class="visual" type="capsule" fromto="0 0 -0.28 0 0 -0.38" size="0.053" material="sock"/>
          <body name="{p}foot_{side}" pos="0 0 -0.42">
            {_j(p, f"ankle_y_{side}")}
            {_j(p, f"ankle_x_{side}")}
            <geom name="{p}foot_{side}" class="leg" type="box" pos="0.045 0 -0.045" size="0.11 0.045 0.025"
                  material="shoe" friction="1.2 0.02 0.01"/>
          </body>
        </body>
      </body>"""

    return f"""
    <body name="{p}pelvis" pos="{pos[0]:.3f} {pos[1]:.3f} {PELVIS_HEIGHT}">
      <freejoint name="{p}root"/>
      <geom name="{p}pelvis" class="upper" type="capsule" fromto="0 -0.08 0 0 0.08 0" size="0.1" material="shorts"/>
      <body name="{p}torso" pos="0 0 0.1">
        {_j(p, "abdomen_z")}
        {_j(p, "abdomen_y")}
        {_j(p, "abdomen_x")}
        <geom name="{p}chest" class="upper" type="ellipsoid" pos="0 0 0.21" size="0.11 0.16 0.22" material="{shirt}"/>
        <geom name="{p}shoulders" class="upper" type="capsule" fromto="0 -0.15 0.39 0 0.15 0.39" size="0.06" material="{shirt}"/>
        <geom name="{p}neck" class="upper" type="capsule" fromto="0 0 0.42 0 0 0.5" size="0.045" material="skin"/>
        <geom name="{p}head" class="upper" type="sphere" pos="0 0 0.6" size="0.1" material="skin"/>
        <geom name="{p}hair" class="visual" type="ellipsoid" pos="-0.02 0 0.635" size="0.095 0.098 0.075" material="hair"/>
        <geom name="{p}eye_r" class="visual" type="sphere" pos="0.088 -0.035 0.615" size="0.013" material="eye"/>
        <geom name="{p}eye_l" class="visual" type="sphere" pos="0.088 0.035 0.615" size="0.013" material="eye"/>
{arm("r", -1)}
{arm("l", 1)}
      </body>
{leg("r", -1)}
{leg("l", 1)}
    </body>"""


def _materials(n_shirts: int) -> str:
    shirts = "\n".join(
        f'    <material name="shirt{i}" rgba="{r} {g} {b} 1" specular="0.2" shininess="0.3"/>'
        for i, (r, g, b) in enumerate(SHIRT_COLORS[:n_shirts])
    )
    return f"""
    <material name="skin" rgba="0.93 0.75 0.62 1" specular="0.15" shininess="0.2"/>
    <material name="hair" rgba="0.18 0.12 0.08 1"/>
    <material name="eye" rgba="0.05 0.05 0.06 1"/>
    <material name="shorts" rgba="0.12 0.14 0.22 1"/>
    <material name="sock" rgba="0.95 0.95 0.95 1"/>
    <material name="shoe" rgba="0.12 0.12 0.14 1" specular="0.4" shininess="0.5"/>
{shirts}"""


# Çarpışma grupları: bacaklar hem zeminle hem birbirleriyle çarpışır (bacaklar iç içe
# geçemez), gövde ve kollar yalnızca zeminle. Görsel parçaların kütlesi ve çarpışması yok.
_DEFAULTS = """
  <default>
    <joint type="hinge" limited="true" damping="0.5" armature="0.01"/>
    <geom condim="3" friction="1 0.02 0.01" density="1000"/>
    <default class="leg"><geom contype="2" conaffinity="2"/></default>
    <default class="upper"><geom contype="4" conaffinity="0"/></default>
    <default class="visual"><geom contype="0" conaffinity="0" density="0" group="1"/></default>
    <default class="decor"><geom contype="0" conaffinity="0" density="0"/></default>
  </default>"""


def _actuators() -> str:
    return "\n".join(
        f'    <position name="{j.name}" joint="{j.name}" kp="{j.kp}" kv="{j.kv}" '
        f'forcelimited="true" forcerange="{-j.force} {j.force}"/>'
        for j in JOINTS
    )


def physics_xml() -> str:
    return f"""
<mujoco model="humanoid3d">
  <compiler angle="degree" inertiafromgeom="true"/>
  <option timestep="{TIMESTEP}" integrator="implicitfast"/>
{_DEFAULTS}
  <asset>{_materials(1)}
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.1" contype="1" conaffinity="15" rgba="0.3 0.5 0.3 1"/>
{humanoid_xml()}
  </worldbody>
  <actuator>
{_actuators()}
  </actuator>
</mujoco>"""


def _track_decor(lanes: int) -> str:
    """Atletizm pisti: kiremit rengi zemin, kulvar çizgileri, 10 m işaretleri, başlangıç çizgisi."""
    width = lanes * LANE_WIDTH
    cy = (lanes - 1) * LANE_WIDTH / 2
    x0, x1 = -30.0, TRACK_LENGTH
    cx, hx = (x0 + x1) / 2, (x1 - x0) / 2
    parts = [
        f'<geom class="decor" type="box" pos="{cx} {cy} 0.0008" size="{hx} {width / 2 + 0.35} 0.0006" material="track"/>'
    ]
    for k in range(lanes + 1):
        y = -LANE_WIDTH / 2 + k * LANE_WIDTH
        parts.append(f'<geom class="decor" type="box" pos="{cx} {y:.3f} 0.0016" size="{hx} 0.025 0.0002" material="line"/>')
    for m in range(10, int(TRACK_LENGTH), 10):
        parts.append(f'<geom class="decor" type="box" pos="{m} {cy} 0.0017" size="0.03 {width / 2} 0.0002" material="line"/>')
    parts.append(f'<geom class="decor" type="box" pos="0 {cy} 0.0018" size="0.06 {width / 2} 0.0002" material="start"/>')
    return "\n    ".join(parts)


def display_xml(lanes: int) -> str:
    """Görüntü modeli: `lanes` kulvarda hümanoidler, pist, gökyüzü ve gölgeli güneş."""
    lanes = max(1, min(lanes, MAX_LANES))
    bodies = "\n".join(
        humanoid_xml(prefix=f"a{k}_", pos=(0.0, k * LANE_WIDTH), shirt=f"shirt{k}") for k in range(lanes)
    )
    cy = (lanes - 1) * LANE_WIDTH / 2
    return f"""
<mujoco model="stadium">
  <compiler angle="degree" inertiafromgeom="true"/>
  <option timestep="{TIMESTEP}"/>
  <statistic extent="6" center="0 {cy} 1"/>
  <visual>
    <global offwidth="1920" offheight="1080" fovy="42"/>
    <quality shadowsize="4096" offsamples="4"/>
    <!-- shadowclip < 5: zeminde hareli gölge lekeleri (shadow acne) oluşuyordu -->
    <map znear="0.02" zfar="300" haze="0.35" shadowclip="6"/>
    <headlight ambient="0.38 0.38 0.42" diffuse="0.35 0.35 0.35" specular="0.05 0.05 0.05"/>
    <rgba haze="0.72 0.8 0.92 1"/>
  </visual>
{_DEFAULTS}
  <asset>{_materials(lanes)}
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.36 0.55 0.86" rgb2="0.88 0.92 0.98" width="512" height="512"/>
    <texture name="grass" type="2d" builtin="checker" rgb1="0.27 0.5 0.28" rgb2="0.24 0.46 0.25" width="512" height="512"
             mark="random" random="0.08" markrgb="0.32 0.56 0.32"/>
    <material name="grass" texture="grass" texrepeat="60 60" reflectance="0.02"/>
    <texture name="track" type="2d" builtin="flat" rgb1="0.72 0.31 0.23" width="256" height="256"
             mark="random" random="0.15" markrgb="0.64 0.27 0.2"/>
    <material name="track" texture="track" texrepeat="200 6" specular="0.05"/>
    <material name="line" rgba="0.97 0.97 0.97 1"/>
    <material name="start" rgba="1 1 1 1" emission="0.2"/>
  </asset>
  <worldbody>
    <light name="sun" directional="true" pos="0 {cy} 30" dir="-0.35 0.45 -1" diffuse="0.75 0.73 0.68"
           specular="0.2 0.2 0.2" castshadow="true" mode="fixed"/>
    <geom name="floor" type="plane" size="0 0 0.1" material="grass" contype="0" conaffinity="0"/>
    {_track_decor(lanes)}
{bodies}
  </worldbody>
</mujoco>"""


def load_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(physics_xml())


def load_display_model(lanes: int) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(display_xml(lanes))


def joint_ranges(model: mujoco.MjModel) -> np.ndarray:
    """Eylem sırasındaki eklemlerin (alt, üst) sınırları, radyan (NUM_JOINTS, 2)."""
    ids = [model.joint(j.name).id for j in JOINTS]
    return model.jnt_range[ids].copy()


def joint_addresses(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    """Eylem sırasındaki eklemlerin qpos ve qvel adresleri.

    Dikkat: qpos, XML gövde ağacının sırasını izler (bel → kollar → bacaklar); eylem
    sırası (JOINTS) farklıdır. Eklemlere her zaman bu adreslerle erişilmelidir.
    """
    ids = [model.joint(j.name).id for j in JOINTS]
    return model.jnt_qposadr[ids].copy(), model.jnt_dofadr[ids].copy()


@lru_cache(maxsize=1)
def _standing_qpos() -> tuple[float, ...]:
    model = load_model()
    qpos = np.zeros(model.nq)
    qpos[:FREE_QPOS] = (0.0, 0.0, PELVIS_HEIGHT, 1.0, 0.0, 0.0, 0.0)
    qpos[joint_addresses(model)[0]] = DEFAULT_POSE
    return tuple(qpos)


def standing_qpos() -> np.ndarray:
    """Ayakta duruş (qpos düzeninde): kök konumu + birim dörtlü + varsayılan eklem açıları."""
    return np.array(_standing_qpos())


def lane_y(lane: int) -> float:
    return lane * LANE_WIDTH


def yaw_of(quat: np.ndarray) -> float:
    """Dörtlüden (w, x, y, z) sapma açısı (z ekseni etrafında)."""
    w, x, y, z = quat
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
