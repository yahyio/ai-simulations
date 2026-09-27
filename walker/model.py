"""Düzlemsel (2B, x-z) hümanoid: gövde, kafa, bel, iki kol, iki bacak — 11 motorlu eklem.

Model MuJoCo MJCF olarak üretilir. Uzuvlar yalnızca zeminle çarpışır (düzlemsel
modelde iki bacak aynı düzlemde olduğundan birbirinden geçebilmeleri gerekir).
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco

PELVIS_HEIGHT = 1.0      # başlangıçta leğen kemiğinin yerden yüksekliği (m)
TIMESTEP = 0.004         # fizik adımı (s)


@dataclass(frozen=True)
class Motor:
    joint: str
    gear: float          # en büyük tork (N·m) = gear × kontrol (kontrol ∈ [-1, 1])
    range_deg: tuple[float, float]


# Eklem ekseni (0, -1, 0): pozitif açı uzvu öne doğru döndürür.
MOTORS = (
    Motor("abdomen", 100, (-35, 30)),
    Motor("hip_r", 150, (-30, 110)),
    Motor("knee_r", 150, (-150, 0)),
    Motor("ankle_r", 60, (-45, 45)),
    Motor("hip_l", 150, (-30, 110)),
    Motor("knee_l", 150, (-150, 0)),
    Motor("ankle_l", 60, (-45, 45)),
    Motor("shoulder_r", 30, (-60, 170)),
    Motor("elbow_r", 20, (0, 150)),
    Motor("shoulder_l", 30, (-60, 170)),
    Motor("elbow_l", 20, (0, 150)),
)
FOOT_GEOMS = ("foot_r", "foot_l")


def _leg(side: str) -> str:
    return f"""
      <body name="thigh_{side}" pos="0 0 -0.02">
        <joint name="hip_{side}"/>
        <geom name="thigh_{side}" type="capsule" fromto="0 0 0 0 0 -0.44" size="0.06"/>
        <body name="shin_{side}" pos="0 0 -0.44">
          <joint name="knee_{side}"/>
          <geom name="shin_{side}" type="capsule" fromto="0 0 0 0 0 -0.44" size="0.05"/>
          <body name="foot_{side}" pos="0 0 -0.44">
            <joint name="ankle_{side}"/>
            <geom name="foot_{side}" type="capsule" fromto="-0.06 0 -0.04 0.16 0 -0.04" size="0.045"
                  friction="1.2 0.1 0.1"/>
          </body>
        </body>
      </body>"""


def _arm(side: str) -> str:
    return f"""
        <body name="upper_arm_{side}" pos="0 0 0.40">
          <joint name="shoulder_{side}"/>
          <geom name="upper_arm_{side}" type="capsule" fromto="0 0 0 0 0 -0.28" size="0.04"/>
          <body name="forearm_{side}" pos="0 0 -0.28">
            <joint name="elbow_{side}"/>
            <geom name="forearm_{side}" type="capsule" fromto="0 0 0 0 0 -0.27" size="0.035"/>
          </body>
        </body>"""


def build_xml() -> str:
    ranges = "\n".join(
        f'    <default class="{m.joint}"><joint range="{m.range_deg[0]} {m.range_deg[1]}"/></default>'
        for m in MOTORS
    )
    motors = "\n".join(f'    <motor joint="{m.joint}" gear="{m.gear}"/>' for m in MOTORS)
    xml = f"""
<mujoco model="duzlemsel_humanoid">
  <compiler angle="degree" inertiafromgeom="true"/>
  <option timestep="{TIMESTEP}" integrator="implicitfast"/>
  <default>
    <joint type="hinge" axis="0 -1 0" limited="true" damping="1" armature="0.02"/>
    <geom contype="1" conaffinity="0" condim="3" friction="0.9 0.1 0.1" density="1000"/>
    <motor ctrllimited="true" ctrlrange="-1 1"/>
{ranges}
  </default>
  <worldbody>
    <geom name="floor" type="plane" size="1000 2 0.1" contype="0" conaffinity="1" friction="1 0.1 0.1"/>
    <body name="pelvis" pos="0 0 {PELVIS_HEIGHT}">
      <joint name="rootx" type="slide" axis="1 0 0" limited="false" damping="0" armature="0"/>
      <joint name="rootz" type="slide" axis="0 0 1" limited="false" damping="0" armature="0"/>
      <joint name="rooty" type="hinge" axis="0 1 0" limited="false" damping="0" armature="0"/>
      <geom name="pelvis" type="capsule" fromto="0 0 -0.02 0 0 0.08" size="0.1"/>
      <body name="chest" pos="0 0 0.1">
        <joint name="abdomen"/>
        <geom name="chest" type="capsule" fromto="0 0 0.02 0 0 0.36" size="0.11"/>
        <geom name="head" type="sphere" pos="0 0 0.58" size="0.1"/>
{_arm("l")}
{_arm("r")}
      </body>
{_leg("l")}
{_leg("r")}
    </body>
  </worldbody>
  <actuator>
{motors}
  </actuator>
</mujoco>"""
    # Eklemlere sınıflarını (hareket aralıklarını) ata.
    for m in MOTORS:
        xml = xml.replace(f'<joint name="{m.joint}"/>', f'<joint name="{m.joint}" class="{m.joint}"/>')
    return xml


def load_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(build_xml())
