"""3B sahne: atletizm pisti, kulvarlarda hümanoidler, gölgeli MuJoCo görüntüsü.

Eğitimdeki karakterlerin durumu (qpos) görüntü modelindeki kulvarlara kopyalanır ve
MuJoCo'nun ekran dışı çizicisiyle görüntülenir; çıkan kare pygame penceresine aktarılır.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np
import pygame

from common import theme

from .model import FREE_QPOS, LANE_WIDTH, NUM_JOINTS, load_display_model, standing_qpos

QPOS_PER_RUNNER = FREE_QPOS + NUM_JOINTS
CAMERA_SMOOTHING = 5.0          # 1/s
CAMERA_SNAP_M = 4.0             # bundan büyük sıçramada kamera yumuşatmadan atlar
ORBIT_DEG_PER_PX = 0.3
ZOOM_STEP = 0.9
DISTANCE_LIMITS = (1.5, 25.0)
ELEVATION_LIMITS = (-80.0, -2.0)
GOLD = np.array([1.0, 0.78, 0.1, 1.0])
LABEL_EVERY_M = 10


@dataclass(frozen=True)
class CameraPreset:
    name: str
    azimuth: float
    elevation: float
    distance: float


PRESETS = (
    # Varsayılan: önden çapraz. Yandan bakınca yakın kulvarlardaki koşucular öndekini örtüyor.
    CameraPreset("çapraz", 135.0, -14.0, 6.5),
    CameraPreset("yan", 100.0, -12.0, 5.0),
    CameraPreset("arkadan", 20.0, -14.0, 4.0),
    CameraPreset("önden", 200.0, -8.0, 4.5),
    CameraPreset("kuşbakışı", 90.0, -55.0, 11.0),
)


class Stadium:
    """Kulvarlarda koşan hümanoidleri çizen 3B sahne."""

    def __init__(self, size: tuple[int, int], lanes: int):
        self.width, self.height = size
        self.lanes = lanes
        self.model = load_display_model(lanes)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
        self.option = mujoco.MjvOption()
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.preset = 0
        self._apply_preset(PRESETS[0])
        self.camera.lookat[:] = (0.0, 0.0, 0.95)
        self.sun = self.model.light("sun").id
        self._sun_offset = self.model.light_pos[self.sun].copy()
        self.track_center_y = (lanes - 1) * LANE_WIDTH / 2
        self.track_half_width = lanes * LANE_WIDTH / 2
        self.tan_half_fovy = math.tan(math.radians(self.model.vis.global_.fovy) / 2)
        self.set_poses(np.tile(standing_qpos(), (lanes, 1)))

    # ------------------------------------------------------------------ #
    def _apply_preset(self, preset: CameraPreset) -> None:
        self.camera.azimuth = preset.azimuth
        self.camera.elevation = preset.elevation
        self.camera.distance = preset.distance

    def cycle_preset(self) -> str:
        self.preset = (self.preset + 1) % len(PRESETS)
        self._apply_preset(PRESETS[self.preset])
        return PRESETS[self.preset].name

    def orbit(self, dx: float, dy: float) -> None:
        self.camera.azimuth = (self.camera.azimuth - dx * ORBIT_DEG_PER_PX) % 360
        self.camera.elevation = float(np.clip(self.camera.elevation - dy * ORBIT_DEG_PER_PX, *ELEVATION_LIMITS))

    def zoom(self, steps: float) -> None:
        self.camera.distance = float(np.clip(self.camera.distance * ZOOM_STEP**steps, *DISTANCE_LIMITS))

    # ------------------------------------------------------------------ #
    def set_poses(self, qpos: np.ndarray) -> None:
        """Karakter durumlarını kulvarlara yerleştirir (ilk `lanes` karakter)."""
        count = min(len(qpos), self.lanes)
        for k in range(count):
            block = self.data.qpos[k * QPOS_PER_RUNNER:(k + 1) * QPOS_PER_RUNNER]
            block[:] = qpos[k]
            block[1] += k * LANE_WIDTH
        mujoco.mj_kinematics(self.model, self.data)

    def runner_position(self, lane: int) -> np.ndarray:
        """Kulvardaki karakterin leğen konumu (dünya, görüntü modeli)."""
        return self.data.qpos[lane * QPOS_PER_RUNNER:lane * QPOS_PER_RUNNER + 3].copy()

    def follow(self, lane: int, dt: float) -> None:
        target = self.runner_position(lane)
        target[2] = 0.95
        gap = float(np.linalg.norm(target - self.camera.lookat))
        blend = 1.0 if gap > CAMERA_SNAP_M else min(1.0, CAMERA_SMOOTHING * dt)
        self.camera.lookat[:] = self.camera.lookat + (target - self.camera.lookat) * blend

    # ------------------------------------------------------------------ #
    def render(self, record_m: float | None = None, marker_lane: int | None = None) -> pygame.Surface:
        # Güneş kamerayı izlesin ki gölgeler hep görüş alanında hesaplansın.
        self.model.light_pos[self.sun] = self._sun_offset + np.array([self.camera.lookat[0], 0.0, 0.0])
        mujoco.mj_camlight(self.model, self.data)
        self.renderer.update_scene(self.data, self.camera, self.option)
        scene = self.renderer.scene
        if record_m is not None and record_m > 0.5:
            self._add_geom(scene, mujoco.mjtGeom.mjGEOM_BOX, [0.05, self.track_half_width, 0.004],
                           [record_m, self.track_center_y, 0.004], np.eye(3), GOLD)
        if marker_lane is not None:
            pos = self.runner_position(marker_lane)
            down = np.diag([1.0, -1.0, -1.0])   # ok aşağıyı göstersin
            self._add_geom(scene, mujoco.mjtGeom.mjGEOM_ARROW, [0.035, 0.035, 0.25],
                           [pos[0], pos[1], 2.3], down, GOLD)
        pixels = self.renderer.render()
        return pygame.image.frombuffer(pixels.tobytes(), (self.width, self.height), "RGB")

    @staticmethod
    def _add_geom(scene: mujoco.MjvScene, kind, size, pos, mat, rgba) -> None:
        if scene.ngeom >= scene.maxgeom:
            return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, kind, np.asarray(size, float), np.asarray(pos, float),
                            np.asarray(mat, float).ravel(), np.asarray(rgba, np.float32))
        geom.emission = 0.35
        scene.ngeom += 1

    # ------------------------------------------------------------------ #
    def project(self, points: np.ndarray) -> list[tuple[float, float] | None]:
        """Dünya noktalarını (M, 3) son çizilen karenin piksel koordinatlarına çevirir."""
        cam = self.renderer.scene.camera[0]
        pos, forward, up = np.array(cam.pos), np.array(cam.forward), np.array(cam.up)
        right = np.cross(forward, up)
        rel = np.atleast_2d(points) - pos
        depth, xc, yc = rel @ forward, rel @ right, rel @ up
        aspect = self.width / self.height
        out: list[tuple[float, float] | None] = []
        for z, x, y in zip(depth, xc, yc):
            if z < 0.1:
                out.append(None)
                continue
            sx = (x / z / (self.tan_half_fovy * aspect) + 1) * 0.5 * self.width
            sy = (1 - y / z / self.tan_half_fovy) * 0.5 * self.height
            out.append((sx, sy))
        return out

    def draw_labels(self, surface: pygame.Surface, origin: tuple[int, int], record_m: float | None,
                    marker_lane: int | None, marker_text: str = "") -> None:
        """Metre işaretleri, rekor ve lider etiketleri (3B noktaların izdüşümüyle)."""
        focus = self.camera.lookat[0]
        near_side = -LANE_WIDTH / 2 - 0.45
        marks = [m for m in range(0, 400, LABEL_EVERY_M) if abs(m - focus) < self.camera.distance * 4 + 15]
        points = [(m, near_side, 0.0) for m in marks]
        labels = [f"{m} m" for m in marks]
        colors = [(245, 245, 240)] * len(marks)
        if record_m is not None and record_m > 0.5:
            points.append((record_m, self.track_center_y + self.track_half_width + 0.4, 0.3))
            labels.append(f"REKOR {record_m:.1f} m")
            colors.append(theme.ACCENT)
        if marker_lane is not None and marker_text:
            pos = self.runner_position(marker_lane)
            points.append((pos[0], pos[1], 2.62))
            labels.append(marker_text)
            colors.append(theme.ACCENT)
        for xy, label, color in zip(self.project(np.array(points, dtype=float)), labels, colors):
            if xy is None or not (0 <= xy[0] <= self.width and 0 <= xy[1] <= self.height):
                continue
            px, py = origin[0] + xy[0], origin[1] + xy[1]
            theme.text(surface, label, (px + 1, py + 1), 15, (20, 20, 24), bold=True, anchor="midbottom")
            theme.text(surface, label, (px, py), 15, color, bold=True, anchor="midbottom")

    def close(self) -> None:
        self.renderer.close()
