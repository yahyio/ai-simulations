"""Yandan görünüm: gökyüzü, paralaks tepeler, metre işaretli zemin ve hümanoidler."""
from __future__ import annotations

import math

import mujoco
import numpy as np
import pygame

from racing import theme

PIXELS_PER_METER = 170
GROUND_FROM_BOTTOM = 150
CAMERA_ANCHOR = 0.38        # lider ekranın solundan bu oranda durur
CAMERA_SMOOTHING = 6.0      # 1/s
CAMERA_SNAP_M = 3.0         # bundan büyük sıçramada kamera yumuşatmadan atlar
GHOST_ALPHA = 55
GRASS_PX = 26

SKY_TOP = (22, 30, 52)
SKY_BOTTOM = (86, 118, 160)
HILL_FAR = (44, 62, 78)
HILL_NEAR = (38, 70, 64)
GRASS = (46, 98, 58)
GRASS_DARK = (38, 84, 50)
SOIL = (58, 48, 40)
SKIN = (236, 196, 160)
SHORTS = (38, 48, 74)
SHOE = (28, 28, 32)
GHOST = (220, 228, 240)
BACK_SHADE = 0.3

SHIRTS = {"yurume": (70, 150, 240), "kosu": (240, 120, 60)}

# Çizim sırası: arkadaki (sol) uzuvlar → gövde → öndeki (sağ) uzuvlar
DRAW_ORDER = (
    "upper_arm_l", "forearm_l", "thigh_l", "shin_l", "foot_l",
    "pelvis", "chest", "head",
    "thigh_r", "shin_r", "foot_r", "upper_arm_r", "forearm_r",
)
PART_BASE_COLORS = {
    "head": SKIN, "pelvis": SHORTS, "thigh": SHORTS, "shin": SKIN, "foot": SHOE, "forearm": SKIN,
}


def part_color(name: str, shirt) -> tuple[int, int, int]:
    back = name.endswith("_l")
    base = name[:-2] if name.endswith(("_l", "_r")) else name
    color = PART_BASE_COLORS.get(base, shirt)  # gövde ve üst kol: tişört
    return theme.lerp_color(color, (0, 0, 0), BACK_SHADE) if back else color


class BodyGeometry:
    """Modelin çizilecek geom'ları (kapsül uç noktaları / küre merkezleri) için indeksler."""

    def __init__(self, model: mujoco.MjModel):
        self.ids = [model.geom(name).id for name in DRAW_ORDER]
        self.names = DRAW_ORDER
        self.radius = model.geom_size[self.ids, 0].copy()
        self.half = model.geom_size[self.ids, 1].copy()
        self.is_sphere = model.geom_type[self.ids] == mujoco.mjtGeom.mjGEOM_SPHERE
        self.head = DRAW_ORDER.index("head")
        self.chest_id = model.geom("chest").id

    def segments(self, d: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        """(uç1, uç2) — x-z düzleminde, metre. Kürelerde iki uç aynıdır."""
        pos = d.geom_xpos[self.ids][:, [0, 2]]
        axis = d.geom_xmat[self.ids].reshape(-1, 3, 3)[:, :, 2][:, [0, 2]]
        offset = axis * np.where(self.is_sphere, 0.0, self.half)[:, None]
        return pos - offset, pos + offset

    def eye(self, d: mujoco.MjData) -> np.ndarray:
        """Gözün dünya konumu: kafanın, gövdenin baktığı yöne (+x) doğru ön-üst kısmı."""
        head = d.geom_xpos[self.ids[self.head]][[0, 2]]
        up = head - d.geom_xpos[self.chest_id][[0, 2]]   # gövdeden kafaya: kapsül ekseni kuralından bağımsız
        up = up / max(float(np.linalg.norm(up)), 1e-9)
        forward = np.array([up[1], -up[0]])
        return head + (forward * 0.55 + up * 0.2) * self.radius[self.head]


class Scene:
    def __init__(self, viewport: pygame.Rect, model: mujoco.MjModel, shirt):
        self.viewport = viewport
        self.geometry = BodyGeometry(model)
        self.shirt = shirt
        self.ground_y = viewport.bottom - GROUND_FROM_BOTTOM
        self.camera_x = 0.0
        self.sky = self._make_sky()
        self.layer = pygame.Surface(viewport.size, pygame.SRCALPHA)
        rng = np.random.default_rng(3)
        self.hills = [(0.15, 170, HILL_FAR, rng.uniform(0, 6, 4)), (0.35, 110, HILL_NEAR, rng.uniform(0, 6, 4))]

    def _make_sky(self) -> pygame.Surface:
        surf = pygame.Surface(self.viewport.size)
        h = self.viewport.h
        for y in range(h):
            pygame.draw.line(surf, theme.lerp_color(SKY_TOP, SKY_BOTTOM, y / h), (0, y), (self.viewport.w, y))
        return surf.convert()

    # ------------------------------------------------------------------ #
    def follow(self, target_x: float, dt: float) -> None:
        if abs(target_x - self.camera_x) > CAMERA_SNAP_M:
            self.camera_x = target_x
        else:
            self.camera_x += (target_x - self.camera_x) * min(1.0, CAMERA_SMOOTHING * dt)

    def to_screen(self, x: float, z: float) -> tuple[float, float]:
        vp = self.viewport
        return vp.x + vp.w * CAMERA_ANCHOR + (x - self.camera_x) * PIXELS_PER_METER, self.ground_y - z * PIXELS_PER_METER

    def draw_world(self, screen: pygame.Surface, record_m: float | None = None) -> None:
        vp = self.viewport
        screen.blit(self.sky, vp.topleft)
        for parallax, height, color, phases in self.hills:
            points = [(vp.x, self.ground_y)]
            for sx in range(0, vp.w + 20, 20):
                u = (sx / PIXELS_PER_METER + self.camera_x * parallax) * 0.35
                h = sum(math.sin(u * (k + 1) * 0.7 + p) / (k + 1) for k, p in enumerate(phases))
                points.append((vp.x + sx, self.ground_y - height * (0.55 + 0.25 * h)))
            points.append((vp.right, self.ground_y))
            pygame.draw.polygon(screen, color, points)

        pygame.draw.rect(screen, GRASS, (vp.x, self.ground_y, vp.w, GRASS_PX))
        pygame.draw.rect(screen, SOIL, (vp.x, self.ground_y + GRASS_PX, vp.w, vp.bottom - self.ground_y - GRASS_PX))
        left = self.camera_x - vp.w * CAMERA_ANCHOR / PIXELS_PER_METER
        right = left + vp.w / PIXELS_PER_METER
        clip = screen.get_clip()
        screen.set_clip(vp)
        for m in range(math.floor(left) - 1, math.ceil(right) + 1):
            x, _ = self.to_screen(m, 0)
            if m % 2 == 0:
                pygame.draw.rect(screen, GRASS_DARK, (x, self.ground_y, PIXELS_PER_METER, GRASS_PX))
            major = m % 5 == 0
            pygame.draw.line(screen, (230, 230, 220), (x, self.ground_y + GRASS_PX),
                             (x, self.ground_y + (46 if major else 34)), 2)
            if major:
                theme.text(screen, f"{m} m", (x, self.ground_y + 52), 15, (230, 225, 210), bold=True, anchor="midtop")
        if record_m is not None and record_m > 0.5:
            self._flag(screen, record_m, f"REKOR {record_m:.1f} m")
        screen.set_clip(clip)

    def _flag(self, screen, x_m: float, label: str) -> None:
        x, y = self.to_screen(x_m, 0)
        pygame.draw.line(screen, (240, 240, 240), (x, y), (x, y - 190), 3)
        pygame.draw.polygon(screen, theme.ACCENT, [(x, y - 190), (x + 46, y - 176), (x, y - 162)])
        theme.text(screen, label, (x + 6, y - 216), 13, theme.ACCENT, bold=True)

    # ------------------------------------------------------------------ #
    def draw_ghosts(self, screen: pygame.Surface, datas: list[mujoco.MjData], skip: int) -> None:
        self.layer.fill((0, 0, 0, 0))
        offset = (-self.viewport.x, -self.viewport.y)
        for i, d in enumerate(datas):
            if i != skip:
                self._draw_body(self.layer, d, lambda _name: GHOST, offset)
        self.layer.set_alpha(GHOST_ALPHA)
        screen.blit(self.layer, self.viewport.topleft)

    def draw_character(self, screen: pygame.Surface, d: mujoco.MjData) -> None:
        sx, sy = self.to_screen(d.qpos[0], 0)
        shadow = pygame.Rect(0, 0, 90, 14)
        shadow.center = (sx, sy + 2)
        pygame.draw.ellipse(screen, (24, 60, 34), shadow)
        self._draw_body(screen, d, lambda name: part_color(name, self.shirt), (0, 0))
        eye = self.to_screen(*self.geometry.eye(d))
        pygame.draw.circle(screen, (30, 30, 36), eye, 3)

    def _draw_body(self, surface, d: mujoco.MjData, color_of, offset) -> None:
        geo = self.geometry
        p1, p2 = geo.segments(d)
        ox, oy = offset
        for k, name in enumerate(geo.names):
            ax, ay = self.to_screen(*p1[k])
            bx, by = self.to_screen(*p2[k])
            a, b = (ax + ox, ay + oy), (bx + ox, by + oy)
            r = geo.radius[k] * PIXELS_PER_METER
            color = color_of(name)
            if not geo.is_sphere[k]:
                pygame.draw.line(surface, color, a, b, max(1, int(2 * r)))
                pygame.draw.circle(surface, color, b, r)
            pygame.draw.circle(surface, color, a, r)
