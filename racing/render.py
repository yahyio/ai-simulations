"""Pist ve arabaların çizimi: dünya koordinatlarından ekran görünümüne."""
from __future__ import annotations

import math
from collections import deque

import numpy as np
import pygame

from . import theme
from .env import RaceEnv
from .track import GRID, Track, box_sum, erode, shift

GRASS = np.array((34, 70, 42))
GRASS_STRIPE = 7
GRASS_STRIPE_PX = 60
RUNOFF = np.array((26, 50, 32))
ASPHALT = np.array((54, 56, 62))
EDGE_LINE = (226, 229, 234)
KERB_RED = (206, 44, 44)
KERB_WHITE = (238, 238, 238)
EDGE_LINE_PX = 2
KERB_PX = 6
RUNOFF_PX = 4
KERB_STRIPE_PX = 12
KERB_BLUR_CELLS = 16
KERB_CORNER_SHARE = 25      # pistin en kıvrımlı % kaçı viraj sayılsın
KERB_MAX_STRAIGHTNESS = 0.995  # bundan düz bölgeler asla viraj sayılmaz
CHECKER_PX = 5
TRAIL_LENGTH = 240
ANGLE_BUCKETS = 180
SUPERSAMPLE = 4
GHOST_ALPHA = 80

SENSORS_OFF, SENSORS_LEADER, SENSORS_ALL = 0, 1, 2
SENSOR_MODE_LABELS = ("kapalı", "lider", "tümü")


def _erode_n(mask: np.ndarray, n: int) -> np.ndarray:
    for _ in range(n):
        mask = erode(mask)
    return mask


def _dilate_n(mask: np.ndarray, n: int) -> np.ndarray:
    for _ in range(n):
        grown = mask.copy()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            grown |= shift(mask, dy, dx, False)
        mask = grown
    return mask


def build_track_image(track: Track, pad_y: int = 0, pad_x: int = 0) -> np.ndarray:
    """Pist maskesinden dekoratif bir pist görseli üretir; kenarlara çim payı eklenebilir.

    Pay değerleri GRID'in katı olmalıdır (ilerleme/yön alanlarıyla hizalı kalsın).
    """
    gpy, gpx = pad_y // GRID, pad_x // GRID
    mask = np.pad(track.mask, ((pad_y, pad_y), (pad_x, pad_x)))
    start = np.pad(track.start_line_mask, ((pad_y, pad_y), (pad_x, pad_x)))
    progress = np.pad(track.progress, ((gpy, gpy), (gpx, gpx)), constant_values=-1.0)
    direction = np.pad(track.direction, ((gpy, gpy), (gpx, gpx), (0, 0)))
    h, w = mask.shape
    rng = np.random.default_rng(7)
    grain = rng.normal(0.0, 1.0, (h, w, 1))
    yy, xx = np.mgrid[0:h, 0:w]

    stripes = (((xx + yy) // GRASS_STRIPE_PX) % 2)[..., None]
    grass = GRASS + stripes * GRASS_STRIPE + grain * 3
    runoff_zone = (_dilate_n(mask, RUNOFF_PX) & ~mask)[..., None]
    grass = np.where(runoff_zone, RUNOFF + grain * 3, grass)
    img = np.where(mask[..., None], ASPHALT + grain * 4, grass)

    gy = np.clip(yy // GRID, 0, progress.shape[0] - 1)
    gx = np.clip(xx // GRID, 0, progress.shape[1] - 1)

    # Virajlar: pist yönü yakın çevrede ne kadar değişiyorsa ortalama yön vektörü o kadar kısalır.
    valid = np.abs(direction).sum(axis=2) > 0
    count = np.maximum(box_sum(valid.astype(np.float64), KERB_BLUR_CELLS), 1e-9)
    mean_x = box_sum(direction[..., 0].astype(np.float64), KERB_BLUR_CELLS) / count
    mean_y = box_sum(direction[..., 1].astype(np.float64), KERB_BLUR_CELLS) / count
    straightness = np.hypot(mean_x, mean_y)
    threshold = min(float(np.percentile(straightness[valid], KERB_CORNER_SHARE)), KERB_MAX_STRAIGHTNESS)
    corner = (straightness < threshold)[gy, gx]

    edge_line = mask & ~_erode_n(mask, EDGE_LINE_PX)
    kerb_zone = mask & ~_erode_n(mask, KERB_PX) & corner
    img[edge_line] = EDGE_LINE
    stripe = ((progress[gy, gx] // KERB_STRIPE_PX) % 2).astype(bool)
    img[kerb_zone & stripe] = KERB_RED
    img[kerb_zone & ~stripe] = KERB_WHITE

    checker = ((xx // CHECKER_PX + yy // CHECKER_PX) % 2).astype(bool)
    img[start & checker] = (245, 245, 245)
    img[start & ~checker] = (20, 20, 20)
    return np.clip(img, 0, 255).astype(np.uint8)


def make_car_sprite(color, length: int, width: int) -> pygame.Surface:
    """+x yönüne bakan, kenarları yumuşatılmış üstten görünüş F1 arabası."""
    L, W = max(length, 4) * SUPERSAMPLE, max(width, 3) * SUPERSAMPLE
    surf = pygame.Surface((L, W), pygame.SRCALPHA)
    dark = theme.lerp_color(color, (0, 0, 0), 0.45)
    tyre = (16, 16, 18)
    wl, ww = L * 0.2, W * 0.26
    for fx in (0.1, 0.66):
        for fy in (0.0, W - ww):
            pygame.draw.rect(surf, tyre, (fx * L, fy, wl, ww), border_radius=int(ww / 3))
    pygame.draw.rect(surf, dark, (0, W * 0.1, L * 0.1, W * 0.8), border_radius=2)
    body = [(L * 0.06, W * 0.32), (L * 0.6, W * 0.24), (L * 0.97, W * 0.43),
            (L * 0.97, W * 0.57), (L * 0.6, W * 0.76), (L * 0.06, W * 0.68)]
    pygame.draw.polygon(surf, color, body)
    pygame.draw.ellipse(surf, color, (L * 0.28, W * 0.2, L * 0.32, W * 0.6))
    pygame.draw.rect(surf, dark, (L * 0.87, W * 0.04, L * 0.08, W * 0.92), border_radius=2)
    pygame.draw.ellipse(surf, (22, 22, 26), (L * 0.4, W * 0.36, L * 0.2, W * 0.28))
    pygame.draw.circle(surf, (245, 245, 245), (L * 0.5, W * 0.5), W * 0.1)
    return pygame.transform.smoothscale(surf, (max(length, 4), max(width, 3)))


class Renderer:
    """Pisti bir kez hazırlar; her karede arabaları, izleri ve sensörleri çizer."""

    def __init__(self, track: Track, viewport: pygame.Rect, car_length: float, car_width: float):
        self.track = track
        self.viewport = viewport
        self.scale = min(viewport.w / track.width, viewport.h / track.height)
        # Görüntü alanının pistten artan kısmını çimle doldur (siyah şerit kalmasın).
        pad_x = int(max(0.0, (viewport.w / self.scale - track.width) / 2)) // GRID * GRID
        pad_y = int(max(0.0, (viewport.h / self.scale - track.height) / 2)) // GRID * GRID
        image = build_track_image(track, pad_y, pad_x)
        surface = pygame.surfarray.make_surface(image.transpose(1, 0, 2))
        size = (round(image.shape[1] * self.scale), round(image.shape[0] * self.scale))
        self.background = pygame.transform.smoothscale(surface, size).convert()
        self._background_pos = (viewport.x + (viewport.w - size[0]) / 2, viewport.y + (viewport.h - size[1]) / 2)
        self.origin = (self._background_pos[0] + pad_x * self.scale, self._background_pos[1] + pad_y * self.scale)
        self.sprite_size = (round(car_length * self.scale), round(car_width * self.scale))
        self._sprites: dict[tuple[int, bool], pygame.Surface] = {}
        self._rotated: dict[tuple[int, bool, int], pygame.Surface] = {}
        self._trail: deque = deque(maxlen=TRAIL_LENGTH)
        self._trail_car = -1
        self._last_step = 0
        self._cover: pygame.Surface | None = None

    def to_screen(self, x: float, y: float) -> tuple[float, float]:
        return self.origin[0] + x * self.scale, self.origin[1] + y * self.scale

    def draw_background(self, screen: pygame.Surface) -> None:
        screen.blit(self.background, self._background_pos)

    def draw_cover(self, screen: pygame.Surface) -> None:
        """Pisti tüm ekranı kaplayacak şekilde (kırparak) çizer — menü arka planı."""
        size = screen.get_size()
        if self._cover is None or self._cover.get_size() != size:
            src_w, src_h = self.background.get_size()
            scale = max(size[0] / src_w, size[1] / src_h)
            scaled = pygame.transform.smoothscale(
                self.background, (max(size[0], round(src_w * scale)), max(size[1], round(src_h * scale)))
            )
            crop = pygame.Rect(0, 0, *size)
            crop.center = scaled.get_rect().center
            self._cover = scaled.subsurface(crop).copy()
        screen.blit(self._cover, (0, 0))

    # ------------------------------------------------------------------ #
    def draw(self, screen: pygame.Surface, env: RaceEnv, sensor_mode: int, show_trail: bool, time_s: float) -> None:
        self.draw_background(screen)
        leader = env.leader()
        self._update_trail(env, leader)
        if show_trail:
            self._draw_trail(screen, theme.CAR_COLORS[leader % len(theme.CAR_COLORS)])

        running = env.running
        if sensor_mode != SENSORS_OFF:
            ends = env.sensor_endpoints()
            targets = np.flatnonzero(running) if sensor_mode == SENSORS_ALL else [leader]
            for i in targets:
                self._draw_sensors(screen, env.x[i], env.y[i], ends[i], env.sensors[i])

        # Önce elenenler (soluk), sonra yarışanlar, en üstte lider.
        order = sorted(range(env.num_cars), key=lambda i: (bool(running[i]), i == leader))
        for i in order:
            self._draw_car(screen, i, env.x[i], env.y[i], env.heading[i], ghost=not running[i])

        if running[leader]:
            cx, cy = self.to_screen(env.x[leader], env.y[leader])
            pulse = 0.5 + 0.5 * math.sin(time_s * 6.0)
            radius = self.sprite_size[0] * (0.75 + 0.15 * pulse)
            pygame.draw.circle(screen, theme.ACCENT, (cx, cy), radius, width=2)

    def _update_trail(self, env: RaceEnv, leader: int) -> None:
        if env.steps < self._last_step or leader != self._trail_car:
            self._trail.clear()
            self._trail_car = leader
        self._last_step = env.steps
        if env.running[leader]:
            self._trail.append(self.to_screen(env.x[leader], env.y[leader]))

    def _draw_trail(self, screen: pygame.Surface, color) -> None:
        points = list(self._trail)
        count = len(points)
        for k in range(1, count):
            shade = theme.lerp_color((40, 60, 45), color, k / count)
            pygame.draw.line(screen, shade, points[k - 1], points[k], 2)

    def _draw_sensors(self, screen, x, y, ends, readings) -> None:
        start = self.to_screen(x, y)
        for (ex, ey), reading in zip(ends, readings):
            color = theme.lerp_color(theme.BAD, theme.GOOD, float(reading))
            end = self.to_screen(ex, ey)
            pygame.draw.aaline(screen, color, start, end)
            pygame.draw.circle(screen, color, end, 3)

    def _sprite(self, index: int, ghost: bool, heading: float) -> pygame.Surface:
        color_idx = index % len(theme.CAR_COLORS)
        bucket = int(round(math.degrees(heading) / (360 / ANGLE_BUCKETS))) % ANGLE_BUCKETS
        key = (color_idx, ghost, bucket)
        if key not in self._rotated:
            base_key = (color_idx, ghost)
            if base_key not in self._sprites:
                sprite = make_car_sprite(theme.CAR_COLORS[color_idx], *self.sprite_size)
                if ghost:
                    sprite.fill((255, 255, 255, GHOST_ALPHA), special_flags=pygame.BLEND_RGBA_MULT)
                self._sprites[base_key] = sprite
            self._rotated[key] = pygame.transform.rotate(self._sprites[base_key], -bucket * (360 / ANGLE_BUCKETS))
        return self._rotated[key]

    def _draw_car(self, screen, index: int, x: float, y: float, heading: float, ghost: bool) -> None:
        sprite = self._sprite(index, ghost, heading)
        screen.blit(sprite, sprite.get_rect(center=self.to_screen(x, y)))
