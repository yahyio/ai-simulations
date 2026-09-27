"""Pist analizi: sürülebilir alan, başlangıç pozu, ilerleme alanı ve yön alanı.

Pist görselinde koyu renk sürülebilir alanı, kırmızı çizgi başlangıç/bitişi gösterir.
Elle checkpoint çizmeye gerek yoktur: başlangıç çizgisinden itibaren pist boyunca
jeodezik mesafe (8-komşulu Dijkstra) hesaplanır. Bu "ilerleme alanı" hem ödül
şekillendirmede hem de tur/ters yön tespitinde kullanılır.
"""
from __future__ import annotations

import heapq
import math
from pathlib import Path

import numpy as np
import pygame

GRID = 2  # İlerleme/yön alanlarının çözünürlüğü (piksel/hücre)
MIN_TRACK_PIXELS = 500
FILL_ITERATIONS = 4
DIRECTION_BLUR_RADIUS = 3
TANGENT_FALLBACK_WEIGHT = 0.05
SQRT2 = math.sqrt(2.0)
NEIGHBORS = (
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, SQRT2), (-1, 1, SQRT2), (1, -1, SQRT2), (1, 1, SQRT2),
)


class TrackError(RuntimeError):
    """Pist görseli yüklenemediğinde veya kullanılamaz olduğunda."""


# --------------------------------------------------------------------------- #
# Saf numpy yardımcıları (test edilebilir, pygame gerektirmez)
# --------------------------------------------------------------------------- #
def color_mask(rgb: np.ndarray, color, tolerance: float) -> np.ndarray:
    """RGB görüntüde `color` rengine `tolerance` mesafeden yakın pikseller."""
    diff = rgb[..., :3].astype(np.int32) - np.asarray(color, dtype=np.int32)
    return (diff * diff).sum(axis=2) <= tolerance * tolerance


def downsample_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    """Maskeyi `factor` katı küçültür (blokların çoğunluğu)."""
    h = mask.shape[0] // factor * factor
    w = mask.shape[1] // factor * factor
    blocks = mask[:h, :w].reshape(h // factor, factor, w // factor, factor)
    return blocks.mean(axis=(1, 3)) >= 0.5


def shift(a: np.ndarray, dy: int, dx: int, fill) -> np.ndarray:
    """out[y, x] = a[y + dy, x + dx]; dışarı taşan yerler `fill`."""
    out = np.full_like(a, fill)
    h, w = a.shape[:2]
    ys = slice(max(0, -dy), min(h, h - dy))
    xs = slice(max(0, -dx), min(w, w - dx))
    yd = slice(max(0, dy), min(h, h + dy))
    xd = slice(max(0, dx), min(w, w + dx))
    out[ys, xs] = a[yd, xd]
    return out


def box_sum(a: np.ndarray, radius: int) -> np.ndarray:
    """(2r+1)x(2r+1) pencere toplamı (kümülatif toplam ile)."""
    c = np.pad(a, radius + 1)[:-1, :-1].cumsum(axis=0).cumsum(axis=1)
    k = 2 * radius + 1
    return c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]


def geodesic_distance(passable: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    """`seeds` hücrelerinden 8-komşulu Dijkstra mesafesi; ulaşılamayan: inf."""
    h, w = passable.shape
    open_cells = passable.ravel().tolist()
    dist = [math.inf] * (h * w)
    heap = []
    for i in np.flatnonzero(seeds & passable).tolist():
        dist[i] = 0.0
        heap.append((0.0, i))
    heapq.heapify(heap)

    while heap:
        d, i = heapq.heappop(heap)
        if d > dist[i]:
            continue
        y, x = divmod(i, w)
        for dy, dx, cost in NEIGHBORS:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                j = ny * w + nx
                nd = d + cost
                if open_cells[j] and nd < dist[j]:
                    dist[j] = nd
                    heapq.heappush(heap, (nd, j))
    return np.asarray(dist, dtype=np.float64).reshape(h, w)


def principal_axis(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Nokta bulutunun en büyük varyans yönü (birim vektör)."""
    pts = np.stack([xs - xs.mean(), ys - ys.mean()]).astype(np.float64)
    vals, vecs = np.linalg.eigh(pts @ pts.T / max(len(xs), 1))
    axis = vecs[:, int(np.argmax(vals))]
    return axis / np.linalg.norm(axis)


def erode(mask: np.ndarray, neighbors: int = 4) -> np.ndarray:
    """4- veya 8-komşulu erozyon."""
    out = mask.copy()
    for dy, dx, _ in NEIGHBORS[:neighbors]:
        out &= shift(mask, dy, dx, False)
    return out


def edge_distance(mask: np.ndarray) -> np.ndarray:
    """Her hücrenin maske kenarına yaklaşık uzaklığı (4/8 komşu sırayla: sekizgen metrik)."""
    depth = np.zeros(mask.shape)
    current, k = mask, 0
    while current.any():
        depth += current
        current = erode(current, 4 if k % 2 == 0 else 8)
        k += 1
    return depth


def deepest_point(mask: np.ndarray) -> tuple[float, float, int]:
    """Kenarlara en uzak nokta (x, y) ve derinliği (erozyon adımı)."""
    current, last, depth = mask, mask, 0
    while current.any():
        last, current = current, erode(current)
        depth += 1
    ys, xs = np.nonzero(last)
    mid = len(xs) // 2
    return float(xs[mid]), float(ys[mid]), depth


def canonical_normal(n: np.ndarray) -> np.ndarray:
    """Normalin işaretini belirli bir yöne sabitler (+x, x≈0 ise +y)."""
    if n[0] < -1e-6 or (abs(n[0]) <= 1e-6 and n[1] < 0):
        return -n
    return n


def march_extent(mask: np.ndarray, cx: float, cy: float, direction: np.ndarray, limit: int = 4000) -> float:
    """(cx, cy)'den `direction` yönünde maskenin dışına çıkana kadar mesafe."""
    h, w = mask.shape
    for step in range(limit):
        x = int(cx + direction[0] * step * 0.5)
        y = int(cy + direction[1] * step * 0.5)
        if not (0 <= x < w and 0 <= y < h and mask[y, x]):
            return step * 0.5
    return limit * 0.5


def fill_unknown(values: np.ndarray, known: np.ndarray, iterations: int) -> tuple[np.ndarray, np.ndarray]:
    """Bilinmeyen hücreleri komşu bilinen değerlerin minimumu ile doldurur."""
    values = values.copy()
    known = known.copy()
    for _ in range(iterations):
        source = np.where(known, values, np.inf)
        best = np.full(values.shape, np.inf)
        for dy, dx, _ in NEIGHBORS:
            best = np.minimum(best, shift(source, dy, dx, np.inf))
        grow = ~known & np.isfinite(best)
        values[grow] = best[grow]
        known |= grow
    return values, known


def _blur_normalize(vx: np.ndarray, vy: np.ndarray, weight: np.ndarray, passes: int) -> np.ndarray:
    """Vektör alanını geçerli hücreler üzerinde yumuşatıp birim uzunluğa getirir."""
    for _ in range(passes):
        count = box_sum(weight, DIRECTION_BLUR_RADIUS)
        denom = np.maximum(count, 1e-9)
        vx = np.where(count > 0, box_sum(vx * weight, DIRECTION_BLUR_RADIUS) / denom, 0.0)
        vy = np.where(count > 0, box_sum(vy * weight, DIRECTION_BLUR_RADIUS) / denom, 0.0)
        weight = (count > 0).astype(np.float64)
    norm = np.hypot(vx, vy)
    scale = np.where(norm > 1e-9, 1.0 / np.maximum(norm, 1e-9), 0.0)
    return np.stack([vx * scale, vy * scale], axis=-1).astype(np.float32)


def progress_gradient(dist: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """İlerleme alanının yumuşatılmış gradyanı (h, w, 2).

    Viraj içlerine yapışan en kısa yolları izlediği için virajlarda pist teğetinden
    sapar; yalnızca yön işaretini belirlemek için kullanılır.
    """
    vx = np.zeros(dist.shape)
    vy = np.zeros(dist.shape)
    safe = np.where(valid, dist, 0.0)
    for dy, dx, cost in NEIGHBORS:
        neighbor = shift(safe, dy, dx, 0.0)
        neighbor_ok = shift(valid, dy, dx, False)
        diff = neighbor - safe
        ok = valid & neighbor_ok & (np.abs(diff) < 3.0 * cost)  # başlangıç dikişini atla
        vx += np.where(ok, diff * dx / cost**2, 0.0)
        vy += np.where(ok, diff * dy / cost**2, 0.0)
    return _blur_normalize(vx, vy, valid.astype(np.float64), passes=2)


def direction_field(dist: np.ndarray, valid: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Pistin gidiş yönü (h, w, 2): duvarlara paralel, ilerleme yönünde birim vektörler.

    Kenara uzaklık alanının gradyanı duvarlara diktir; 90° çevrilince duvarlara
    paralel olur. İşareti ilerleme gradyanına göre seçilir. Halka pistte teğetten
    ortalama ~2° sapar (yalnız ilerleme gradyanı ~37° sapıyordu).
    """
    reference = progress_gradient(dist, valid)
    gy, gx = np.gradient(edge_distance(mask))
    tx, ty = -gy, gx
    sign = np.where(tx * reference[..., 0] + ty * reference[..., 1] < 0, -1.0, 1.0)
    # Sırt çizgisinde (pist ortası) gradyan sıfıra yaklaşır; orada referans yön devralır.
    vx = tx * sign + TANGENT_FALLBACK_WEIGHT * reference[..., 0]
    vy = ty * sign + TANGENT_FALLBACK_WEIGHT * reference[..., 1]
    return _blur_normalize(vx, vy, valid.astype(np.float64), passes=1)


# --------------------------------------------------------------------------- #
# Pist
# --------------------------------------------------------------------------- #
class Track:
    """Bir pist görselinden türetilen tüm geometrik bilgiler."""

    def __init__(self, rgb: np.ndarray, track_cfg: dict, name: str = "track"):
        self.name = name
        self.height, self.width = rgb.shape[:2]
        tol = track_cfg["color_tolerance"]
        self.start_line_mask = color_mask(rgb, track_cfg["start_line_color"], tol)
        self.mask = color_mask(rgb, track_cfg["track_color"], tol) | self.start_line_mask
        if int(self.mask.sum()) < MIN_TRACK_PIXELS:
            raise TrackError(
                "Pistte sürülebilir alan bulunamadı. Pist koyu renkte, "
                "başlangıç çizgisi kırmızı olmalı (config.yaml > track)."
            )

        self.grid_mask = downsample_mask(self.mask, GRID)
        center, normal, along = self._find_start(bool(track_cfg.get("reverse", False)))
        self.start_center = center
        self.start_normal = normal
        self.start_heading = math.atan2(normal[1], normal[0])
        self._build_progress(center, normal, along)

    @classmethod
    def from_file(cls, path: str | Path, track_cfg: dict) -> "Track":
        path = Path(path)
        if not path.exists():
            raise TrackError(f"Pist dosyası bulunamadı: {path}")
        try:
            surface = pygame.image.load(str(path))
        except pygame.error as exc:
            raise TrackError(f"Pist görseli okunamadı ({path}): {exc}") from exc
        rgb = pygame.surfarray.array3d(surface).transpose(1, 0, 2)
        return cls(rgb, track_cfg, name=path.stem)

    # ------------------------------------------------------------------ #
    def _find_start(self, reverse: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Başlangıç çizgisinin merkezi, gidiş normali ve çizgi yönü (dünya px)."""
        ys, xs = np.nonzero(self.start_line_mask)
        if len(xs) >= 5:
            center = np.array([xs.mean(), ys.mean()])
            along = principal_axis(xs, ys)
            normal = np.array([-along[1], along[0]])
        else:
            # Kırmızı çizgi yok: pistin en geniş noktasını ve yerel yönünü kullan.
            gx, gy, depth = deepest_point(self.grid_mask)
            radius = max(3 * depth, 6)
            yy, xx = np.nonzero(self.grid_mask)
            near = (xx - gx) ** 2 + (yy - gy) ** 2 <= radius**2
            normal = principal_axis(xx[near].astype(float), yy[near].astype(float))
            along = np.array([-normal[1], normal[0]])
            center = np.array([gx, gy]) * GRID + GRID / 2

        normal = canonical_normal(normal)
        if reverse:
            normal = -normal
        return center, normal, along

    def _build_progress(self, center: np.ndarray, normal: np.ndarray, along: np.ndarray) -> None:
        grid = self.grid_mask
        h, w = grid.shape
        cx, cy = center / GRID
        half = max(march_extent(grid, cx, cy, along), march_extent(grid, cx, cy, -along)) + 2.0

        yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
        rx, ry = xx - cx, yy - cy
        along_n = rx * normal[0] + ry * normal[1]
        band = grid & (np.abs(rx * along[0] + ry * along[1]) <= half)
        wall = band & (np.abs(along_n) <= 1.0)
        seeds = band & (along_n > 1.0) & (along_n <= 2.5)
        behind = band & (along_n < -1.0) & (along_n >= -2.5)

        dist = geodesic_distance(grid & ~wall, seeds)
        closing = dist[behind & np.isfinite(dist)]
        if closing.size == 0:
            raise TrackError(
                "Pist kapalı bir döngü oluşturmuyor: başlangıç çizgisinden çıkıp "
                "arkasına dönen bir yol bulunamadı."
            )
        self.length = float(np.median(closing)) * GRID  # piksel

        reached = np.isfinite(dist)
        filled, known = fill_unknown(np.where(reached, dist, 0.0), reached, FILL_ITERATIONS)
        self.progress = np.where(known, filled * GRID, -1.0).astype(np.float32)
        self.direction = direction_field(dist, reached, grid)
        self.wall_mask = wall

    # ------------------------------------------------------------------ #
    # Vektörize sorgular (dünya koordinatları, piksel)
    # ------------------------------------------------------------------ #
    def on_track(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        xi = np.floor(xs).astype(np.int64)
        yi = np.floor(ys).astype(np.int64)
        inside = (xi >= 0) & (xi < self.width) & (yi >= 0) & (yi < self.height)
        result = np.zeros(xi.shape, dtype=bool)
        result[inside] = self.mask[yi[inside], xi[inside]]
        return result

    def _grid_index(self, xs, ys) -> tuple[np.ndarray, np.ndarray]:
        h, w = self.progress.shape
        gx = np.clip(np.floor(np.asarray(xs) / GRID).astype(np.int64), 0, w - 1)
        gy = np.clip(np.floor(np.asarray(ys) / GRID).astype(np.int64), 0, h - 1)
        return gy, gx

    def progress_at(self, xs, ys) -> np.ndarray:
        """Başlangıçtan itibaren pist boyunca mesafe (px); bilinmiyorsa -1."""
        return self.progress[self._grid_index(xs, ys)]

    def direction_at(self, xs, ys) -> np.ndarray:
        """Pistin o noktadaki gidiş yönü (birim vektör, ..., 2)."""
        return self.direction[self._grid_index(xs, ys)]

    def spawn_pose(self, offset: float) -> tuple[float, float, float]:
        """Başlangıç çizgisinin `offset` px önünde (x, y, yön)."""
        pos = self.start_center + self.start_normal * offset
        return float(pos[0]), float(pos[1]), self.start_heading
