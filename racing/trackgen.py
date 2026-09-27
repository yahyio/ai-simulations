"""Rastgele pist üretici: seçilebilir zorlukta, kendini kesmeyen kapalı döngü pistler.

Üretilen görsel, elle çizilmiş pistlerle aynı biçimdedir (beyaz zemin, siyah pist,
kırmızı başlangıç çizgisi); böylece `Track` analizi ve eğitim hiç değişmeden çalışır.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pygame

from .track import Track, TrackError

IMAGE_SIZE = (1336, 794)
MARGIN = 24
SAMPLES_PER_SEGMENT = 40
MAX_ATTEMPTS = 200
START_LINE_PX = 10
OVERLAP_CLEARANCE = 1.2   # iki pist parçası arasındaki en az mesafe (genişlik katı)
MIN_TURN_RADIUS = 0.35    # en keskin viraj yarıçapı (genişlik katı)
MAX_STRETCH = 1.8         # eksenler arası en fazla ölçek farkı
TRACKS_DIR = Path("tracks")


@dataclass(frozen=True)
class Difficulty:
    key: str
    label: str
    points: tuple[int, int]         # kontrol noktası sayısı aralığı
    radius_jitter: float            # yarıçap sapması → viraj keskinliği
    angle_jitter: float             # açısal düzensizlik
    width: tuple[int, int]          # pist genişliği (px)
    description: str


DIFFICULTIES = {
    "kolay": Difficulty("kolay", "Kolay", (6, 8), 0.24, 0.25, (96, 112), "geniş pist, yumuşak virajlar"),
    "orta": Difficulty("orta", "Orta", (9, 12), 0.30, 0.30, (72, 84), "sık virajlar, orta genişlik"),
    "zor": Difficulty("zor", "Zor", (13, 17), 0.44, 0.38, (54, 64), "dar pist, keskin virajlar ve şikanlar"),
}
DIFFICULTY_ORDER = ("kolay", "orta", "zor", "rastgele")
DIFFICULTY_LABELS = {"kolay": "Kolay", "orta": "Orta", "zor": "Zor", "rastgele": "Rastgele"}


@dataclass(frozen=True)
class GeneratedTrack:
    rgb: np.ndarray
    difficulty: Difficulty
    seed: int
    width: int

    @property
    def name(self) -> str:
        return f"{self.difficulty.key}_{self.seed}"


def resolve_difficulty(key: str, rng: np.random.Generator) -> Difficulty:
    if key == "rastgele":
        return DIFFICULTIES[str(rng.choice(list(DIFFICULTIES)))]
    if key not in DIFFICULTIES:
        raise ValueError(f"Bilinmeyen zorluk: {key!r} (seçenekler: {', '.join(DIFFICULTY_ORDER)})")
    return DIFFICULTIES[key]


# --------------------------------------------------------------------------- #
# Geometri
# --------------------------------------------------------------------------- #
def control_points(diff: Difficulty, rng: np.random.Generator) -> np.ndarray:
    """Merkez etrafında açıya göre sıralı, düzensiz noktalar (yıldız biçimli → döngü kendini kesmez)."""
    n = int(rng.integers(diff.points[0], diff.points[1] + 1))
    base = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
    angles = base + rng.uniform(-diff.angle_jitter, diff.angle_jitter, n) * (2 * math.pi / n)
    radii = 1.0 + rng.uniform(-diff.radius_jitter, diff.radius_jitter, n)
    return np.stack([np.cos(angles) * radii, np.sin(angles) * radii], axis=1)


def catmull_rom_closed(points: np.ndarray, samples: int = SAMPLES_PER_SEGMENT) -> np.ndarray:
    """Kapalı Catmull-Rom eğrisi (tüm kontrol noktalarından geçer)."""
    t = np.linspace(0.0, 1.0, samples, endpoint=False)[:, None]
    t2, t3 = t * t, t * t * t
    out = []
    n = len(points)
    for i in range(n):
        p0, p1, p2, p3 = (points[(i + k) % n] for k in (-1, 0, 1, 2))
        out.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    return np.concatenate(out)


def fit_to_image(curve: np.ndarray, width: int, size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """Eğriyi, pist kenarları görsele sığacak şekilde ölçekleyip ortalar."""
    lo, hi = curve.min(axis=0), curve.max(axis=0)
    pad = width / 2 + MARGIN
    scale = np.array([(size[0] - 2 * pad) / (hi[0] - lo[0]), (size[1] - 2 * pad) / (hi[1] - lo[1])])
    scale = np.minimum(scale, scale.min() * MAX_STRETCH)  # alanı doldur ama şekli fazla bozma
    center = (lo + hi) / 2
    return (curve - center) * scale + np.array(size) / 2


def arc_lengths(curve: np.ndarray) -> np.ndarray:
    seg = np.linalg.norm(np.diff(curve, axis=0, append=curve[:1]), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)[:-1]])


def is_well_formed(curve: np.ndarray, width: int) -> bool:
    """Pistin farklı kısımları üst üste binmiyor ve virajlar çok keskin değil mi?"""
    s = arc_lengths(curve)
    total = s[-1] + np.linalg.norm(curve[0] - curve[-1])
    along = np.abs(s[:, None] - s[None, :])
    along = np.minimum(along, total - along)          # döngü üzerindeki mesafe
    apart = np.linalg.norm(curve[:, None] - curve[None, :], axis=2)
    far_along = along > 2.5 * width
    if np.any(apart[far_along] < OVERLAP_CLEARANCE * width):
        return False

    prev, nxt = np.roll(curve, 1, axis=0), np.roll(curve, -1, axis=0)
    a = np.linalg.norm(curve - prev, axis=1)
    b = np.linalg.norm(nxt - curve, axis=1)
    c = np.linalg.norm(nxt - prev, axis=1)
    u, v = curve - prev, nxt - curve
    cross = np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])
    radius = a * b * c / np.maximum(2 * cross, 1e-9)   # üç noktadan geçen çemberin yarıçapı
    return float(radius.min()) > MIN_TURN_RADIUS * width


def straightest_index(curve: np.ndarray, window: int = 12) -> int:
    """Başlangıç çizgisi için en düz yer (yön değişimi en küçük)."""
    d = np.diff(curve, axis=0, append=curve[:1])
    heading = np.arctan2(d[:, 1], d[:, 0])
    turn = np.abs(np.angle(np.exp(1j * (np.roll(heading, -1) - heading))))
    padded = np.concatenate([turn[-window:], turn, turn[:window]])
    return int(np.argmin(np.convolve(padded, np.ones(2 * window + 1), mode="valid")))


# --------------------------------------------------------------------------- #
# Çizim
# --------------------------------------------------------------------------- #
def rasterize(curve: np.ndarray, width: int, size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """Eğriyi `width` kalınlığında siyah şerit olarak beyaz zemine çizer; (H, W, 3) döner."""
    surface = pygame.Surface(size)
    surface.fill((255, 255, 255))
    pts = [tuple(p) for p in curve]
    for i, p in enumerate(pts):
        pygame.draw.line(surface, (0, 0, 0), p, pts[(i + 1) % len(pts)], width)
        pygame.draw.circle(surface, (0, 0, 0), p, width / 2)
    return pygame.surfarray.array3d(surface).transpose(1, 0, 2).copy()


def paint_start_line(rgb: np.ndarray, curve: np.ndarray, index: int) -> None:
    """Pisti enlemesine kesen kırmızı başlangıç çizgisi (yalnız pist piksellerine)."""
    p = curve[index]
    d = curve[(index + 1) % len(curve)] - curve[index - 1]
    d = d / np.linalg.norm(d)
    h, w = rgb.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    rx, ry = xx - p[0], yy - p[1]
    across = np.abs(rx * d[0] + ry * d[1]) <= START_LINE_PX / 2
    near = np.hypot(rx, ry) < 200
    track = rgb.sum(axis=2) < 3 * 128
    rgb[track & near & across] = (255, 0, 0)


# --------------------------------------------------------------------------- #
def generate(difficulty: str = "orta", seed: int | None = None, track_cfg: dict | None = None) -> GeneratedTrack:
    """Zorluğa uygun, doğrulanmış rastgele bir pist üretir."""
    if seed is None:
        seed = int(np.random.default_rng().integers(0, 1_000_000))
    rng = np.random.default_rng(seed)
    diff = resolve_difficulty(difficulty, rng)
    for _ in range(MAX_ATTEMPTS):
        width = int(rng.integers(diff.width[0], diff.width[1] + 1))
        curve = fit_to_image(catmull_rom_closed(control_points(diff, rng)), width)
        if not is_well_formed(curve, width):
            continue
        rgb = rasterize(curve, width)
        paint_start_line(rgb, curve, straightest_index(curve))
        if track_cfg is not None:
            try:
                Track(rgb, track_cfg)  # analizden geçmeyen pisti asla döndürme
            except TrackError:
                continue
        return GeneratedTrack(rgb=rgb, difficulty=diff, seed=seed, width=width)
    raise TrackError(f"{MAX_ATTEMPTS} denemede geçerli pist üretilemedi (zorluk: {diff.label}).")


def save(track: GeneratedTrack, directory: Path = TRACKS_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{track.name}.png"
    pygame.image.save(pygame.surfarray.make_surface(track.rgb.transpose(1, 0, 2)), str(path))
    return path
