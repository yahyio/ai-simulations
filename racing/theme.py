"""Görsel tema: renk paleti ve yazı tipleri (tek yerden yönetilir)."""
from __future__ import annotations

import pygame

# Arayüz renkleri — gece pit-lane paneli
BG = (14, 16, 22)
PANEL = (21, 24, 32)
PANEL_EDGE = (38, 43, 56)
TEXT = (232, 235, 242)
TEXT_DIM = (140, 147, 163)
TEXT_FAINT = (88, 95, 112)
ACCENT = (255, 196, 0)       # pit-lane sarısı: vurgular, rekor
GOOD = (64, 214, 128)        # tur tamamlandı / olumlu
BAD = (240, 84, 84)          # kaza / olumsuz
INFO = (88, 170, 255)        # öğrenme metrikleri
OVERLAY = (8, 10, 14, 200)

# Araba renkleri — birbirinden kolay ayırt edilen canlı tonlar
CAR_COLORS = [
    (255, 196, 0), (240, 84, 84), (88, 170, 255), (64, 214, 128),
    (255, 128, 64), (190, 120, 255), (0, 214, 214), (255, 105, 180),
    (170, 220, 60), (255, 255, 255), (120, 140, 255), (255, 160, 120),
    (60, 190, 160), (230, 200, 120), (180, 180, 200), (255, 90, 140),
]

_FONT_FAMILIES = {
    "ui": "segoeui,helveticaneue,arial,dejavusans",
    "mono": "consolas,menlo,dejavusansmono,couriernew",
}
_cache: dict[tuple[str, int, bool], pygame.font.Font] = {}


def font(size: int, kind: str = "ui", bold: bool = False) -> pygame.font.Font:
    """Önbellekli yazı tipi; sistemde yoksa pygame varsayılanına düşer."""
    key = (kind, size, bold)
    if key not in _cache:
        try:
            _cache[key] = pygame.font.SysFont(_FONT_FAMILIES[kind], size, bold=bold)
        except (OSError, RuntimeError):
            _cache[key] = pygame.font.Font(None, int(size * 1.3))
    return _cache[key]


def lerp_color(a, b, t: float) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def text(surface: pygame.Surface, value: str, pos, size: int = 16, color=TEXT,
         kind: str = "ui", bold: bool = False, anchor: str = "topleft") -> pygame.Rect:
    """Metni çizip kapladığı alanı döndürür. anchor: topleft, topright, center, midleft..."""
    rendered = font(size, kind, bold).render(value, True, color)
    rect = rendered.get_rect(**{anchor: pos})
    surface.blit(rendered, rect)
    return rect


def panel(surface: pygame.Surface, rect: pygame.Rect, color=PANEL, radius: int = 10, edge=PANEL_EDGE) -> None:
    pygame.draw.rect(surface, color, rect, border_radius=radius)
    if edge is not None:
        pygame.draw.rect(surface, edge, rect, width=1, border_radius=radius)
