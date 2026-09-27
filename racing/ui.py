"""Yarış simülasyonunun ayarlar ekranı (ortak menü bileşenleri `common.widgets` içinde)."""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import pygame

from common import theme
from common.widgets import MENU_FPS

from .config import DEFAULTS, RANGES


@dataclass(frozen=True)
class Setting:
    label: str
    section: str
    key: str
    step: float = 1.0
    kind: str = "linear"  # linear | log | bool
    note: str = ""


SETTINGS = (
    Setting("Araba sayısı", "simulation", "num_cars", 1),
    Setting("Başlangıç sim hızı (adım/kare)", "simulation", "steps_per_frame", 1),
    Setting("Bölüm adım sınırı", "simulation", "max_steps_per_episode", 500),
    Setting("Azami hız", "car", "max_speed", 0.5),
    Setting("İvme", "car", "acceleration", 0.02),
    Setting("Dönüş hızı (°/adım)", "car", "turn_rate_deg", 0.5),
    Setting("Savrulma (yüksek hızda)", "car", "understeer", 0.05),
    Setting("Sensör sayısı", "car", "sensor_count", 2, note="değişirse yeni eğitim gerekir"),
    Setting("Sensör menzili", "car", "sensor_range", 20),
    Setting("Öğrenme oranı", "learning", "learning_rate", 2.0, "log"),
    Setting("İndirim faktörü (γ)", "learning", "discount_factor", 0.005),
    Setting("Keşfin azalacağı adım", "learning", "epsilon_decay_steps", 2.0, "log"),
    Setting("Sensörleri göster", "display", "show_sensors", kind="bool"),
    Setting("İz çizgisini göster", "display", "show_trail", kind="bool"),
    Setting("Pisti ters yönde sür", "track", "reverse", kind="bool"),
)


def adjust(setting: Setting, value, direction: int):
    """Bir ayarı sınırlar içinde bir adım değiştirip yeni değeri döndürür."""
    if setting.kind == "bool":
        return not value
    lo, hi = RANGES.get((setting.section, setting.key), (-math.inf, math.inf))
    if setting.kind == "log":
        new = value * (setting.step if direction > 0 else 1 / setting.step)
    else:
        new = value + direction * setting.step
    new = min(max(new, lo), hi)
    return int(round(new)) if isinstance(value, int) else round(float(new), 6)


def format_value(value) -> str:
    if isinstance(value, bool):
        return "AÇIK" if value else "KAPALI"
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".") if value < 0.01 else f"{value:.3f}".rstrip("0").rstrip(".")
    return f"{value:,}".replace(",", ".")


class SettingsScreen:
    def __init__(self, screen: pygame.Surface, clock: pygame.time.Clock):
        self.screen = screen
        self.clock = clock

    def run(self, config: dict) -> dict | None:
        """Düzenlenmiş yapılandırmanın kopyasını döndürür; iptal edilirse None."""
        draft = copy.deepcopy(config)
        selected = 0
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if event.type != pygame.KEYDOWN:
                    continue
                setting = SETTINGS[selected]
                section = draft[setting.section]
                if event.key == pygame.K_ESCAPE:
                    return None
                if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    return draft
                if event.key == pygame.K_UP:
                    selected = (selected - 1) % len(SETTINGS)
                elif event.key == pygame.K_DOWN:
                    selected = (selected + 1) % len(SETTINGS)
                elif event.key in (pygame.K_LEFT, pygame.K_RIGHT, pygame.K_SPACE):
                    direction = -1 if event.key == pygame.K_LEFT else 1
                    section[setting.key] = adjust(setting, section[setting.key], direction)
                elif event.key == pygame.K_d:
                    section[setting.key] = copy.deepcopy(DEFAULTS[setting.section][setting.key])
            self._draw(draft, selected)
            pygame.display.flip()
            self.clock.tick(MENU_FPS)

    def _draw(self, draft: dict, selected: int) -> None:
        screen = self.screen
        w, h = screen.get_size()
        screen.fill(theme.BG)
        theme.text(screen, "Ayarlar", (72, 44), 40, theme.TEXT, bold=True)
        theme.text(screen, "Değişiklikler config.yaml dosyasına kaydedilir.", (72, 96), 15, theme.TEXT_DIM)

        row_h = 38
        width = min(w - 144, 820)
        for i, setting in enumerate(SETTINGS):
            y = 136 + i * row_h
            rect = pygame.Rect(60, y, width + 24, row_h - 4)
            active = i == selected
            if active:
                pygame.draw.rect(screen, theme.PANEL_EDGE, rect, border_radius=8)
                pygame.draw.rect(screen, theme.ACCENT, (rect.x, rect.y + 8, 4, rect.h - 16), border_radius=2)
            value = draft[setting.section][setting.key]
            default = DEFAULTS[setting.section][setting.key]
            theme.text(screen, setting.label, (rect.x + 18, y + 7), 17, theme.TEXT if active else theme.TEXT_DIM)
            if setting.note and active:
                theme.text(screen, setting.note, (rect.x + 330, y + 10), 12, theme.ACCENT)
            label = f"‹  {format_value(value)}  ›" if active else format_value(value)
            theme.text(screen, label, (rect.right - 18, y + 6), 17, theme.TEXT if value == default else theme.INFO,
                       kind="mono", bold=active, anchor="topright")

        hint = "↑ ↓ seç  ·  ← → değiştir  ·  D varsayılan  ·  Enter kaydet  ·  Esc iptal"
        theme.text(screen, hint, (72, h - 36), 13, theme.TEXT_FAINT)
        theme.text(screen, "Mavi değerler varsayılandan farklı", (w - 72, h - 36), 13, theme.INFO, anchor="topright")
