"""Menü, ayarlar ve diyalog ekranları."""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Callable

import pygame

from . import theme
from .config import DEFAULTS, RANGES

MENU_FPS = 60
BUTTON_W, BUTTON_H, BUTTON_GAP = 340, 54, 10


def _dim(screen: pygame.Surface, alpha: int = 200) -> None:
    veil = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
    veil.fill((*theme.BG, alpha))
    screen.blit(veil, (0, 0))


# --------------------------------------------------------------------------- #
# Ana menü
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MenuItem:
    key: str
    label: str
    hint: str
    enabled: bool = True


class MainMenu:
    def __init__(self, screen: pygame.Surface, clock: pygame.time.Clock,
                 draw_backdrop: Callable[[pygame.Surface], None], summary: list[tuple[str, str]],
                 title: str = "AI RACING",
                 subtitle: str = "Pekiştirmeli öğrenme ile kendi kendine süren arabalar"):
        self.screen = screen
        self.clock = clock
        self.draw_backdrop = draw_backdrop
        self.summary = summary
        self.title = title
        self.subtitle = subtitle

    def run(self, items: list[MenuItem]) -> str:
        enabled = [i for i, item in enumerate(items) if item.enabled]
        if not enabled:
            return "quit"
        selected = enabled[0]
        rects = self._layout(items)
        t0 = pygame.time.get_ticks()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return "quit"
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return "quit"
                    if event.key in (pygame.K_UP, pygame.K_w):
                        selected = enabled[(enabled.index(selected) - 1) % len(enabled)]
                    elif event.key in (pygame.K_DOWN, pygame.K_s):
                        selected = enabled[(enabled.index(selected) + 1) % len(enabled)]
                    elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
                        return items[selected].key
                elif event.type == pygame.MOUSEMOTION:
                    for i, rect in enumerate(rects):
                        if rect.collidepoint(event.pos) and items[i].enabled:
                            selected = i
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for i, rect in enumerate(rects):
                        if rect.collidepoint(event.pos) and items[i].enabled:
                            return items[i].key
            self._draw(items, rects, selected, (pygame.time.get_ticks() - t0) / 1000)
            pygame.display.flip()
            self.clock.tick(MENU_FPS)

    def _layout(self, items: list[MenuItem]) -> list[pygame.Rect]:
        top = self.screen.get_height() // 2 - (len(items) * (BUTTON_H + BUTTON_GAP)) // 2 + 40
        return [pygame.Rect(72, top + i * (BUTTON_H + BUTTON_GAP), BUTTON_W, BUTTON_H) for i in range(len(items))]

    def _draw(self, items, rects, selected: int, t: float) -> None:
        screen = self.screen
        screen.fill(theme.BG)
        self.draw_backdrop(screen)
        # Soldan sağa açılan perde: menü okunaklı kalır, pist arkada görünür.
        w, h = screen.get_size()
        veil = pygame.Surface((w, h), pygame.SRCALPHA)
        for x in range(0, w, 8):
            alpha = int(235 * max(0.0, 1.0 - x / (w * 0.75)) + 70)
            pygame.draw.rect(veil, (*theme.BG, min(alpha, 245)), (x, 0, 8, h))
        screen.blit(veil, (0, 0))

        title_y = rects[0].y - 150
        theme.text(screen, self.title, (72, title_y), 64, theme.TEXT, bold=True)
        underline = pygame.Rect(76, title_y + 84, 90 + int(20 * math.sin(t * 2)), 5)
        pygame.draw.rect(screen, theme.ACCENT, underline, border_radius=3)
        theme.text(screen, self.subtitle, (72, title_y + 100), 18, theme.TEXT_DIM)

        for i, (item, rect) in enumerate(zip(items, rects)):
            active = i == selected
            pygame.draw.rect(screen, theme.PANEL_EDGE if active else theme.PANEL, rect, border_radius=10)
            if active:
                pygame.draw.rect(screen, theme.ACCENT, (rect.x, rect.y + 10, 4, rect.h - 20), border_radius=2)
            color = theme.TEXT if item.enabled else theme.TEXT_FAINT
            theme.text(screen, item.label, (rect.x + 22, rect.y + 7), 20, color, bold=active)
            theme.text(screen, item.hint, (rect.x + 22, rect.y + 32), 12,
                       theme.TEXT_DIM if item.enabled else theme.TEXT_FAINT)

        self._draw_summary(rects[-1].bottom + 36)
        theme.text(screen, "↑ ↓ seç  ·  Enter onayla  ·  Esc çıkış", (72, h - 36), 13, theme.TEXT_FAINT)

    def _draw_summary(self, y: int) -> None:
        if not self.summary:
            return
        col = BUTTON_W // len(self.summary) + 10
        for i, (label, value) in enumerate(self.summary):
            theme.text(self.screen, label, (72 + i * col, y), 11, theme.TEXT_FAINT, bold=True)
            theme.text(self.screen, value, (72 + i * col, y + 14), 20, theme.TEXT, kind="mono", bold=True)


# --------------------------------------------------------------------------- #
# Ayarlar
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Diyaloglar
# --------------------------------------------------------------------------- #
def _dialog_box(screen: pygame.Surface, title: str, message: str, accent) -> pygame.Rect:
    w, h = screen.get_size()
    lines = message.split("\n")
    box = pygame.Rect(0, 0, min(600, w - 80), 150 + 22 * len(lines))
    box.center = (w // 2, h // 2)
    theme.panel(screen, box, radius=14)
    pygame.draw.rect(screen, accent, (box.x, box.y, box.w, 4), border_top_left_radius=14, border_top_right_radius=14)
    theme.text(screen, title, (box.x + 28, box.y + 26), 24, theme.TEXT, bold=True)
    for i, line in enumerate(lines):
        theme.text(screen, line, (box.x + 28, box.y + 68 + i * 22), 15, theme.TEXT_DIM)
    return box


def confirm(screen: pygame.Surface, clock: pygame.time.Clock, title: str, message: str) -> bool:
    """Evet/Hayır sorusu. E/Enter evet, H/Esc hayır."""
    backdrop = screen.copy()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_e, pygame.K_y, pygame.K_RETURN, pygame.K_KP_ENTER):
                    return True
                if event.key in (pygame.K_h, pygame.K_n, pygame.K_ESCAPE):
                    return False
        screen.blit(backdrop, (0, 0))
        _dim(screen)
        box = _dialog_box(screen, title, message, theme.BAD)
        theme.text(screen, "E / Enter: evet     H / Esc: hayır", (box.x + 28, box.bottom - 40), 14, theme.ACCENT)
        pygame.display.flip()
        clock.tick(MENU_FPS)


def show_message(screen: pygame.Surface, clock: pygame.time.Clock, title: str, message: str,
                 accent=theme.INFO) -> None:
    """Bilgi mesajı; bir tuşa veya tıklamaya kadar bekler."""
    backdrop = screen.copy()
    while True:
        for event in pygame.event.get():
            if event.type in (pygame.QUIT, pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN):
                return
        screen.blit(backdrop, (0, 0))
        _dim(screen)
        box = _dialog_box(screen, title, message, accent)
        theme.text(screen, "Devam etmek için bir tuşa basın", (box.x + 28, box.bottom - 40), 14, theme.TEXT_FAINT)
        pygame.display.flip()
        clock.tick(MENU_FPS)


def draw_loading(screen: pygame.Surface, message: str) -> None:
    screen.fill(theme.BG)
    w, h = screen.get_size()
    theme.text(screen, "AI RACING", (w // 2, h // 2 - 30), 40, theme.TEXT, bold=True, anchor="center")
    theme.text(screen, message, (w // 2, h // 2 + 20), 16, theme.TEXT_DIM, anchor="center")
    pygame.display.flip()
    pygame.event.pump()
