"""Ortak pygame bileşenleri: ana menü, onay ve bilgi pencereleri, yükleme ekranı."""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import pygame

from . import theme

MENU_FPS = 60
BUTTON_W, BUTTON_H, BUTTON_GAP = 340, 54, 10


def dim(screen: pygame.Surface, alpha: int = 200) -> None:
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
                 title: str, subtitle: str):
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
        # Soldan sağa açılan perde: menü okunaklı kalır, sahne arkada görünür.
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
        dim(screen)
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
        dim(screen)
        box = _dialog_box(screen, title, message, accent)
        theme.text(screen, "Devam etmek için bir tuşa basın", (box.x + 28, box.bottom - 40), 14, theme.TEXT_FAINT)
        pygame.display.flip()
        clock.tick(MENU_FPS)


def draw_loading(screen: pygame.Surface, message: str, title: str) -> None:
    screen.fill(theme.BG)
    w, h = screen.get_size()
    theme.text(screen, title, (w // 2, h // 2 - 30), 40, theme.TEXT, bold=True, anchor="center")
    theme.text(screen, message, (w // 2, h // 2 + 20), 16, theme.TEXT_DIM, anchor="center")
    pygame.display.flip()
    pygame.event.pump()
