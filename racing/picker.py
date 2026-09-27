"""Pist oluşturma ekranı: zorluk seç, rastgele pist üret, önizle, kullan."""
from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np
import pygame

from common import theme

from . import trackgen
from .render import build_track_image
from .track import Track, TrackError

PICKER_FPS = 60
PREVIEW_TOP = 190
PREVIEW_BOTTOM_GAP = 90


@dataclass(frozen=True)
class Pick:
    """Kullanıcının seçimi: üretilmiş bir pist ya da varsayılan pist (generated=None)."""
    generated: trackgen.GeneratedTrack | None
    track: Track | None


class TrackPicker:
    def __init__(self, screen: pygame.Surface, clock: pygame.time.Clock, track_cfg: dict):
        self.screen = screen
        self.clock = clock
        self.track_cfg = {**track_cfg, "reverse": False}
        self.rng = np.random.default_rng()
        self.level = 1  # "orta"
        self.generated: trackgen.GeneratedTrack | None = None
        self.track: Track | None = None
        self.preview: pygame.Surface | None = None
        self.error = ""

    def run(self) -> Pick | None:
        """Seçilen pisti döndürür; iptal edilirse None."""
        self._regenerate()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if event.type != pygame.KEYDOWN:
                    continue
                if event.key == pygame.K_ESCAPE:
                    return None
                if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER) and self.generated is not None:
                    return Pick(self.generated, self.track)
                if event.key == pygame.K_o:
                    return Pick(None, None)
                if event.key in (pygame.K_LEFT, pygame.K_RIGHT):
                    step = -1 if event.key == pygame.K_LEFT else 1
                    self.level = (self.level + step) % len(trackgen.DIFFICULTY_ORDER)
                    self._regenerate()
                elif event.key in (pygame.K_r, pygame.K_SPACE):
                    self._regenerate()
            self._draw()
            pygame.display.flip()
            self.clock.tick(PICKER_FPS)

    # ------------------------------------------------------------------ #
    def _regenerate(self) -> None:
        """Pisti arka planda üretir; bu sırada pencere çizilmeye ve olayları işlemeye devam eder."""
        key = trackgen.DIFFICULTY_ORDER[self.level]
        seed = int(self.rng.integers(0, 1_000_000))
        result: dict = {}

        def work() -> None:
            try:
                generated = trackgen.generate(key, seed, self.track_cfg)
                track = Track(generated.rgb, self.track_cfg, name=generated.name)
                result["value"] = (generated, track, build_track_image(track))
            except TrackError as exc:
                result["error"] = str(exc)

        worker = threading.Thread(target=work, daemon=True)
        worker.start()
        while worker.is_alive():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.event.post(event)  # ana döngü kapatsın
            self._draw(busy=True)
            pygame.display.flip()
            self.clock.tick(PICKER_FPS)
        if "error" in result:
            self.error = result["error"]
            return
        generated, track, image = result["value"]
        self.error = ""
        self.generated, self.track = generated, track
        self.preview = self._make_preview(track, image)

    def _preview_rect(self) -> pygame.Rect:
        w, h = self.screen.get_size()
        return pygame.Rect(72, PREVIEW_TOP, w - 144, h - PREVIEW_TOP - PREVIEW_BOTTOM_GAP)

    def _make_preview(self, track: Track, image: np.ndarray) -> pygame.Surface:
        area = self._preview_rect()
        surface = pygame.surfarray.make_surface(image.transpose(1, 0, 2))
        scale = min(area.w / track.width, area.h / track.height)
        return pygame.transform.smoothscale(surface, (round(track.width * scale), round(track.height * scale)))

    def _draw(self, busy: bool = False) -> None:
        screen = self.screen
        h = screen.get_height()
        screen.fill(theme.BG)
        theme.text(screen, "Pist oluştur", (72, 40), 40, theme.TEXT, bold=True)
        theme.text(screen, "Her pist rastgele üretilir; kendini kesmez ve kapalı bir döngüdür.", (72, 92), 15,
                   theme.TEXT_DIM)

        x = 72
        for i, key in enumerate(trackgen.DIFFICULTY_ORDER):
            active = i == self.level
            label = theme.font(16, bold=active).render(trackgen.DIFFICULTY_LABELS[key], True,
                                                       theme.BG if active else theme.TEXT_DIM)
            chip = label.get_rect().inflate(28, 14)
            chip.topleft = (x, 130)
            pygame.draw.rect(screen, theme.ACCENT if active else theme.PANEL, chip, border_radius=chip.h // 2)
            screen.blit(label, label.get_rect(center=chip.center))
            x = chip.right + 10

        info = ""
        if self.generated is not None:
            g = self.generated
            info = f"{g.difficulty.label}: {g.difficulty.description}   ·   genişlik {g.width} px"
            if self.track is not None:
                info += f"   ·   tur {self.track.length:.0f} px   ·   tohum {g.seed}"
        theme.text(screen, info, (x + 16, 138), 14, theme.TEXT_DIM)

        area = self._preview_rect()
        pygame.draw.rect(screen, theme.PANEL, area, border_radius=12)
        if self.preview is not None:
            screen.blit(self.preview, self.preview.get_rect(center=area.center))
        if busy:
            veil = pygame.Surface(area.size, pygame.SRCALPHA)
            veil.fill((*theme.BG, 170))
            screen.blit(veil, area)
            theme.text(screen, "Pist üretiliyor…", area.center, 22, theme.ACCENT, bold=True, anchor="center")
        if self.error:
            theme.text(screen, self.error, (area.centerx, area.bottom - 30), 14, theme.BAD, anchor="center")

        hint = "← → zorluk  ·  R yeni pist  ·  Enter kullan  ·  O varsayılan piste dön  ·  Esc iptal"
        theme.text(screen, hint, (72, h - 40), 14, theme.TEXT_FAINT)
