"""AI Simülasyonları — başlatıcı.

Kullanım:
    python main.py              # simülasyon seçme penceresi
    python -m racing --help     # AI Racing komut satırı seçenekleri
    python -m walker --help     # AI Walker 3D komut satırı seçenekleri
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
PREVIEWS = (Path("docs/images/racing.png"), Path("docs/images/walker.png"))
WINDOW = (1440, 810)
SLIDE_SECONDS = 5.0


class _Slideshow:
    """Menü arka planı: önizleme görsellerini sırayla, yumuşak geçişle gösterir."""

    def __init__(self, size: tuple[int, int]):
        import pygame

        self.images = [
            pygame.transform.smoothscale(pygame.image.load(str(p)).convert(), size) for p in PREVIEWS if p.exists()
        ]
        self.start = time.perf_counter()

    def __call__(self, screen) -> None:
        if not self.images:
            return
        elapsed = time.perf_counter() - self.start
        index = int(elapsed // SLIDE_SECONDS) % len(self.images)
        screen.blit(self.images[index], (0, 0))
        fade = (elapsed % SLIDE_SECONDS) / SLIDE_SECONDS
        if len(self.images) > 1 and fade > 0.8:
            nxt = self.images[(index + 1) % len(self.images)]
            nxt.set_alpha(int(255 * (fade - 0.8) / 0.2))
            screen.blit(nxt, (0, 0))
            nxt.set_alpha(None)


def choose() -> str:
    import pygame

    from common.widgets import MainMenu, MenuItem

    pygame.init()
    try:
        screen = pygame.display.set_mode(WINDOW, pygame.SCALED)
        pygame.display.set_caption("AI Simülasyonları")
        items = [
            MenuItem("racing", "AI Racing", "16 araba pistte tur atmayı öğrenir · rastgele pistler"),
            MenuItem("walker", "AI Walker 3D", "hümanoidler sıfırdan yürümeyi ve koşmayı öğrenir"),
            MenuItem("quit", "Çıkış", ""),
        ]
        menu = MainMenu(screen, pygame.time.Clock(), _Slideshow(WINDOW), [], "AI SİMÜLASYONLARI",
                        "Hiçbir şey bilmeden, deneme yanılmayla öğrenen iki simülasyon")
        return menu.run(items)
    finally:
        pygame.quit()


def main() -> int:
    os.chdir(PROJECT_DIR)
    choice = choose()
    if choice == "racing":
        from racing.__main__ import main as racing_main

        return racing_main([])
    if choice == "walker":
        from walker.__main__ import main as walker_main

        return walker_main([])
    return 0


if __name__ == "__main__":
    sys.exit(main())
