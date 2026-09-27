"""Yürüme simülasyonunun yan paneli: skorlar, öndeki karakter, öğrenme metrikleri, grafikler."""
from __future__ import annotations

from dataclasses import dataclass

import pygame

from common import theme

from .trainer import WalkTrainer

PANEL_WIDTH = 340
PAD = 20
CHART_HEIGHT = 72
CONTROLS = (
    ("Boşluk", "duraklat"), ("+ / −", "hız"),
    ("V", "turbo"), ("C", "kamera açısı"),
    ("Sürükle", "kamerayı döndür"), ("Tekerlek", "yakınlaş"),
    ("S", "kaydet"), ("Esc", "menü"),
)


@dataclass(frozen=True)
class HudState:
    paused: bool
    turbo: bool
    speed: int
    steps_per_sec: float
    camera: str


def _count(n: float) -> str:
    if n >= 1_000_000:
        return f"{n / 1e6:.2f}M"
    return f"{n / 1000:.1f}k" if n >= 10_000 else f"{int(n)}"


def line_chart(screen, rect: pygame.Rect, series: list[tuple[list[float], tuple]], empty: str, unit: str) -> None:
    pygame.draw.rect(screen, theme.BG, rect, border_radius=6)
    values = [v for data, _ in series for v in data]
    if len(values) < 2:
        theme.text(screen, empty, rect.center, 13, theme.TEXT_FAINT, anchor="center")
        return
    hi = max(values) or 1.0
    inner = rect.inflate(-14, -18)
    for data, color in series:
        if len(data) < 2:
            continue
        step = inner.w / (len(data) - 1)
        pts = [(inner.x + step * i, inner.bottom - inner.h * max(v, 0) / hi) for i, v in enumerate(data)]
        pygame.draw.aalines(screen, color, False, pts)
    theme.text(screen, f"{hi:.1f} {unit}", (rect.x + 5, rect.y + 3), 11, theme.TEXT_FAINT, kind="mono")


class WalkHud:
    def __init__(self, rect: pygame.Rect):
        self.rect = rect

    def draw(self, screen: pygame.Surface, trainer: WalkTrainer, state: HudState) -> None:
        r = self.rect
        pygame.draw.rect(screen, theme.PANEL, r)
        pygame.draw.line(screen, theme.PANEL_EDGE, r.topleft, r.bottomleft)
        x, w = r.x + PAD, r.w - 2 * PAD
        y = self._header(screen, trainer, state, x, r.y + PAD, w)
        y = self._scoreboard(screen, trainer, x, y + 12, w)
        y = self._leader(screen, trainer, x, y + 16, w)
        y = self._learning(screen, trainer, state, x, y + 14, w)
        if trainer.training:
            self._charts(screen, trainer, x, y + 12, w)
        self._controls(screen, x, r.bottom - PAD, w)

    def _header(self, screen, trainer: WalkTrainer, state: HudState, x, y, w) -> int:
        theme.text(screen, "AI WALKER 3D", (x, y), 24, theme.TEXT, bold=True)
        if state.paused:
            label, color = "DURAKLATILDI", theme.ACCENT
        elif trainer.updating:
            label, color = "ÖĞRENİYOR", theme.INFO
        elif trainer.training:
            label, color = ("DENİYOR · TURBO" if state.turbo else "DENİYOR"), theme.GOOD
        else:
            label, color = "İZLEME", theme.INFO
        chip = theme.font(12, bold=True).render(label, True, theme.BG)
        box = chip.get_rect().inflate(14, 6)
        box.topright = (x + w, y + 6)
        pygame.draw.rect(screen, color, box, border_radius=box.h // 2)
        screen.blit(chip, chip.get_rect(center=box.center))
        task = trainer.task
        theme.text(screen, f"{task.label} · {task.description}", (x, y + 34), 13, theme.TEXT_DIM)
        return y + 56

    def _scoreboard(self, screen, trainer: WalkTrainer, x, y, w) -> int:
        s = trainer.stats
        if trainer.training:
            cells = (("DENEME", _count(s.episodes), theme.TEXT),
                     ("EN UZAK", f"{s.best_distance:.1f} m", theme.ACCENT),
                     ("EN HIZLI", f"{s.best_speed:.2f} m/s", theme.GOOD))
        else:
            eps = trainer.watch_episodes
            cells = (("DENEME", str(len(eps)), theme.TEXT),
                     ("SON", f"{eps[-1].distance:.1f} m" if eps else "—", theme.GOOD),
                     ("EN UZAK", f"{max(e.distance for e in eps):.1f} m" if eps else "—", theme.ACCENT))
        col = w / 3
        for i, (label, value, color) in enumerate(cells):
            theme.text(screen, label, (x + i * col, y), 11, theme.TEXT_FAINT, bold=True)
            theme.text(screen, value, (x + i * col, y + 14), 20, color, kind="mono", bold=True)
        return y + 44

    def _leader(self, screen, trainer: WalkTrainer, x, y, w) -> int:
        lane = trainer.leader()
        info = trainer.runner(lane)
        theme.text(screen, f"EN ÖNDEKİ · kulvar {lane + 1}", (x, y), 11, theme.TEXT_FAINT, bold=True)
        y += 20
        rows = (
            ("Mesafe", f"{info.distance:6.2f} m"),
            ("Hız", f"{info.speed:6.2f} m/s"),
            ("Ayakta kalma", f"{info.seconds:5.1f} s"),
        )
        for label, value in rows:
            theme.text(screen, label, (x, y), 14, theme.TEXT_DIM)
            theme.text(screen, value, (x + w, y), 14, theme.TEXT, kind="mono", anchor="topright")
            y += 22
        theme.text(screen, "Yere basan ayak", (x, y), 14, theme.TEXT_DIM)
        for k, (label, on) in enumerate(zip(("sol", "sağ"), info.contacts)):
            cx = x + w - 70 + k * 42
            pygame.draw.circle(screen, theme.GOOD if on else theme.PANEL_EDGE, (cx, y + 9), 7)
            theme.text(screen, label, (cx + 10, y + 1), 12, theme.TEXT_DIM)
        return y + 24

    def _learning(self, screen, trainer: WalkTrainer, state: HudState, x, y, w) -> int:
        ppo = trainer.ppo
        theme.text(screen, "ÖĞRENME" if trainer.training else "MODEL", (x, y), 11, theme.TEXT_FAINT, bold=True)
        y += 20
        speed = "turbo" if state.turbo else f"x{state.speed}"
        rows = [("Toplam deneyim", f"{_count(ppo.total_steps)} adım")]
        if trainer.training:
            rows += [("Güncelleme", str(ppo.updates)),
                     ("Keşif (gürültü)", f"{ppo.action_std:.3f}"),
                     ("Paralel karakter", f"{trainer.env.num_envs}")]
        else:
            source = trainer.model_source
            rows.append(("Model", "hazır (depodan)" if source and "models" in source.parts else "yerel en iyi"))
        rows += [("Sim hızı", f"{speed} · {state.steps_per_sec:,.0f} adım/sn"), ("Kamera", state.camera)]
        for label, value in rows:
            theme.text(screen, label, (x, y), 14, theme.TEXT_DIM)
            theme.text(screen, value, (x + w, y), 14, theme.TEXT, kind="mono", anchor="topright")
            y += 22
        if trainer.training:
            fill = ppo.t / ppo.cfg.rollout_steps
            label = "sinir ağı güncelleniyor…" if trainer.updating else "deneyim toplanıyor"
            theme.text(screen, label, (x, y), 12, theme.INFO if trainer.updating else theme.TEXT_FAINT)
            bar = pygame.Rect(x, y + 18, w, 5)
            pygame.draw.rect(screen, theme.PANEL_EDGE, bar, border_radius=3)
            done = bar.copy()
            done.w = int(w * (1.0 if trainer.updating else fill))
            if done.w:
                pygame.draw.rect(screen, theme.INFO if trainer.updating else theme.GOOD, done, border_radius=3)
            y += 28
        return y

    def _charts(self, screen, trainer: WalkTrainer, x, y, w) -> None:
        s = trainer.stats
        theme.text(screen, "MESAFE · ortalama ve en iyi / güncelleme", (x, y), 11, theme.TEXT_FAINT, bold=True)
        line_chart(screen, pygame.Rect(x, y + 18, w, CHART_HEIGHT),
                   [(s.history_best_distance, theme.lerp_color(theme.ACCENT, theme.BG, 0.45)),
                    (s.history_mean_distance, theme.ACCENT)],
                   "İlk güncellemeden sonra görünecek", "m")
        y += CHART_HEIGHT + 32
        theme.text(screen, "AYAKTA KALMA SÜRESİ · ortalama", (x, y), 11, theme.TEXT_FAINT, bold=True)
        line_chart(screen, pygame.Rect(x, y + 18, w, CHART_HEIGHT), [(s.history_mean_seconds, theme.GOOD)],
                   "İlk güncellemeden sonra görünecek", "s")

    def _controls(self, screen, x, bottom, w) -> None:
        rows = (len(CONTROLS) + 1) // 2
        y = bottom - rows * 20
        pygame.draw.line(screen, theme.PANEL_EDGE, (x, y - 10), (x + w, y - 10))
        for i, (key, action) in enumerate(CONTROLS):
            cx, cy = x + (i % 2) * w / 2, y + (i // 2) * 20
            rect = theme.text(screen, key, (cx, cy), 12, theme.ACCENT, kind="mono", bold=True)
            theme.text(screen, action, (cx + max(rect.w, 56) + 6, cy), 12, theme.TEXT_DIM)
