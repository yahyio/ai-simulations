"""Yan bilgi paneli: skorlar, anlık yarış durumu, öğrenme metrikleri ve gelişim grafikleri."""
from __future__ import annotations

from dataclasses import dataclass

import pygame

from common import theme

from .env import Status
from .trainer import Trainer

PANEL_WIDTH = 340
PAD = 20
CHART_EPISODES = 120
CHART_HEIGHT = 74
CONTROLS = (
    ("Boşluk", "duraklat"), ("+ / −", "hız"),
    ("V", "turbo"), ("T", "sensörler"),
    ("S", "kaydet"), ("R", "yarışı yenile"),
    ("Esc", "menü"),
)


@dataclass(frozen=True)
class HudState:
    paused: bool
    turbo: bool
    steps_per_frame: int
    steps_per_sec: float
    sensor_label: str


def _format_time(seconds: float | None) -> str:
    return "—" if seconds is None else f"{seconds:.2f} s"


def _format_count(n: int) -> str:
    return f"{n / 1000:.1f}k" if n >= 10_000 else str(n)


class Hud:
    def __init__(self, rect: pygame.Rect):
        self.rect = rect

    def draw(self, screen: pygame.Surface, trainer: Trainer, state: HudState) -> None:
        r = self.rect
        pygame.draw.rect(screen, theme.PANEL, r)
        pygame.draw.line(screen, theme.PANEL_EDGE, r.topleft, r.bottomleft)
        x, w = r.x + PAD, r.w - 2 * PAD

        y = self._header(screen, trainer, state, x, r.y + PAD, w)
        y = self._scoreboard(screen, trainer, x, y + 14, w)
        y = self._race(screen, trainer, x, y + 18, w)
        y = self._learning(screen, trainer, state, x, y + 18, w)
        self._charts(screen, trainer, x, y + 18, w)
        self._controls(screen, x, r.bottom - PAD, w)

    # ------------------------------------------------------------------ #
    def _header(self, screen, trainer: Trainer, state: HudState, x, y, w) -> int:
        theme.text(screen, "AI RACING", (x, y), 24, theme.TEXT, bold=True)
        if state.paused:
            label, color = "DURAKLATILDI", theme.ACCENT
        elif trainer.training:
            label, color = ("EĞİTİM · TURBO" if state.turbo else "EĞİTİM"), theme.GOOD
        else:
            label, color = "İZLEME", theme.INFO
        chip = theme.font(12, bold=True).render(label, True, theme.BG)
        chip_rect = chip.get_rect().inflate(14, 6)
        chip_rect.topright = (x + w, y + 6)
        pygame.draw.rect(screen, color, chip_rect, border_radius=chip_rect.h // 2)
        screen.blit(chip, chip.get_rect(center=chip_rect.center))

        env = trainer.env
        info = f"{env.track.name} · {env.num_cars} araba · {env.track.length:.0f} px tur"
        theme.text(screen, info, (x, y + 34), 13, theme.TEXT_DIM)
        return y + 56

    def _scoreboard(self, screen, trainer: Trainer, x, y, w) -> int:
        stats = trainer.stats
        if trainer.training:
            cells = (("BÖLÜM", str(trainer.episode), theme.TEXT),
                     ("TUR", _format_count(stats.total_laps), theme.GOOD),
                     ("EN İYİ TUR", _format_time(stats.best_lap_time), theme.ACCENT))
        else:
            laps = trainer.watch_laps
            cells = (("DENEME", str(len(laps)), theme.TEXT),
                     ("SON TUR", _format_time(laps[-1] if laps else None), theme.GOOD),
                     ("EN İYİ", _format_time(min(laps) if laps else None), theme.ACCENT))
        col = w / 3
        for i, (label, value, color) in enumerate(cells):
            cx = x + i * col
            theme.text(screen, label, (cx, y), 11, theme.TEXT_FAINT, bold=True)
            theme.text(screen, value, (cx, y + 14), 22 if len(value) < 8 else 18, color, kind="mono", bold=True)
        return y + 46

    def _race(self, screen, trainer: Trainer, x, y, w) -> int:
        env = trainer.env
        theme.text(screen, "BU YARIŞ", (x, y), 11, theme.TEXT_FAINT, bold=True)
        theme.text(screen, f"{env.steps / env.sim_fps:5.1f} s", (x + w, y - 2), 14, theme.TEXT_DIM,
                   kind="mono", anchor="topright")
        y += 20

        # Her araba için bir nokta: yarışıyor (dolu), tur tamam (yeşil halka), elendi (soluk).
        per_row = 16
        size = min(14, int(w / per_row) - 4)
        for i in range(env.num_cars):
            cx = x + (i % per_row) * (size + 4) + size / 2
            cy = y + (i // per_row) * (size + 4) + size / 2
            color = theme.CAR_COLORS[i % len(theme.CAR_COLORS)]
            status = env.status[i]
            if status == Status.RUNNING:
                pygame.draw.circle(screen, color, (cx, cy), size / 2)
            elif status == Status.LAP:
                pygame.draw.circle(screen, theme.GOOD, (cx, cy), size / 2, width=3)
            else:
                pygame.draw.circle(screen, theme.lerp_color(color, theme.PANEL, 0.75), (cx, cy), size / 2)
        y += ((env.num_cars - 1) // per_row + 1) * (size + 4) + 8

        leader = env.leader()
        running = int(env.running.sum())
        laps = int((env.status == Status.LAP).sum())
        theme.text(screen, f"Yarışan {running}/{env.num_cars}   ·   Tur tamam {laps}", (x, y), 14, theme.TEXT)
        y += 24
        y = self._bar(screen, "Lider ilerleme", float(env.progress_fraction()[leader]), x, y, w, theme.ACCENT)
        y = self._bar(screen, "Lider hız", float(env.speed[leader] / env.max_speed), x, y, w, theme.INFO)
        return y

    def _bar(self, screen, label, value, x, y, w, color) -> int:
        theme.text(screen, label, (x, y), 13, theme.TEXT_DIM)
        theme.text(screen, f"{value * 100:3.0f}%", (x + w, y), 13, theme.TEXT, kind="mono", anchor="topright")
        track = pygame.Rect(x, y + 20, w, 6)
        pygame.draw.rect(screen, theme.PANEL_EDGE, track, border_radius=3)
        fill = track.copy()
        fill.w = max(0, int(w * min(max(value, 0.0), 1.0)))
        if fill.w:
            pygame.draw.rect(screen, color, fill, border_radius=3)
        return y + 34

    def _learning(self, screen, trainer: Trainer, state: HudState, x, y, w) -> int:
        agent = trainer.agent
        theme.text(screen, "ÖĞRENME" if trainer.training else "MODEL", (x, y), 11, theme.TEXT_FAINT, bold=True)
        y += 20
        speed = "turbo" if state.turbo else f"x{state.steps_per_frame}"
        if trainer.training:
            rows = (
                ("Keşif (ε)", f"{agent.epsilon:.3f}"),
                ("Kayıp", "—" if agent.loss != agent.loss else f"{agent.loss:.4f}"),
                ("Deneyim", f"{_format_count(len(agent.buffer))} / güncelleme {_format_count(agent.updates)}"),
                ("Sim hızı", f"{speed} · {state.steps_per_sec:,.0f} adım/sn"),
                ("Sensörler", state.sensor_label),
            )
        else:
            rows = (
                ("Eğitim adımı", _format_count(agent.env_steps)),
                ("Sim hızı", f"{speed} · {state.steps_per_sec:,.0f} adım/sn"),
                ("Sensörler", state.sensor_label),
            )
        for label, value in rows:
            theme.text(screen, label, (x, y), 14, theme.TEXT_DIM)
            theme.text(screen, value, (x + w, y), 14, theme.TEXT, kind="mono", anchor="topright")
            y += 22
        return y

    def _charts(self, screen, trainer: Trainer, x, y, w) -> int:
        stats = trainer.stats
        theme.text(screen, "GELİŞİM · en iyi ilerleme / bölüm", (x, y), 11, theme.TEXT_FAINT, bold=True)
        y += 18
        self._progress_chart(screen, pygame.Rect(x, y, w, CHART_HEIGHT), stats.history_progress, stats.history_laps)
        y += CHART_HEIGHT + 16
        theme.text(screen, "TUR SÜRESİ · bölüm başına en iyi", (x, y), 11, theme.TEXT_FAINT, bold=True)
        y += 18
        self._lap_chart(screen, pygame.Rect(x, y, w, CHART_HEIGHT), stats.history_lap_time)
        return y + CHART_HEIGHT

    def _progress_chart(self, screen, rect: pygame.Rect, progress, laps) -> None:
        pygame.draw.rect(screen, theme.BG, rect, border_radius=6)
        data = list(zip(progress, laps))[-CHART_EPISODES:]
        if not data:
            theme.text(screen, "Henüz tamamlanan bölüm yok", rect.center, 13, theme.TEXT_FAINT, anchor="center")
            return
        bar_w = rect.w / CHART_EPISODES
        for i, (value, lap_count) in enumerate(data):
            h = max(1, int((rect.h - 8) * value))
            color = theme.GOOD if lap_count else theme.lerp_color(theme.INFO, theme.BG, 0.35)
            pygame.draw.rect(screen, color, (rect.x + i * bar_w, rect.bottom - 4 - h, max(1, bar_w - 1), h))

    def _lap_chart(self, screen, rect: pygame.Rect, lap_times) -> None:
        pygame.draw.rect(screen, theme.BG, rect, border_radius=6)
        recent = lap_times[-CHART_EPISODES:]
        points = [(i, t) for i, t in enumerate(recent) if t is not None]
        if len(points) < 2:
            theme.text(screen, "Tur tamamlanınca burada görünecek", rect.center, 13, theme.TEXT_FAINT, anchor="center")
            return
        times = [t for _, t in points]
        lo, hi = min(times), max(times)
        span = max(hi - lo, 1e-6)
        inner = rect.inflate(-12, -16)

        def to_xy(i: int, t: float) -> tuple[float, float]:
            return inner.x + inner.w * i / max(CHART_EPISODES - 1, 1), inner.bottom - inner.h * (hi - t) / span

        pygame.draw.aalines(screen, theme.ACCENT, False, [to_xy(i, t) for i, t in points])
        best_i, best_t = min(points, key=lambda p: p[1])
        pygame.draw.circle(screen, theme.GOOD, to_xy(best_i, best_t), 4)
        # Kısa tur yukarıda çizilir (iyileşme yukarı doğru); etiketler de buna göre.
        theme.text(screen, f"{lo:.2f}", (rect.x + 4, rect.y + 2), 11, theme.GOOD, kind="mono")
        theme.text(screen, f"{hi:.2f}", (rect.x + 4, rect.bottom - 14), 11, theme.TEXT_FAINT, kind="mono")

    def _controls(self, screen, x, bottom, w) -> None:
        rows = (len(CONTROLS) + 1) // 2
        y = bottom - rows * 20
        pygame.draw.line(screen, theme.PANEL_EDGE, (x, y - 10), (x + w, y - 10))
        col = w / 2
        for i, (key, action) in enumerate(CONTROLS):
            cx = x + (i % 2) * col
            cy = y + (i // 2) * 20
            rect = theme.text(screen, key, (cx, cy), 12, theme.ACCENT, kind="mono", bold=True)
            theme.text(screen, action, (cx + max(rect.w, 44) + 8, cy), 12, theme.TEXT_DIM)
