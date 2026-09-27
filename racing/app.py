"""Pencereli uygulama: menü → eğitim / izleme oturumu → menü."""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import pygame

from common import theme
from common.widgets import MainMenu, MenuItem, confirm, draw_loading, show_message

from . import trackgen
from .config import DEFAULTS, save_config
from .hud import PANEL_WIDTH, Hud, HudState
from .picker import TrackPicker
from .render import SENSOR_MODE_LABELS, SENSORS_ALL, SENSORS_LEADER, SENSORS_OFF, Renderer
from .storage import EPISODE_LOG_PATH, STATS_PATH, has_model, load_stats, reset_all
from .track import Track, TrackError
from .trainer import Mode, StepEvents, Trainer
from .ui import SettingsScreen

TITLE = "AI RACING"
SUBTITLE = "Pekiştirmeli öğrenme ile kendi kendine süren arabalar"
SPEEDS = (1, 2, 4, 8, 16)
TURBO_BUDGET_S = 0.045      # Turbo modda kare başına simülasyona ayrılan süre
TOAST_SECONDS = 3.0
STEP_RATE_WINDOW_S = 1.0


@dataclass
class Toast:
    text: str
    color: tuple
    expires: float


class App:
    def __init__(self, config: dict):
        pygame.init()
        self.config = config
        display = config["display"]
        self.screen = pygame.display.set_mode((display["width"], display["height"]), pygame.SCALED)
        pygame.display.set_caption("AI Racing — kendi kendine süren arabalar")
        self.clock = pygame.time.Clock()
        w, h = self.screen.get_size()
        self.panel_rect = pygame.Rect(w - PANEL_WIDTH, 0, PANEL_WIDTH, h)
        self.viewport = pygame.Rect(0, 0, w - PANEL_WIDTH, h)
        self.track: Track | None = None
        self.renderer: Renderer | None = None

    # ------------------------------------------------------------------ #
    def run(self) -> None:
        try:
            if not self._load_track():
                return
            while self._menu_round():
                pass
        finally:
            pygame.quit()

    def _load_track(self) -> bool:
        draw_loading(self.screen, "Pist analiz ediliyor…", TITLE)
        try:
            self.track = Track.from_file(self.config["track"]["file"], self.config["track"])
        except TrackError as exc:
            show_message(self.screen, self.clock, "Pist yüklenemedi", str(exc), theme.BAD)
            return False
        car = self.config["car"]
        self.renderer = Renderer(self.track, self.viewport, car["length"], car["width"])
        return True

    def _menu_round(self) -> bool:
        """Menüyü bir kez gösterir ve seçimi uygular. Çıkılacaksa False döner."""
        model = has_model()
        any_data = model or STATS_PATH.exists() or EPISODE_LOG_PATH.exists()
        items = [
            MenuItem("train", "Yeni eğitim", "Sıfırdan öğrenmeye başla"),
            MenuItem("resume", "Eğitime devam et",
                     "Kayıtlı modelden sürdür" if model else "Henüz kayıtlı model yok", model),
            MenuItem("watch", "En iyi modeli izle",
                     "Keşif kapalı, tek araba" if model else "Önce eğitim gerekli", model),
            MenuItem("track", "Pist oluştur", f"Rastgele, zorluk seçilebilir · şu an: {self.track.name}"),
            MenuItem("settings", "Ayarlar", "Fizik, öğrenme ve görünüm"),
            MenuItem("reset", "Verileri sıfırla", "Model, istatistik ve günlükleri sil", any_data),
            MenuItem("quit", "Çıkış", "Kaydet ve kapat"),
        ]
        menu = MainMenu(self.screen, self.clock, self.renderer.draw_cover, self._summary(), TITLE, SUBTITLE)
        choice = menu.run(items)

        if choice == "quit":
            return False
        if choice == "train":
            if any_data and not confirm(self.screen, self.clock, "Yeni eğitim",
                                        "Mevcut model, istatistikler ve günlükler silinecek.\nDevam edilsin mi?"):
                return True
            reset_all()
            return self._session(Mode.TRAIN, resume=False)
        if choice == "resume":
            return self._session(Mode.TRAIN, resume=True)
        if choice == "watch":
            return self._session(Mode.WATCH, resume=False)
        if choice == "track":
            self._pick_track()
        elif choice == "settings":
            self._settings()
        elif choice == "reset" and confirm(self.screen, self.clock, "Verileri sıfırla",
                                           "Model, istatistikler ve günlükler kalıcı olarak silinecek."):
                removed = reset_all()
                show_message(self.screen, self.clock, "Sıfırlandı", f"{len(removed)} dosya silindi.", theme.GOOD)
        return True

    def _summary(self) -> list[tuple[str, str]]:
        stats = load_stats()
        if not stats.episodes:
            return []
        best = f"{stats.best_lap_time:.2f} s" if stats.best_lap_time is not None else "—"
        return [("BÖLÜM", str(stats.episodes)), ("TUR", str(stats.total_laps)), ("EN İYİ", best)]

    def _pick_track(self) -> None:
        pick = TrackPicker(self.screen, self.clock, self.config["track"]).run()
        if pick is None:
            return
        updated = copy.deepcopy(self.config)
        if pick.generated is None:
            updated["track"]["file"] = DEFAULTS["track"]["file"]
        else:
            updated["track"]["file"] = trackgen.save(pick.generated).as_posix()
            updated["track"]["reverse"] = False
        save_config(updated)
        self.config.clear()
        self.config.update(updated)
        if pick.track is None:
            if not self._load_track():
                return
        else:
            car = self.config["car"]
            self.track = pick.track
            self.renderer = Renderer(self.track, self.viewport, car["length"], car["width"])
        show_message(self.screen, self.clock, f"Pist hazır: {self.track.name}",
                     "Eğitime devam et: mevcut model yeni pisti sürmeyi dener (sensörlerle genelleme).\n"
                     "Yeni eğitim: sıfırdan öğrenir. Tur rekorları piste özeldir.", theme.GOOD)

    def _settings(self) -> None:
        updated = SettingsScreen(self.screen, self.clock).run(self.config)
        if updated is None or updated == self.config:
            return
        track_changed = updated["track"] != self.config["track"]
        car_changed = updated["car"] != self.config["car"]
        save_config(updated)
        self.config.clear()
        self.config.update(updated)
        if track_changed:
            self._load_track()
        elif car_changed and self.track is not None:
            car = self.config["car"]
            self.renderer = Renderer(self.track, self.viewport, car["length"], car["width"])

    def _session(self, mode: Mode, resume: bool) -> bool:
        draw_loading(self.screen, "Model hazırlanıyor…", TITLE)
        try:
            trainer = Trainer(self.config, self.track, mode=mode, resume=resume)
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            show_message(self.screen, self.clock, "Başlatılamadı", str(exc), theme.BAD)
            return True
        return Session(self, trainer).run()


class Session:
    """Tek bir eğitim ya da izleme oturumu (Esc ile menüye döner)."""

    def __init__(self, app: App, trainer: Trainer):
        self.app = app
        self.trainer = trainer
        self.hud = Hud(app.panel_rect)
        display = app.config["display"]
        self.fps = display["fps"]
        start_speed = app.config["simulation"]["steps_per_frame"] if trainer.training else 1
        self.speed_index = min(range(len(SPEEDS)), key=lambda i: abs(SPEEDS[i] - start_speed))
        self.sensor_mode = SENSORS_LEADER if display["show_sensors"] else SENSORS_OFF
        self.show_trail = display["show_trail"]
        self.paused = False
        self.turbo = False
        self.toasts: list[Toast] = []
        self._rate_steps = 0
        self._rate_start = time.perf_counter()
        self.steps_per_sec = 0.0

    def run(self) -> bool:
        """Oturumu çalıştırır. Uygulama kapatılacaksa False döner."""
        try:
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return False
                    if event.type == pygame.KEYDOWN and not self._on_key(event.key):
                        return True
                if not self.paused:
                    self._advance()
                self._draw()
                pygame.display.flip()
                self.app.clock.tick(self.fps)
        finally:
            self.trainer.save()

    # ------------------------------------------------------------------ #
    def _on_key(self, key: int) -> bool:
        """Tuşu işler; menüye dönülecekse False."""
        if key == pygame.K_ESCAPE:
            return False
        if key == pygame.K_SPACE:
            self.paused = not self.paused
        elif key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
            self.turbo = False
            self.speed_index = min(self.speed_index + 1, len(SPEEDS) - 1)
        elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.turbo = False
            self.speed_index = max(self.speed_index - 1, 0)
        elif key == pygame.K_v:
            self.turbo = not self.turbo
            self._toast("Turbo açık — en yüksek hız" if self.turbo else "Turbo kapalı", theme.INFO)
        elif key == pygame.K_t:
            self.sensor_mode = (self.sensor_mode + 1) % (SENSORS_ALL + 1)
            self._toast(f"Sensörler: {SENSOR_MODE_LABELS[self.sensor_mode]}", theme.INFO)
        elif key == pygame.K_s and self.trainer.training:
            self.trainer.save()
            self._toast("Model ve istatistikler kaydedildi", theme.GOOD)
        elif key == pygame.K_r:
            self.trainer.restart_episode()
            self._toast("Yarış yeniden başlatıldı", theme.INFO)
        return True

    def _advance(self) -> None:
        if self.turbo:
            deadline = time.perf_counter() + TURBO_BUDGET_S
            steps = 0
            while time.perf_counter() < deadline:
                self._handle(self.trainer.step())
                steps += 1
        else:
            steps = SPEEDS[self.speed_index]
            for _ in range(steps):
                self._handle(self.trainer.step())
        self._count_steps(steps)

    def _count_steps(self, steps: int) -> None:
        self._rate_steps += steps
        elapsed = time.perf_counter() - self._rate_start
        if elapsed >= STEP_RATE_WINDOW_S:
            self.steps_per_sec = self._rate_steps / elapsed
            self._rate_steps = 0
            self._rate_start = time.perf_counter()

    def _handle(self, events: StepEvents) -> None:
        summary = events.episode
        if summary is None:
            return
        if not self.trainer.training:
            if summary.best_lap_time is not None:
                self._toast(f"Tur tamamlandı: {summary.best_lap_time:.2f} s", theme.GOOD)
            else:
                self._toast(f"Tur tamamlanamadı (%{summary.best_progress * 100:.0f})", theme.BAD)
            return
        if summary.laps and self.trainer.stats.first_lap_episode == summary.episode:
            self._toast(f"İlk tur tamamlandı! (bölüm {summary.episode})", theme.GOOD)
        if events.new_record:
            self._toast(f"Yeni tur rekoru: {summary.best_lap_time:.2f} s", theme.ACCENT)

    def _toast(self, text: str, color) -> None:
        now = time.perf_counter()
        self.toasts = [t for t in self.toasts if t.expires > now][-3:] + [Toast(text, color, now + TOAST_SECONDS)]

    # ------------------------------------------------------------------ #
    def _draw(self) -> None:
        screen = self.app.screen
        screen.fill(theme.BG)
        now = time.perf_counter()
        self.app.renderer.draw(screen, self.trainer.env, self.sensor_mode, self.show_trail, now)
        self.hud.draw(screen, self.trainer, HudState(
            paused=self.paused,
            turbo=self.turbo,
            steps_per_frame=SPEEDS[self.speed_index],
            steps_per_sec=self.steps_per_sec,
            sensor_label=SENSOR_MODE_LABELS[self.sensor_mode],
        ))
        if self.paused:
            self._banner(screen, "DURAKLATILDI", "Devam etmek için Boşluk")
        self._draw_toasts(screen, now)

    def _banner(self, screen, title: str, subtitle: str) -> None:
        box = pygame.Rect(0, 0, 360, 96)
        box.center = self.app.viewport.center
        veil = pygame.Surface(box.size, pygame.SRCALPHA)
        pygame.draw.rect(veil, theme.OVERLAY, veil.get_rect(), border_radius=14)
        screen.blit(veil, box)
        theme.text(screen, title, (box.centerx, box.y + 18), 28, theme.ACCENT, bold=True, anchor="midtop")
        theme.text(screen, subtitle, (box.centerx, box.y + 60), 14, theme.TEXT_DIM, anchor="midtop")

    def _draw_toasts(self, screen, now: float) -> None:
        self.toasts = [t for t in self.toasts if t.expires > now]
        y = 18
        for toast in self.toasts:
            fade = min(1.0, (toast.expires - now) / 0.4)
            rendered = theme.font(16, bold=True).render(toast.text, True, toast.color)
            box = rendered.get_rect().inflate(28, 16)
            box.midtop = (self.app.viewport.centerx, y)
            veil = pygame.Surface(box.size, pygame.SRCALPHA)
            pygame.draw.rect(veil, (*theme.PANEL, int(230 * fade)), veil.get_rect(), border_radius=box.h // 2)
            pygame.draw.rect(veil, (*toast.color, int(200 * fade)), veil.get_rect(), width=1, border_radius=box.h // 2)
            screen.blit(veil, box)
            rendered.set_alpha(int(255 * fade))
            screen.blit(rendered, rendered.get_rect(center=box.center))
            y += box.h + 8
