"""AI Walker 3D penceresi: menü → eğitim / izleme → menü."""
from __future__ import annotations

import textwrap
import time
from dataclasses import dataclass

import pygame

from common import theme
from common.widgets import MainMenu, MenuItem, confirm, draw_loading, show_message

from .env import TASKS
from .hud import PANEL_WIDTH, HudState, WalkHud
from .model import MAX_LANES
from .render import PRESETS, Stadium
from .trainer import (
    IncompatibleModelError,
    Mode,
    WalkTrainer,
    has_legacy_model,
    has_model,
    load_stats,
    reset_task,
    watch_source,
)

WINDOW = (1440, 810)
FPS = 60
SPEEDS = (1, 2, 4, 8)
UPDATE_BUDGET_S = 0.012   # normal hızda, öğrenme için kare başına ayrılan süre
TURBO_BUDGET_S = 0.045
TOAST_SECONDS = 3.0
RECORD_TOAST_STEP_M = 1.0  # rekor bildirimi için en az iyileşme
MENU_ORBIT_DEG = 0.12      # menü arka planında kameranın kare başına dönüşü
NO_GL_HINT = "Pencere olmadan eğitim yine de çalışır:\npython -m walker --headless"


def describe_error(exc: Exception) -> str:
    """Hata iletişim kutusuna sığacak kısa açıklama."""
    return textwrap.shorten(f"{type(exc).__name__}: {exc}", width=72, placeholder="…")


@dataclass
class Toast:
    text: str
    color: tuple
    expires: float


class WalkerApp:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(WINDOW, pygame.SCALED)
        pygame.display.set_caption("AI Walker 3D — deneme yanılmayla yürümeyi öğrenen hümanoidler")
        self.clock = pygame.time.Clock()
        w, h = WINDOW
        self.panel = pygame.Rect(w - PANEL_WIDTH, 0, PANEL_WIDTH, h)
        self.viewport = pygame.Rect(0, 0, w - PANEL_WIDTH, h)
        self._menu_stadium: Stadium | None = None
        self._render_error: str | None = None   # 3B görüntü açılamadıysa nedeni

    def run(self) -> None:
        try:
            while self._menu_round():
                pass
        finally:
            if self._menu_stadium is not None:
                self._menu_stadium.close()
            pygame.quit()

    # ------------------------------------------------------------------ #
    def _backdrop(self, screen: pygame.Surface) -> None:
        if self._menu_stadium is None and self._render_error is None:
            try:
                stadium = Stadium(WINDOW, MAX_LANES)
            except Exception as exc:   # OpenGL bağlamı yok: sürücü, uzak masaüstü, ekransız makine…
                self._render_error = describe_error(exc)
            else:
                stadium.camera.lookat[:] = (1.5, stadium.track_center_y, 1.0)
                stadium.camera.azimuth, stadium.camera.elevation, stadium.camera.distance = 150.0, -9.0, 8.5
                self._menu_stadium = stadium
        stadium = self._menu_stadium
        if stadium is None:
            screen.fill(theme.BG)
            return
        stadium.camera.azimuth = (stadium.camera.azimuth + MENU_ORBIT_DEG) % 360
        screen.blit(stadium.render(), (0, 0))

    def _menu_round(self) -> bool:
        items = []
        for key, verb in (("yurume", "Yürümeyi"), ("kosu", "Koşmayı")):
            if has_model(key):
                stats = load_stats(key)
                hint = f"kayıtlı modelden devam · {stats.episodes} deneme, en uzak {stats.best_distance:.1f} m"
            elif has_legacy_model(key):
                hint = "eski (2B) kayıt bulundu — onayla silinip sıfırdan başlar"
            else:
                hint = "sıfırdan — ne ayakta durmayı ne adım atmayı biliyor"
            items.append(MenuItem(f"train:{key}", f"{verb} öğren", hint))
        for key, noun in (("yurume", "Yürüyüşü"), ("kosu", "Koşuyu")):
            source = watch_source(key)
            hint = ("önce eğitim gerekli" if source is None else
                    "depoyla gelen hazır model" if "models" in source.parts else "en iyi model, 8 kulvarda")
            items.append(MenuItem(f"watch:{key}", f"{noun} izle", hint, source is not None))
        any_model = any(has_model(k) or has_legacy_model(k) for k in TASKS)
        items.append(MenuItem("reset", "Verileri sıfırla", "bu bilgisayarda eğitilen modelleri sil", any_model))
        items.append(MenuItem("quit", "Çıkış", "kaydet ve kapat"))

        choice = MainMenu(self.screen, self.clock, self._backdrop, [], title="AI WALKER 3D",
                          subtitle="Hiçbir şey bilmeden, deneme yanılmayla yürümeyi öğrenen hümanoidler").run(items)
        if choice == "quit":
            return False
        if choice == "reset":
            if confirm(self.screen, self.clock, "Verileri sıfırla",
                       "Yürüme ve koşu modelleri, istatistikler ve günlükler silinecek.\n"
                       "(Depoyla gelen hazır modeller korunur.)"):
                removed = sum(len(reset_task(k)) for k in TASKS)
                show_message(self.screen, self.clock, "Sıfırlandı", f"{removed} dosya silindi.", theme.GOOD)
            return True
        action, task = choice.split(":")
        return self._session(task, Mode.TRAIN if action == "train" else Mode.WATCH)

    def _session(self, task: str, mode: Mode) -> bool:
        if self._render_error is not None:
            show_message(self.screen, self.clock, "3B görüntü açılamadı", f"{self._render_error}\n{NO_GL_HINT}",
                         theme.BAD)
            return True
        draw_loading(self.screen, "Hümanoidler hazırlanıyor…", "AI WALKER 3D")
        try:
            trainer = WalkTrainer(task, mode=mode, resume=True)
        except IncompatibleModelError:
            if not confirm(self.screen, self.clock, "Eski kayıt uyumsuz",
                           "Bu bilgisayardaki kayıt eski 2B sürümden kalma ve 3B ile kullanılamaz.\n"
                           "Silinip sıfırdan başlansın mı?"):
                return True
            reset_task(task)
            return self._session(task, mode)
        except (FileNotFoundError, ValueError, OSError) as exc:
            show_message(self.screen, self.clock, "Başlatılamadı", str(exc), theme.BAD)
            return True
        try:
            session = WalkSession(self, trainer)
        except Exception as exc:   # 3B görüntü açılamadı: eğitim süreçleri açık kalmasın
            trainer.close()
            show_message(self.screen, self.clock, "3B görüntü açılamadı", f"{describe_error(exc)}\n{NO_GL_HINT}",
                         theme.BAD)
            return True
        try:
            return session.run()
        finally:
            session.close()


class WalkSession:
    def __init__(self, app: WalkerApp, trainer: WalkTrainer):
        self.app = app
        self.trainer = trainer
        self.hud = WalkHud(app.panel)
        self.speed_index = 1 if trainer.training else 0
        self.turbo = False
        self.paused = False
        self.dragging = False
        self.toasts: list[Toast] = []
        self._last_record_toast = trainer.stats.best_distance
        self._rate_steps, self._rate_start, self.steps_per_sec = 0, time.perf_counter(), 0.0
        # En son: açılamazsa (OpenGL yok) geride kapatılması gereken başka bir şey kalmasın.
        self.stadium = Stadium(app.viewport.size, trainer.display_count)

    def run(self) -> bool:
        """Oturumu çalıştırır. Uygulama kapatılacaksa False döner."""
        try:
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return False
                    if event.type == pygame.KEYDOWN and not self._on_key(event.key):
                        return True
                    self._on_mouse(event)
                dt = self.app.clock.tick(FPS) / 1000
                if not self.paused:
                    self._advance()
                self._draw(dt)
                pygame.display.flip()
        finally:
            self.trainer.save()

    def close(self) -> None:
        self.stadium.close()
        self.trainer.close()

    # ------------------------------------------------------------------ #
    def _on_key(self, key: int) -> bool:
        if key == pygame.K_ESCAPE:
            return False
        if key == pygame.K_SPACE:
            self.paused = not self.paused
        elif key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
            self.turbo, self.speed_index = False, min(self.speed_index + 1, len(SPEEDS) - 1)
        elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.turbo, self.speed_index = False, max(self.speed_index - 1, 0)
        elif key == pygame.K_v:
            self.turbo = not self.turbo
            self._toast("Turbo açık — en yüksek hız" if self.turbo else "Turbo kapalı", theme.INFO)
        elif key == pygame.K_c:
            self._toast(f"Kamera: {self.stadium.cycle_preset()}", theme.INFO)
        elif key == pygame.K_s and self.trainer.training:
            self.trainer.save()
            self._toast("Model ve istatistikler kaydedildi", theme.GOOD)
        return True

    def _on_mouse(self, event: pygame.event.Event) -> None:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.app.viewport.collidepoint(event.pos):
            self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self.stadium.orbit(*event.rel)
        elif event.type == pygame.MOUSEWHEEL:
            self.stadium.zoom(event.y)

    def _advance(self) -> None:
        start = time.perf_counter()
        env_steps = 0
        if self.turbo:
            while time.perf_counter() - start < TURBO_BUDGET_S:
                env_steps += self._tick()
        else:
            target = SPEEDS[self.speed_index]
            while env_steps < target:
                if self.trainer.updating and time.perf_counter() - start > UPDATE_BUDGET_S:
                    break  # öğrenme kalan karelere yayılır; pencere donmaz
                env_steps += self._tick()
        self._rate_steps += env_steps * self.trainer.env.num_envs
        elapsed = time.perf_counter() - self._rate_start
        if elapsed >= 1.0:
            self.steps_per_sec = self._rate_steps / elapsed
            self._rate_steps, self._rate_start = 0, time.perf_counter()

    def _tick(self) -> int:
        """Bir iş birimi yürütür; ortam adımıysa 1 döner."""
        tick = self.trainer.tick()
        if tick.updating or tick.update_done:
            return 0
        if not self.trainer.training:
            for ep in tick.finished:
                verdict = "düştü" if ep.fell else "süre doldu"
                self._toast(f"Kulvar {ep.env + 1}: {ep.distance:.1f} m · {ep.speed:.2f} m/s · {verdict}",
                            theme.BAD if ep.fell else theme.GOOD)
        best = self.trainer.stats.best_distance
        if tick.new_record and best >= self._last_record_toast + RECORD_TOAST_STEP_M:
            self._last_record_toast = best
            self._toast(f"Yeni rekor: {best:.1f} m", theme.ACCENT)
        return 1

    def _toast(self, text: str, color) -> None:
        now = time.perf_counter()
        self.toasts = [t for t in self.toasts if t.expires > now][-3:] + [Toast(text, color, now + TOAST_SECONDS)]

    # ------------------------------------------------------------------ #
    def _draw(self, dt: float) -> None:
        screen = self.app.screen
        trainer = self.trainer
        self.stadium.set_poses(trainer.qpos[: trainer.display_count])
        leader = trainer.leader()
        self.stadium.follow(leader, dt)
        record = trainer.stats.best_distance if trainer.training else None
        screen.blit(self.stadium.render(record, marker_lane=leader), self.app.viewport.topleft)
        info = trainer.runner(leader)
        self.stadium.draw_labels(screen, self.app.viewport.topleft, record, leader,
                                 f"Kulvar {leader + 1} · {info.distance:.1f} m")
        self.hud.draw(screen, trainer, HudState(self.paused, self.turbo, SPEEDS[self.speed_index],
                                                self.steps_per_sec, PRESETS[self.stadium.preset].name))
        if self.paused:
            self._banner(screen, "DURAKLATILDI", "Devam etmek için Boşluk")
        self._draw_toasts(screen)

    def _banner(self, screen, title: str, subtitle: str) -> None:
        box = pygame.Rect(0, 0, 360, 96)
        box.center = (self.app.viewport.centerx, 120)
        veil = pygame.Surface(box.size, pygame.SRCALPHA)
        pygame.draw.rect(veil, theme.OVERLAY, veil.get_rect(), border_radius=14)
        screen.blit(veil, box)
        theme.text(screen, title, (box.centerx, box.y + 18), 28, theme.ACCENT, bold=True, anchor="midtop")
        theme.text(screen, subtitle, (box.centerx, box.y + 60), 14, theme.TEXT_DIM, anchor="midtop")

    def _draw_toasts(self, screen) -> None:
        now = time.perf_counter()
        self.toasts = [t for t in self.toasts if t.expires > now]
        y = 18
        for toast in self.toasts:
            fade = min(1.0, (toast.expires - now) / 0.4)
            rendered = theme.font(16, bold=True).render(toast.text, True, toast.color)
            box = rendered.get_rect().inflate(28, 16)
            box.midtop = (self.app.viewport.centerx, y)
            veil = pygame.Surface(box.size, pygame.SRCALPHA)
            pygame.draw.rect(veil, (*theme.PANEL, int(230 * fade)), veil.get_rect(), border_radius=box.h // 2)
            screen.blit(veil, box)
            rendered.set_alpha(int(255 * fade))
            screen.blit(rendered, rendered.get_rect(center=box.center))
            y += box.h + 8
