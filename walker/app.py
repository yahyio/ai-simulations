"""Yürüme simülasyonu penceresi: menü → eğitim / izleme → menü."""
from __future__ import annotations

import time
from dataclasses import dataclass

import mujoco
import pygame

from racing import theme
from racing.ui import MainMenu, MenuItem, confirm, draw_loading, show_message

from .env import TASKS
from .hud import PANEL_WIDTH, HudState, WalkHud
from .model import load_model
from .render import SHIRTS, Scene
from .trainer import Mode, WalkTrainer, has_model, load_stats, reset_task

WINDOW = (1440, 810)
FPS = 60
SPEEDS = (1, 2, 4, 8)
UPDATE_BUDGET_S = 0.012   # normal hızda, öğrenme için kare başına ayrılan süre
TURBO_BUDGET_S = 0.045
TOAST_SECONDS = 3.0
RECORD_TOAST_STEP_M = 1.0  # rekor bildirimi için en az iyileşme


@dataclass
class Toast:
    text: str
    color: tuple
    expires: float


class WalkerApp:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(WINDOW, pygame.SCALED)
        pygame.display.set_caption("AI Walker — deneme yanılmayla yürümeyi öğrenen hümanoid")
        self.clock = pygame.time.Clock()
        w, h = WINDOW
        self.panel = pygame.Rect(w - PANEL_WIDTH, 0, PANEL_WIDTH, h)
        self.viewport = pygame.Rect(0, 0, w - PANEL_WIDTH, h)
        model = load_model()
        self._idle = mujoco.MjData(model)
        mujoco.mj_forward(model, self._idle)
        self._menu_scene = Scene(pygame.Rect(0, 0, w, h), model, SHIRTS["yurume"])
        self._menu_scene.camera_x = -2.2

    def run(self) -> None:
        try:
            while self._menu_round():
                pass
        finally:
            pygame.quit()

    # ------------------------------------------------------------------ #
    def _backdrop(self, screen: pygame.Surface) -> None:
        self._menu_scene.draw_world(screen)
        self._menu_scene.draw_character(screen, self._idle)

    def _menu_round(self) -> bool:
        items = []
        for key, verb in (("yurume", "Yürümeyi"), ("kosu", "Koşmayı")):
            stats = load_stats(key)
            hint = (f"kayıtlı modelden devam · {stats.episodes} deneme, en uzak {stats.best_distance:.1f} m"
                    if has_model(key) else "sıfırdan — ne ayakta durmayı ne adım atmayı biliyor")
            items.append(MenuItem(f"train:{key}", f"{verb} öğren", hint))
        for key, noun in (("yurume", "Yürüyüşü"), ("kosu", "Koşuyu")):
            ok = has_model(key)
            items.append(MenuItem(f"watch:{key}", f"{noun} izle",
                                  "en iyi model, keşif kapalı" if ok else "önce eğitim gerekli", ok))
        any_model = any(has_model(k) for k in TASKS)
        items.append(MenuItem("reset", "Verileri sıfırla", "tüm yürüme/koşu modellerini sil", any_model))
        items.append(MenuItem("quit", "Çıkış", "kaydet ve kapat"))

        choice = MainMenu(self.screen, self.clock, self._backdrop, [], title="AI WALKER",
                          subtitle="Hiçbir şey bilmeden, deneme yanılmayla yürümeyi öğrenen hümanoid").run(items)
        if choice == "quit":
            return False
        if choice == "reset":
            if confirm(self.screen, self.clock, "Verileri sıfırla",
                       "Yürüme ve koşu modelleri, istatistikler ve günlükler silinecek."):
                removed = sum(len(reset_task(k)) for k in TASKS)
                show_message(self.screen, self.clock, "Sıfırlandı", f"{removed} dosya silindi.", theme.GOOD)
            return True
        action, task = choice.split(":")
        return self._session(task, Mode.TRAIN if action == "train" else Mode.WATCH)

    def _session(self, task: str, mode: Mode) -> bool:
        draw_loading(self.screen, "Hümanoid hazırlanıyor…")
        try:
            trainer = WalkTrainer(task, mode=mode, resume=True)
        except (FileNotFoundError, ValueError) as exc:
            show_message(self.screen, self.clock, "Başlatılamadı", str(exc), theme.BAD)
            return True
        return WalkSession(self, trainer).run()


class WalkSession:
    def __init__(self, app: WalkerApp, trainer: WalkTrainer):
        self.app = app
        self.trainer = trainer
        self.scene = Scene(app.viewport, trainer.env.model, SHIRTS[trainer.task.key])
        self.hud = WalkHud(app.panel)
        self.speed_index = 0 if not trainer.training else 1
        self.turbo = False
        self.paused = False
        self.ghosts = True
        self.toasts: list[Toast] = []
        self._last_record_toast = trainer.stats.best_distance
        self._rate_steps, self._rate_start, self.steps_per_sec = 0, time.perf_counter(), 0.0

    def run(self) -> bool:
        try:
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return False
                    if event.type == pygame.KEYDOWN and not self._on_key(event.key):
                        return True
                dt = self.app.clock.tick(FPS) / 1000
                if not self.paused:
                    self._advance()
                self._draw(dt)
                pygame.display.flip()
        finally:
            self.trainer.save()

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
        elif key == pygame.K_g:
            self.ghosts = not self.ghosts
        elif key == pygame.K_s and self.trainer.training:
            self.trainer.save()
            self._toast("Model ve istatistikler kaydedildi", theme.GOOD)
        return True

    # ------------------------------------------------------------------ #
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
                self._toast(f"{ep.distance:.1f} m · {ep.speed:.2f} m/s · {verdict}", theme.BAD if ep.fell else theme.GOOD)
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
        env = self.trainer.env
        leader = self.trainer.leader()
        self.scene.follow(env.datas[leader].qpos[0], dt)
        record = self.trainer.stats.best_distance if self.trainer.training else None
        self.scene.draw_world(screen, record)
        if self.ghosts and env.num_envs > 1:
            self.scene.draw_ghosts(screen, env.datas, skip=leader)
        self.scene.draw_character(screen, env.datas[leader])
        self.hud.draw(screen, self.trainer, HudState(self.paused, self.turbo, SPEEDS[self.speed_index],
                                                     self.steps_per_sec, self.ghosts))
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
