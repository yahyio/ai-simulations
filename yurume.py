"""AI Walker — deneme yanılmayla yürümeyi/koşmayı öğrenen hümanoid.

Kullanım:
    python yurume.py                                        # pencereli uygulama
    python yurume.py --headless --task kosu --steps 3000000  # pencere olmadan hızlı eğitim
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
TASK_CHOICES = ("yurume", "kosu")
log = logging.getLogger("walker")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deneme yanılmayla yürümeyi öğrenen hümanoid.")
    parser.add_argument("--headless", action="store_true", help="pencere açmadan eğit (en hızlısı)")
    parser.add_argument("--task", choices=TASK_CHOICES, default="yurume", help="yurume veya kosu")
    parser.add_argument("--steps", type=float, default=3e6, help="başsız modda eğitilecek toplam adım")
    parser.add_argument("--fresh", action="store_true", help="kayıtlı modeli yok say, sıfırdan başla")
    args = parser.parse_args(argv)
    if args.steps < 1:
        parser.error("--steps en az 1 olmalı")
    return args


def run_headless(task: str, steps: float, fresh: bool) -> int:
    from walker.trainer import Mode, WalkTrainer

    try:
        trainer = WalkTrainer(task, mode=Mode.TRAIN, resume=not fresh)
    except ValueError as exc:
        log.error("Başlatılamadı: %s", exc)
        return 1
    first = trainer.ppo.total_steps
    target = first + int(steps)
    log.info("Eğitim: %s · %d paralel karakter · hedef %.1fM adım",
             trainer.task.label, trainer.env.num_envs, target / 1e6)
    start, last_update = time.time(), trainer.ppo.updates
    try:
        while trainer.ppo.total_steps < target:
            trainer.tick()
            if trainer.ppo.updates != last_update and trainer.ppo.updates % 10 == 0:
                last_update = trainer.ppo.updates
                s = trainer.stats
                rate = (trainer.ppo.total_steps - first) / max(time.time() - start, 1e-9)
                log.info("%5.2fM adım | deneme %6d | son 100 ort. %5.1f m | en uzak %5.1f m | en hızlı %.2f m/s | %4.0f adım/sn",
                         trainer.ppo.total_steps / 1e6, s.episodes, s.recent_mean_distance, s.best_distance,
                         s.best_speed, rate)
    except KeyboardInterrupt:
        log.info("Durduruldu.")
    finally:
        trainer.save()
    log.info("Kaydedildi: %s", trainer.paths["model"])
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    os.chdir(PROJECT_DIR)
    args = parse_args(argv)
    if args.headless:
        return run_headless(args.task, args.steps, args.fresh)
    from walker.app import WalkerApp

    WalkerApp().run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
