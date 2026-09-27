"""AI Araba Simülasyonu — giriş noktası.

Kullanım:
    python main.py                           # pencereli uygulama (menü)
    python main.py --headless --episodes 300  # pencere olmadan hızlı eğitim
    python main.py --headless --resume        # kayıtlı modelden devam
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
log = logging.getLogger("racing")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pekiştirmeli öğrenme ile kendi kendine süren arabalar.")
    parser.add_argument("--headless", action="store_true", help="pencere açmadan eğit (en hızlısı)")
    parser.add_argument("--episodes", type=int, default=200, help="başsız modda eğitilecek bölüm sayısı")
    parser.add_argument("--resume", action="store_true", help="başsız modda kayıtlı modelden devam et")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="yapılandırma dosyası")
    parser.add_argument("--new-track", choices=("kolay", "orta", "zor", "rastgele"),
                        help="rastgele yeni bir pist üret ve onu kullan")
    parser.add_argument("--seed", type=int, help="--new-track için tohum (aynı tohum = aynı pist)")
    args = parser.parse_args(argv)
    if args.episodes < 1:
        parser.error("--episodes en az 1 olmalı")
    return args


def run_headless(config: dict, episodes: int, resume: bool) -> int:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    from racing.track import Track, TrackError
    from racing.trainer import Trainer

    try:
        track = Track.from_file(config["track"]["file"], config["track"])
        trainer = Trainer(config, track, resume=resume)
    except (TrackError, ValueError) as exc:
        log.error("Başlatılamadı: %s", exc)
        return 1

    target = trainer.episode + episodes
    log.info("Eğitim: %d bölüm · %d araba · cihaz %s · pist %.0f px",
             episodes, trainer.env.num_cars, trainer.agent.device, track.length)
    try:
        while trainer.episode < target:
            summary = trainer.step().episode
            if summary is None:
                continue
            best = f"{summary.best_lap_time:6.2f} s" if summary.best_lap_time is not None else "     —  "
            log.info("bölüm %5d | tur %2d/%d | en iyi %s | ilerleme %3.0f%% | ε %.3f",
                     summary.episode, summary.laps, trainer.env.num_cars, best,
                     summary.best_progress * 100, summary.epsilon)
    except KeyboardInterrupt:
        log.info("Durduruldu.")
    finally:
        trainer.save()

    stats = trainer.stats
    best = f"{stats.best_lap_time:.2f} s" if stats.best_lap_time is not None else "—"
    log.info("Bitti: toplam %d tur, en iyi tur %s. Model: data/model.pt", stats.total_laps, best)
    return 0


def use_new_track(config: dict, difficulty: str, seed: int | None, config_path: Path) -> dict | None:
    """Pist üretir, kaydeder ve yapılandırmayı ona yönlendirir."""
    from racing import trackgen
    from racing.config import deep_merge, save_config
    from racing.track import TrackError

    try:
        generated = trackgen.generate(difficulty, seed, config["track"])
    except TrackError as exc:
        log.error("Pist üretilemedi: %s", exc)
        return None
    path = trackgen.save(generated)
    updated = deep_merge(config, {"track": {"file": path.as_posix(), "reverse": False}})
    save_config(updated, config_path)
    log.info("Yeni pist: %s (%s, tohum %d, genişlik %d px)",
             path, generated.difficulty.label, generated.seed, generated.width)
    return updated


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    os.chdir(PROJECT_DIR)  # göreli yollar (track.png, data/, logs/) her yerden çalışsın
    args = parse_args(argv)

    from racing.config import ConfigError, load_config

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        log.error("Yapılandırma hatası: %s", exc)
        return 2

    if args.new_track:
        config = use_new_track(config, args.new_track, args.seed, args.config)
        if config is None:
            return 1

    if args.headless:
        return run_headless(config, args.episodes, args.resume)

    from racing.app import App

    App(config).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
