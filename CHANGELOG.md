# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to
[Semantic Versioning](https://semver.org/).

## [2.0.0] — 2026-09-27

### Added

- **AI Walker 3D** — a 21-joint MuJoCo humanoid that learns to walk and run from scratch with PPO.
  - Multi-process vectorized environments: 48 humanoids at 4 000–6 000 steps/s on a 6-core CPU.
  - Reward built from task goals and anti-exploit terms only: upright-gated speed, a gait clock for foot
    contacts, alternating steps, anti-slip, posture and effort terms. No demonstrations or reference motions.
  - 3D stadium renderer with camera presets, mouse orbit and zoom; eight humanoids race side by side.
  - Pretrained walking and running policies in `models/walker/`, so the viewer works right after cloning.
  - Versioned checkpoints: incompatible or legacy models are detected with a clear message instead of a crash.
- Launcher (`python main.py`) to pick a simulation; both simulations also run as packages
  (`python -m racing`, `python -m walker`).
- Procedural race tracks with Easy / Medium / Hard / Random difficulty and an in-app track picker.
- GitHub Actions CI (ruff and pytest on Linux and Windows), MIT license, English and Turkish READMEs.

### Changed

- **AI Racing rewritten from scratch** around a geodesic progress field and a shared Dueling Double DQN.
  Version 1 never completed a lap in 3 092 recorded episodes; the rewrite completes its first lap around episode 5.
- Shared pygame theme and widgets moved to the `common/` package.

### Removed

- The 2D planar walker prototype, superseded by the 3D humanoid.

## [1.0.0]

- Original racing simulator: Q-learning and DQN agents driving on a PNG track with hand-placed checkpoints.

[2.0.0]: https://github.com/yahyio/ai-simulations/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/yahyio/ai-simulations/releases/tag/v1.0.0
