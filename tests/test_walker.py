import math
import os

import mujoco
import numpy as np
import pytest
import torch

from walker.__main__ import parse_args
from walker.env import (
    CONTACT_WEIGHT,
    CONTROL_DT,
    FALL_PENALTY,
    OBS_DIM,
    SLIP_WEIGHT,
    TASKS,
    Episode,
    WalkerEnv,
    forward_reward,
)
from walker.model import (
    DEFAULT_POSE,
    FREE_QPOS,
    JOINTS,
    LANE_WIDTH,
    NUM_JOINTS,
    joint_addresses,
    load_display_model,
    load_model,
    standing_qpos,
    yaw_of,
)
from walker.ppo import CHECKPOINT_VERSION, PPO, PPOConfig, RunningMeanStd, read_checkpoint_header
from walker.trainer import (
    IncompatibleModelError,
    Mode,
    WalkStats,
    WalkTrainer,
    checkpoint_compatible,
    has_legacy_model,
    has_model,
    paths,
    reset_task,
    watch_source,
)
from walker.vecenv import SINGLE_THREAD_ENV, LocalVecEnv, SubprocVecEnv, _child_environment

SMALL = PPOConfig(num_envs=4, rollout_steps=16, epochs=2, minibatches=2, hidden=(32, 32))
ZERO = np.zeros((1, NUM_JOINTS))


@pytest.fixture(autouse=True)
def in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def test_model_has_21_pd_joints_and_human_proportions():
    model = load_model()
    assert model.nu == NUM_JOINTS == 21
    assert model.nq == FREE_QPOS + NUM_JOINTS
    assert 55 < model.body_mass.sum() < 80
    d = mujoco.MjData(model)
    d.qpos[:] = standing_qpos()
    mujoco.mj_forward(model, d)
    soles = [d.geom_xpos[model.geom(f).id][2] - 0.025 for f in ("foot_l", "foot_r")]
    head_top = d.geom_xpos[model.geom("head").id][2] + 0.1
    assert all(0.0 <= z < 0.02 for z in soles)
    assert 1.7 < head_top < 1.8


def test_standing_pose_is_symmetric_despite_tree_order():
    """Gerileme: qpos ağaç sırasını (bel → kollar → bacaklar) izler, eylem sırası farklıdır."""
    model = load_model()
    qadr, _ = joint_addresses(model)
    assert list(qadr) != list(range(FREE_QPOS, FREE_QPOS + NUM_JOINTS))   # sıralar gerçekten farklı
    d = mujoco.MjData(model)
    d.qpos[:] = standing_qpos()
    mujoco.mj_forward(model, d)
    feet = d.geom_xpos[[model.geom("foot_r").id, model.geom("foot_l").id]]
    assert feet[0, 1] == pytest.approx(-feet[1, 1], abs=1e-6)   # sağ-sol ayna simetrisi
    assert feet[0, 2] == pytest.approx(feet[1, 2], abs=1e-6)
    names = [j.name for j in JOINTS]
    assert d.qpos[qadr[names.index("knee_r")]] == pytest.approx(DEFAULT_POSE[names.index("knee_r")])


def test_display_model_places_runners_in_lanes():
    for lanes in (1, 8):
        model = load_display_model(lanes)
        assert model.nq == lanes * (FREE_QPOS + NUM_JOINTS)
    model = load_display_model(3)
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    ys = [d.xpos[model.body(f"a{k}_pelvis").id][1] for k in range(3)]
    assert ys == pytest.approx([0.0, LANE_WIDTH, 2 * LANE_WIDTH])


def test_yaw_of_quaternion():
    half = math.radians(30) / 2
    assert yaw_of(np.array([math.cos(half), 0, 0, math.sin(half)])) == pytest.approx(math.radians(30))


# --------------------------------------------------------------------------- #
# Ortam ve ödül
# --------------------------------------------------------------------------- #
def test_env_shapes_and_reset():
    env = WalkerEnv(3, TASKS["yurume"], seed=0)
    obs = env.reset()
    assert obs.shape == (3, OBS_DIM) and env.act_dim == 21
    assert np.allclose(env.positions(), 0.0)
    assert env.upright(env.datas[0]) == pytest.approx(1.0, abs=0.01)


def test_zero_action_targets_default_pose():
    env = WalkerEnv(1, TASKS["yurume"], seed=0)
    np.testing.assert_allclose(env.targets(np.zeros(21)), DEFAULT_POSE)
    assert np.all(env.targets(np.full(21, 5.0)) <= env.hi)


def test_passive_character_falls_with_penalty_and_resets():
    env = WalkerEnv(1, TASKS["kosu"], seed=0)
    env.reset()
    for _ in range(300):
        _, rewards, terminated, truncated, _, finished = env.step(ZERO)
        if finished:
            break
    assert terminated[0] and not truncated[0] and finished[0].fell
    assert rewards[0] < -FALL_PENALTY / 2
    assert env.steps[0] == 0  # yeni deneme başladı


def test_time_limit_truncates():
    env = WalkerEnv(1, TASKS["kosu"], seed=0, max_steps=5)
    env.reset()
    for _ in range(5):
        _, _, terminated, truncated, _, finished = env.step(ZERO)
    assert truncated[0] and not terminated[0] and not finished[0].fell


def test_forward_reward_shapes():
    walk, run = TASKS["yurume"], TASKS["kosu"]
    assert forward_reward(run, 4.0) > forward_reward(run, 2.0)
    assert forward_reward(run, 20.0) == forward_reward(run, run.max_speed)   # üst sınır
    assert forward_reward(walk, 1.4) > forward_reward(walk, 1.0)
    assert forward_reward(walk, 1.4) > forward_reward(walk, 3.0)            # hedeften hızlısı cezalı


def _tilt(env: WalkerEnv, pitch_deg: float) -> mujoco.MjData:
    d = env.datas[0]
    half = math.radians(pitch_deg) / 2
    d.qpos[3:7] = (math.cos(half), 0.0, math.sin(half), 0.0)
    mujoco.mj_forward(env.model, d)
    return d


def test_speed_is_not_rewarded_while_diving_forward():
    env = WalkerEnv(1, TASKS["yurume"], seed=0)
    env.reset()
    d = _tilt(env, 60)
    no_contacts = np.zeros(2)
    fast = env._reward(0, d, np.zeros(21), 1.4, 0.0, no_contacts)
    still = env._reward(0, d, np.zeros(21), 0.0, 0.0, no_contacts)
    assert fast == pytest.approx(still)   # öne düşerken kazanılan hız değersiz

    upright = _tilt(env, 0)
    assert env._reward(0, upright, np.zeros(21), 1.4, 0.0, no_contacts) > \
        env._reward(0, upright, np.zeros(21), 0.0, 0.0, no_contacts)


def test_crossed_feet_are_penalized():
    env = WalkerEnv(1, TASKS["yurume"], seed=0)
    env.reset()
    d = env.datas[0]
    base = env._reward(0, d, np.zeros(21), 0.0, 0.0, np.zeros(2))
    names = [j.name for j in JOINTS]
    d.qpos[env.qadr[names.index("hip_x_r")]] = math.radians(15)   # sağ bacak içe
    d.qpos[env.qadr[names.index("hip_x_l")]] = math.radians(-15)  # sol bacak içe → çapraz
    mujoco.mj_forward(env.model, d)
    assert env._reward(0, d, np.zeros(21), 0.0, 0.0, np.zeros(2)) < base - 0.5


def test_step_reward_prefers_alternating_feet_and_is_capped():
    env = WalkerEnv(1, TASKS["yurume"], seed=0)
    env.reset()
    left, right = np.array([0.0, 1.0]), np.array([1.0, 0.0])   # FOOT_GEOMS = (sağ, sol)
    env.air_time[0] = [0.0, 0.5]
    first = env._step_reward(0, left)
    env.air_time[0] = [0.0, 0.5]
    same_again = env._step_reward(0, left)
    env.air_time[0] = [0.5, 0.0]
    alternate = env._step_reward(0, right)
    assert first > 0 and alternate > 0 and same_again < 0
    env.last_foot[0] = 1
    env.air_time[0] = [5.0, 0.0]                 # bacağı saniyelerce havada tutmak
    long_hold = env._step_reward(0, right)
    task = TASKS["yurume"]
    assert long_hold == pytest.approx(task.step_air_weight * (task.step_air_cap - task.step_air_target))


def test_landing_on_both_feet_at_once_is_a_hop_not_a_step():
    env = WalkerEnv(1, TASKS["yurume"], seed=0)
    env.reset()
    env.air_time[0] = [0.5, 0.5]
    hop = env._step_reward(0, np.array([1.0, 1.0]))
    task = TASKS["yurume"]
    assert hop == pytest.approx(-2 * task.step_air_weight * task.step_air_target)


@pytest.mark.parametrize("key", ["yurume", "kosu"])
def test_gait_clock_alternates_feet_with_task_specific_stance(key):
    env = WalkerEnv(1, TASKS[key], seed=0)
    env.reset()
    task = TASKS[key]
    pattern = []
    for step in range(round(task.gait_period / CONTROL_DT)):
        env.steps[0] = step
        pattern.append(env.stance(0))
    pattern = np.array(pattern)
    assert pattern.mean(axis=0) == pytest.approx([task.stance_fraction] * 2, abs=0.03)
    assert not np.array_equal(pattern[:, 0], pattern[:, 1])      # sağ ve sol yarım tur arayla
    flight = ~pattern.any(axis=1)
    assert flight.any() == (key == "kosu")                        # uçuş anı yalnızca koşuda


def test_observation_ends_with_the_gait_clock():
    env = WalkerEnv(1, TASKS["yurume"], seed=0)
    obs = env.reset()
    np.testing.assert_allclose(obs[0, -2:], [0.0, 1.0], atol=1e-6)   # sin, cos: tur başı
    steps = 10
    for _ in range(steps):
        obs, *_ = env.step(ZERO)
    angle = 2 * math.pi * steps * CONTROL_DT / TASKS["yurume"].gait_period
    np.testing.assert_allclose(obs[0, -2:], [math.sin(angle), math.cos(angle)], atol=1e-6)


def test_feet_matching_the_clock_earn_more_than_feet_against_it():
    env = WalkerEnv(1, TASKS["kosu"], seed=0)
    env.reset()
    d = env.datas[0]
    env.steps[0] = 0                                  # koşuda tur başı: sağ yerde, sol havada
    in_step = env._reward(0, d, np.zeros(21), 0.0, 0.0, np.array([1.0, 0.0]))
    env.air_time[0] = 0.0
    against = env._reward(0, d, np.zeros(21), 0.0, 0.0, np.array([0.0, 1.0]))
    assert in_step - against == pytest.approx(2 * CONTACT_WEIGHT, abs=0.05)


@pytest.mark.parametrize("key", ["yurume", "kosu"])
def test_flight_is_penalized_only_when_walking(key):
    env = WalkerEnv(1, TASKS[key], seed=0)
    env.reset()
    d = env.datas[0]
    env.steps[0] = 0                                  # tur başı: sağ ayak yerde olmalı
    grounded = env._reward(0, d, np.zeros(21), 0.0, 0.0, np.array([1.0, 0.0]))
    env.air_time[0] = 0.0
    airborne = env._reward(0, d, np.zeros(21), 0.0, 0.0, np.array([0.0, 0.0]))
    expected = CONTACT_WEIGHT + TASKS[key].flight_penalty   # sağ ayak ritimden çıktı (+ uçuş cezası)
    assert grounded - airborne == pytest.approx(expected, abs=0.05)


def test_sliding_a_planted_foot_is_penalized():
    env = WalkerEnv(1, TASKS["yurume"], seed=0)
    env.reset()
    d, both = env.datas[0], np.array([1.0, 1.0])
    still = env._reward(0, d, np.zeros(21), 0.0, 0.0, both)
    env.air_time[0] = 0.0
    env.prev_feet[0, :, 0] -= 0.02                    # iki ayak da 20 ms'de 2 cm kaydı: 1 m/s
    sliding = env._reward(0, d, np.zeros(21), 0.0, 0.0, both)
    assert still - sliding == pytest.approx(2 * SLIP_WEIGHT * 1.0, rel=1e-6)


# --------------------------------------------------------------------------- #
# Paralel ortamlar
# --------------------------------------------------------------------------- #
def test_subprocess_env_matches_local_interface():
    local = LocalVecEnv(4, "kosu", seed=0)
    remote = SubprocVecEnv(4, "kosu", seed=0, workers=2)
    try:
        assert remote.obs_dim == local.obs_dim == OBS_DIM
        assert remote.reset().shape == local.reset().shape == (4, OBS_DIM)
        finished = []
        for _ in range(150):
            batch = remote.step(np.zeros((4, 21)))
            finished += batch.finished
        assert batch.qpos.shape == (4, FREE_QPOS + NUM_JOINTS)
        assert finished and {ep.env for ep in finished} <= {0, 1, 2, 3}
        assert any(ep.env >= 2 for ep in finished)   # ikinci sürecin denemeleri doğru numaralanır
    finally:
        remote.close()
    assert not any(p.is_alive() for p in remote.procs)
    assert all(conn.closed for conn in remote.conns)


def test_child_environment_is_temporary(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
    with _child_environment(SINGLE_THREAD_ENV):
        assert os.environ["OMP_NUM_THREADS"] == os.environ["OPENBLAS_NUM_THREADS"] == "1"
    assert os.environ["OMP_NUM_THREADS"] == "8" and "OPENBLAS_NUM_THREADS" not in os.environ


# --------------------------------------------------------------------------- #
# PPO
# --------------------------------------------------------------------------- #
def test_running_mean_std_matches_numpy():
    data = np.random.default_rng(0).normal(3.0, 2.0, size=(1000, 4))
    rms = RunningMeanStd((4,))
    for chunk in np.split(data, 10):
        rms.update(chunk)
    np.testing.assert_allclose(rms.mean, data.mean(axis=0), atol=1e-3)
    np.testing.assert_allclose(rms.var, data.var(axis=0), rtol=1e-3)


def test_ppo_collects_updates_and_roundtrips(tmp_path):
    env = LocalVecEnv(SMALL.num_envs, "kosu", seed=0)
    ppo = PPO(env.obs_dim, env.act_dim, SMALL, seed=0)
    obs = env.reset()
    full = False
    while not full:
        a, norm, logp, v = ppo.act(obs)
        batch = env.step(a)
        full = ppo.record(norm, a, logp, v, batch.rewards, batch.terminated, batch.truncated, batch.final_obs)
        obs = batch.obs
    before = [p.detach().clone() for p in ppo.net.parameters()]
    steps = sum(1 for _ in ppo.update(obs))
    assert steps == SMALL.epochs * SMALL.minibatches and ppo.updates == 1 and ppo.t == 0
    assert any(not torch.equal(a, b) for a, b in zip(before, ppo.net.parameters()))
    assert ppo.stats["kl"] >= 0

    path = tmp_path / "m.pt"
    ppo.save(path)
    assert read_checkpoint_header(path) == (CHECKPOINT_VERSION, OBS_DIM, 21, [32, 32])
    other = PPO(env.obs_dim, env.act_dim, SMALL)
    other.load(path)
    np.testing.assert_allclose(other.obs_rms.mean, ppo.obs_rms.mean)
    assert other.total_steps == ppo.total_steps


def test_slim_checkpoint_is_smaller_and_loadable(tmp_path):
    ppo = PPO(OBS_DIM, 21, SMALL, seed=0)
    full, slim = tmp_path / "full.pt", tmp_path / "slim.pt"
    ppo.save(full)
    ppo.save(slim, include_optimizer=False)
    assert slim.stat().st_size < full.stat().st_size
    other = PPO(OBS_DIM, 21, SMALL)
    other.load(slim)
    for a, b in zip(ppo.net.parameters(), other.net.parameters()):
        assert torch.equal(a, b)


def test_ppo_load_rejects_corrupt_and_incompatible(tmp_path):
    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"cop")
    with pytest.raises(ValueError, match="okunamadı"):
        PPO(OBS_DIM, 21, SMALL).load(bad)
    assert read_checkpoint_header(bad) is None
    good = tmp_path / "good.pt"
    PPO(OBS_DIM, 21, SMALL).save(good)
    with pytest.raises(ValueError, match="uyumsuz"):
        PPO(OBS_DIM + 1, 21, SMALL).load(good)


def test_ppo_config_validation():
    with pytest.raises(ValueError, match="minibatches"):
        PPOConfig(num_envs=1, rollout_steps=2, minibatches=4)
    with pytest.raises(ValueError):
        PPOConfig(hidden=(0,))


# --------------------------------------------------------------------------- #
# Eğitim çekirdeği
# --------------------------------------------------------------------------- #
def test_stats_record_best_distance_and_history():
    stats = WalkStats(task="kosu")
    assert stats.record(Episode(0, 10.0, 5.0, 400, True))
    assert not stats.record(Episode(1, 10.0, 3.0, 400, True))
    stats.close_update([Episode(0, 1, 5.0, 400, True), Episode(1, 1, 3.0, 200, True)])
    assert stats.best_distance == 5.0 and stats.history_best_distance == [5.0]
    assert stats.history_mean_distance == [4.0]


def test_trainer_trains_saves_resumes_and_resets():
    trainer = WalkTrainer("kosu", Mode.TRAIN, resume=False, cfg=SMALL, seed=0, workers=0)
    try:
        saw_update = False
        for _ in range(SMALL.rollout_steps + 20):
            saw_update |= trainer.tick().updating
        assert saw_update and trainer.ppo.updates >= 1
        info = trainer.runner(trainer.leader())
        assert info.seconds >= 0 and len(info.contacts) == 2
        env = trainer.env.env
        for i in range(trainer.display_count):   # panel ayak temaslarını gözlemin doğru yerinden okur
            right, left = env.foot_contacts(env.datas[i]) > 0
            assert trainer.runner(i).contacts == (bool(left), bool(right))
        trainer.save()
    finally:
        trainer.close()
    assert has_model("kosu", SMALL) and not has_legacy_model("kosu", SMALL)
    resumed = WalkTrainer("kosu", Mode.TRAIN, resume=True, cfg=SMALL, workers=0)
    resumed.close()
    assert resumed.ppo.total_steps == trainer.ppo.total_steps
    assert {p.name for p in reset_task("kosu")} >= {"model.pt", "stats.json"}
    assert not has_model("kosu", SMALL)


def test_watch_needs_a_compatible_model_and_falls_back_to_bundled():
    with pytest.raises(FileNotFoundError):
        WalkTrainer("yurume", Mode.WATCH, cfg=SMALL)
    bundled = paths("yurume")["bundled"]
    bundled.parent.mkdir(parents=True)
    PPO(OBS_DIM, 21, SMALL).save(bundled)
    assert watch_source("yurume", SMALL) == bundled
    watcher = WalkTrainer("yurume", Mode.WATCH, cfg=SMALL)
    try:
        assert watcher.model_source == bundled and watcher.env.num_envs == 8
        tick = watcher.tick()
        assert not tick.updating and watcher.stats.episodes == 0
    finally:
        watcher.close()


def test_legacy_checkpoint_is_detected_and_blocks_resume():
    legacy = paths("yurume")["model"]
    legacy.parent.mkdir(parents=True)
    torch.save({"version": 1, "obs_dim": 29, "act_dim": 11, "hidden": 128}, legacy)   # eski 2B biçimi
    assert read_checkpoint_header(legacy) == (1, 29, 11, [128])
    assert has_legacy_model("yurume", SMALL) and not checkpoint_compatible(legacy, SMALL)
    assert watch_source("yurume", SMALL) is None
    with pytest.raises(IncompatibleModelError):
        WalkTrainer("yurume", Mode.TRAIN, resume=True, cfg=SMALL, workers=0)


def test_unknown_task_raises():
    with pytest.raises(ValueError):
        WalkTrainer("ucma", cfg=SMALL, workers=0)


# --------------------------------------------------------------------------- #
# 3B görüntü (GL bağlamı olmayan makinelerde, ör. CI, atlanır)
# --------------------------------------------------------------------------- #
@pytest.fixture
def stadium():
    import pygame

    from walker.render import Stadium

    pygame.init()
    try:
        scene = Stadium((160, 120), 2)
    except Exception as exc:  # OpenGL bağlamı yok
        pytest.skip(f"ekran dışı OpenGL kullanılamıyor: {exc}")
    yield scene
    scene.close()


def test_stadium_renders_and_projects_lookat_to_center(stadium):
    surface = stadium.render(record_m=5.0, marker_lane=0)
    assert surface.get_size() == (160, 120)
    (x, y), = stadium.project(np.array([stadium.camera.lookat]))
    assert abs(x - 80) < 2 and abs(y - 60) < 2
    stadium.set_poses(np.tile(standing_qpos(), (2, 1)))
    assert stadium.runner_position(1)[1] == pytest.approx(LANE_WIDTH)


def test_error_description_fits_a_dialog():
    from walker.app import describe_error

    text = describe_error(RuntimeError("OpenGL bağlamı oluşturulamadı " * 10))
    assert text.startswith("RuntimeError: OpenGL") and len(text) <= 72


def test_cli_args():
    args = parse_args(["--headless", "--task", "kosu", "--steps", "1000", "--workers", "2"])
    assert args.task == "kosu" and args.steps == 1000 and args.workers == 2
    with pytest.raises(SystemExit):
        parse_args(["--task", "ucma"])
