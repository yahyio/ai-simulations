import mujoco
import numpy as np
import pytest

from walker.env import MIN_PELVIS_HEIGHT, TASKS, Episode, WalkerEnv, forward_reward
from walker.model import MOTORS, PELVIS_HEIGHT, load_model
from walker.ppo import PPO, PPOConfig, RunningMeanStd
from walker.render import BodyGeometry, part_color
from walker.trainer import Mode, WalkStats, WalkTrainer, has_model, reset_task
from yurume import parse_args

SMALL = PPOConfig(num_envs=4, rollout_steps=16, epochs=2, minibatches=2, hidden=32)


@pytest.fixture(autouse=True)
def in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def test_model_has_eleven_motors_and_stands_on_ground():
    model = load_model()
    assert model.nu == len(MOTORS) == 11
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    feet = [d.geom_xpos[model.geom(n).id][2] - model.geom_size[model.geom(n).id][0] for n in ("foot_l", "foot_r")]
    assert all(0.0 <= z < 0.03 for z in feet)
    assert 55 < model.body_mass.sum() < 65


def test_env_shapes_and_reset():
    env = WalkerEnv(3, TASKS["yurume"], seed=0)
    obs = env.reset()
    assert obs.shape == (3, env.obs_dim) and env.act_dim == 11
    assert np.allclose(env.positions(), 0.0)


def test_limp_character_falls_and_is_reset():
    env = WalkerEnv(1, TASKS["kosu"], seed=0)
    env.reset()
    for _ in range(200):
        _, _, terminated, truncated, final_obs, finished = env.step(np.zeros((1, env.act_dim)))
        if finished:
            break
    assert terminated[0] and not truncated[0] and finished[0].fell
    assert final_obs[0][0] + PELVIS_HEIGHT < MIN_PELVIS_HEIGHT or abs(final_obs[0][1]) > 0.5
    assert env.steps[0] == 0  # yeni deneme başladı


def test_time_limit_truncates():
    env = WalkerEnv(1, TASKS["kosu"], seed=0, max_steps=5)
    env.reset()
    for _ in range(5):
        _, _, terminated, truncated, _, finished = env.step(np.zeros((1, env.act_dim)))
    assert truncated[0] and not terminated[0] and not finished[0].fell


def test_forward_reward_shapes():
    walk, run = TASKS["yurume"], TASKS["kosu"]
    assert forward_reward(run, 4.0) > forward_reward(run, 2.0)
    assert forward_reward(walk, 1.5) > forward_reward(walk, 1.0)
    assert forward_reward(walk, 1.5) > forward_reward(walk, 3.0)  # hedeften hızlısı cezalı


def test_step_reward_prefers_alternating_feet():
    env = WalkerEnv(1, TASKS["yurume"], seed=0)
    env.reset()
    left, right, air = np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.0, 0.0])
    env.air_time[0] = [0.5, 0.0]
    first = env._step_reward(0, left)             # sol ayak 0.5 s havadan indi: ödül
    env.air_time[0] = [0.5, 0.0]
    same_again = env._step_reward(0, left)        # yine sol: ceza
    env.air_time[0] = [0.0, 0.5]
    alternate = env._step_reward(0, right)        # sağ: ödül
    assert first > 0 and alternate > 0 and same_again < 0
    assert env._step_reward(0, air) == 0.0


def test_upright_is_one_when_standing():
    env = WalkerEnv(1, TASKS["yurume"], seed=0)
    env.reset()
    assert env.upright(env.datas[0]) == pytest.approx(1.0, abs=0.01)


def test_running_mean_std_matches_numpy():
    data = np.random.default_rng(0).normal(3.0, 2.0, size=(1000, 4))
    rms = RunningMeanStd((4,))
    for chunk in np.split(data, 10):
        rms.update(chunk)
    np.testing.assert_allclose(rms.mean, data.mean(axis=0), atol=1e-3)
    np.testing.assert_allclose(rms.var, data.var(axis=0), rtol=1e-3)


def test_ppo_collects_updates_and_roundtrips(tmp_path):
    env = WalkerEnv(SMALL.num_envs, TASKS["kosu"], seed=0)
    ppo = PPO(env.obs_dim, env.act_dim, SMALL, seed=0)
    obs = env.reset()
    full = False
    while not full:
        a, norm, logp, v = ppo.act(obs)
        obs, r, term, trunc, final, _ = env.step(a)
        full = ppo.record(norm, a, logp, v, r, term, trunc, final)
    before = [p.detach().clone() for p in ppo.net.parameters()]
    steps = sum(1 for _ in ppo.update(obs))
    assert steps >= SMALL.minibatches and ppo.updates == 1 and ppo.t == 0
    assert any(not np.allclose(a, b.detach()) for a, b in zip(before, ppo.net.parameters()))

    path = tmp_path / "m.pt"
    ppo.save(path)
    other = PPO(env.obs_dim, env.act_dim, SMALL)
    other.load(path)
    np.testing.assert_allclose(other.obs_rms.mean, ppo.obs_rms.mean)
    assert other.total_steps == ppo.total_steps


def test_ppo_config_rejects_too_many_minibatches():
    with pytest.raises(ValueError, match="minibatches"):
        PPOConfig(num_envs=1, rollout_steps=2, minibatches=4)


def test_ppo_load_rejects_corrupt_and_incompatible(tmp_path):
    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"cop")
    with pytest.raises(ValueError, match="okunamadı"):
        PPO(29, 11, SMALL).load(bad)
    good = tmp_path / "good.pt"
    PPO(29, 11, SMALL).save(good)
    with pytest.raises(ValueError, match="uyumsuz"):
        PPO(30, 11, SMALL).load(good)


def test_stats_record_best_distance_and_history():
    stats = WalkStats(task="kosu")
    assert stats.record(Episode(0, 10.0, 5.0, 400, True))
    assert not stats.record(Episode(1, 10.0, 3.0, 400, True))
    stats.close_update([Episode(0, 1, 5.0, 400, True), Episode(1, 1, 3.0, 200, True)])
    assert stats.best_distance == 5.0 and stats.history_best_distance == [5.0]
    assert stats.history_mean_distance == [4.0]


def test_trainer_runs_through_update_and_saves():
    trainer = WalkTrainer("kosu", Mode.TRAIN, resume=False, cfg=SMALL, seed=0)
    saw_update = False
    for _ in range(SMALL.rollout_steps + 20):
        saw_update |= trainer.tick().updating
    assert saw_update
    trainer.save()
    assert has_model("kosu")
    resumed = WalkTrainer("kosu", Mode.TRAIN, resume=True, cfg=SMALL)
    assert resumed.ppo.total_steps == trainer.ppo.total_steps
    assert {p.name for p in reset_task("kosu")} >= {"model.pt", "stats.json"}
    assert not has_model("kosu")


def test_watch_needs_model():
    with pytest.raises(FileNotFoundError):
        WalkTrainer("yurume", Mode.WATCH, cfg=SMALL)


def test_unknown_task_raises():
    with pytest.raises(ValueError):
        WalkTrainer("ucma", cfg=SMALL)


def test_body_geometry_segments_have_limb_lengths():
    model = load_model()
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    geo = BodyGeometry(model)
    p1, p2 = geo.segments(d)
    thigh = geo.names.index("thigh_r")
    assert np.linalg.norm(p2[thigh] - p1[thigh]) == pytest.approx(0.44, abs=1e-6)
    assert np.allclose(p1[geo.head], p2[geo.head])
    assert geo.eye(d)[0] > p1[geo.head][0]  # göz öne bakıyor


def test_back_limbs_are_darker():
    shirt = (100, 150, 200)
    assert sum(part_color("thigh_l", shirt)) < sum(part_color("thigh_r", shirt))
    assert part_color("chest", shirt) == shirt


def test_cli_args():
    args = parse_args(["--headless", "--task", "kosu", "--steps", "1000"])
    assert args.task == "kosu" and args.steps == 1000
    with pytest.raises(SystemExit):
        parse_args(["--task", "ucma"])
