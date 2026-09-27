import numpy as np
import pytest
import torch

from racing.agent import DQNAgent, NStepAccumulator, QNetwork, ReplayBuffer
from racing.env import StepResult

OBS_DIM, N_ACTIONS = 5, 9


def make_agent(config, num_envs=2, **learning) -> DQNAgent:
    cfg = {**config["learning"], **learning}
    return DQNAgent(OBS_DIM, N_ACTIONS, cfg, num_envs, seed=0)


def step_result(obs, rewards, terminated=None, truncated=None, active=None) -> StepResult:
    n = len(rewards)
    return StepResult(
        obs=np.asarray(obs, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=float),
        terminated=np.zeros(n, bool) if terminated is None else np.asarray(terminated),
        truncated=np.zeros(n, bool) if truncated is None else np.asarray(truncated),
        active=np.ones(n, bool) if active is None else np.asarray(active),
        laps=np.zeros(n, bool),
    )


def test_nstep_returns_discounted_sum_and_bootstrap():
    acc = NStepAccumulator(num_envs=1, n=3, gamma=0.5)
    assert acc.push(0, "s0", 0, 1.0, "s1", False, False) == []
    assert acc.push(0, "s1", 1, 2.0, "s2", False, False) == []
    (obs, action, ret, next_obs, done, discount), = acc.push(0, "s2", 2, 4.0, "s3", False, False)
    assert (obs, action, next_obs, done) == ("s0", 0, "s3", 0.0)
    assert ret == pytest.approx(1 + 0.5 * 2 + 0.25 * 4)
    assert discount == pytest.approx(0.125)


def test_nstep_terminal_flushes_every_pending_transition():
    acc = NStepAccumulator(num_envs=1, n=5, gamma=0.9)
    acc.push(0, "s0", 0, 1.0, "s1", False, False)
    out = acc.push(0, "s1", 1, 10.0, "s2", True, False)
    assert [t[0] for t in out] == ["s0", "s1"]
    assert all(t[4] == 1.0 for t in out)
    assert out[1][2] == pytest.approx(10.0)
    assert len(acc.queues[0]) == 0


def test_nstep_truncation_keeps_bootstrapping():
    acc = NStepAccumulator(num_envs=1, n=5, gamma=0.9)
    out = acc.push(0, "s0", 0, 1.0, "s1", False, True)
    assert out[0][4] == 0.0
    assert out[0][5] == pytest.approx(0.9)


def test_replay_buffer_wraps_around():
    buf = ReplayBuffer(4, OBS_DIM, torch.device("cpu"), torch.Generator().manual_seed(0))
    for k in range(3):
        buf.add_batch(np.full((2, OBS_DIM), k), [k, k], [k, k], np.zeros((2, OBS_DIM)), [0, 0], [1, 1])
    assert len(buf) == 4
    assert sorted(buf.actions.tolist()) == [1, 1, 2, 2]
    batch = buf.sample(8)
    assert batch["obs"].shape == (8, OBS_DIM)


def test_dueling_network_output_shape():
    net = QNetwork(OBS_DIM, N_ACTIONS, 16)
    assert net(torch.zeros(3, OBS_DIM)).shape == (3, N_ACTIONS)


def test_act_is_greedy_without_exploration(config):
    agent = make_agent(config)
    obs = np.random.default_rng(0).normal(size=(6, OBS_DIM)).astype(np.float32)
    first = agent.act(obs, 0.0)
    assert np.array_equal(first, agent.act(obs, 0.0))
    assert first.dtype.kind == "i" and first.min() >= 0 and first.max() < N_ACTIONS


def test_epsilon_decays_linearly(config):
    agent = make_agent(config, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay_steps=100)
    assert agent.epsilon == pytest.approx(1.0)
    agent.env_steps = 50
    assert agent.epsilon == pytest.approx(0.55)
    agent.env_steps = 1000
    assert agent.epsilon == pytest.approx(0.1)


def test_observe_only_stores_active_cars(config):
    agent = make_agent(config, n_step=1)
    obs = np.zeros((2, OBS_DIM), np.float32)
    agent.observe(obs, np.array([1, 2]), step_result(obs, [1.0, 1.0], active=[True, False]))
    assert len(agent.buffer) == 1
    assert agent.env_steps == 1


def test_training_learns_simple_bandit(config):
    """Eylem 3 her zaman +1, diğerleri 0 ödül verir → ajan 3'ü seçmeyi öğrenmeli."""
    agent = make_agent(config, n_step=1, learning_starts=16, batch_size=32, train_every=1, learning_rate=0.01)
    rng = np.random.default_rng(0)
    for _ in range(300):
        obs = rng.normal(size=(2, OBS_DIM)).astype(np.float32)
        actions = rng.integers(0, N_ACTIONS, size=2)
        rewards = (actions == 3).astype(float)
        agent.observe(obs, actions, step_result(obs, rewards, terminated=[True, True]))
        agent.train()
    test_obs = rng.normal(size=(20, OBS_DIM)).astype(np.float32)
    assert (agent.act(test_obs, 0.0) == 3).mean() > 0.9
    assert agent.updates > 0


def test_save_and_load_roundtrip(config, tmp_path):
    agent = make_agent(config)
    agent.env_steps = 1234
    path = tmp_path / "m.pt"
    agent.save(path)
    other = make_agent(config)
    other.load(path)
    assert other.env_steps == 1234
    for a, b in zip(agent.policy.parameters(), other.policy.parameters()):
        assert torch.equal(a, b)


def test_load_rejects_corrupted_file(config, tmp_path):
    path = tmp_path / "m.pt"
    path.write_bytes(b"bu bir model degil")
    with pytest.raises(ValueError, match="okunamadı"):
        make_agent(config).load(path)


def test_load_rejects_incompatible_model(config, tmp_path):
    path = tmp_path / "m.pt"
    make_agent(config).save(path)
    bigger = DQNAgent(OBS_DIM + 2, N_ACTIONS, config["learning"], 1)
    with pytest.raises(ValueError, match="uyumsuz"):
        bigger.load(path)
