import math

import numpy as np
import pytest

from racing.config import deep_merge, validate
from racing.env import ACTIONS, NUM_ACTIONS, RaceEnv, Status
from tests.conftest import RING_CENTER, RING_INNER, RING_OUTER

MID_Y = RING_CENTER - (RING_INNER + RING_OUTER) / 2  # halkanın tepesinde orta hat
GAS_STRAIGHT = int(np.flatnonzero((ACTIONS[:, 0] == 0) & (ACTIONS[:, 1] == 1))[0])
BRAKE_STRAIGHT = int(np.flatnonzero((ACTIONS[:, 0] == 0) & (ACTIONS[:, 1] == -1))[0])


def make_env(track, config, num_cars=1, **overrides) -> RaceEnv:
    cfg = validate(deep_merge(config, overrides))
    return RaceEnv(track, cfg, num_cars, np.random.default_rng(0))


def place(env: RaceEnv, x: float, y: float, heading: float, speed: float) -> None:
    env.x[:] = x
    env.y[:] = y
    env.heading[:] = heading
    env.speed[:] = speed
    env.progress[:] = env.track.progress_at(env.x, env.y)


def test_action_table_covers_all_combinations():
    assert NUM_ACTIONS == 9
    assert {tuple(a) for a in ACTIONS} == {(s, t) for s in (-1, 0, 1) for t in (-1, 0, 1)}


def test_reset_places_cars_ahead_of_start_line(ring_track, config):
    env = make_env(ring_track, config, num_cars=4)
    obs = env.reset()
    assert obs.shape == (4, env.obs_dim)
    assert np.all(env.x > ring_track.start_center[0])
    assert np.all(env.running)
    assert np.all((obs[:, : len(env.sensor_angles)] >= 0) & (obs[:, : len(env.sensor_angles)] <= 1))


def test_accelerating_forward_earns_progress_reward(ring_track, config):
    env = make_env(ring_track, config)
    rewards = [env.step(np.array([GAS_STRAIGHT])).rewards[0] for _ in range(8)]
    assert env.distance[0] > 5
    assert sum(rewards) > 0
    assert env.status[0] == Status.RUNNING


def test_driving_off_track_crashes(ring_track, config):
    env = make_env(ring_track, config)
    place(env, RING_CENTER + 5, MID_Y, -math.pi / 2, 4.0)  # dışa doğru (yukarı)
    for _ in range(20):
        result = env.step(np.array([GAS_STRAIGHT]))
        if result.terminated[0]:
            break
    assert env.status[0] == Status.CRASHED
    assert result.rewards[0] < 0
    assert not result.truncated[0]


def test_crossing_line_after_full_loop_counts_as_lap(ring_track, config):
    env = make_env(ring_track, config)
    place(env, RING_CENTER - 6, MID_Y, 0.0, 4.0)  # çizginin hemen arkası, ileri bakıyor
    assert env.progress[0] > 0.75 * ring_track.length
    laps = False
    for _ in range(5):
        result = env.step(np.array([GAS_STRAIGHT]))
        laps |= bool(result.laps[0])
        if result.terminated[0]:
            break
    assert laps and env.status[0] == Status.LAP
    assert env.lap_times[0] == pytest.approx(env.steps / env.sim_fps)
    assert env.progress_fraction()[0] == 1.0


def test_crossing_line_backwards_is_wrong_way(ring_track, config):
    env = make_env(ring_track, config)
    place(env, RING_CENTER + 6, MID_Y, math.pi, 4.0)  # çizginin önü, geri bakıyor
    for _ in range(5):
        result = env.step(np.array([GAS_STRAIGHT]))
        if result.terminated[0]:
            break
    assert env.status[0] == Status.WRONG_WAY
    assert not result.laps[0]


def test_standing_still_triggers_idle(ring_track, config):
    env = make_env(ring_track, config, simulation={"idle_timeout_steps": 10})
    for _ in range(10):
        result = env.step(np.array([BRAKE_STRAIGHT]))
    assert env.status[0] == Status.IDLE
    assert result.terminated[0]


def test_step_limit_truncates_instead_of_terminating(ring_track, config):
    env = make_env(ring_track, config, simulation={"idle_timeout_steps": 10000, "max_steps_per_episode": 100})
    for _ in range(100):
        result = env.step(np.array([BRAKE_STRAIGHT]))
    assert env.status[0] == Status.TIMEOUT
    assert result.truncated[0] and not result.terminated[0]
    assert env.episode_over


def test_finished_cars_are_frozen(ring_track, config):
    env = make_env(ring_track, config, simulation={"idle_timeout_steps": 10})
    for _ in range(10):
        env.step(np.array([BRAKE_STRAIGHT]))
    before = (env.x.copy(), env.steps)
    result = env.step(np.array([GAS_STRAIGHT]))
    assert not result.active.any()
    np.testing.assert_array_equal(env.x, before[0])
    assert env.steps == before[1]


def test_observation_heading_error_is_zero_when_aligned(ring_track, config):
    env = make_env(ring_track, config)
    dx = 20.0
    tangent = math.atan2(dx, RING_CENTER - MID_Y)  # halkanın o noktadaki teğet açısı
    place(env, RING_CENTER + dx, MID_Y, tangent, 0.0)
    obs = env.observe()[0]
    n = len(env.sensor_angles)
    assert obs[n + 1] == pytest.approx(0.0, abs=0.1)   # sin(hata)
    assert obs[n + 2] == pytest.approx(1.0, abs=0.02)  # cos(hata)


def test_leader_prefers_running_car(ring_track, config):
    env = make_env(ring_track, config, num_cars=2)
    env.distance[:] = [100.0, 50.0]
    env.status[0] = Status.CRASHED
    assert env.leader() == 1
