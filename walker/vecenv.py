"""Paralel ortamlar: karakterleri işlemci çekirdeklerine dağıtır.

`LocalVecEnv` her şeyi aynı süreçte çalıştırır (testler, izleme modu). `SubprocVecEnv`
ortamları ayrı süreçlere böler; ana süreç sinir ağını çalıştırırken çekirdekler fiziği
hesaplar. İkisinin arayüzü aynıdır.
"""
from __future__ import annotations

import contextlib
import multiprocessing as mp
import os
from dataclasses import dataclass, replace
from multiprocessing.connection import Connection

import numpy as np

from .env import MAX_EPISODE_STEPS, TASKS, Episode, WalkerEnv

# Fizik süreçleri yalnızca küçük numpy işlemleri yapar. Çok iş parçacıklı BLAS orada hız kazandırmaz,
# ama her süreçte iş parçacığı başına bellek ayırır: süreç başına ~520 MB yerine ~150 MB (ölçüldü).
# 12 süreçle Windows'un bellek sınırı dolup eğitim "not enough memory" ile durabiliyordu.
SINGLE_THREAD_ENV = {name: "1" for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}


@dataclass(frozen=True)
class StepBatch:
    obs: np.ndarray            # (N, obs_dim)
    rewards: np.ndarray        # (N,)
    terminated: np.ndarray     # (N,) düştü
    truncated: np.ndarray      # (N,) süre doldu
    final_obs: np.ndarray      # (N, obs_dim) bitenlerin son gözlemi
    finished: tuple[Episode, ...]
    qpos: np.ndarray           # (N, nq) çizim için
    steps: np.ndarray          # (N,) mevcut denemenin adım sayısı


@contextlib.contextmanager
def _child_environment(values: dict[str, str]):
    """Yeni başlatılan süreçlerin ortam değişkenlerini geçici olarak ayarlar; ana süreç etkilenmez."""
    saved = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def default_workers() -> int:
    """Fiziksel çekirdek sayısına yakın (mantıksal çekirdeklerin yarısı)."""
    return max(1, (os.cpu_count() or 2) // 2)


class LocalVecEnv:
    def __init__(self, num_envs: int, task_key: str, seed: int | None = None, max_steps: int = MAX_EPISODE_STEPS):
        self.env = WalkerEnv(num_envs, TASKS[task_key], seed=seed, max_steps=max_steps)
        self.num_envs = num_envs
        self.obs_dim, self.act_dim = self.env.obs_dim, self.env.act_dim
        self.qpos = self.env.snapshot()
        self.steps = np.zeros(num_envs, dtype=np.int64)

    def reset(self) -> np.ndarray:
        obs = self.env.reset()
        self.qpos, self.steps = self.env.snapshot(), self.env.steps.copy()
        return obs

    def step(self, actions: np.ndarray) -> StepBatch:
        obs, rew, term, trunc, final, finished = self.env.step(actions)
        self.qpos, self.steps = self.env.snapshot(), self.env.steps.copy()
        return StepBatch(obs, rew, term, trunc, final, tuple(finished), self.qpos, self.steps)

    def close(self) -> None:
        pass


def _worker(conn: Connection, num_envs: int, task_key: str, seed: int | None, max_steps: int) -> None:
    env = WalkerEnv(num_envs, TASKS[task_key], seed=seed, max_steps=max_steps)
    try:
        while True:
            command, payload = conn.recv()
            if command == "step":
                result = env.step(payload)
                conn.send((*result, env.snapshot(), env.steps.copy()))
            elif command == "reset":
                conn.send((env.reset(), env.snapshot(), env.steps.copy()))
            elif command == "close":
                break
    except (EOFError, KeyboardInterrupt, BrokenPipeError):
        pass
    finally:
        conn.close()


class SubprocVecEnv:
    def __init__(self, num_envs: int, task_key: str, seed: int | None = None, workers: int | None = None,
                 max_steps: int = MAX_EPISODE_STEPS):
        workers = max(1, min(default_workers() if workers is None else workers, num_envs))
        sizes = [num_envs // workers + (1 if k < num_envs % workers else 0) for k in range(workers)]
        self.offsets = np.cumsum([0] + sizes[:-1])
        self.num_envs = num_envs
        probe = WalkerEnv(1, TASKS[task_key])
        self.obs_dim, self.act_dim = probe.obs_dim, probe.act_dim

        ctx = mp.get_context("spawn")   # Windows ile aynı davranış her yerde
        self.conns: list[Connection] = []
        self.procs = []
        with _child_environment(SINGLE_THREAD_ENV):
            for k, size in enumerate(sizes):
                parent, child = ctx.Pipe()
                worker_seed = None if seed is None else seed + 1000 * k
                proc = ctx.Process(target=_worker, args=(child, size, task_key, worker_seed, max_steps), daemon=True)
                proc.start()
                child.close()
                self.conns.append(parent)
                self.procs.append(proc)
        self.sizes = sizes
        self.qpos = np.zeros((num_envs, probe.model.nq))
        self.steps = np.zeros(num_envs, dtype=np.int64)

    def reset(self) -> np.ndarray:
        for conn in self.conns:
            conn.send(("reset", None))
        parts = [conn.recv() for conn in self.conns]
        self.qpos = np.concatenate([p[1] for p in parts])
        self.steps = np.concatenate([p[2] for p in parts])
        return np.concatenate([p[0] for p in parts])

    def step_async(self, actions: np.ndarray) -> None:
        for conn, start, size in zip(self.conns, self.offsets, self.sizes):
            conn.send(("step", actions[start:start + size]))

    def step_wait(self) -> StepBatch:
        parts = [conn.recv() for conn in self.conns]
        finished = tuple(
            replace(ep, env=int(start) + ep.env)
            for part, start in zip(parts, self.offsets) for ep in part[5]
        )
        self.qpos = np.concatenate([p[6] for p in parts])
        self.steps = np.concatenate([p[7] for p in parts])
        return StepBatch(
            obs=np.concatenate([p[0] for p in parts]),
            rewards=np.concatenate([p[1] for p in parts]),
            terminated=np.concatenate([p[2] for p in parts]),
            truncated=np.concatenate([p[3] for p in parts]),
            final_obs=np.concatenate([p[4] for p in parts]),
            finished=finished,
            qpos=self.qpos,
            steps=self.steps,
        )

    def step(self, actions: np.ndarray) -> StepBatch:
        self.step_async(actions)
        return self.step_wait()

    def close(self) -> None:
        for conn in self.conns:
            with contextlib.suppress(BrokenPipeError, OSError):
                conn.send(("close", None))
        for proc in self.procs:
            proc.join(timeout=2)
            if proc.is_alive():
                proc.terminate()
        for conn in self.conns:
            conn.close()


def make_vec_env(num_envs: int, task_key: str, seed: int | None = None, workers: int | None = None,
                 max_steps: int = MAX_EPISODE_STEPS) -> LocalVecEnv | SubprocVecEnv:
    """workers=0 → aynı süreçte; aksi halde çekirdeklere dağıtılır."""
    if workers == 0 or num_envs == 1:
        return LocalVecEnv(num_envs, task_key, seed=seed, max_steps=max_steps)
    return SubprocVecEnv(num_envs, task_key, seed=seed, workers=workers, max_steps=max_steps)
