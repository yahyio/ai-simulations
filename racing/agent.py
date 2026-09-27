"""Dueling Double DQN ajanı: n-adımlı getiriler, deneyim tekrarı, yumuşak hedef güncelleme.

Tüm arabalar tek bir ortak sinir ağını paylaşır; her araba deneyim toplayan bir
"işçi" gibidir. Böylece N araba, tek arabaya göre N kat daha hızlı deneyim toplar.
"""
from __future__ import annotations

import copy
import os
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn

CHECKPOINT_VERSION = 2
GRAD_CLIP_NORM = 10.0
LOSS_EMA = 0.1
LOSS_SYNC_EVERY = 50


GPU_MIN_HIDDEN = 1024   # Daha küçük ağlarda CPU, GPU senkronizasyon maliyetinden hızlıdır
MAX_CPU_THREADS = 4     # Küçük matrislerde çok iş parçacığı yavaşlatır (ölçüldü)


def resolve_device(preference: str, hidden_size: int) -> torch.device:
    """'auto': büyük ağlarda (varsa) GPU, küçük ağlarda CPU."""
    wants_gpu = preference == "cuda" or (preference == "auto" and hidden_size >= GPU_MIN_HIDDEN)
    if wants_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    torch.set_num_threads(max(1, min(MAX_CPU_THREADS, (os.cpu_count() or 2) // 2)))
    return torch.device("cpu")


class QNetwork(nn.Module):
    """Dueling mimari: Q(s, a) = V(s) + A(s, a) - ortalama(A)."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.value = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))
        self.advantage = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, n_actions))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.body(x)
        advantage = self.advantage(h)
        return self.value(h) + advantage - advantage.mean(dim=1, keepdim=True)


class ReplayBuffer:
    """Sabit boyutlu halka tampon; veriler doğrudan hesaplama cihazında (GPU/CPU) tutulur."""

    def __init__(self, capacity: int, obs_dim: int, device: torch.device, generator: torch.Generator):
        self.capacity = int(capacity)
        self.device = device
        self.generator = generator
        self.obs = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=device)
        self.next_obs = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros(capacity, dtype=torch.int64, device=device)
        self.rewards = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.dones = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.discounts = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.pos = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add_batch(self, obs, actions, rewards, next_obs, dones, discounts) -> None:
        count = len(actions)
        if count == 0:
            return
        idx = torch.as_tensor((self.pos + np.arange(count)) % self.capacity, device=self.device)
        for store, values in ((self.obs, obs), (self.next_obs, next_obs), (self.actions, actions),
                              (self.rewards, rewards), (self.dones, dones), (self.discounts, discounts)):
            store[idx] = torch.as_tensor(np.asarray(values), dtype=store.dtype).to(self.device, non_blocking=True)
        self.pos = int((self.pos + count) % self.capacity)
        self.size = min(self.size + count, self.capacity)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        idx = torch.randint(0, self.size, (batch_size,), device=self.device, generator=self.generator)
        return {
            "obs": self.obs[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_obs": self.next_obs[idx],
            "dones": self.dones[idx],
            "discounts": self.discounts[idx],
        }


class NStepAccumulator:
    """Her araba için son n geçişi tutup n-adımlı getiri üretir."""

    def __init__(self, num_envs: int, n: int, gamma: float):
        self.n = n
        self.gamma = gamma
        self.queues: list[deque] = [deque() for _ in range(num_envs)]

    def clear(self) -> None:
        for q in self.queues:
            q.clear()

    def push(self, env: int, obs, action: int, reward: float, next_obs, terminated: bool, truncated: bool) -> list[tuple]:
        q = self.queues[env]
        q.append((obs, action, reward))
        out = []
        if terminated or truncated:
            while q:
                out.append(self._emit(q, next_obs, terminated))
                q.popleft()
        elif len(q) >= self.n:
            out.append(self._emit(q, next_obs, False))
            q.popleft()
        return out

    def _emit(self, q: deque, next_obs, done: bool) -> tuple:
        ret = sum(self.gamma**k * r for k, (_, _, r) in enumerate(q))
        obs0, action0, _ = q[0]
        return obs0, action0, ret, next_obs, float(done), self.gamma ** len(q)


class DQNAgent:
    """Paylaşılan ağ ile birden çok arabayı süren öğrenen ajan."""

    def __init__(self, obs_dim: int, n_actions: int, learning_cfg: dict, num_envs: int, seed: int | None = None):
        cfg = learning_cfg
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden_size = int(cfg["hidden_size"])
        self.device = resolve_device(cfg["device"], self.hidden_size)
        self.rng = np.random.default_rng(seed)
        self.torch_rng = torch.Generator(device=self.device)
        if seed is not None:
            torch.manual_seed(seed)
            self.torch_rng.manual_seed(seed)
        else:
            self.torch_rng.seed()

        self.gamma = float(cfg["discount_factor"])
        self.batch_size = int(cfg["batch_size"])
        self.tau = float(cfg["target_tau"])
        self.learning_starts = int(cfg["learning_starts"])
        self.gradient_steps = int(cfg["gradient_steps"])
        self.train_every = int(cfg["train_every"])
        self.eps_start = float(cfg["epsilon_start"])
        self.eps_end = float(cfg["epsilon_end"])
        self.eps_decay_steps = int(cfg["epsilon_decay_steps"])
        self.learning_rate = float(cfg["learning_rate"])

        self.policy = QNetwork(obs_dim, n_actions, self.hidden_size).to(self.device)
        self.target = copy.deepcopy(self.policy)
        self.target.requires_grad_(False)
        self._policy_params = list(self.policy.parameters())
        self._target_params = list(self.target.parameters())
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        self.buffer = ReplayBuffer(int(cfg["buffer_size"]), obs_dim, self.device, self.torch_rng)
        self.nstep = NStepAccumulator(num_envs, int(cfg["n_step"]), self.gamma)

        self.env_steps = 0
        self.updates = 0
        self._calls = 0
        self._loss_sum: torch.Tensor | None = None
        self._loss_count = 0
        self.loss = float("nan")

    # ------------------------------------------------------------------ #
    @property
    def epsilon(self) -> float:
        frac = min(1.0, self.env_steps / max(1, self.eps_decay_steps))
        return self.eps_start + (self.eps_end - self.eps_start) * frac

    def act(self, obs: np.ndarray, epsilons: np.ndarray | float = 0.0) -> np.ndarray:
        """Epsilon-greedy eylem seçimi (her araba için ayrı epsilon olabilir)."""
        with torch.no_grad():
            q = self.policy(torch.as_tensor(obs, dtype=torch.float32, device=self.device))
            actions = q.argmax(dim=1).cpu().numpy()
        explore = self.rng.random(len(actions)) < np.broadcast_to(epsilons, actions.shape)
        if explore.any():
            actions[explore] = self.rng.integers(0, self.n_actions, size=int(explore.sum()))
        return actions

    def observe(self, obs: np.ndarray, actions: np.ndarray, result) -> None:
        """Ortam adımının sonucunu n-adımlı tampon üzerinden deneyim havuzuna ekler."""
        transitions = []
        for i in np.flatnonzero(result.active):
            transitions.extend(self.nstep.push(
                i, obs[i], int(actions[i]), float(result.rewards[i]), result.obs[i],
                bool(result.terminated[i]), bool(result.truncated[i]),
            ))
        self.env_steps += int(result.active.sum())
        if transitions:
            o, a, r, no, d, g = zip(*transitions)
            self.buffer.add_batch(np.stack(o), np.array(a), np.array(r), np.stack(no), np.array(d), np.array(g))

    def train(self) -> bool:
        """Her `train_every` çağrıda bir, `gradient_steps` kez ağı günceller."""
        self._calls += 1
        if len(self.buffer) < max(self.batch_size, self.learning_starts) or self._calls % self.train_every:
            return False
        for _ in range(self.gradient_steps):
            self._update()
        if self._loss_count >= LOSS_SYNC_EVERY:
            self._sync_loss()
        return True

    def _sync_loss(self) -> None:
        """Kayıp değerini GPU'dan seyrek okur (her adımda senkronizasyon yavaştır)."""
        if self._loss_sum is None or self._loss_count == 0:
            return
        value = float(self._loss_sum.item()) / self._loss_count
        self.loss = value if np.isnan(self.loss) else (1 - LOSS_EMA) * self.loss + LOSS_EMA * value
        self._loss_sum, self._loss_count = None, 0

    def _update(self) -> None:
        batch = self.buffer.sample(self.batch_size)
        q = self.policy(batch["obs"]).gather(1, batch["actions"].unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_actions = self.policy(batch["next_obs"]).argmax(dim=1, keepdim=True)  # Double DQN
            next_q = self.target(batch["next_obs"]).gather(1, next_actions).squeeze(1)
            target = batch["rewards"] + (1.0 - batch["dones"]) * batch["discounts"] * next_q

        loss = nn.functional.smooth_l1_loss(q, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), GRAD_CLIP_NORM)
        self.optimizer.step()

        with torch.no_grad():
            torch._foreach_lerp_(self._target_params, self._policy_params, self.tau)
            detached = loss.detach()
            self._loss_sum = detached if self._loss_sum is None else self._loss_sum + detached
        self._loss_count += 1
        self.updates += 1

    def end_episode(self) -> None:
        """Bölüm yarıda kesildiğinde yarım kalan n-adım kuyruklarını atar."""
        self.nstep.clear()

    # ------------------------------------------------------------------ #
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save({
            "version": CHECKPOINT_VERSION,
            "obs_dim": self.obs_dim,
            "n_actions": self.n_actions,
            "hidden_size": self.hidden_size,
            "policy": self.policy.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "env_steps": self.env_steps,
            "updates": self.updates,
        }, tmp)
        tmp.replace(path)

    def load(self, path: Path) -> None:
        """Kayıtlı modeli yükler; uyumsuzsa ValueError fırlatır."""
        try:
            data = torch.load(path, map_location=self.device, weights_only=True)
        except Exception as exc:  # bozuk/yarım dosya: UnpicklingError, EOFError, BadZipFile…
            raise ValueError(f"Model dosyası okunamadı ({path}): {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"Model dosyası tanınmayan biçimde: {path}")
        expected = (CHECKPOINT_VERSION, self.obs_dim, self.n_actions, self.hidden_size)
        found = (data.get("version"), data.get("obs_dim"), data.get("n_actions"), data.get("hidden_size"))
        if found != expected:
            raise ValueError(
                f"Kayıtlı model mevcut ayarlarla uyumsuz (sürüm/gözlem/eylem/gizli katman: "
                f"{found} ≠ {expected}). Sensör sayısı veya ağ boyutu değişmiş olabilir."
            )
        self.policy.load_state_dict(data["policy"])
        self.target.load_state_dict(data["target"])
        self.optimizer.load_state_dict(data["optimizer"])
        for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate
        self.env_steps = int(data["env_steps"])
        self.updates = int(data["updates"])
