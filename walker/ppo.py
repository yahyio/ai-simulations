"""PPO (Proximal Policy Optimization) — sürekli eylemler (eklem torkları) için.

Güncelleme mini-parti adımlarına bölünmüş bir üreteçtir; arayüz her karede birkaç
adım işleyerek öğrenme sırasında donmadan akmaya devam eder.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch import nn

CHECKPOINT_VERSION = 1
OBS_CLIP = 10.0
REWARD_CLIP = 10.0
MAX_CPU_THREADS = 4


@dataclass(frozen=True)
class PPOConfig:
    num_envs: int = 32
    rollout_steps: int = 128        # ortam başına, her güncellemeden önce
    epochs: int = 10
    minibatches: int = 8
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.0
    max_grad_norm: float = 0.5
    hidden: int = 128
    init_log_std: float = -0.5
    target_kl: float = 0.03         # politika bundan fazla değişirse o turu erken bitir

    def __post_init__(self) -> None:
        if min(self.num_envs, self.rollout_steps, self.epochs, self.minibatches, self.hidden) < 1:
            raise ValueError("PPOConfig: sayısal alanlar en az 1 olmalı")
        if self.num_envs * self.rollout_steps < self.minibatches:
            raise ValueError("PPOConfig: num_envs × rollout_steps, minibatches sayısından küçük olamaz")


class RunningMeanStd:
    """Akan veri için ortalama/varyans (Chan'ın paralel algoritması)."""

    def __init__(self, shape: tuple[int, ...] = ()):
        self.mean = np.zeros(shape)
        self.var = np.ones(shape)
        self.count = 1e-4

    def update(self, batch: np.ndarray) -> None:
        b_mean, b_var, b_count = batch.mean(axis=0), batch.var(axis=0), batch.shape[0]
        delta = b_mean - self.mean
        total = self.count + b_count
        self.mean = self.mean + delta * b_count / total
        m2 = self.var * self.count + b_var * b_count + delta**2 * self.count * b_count / total
        self.var = m2 / total
        self.count = total

    def state(self) -> dict[str, torch.Tensor]:
        return {"mean": torch.as_tensor(self.mean), "var": torch.as_tensor(self.var),
                "count": torch.tensor(self.count, dtype=torch.float64)}

    def load(self, state: dict[str, torch.Tensor]) -> None:
        self.mean = state["mean"].numpy().copy()
        self.var = state["var"].numpy().copy()
        self.count = float(state["count"])


def _layer(inp: int, out: int, std: float = np.sqrt(2)) -> nn.Linear:
    layer = nn.Linear(inp, out)
    nn.init.orthogonal_(layer.weight, std)
    nn.init.zeros_(layer.bias)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int, init_log_std: float):
        super().__init__()
        self.actor = nn.Sequential(
            _layer(obs_dim, hidden), nn.Tanh(), _layer(hidden, hidden), nn.Tanh(), _layer(hidden, act_dim, 0.01),
        )
        self.critic = nn.Sequential(
            _layer(obs_dim, hidden), nn.Tanh(), _layer(hidden, hidden), nn.Tanh(), _layer(hidden, 1, 1.0),
        )
        self.log_std = nn.Parameter(torch.full((act_dim,), init_log_std))

    def distribution(self, obs: torch.Tensor) -> torch.distributions.Normal:
        return torch.distributions.Normal(self.actor(obs), self.log_std.exp().expand(obs.shape[0], -1))

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)


class PPO:
    def __init__(self, obs_dim: int, act_dim: int, cfg: PPOConfig = PPOConfig(), seed: int | None = None):
        torch.set_num_threads(max(1, min(MAX_CPU_THREADS, (os.cpu_count() or 2) // 2)))
        if seed is not None:
            torch.manual_seed(seed)
        self.cfg = cfg
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.net = ActorCritic(obs_dim, act_dim, cfg.hidden, cfg.init_log_std)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=cfg.learning_rate, eps=1e-5)
        self.obs_rms = RunningMeanStd((obs_dim,))
        self.ret_rms = RunningMeanStd(())
        self._ret_acc = np.zeros(cfg.num_envs)

        T, N = cfg.rollout_steps, cfg.num_envs
        self.buf_obs = np.zeros((T, N, obs_dim), dtype=np.float32)
        self.buf_act = np.zeros((T, N, act_dim), dtype=np.float32)
        self.buf_logp = np.zeros((T, N), dtype=np.float32)
        self.buf_rew = np.zeros((T, N), dtype=np.float32)
        self.buf_done = np.zeros((T, N), dtype=np.float32)
        self.buf_val = np.zeros((T, N), dtype=np.float32)
        self.t = 0
        self.total_steps = 0
        self.updates = 0
        self.stats = {"policy_loss": float("nan"), "value_loss": float("nan"), "kl": float("nan")}

    # ------------------------------------------------------------------ #
    def normalize(self, obs: np.ndarray) -> np.ndarray:
        z = (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8)
        return np.clip(z, -OBS_CLIP, OBS_CLIP).astype(np.float32)

    @property
    def action_std(self) -> float:
        return float(self.net.log_std.exp().mean())

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False, learn: bool = True):
        """Eylem seçer. learn=True ise gözlem istatistikleri güncellenir.

        Döner: (eylemler, normalize gözlem, log-olasılık, değer)
        """
        if learn:
            self.obs_rms.update(obs)
        norm = self.normalize(obs)
        x = torch.as_tensor(norm)
        dist = self.net.distribution(x)
        action = dist.mean if deterministic else dist.sample()
        return (action.numpy(), norm, dist.log_prob(action).sum(-1).numpy(), self.net.value(x).numpy())

    @torch.no_grad()
    def record(self, norm_obs, actions, logp, values, rewards, terminated, truncated, final_obs) -> bool:
        """Bir ortam adımını tampona yazar. Tampon dolduysa True (güncelleme zamanı)."""
        cfg = self.cfg
        ended = terminated | truncated
        self._ret_acc = self._ret_acc * cfg.gamma + rewards
        self.ret_rms.update(self._ret_acc)
        self._ret_acc[ended] = 0.0
        scaled = np.clip(rewards / np.sqrt(self.ret_rms.var + 1e-8), -REWARD_CLIP, REWARD_CLIP)
        if truncated.any():
            # Süre doldu ama düşmedi: gelecekteki ödüller hâlâ vardı, değer tahminiyle ekle.
            tail = self.net.value(torch.as_tensor(self.normalize(final_obs[truncated]))).numpy()
            scaled[truncated] += cfg.gamma * tail

        self.buf_obs[self.t] = norm_obs
        self.buf_act[self.t] = actions
        self.buf_logp[self.t] = logp
        self.buf_val[self.t] = values
        self.buf_rew[self.t] = scaled
        self.buf_done[self.t] = ended
        self.t += 1
        self.total_steps += cfg.num_envs
        return self.t >= cfg.rollout_steps

    def update(self, next_obs: np.ndarray) -> Iterator[None]:
        """Toplanan deneyimle politikayı iyileştirir; her mini-partiden sonra durur (yield)."""
        cfg = self.cfg
        with torch.no_grad():
            next_value = self.net.value(torch.as_tensor(self.normalize(next_obs))).numpy()
        advantages = np.zeros_like(self.buf_rew)
        last = np.zeros(cfg.num_envs, dtype=np.float32)
        for t in reversed(range(cfg.rollout_steps)):
            following = next_value if t == cfg.rollout_steps - 1 else self.buf_val[t + 1]
            alive = 1.0 - self.buf_done[t]
            delta = self.buf_rew[t] + cfg.gamma * following * alive - self.buf_val[t]
            last = delta + cfg.gamma * cfg.gae_lambda * alive * last
            advantages[t] = last
        returns = advantages + self.buf_val

        b_obs = torch.as_tensor(self.buf_obs.reshape(-1, self.obs_dim))
        b_act = torch.as_tensor(self.buf_act.reshape(-1, self.act_dim))
        b_logp = torch.as_tensor(self.buf_logp.reshape(-1))
        b_adv = torch.as_tensor(advantages.reshape(-1))
        b_ret = torch.as_tensor(returns.reshape(-1))
        size = b_obs.shape[0]
        mb = size // cfg.minibatches

        for _ in range(cfg.epochs):
            perm = torch.randperm(size)
            kls = []
            for start in range(0, size, mb):
                idx = perm[start:start + mb]
                kls.append(self._minibatch(b_obs[idx], b_act[idx], b_logp[idx], b_adv[idx], b_ret[idx]))
                yield
            if float(np.mean(kls)) > cfg.target_kl:
                break
        self.t = 0
        self.updates += 1

    def _minibatch(self, obs, act, old_logp, adv, ret) -> float:
        cfg = self.cfg
        dist = self.net.distribution(obs)
        logp = dist.log_prob(act).sum(-1)
        ratio = (logp - old_logp).exp()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        policy_loss = torch.max(-adv * ratio, -adv * ratio.clamp(1 - cfg.clip, 1 + cfg.clip)).mean()
        value_loss = 0.5 * (self.net.value(obs) - ret).pow(2).mean()
        entropy = dist.entropy().sum(-1).mean()
        loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
        self.optimizer.step()

        with torch.no_grad():
            kl = float(((ratio - 1) - (logp - old_logp)).mean())
        self.stats = {"policy_loss": float(policy_loss), "value_loss": float(value_loss), "kl": kl}
        return kl

    # ------------------------------------------------------------------ #
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save({
            "version": CHECKPOINT_VERSION,
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "hidden": self.cfg.hidden,
            "net": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "obs_rms": self.obs_rms.state(),
            "ret_rms": self.ret_rms.state(),
            "total_steps": self.total_steps,
            "updates": self.updates,
        }, tmp)
        tmp.replace(path)

    def load(self, path: Path) -> None:
        """Kayıtlı modeli yükler; bozuk ya da uyumsuzsa ValueError."""
        try:
            data = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:  # bozuk/yarım dosya
            raise ValueError(f"Model dosyası okunamadı ({path}): {exc}") from exc
        expected = (CHECKPOINT_VERSION, self.obs_dim, self.act_dim, self.cfg.hidden)
        found = tuple(data.get(k) for k in ("version", "obs_dim", "act_dim", "hidden")) if isinstance(data, dict) else None
        if found != expected:
            raise ValueError(f"Kayıtlı model bu sürümle uyumsuz ({found} ≠ {expected}).")
        self.net.load_state_dict(data["net"])
        self.optimizer.load_state_dict(data["optimizer"])
        self.obs_rms.load(data["obs_rms"])
        self.ret_rms.load(data["ret_rms"])
        self.total_steps = int(data["total_steps"])
        self.updates = int(data["updates"])
