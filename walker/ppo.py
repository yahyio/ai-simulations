"""PPO (Proximal Policy Optimization) — sürekli eylemler (eklem hedef açıları) için.

- Gözlem normalizasyonu ve getiri ölçekleme (akan istatistiklerle).
- GAE avantajları; süre dolunca (düşmeden) değer tahminiyle bootstrap.
- KL'ye göre kendini ayarlayan öğrenme oranı (bacaklı robot eğitiminde yaygın yöntem).
- Güncelleme mini-parti adımlarına bölünmüş bir üreteçtir; arayüz her karede birkaç
  adım işleyerek öğrenme sırasında donmadan akmaya devam eder.
"""
from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

CHECKPOINT_VERSION = 4   # 3: eklemler eylem sırasında · 4: gözlemde adım saati
OBS_CLIP = 10.0
REWARD_CLIP = 10.0
MAX_CPU_THREADS = 4
LR_BOUNDS = (1e-5, 1e-2)
LR_FACTOR = 1.5


@dataclass(frozen=True)
class PPOConfig:
    num_envs: int = 48
    rollout_steps: int = 128        # ortam başına, her güncellemeden önce
    epochs: int = 5
    minibatches: int = 4
    learning_rate: float = 3e-4
    # İsteğe bağlı KL denetimleri. 21 boyutlu, dar (σ≈0.3) bir politikada KL küçük adımlarda
    # bile hızla büyüdüğünden adaptif oran öğrenmeyi durma noktasına getiriyordu (ölçüldü);
    # varsayılan: sabit oran + PPO kırpması.
    desired_kl: float | None = None     # verilirse öğrenme oranı KL'ye göre kendini ayarlar
    target_kl: float | None = None      # verilirse tur ortalama KL'si bunu aşınca güncelleme erken biter
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    value_coef: float = 1.0
    entropy_coef: float = 0.002
    max_grad_norm: float = 1.0
    hidden: tuple[int, ...] = (256, 256)
    init_log_std: float = -1.2      # std ≈ 0.3: başlangıç denemeleri sert ama çok sert değil
    device: str = "cpu"             # cpu | cuda (küçük ağlarda CPU genelde daha hızlı; ölçüldü)

    def __post_init__(self) -> None:
        if min(self.num_envs, self.rollout_steps, self.epochs, self.minibatches, *self.hidden) < 1:
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
        self.mean = state["mean"].cpu().numpy().copy()
        self.var = state["var"].cpu().numpy().copy()
        self.count = float(state["count"])


def _layer(inp: int, out: int, std: float = np.sqrt(2)) -> nn.Linear:
    layer = nn.Linear(inp, out)
    nn.init.orthogonal_(layer.weight, std)
    nn.init.zeros_(layer.bias)
    return layer


def _mlp(inp: int, hidden: tuple[int, ...], out: int, out_std: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    size = inp
    for width in hidden:
        layers += [_layer(size, width), nn.ELU()]
        size = width
    layers.append(_layer(size, out, out_std))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: tuple[int, ...], init_log_std: float):
        super().__init__()
        self.actor = _mlp(obs_dim, hidden, act_dim, 0.01)
        self.critic = _mlp(obs_dim, hidden, 1, 1.0)
        self.log_std = nn.Parameter(torch.full((act_dim,), init_log_std))

    def distribution(self, obs: torch.Tensor) -> torch.distributions.Normal:
        return torch.distributions.Normal(self.actor(obs), self.log_std.exp().expand(obs.shape[0], -1))

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)


def expected_header(obs_dim: int, act_dim: int, cfg: PPOConfig) -> tuple:
    """Uyumlu bir model dosyasının taşıması gereken (sürüm, gözlem, eylem, katmanlar) bilgisi."""
    return CHECKPOINT_VERSION, obs_dim, act_dim, list(cfg.hidden)


def _header(data) -> tuple | None:
    if not isinstance(data, dict):
        return None
    hidden = data.get("hidden")
    # Eski (2B) kayıtlarda katman genişliği tek bir tamsayıydı.
    layers = list(hidden) if isinstance(hidden, (list, tuple)) else [hidden] if isinstance(hidden, int) else []
    return data.get("version"), data.get("obs_dim"), data.get("act_dim"), layers


def read_checkpoint_header(path: Path) -> tuple | None:
    """Model dosyasının üst bilgisi; okunamıyorsa None."""
    try:
        return _header(torch.load(path, map_location="cpu", weights_only=True))
    except Exception:  # bozuk/yarım ya da tanınmayan dosya: uyumsuz sayılır
        return None


def resolve_device(preference: str) -> torch.device:
    if preference == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class PPO:
    def __init__(self, obs_dim: int, act_dim: int, cfg: PPOConfig = PPOConfig(), seed: int | None = None):
        torch.set_num_threads(max(1, min(MAX_CPU_THREADS, (os.cpu_count() or 2) // 2)))
        if seed is not None:
            torch.manual_seed(seed)
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.net = ActorCritic(obs_dim, act_dim, cfg.hidden, cfg.init_log_std).to(self.device)
        self.lr = cfg.learning_rate
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, eps=1e-5)
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
        x = torch.as_tensor(norm, device=self.device)
        dist = self.net.distribution(x)
        action = dist.mean if deterministic else dist.sample()
        return (action.cpu().numpy(), norm, dist.log_prob(action).sum(-1).cpu().numpy(),
                self.net.value(x).cpu().numpy())

    def _values(self, obs: np.ndarray) -> np.ndarray:
        return self.net.value(torch.as_tensor(self.normalize(obs), device=self.device)).cpu().numpy()

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
            scaled[truncated] += cfg.gamma * self._values(final_obs[truncated])

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
            next_value = self._values(next_obs)
        advantages = np.zeros_like(self.buf_rew)
        last = np.zeros(cfg.num_envs, dtype=np.float32)
        for t in reversed(range(cfg.rollout_steps)):
            following = next_value if t == cfg.rollout_steps - 1 else self.buf_val[t + 1]
            alive = 1.0 - self.buf_done[t]
            delta = self.buf_rew[t] + cfg.gamma * following * alive - self.buf_val[t]
            last = delta + cfg.gamma * cfg.gae_lambda * alive * last
            advantages[t] = last
        returns = advantages + self.buf_val

        def tensor(array: np.ndarray, width: int | None = None) -> torch.Tensor:
            flat = array.reshape(-1, width) if width else array.reshape(-1)
            return torch.as_tensor(flat, device=self.device)

        b_obs = tensor(self.buf_obs, self.obs_dim)
        b_act = tensor(self.buf_act, self.act_dim)
        b_logp = tensor(self.buf_logp)
        b_adv = tensor(advantages)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
        b_ret = tensor(returns)
        with torch.no_grad():   # veriyi toplayan (eski) politikanın dağılımı — analitik KL için
            b_mu_old = self.net.actor(b_obs)
            std_old = self.net.log_std.exp().detach().clone()
        size = b_obs.shape[0]
        mb = size // cfg.minibatches

        for _ in range(cfg.epochs):
            perm = torch.randperm(size, device=self.device)
            kls = []
            for start in range(0, mb * cfg.minibatches, mb):
                idx = perm[start:start + mb]
                self._minibatch(b_obs[idx], b_act[idx], b_logp[idx], b_adv[idx], b_ret[idx], b_mu_old[idx], std_old)
                kls.append(self.stats["kl"])
                yield
            if cfg.target_kl is not None and float(np.mean(kls)) > cfg.target_kl:
                break
        self.t = 0
        self.updates += 1

    def _minibatch(self, obs, act, old_logp, adv, ret, mu_old, std_old) -> None:
        cfg = self.cfg
        dist = self.net.distribution(obs)
        logp = dist.log_prob(act).sum(-1)
        ratio = (logp - old_logp).exp()
        policy_loss = torch.max(-adv * ratio, -adv * ratio.clamp(1 - cfg.clip, 1 + cfg.clip)).mean()
        value_loss = 0.5 * (self.net.value(obs) - ret).pow(2).mean()
        entropy = dist.entropy().sum(-1).mean()
        loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

        with torch.no_grad():
            # İki köşegen Gauss arasındaki KL (örneklemeye dayalı tahminden çok daha az gürültülü).
            mu, std = dist.mean, dist.stddev
            kl = float((torch.log(std / std_old) + (std_old**2 + (mu_old - mu) ** 2) / (2 * std**2) - 0.5)
                       .sum(-1).mean())

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
        self.optimizer.step()

        if cfg.desired_kl is not None:
            # Politika çok değiştiyse yavaşla, az değiştiyse hızlan.
            if kl > 2.0 * cfg.desired_kl:
                self.lr = max(LR_BOUNDS[0], self.lr / LR_FACTOR)
            elif kl < 0.5 * cfg.desired_kl:
                self.lr = min(LR_BOUNDS[1], self.lr * LR_FACTOR)
            for group in self.optimizer.param_groups:
                group["lr"] = self.lr
        self.stats = {"policy_loss": policy_loss.detach().item(), "value_loss": value_loss.detach().item(), "kl": kl}

    # ------------------------------------------------------------------ #
    def save(self, path: Path, include_optimizer: bool = True) -> None:
        """Modeli kaydeder. include_optimizer=False: yalnızca izlemeye yetecek hafif kayıt."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        data = {
            "version": CHECKPOINT_VERSION,
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "hidden": list(self.cfg.hidden),
            "net": self.net.state_dict(),
            "obs_rms": self.obs_rms.state(),
            "ret_rms": self.ret_rms.state(),
            "lr": self.lr,
            "total_steps": self.total_steps,
            "updates": self.updates,
        }
        if include_optimizer:
            data["optimizer"] = self.optimizer.state_dict()
        torch.save(data, tmp)
        tmp.replace(path)

    def load(self, path: Path) -> None:
        """Kayıtlı modeli yükler; bozuk ya da uyumsuzsa ValueError."""
        try:
            data = torch.load(path, map_location=self.device, weights_only=True)
        except Exception as exc:  # bozuk/yarım dosya
            raise ValueError(f"Model dosyası okunamadı ({path}): {exc}") from exc
        expected = expected_header(self.obs_dim, self.act_dim, self.cfg)
        found = _header(data)
        if found != expected:
            raise ValueError(f"Kayıtlı model bu sürümle uyumsuz ({found} ≠ {expected}).")
        self.net.load_state_dict(data["net"])
        if "optimizer" in data:   # hafif (yalnızca izleme) kayıtlarda yok
            self.optimizer.load_state_dict(data["optimizer"])
        self.obs_rms.load(data["obs_rms"])
        self.ret_rms.load(data["ret_rms"])
        self.lr = float(data["lr"])
        for group in self.optimizer.param_groups:
            group["lr"] = self.lr
        self.total_steps = int(data["total_steps"])
        self.updates = int(data["updates"])
