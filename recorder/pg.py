"""Policy-gradient learners with networks: A2C, PPO and TRPO, for a softmax policy over a few actions or a Gaussian one
over a real-valued action. Several workers play at once; GAE judges every step; the critic is a separate network.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nets import MLP, Adam


@dataclass
class PGConfig:
    algo: str = "ppo"  # "a2c", "ppo" or "trpo"
    workers: int = 8
    steps: int = 128  # steps per worker between updates
    gamma: float = 0.99
    lam: float = 0.95  # GAE; 1 gives n-step returns
    lr: float = 3e-4  # the policy's step size (A2C, PPO)
    lr_v: float = 1e-3  # the critic's
    epochs: int = 10  # PPO: passes over each batch
    minibatch: int = 64  # PPO
    clip: float = 0.2  # PPO; 0: no clip
    ent: float = 0.0  # entropy bonus β
    delta: float = 0.01  # TRPO: the trust region
    v_epochs: int = 10  # critic passes per batch (in minibatches)
    hidden: tuple[int, ...] = (64, 64)
    norm_adv: bool = True
    log_std0: float = 0.0  # Gaussian: the first spread, in the action's unit
    reward_scale: float = 1.0  # rewards are multiplied by this for learning (the recordings keep the real ones)


class Policy:
    """π(a | s, θ): a softmax over logits, or a Gaussian with a learned mean and a log spread that is a free parameter."""

    def __init__(self, obs_dim: int, space, cfg: PGConfig, rng: np.random.Generator) -> None:
        self.discrete = hasattr(space, "n")
        out = space.n if self.discrete else space.shape[0]
        self.net = MLP([obs_dim, *cfg.hidden, out], rng, act="tanh", out_scale=0.01)
        self.log_std = None if self.discrete else np.full(out, cfg.log_std0)
        if not self.discrete:
            self.low, self.high = space.low, space.high

    @property
    def params(self) -> list[np.ndarray]:
        return self.net.params + ([] if self.discrete else [self.log_std])

    def dist(self, s: np.ndarray):
        """What the policy says in states s: probabilities, or means and spreads."""
        out = self.net(s)
        if self.discrete:
            z = np.exp(out - out.max(axis=1, keepdims=True))
            return z / z.sum(axis=1, keepdims=True)
        return out, np.exp(self.log_std)

    def sample(self, s: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if self.discrete:
            p = self.dist(s)
            # the index of the first cumulative probability above a uniform draw (rounding can never pass the last action)
            return np.minimum((rng.random((len(s), 1)) > np.cumsum(p, axis=1)).sum(axis=1), p.shape[1] - 1)
        mu, sd = self.dist(s)
        return mu + sd * rng.standard_normal(mu.shape)

    def log_prob(self, s: np.ndarray, a: np.ndarray) -> np.ndarray:
        if self.discrete:
            return np.log(self.dist(s)[np.arange(len(s)), a] + 1e-12)
        mu, sd = self.dist(s)
        return (-0.5 * ((a - mu) / sd) ** 2 - np.log(sd) - 0.5 * np.log(2 * np.pi)).sum(axis=1)

    def grad(self, s: np.ndarray, a: np.ndarray, w: np.ndarray, beta: float = 0.0) -> list[np.ndarray]:
        """The gradient of Σ_i w_i ln π(a_i | s_i) + β Σ_i H(π(· | s_i)), one array per parameter."""
        out, cache = self.net.forward(s)
        if self.discrete:
            z = np.exp(out - out.max(axis=1, keepdims=True))
            p = z / z.sum(axis=1, keepdims=True)
            d = -p * w[:, None]
            d[np.arange(len(s)), a] += w
            if beta:
                lp = np.log(p + 1e-12)
                H = -(p * lp).sum(axis=1, keepdims=True)
                d += beta * (-p * (lp + H))
            return self.net.backward(cache, d)
        sd = np.exp(self.log_std)
        zt = (a - out) / sd
        d_mu = w[:, None] * zt / sd
        d_ls = (w[:, None] * (zt**2 - 1)).sum(axis=0) + beta * len(s)  # ∂H/∂ln σ = 1 per sample and dimension
        return self.net.backward(cache, d_mu) + [d_ls]

    def kl_from(self, s: np.ndarray, old) -> float:
        """The average KL divergence KL(π_old ‖ π) over states s; old is what dist returned before the update."""
        if self.discrete:
            p = self.dist(s)
            return float((old * (np.log(old + 1e-12) - np.log(p + 1e-12))).sum(axis=1).mean())
        mu, sd = self.dist(s)
        mu0, sd0 = old
        return float((np.log(sd / sd0) + (sd0**2 + (mu0 - mu) ** 2) / (2 * sd**2) - 0.5).sum(axis=1).mean())

    def fisher_vector(self, s: np.ndarray, v: list[np.ndarray], eps: float = 1e-5) -> list[np.ndarray]:
        """F v, with F the Fisher information of the policy averaged over states s: the network's Jacobian carries v to
        its outputs (a central difference), the Fisher matrix of the distribution acts there, and backward carries it
        back. Exact for the softmax and for the Gaussian with a state-independent spread."""
        net_v = v[: len(self.net.params)]
        for p, d in zip(self.net.params, net_v):
            p += eps * d
        up = self.net(s)
        for p, d in zip(self.net.params, net_v):
            p -= 2 * eps * d
        down = self.net(s)
        for p, d in zip(self.net.params, net_v):
            p += eps * d
        jv = (up - down) / (2 * eps)
        out, cache = self.net.forward(s)
        n = len(s)
        if self.discrete:
            z = np.exp(out - out.max(axis=1, keepdims=True))
            p = z / z.sum(axis=1, keepdims=True)
            g = (p * jv - p * (p * jv).sum(axis=1, keepdims=True)) / n  # (diag π − π πᵀ) J v
            return self.net.backward(cache, g)
        var = np.exp(2 * self.log_std)
        return self.net.backward(cache, jv / var / n) + [2.0 * v[-1]]

    def action_for_env(self, a: np.ndarray) -> np.ndarray:
        return a if self.discrete else np.clip(a, self.low, self.high)


def gae(r, v, v_next, done, cut, gamma, lam):
    """Advantages and λ-returns for T steps of N workers (arrays T × N). done: the episode ended (no bootstrap); cut: it
    was cut by the time limit (bootstrap from the value of the last state, in v_next)."""
    T = len(r)
    adv = np.zeros_like(r)
    a = np.zeros(r.shape[1])
    for t in range(T - 1, -1, -1):
        delta = r[t] + gamma * v_next[t] * (1 - done[t]) - v[t]
        a = delta + gamma * lam * (1 - done[t]) * (1 - cut[t]) * a
        adv[t] = a
    return adv, adv + v


def conjugate_gradient(fvp, b: list[np.ndarray], iters: int = 10) -> list[np.ndarray]:
    """Solve F x = b approximately, with F available only through products F v."""
    x = [np.zeros_like(g) for g in b]
    r = [g.copy() for g in b]
    p = [g.copy() for g in b]
    rr = sum(float((g * g).sum()) for g in r)
    for _ in range(iters):
        fp = fvp(p)
        alpha = rr / (sum(float((a * c).sum()) for a, c in zip(p, fp)) + 1e-10)
        for xi, pi, ri, fi in zip(x, p, r, fp):
            xi += alpha * pi
            ri -= alpha * fi
        rr_new = sum(float((g * g).sum()) for g in r)
        if rr_new < 1e-10:
            break
        for pi, ri in zip(p, r):
            pi *= rr_new / rr
            pi += ri
        rr = rr_new
    return x


class PG:
    """A2C, PPO or TRPO with a policy network, a critic network and N workers."""

    def __init__(self, obs_dim: int, space, cfg: PGConfig, rng: np.random.Generator) -> None:
        self.cfg, self.rng = cfg, rng
        self.pi = Policy(obs_dim, space, cfg, rng)
        self.v = MLP([obs_dim, *cfg.hidden, 1], rng, act="tanh")
        self.opt_pi = Adam(self.pi.params, cfg.lr, clip=0.5)
        self.opt_v = Adam(self.v.params, cfg.lr_v, clip=0.5)
        self.stats: dict[str, float] = {}

    def value(self, s: np.ndarray) -> np.ndarray:
        return self.v(s)[:, 0]

    def update(self, s, a, adv, ret) -> None:
        """One update from a batch: s, a, advantages and λ-returns, flattened over steps and workers."""
        cfg = self.cfg
        if cfg.norm_adv:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        old = self.pi.dist(s)
        old = (old[0].copy(), old[1].copy()) if isinstance(old, tuple) else old.copy()
        logp_old = self.pi.log_prob(s, a)
        n = len(s)
        if cfg.algo == "a2c":
            g = self.pi.grad(s, a, adv / n, cfg.ent / n)
            self.opt_pi.step([-x for x in g])
            self.stats.update(clipped=0.0)
        elif cfg.algo == "ppo":
            clipped = 0
            for _ in range(cfg.epochs):
                order = self.rng.permutation(n)
                clipped = 0
                for k in range(0, n, cfg.minibatch):
                    i = order[k : k + cfg.minibatch]
                    r = np.exp(self.pi.log_prob(s[i], a[i]) - logp_old[i])
                    # a sample pushes only while its ratio is inside 1 ± ε in the direction its advantage favors
                    out = (adv[i] > 0) & (r > 1 + cfg.clip) | (adv[i] < 0) & (r < 1 - cfg.clip) if cfg.clip else np.zeros(len(i), bool)
                    clipped += int(out.sum())
                    w = np.where(out, 0.0, adv[i] * r) / len(i)
                    g = self.pi.grad(s[i], a[i], w, cfg.ent / len(i))
                    self.opt_pi.step([-x for x in g])
            self.stats.update(clipped=clipped / n)
        else:  # trpo
            g = self.pi.grad(s, a, adv / n)
            step = conjugate_gradient(lambda v: [f + 0.1 * x for f, x in zip(self.pi.fisher_vector(s, v), v)], g)
            shs = sum(float((x * fx).sum()) for x, fx in zip(step, self.pi.fisher_vector(s, step)))
            scale = np.sqrt(2 * cfg.delta / (shs + 1e-10))
            before = [p.copy() for p in self.pi.params]
            surr0 = float(np.mean(adv))  # the surrogate at the old policy: mean ratio (1) × advantage
            accepted = 0.0
            for frac in 0.5 ** np.arange(10):
                for p, b0, x in zip(self.pi.params, before, step):
                    p[...] = b0 + frac * scale * x
                surr = float(np.mean(np.exp(self.pi.log_prob(s, a) - logp_old) * adv))
                if self.pi.kl_from(s, old) <= cfg.delta and surr > surr0:
                    accepted = frac
                    break
            else:
                for p, b0 in zip(self.pi.params, before):
                    p[...] = b0
            self.stats.update(step=accepted)
        self.stats.update(kl=self.pi.kl_from(s, old))
        # the critic: a few passes of minibatch steps toward the λ-returns
        for _ in range(cfg.v_epochs):
            order = self.rng.permutation(n)
            for k in range(0, n, cfg.minibatch):
                i = order[k : k + cfg.minibatch]
                out, cache = self.v.forward(s[i])
                self.opt_v.step(self.v.backward(cache, (out[:, 0] - ret[i])[:, None] / len(i)))
