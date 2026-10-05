"""DQN and its parts, each of which can be switched off or upgraded, so that one implementation records every comparison
of Part 9: experience replay or only the latest experience, a target network or none, double Q-learning, a dueling
network, prioritized replay.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from nets import MLP, Adam


@dataclass
class DQNConfig:
    lr: float = 1e-3
    gamma: float = 0.99
    batch: int = 64
    buffer: int = 50_000  # replay capacity; equal to batch means "only the latest experience"
    start: int = 1_000  # steps of random play before learning starts
    train_every: int = 1
    target_every: int = 500  # steps between copies to the target network; 0: no target network
    tau: float = 1.0  # 1: copy the network; below 1, move the target that fraction of the way (soft updates)
    grad_steps: int = 1  # gradient steps each time it trains
    eps: tuple[float, float, int] = (1.0, 0.05, 10_000)  # ε from, to, over how many steps
    double: bool = False
    dueling: bool = False
    prioritized: bool = False
    alpha: float = 0.6  # prioritization strength
    beta: tuple[float, float] = (0.4, 1.0)  # importance-sampling correction, annealed over the run
    hidden: tuple[int, ...] = (64, 64)
    clip: float = 10.0  # gradient norm limit
    huber: bool = True  # Huber loss (squared for small errors, linear for large); False: squared error
    extra: dict = field(default_factory=dict)


class Replay:
    """A circular buffer of transitions, sampled uniformly. With capacity = batch it is just the latest experience."""

    def __init__(self, capacity: int, obs_dim: int) -> None:
        self.s = np.zeros((capacity, obs_dim))
        self.s2 = np.zeros((capacity, obs_dim))
        self.a = np.zeros(capacity, dtype=np.int64)
        self.r = np.zeros(capacity)
        self.done = np.zeros(capacity)
        self.capacity, self.next, self.size = capacity, 0, 0

    def add(self, s, a, r, s2, done) -> int:
        i = self.next
        self.s[i], self.a[i], self.r[i], self.s2[i], self.done[i] = s, a, r, s2, done
        self.next = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return i

    def sample(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        if self.size <= n:  # the whole (small) buffer: the latest experience
            idx = np.arange(self.size)
        else:
            idx = rng.integers(0, self.size, n)
        return idx, np.ones(len(idx))

    def update(self, idx: np.ndarray, td: np.ndarray) -> None:
        pass


class PrioritizedReplay(Replay):
    """Proportional prioritized replay (Schaul et al., 2016): a transition is sampled with probability proportional to
    (|δ| + ε)^α, using a sum tree; the importance-sampling weights (N P(i))^-β, divided by their largest, undo the bias."""

    def __init__(self, capacity: int, obs_dim: int, alpha: float) -> None:
        super().__init__(capacity, obs_dim)
        self.alpha, self.top = alpha, 1.0
        self.tree = np.zeros(2 * capacity)  # leaves at capacity..2·capacity−1
        self.beta = 0.4

    def _set(self, i: int, p: float) -> None:
        j = i + self.capacity
        delta = p - self.tree[j]
        while j >= 1:
            self.tree[j] += delta
            j //= 2

    def add(self, s, a, r, s2, done) -> int:
        i = super().add(s, a, r, s2, done)
        self._set(i, self.top)  # new transitions get the largest priority so far: each is replayed at least once soon
        return i

    def sample(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        total, idx = self.tree[1], np.zeros(n, dtype=np.int64)
        # stratified: one sample from each of n equal slices of the total priority
        for k, u in enumerate((np.arange(n) + rng.random(n)) * total / n):
            j = 1
            while j < self.capacity:
                j *= 2
                if u > self.tree[j]:
                    u -= self.tree[j]
                    j += 1
            idx[k] = min(j - self.capacity, self.size - 1)
        p = self.tree[idx + self.capacity] / total
        w = (self.size * p) ** -self.beta
        return idx, w / w.max()

    def update(self, idx: np.ndarray, td: np.ndarray) -> None:
        pr = (np.abs(td) + 1e-3) ** self.alpha
        self.top = max(self.top, float(pr.max()))
        for i, p in zip(idx, pr):
            self._set(int(i), float(p))


class QNet:
    """Action values from a network: plain, or dueling (one value V(s) and one advantage per action, Q = V + A − mean A)."""

    def __init__(self, obs_dim: int, n_actions: int, cfg: DQNConfig, rng: np.random.Generator) -> None:
        self.dueling, self.n = cfg.dueling, n_actions
        self.net = MLP([obs_dim, *cfg.hidden, n_actions + (1 if cfg.dueling else 0)], rng)

    @property
    def params(self):
        return self.net.params

    def copy_from(self, other: QNet) -> None:
        self.net.copy_from(other.net)

    def forward(self, s: np.ndarray):
        out, cache = self.net.forward(s)
        if not self.dueling:
            return out, cache
        v, a = out[:, :1], out[:, 1:]
        return v + a - a.mean(axis=1, keepdims=True), cache

    def __call__(self, s: np.ndarray) -> np.ndarray:
        return self.forward(s)[0]

    def backward(self, cache, dq: np.ndarray):
        if self.dueling:
            dq = np.concatenate([dq.sum(axis=1, keepdims=True), dq - dq.mean(axis=1, keepdims=True)], axis=1)
        return self.net.backward(cache, dq)


class DQN:
    """The learner: ε-greedy acting, a replay memory, a target network, and one gradient step per training step."""

    def __init__(self, obs_dim: int, n_actions: int, cfg: DQNConfig, rng: np.random.Generator, total_steps: int) -> None:
        self.cfg, self.rng, self.n, self.total = cfg, rng, n_actions, total_steps
        self.q = QNet(obs_dim, n_actions, cfg, rng)
        self.target = QNet(obs_dim, n_actions, cfg, rng)
        self.target.copy_from(self.q)
        self.opt = Adam(self.q.params, cfg.lr, cfg.clip)
        self.memory = PrioritizedReplay(cfg.buffer, obs_dim, cfg.alpha) if cfg.prioritized else Replay(cfg.buffer, obs_dim)
        self.t = 0
        self.log = {"loss": [], "td": [], "q": []}  # per training step, emptied at each snapshot

    def epsilon(self) -> float:
        hi, lo, over = self.cfg.eps
        return lo + (hi - lo) * max(0.0, 1 - self.t / over)

    def act(self, s: np.ndarray, greedy: bool = False) -> int:
        if not greedy and self.rng.random() < self.epsilon():
            return int(self.rng.integers(self.n))
        return int(np.argmax(self.q(s[None])[0]))

    def observe(self, s, a, r, s2, done: bool) -> None:
        """One step of experience: store it, learn from a batch, keep the target network in step."""
        cfg = self.cfg
        self.memory.add(s, a, r, s2, float(done))
        self.t += 1
        if cfg.prioritized:
            b0, b1 = cfg.beta
            self.memory.beta = b0 + (b1 - b0) * min(1.0, self.t / self.total)
        if self.t >= cfg.start and self.t % cfg.train_every == 0:
            for _ in range(cfg.grad_steps):
                self.learn()
        if cfg.target_every and self.t % cfg.target_every == 0:
            if cfg.tau < 1:
                self.target.net.soft_update(self.q.net, cfg.tau)
            else:
                self.target.copy_from(self.q)

    def learn(self) -> None:
        cfg, m = self.cfg, self.memory
        idx, w = m.sample(cfg.batch, self.rng)
        s, a, r, s2, done = m.s[idx], m.a[idx], m.r[idx], m.s2[idx], m.done[idx]
        nxt = self.target if cfg.target_every else self.q  # no target network: bootstrap from the network being trained
        q2 = nxt(s2)
        if cfg.double:  # double DQN: the online network picks the next action, the target network values it
            best = np.argmax(self.q(s2), axis=1)
            boot = q2[np.arange(len(idx)), best]
        else:
            boot = q2.max(axis=1)
        y = r + cfg.gamma * (1 - done) * boot
        q, cache = self.q.forward(s)
        td = q[np.arange(len(idx)), a] - y
        # Huber loss: squared for small errors, linear for large ones, so a single surprise cannot blow up the step
        dq = np.zeros_like(q)
        dq[np.arange(len(idx)), a] = (np.clip(td, -1.0, 1.0) if cfg.huber else td) * w / len(idx)
        self.opt.step(self.q.backward(cache, dq))
        m.update(idx, td)
        self.log["loss"].append(float(np.mean(w * np.where(np.abs(td) < 1, 0.5 * td**2, np.abs(td) - 0.5))))
        self.log["td"].append(float(np.mean(np.abs(td))))
        self.log["q"].append(float(q.max(axis=1).mean()))
