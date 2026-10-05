"""Off-policy actor-critics for continuous actions: DDPG, TD3 and SAC.

All three keep a replay memory and critics that learn Q(s, a) from targets computed by slowly following copies of the
networks. The actor climbs the critic: DDPG and TD3 have a deterministic actor, a = high · tanh(net(s)), and follow the
critic's gradient with respect to the action; SAC has a squashed Gaussian, a = high · tanh(μ + σ ε), and climbs the
critic plus an entropy bonus through the same reparameterization. TD3 adds twin critics (the smaller target wins),
delayed actor updates and noise on the target action; SAC uses twin critics and tunes the bonus's weight α so that the
policy's entropy stays near a target.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nets import MLP, Adam

LOG_STD = (-5.0, 2.0)  # SAC: the log spread is kept in this range


@dataclass
class ACConfig:
    algo: str = "td3"  # "ddpg", "td3" or "sac"
    lr_actor: float = 1e-3
    lr_critic: float = 1e-3
    gamma: float = 0.99
    tau: float = 0.005  # how far the target copies move toward the networks at every update
    batch: int = 128
    buffer: int = 100_000
    start: int = 1_000  # steps of uniformly random actions before learning starts
    noise: float = 0.1  # DDPG, TD3: exploration noise, as a share of the largest action
    policy_noise: float = 0.2  # TD3: noise on the target action (share of the largest action)
    noise_clip: float = 0.5  # TD3: limit on that noise (share of the largest action)
    delay: int = 2  # TD3: critic updates per actor (and target) update
    alpha: float = 0.2  # SAC: the entropy weight, or where its tuning starts
    auto_alpha: bool = True  # SAC: tune α toward the target entropy, −(number of action dimensions)
    lr_alpha: float = 3e-4
    hidden: tuple[int, ...] = (64, 64)
    reward_scale: float = 1.0  # rewards are multiplied by this for learning (the recordings keep the real ones)


class Memory:
    """The last `capacity` transitions, with continuous actions; sampled uniformly."""

    def __init__(self, capacity: int, obs_dim: int, act_dim: int) -> None:
        self.S, self.S2 = np.zeros((capacity, obs_dim)), np.zeros((capacity, obs_dim))
        self.A, self.R, self.D = np.zeros((capacity, act_dim)), np.zeros(capacity), np.zeros(capacity)
        self.capacity, self.size, self.at = capacity, 0, 0

    def add(self, s, a, r, s2, done) -> None:
        i = self.at
        self.S[i], self.A[i], self.R[i], self.S2[i], self.D[i] = s, a, r, s2, done
        self.at, self.size = (i + 1) % self.capacity, min(self.size + 1, self.capacity)

    def sample(self, n: int, rng: np.random.Generator):
        i = rng.integers(0, self.size, n)
        return self.S[i], self.A[i], self.R[i], self.S2[i], self.D[i]


class AC:
    def __init__(self, obs_dim: int, space, cfg: ACConfig, rng: np.random.Generator, total: int) -> None:
        self.cfg, self.rng, self.total, self.t, self.updates = cfg, rng, total, 0, 0
        self.obs_dim, self.act_dim, self.high = obs_dim, space.shape[0], float(space.high[0])
        sac = cfg.algo == "sac"
        self.actor = MLP([obs_dim, *cfg.hidden, (2 if sac else 1) * self.act_dim], rng, out_scale=0.1)
        self.critics = [MLP([obs_dim + self.act_dim, *cfg.hidden, 1], rng) for _ in range(1 if cfg.algo == "ddpg" else 2)]
        self.critic_targets = [MLP([obs_dim + self.act_dim, *cfg.hidden, 1], rng) for _ in self.critics]
        for c, ct in zip(self.critics, self.critic_targets):
            ct.copy_from(c)
        if not sac:
            self.actor_target = MLP([obs_dim, *cfg.hidden, self.act_dim], rng)
            self.actor_target.copy_from(self.actor)
        self.opt_actor = Adam(self.actor.params, cfg.lr_actor)
        self.opt_critics = [Adam(c.params, cfg.lr_critic) for c in self.critics]
        self.log_alpha = np.log(cfg.alpha)
        self.alpha_opt_m = self.alpha_opt_v = 0.0
        self.memory = Memory(cfg.buffer, obs_dim, self.act_dim)
        self.log = {"q": [], "loss": [], "alpha": []}

    # ---- the actor ----
    def _sac_parts(self, s: np.ndarray, eps: np.ndarray | None):
        out, cache = self.actor.forward(s)
        mu, log_std_raw = out[:, : self.act_dim], out[:, self.act_dim :]
        log_std = np.clip(log_std_raw, *LOG_STD)
        std = np.exp(log_std)
        u = mu + std * (eps if eps is not None else 0.0)
        return mu, log_std, log_std_raw, std, u, cache

    def _sac_logp(self, eps: np.ndarray, log_std: np.ndarray, u: np.ndarray) -> np.ndarray:
        """log π(a | s) of a = high · tanh(u), u = μ + σ ε: the Gaussian's, minus the log of the squashing's slope."""
        gauss = (-0.5 * eps**2 - log_std - 0.5 * np.log(2 * np.pi)).sum(axis=1)
        return gauss - np.log(self.high * (1 - np.tanh(u) ** 2) + 1e-6).sum(axis=1)

    def policy(self, s: np.ndarray, greedy: bool = False, rng: np.random.Generator | None = None) -> np.ndarray:
        """Actions for states s: the deterministic actor (plus noise when exploring), or SAC's sample (its mean when greedy)."""
        rng = rng or self.rng
        if self.cfg.algo == "sac":
            eps = None if greedy else rng.standard_normal((len(s), self.act_dim))
            u = self._sac_parts(s, eps)[4]
            return self.high * np.tanh(u)
        a = self.high * np.tanh(self.actor(s))
        if not greedy:
            a = a + self.cfg.noise * self.high * rng.standard_normal(a.shape)
        return np.clip(a, -self.high, self.high)

    def spread(self, s: np.ndarray) -> np.ndarray:
        """How widely the policy tries actions around its choice: SAC's σ, or the exploration noise (in action units)."""
        if self.cfg.algo == "sac":  # σ of u, carried through the squashing's slope at the mean: torque units
            mu, _, _, std, _, _ = self._sac_parts(s, None)
            return (self.high * std * (1 - np.tanh(mu) ** 2))[:, 0]
        return np.full(len(s), self.cfg.noise * self.high)

    def value(self, s: np.ndarray) -> np.ndarray:
        """Q(s, the greedy action) by the first critic: what the agent thinks a state is worth."""
        a = self.policy(s, greedy=True)
        return self.critics[0](np.hstack([s, a]))[:, 0]

    # ---- acting and learning ----
    def act(self, s: np.ndarray) -> np.ndarray:
        if self.t < self.cfg.start:
            return self.rng.uniform(-self.high, self.high, self.act_dim)
        return self.policy(s[None])[0]

    def observe(self, s, a, r, s2, done) -> None:
        self.memory.add(s, a, r * self.cfg.reward_scale, s2, float(done))
        self.t += 1
        if self.t >= self.cfg.start:
            self.learn()

    def learn(self) -> None:
        cfg, B = self.cfg, self.cfg.batch
        S, A, R, S2, D = self.memory.sample(B, self.rng)
        self.updates += 1
        alpha = float(np.exp(self.log_alpha))
        # the targets, from the slowly following copies
        if cfg.algo == "sac":
            eps2 = self.rng.standard_normal((B, self.act_dim))
            _, log_std2, _, _, u2, _ = self._sac_parts(S2, eps2)
            A2, logp2 = self.high * np.tanh(u2), self._sac_logp(eps2, log_std2, u2)
        else:
            A2 = self.high * np.tanh(self.actor_target(S2))
            if cfg.algo == "td3":  # smooth the target: nearby actions should be worth about the same
                noise = np.clip(cfg.policy_noise * self.high * self.rng.standard_normal(A2.shape), -cfg.noise_clip * self.high, cfg.noise_clip * self.high)
                A2 = np.clip(A2 + noise, -self.high, self.high)
        X2 = np.hstack([S2, A2])
        q2 = np.min([ct(X2)[:, 0] for ct in self.critic_targets], axis=0)
        if cfg.algo == "sac":
            q2 = q2 - alpha * logp2
        y = R + cfg.gamma * (1 - D) * q2
        # the critics: a regression step toward the targets
        X = np.hstack([S, A])
        loss = 0.0
        for c, opt in zip(self.critics, self.opt_critics):
            out, cache = c.forward(X)
            err = out[:, 0] - y
            loss += float((err**2).mean())
            opt.step(c.backward(cache, err[:, None] / B))
        self.log["loss"].append(loss / len(self.critics))
        self.log["q"].append(float(y.mean()))
        # the actor, and the targets with it (TD3: every `delay` critic updates)
        if cfg.algo == "td3" and self.updates % cfg.delay:
            return
        if cfg.algo == "sac":
            self._actor_step_sac(S, alpha)
        else:
            self._actor_step_det(S)
            self.actor_target.soft_update(self.actor, cfg.tau)
        for c, ct in zip(self.critics, self.critic_targets):
            ct.soft_update(c, cfg.tau)

    def _actor_step_det(self, S: np.ndarray) -> None:
        """Climb Q(s, μ(s)): the critic's gradient with respect to the action, carried back through tanh and the actor."""
        z, acache = self.actor.forward(S)
        a = self.high * np.tanh(z)
        c = self.critics[0]
        _, ccache = c.forward(np.hstack([S, a]))
        dq_da = c.input_grad(ccache, np.ones((len(S), 1)))[:, self.obs_dim :]
        dl_dz = -dq_da * self.high * (1 - np.tanh(z) ** 2) / len(S)  # loss = −mean Q
        self.opt_actor.step(self.actor.backward(acache, dl_dz))

    def _actor_step_sac(self, S: np.ndarray, alpha: float) -> None:
        """Climb min(Q1, Q2)(s, a) − α log π(a | s), a = high · tanh(μ + σ ε), through the sample (reparameterization)."""
        n = len(S)
        eps = self.rng.standard_normal((n, self.act_dim))
        mu, log_std, log_std_raw, std, u, acache = self._sac_parts(S, eps)
        a, t = self.high * np.tanh(u), np.tanh(u)
        X = np.hstack([S, a])
        outs = [c.forward(X) for c in self.critics]
        qs = np.stack([o[0][:, 0] for o in outs])
        pick = np.argmin(qs, axis=0)  # the gradient of the min is the gradient of whichever critic is smaller
        dq_da = np.zeros((n, self.act_dim))
        for k, (c, (_, cache)) in enumerate(zip(self.critics, outs)):
            mask = (pick == k).astype(float)[:, None]
            dq_da += mask * c.input_grad(cache, np.ones((n, 1)))[:, self.obs_dim :]
        # loss = mean(α log π − min Q); log π depends on u through the squashing (d/du of −log(1 − tanh²) is 2 tanh)
        dl_du = (alpha * 2 * t - dq_da * self.high * (1 - t**2)) / n
        dl_dmu = dl_du
        dl_dlogstd = (dl_du * std * eps - alpha / n) * ((log_std_raw > LOG_STD[0]) & (log_std_raw < LOG_STD[1]))
        self.opt_actor.step(self.actor.backward(acache, np.hstack([dl_dmu, dl_dlogstd])))
        if self.cfg.auto_alpha:  # α rises when the policy is less random than the target entropy, and falls when more
            logp = self._sac_logp(eps, log_std, u)
            g = -float((logp - self.act_dim).mean())  # d/d(log α) of −log α · (log π + target), target = −act_dim
            self.alpha_opt_m = 0.9 * self.alpha_opt_m + 0.1 * g
            self.alpha_opt_v = 0.999 * self.alpha_opt_v + 0.001 * g * g
            k = self.updates
            self.log_alpha -= self.cfg.lr_alpha * (self.alpha_opt_m / (1 - 0.9**k)) / (np.sqrt(self.alpha_opt_v / (1 - 0.999**k)) + 1e-8)
        self.log["alpha"].append(float(np.exp(self.log_alpha)))
