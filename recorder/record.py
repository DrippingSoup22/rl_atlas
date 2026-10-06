"""Record training runs for RL Atlas.

    python recorder/record.py              # every recording in the catalog
    python recorder/record.py dqn-cartpole # some of them
    python recorder/record.py --sweep      # the sweeps: each recording's knobs over several values, curves only

Trains small networks on Gymnasium's worlds and writes what the Lab and the stories play back, to
content/recordings/<name>.json: for every seed, the average return of the training episodes in each block of steps
and the return of one test episode after each block (the same start each time, played as the snapshots play it);
for one seed, a snapshot after every block: what the network thinks of a grid of states (values, and the action it
prefers), one test episode played with it, and the block's statistics. Needs Python 3.11+, NumPy and Gymnasium; every
run is reproducible from its seed. Seeds run in parallel, one process each.

A sweep trains a recording's settings again with one knob changed, for each of several values and every seed, and
keeps only the curves (training and test returns), in content/recordings/sweeps/<name>.json: how often a setting ends
well, and how that moves with the knob.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # one thread per process: faster for small networks, and reproducible
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import base64
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import gymnasium as gym
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dqn import DQN, DQNConfig  # noqa: E402
from pg import PG, PGConfig, gae  # noqa: E402
from ac import AC, ACConfig  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("RECORDINGS_OUT", ROOT / "content" / "recordings"))  # elsewhere, to compare before replacing
TEST_SEED = 2024  # every snapshot's test episode starts from the same state, so snapshots compare like with like
log = logging.getLogger("record")


# ---- the worlds: Gymnasium's id, the slice of states a snapshot maps, and how states are drawn ----
def cartpole_grid(xs, ys):
    """Angle × angular velocity, with the cart at rest in the middle."""
    th, om = np.meshgrid(xs, ys)
    z = np.zeros_like(th)
    return np.stack([z, z, th, om], axis=-1).reshape(-1, 4)


def mountaincar_grid(xs, ys):
    x, v = np.meshgrid(xs, ys)
    return np.stack([x, v], axis=-1).reshape(-1, 2)


def pendulum_grid(xs, ys):
    """Angle (0 = upright) × angular velocity, as the observation (cos θ, sin θ, θ̇)."""
    th, om = np.meshgrid(xs, ys)
    return np.stack([np.cos(th), np.sin(th), om], axis=-1).reshape(-1, 3)


WORLDS = {
    "cartpole": {
        "gym": "CartPole-v1", "grid": cartpole_grid, "x": ("angle", -0.21, 0.21), "y": ("angular velocity", -2.0, 2.0),
        "state": lambda o: o, "ranges": [(-2.4, 2.4), (-3.0, 3.0), (-0.21, 0.21), (-3.5, 3.5)],
    },
    "mountain-car": {
        "gym": "MountainCar-v0", "grid": mountaincar_grid, "x": ("position", -1.2, 0.5), "y": ("speed", -0.07, 0.07),
        "state": lambda o: o, "ranges": [(-1.2, 0.6), (-0.07, 0.07)],
    },
    "pendulum": {
        "gym": "Pendulum-v1", "grid": pendulum_grid, "x": ("angle", -np.pi, np.pi), "y": ("angular velocity", -8.0, 8.0),
        "state": lambda o: np.array([np.arctan2(o[1], o[0]), o[2]]), "ranges": [(-np.pi, np.pi), (-8.0, 8.0)],
    },
}
GRID = 31  # points per side of a snapshot's map


# ---- the catalog: what to record ----
# DQN on CartPole: tuned until most seeds balance for the full 500 steps. A squared loss, not Huber's: CartPole's values
# run up to 1/(1 − γ) = 100, and clipped errors learn them too slowly. Each comparison of Part 9 changes one thing.
DQN_CARTPOLE = dict(lr=5e-4, buffer=10_000, batch=128, train_every=4, target_every=500, eps=(1.0, 0.05, 20_000), huber=False)
SEEDS = list(range(1, 21))  # 20 seeds per recording: enough to tell a setting that works 9 times in 10 from one that works 6
CARTPOLE = dict(world="cartpole", learner="dqn", steps=200_000, block=5_000, seeds=SEEDS)
CATALOG = {
    "dqn-cartpole": dict(CARTPOLE, cfg=DQN_CARTPOLE, station="dqn", title="DQN"),
    "dqn-cartpole-no-replay": dict(CARTPOLE, cfg=dict(DQN_CARTPOLE, buffer=128), station="dqn", title="DQN without replay"),
    "dqn-cartpole-no-target": dict(CARTPOLE, cfg=dict(DQN_CARTPOLE, target_every=0), station="dqn", title="DQN without a target network"),
    "double-dqn-cartpole": dict(CARTPOLE, cfg=dict(DQN_CARTPOLE, double=True), station="dqn-extensions", title="Double DQN"),
    "dueling-dqn-cartpole": dict(CARTPOLE, cfg=dict(DQN_CARTPOLE, dueling=True), station="dqn-extensions", title="Dueling DQN"),
    "prioritized-dqn-cartpole": dict(CARTPOLE, cfg=dict(DQN_CARTPOLE, prioritized=True), station="dqn-extensions", title="Prioritized replay"),
    # Policy gradients with networks, the deep versions of Part 10's methods
    "a2c-cartpole": dict(world="cartpole", learner="pg", station="a2c", title="A2C", steps=120_000, block=3_000, seeds=SEEDS,
                         cfg=dict(algo="a2c", workers=8, steps=5, lr=1e-3, lam=1.0, v_epochs=1, minibatch=40, lr_v=1e-3)),
    "ppo-cartpole": dict(world="cartpole", learner="pg", station="ppo", title="PPO", steps=100_000, block=2_500, seeds=SEEDS,
                         cfg=dict(algo="ppo", workers=8, steps=128, epochs=10, minibatch=64, lr=3e-4, clip=0.2, lam=0.95)),
    "trpo-cartpole": dict(world="cartpole", learner="pg", station="trpo", title="TRPO", steps=100_000, block=2_500, seeds=SEEDS,
                          cfg=dict(algo="trpo", workers=8, steps=256, delta=0.01, lam=0.95, v_epochs=10, minibatch=64)),
    "ppo-pendulum": dict(world="pendulum", learner="pg", station="ppo", title="PPO", steps=200_000, block=5_000, seeds=SEEDS,
                         cfg=dict(algo="ppo", workers=4, steps=512, epochs=10, minibatch=64, lr=1e-3, gamma=0.9, lam=0.95, reward_scale=0.1)),
    # Off-policy actor-critics for continuous actions (Part 11)
    "ddpg-pendulum": dict(world="pendulum", learner="ac", station="ddpg", title="DDPG", steps=60_000, block=1_500, seeds=SEEDS, sweep_seeds=SEEDS[:10],
                          cfg=dict(algo="ddpg", reward_scale=0.1)),
    "td3-pendulum": dict(world="pendulum", learner="ac", station="td3", title="TD3", steps=60_000, block=1_500, seeds=SEEDS, sweep_seeds=SEEDS[:10],
                         cfg=dict(algo="td3", reward_scale=0.1)),
    "sac-pendulum": dict(world="pendulum", learner="ac", station="sac", title="SAC", steps=60_000, block=1_500, seeds=SEEDS, sweep_seeds=SEEDS[:10],
                         cfg=dict(algo="sac", reward_scale=0.1)),
}

# The knobs each recording's sweep tries, with the recording's own value among them. target_every = 0: no target
# network; buffer = 128 (the batch size): no replay, each batch is the latest experience. DQN's memory is 10,000 steps:
# over 20 seeds, 16 end well with it, 11 with 100,000, and every run with 10,000 ends at 400 steps or more.
SWEEPS = {
    "dqn-cartpole": {"lr": [1e-4, 2.5e-4, 5e-4, 1e-3, 2.5e-3], "target_every": [0, 100, 500, 2000], "buffer": [128, 1_000, 10_000, 100_000],
                     "huber": [0, 1]},
    "a2c-cartpole": {"lr": [3e-4, 1e-3, 3e-3, 1e-2], "steps": [1, 5, 20]},
    "ppo-cartpole": {"clip": [0.0, 0.1, 0.2, 0.3, 0.5], "epochs": [1, 4, 10, 30], "lr": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]},
    "trpo-cartpole": {"delta": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]},
    "ppo-pendulum": {"lr": [3e-4, 1e-3, 3e-3], "gamma": [0.9, 0.95, 0.99], "clip": [0.0, 0.1, 0.2, 0.3, 0.5], "epochs": [1, 4, 10, 30]},
    "ddpg-pendulum": {"tau": [0.001, 0.005, 0.02, 0.1], "noise": [0.0, 0.1, 0.3, 0.6], "reward_scale": [0.01, 0.1, 1.0]},
    "td3-pendulum": {"delay": [1, 2, 4, 8], "policy_noise": [0.0, 0.2, 0.5, 1.0]},
    "sac-pendulum": {"alpha": ["auto", 0.01, 0.05, 0.2, 1.0], "tau": [0.001, 0.005, 0.02, 0.1]},
}
# Sweep values that set more than one knob: SAC's α is either tuned ("auto") or fixed.
SWEEP_AS = {"alpha": lambda v: {"auto_alpha": True} if v == "auto" else {"alpha": v, "auto_alpha": False}}


# ---- packing numbers: quantized to bytes, base64 ----
def pack(values, lo: float | None = None, hi: float | None = None, bits: int = 8) -> dict:
    v = np.asarray(values, dtype=float).ravel()
    lo = float(v.min() if lo is None else lo)
    hi = float(v.max() if hi is None else hi)
    if hi - lo < 1e-9:
        hi = lo + 1.0
    top = 2**bits - 1
    q = np.round((np.clip(v, lo, hi) - lo) / (hi - lo) * top).astype(np.uint8 if bits == 8 else "<u2")
    return {"lo": round(lo, 6), "hi": round(hi, 6), "bits": bits, "data": base64.b64encode(q.tobytes()).decode()}


def rounded(d: dict, digits: int = 4) -> dict:
    return {k: (round(float(v), digits) if isinstance(v, (float, np.floating)) else v) for k, v in d.items()}


# ---- what a snapshot holds ----
def grid_axes(world: dict):
    return np.linspace(world["x"][1], world["x"][2], GRID), np.linspace(world["y"][1], world["y"][2], GRID)


def test_episode(world: dict, choose, seed: int, rng: np.random.Generator) -> tuple[dict, float]:
    """One episode from the test start, with choose(obs) → (action, numbers to keep for this step)."""
    env = gym.make(world["gym"])
    obs, _ = env.reset(seed=seed)
    states, actions, extra, ret, done = [], [], [], 0.0, False
    while not done:
        a, numbers = choose(obs, rng)
        states.append(world["state"](obs))
        actions.append(a)
        extra.append(numbers)
        obs, r, term, trunc, _ = env.step(a)
        ret += r
        done = term or trunc
    states.append(world["state"](obs))
    S = np.array(states)
    ep = {
        "steps": len(actions), "return": round(ret, 2),
        # 8 bits are plenty to draw a state or a bar: CartPole's angle to a tenth of a degree, its values to 0.2
        "s": [pack(S[:, j], lo, hi) for j, (lo, hi) in enumerate(world["ranges"])],
        "a": pack(np.array(actions, dtype=float).ravel()),
        # per step: the action values (DQN), or the probability and the value (PG); each column on its own scale, so a
        # probability keeps its precision next to a value of 90
        "x": [pack(col) for col in np.array(extra, dtype=float).T],
    }
    return ep, ret


def test_dqn(agent: DQN, world: dict, rng: np.random.Generator) -> dict:
    return test_episode(world, lambda o, r: (agent.act(o, greedy=True), agent.q(o[None])[0]), TEST_SEED, rng)[0]


def snapshot_dqn(agent: DQN, world: dict, rng: np.random.Generator) -> dict:
    xs, ys = grid_axes(world)
    q = agent.q(world["grid"](xs, ys))
    return {"v": pack(q.max(axis=1)), "act": pack(q.argmax(axis=1), 0, agent.n - 1), "test": test_dqn(agent, world, rng), "eps": round(agent.epsilon(), 4)}


# ---- training ----
def run_dqn(spec: dict, seed: int, snapshots: bool) -> dict:
    world = WORLDS[spec["world"]]
    cfg = DQNConfig(**spec["cfg"])
    env = gym.make(world["gym"])
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    agent = DQN(env.observation_space.shape[0], env.action_space.n, cfg, rng, spec["steps"])
    block, blocks = spec["block"], spec["steps"] // spec["block"]
    per_block, qs, tests = [[] for _ in range(blocks)], [], []
    shots = [snapshot_dqn(agent, world, np.random.default_rng(seed + 10_000))] if snapshots else []
    ret = 0.0
    for t in range(spec["steps"]):
        a = agent.act(obs)
        obs2, r, term, trunc, _ = env.step(a)
        agent.observe(obs, a, r, obs2, term)
        obs, ret = obs2, ret + r
        if term or trunc:
            per_block[t // block].append(ret)
            ret = 0.0
            obs, _ = env.reset(seed=int(rng.integers(1 << 30)))
        if (t + 1) % block == 0:
            k = (t + 1) // block
            qs.append(round(float(np.mean(agent.log["q"])), 2) if agent.log["q"] else None)  # every seed: overestimation shows here
            if snapshots:
                shot = snapshot_dqn(agent, world, np.random.default_rng(seed + 10_000 + k))
                L = agent.log
                shot["stats"] = rounded({"loss": np.mean(L["loss"]) if L["loss"] else 0.0, "td": np.mean(L["td"]) if L["td"] else 0.0,
                                         "q": np.mean(L["q"]) if L["q"] else 0.0})
                shots.append(shot)
            # every seed plays the shown seed's test episode: the greedy policy, judged apart from its exploration
            tests.append(shots[-1]["test"]["return"] if snapshots else test_dqn(agent, world, np.random.default_rng(seed + 10_000 + k))["return"])
            agent.log = {"loss": [], "td": [], "q": []}
    return {"seed": seed, "train": [round(float(np.mean(b)), 2) if b else None for b in per_block],
            "episodes": [len(b) for b in per_block], "q": qs, "test": tests, "snapshots": shots}


def test_pg(agent: PG, world: dict, rng: np.random.Generator) -> dict:
    """A test episode with actions drawn from the policy, keeping per step its probabilities (or its mean and spread)
    and the critic's value."""
    pi = agent.pi

    def choose(o, r):
        a = pi.sample(o[None], r)[0]
        val = float(agent.value(o[None])[0])
        if pi.discrete:  # the probability of the last action is enough with two (the other is its complement)
            return int(a), [float(pi.dist(o[None])[0, -1]), val]
        mu, sd = pi.dist(o[None])
        return pi.action_for_env(a), [float(mu[0, 0]), float(sd[0]), val]

    return test_episode(world, choose, TEST_SEED, rng)[0]


def snapshot_pg(agent: PG, world: dict, rng: np.random.Generator) -> dict:
    """The critic's values over the grid, the policy's choice there (P(second action) for a softmax, the mean action for
    a Gaussian), and a test episode."""
    xs, ys = grid_axes(world)
    g = world["grid"](xs, ys)
    v, pi = agent.value(g), agent.pi
    ep = test_pg(agent, world, rng)
    choice = pi.dist(g)[:, -1] if pi.discrete else pi.dist(g)[0][:, 0]
    return {"v": pack(v), "mean": pack(choice), "test": ep}


def run_pg(spec: dict, seed: int, snapshots: bool) -> dict:
    world = WORLDS[spec["world"]]
    cfg = PGConfig(**spec["cfg"])
    rng = np.random.default_rng(seed)
    envs = [gym.make(world["gym"]) for _ in range(cfg.workers)]
    obs = np.stack([e.reset(seed=seed * 100 + k)[0] for k, e in enumerate(envs)])
    agent = PG(obs.shape[1], envs[0].action_space, cfg, rng)
    N, T, block, blocks = cfg.workers, cfg.steps, spec["block"], spec["steps"] // spec["block"]
    per_block, tests = [[] for _ in range(blocks)], []
    stats: list[dict] = []
    shots = [snapshot_pg(agent, world, np.random.default_rng(seed + 10_000))] if snapshots else []
    ep, t = np.zeros(N), 0
    while t < spec["steps"]:
        S = np.zeros((T, N, obs.shape[1]))
        A, R, V, VN, D, C = [], np.zeros((T, N)), np.zeros((T, N)), np.zeros((T, N)), np.zeros((T, N)), np.zeros((T, N))
        for k in range(T):
            a = agent.pi.sample(obs, rng)
            S[k], V[k] = obs, agent.value(obs)
            A.append(a)
            nxt = np.zeros_like(obs)
            for i, e in enumerate(envs):
                o, r, term, trunc, _ = e.step(int(a[i]) if agent.pi.discrete else agent.pi.action_for_env(a[i]))
                R[k, i], D[k, i], C[k, i], nxt[i] = r * cfg.reward_scale, term, trunc, o
                ep[i] += r
                if term or trunc:
                    if t // block < blocks:
                        per_block[t // block].append(ep[i])
                    ep[i] = 0.0
            VN[k] = agent.value(nxt)  # the value of the state each step led to, before any reset: for cut episodes
            for i, e in enumerate(envs):
                if D[k, i] or C[k, i]:
                    nxt[i] = e.reset(seed=int(rng.integers(1 << 30)))[0]
            obs = nxt
            t += N
            if t % block < N and t // block <= blocks and snapshots and len(shots) <= t // block:
                shot = snapshot_pg(agent, world, np.random.default_rng(seed + 10_000 + t // block))
                shot["stats"] = rounded({k2: float(np.mean([s_[k2] for s_ in stats])) for k2 in (stats[0] if stats else {})})
                shots.append(shot)
                stats = []
            if t % block < N and t // block <= blocks and len(tests) < t // block:  # every seed: the same test episode
                tests.append(shots[-1]["test"]["return"] if snapshots else test_pg(agent, world, np.random.default_rng(seed + 10_000 + t // block))["return"])
        adv, ret = gae(R, V, VN, D, C, cfg.gamma, cfg.lam)
        A = np.array(A)
        agent.update(S.reshape(T * N, -1), A.reshape(T * N, *A.shape[2:]), adv.reshape(-1), ret.reshape(-1))
        stats.append(dict(agent.stats))
    return {"seed": seed, "train": [round(float(np.mean(b)), 2) if b else None for b in per_block],
            "episodes": [len(b) for b in per_block], "test": tests, "snapshots": shots}


def test_ac(agent: AC, world: dict, rng: np.random.Generator) -> dict:
    """A test episode played greedily, keeping per step the action, the spread the policy explores with, and the value."""
    def choose(o, r):
        a = agent.policy(o[None], greedy=True)[0]
        return a, [float(a[0]), float(agent.spread(o[None])[0]), float(agent.value(o[None])[0])]

    return test_episode(world, choose, TEST_SEED, rng)[0]


def snapshot_ac(agent: AC, world: dict, rng: np.random.Generator) -> dict:
    """What the critic thinks of the grid (Q of the greedy action), the actor's greedy action there, and a test episode."""
    xs, ys = grid_axes(world)
    g = world["grid"](xs, ys)
    return {"v": pack(agent.value(g)), "mean": pack(agent.policy(g, greedy=True)[:, 0]), "test": test_ac(agent, world, rng)}


def run_ac(spec: dict, seed: int, snapshots: bool) -> dict:
    world = WORLDS[spec["world"]]
    cfg = ACConfig(**spec["cfg"])
    env = gym.make(world["gym"])
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    agent = AC(env.observation_space.shape[0], env.action_space, cfg, rng, spec["steps"])
    block, blocks = spec["block"], spec["steps"] // spec["block"]
    per_block, qs, tests = [[] for _ in range(blocks)], [], []
    shots = [snapshot_ac(agent, world, np.random.default_rng(seed + 10_000))] if snapshots else []
    ret = 0.0
    for t in range(spec["steps"]):
        a = agent.act(obs)
        obs2, r, term, trunc, _ = env.step(a)
        agent.observe(obs, a, r, obs2, term)
        obs, ret = obs2, ret + r
        if term or trunc:
            per_block[t // block].append(ret)
            ret = 0.0
            obs, _ = env.reset(seed=int(rng.integers(1 << 30)))
        if (t + 1) % block == 0:
            L = agent.log
            qs.append(round(float(np.mean(L["q"])) / cfg.reward_scale, 2) if L["q"] else None)  # in the reward's own units
            if snapshots:
                shot = snapshot_ac(agent, world, np.random.default_rng(seed + 10_000 + (t + 1) // block))
                shot["stats"] = rounded({"q": qs[-1] or 0.0, "loss": float(np.mean(L["loss"])) if L["loss"] else 0.0,
                                         **({"alpha": float(L["alpha"][-1])} if L["alpha"] else {})})
                shots.append(shot)
            tests.append(shots[-1]["test"]["return"] if snapshots else test_ac(agent, world, np.random.default_rng(seed + 10_000 + (t + 1) // block))["return"])
            agent.log = {"q": [], "loss": [], "alpha": []}
    return {"seed": seed, "train": [round(float(np.mean(b)), 2) if b else None for b in per_block],
            "episodes": [len(b) for b in per_block], "q": qs, "test": tests, "snapshots": shots}


RUNNERS = {"dqn": run_dqn, "pg": run_pg, "ac": run_ac}


def record(name: str, spec: dict) -> None:
    t0 = time.time()
    seeds, shown = spec["seeds"], spec.get("shown", spec["seeds"][0])
    with ProcessPoolExecutor(max_workers=min(4, len(seeds))) as pool:
        results = list(pool.map(RUNNERS[spec["learner"]], [spec] * len(seeds), seeds, [s == shown for s in seeds]))
    world = WORLDS[spec["world"]]
    main = next(r for r in results if r["seed"] == shown)
    out = {
        "name": name, "title": spec["title"], "station": spec["station"], "world": spec["world"], "learner": spec["learner"],
        "config": {k: v for k, v in spec["cfg"].items()},
        "steps": spec["steps"], "block": spec["block"], "seeds": seeds, "shown": shown,
        "grid": {"x": [world["x"][0], world["x"][1], world["x"][2], GRID], "y": [world["y"][0], world["y"][1], world["y"][2], GRID]},
        "curves": [{"seed": r["seed"], "train": r["train"], "test": r["test"], "episodes": r["episodes"], **({"q": r["q"]} if "q" in r else {})} for r in results],
        "snapshots": main["snapshots"],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.json"
    path.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    tests = [s["test"]["return"] for s in main["snapshots"]]
    log.info("%s: %.0f s, %d KB; test returns %s", name, time.time() - t0, path.stat().st_size // 1024, " ".join(f"{x:.0f}" for x in tests))


def current(spec: dict, knob: str):
    """The recording's own value of a knob: its setting, or the learner's default."""
    cfg = {"dqn": DQNConfig, "pg": PGConfig, "ac": ACConfig}[spec["learner"]](**spec["cfg"])
    if knob == "alpha" and getattr(cfg, "auto_alpha", False):
        return "auto"
    v = getattr(cfg, knob)
    return list(v) if isinstance(v, tuple) else int(v) if isinstance(v, bool) else v  # a switch is swept as 0 and 1


def sweep(name: str, knobs: dict, merge: bool = False) -> None:
    """Every value of every knob, every seed, all at once over the processes; the curves only. With merge, the knobs
    join those already in the sweep's file instead of replacing it."""
    t0, spec = time.time(), CATALOG[name]
    seeds = spec.get("sweep_seeds", spec["seeds"])  # each run of an actor-critic takes minutes: fewer seeds per value
    jobs = [(knob, v, s) for knob, values in knobs.items() for v in values for s in seeds]
    specs = [dict(spec, cfg=dict(spec["cfg"], **(SWEEP_AS[knob](v) if knob in SWEEP_AS else {knob: v}))) for knob, v, _ in jobs]
    with ProcessPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(RUNNERS[spec["learner"]], specs, [s for *_, s in jobs], [False] * len(jobs)))
    train = {job: r["train"] for job, r in zip(jobs, results)}
    test = {job: r["test"] for job, r in zip(jobs, results)}
    out = {"name": name, "world": spec["world"], "steps": spec["steps"], "block": spec["block"], "seeds": seeds,
           "config": spec["cfg"], "knobs": {knob: {"values": values, "current": current(spec, knob),
                                                   "train": [[train[(knob, v, s)] for s in seeds] for v in values],
                                                   "test": [[test[(knob, v, s)] for s in seeds] for v in values]}
                                           for knob, values in knobs.items()}}
    path = OUT / "sweeps" / f"{name}.json"
    if merge and path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        out["knobs"] = {**old["knobs"], **out["knobs"]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    for knob, values in knobs.items():  # the last tenth of training, averaged, per value: a first look at the odds
        ends = [[np.mean([x for x in c[-max(1, len(c) // 10):] if x is not None] or [np.nan]) for c in row] for row in out["knobs"][knob]["train"]]
        log.info("%s %s: %s", name, knob, " · ".join(f"{v}: {' '.join(f'{e:.0f}' for e in row)}" for v, row in zip(values, ends)))
    log.info("%s: %d runs, %.0f s, %d KB", name, len(jobs), time.time() - t0, path.stat().st_size // 1024)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    parser = argparse.ArgumentParser(description="Record training runs for RL Atlas.")
    parser.add_argument("names", nargs="*", help="recordings (or sweeps) to make (default: all)")
    parser.add_argument("--sweep", action="store_true", help="make the sweeps instead of the recordings")
    parser.add_argument("--knobs", help="with --sweep: only these knobs (comma-separated), merged into the sweep's file")
    args = parser.parse_args()
    known = SWEEPS if args.sweep else CATALOG
    unknown = [n for n in args.names if n not in known]
    if unknown:
        log.error("unknown %s: %s (known: %s)", "sweeps" if args.sweep else "recordings", ", ".join(unknown), ", ".join(known))
        return 1
    for name in args.names or known:
        if not args.sweep:
            record(name, CATALOG[name])
        elif args.knobs:
            sweep(name, {k: SWEEPS[name][k] for k in args.knobs.split(",")}, merge=True)
        else:
            sweep(name, SWEEPS[name])
    return 0


if __name__ == "__main__":
    sys.exit(main())
