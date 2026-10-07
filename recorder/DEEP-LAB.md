# The deep Lab: knobs you can turn on runs nobody can train live

A plan to decide on, not yet built. The live Lab can now separate the settings from luck: a bench of 20 seeds, the
typical run, any seed on demand. The deep presets (CartPole and Pendulum) can't do any of that yet. Each one plays one
recorded seed, and only a few knobs were swept, one at a time. This note weighs the options at each decision point and
recommends one.

## Status: option D, being built

The spike settled the open questions, and the build follows option D below.

- **It learns.** The JavaScript trainer (`lab/deep/`) learns CartPole like the NumPy recorder. On 20 seeds of
  `dqn-cartpole`, 15 end with a 500-step test (NumPy 16) and 13 end well by the Lab's rule (NumPy 16). On 20 more
  seeds NumPy itself scored 13: the gap is luck.
- **It is exact.** The same seed gives the same run, bit for bit, in Node and in a browser's Worker: seed 2 trained in
  Chromium ended where the offline recording did. The trainer's math (`lab/deep/dmath.js`) uses only + − × ÷ and the
  square root, which every engine rounds alike.
- **It is fast enough.** One 200k-step DQN seed takes about 75 s on one core. The network's loops work on two rows
  and four units at a time, 1.8 times faster than the first version. A bench of 20 seeds takes 5–10 minutes on a
  laptop's spare cores: offered on request, never automatic.

What runs where:
- `recorder/deep.js` trains recordings and sweeps in Node, one thread per core, resumable (`recorder/.cache`).
- The page (`lab/recorded.js`, `lab/lab.js`) treats a recording made by this trainer as a starting point:
  - its knobs (those of its sweep) and its seed can be changed;
  - a seed of the recording plays at once, and any other seed or setting trains in a Worker, with its progress shown;
  - the odds come from the recording (its settings), the sweep (one knob turned), or a bench trained here on request.
- Network shape: depth (1–3), width (16–128) and activation are knobs of the DQN sweep.

Done so far: the DQN family on CartPole. Still to port: A2C, PPO, TRPO, DDPG, TD3, SAC and Pendulum's recordings;
the network view.

## What we have, and what it costs

- **The recorder** (`recorder/`) trains its own NumPy networks on Gymnasium's CartPole and Pendulum. Every network is
  the same: two hidden layers of 64 units.
  - DQN, 200k steps: ~1.1 min per seed.
  - PPO and A2C: 0.2–0.5 min per seed.
  - DDPG, TD3 and SAC on Pendulum, 60k steps: 1.6–2.6 min per seed.
- **The recordings** take 4 MB of the 7.5 MB page. Each recording stores the curves of all 20 seeds, plus one seed's
  snapshots: a value map and a test episode after every block.
- **Training in the page is possible, but slow.** A plain-JavaScript benchmark of the same 64 × 64 network (training
  step plus Adam), on one core of this container:

  | Network | Batch | Updates per second |
  |---|---|---|
  | 4 → 32 → 32 → 2 | 32 | ~5,100 |
  | 4 → 64 → 64 → 2 | 32 | ~1,700 |
  | 4 → 64 → 64 → 2 | 64 | ~850 |
  | 4 → 64 → 64 → 64 → 2 | 64 | ~390 |
  | 4 → 128 → 128 → 2 | 64 | ~235 |
  | critic 4 → 64 → 64 → 1 | 256 | ~245 |

  That makes one DQN seed on CartPole take about 4 minutes (one update per step), or about 1 minute when it trains
  every 4 steps, as most DQN setups do. One SAC seed on Pendulum would take 5–15 minutes. The code was not optimized;
  twice as fast is realistic, ten times is not.
  - **One run on demand:** feasible. Watching a network learn for a minute is a lesson in itself.
  - **A bench of 20 seeds:** 5–20 minutes even on 4 cores, so it cannot be live.

## Decision 1 · Where the runs come from

| Option | Freedom | Speed | Verdict |
|---|---|---|---|
| A. Everything precomputed, on a lattice of settings | only the lattice | instant | honest, but closed |
| B. Everything trained in the page | total | minutes per seed, a bench in tens of minutes | too slow for odds |
| C. Interpolate between precomputed settings | looks total | instant | **no**: it would show runs that never happened |
| **D. Both, from one trainer** | total | instant on the lattice, about 1 minute per seed elsewhere | **recommended** |

**D in detail.**
- The trainer is rewritten in JavaScript, and the same code runs in two places:
  - offline in Node, to precompute the lattice;
  - in the page, in a Web Worker, for anything off it.
- Same code, same seed, same settings: the same run, bit for bit. So a precomputed point *is* the bench for those
  settings. It works the same as the live bench's cache, and any run on it can be replayed live to check it.
- Off the lattice, a different value or a seed outside 1–20 trains live, with its curve growing as it goes ("training
  here, about a minute"). Its odds stay unknown until a bench is run, and the page says so.
- **Cost:** the NumPy recorder is retired, and every recording and sweep is remade with the JavaScript trainer.
- **Determinism:** `Math.exp`, `Math.tanh` and `Math.log` are not guaranteed to round the same way in every browser.
  The trainer must use its own versions of them. The same issue already touches the live presets' softmax policies,
  and the fix covers both.

## Decision 2 · The network

The network should be worth turning knobs on, yet small enough to train in the page.

| Option | Parameters | Verdict |
|---|---|---|
| Today's 2 × 64, fixed | ~4.5k | no model knobs at all |
| SB3 RL Zoo's tuned CartPole DQN: 2 × 256, lr 2.3e-3, 128 gradient steps every 256 | ~67k | too heavy to train live |
| **One plain MLP whose shape is a knob** | 0.2k–17k | **recommended** |

**Recommended network:** input → 1 to 3 hidden layers → output. The same family serves Q-networks, actors and critics.

| Knob | Values | Default | What it teaches |
|---|---|---|---|
| Depth (hidden layers) | 1, 2, 3 | 2 | Deeper isn't automatically better on small problems. |
| Width (units per layer) | 16, 32, 64, 128 | 64 | Too narrow can't represent the values; wide is slower and needs a smaller step. |
| Activation | ReLU, tanh | ReLU (tanh for policy gradients, as now) | Dead units against saturation. |
| Step size (Adam) | 4 to 5 values on a log scale | as now | The knob that interacts with all the others. |

- Widths stop at 128, so a live run stays near a minute; the largest network, 3 × 128, has about 34k parameters.
- **A view of the network**, new to the Lab: layers as columns of units, colored by their activation on the current
  state, with edges shaded by weight (sampled when the layer is wide). The depth and width knobs reshape it, so the
  model is something you can see.
- Possible later additions: input normalization and layer normalization. Both are real issues in deep RL, but each one
  doubles the lattice.

## Decision 3 · Which combinations get precomputed

A full grid explodes. For DQN on CartPole: 5 step sizes × 4 target periods × 4 buffers × 4 widths × 3 depths gives 960
settings. At 20 seeds and a minute each, that is about 320 CPU-hours.

| Option | Points (DQN example) | Captures interactions | Verdict |
|---|---|---|---|
| Full grid | ~960 | all | unaffordable |
| One knob at a time, around a base setting (the "star"; what the sweeps do now) | ~25 | none | cheap, misses the lesson that a good step size depends on the network |
| Random combinations, the reader snapped to the nearest one | any | blurred | confusing: the setting shown is not the one chosen |
| **The star, plus two-knob planes where the interaction is the lesson** | ~45–60 | the ones that teach | **recommended** |

- **Planes for DQN:** step size × width, and target period × step size.
- **Planes for PPO:** clip × epochs, and step size × width.
- **Planes for SAC and TD3:** to be chosen once the world question (Decision 5) is settled.
- Everything else is covered by live training (Decision 1).
- The knob ladders in the Lab show which values are precomputed with a small mark. Any other combination trains live.

**Budget for CartPole:**
- DQN: about 50 points × 20 seeds × ~1 minute is about 17 CPU-hours, or about 4 hours on this container's 4 cores.
- Double, dueling and prioritized DQN, A2C, PPO and TRPO: their own stars, smaller.
- Together: about 8–10 hours, run once in the background.

## Decision 4 · What gets stored

All seeds' curves for every point fit easily: 40 blocks × 2 curves × 20 seeds × ~150 points is about 1 MB.

Snapshots don't fit: a full set of value maps and test episodes per point would add 15–30 MB.

| Option | Verdict |
|---|---|
| Snapshots for every point | too big for one offline file |
| Store network weights instead of maps | bigger still |
| **Snapshots only for the base point and the planes' typical seeds, quantized to 8 bits; other points store curves only, and their typical run is replayed live when someone plays it** | **recommended** |

The live replay is exact (Decision 1), takes about a minute, and is kept for the rest of the visit.

## Decision 5 · The worlds

The runs review found that Pendulum at 60k steps is too easy: SAC's α, TD3's delay and DDPG's noise all end within a
few points of each other.

| Option | Verdict |
|---|---|
| Keep Pendulum and judge its knobs by speed and stability, not the final return | cheap, honest; **recommended for now** |
| A harder continuous world: continuous Lunar Lander (Box2D physics) or Hopper (MuJoCo) | physics too heavy to port to the page; rules out live training |
| Continuous Mountain Car | sparse reward: most methods fail, which teaches exploration more than the knobs |
| Add Acrobot (discrete, harder than CartPole, simple physics) | a good later addition for the DQN family |

CartPole and Pendulum physics are a few lines each, and get ported exactly from Gymnasium's equations.

## Decision 6 · How the Lab presents it

- **The same frame as the live presets:**
  - the odds, the band and the seed strip;
  - the typical seed by default;
  - type a seed or roll the dice.
  - On the lattice, all of that is instant. Off it, the run trains live, and the bench can be started on request.
- **Play bar:** for a run trained live, it follows the training as it streams in.
- **Speed ladder:** a block (5,000 steps) is the unit. "Step by step" plays the test episode, the network view showing
  each decision.
- **Stories:** they re-pick their shown seed from the precomputed bench (the typical one, or the one that shows the
  idea; the `tools/story-runs.js` check applies). They then get the same fingerprint test.

## Recommended path, in phases

1. **The trainer in JavaScript.**
   - Contents: the networks with deterministic math, Adam, replay, the DQN family, A2C, PPO and TRPO, DDPG, TD3 and
     SAC, and the CartPole and Pendulum physics.
   - Tested against the NumPy recorder statistically (its curves fall in the same band), not bit for bit.
2. **Training in the page.** A Web Worker that trains, streams blocks and snapshots, and measures its own speed. "Train
   it here" for any setting or seed.
3. **The lattice runner.** Node, one process per core, resumable. The file format: curves per point; snapshots for
   chosen points. CartPole first.
4. **The Lab.**
   - knob ladders with the precomputed values marked;
   - the network knobs and the network view;
   - the bench read from the lattice, falling back to live training.
5. **Pendulum, the stories, and cleanup.**
   - Pendulum's actor–critics, once the budget is set;
   - stories re-pinned;
   - the NumPy recorder retired, or kept only as a cross-check.

## Questions to settle before building

1. Replace the NumPy recorder with a JavaScript trainer, remaking every recording? (Needed for exact live replay.)
2. The network knobs: depth 1–3, width 16–128, ReLU or tanh. Too much, too little?
3. The offline compute budget: about 10 hours for CartPole is fine? Pendulum's actor–critics cost several times more.
   They could use 10 seeds, or a shorter run.
4. Pendulum: keep it and judge its knobs by speed and stability, or look for another continuous world?
