# Runs review: what is costly, what is weak

A profile of every run the atlas shows. It covers the 13 recordings and 8 sweeps trained offline
(`recorder/`), and the 56 Lab presets the browser computes. Nothing was retrained. The recordings and sweeps
were read from their files. The presets were replayed over 20 seeds each, which is the count the Lab itself uses for
its odds. "Shown run" means the run a story or the Lab plays back.

## 1. What is costly to run

**Offline recordings** run on this 4-core container, one process per seed. The times come from the sweep logs.

| World and algorithm | CPU per seed | Recording (20 seeds) | Sweep |
|---|---|---|---|
| SAC, Pendulum, 60k steps | ~2.6 min | ~13 min | 90 runs, 59 min |
| DDPG, Pendulum, 60k steps | ~1.7 min | ~8.5 min | 110 runs, 46 min |
| TD3, Pendulum, 60k steps | ~1.6 min | ~8 min | 80 runs, 33 min |
| DQN family, CartPole, 200k steps | ~1.1 min | ~6 min each (6 recordings: ~34 min) | 40 runs, 11 min |
| PPO, CartPole, 100k steps | ~0.5 min | ~2.5 min | 300 runs, 38 min |
| A2C, CartPole, 120k steps | ~0.2 min | ~1 min | 140 runs, 8 min |

- The expensive part is the actor–critics on Pendulum, and the sweeps above all, because they multiply values by seeds.
- Re-recording everything would take about 1.5 hours. Re-running every sweep would take about 4 hours.
- **Every recording shows seed 1.** The shown seed was never chosen: `record.py` defaults `shown` to the first seed.

**In-browser presets are cheap.** Every run takes under 1.1 s; most take under 0.1 s.

| Run | Time per run |
|---|---|
| Blackjack (100k episodes) | ~1 s |
| TRPO/GAE runs that never reach the goal (they hit the 1,000-step cap) | ~0.5 s |
| Actor–critic on the cliff | ~0.3 s |
| Mountain Car | 0.06–0.23 s |

Mountain Car is not costly, and its preset seed is typical: 20 of 20 seeds get out in under 150 steps.

## 2. Recordings: how good is the shown run?

"Solved" means a test episode reaches 475 on CartPole, or −200 on Pendulum.

| Recording | Used by | Shown seed (seed 1) | Verdict |
|---|---|---|---|
| **dqn-cartpole** | 6 stories (DQN, neural networks, experience replay, target network, learning curves, DQN extensions) | Solves at block 30 (150k steps); the median seed solves at block 18. It sits on a long plateau around 200–250 for blocks 3–28. | **Top candidate.** It is the most-seen run, and it is a slow seed. |
| **double-dqn-cartpole** | DQN extensions | Solves at block 22 (median 18), then falls back to 188 and 159 before settling at block 32. | Candidate |
| **dueling-dqn-cartpole** | DQN extensions | Solves at block 25 (median 16) | Candidate, minor |
| prioritized-dqn-cartpole | DQN extensions | Solves at block 18 (median 14) | Fine |
| a2c-cartpole | Lab only | Solves at block 20 (median 12), and wobbles after | Low priority |
| ppo-pendulum | Lab only | Test episodes collapse at blocks 9 (−1115), 30 and 35 (about −265) | Medium; Lab only |
| ddpg-pendulum | DDPG, TD3, SAC stories | Worst final of 20 seeds (−134 against a median of −125), and one bad test episode (−383) at block 34 | Low: the gaps are invisible on Pendulum |
| td3, sac, ppo-cartpole, trpo-cartpole | stories and Lab | At or near the median | Fine |
| dqn-no-replay, dqn-no-target | DQN, replay and target stories | Bad on purpose. The no-replay drop at the last block (500 → 234) is that story's point. | Fine |

## 3. Sweeps: does the knob show its effect?

- **They teach well:**
  - DQN: the target period, Huber loss and buffer size.
  - A2C: the learning rate.
  - PPO on Pendulum: clip and epochs. No clip fails on 17 of 20 seeds.
  - TRPO: δ, which moves the speed by a factor of 6.
- **They are flat, because Pendulum at 60k steps is too easy.** Every value ends within a few points of the others.
  - SAC α: all five values end between −123 and −124. Only α = 1 learns slower.
  - TD3 policy noise: no effect at all, even on speed.
  - TD3 delay: it only slows learning (6k steps to solve at delay 1, 19k at delay 8).
  - DDPG noise and reward scale: the final return is flat; the knob only changes speed.
- **PPO on CartPole:** every clip, epoch count and learning rate reaches 500. With **no clip**, learning is fastest (7.5k steps against 12.5k).
  - This is the opposite of the story's selling point, if anyone reads that chart for speed. The real case for clipping is the Pendulum sweep above.

The SAC and TD3 texts already say this honestly ("on Pendulum DDPG's overestimation does no visible harm"). The
overestimation chart (the critics' targets) does sell TD3. What the sweeps cannot do is show TD3 or SAC *winning*.

## 4. Lab presets: the played seed against 20 seeds

The stories choose their seeds by hand (UCB's story plays seed 58, for example) and average their charts over 200
seeds, so they are fine. The **Lab presets** mostly play seed 1. Their charts average many runs and are right, but the
animated run, the filmstrip and the first impression come from the played seed.

**The hero underperforms, or the lesson is reversed in the played run:**

| Preset | Played run | Typical (median of 20) |
|---|---|---|
| bandit-ucb | UCB picks the best arm 88% of the time, **ε-greedy 90%**: the lesson is reversed | UCB 93% |
| bandit-gradient | no baseline 100% > **with baseline 97%**: reversed. The odds say baseline 16/20, no baseline 11/20. | baseline 98% |
| bandit-optimistic | optimistic 78% (rank 17 of 20) | 100% |
| bandit-drift | constant α 75% | 85% |
| corridor-baseline | **baseline −13.0 is worse than REINFORCE −12.9**: reversed | baseline −11.6 |
| n-step-sarsa | n = 4 takes 305 steps (rank 18 of 20); n = 16 takes 141 | 172 and 109 |
| trpo-cliff | δ = 0.2 never reaches the goal, though 15 of 20 seeds do | 17 steps |
| walk-features | seed 2 is rough for all four feature sets (rank 16–18); their order holds | the played errors run 7–60% higher |
| throw | both racers sit in the bottom 2 of 20; their order holds | 0.3 and 2.9 farther |

**Lucky the other way:** in frozen-mc, on-policy MC's played run (seed 30) gets 0.70. The median is 0.23, and only 6
of 20 seeds succeed. This seed was probably picked on purpose, but it oversells MC in the Lab.

**These are fine:** cliff, Mountain Car (all three presets), Dyna, prioritized sweeping, the deadly triad, A2C, PPO and GAE on
the maze and the cliff, offline, imitation and the model cliffs. The runs that fail are meant to fail: Q-learning's
odds on the cliff, the greedy bandit, max-bias Q-learning, λ = 0, no planning, values starting at −150, the triad,
no entropy bonus, offline Q-learning and behavior cloning. The small gem in seeds-gems is shown on purpose.

## 5. Continuous control stories are stills

- DDPG, TD3 and SAC show snapshots only; only DPG's throw animates.
- Each recording already stores, for its shown seed, one full test episode per block, with the state at every step.
- Playing that episode, so the pendulum swings up as the policy learned it, is a view change. It needs **no rerun**.
- The same goes for the CartPole stories.

## 6. Candidates, by cost

1. **Free (edit `lab.toml`, minutes):** re-pick the played seed of the presets in §4. The candidates are
   UCB, the gradient bandit, the optimistic start, drift, corridor-baseline, n-step SARSA, and maybe walk-features,
   throw and TRPO δ = 0.2. Pick a seed near the median where the lesson shows. Check each intro text for numbers.
2. **Cheap rerun (1–3 min each):** re-pick the shown seed of **dqn-cartpole** (6 stories), then double-DQN and
   dueling.
   - Every seed's curve is already in the files, so a seed near the median can be chosen before anything runs.
   - A small `--shown` option in `record.py` would retrain only that seed for its snapshots and keep the 20 curves. That makes one DQN reshow about 1 minute instead of 6.
   - The story numbers tied to seed 1 must then be updated (for example, "holds the pole after 150,000 steps").
3. **Code only:** animate the test episodes in the Pendulum and CartPole stories (§5).
4. **Expensive (hours), worth a decision first:** the continuous sweeps that are flat on Pendulum. The options are:
   - a harder world, such as Gymnasium's continuous Lunar Lander (needs Box2D) or MuJoCo's Hopper (needs MuJoCo; slow on 4 cores)
   - judging those knobs by speed or stability instead of the final return
   - dropping the flat knobs and keeping only the ones that teach.
