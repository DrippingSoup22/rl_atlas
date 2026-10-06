+++
summary = "Every method has knobs, and in reinforcement learning their values decide not only how fast a run learns but whether it learns at all. Most knobs have a sweet spot with failure on both sides; the useful measure of a setting is how often it works over many seeds, and the useful search changes one knob at a time on a log scale, starting from values known to work."
prereqs = ["entropy-bonus", "dqn", "ppo"]
lab = "entropy-gems"
sources = [
  { text = "Henderson, Islam, Bachman, Pineau, Precup & Meger (2018), Deep reinforcement learning that matters, AAAI", url = "https://arxiv.org/abs/1709.06560" },
  { text = "Andrychowicz et al. (2021), What matters in on-policy reinforcement learning? A large-scale empirical study, ICLR", url = "https://arxiv.org/abs/2006.05990" },
  { text = "Bergstra & Bengio (2012), Random search for hyper-parameter optimization, Journal of Machine Learning Research 13", url = "https://www.jmlr.org/papers/v13/bergstra12a.html" },
  { text = "Engstrom et al. (2020), Implementation matters in deep policy gradients: a case study on PPO and TRPO, ICLR", url = "https://arxiv.org/abs/2005.12729" },
  { text = "Raffin et al. (2021), Stable-Baselines3: reliable reinforcement learning implementations, JMLR 22 (with tuned settings in RL Baselines3 Zoo)", url = "https://jmlr.org/papers/v22/20-1364.html" },
]

[story]
scene = "grid"
env = "two-gems"
seed = 1
average = 40

[story.runs]
low = { algorithm = "a2c", alpha = 2.0, alphaW = 0.3, workers = 4, n = 5, beta = 0.0, gamma = 0.95, maxSteps = 500, units = 200, name = "β = 0" }
right = { algorithm = "a2c", alpha = 2.0, alphaW = 0.3, workers = 4, n = 5, beta = 0.1, gamma = 0.95, maxSteps = 500, units = 200, name = "β = 0.1" }
high = { algorithm = "a2c", alpha = 2.0, alphaW = 0.3, workers = 4, n = 5, beta = 0.5, gamma = 0.95, maxSteps = 500, units = 200, name = "β = 0.5" }
+++

## Story

::: step {run = "low", at = 200, values = true}
**One knob, three values.** A small gem two steps from the start pays 0.3; a big one seven steps away pays 1. Four A2C workers learn a policy here, and the only knob that changes is β, the weight of the entropy bonus ([[entropy-bonus]]). With **β = 0** there is no bonus. The first gem the policy finds is the small one, and it learns to go there and stop looking. Over its last 20 rounds, its episodes take 2 steps and earn $\rew{0.3}$. Of the 40 runs averaged at the end of this story, none ends up going for the big gem.
:::

::: step {run = "right", at = 200, values = true}
**β = 0.1.** The bonus keeps the policy undecided long enough for its workers to stumble on the big gem, and then lets it commit. Over its last 20 rounds, its episodes take 13 steps and earn $\rew{0.93}$ on average. Of the 40 runs, 35 end up going for the big gem.
:::

::: step {run = "high", at = 200, values = true}
**β = 0.5.** Too much of a good thing. The bonus now outweighs the gems, and the policy is paid to stay random: its episodes wander for 25 steps and end at whichever gem they bump into, for $\rew{0.59}$ on average. Of the 40 runs, none commits to the big gem. Raising β further makes it worse: at β = 1, 40 runs average $\rew{0.51}$.
:::

::: step {run = "right", at = 200, curves = ["low", "right", "high"], metric = "return"}
**The average of 40 runs per value.** Too little bonus and the policy commits too early; too much and it never commits. Between them is a narrow band: at β = 0.05, 18 of 40 runs go for the big gem, and at β = 0.2 none does, for $\rew{0.82}$ per episode. That shape, failure on both sides of a sweet spot, is the rule for knobs in reinforcement learning, not the exception. The Lab's sweep draws it for any knob of any preset.
:::

## Textbook

### Why knobs matter more here {#why}

In supervised learning a poor step size usually costs time: the loss falls slowly or noisily, and the model still ends up somewhere reasonable. In reinforcement learning the agent's own behavior produces its data. A knob that makes the policy commit too early stops it from ever seeing the better option; one that makes the values overshoot changes the targets of every later update. So a knob decides whether a run works at all, and runs with the same settings and different seeds can end in different places. A setting is not good or bad, but good with some probability, and that probability is what to measure ([[seeds]]).

### Sweet spots {#sweet-spots}

Almost every knob in this guide has failure on both sides. The Lab's sweeps measure how often: below, how many runs end well, out of 40 seeds for the two gems and out of the 5 seeds recorded for each setting of the deep methods. A CartPole run ends well when its last training episodes balance the pole (450 steps or more), a Pendulum run when they hold the pendulum up (a return of −250 or more).

| Knob | too low | sweet spot | too high |
| --- | --- | --- | --- |
| entropy bonus $\beta$ (two gems) | 0 of 40 go for the big gem ($\beta = 0$) | 35 of 40 ($\beta = 0.1$) | 0 of 40 ($\beta = 0.2$) |
| TRPO's trust region $\delta$ (CartPole) | 4 of 5 ($\delta = 0.001$) | 5 of 5 ($0.003$ to $0.1$) | 0 of 5 ($\delta = 1$) |
| PPO's clip $\epsilon$ (Pendulum) | 0 of 5 (no clip) | 5 of 5 ($0.1$, $0.2$) | 0 of 5 ($\epsilon = 0.5$) |
| PPO's passes per batch (Pendulum) | 4 of 5 (1 pass) | 5 of 5 (4, 10) | 1 of 5 (30 passes) |
| DQN's step size $\alp$ (CartPole) | 4 of 5 ($10^{-4}$) | 5 of 5 ($5 \cdot 10^{-4}$) | 2 of 5 ($2.5 \cdot 10^{-3}$) |
| DQN's target period $C$ (CartPole) | 0 of 5 (no copy) | 5 of 5 (500 steps) | 3 of 5 (2,000 steps) |
| DQN's memory $N$ (CartPole) | 2 of 5 (128 steps) | 5 of 5 (10,000 steps) | 2 of 5 (100,000 steps) |

The reasons differ, and the [[bias-variance]] trade explains several of them. Too little exploration commits early, too much never commits. Too small a step learns slowly, too large a step knocks over what was learned. A target that moves too often chases itself, one that moves too rarely crawls. Knowing why a knob fails on each side tells which way to turn it.

### Searching {#search}

Some habits make the search cheaper:

- **Start from values that work.** For common methods there are published settings, tuned on many problems: begin there, and change one knob at a time. For DQN-like methods: a memory of $10^4$ to $10^6$ steps, a target copied every $10^3$ to $10^4$ steps, a step size near $10^{-4}$ to $10^{-3}$ with Adam. For PPO: $\epsilon = 0.2$, 3 to 10 passes, GAE with $\lambda = 0.95$, a step size near $3 \cdot 10^{-4}$. For SAC: an automatically tuned $\alpha$, $\tau = 0.005$, a step size near $3 \cdot 10^{-4}$.
- **Use log scales.** Step sizes, $\beta$, $\delta$, $1 - \gamma$ and memory sizes matter by their order of magnitude. Try $10^{-4}, 3 \cdot 10^{-4}, 10^{-3}, \dots$, not $0.001, 0.002, 0.003$.
- **Judge by the odds, not by one run.** Compare settings by how many seeds end well, with the same seeds for each value, so that two settings face the same luck ([[seeds]]).
- **Search randomly when several knobs matter.** Varying all knobs at random points beats a grid of the same size, because a grid tries each value of an unimportant knob many times while important ones get few distinct values (Bergstra & Bengio, 2012).
- **Expect interactions.** The best step size depends on the batch size, the reward scale, the network's width. A sweep of one knob holds the others fixed, and its sweet spot moves when they change ([[normalization]]).

### The ones that matter most {#most}

Large studies agree on a short list. For on-policy methods, Andrychowicz et al. (2021) single out the discount, the normalization of observations, and the initialization of the policy's last layer (small, so that the starting policy is close to uniform), with GAE's $\lambda$ and the step size next; many other choices mattered little once in their usual range. For value-based methods, the step size, the target period and the memory size. For all of them, the reward scale: it sets the size of every TD error and so of every step ([[normalization]]). And implementation details that papers rarely mention can matter as much as the knobs they do list (Engstrom et al., 2020).

### Historical remarks {#history}

Henderson et al. (2018) showed how fragile deep RL results were: the same algorithm, with different code, seeds or knobs, could rank first or last. Their paper changed how results are reported, with more seeds and confidence intervals. Libraries with tuned settings for many problems, such as Stable-Baselines3 and its Zoo, made good starting values easy to find.

## Card

### Idea

A knob's value decides whether a run learns, not only how fast, and most knobs fail on both sides of a sweet spot. Measure a setting by how often it works over many seeds, start from known values, and change one knob at a time on a log scale.

::: analogy
Seasoning a dish: too little and it is bland, too much and it is ruined, and the right amount depends on what else is in the pot.
:::

### How to search {#how}

1. Start from published values for the method.
2. Change one knob, on a log scale, with the same seeds for every value.
3. Compare how many runs end well, not the best run.
4. With several knobs, sample them at random instead of on a grid.

### Why it matters {#why}

- In RL the policy makes its own data, so a bad knob can stop learning altogether.
- The same setting can work for one seed and fail for another: only the odds tell settings apart.
- Most knobs fail both ways, so "more" is rarely the direction to search in.

### Pitfalls

- Tuning on one seed, then reporting that seed.
- Comparing settings that faced different seeds.
- Linear grids for knobs that act on a log scale.
- Forgetting interactions: a new batch size or reward scale moves every sweet spot.

### Check yourself {#check}

::: question
Why can too large an entropy bonus be as bad as none?
---
The bonus pays the policy for staying random. Too small and it commits to the first gem it finds; too large and staying random pays more than any gem, so it never commits. On two gems, 35 of 40 runs ended up going for the big gem with β = 0.1, none with β = 0 or β = 0.5.
:::

::: question
Two settings each ran once: the first scored higher. What else would you want to know?
---
How many seeds end well for each, with the same seeds for both. One run per setting compares their luck as much as the settings.
:::

::: question
Why search step sizes on a log scale?
---
Their effect depends on their order of magnitude: $10^{-4}$ and $2 \cdot 10^{-4}$ behave alike, $10^{-4}$ and $10^{-2}$ do not. A linear grid spends most of its points on the large values.
:::
