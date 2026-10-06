+++
summary = "A learning curve plots how well an agent does as it trains, and reading one well is a skill. Training returns include exploration, test returns do not; an average over seeds smooths away the runs that failed; and an agent's own values are a curve worth watching too, since they should track the returns and never pass what the world can pay."
prereqs = ["seeds", "dqn"]
lab = "dqn-cartpole"
sources = [
  { text = "Henderson, Islam, Bachman, Pineau, Precup & Meger (2018), Deep reinforcement learning that matters, AAAI", url = "https://arxiv.org/abs/1709.06560" },
  { text = "Agarwal, Schwarzer, Castro, Courville & Bellemare (2021), Deep reinforcement learning at the edge of the statistical precipice, NeurIPS", url = "https://arxiv.org/abs/2108.13264" },
  { text = "Machado, Bellemare, Talvitie, Veness, Hausknecht & Bowling (2018), Revisiting the Arcade Learning Environment: evaluation protocols and open problems for general agents, JAIR 61", url = "https://arxiv.org/abs/1709.06009" },
]

[story]
scene = "cartpole"
env = "cartpole"

[story.runs]
learn = { recording = "dqn-cartpole", name = "DQN" }
notarget = { recording = "dqn-cartpole-no-target", name = "DQN without a target network" }
+++

## Story

::: step {run = "learn", at = 40, curves = ["learn"], metric = "return"}
**The training curve.** DQN learning to balance a pole, as in [[dqn]]: the average return of the training episodes in each block of 5,000 steps, averaged over 20 seeds. It rises to 273 by block 5, sags to 234 by block 10, then climbs to 480 at the end. The first questions a curve answers: did it learn, how fast, and did it keep what it learned.
:::

::: step {run = "learn", at = 40, curves = ["learn"], metric = "test"}
**The test curve.** After each block, every seed also played one episode without exploration. Averaged over the seeds, it follows the training curve closely here, because DQN's exploration has shrunk to 5% of moves after 20,000 steps; with more exploration the two would part. One test episode is a noisy measure: the run in the picture scored 500, 438, 487, 500, 500, 405, 500, 500, 500 and 500 over its last ten blocks.
:::

::: step {run = "learn", at = 40, curves = ["learn"], metric = "return"}
**What the average hides.** The curve ends at 480, near the ceiling of 500. But it is the average of 20 runs that did not all end alike: 16 average 450 or more over their last four blocks, and the other four end between 402 and 446. An average says what to expect, not what every run does ([[seeds]]). The Lab draws the runs one by one.
:::

::: step {run = "learn", at = 40, curves = ["learn"], metric = "q"}
**The agent's own values are a curve too.** The largest Q-value in each batch, averaged over the seeds, climbs from 4.7 to 101. A state here is worth at most $1 / (1 - \gam) = 100$ to a learner that bootstraps through the time limit, so 101 means overestimation: on 15 of the 20 seeds, the values went past 100 at some point. Values that keep tracking the returns are healthy; values above what the world can pay are not ([[dqn-extensions]]).
:::

::: step {run = "notarget", at = 40, curves = ["notarget"], metric = "q"}
**When the values run away.** The same agent without a target network ([[target-network]]): its average largest Q-value passes 1,000 at block 11 and 10,000 at block 30, peaking near 11,000, while its training episodes never average more than 218 steps. No seed ends well. The value curve showed the trouble long before the return curve could explain it.
:::

## Textbook

### Which return {#which}

- **Training returns** are what the agent earned while learning, exploration included. They are free to collect but mix two things: what the policy knows and what its exploration costs. An ε-greedy agent with ε = 0.1 loses some return to random moves even when its greedy policy is perfect.
- **Test returns** come from separate episodes played without exploration (or with the policy's own randomness, for a stochastic policy). They measure the policy itself, at the cost of extra episodes. One test episode is noisy; several, from varied starts, are better.

Report which one a curve shows. They answer different questions: how well the agent does while it learns, and how good a policy it has learned.

### Which x-axis {#x-axis}

Plot against **steps of experience**, not episodes or updates. Episodes vary in length, often growing as the agent improves, so a curve over episodes stretches some parts and squeezes others; and methods differ in how many updates they make per step. Steps of experience are what an agent pays for in interaction. Wall-clock time is a second axis worth showing when computation is the limit ([[which-algorithm]]).

### Smoothing {#smoothing}

Raw curves are noisy, so they are averaged: over a window of episodes, or, as in this guide's recorded runs, over blocks of steps. Smoothing hides noise but also hides sudden collapses and shifts the curve right. Say how much smoothing was applied, and keep it modest.

### Many seeds {#seeds}

A curve from one seed shows one run's luck. Average several seeds and show their spread: thin lines for each run, or a shaded band. Many papers shade the standard error, others a confidence interval, others the range; they mean different things, so say which. When runs split into groups, say how many ended well instead of averaging them together ([[seeds]]). To compare final performance, average the last part of training for each seed, then compare those numbers across seeds.

### Curves besides the return {#others}

- **Value estimates** should rise toward the returns the agent actually gets, and stay below any known bound. Values above it mean overestimation; values that explode mean divergence ([[deadly-triad]]).
- **Losses** in RL do not fall steadily as in supervised learning: the targets change as the agent learns, and the loss can rise while the agent improves, for instance as values grow toward larger returns.
- **Entropy, KL and clipped fractions** for policy-gradient methods: a policy that turns deterministic early, or moves too far per update, shows it here first ([[ppo]]).

### Historical remarks {#history}

Henderson et al. (2018) showed how much curves in deep RL papers depended on seeds, smoothing and reporting choices. Machado et al. (2018) proposed evaluation protocols for Atari, including reporting the average over the last part of training rather than the best point reached. Agarwal et al. (2021) recommended reporting interval estimates and whole score distributions instead of point estimates of the mean.

## Card

### Idea

Read a learning curve with its axes in mind: training or test returns, over steps of experience, smoothed by how much, averaged over how many seeds. Watch the agent's values too: they should track the returns and stay below what the world can pay.

::: analogy
A patient's temperature chart: one reading tells little, a smoothed trend tells more, and the doctor still wants to know when the readings were taken and with which thermometer.
:::

### Checklist {#checklist}

- Training (with exploration) or test (without)?
- Steps of experience on the x-axis.
- How much smoothing?
- How many seeds, and what does the band show?
- Values: tracking the returns, under the ceiling?

### Pitfalls

- Reading a training curve as the quality of the policy.
- Reporting the best point of a curve instead of its end.
- An average over seeds that hides failed runs.
- Ignoring a value curve that runs away while the returns still look fine.

### Check yourself {#check}

::: question
Why can a training curve sit below the test curve for the same agent?
---
Training episodes include exploration: random moves that cost return even when the greedy policy is good. Test episodes are played without them, so they measure the policy itself.
:::

::: question
Why plot against steps of experience rather than episodes?
---
Episodes change length as the agent learns, so equal intervals of episodes are unequal amounts of experience. Steps are the cost an agent pays to the world, and they put methods with different episode lengths on the same footing.
:::

::: question
DQN's average largest Q-value on CartPole reaches 101. What does that tell you?
---
That it overestimates: with a reward of 1 per step and γ = 0.99, no state is worth more than 100 to a learner that bootstraps through the time limit. The max in the target picks up upward errors. Double DQN reduces it.
:::
