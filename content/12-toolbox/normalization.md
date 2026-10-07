+++
summary = "Networks learn best when their inputs, outputs and targets stay in a moderate range, and RL rarely provides that on its own: observations come in arbitrary units, rewards can be tiny or huge, and values grow as the agent improves. Normalizing observations, scaling rewards, normalizing advantages, clipping gradients and bounding steps are small changes that often decide whether a run learns at all."
prereqs = ["neural-networks", "hyperparameters"]
lab = "reward-units"
sources = [
  { text = "Mnih et al. (2015), Human-level control through deep reinforcement learning, Nature 518", url = "https://www.nature.com/articles/nature14236" },
  { text = "van Hasselt, Guez, Hessel, Mnih & Silver (2016), Learning values across many orders of magnitude (PopArt), NeurIPS", url = "https://arxiv.org/abs/1602.07714" },
  { text = "Engstrom et al. (2020), Implementation matters in deep policy gradients: a case study on PPO and TRPO, ICLR", url = "https://arxiv.org/abs/2005.12729" },
  { text = "Andrychowicz et al. (2021), What matters in on-policy reinforcement learning? A large-scale empirical study, ICLR", url = "https://arxiv.org/abs/2006.05990" },
  { text = "Ba, Kiros & Hinton (2016), Layer normalization", url = "https://arxiv.org/abs/1607.06450" },
]

[story]
scene = "grid"
env = "dyna-maze"
seed = 2
average = 20
formula = '''\step{1}{\pol{\theta} \leftarrow \pol{\theta} + \alp\, \err{\hat A}\, \nabla \ln \pol{\pi(A \mid S, \theta)}} \step{2}{\qquad \err{\hat A} \leftarrow \frac{\err{\hat A} - \text{mean}}{\text{spread}}}'''

[story.runs]
one = { algorithm = "a2c", alpha = 2.0, alphaW = 0.3, workers = 4, n = 5, beta = 0.0, gamma = 0.95, maxSteps = 1000, units = 60, name = "the gem pays 1" }
cent = { algorithm = "a2c", alpha = 2.0, alphaW = 0.3, workers = 4, n = 5, beta = 0.0, gamma = 0.95, maxSteps = 1000, units = 60, rewardScale = 0.01, name = "pays 0.01" }
hundred = { algorithm = "a2c", alpha = 2.0, alphaW = 0.3, workers = 4, n = 5, beta = 0.0, gamma = 0.95, maxSteps = 1000, units = 60, rewardScale = 100.0, name = "pays 100" }
centN = { algorithm = "a2c", alpha = 2.0, alphaW = 0.3, workers = 4, n = 5, beta = 0.0, gamma = 0.95, maxSteps = 1000, units = 60, rewardScale = 0.01, normalize = 1, name = "pays 0.01, normalized" }
oneN = { algorithm = "a2c", alpha = 2.0, alphaW = 0.3, workers = 4, n = 5, beta = 0.0, gamma = 0.95, maxSteps = 1000, units = 60, normalize = 1, name = "pays 1, normalized" }
hundredN = { algorithm = "a2c", alpha = 2.0, alphaW = 0.3, workers = 4, n = 5, beta = 0.0, gamma = 0.95, maxSteps = 1000, units = 60, rewardScale = 100.0, normalize = 1, name = "pays 100, normalized" }
+++

## Story

::: step {run = "one", at = 60, range = 1, formula = 1}
**A2C in the maze**, as in [[a2c|its own entry]]: four workers, one update every five steps, and the gem pays 1. Each step of the policy is the step size times an advantage, so its size depends on how big the rewards are. After 60 rounds the arrows lead to the gem, and the last rounds take 15 to 17 steps.
:::

::: step {run = "cent", at = 60, range = 0.01}
**The same world, with the gem paying 0.01:** cents instead of euros. Every TD error and every advantage is a hundred times smaller, and so is every step of the policy. The critic copes, since its values simply come out in cents. The policy barely moves: after 60 rounds, the four moves from the start still have probability 0.25 each, and the last three rounds take 321, 515 and 261 steps.
:::

::: step {run = "hundred", at = 1, range = 100}
**Now the gem pays 100.** In the first round, a worker reaches the gem a few moves after bumping into the edge below it, and that bump shares the prize: an advantage near 90, where with a gem worth 1 it would be at most 1. Steps that large overshoot. By the end of the round, bumping into that edge has probability 1.0000.
:::

::: step {run = "hundred", at = 2, range = 100}
**Round two never ends.** All four workers end up on that tile, bumping into the edge until their 1,000 steps run out. The bump's advantages now come out negative, but a move with probability 1 cannot unlearn itself: the gradient of $\ln \pol{\pi}$ for it is $1 - \pol{\pi} = 0$. The run is stuck for good.
:::

::: step {run = "one", at = 60, curves = ["cent", "one", "hundred"], metric = "steps"}
**Twenty runs of each.** With the gem worth 1, all 20 runs learn the way. With 0.01, none does within 60 rounds. With 100, the 14 runs that escape the trap are fast; 6 get stuck at 1,000 steps a round for good.
:::

::: step {run = "hundredN", at = 60, range = 100, formula = 2, curves = ["centN", "oneN", "hundredN"], metric = "steps"}
**One line fixes it: normalize each batch's advantages**, subtracting their mean and dividing by their spread. The units cancel, and the three lines lie on top of each other: in cents, euros or hundreds, all 60 runs learn the way, in 14 to 16 steps. [Try the units in the Lab](lab:reward-units).
:::

## Textbook

### Why scale matters {#why}

A network's weights start small, and its gradient steps are sized for inputs and targets of order one. Feed it a position in meters next to a velocity in millimeters per second, or ask it to predict values in the thousands, and every step is too large in some directions and too small in others. Adam rescales each weight's step by the size of its recent gradients, which helps, but it cannot fix a target that is a thousand times larger than the network's outputs can comfortably reach, or a loss dominated by one huge error ([[neural-networks]]). RL makes this worse, because the scales change as the agent learns: values grow as the policy improves, and the states visited drift.

### Observations {#observations}

Standardize each input: subtract a running mean and divide by a running standard deviation, both updated from the states seen so far, and clip the result to a few standard deviations. Many PPO implementations do this by default, and the large study of Andrychowicz et al. (2021) found it one of the choices that matter most for on-policy methods. With linear features, the same concern appears as the step size: with tile coding, the step size is divided by the number of tilings, since each state activates that many weights at once ([[features]]).

### Rewards and values {#rewards}

The size of the rewards sets the size of every value, every TD error and so every step.

- **Clipping.** DQN clipped every Atari reward to $[-1, 1]$, so one set of knobs worked across games whose scores differ by orders of magnitude, at the price of treating a small and a large reward alike (Mnih et al., 2015).
- **Scaling.** Multiply rewards by a constant, or divide them by a running estimate of the spread of the returns, as many PPO implementations do. The recorded runs of this guide on Pendulum scale rewards by 0.1, so that values of a few hundred become a few tens. How much this matters depends on the optimizer: in a sweep of DDPG on Pendulum, 10 seeds per value, rewards scaled by 1 learned as fast as by 0.1 and every run ended well, while a scale of 0.01 more than doubled the steps half the seeds needed to train at −250 or better (15,750 against 6,750). Adam divides each step by the size of recent gradients, which absorbs much of a change of scale, though not all of it.
- **Adaptive targets.** PopArt (van Hasselt et al., 2016) normalizes the value targets with running statistics and rescales the network's last layer to match, so its outputs are preserved while the scale changes underneath.

### Advantages {#advantages}

Policy gradients weight each action's log-probability by its advantage. Normalizing the advantages in each batch to mean 0 and standard deviation 1 keeps the size of the policy's steps independent of the reward scale and of how well the critic has learned so far. PPO and A2C implementations usually do it, as the recorded runs of this guide do ([[ppo]]). It is a safeguard more than a cure: in the study of Andrychowicz et al. (2021) it made little difference on their tasks, whose rewards were already of a sensible size.

::: example {#ex-units} The same A2C, in other units
In the guide's maze, A2C with four workers and a step size of 2 learns the way to the gem in all of 20 runs when the gem pays 1. Pay 0.01 instead and every step of the policy is a hundred times smaller: none of 20 runs learns it within 60 rounds. Pay 100 and the steps overshoot: in 6 runs of 20, one round pushes a useless move, such as bumping into a wall, to probability 1, where its gradient $1 - \pol{\pi}$ vanishes and the run stays stuck. With the advantages of each batch normalized, all 60 runs learn the way, whatever the units, along curves that cannot be told apart.
:::

### Gradients and steps {#gradients}

- **Gradient clipping**: if the norm of the whole gradient exceeds a threshold, scale it down. One bad batch then cannot throw the weights far.
- **The Huber loss**: squared for small TD errors, linear for large ones, so large errors push no harder than a fixed amount. DQN used it on Atari, where rewards were clipped and errors stayed small. The recorded DQN on CartPole uses a squared loss instead: its values legitimately reach 100, and large errors there are signal, not noise. With the Huber loss, one of its 20 runs ends well; with the squared loss, 13 do.
- **Bounded policy steps**: PPO's clip and TRPO's KL bound limit how far one update can move the policy, whatever the scale of the gradient ([[trpo]]).
- **Normalization layers**: layer normalization inside the networks, especially the critics, keeps hidden activations in range as training goes on (Ba et al., 2016).

## Card

### Idea

Keep the numbers a network sees in a moderate range: standardize observations, clip or scale rewards and value targets, normalize advantages per batch, and clip gradients. Cheap changes, often the difference between a run that learns and one that does not.

::: analogy
Mixing a recording: before any clever effects, set every microphone's level so that no voice drowns the others and none is lost in the noise.
:::

### The tools {#tools}

- Observations: running mean and standard deviation, then clip.
- Rewards: clip, scale, or divide by the spread of returns; PopArt for targets.
- Advantages: mean 0, standard deviation 1 in each batch.
- Gradients: clip their norm; Huber loss; bounded policy steps.

### Pitfalls

- Normalizing with statistics that differ between training and testing.
- Clipping rewards in a task where their size carries the meaning.
- Changing the reward scale and keeping knobs tuned for the old one, such as SAC's entropy weight.

### Check yourself {#check}

::: question
Why does DQN clip Atari rewards to [−1, 1], and what does it lose?
---
So that one step size and one set of knobs suit every game, whatever its score scale. It loses the difference between small and large rewards: eating a small pellet and winning a big prize look the same.
:::

::: question
Why normalize advantages in each batch?
---
It keeps the size of the policy's steps independent of the reward scale and of the critic's current accuracy, so one step size works throughout training and across tasks.
:::

::: question
A run's TD errors are in the thousands and its value network learns erratically. What would you try first?
---
Scale the rewards, or normalize the value targets, so that the values the network must output are of order one to a hundred, and clip the gradient norm. Then retune the step size for the new scale.
:::
