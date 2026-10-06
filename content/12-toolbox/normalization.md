+++
summary = "Networks learn best when their inputs, outputs and targets stay in a moderate range, and RL rarely provides that on its own: observations come in arbitrary units, rewards can be tiny or huge, and values grow as the agent improves. Normalizing observations, scaling rewards, normalizing advantages, clipping gradients and bounding steps are small changes that often decide whether a run learns at all."
prereqs = ["neural-networks", "hyperparameters"]
lab = "ddpg-pendulum"
sources = [
  { text = "Mnih et al. (2015), Human-level control through deep reinforcement learning, Nature 518", url = "https://www.nature.com/articles/nature14236" },
  { text = "van Hasselt, Guez, Hessel, Mnih & Silver (2016), Learning values across many orders of magnitude (PopArt), NeurIPS", url = "https://arxiv.org/abs/1602.07714" },
  { text = "Engstrom et al. (2020), Implementation matters in deep policy gradients: a case study on PPO and TRPO, ICLR", url = "https://arxiv.org/abs/2005.12729" },
  { text = "Andrychowicz et al. (2021), What matters in on-policy reinforcement learning? A large-scale empirical study, ICLR", url = "https://arxiv.org/abs/2006.05990" },
  { text = "Ba, Kiros & Hinton (2016), Layer normalization", url = "https://arxiv.org/abs/1607.06450" },
]
+++

## Textbook

### Why scale matters {#why}

A network's weights start small, and its gradient steps are sized for inputs and targets of order one. Feed it a position in meters next to a velocity in millimeters per second, or ask it to predict values in the thousands, and every step is too large in some directions and too small in others. Adam rescales each weight's step by the size of its recent gradients, which helps, but it cannot fix a target that is a thousand times larger than the network's outputs can comfortably reach, or a loss dominated by one huge error ([[neural-networks]]). RL makes this worse, because the scales change as the agent learns: values grow as the policy improves, and the states visited drift.

### Observations {#observations}

Standardize each input: subtract a running mean and divide by a running standard deviation, both updated from the states seen so far, and clip the result to a few standard deviations. Many PPO implementations do this by default, and the large study of Andrychowicz et al. (2021) found it one of the choices that matter most for on-policy methods. With linear features, the same concern appears as the step size: with tile coding, the step size is divided by the number of tilings, since each state activates that many weights at once ([[features]]).

### Rewards and values {#rewards}

The size of the rewards sets the size of every value, every TD error and so every step.

- **Clipping.** DQN clipped every Atari reward to $[-1, 1]$, so one set of knobs worked across games whose scores differ by orders of magnitude, at the price of treating a small and a large reward alike (Mnih et al., 2015).
- **Scaling.** Multiply rewards by a constant, or divide them by a running estimate of the spread of the returns, as many PPO implementations do. The recorded runs of this guide on Pendulum scale rewards by 0.1, so that values of a few hundred become a few tens.
- **Adaptive targets.** PopArt (van Hasselt et al., 2016) normalizes the value targets with running statistics and rescales the network's last layer to match, so its outputs are preserved while the scale changes underneath.

### Advantages {#advantages}

Policy gradients weight each action's log-probability by its advantage. Normalizing the advantages in each batch to mean 0 and standard deviation 1 keeps the size of the policy's steps independent of the reward scale and of how well the critic has learned so far. PPO and A2C implementations usually do it, as the recorded runs of this guide do ([[ppo]]). It is a safeguard more than a cure: in the study of Andrychowicz et al. (2021) it made little difference on their tasks.

### Gradients and steps {#gradients}

- **Gradient clipping**: if the norm of the whole gradient exceeds a threshold, scale it down. One bad batch then cannot throw the weights far.
- **The Huber loss**: squared for small TD errors, linear for large ones, so large errors push no harder than a fixed amount. DQN used it on Atari, where rewards were clipped and errors stayed small. The recorded DQN on CartPole uses a squared loss instead: its values legitimately reach 100, and large errors there are signal, not noise.
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
