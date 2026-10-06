+++
summary = "When an agent does not learn, the cause is usually not the algorithm: it is the world as the agent sees it, a quiet bug, or a knob far from its sweet spot. Check the world first, then make the learner solve something whose answer you know, then watch every number it produces, on several seeds. Each failure has a signature, and the guide's experiments show most of them."
story_in = ["learning-curves", "seeds"]
prereqs = ["hyperparameters", "seeds"]
lab = "deadly-triad"
sources = [
  { text = "Pardo, Tavakoli, Levdik & Kormushev (2018), Time limits in reinforcement learning, ICML", url = "https://arxiv.org/abs/1712.00378" },
  { text = "Engstrom et al. (2020), Implementation matters in deep policy gradients: a case study on PPO and TRPO, ICLR", url = "https://arxiv.org/abs/2005.12729" },
  { text = "Huang, Dossa, Raffin, Kanervisto & Wang (2022), The 37 implementation details of proximal policy optimization, ICLR blog track", url = "https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/" },
  { text = "Henderson, Islam, Bachman, Pineau, Precup & Meger (2018), Deep reinforcement learning that matters, AAAI", url = "https://arxiv.org/abs/1709.06560" },
]
+++

## Textbook

### Start with the world {#world}

Many failures are not in the learner at all. Before touching the algorithm:

- **Play randomly.** The return of a random policy is the baseline every learner must beat. If it is already as good as anything you expect, the reward does not distinguish good behavior from bad ([[reward-design]]).
- **Read the reward.** Its sign, its scale, when it arrives. A reward that only comes at a far-away goal gives nothing to learn from until the goal is found: that is an exploration problem, not a learning one ([[exploration-strategies]]).
- **Know how episodes end.** An episode can *terminate*, because the task is over, or be *cut off* by a time limit. A cut-off state is not worth zero, so its target should still bootstrap from the next state's value. Treating a time limit as a terminal state teaches the agent that the world ends at step 500 (Pardo et al., 2018). The recorded runs in this guide bootstrap through CartPole's limit for this reason.
- **Check what the agent observes.** If the state leaves out something that matters, such as velocities, no learner can do well ([[mdp]]).

### Then a problem you can solve {#known}

Run the learner on something whose answer you know: a small grid where dynamic programming gives the exact values ([[value-iteration]]), a bandit, a two-step chain. If it fails there, the problem is in the code. For networks, a useful test is to make the value network fit a fixed set of targets, with no RL at all: if it cannot learn a regression, it cannot learn from bootstrapped targets either.

### Read the signs {#signs}

Once it runs, the shape of the failure says where to look:

| what you see | often caused by | try |
| --- | --- | --- |
| returns flat from the start | no reward found yet; values that start pessimistic; a step size far too small | explore more, start optimistic, raise the step size |
| learns, then collapses | step size too large; targets that move with every step; a replay memory too small | lower the step size, add a target network, a larger memory |
| values grow without bound | off-policy learning with shared features and bootstrapping; γ near 1 with a large step | on-policy data, a target network, smaller steps |
| values above anything achievable | overestimation by the max | Double Q, check the targets against a known ceiling |
| policy turns deterministic early, stuck on a poor choice | no entropy bonus, a step size too large | an entropy bonus, a smaller step |
| works on some seeds only | a setting at the edge of its sweet spot | measure the odds, move toward the middle of the range |
| good in training, poor in test (or the reverse) | exploration counted in training returns; overfitting to training starts | test greedily, from varied starts |

The guide's experiments show several of these signatures:

- **Divergence.** In Baird's counterexample, the weights grow past 100 in all 20 runs ([[deadly-triad]]).
- **Pessimistic values.** In Mountain Car with values starting at −150, every run ends stuck in the valley ([[exploration-strategies]]).
- **Early commitment.** On the two gems, A2C without an entropy bonus settles on the small gem in every run ([[entropy-bonus]]).
- **Overestimation.** On some seeds, DQN's largest Q-values on CartPole go past 100: more than any state is worth to a learner that bootstraps through the time limit, $1/(1 - \gam)$ ([[dqn-extensions]]).
- **A step that goes too far.** PPO without its clip drifts 20 times further from the policy that collected the data ([[ppo]]).

### Watch everything {#watch}

Plot more than the return. Each number below tells something the return hides:

- **Test returns** played without exploration, next to the training returns.
- **Episode lengths**: a learner that wanders shows it here first.
- **Value estimates** against the returns actually received: they should agree, roughly.
- **TD errors and losses**: they should shrink, but not to zero; a value loss that keeps growing means the targets run away.
- **Entropy, KL and clipped fractions** for policy methods: how fast the policy changes, and how decisive it is.
- **Gradient norms**: sudden spikes come before many collapses.
- **The exploration schedule**: ε, the noise, or SAC's α at each moment.
- **The agent itself**: watching a few episodes, early and late, often reveals a loophole in the reward or a frozen policy faster than any curve ([[reward-design]]).

The Lab charts most of these for each method.

### Quiet bugs {#bugs}

Some bugs let training run and simply make it worse:

- The **sign** of an update: descending what should be climbed.
- A missing $(1 - \text{done})$ in a target, so values bootstrap across the end of an episode.
- **Shapes**: a target of shape $(B)$ minus a prediction of shape $(B, 1)$ broadcasts, silently, to a $B \times B$ matrix.
- Gradients flowing into a target that should be held fixed.
- Testing with exploration still on, or training with it off.
- The **reward scale**: rewards in the hundreds give huge TD errors and huge steps ([[normalization]]).
- Code-level details that papers do not mention: for PPO, the way advantages are normalized, how the value loss is clipped, how learning rates decay. Together they can matter as much as the algorithm (Engstrom et al., 2020; Huang et al., 2022).

### An order of work {#order}

1. The world: random baseline, reward, episode ends, observations.
2. A known problem: small, exact answers.
3. One seed, every number plotted.
4. Several seeds: does it work often, or once?
5. One knob at a time, on a log scale, from published values ([[hyperparameters]]).

## Card

### Idea

When an agent does not learn, check the world first (reward, episode ends, observations), then the learner on a problem with a known answer, then every number it produces, on several seeds. Most failures have a recognizable signature.

::: analogy
A doctor does not start with surgery. First the vital signs, then the simple explanations, then the tests that tell causes apart.
:::

### Signatures {#signatures}

- Flat from the start: no signal yet, or values that start pessimistic.
- Learns, then collapses: steps too large, or a moving target.
- Values without bound: the deadly triad.
- Values above the ceiling: overestimation.
- Deterministic too soon: missing entropy.
- Some seeds only: a setting at the edge of its range.

### Pitfalls

- Debugging the algorithm before checking the world.
- Treating a time limit as the end of the world.
- Watching only the return.
- Concluding from one seed.

### Check yourself {#check}

::: question
CartPole cuts every episode off at 500 steps. Why should the last step's target still include the next state's value?
---
The pole was still up: the cut-off is a rule of the simulator, not the end of the task. If the target treated it as terminal, states late in an episode would look worth less than they are, and the agent would learn that time runs out.
:::

::: question
The value estimates of a DQN agent on CartPole climb past 100. What is wrong, and why is it visible?
---
With rewards of 1 per step and γ = 0.99, no return can exceed 1/(1 − 0.99) = 100, so values above it are overestimates: the max in the target picks up upward errors. A known ceiling makes such errors easy to spot; Double DQN is one fix.
:::

::: question
A run's training returns rise, then fall back to where they started. Name two likely causes.
---
A step size too large, so an update undoes what was learned, or targets that move with every update, such as DQN without a target network. A small replay memory, in which old lessons are quickly overwritten, is a third.
:::
