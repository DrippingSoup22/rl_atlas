+++
summary = "Start from the problem, not from the newest method. Do you know the rules of the world? Is the state small enough for a table? Are the actions discrete or continuous? Are samples cheap or precious? Is there only a fixed dataset? Each answer narrows the choice to a family, and within a family a robust default, well tuned, beats a fancier method used carelessly."
prereqs = ["model-based-free", "on-off-policy"]
lab = "ppo-pendulum"
sources = [
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed.", url = "http://incompleteideas.net/book/the-book-2nd.html" },
  { text = "Hessel et al. (2018), Rainbow: combining improvements in deep reinforcement learning, AAAI", url = "https://arxiv.org/abs/1710.02298" },
  { text = "Schulman, Wolski, Dhariwal, Radford & Klimov (2017), Proximal policy optimization algorithms", url = "https://arxiv.org/abs/1707.06347" },
  { text = "Haarnoja, Zhou, Abbeel & Levine (2018), Soft actor-critic: off-policy maximum entropy deep reinforcement learning with a stochastic actor, ICML", url = "https://arxiv.org/abs/1801.01290" },
  { text = "Levine, Kumar, Tucker & Fu (2020), Offline reinforcement learning: tutorial, review, and perspectives on open problems", url = "https://arxiv.org/abs/2005.01643" },
]
+++

## Textbook

### Questions to ask first {#questions}

1. **Do you know the rules of the world?** If you can compute what each action does, you can plan instead of learning by trial: dynamic programming for small problems ([[value-iteration]]), search such as MCTS for large ones ([[mcts]]). Learned models sit in between ([[model-based-deep]]).
2. **Is the state small enough for a table?** Then tabular methods are exact, simple and fast: Q-learning, SARSA, Expected SARSA ([[expected-sarsa]]), with Dyna if experience is expensive ([[dyna-q]]).
3. **Are the actions discrete or continuous?** Value-based methods need a max over actions, easy for a few discrete ones. Continuous actions call for an actor ([[policy-parameterization]]).
4. **Are samples cheap or precious?** With a fast simulator, run many copies and use an on-policy method that throws data away after a few passes. With a real robot, reuse every sample through replay.
5. **Can you interact at all?** With only a fixed dataset of past behavior, you need offline RL ([[offline-rl]]); with demonstrations of good behavior, start by imitating them ([[imitation]]).

### A map of the choices {#map}

| problem | a good first choice | why |
| --- | --- | --- |
| known model, small | value iteration, policy iteration | exact answers, no exploration needed |
| known model, huge, discrete (games) | MCTS with learned values and policy | plans where it matters |
| small, unknown, discrete | Q-learning or Expected SARSA; Dyna-Q | tables are exact and fast |
| large states (images), discrete actions | DQN with Double and dueling heads, prioritized replay | sample reuse; Rainbow combines the add-ons |
| continuous actions, samples precious | SAC, or TD3 | off-policy: every sample replayed many times |
| continuous or discrete, simulator fast and parallel | PPO | robust, simple, scales with copies of the world |
| fixed dataset, no interaction | offline RL (conservative value methods) | stays near the data it has |
| demonstrations available | imitation learning, then RL | starts from competence instead of chance |
| several agents learning together | multi-agent methods | each agent's world changes as the others learn |

### Sample-efficient is not the same as fast {#efficient}

The guide's Pendulum runs show the trade between the two families of continuous-control methods. Counting the steps of experience until a block of training averages a return of −250 or better:

| method | family | steps of experience, median | seeds |
| --- | --- | --- | --- |
| PPO | on-policy | 35,000 | 20 |
| DDPG | off-policy | 6,000 | 20 |
| TD3 | off-policy | 9,000 | 20 |
| SAC | off-policy | 6,000 | 20 |

The off-policy methods need four to six times fewer steps, because they learn from every transition many times over. But they take a gradient step on a replayed batch after every single step of experience, while PPO spends its computation in large batches after thousands of steps. When steps of experience are cheap, as in a fast simulator, PPO's cheap steps can make up much of the difference in time. When they are expensive, as with a robot or a slow simulator, sample efficiency is what counts ([[ddpg]], [[ppo]]).

### Defaults that hold up {#defaults}

Within a family, a well-tuned standard method is usually the best start:

- **Discrete actions, off-policy**: DQN with Double DQN's target, often with dueling heads and prioritized replay ([[dqn-extensions]]).
- **On-policy**: PPO, with GAE and normalized advantages ([[ppo]]).
- **Continuous, off-policy**: SAC with automatic entropy tuning, or TD3 ([[sac]], [[td3]]).

Published settings for these methods are a better first guess than settings of your own ([[hyperparameters]]), and the odds over several seeds are a better judge than a single run ([[seeds]]).

### What matters more than the choice {#more}

On most problems the difference between two reasonable methods is smaller than the difference made by the reward ([[reward-design]]), the observations, the scale of the inputs and rewards ([[normalization]]), and the knobs. If a standard method fails, the cause is more often one of these than the method itself ([[debugging]]).

## Card

### Idea

Choose by the problem: known model or not, table or features, discrete or continuous actions, cheap or precious samples, live interaction or a fixed dataset. Then start from the standard method of that family, with published settings.

::: analogy
Choosing a vehicle: you ask where you are going, what you carry and what roads there are before you compare engines.
:::

### Quick guide {#quick}

- Known rules: plan (DP, MCTS).
- Small and unknown: tabular Q-learning, Expected SARSA, Dyna-Q.
- Discrete actions, big states: DQN with its extensions.
- Continuous actions, precious samples: SAC or TD3.
- Fast parallel simulator: PPO.
- Only a dataset: offline RL; demonstrations: imitation first.

### Pitfalls

- Picking the newest method instead of the one that fits the problem.
- Comparing methods by their steps of experience alone, or by their computing time alone.
- Blaming the method for a reward or observation problem.
- Judging a method on one seed, with untuned settings.

### Check yourself {#check}

::: question
On Pendulum, SAC needs about 6,000 steps of experience and PPO about 35,000. When might PPO still be the better choice?
---
When steps of experience are cheap: a fast simulator that runs many copies in parallel. SAC takes a gradient step after every step of experience, so its computation per step is much higher. If time and computation are the limit rather than experience, PPO's cheap steps can make up the difference.
:::

::: question
You have a large log of a warehouse robot's past behavior and cannot experiment with the robot. Which family fits, and why not plain DQN?
---
Offline RL. Plain off-policy methods can learn from logged data, but they assume they can correct their mistakes by trying actions; without interaction, their estimates for actions the log never shows can be wildly optimistic, and nothing corrects them. Offline methods stay close to what the data supports.
:::

::: question
Why do value-based methods like DQN not suit continuous actions directly?
---
Their targets and their action choice need the action with the largest value, a max over all actions. With a few discrete actions that is a short list; with a continuous range it is an optimization problem at every step. Actor–critic methods learn an actor that outputs the action instead.
:::
