+++
summary = "Some agents use a model of the world to think ahead before acting; others learn straight from experience and never predict what the world will do."
story_in = ["dyna-q", "model-based-deep"]
prereqs = ["mdp", "prediction-control"]
sources = [
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., §1.6, §8.1–8.3 and Example 8.1", url = "http://incompleteideas.net/book/the-book-2nd.html" },
  { text = "Silver et al. (2017), Mastering the game of Go without human knowledge, Nature 550", url = "https://doi.org/10.1038/nature24270" },
  { text = "Schrittwieser et al. (2020), Mastering Atari, Go, chess and shogi by planning with a learned model, Nature 588", url = "https://doi.org/10.1038/s41586-020-03051-4" },
]
+++

## Textbook

### What a model is {#model}

A **model** of the environment is anything an agent can use to predict how the environment will respond to its actions: given a state and an action, it predicts the next state and the reward. Models come in two kinds.

- A **distribution model** gives every possible outcome with its probability: the dynamics $p(s', \rew{r} \mid s, a)$ of [[mdp]].
- A **sample model** produces one outcome at a time, drawn with the right probabilities: $(s', \rew{r}) \sim p(\cdot, \cdot \mid s, a)$.

A distribution model can always produce samples, but not the other way round, and a sample model is often much easier to obtain. A program that simulates rolling a dozen dice is a few lines long; listing the probability of every possible total and arrangement is a much larger task. Simulators of physical systems and games are sample models.

### Planning and learning {#planning}

Using a model to compute or improve values and policies, without acting in the real environment, is **planning**. Improving them from real experience is **learning**. The distinction is about the source of the experience, not about the updates themselves: a planning method can apply exactly the same update as a learning method, only to experience that the model simulates. Dyna-Q does precisely this, interleaving Q-learning updates on real transitions with Q-learning updates on transitions replayed from a learned model ([[dyna-q]]).

### Model-based methods {#based}

**Model-based** methods use a model, either given or learned.

- **The model is given.** Dynamic programming sweeps the states using $p$ ([[policy-evaluation]], [[value-iteration]]). Search methods look ahead from the current state through simulated futures; AlphaGo Zero combined a tree search over the known rules of Go with learned value and policy networks (Silver et al., 2017; [[mcts]]).
- **The model is learned.** The agent estimates the dynamics from its own experience and plans with the estimate. In the tabular case the model can simply record what happened after each state and action ([[models]]); in deep reinforcement learning it is a neural network. MuZero learned a model that predicts only what planning needs, the rewards, values and policies, and matched AlphaZero without being told the rules (Schrittwieser et al., 2020; [[model-based-deep]]).

### Model-free methods {#free}

**Model-free** methods never represent the dynamics. They learn values or policies directly from sampled trajectories: Monte Carlo methods from complete returns ([[mc-prediction]]), temporal-difference methods from single transitions ([[td0]]), policy-gradient methods by adjusting the policy toward actions that turned out well ([[reinforce]]). They need nothing but the stream of states, actions and rewards, which is why most of the atlas is about them.

### The trade-off {#tradeoffs}

| | Model-based | Model-free |
| --- | --- | --- |
| Use of experience | each real step can be replayed in planning many times | each real step is used once, or a few times with replay |
| Errors | the plan exploits the model's mistakes: *model bias* | no model to be wrong, but noisier estimates |
| Computation per real step | larger: planning costs time | smaller |
| Parts to get right | more: the model and the planner | fewer |

The first row is the main attraction of models: real experience is often expensive (a robot wears out, a patient cannot be treated twice), while simulated experience is cheap. In the maze of Sutton and Barto's Example 8.1, a Dyna agent that planned a few dozen steps after each real step found a good path within a few episodes, while plain Q-learning needed many more. The second row is the main danger: a learned model is wrong somewhere, and a planner is very good at finding those places, so a plan can look excellent in the model and fail in the world.

### A spectrum, not a split {#spectrum}

The two families shade into each other. Experience replay, as in [[dqn]], keeps past transitions and learns from them again: a memory that acts like a sample model, but one that can only replay what has already happened. Dyna combines direct learning with planning, and decision-time planning such as tree search can be added on top of almost any learned value function. The question is less “with or without a model” than how much of the agent's computation goes into predicting the world and how much into predicting value.

## Card

### Idea

A **model** predicts what the world will do: the next state and reward after an action. **Model-based** agents use one to plan, thinking ahead before acting. **Model-free** agents learn values or policies straight from experience and never predict the world.

::: analogy
Planning a trip with a map, against learning a city by walking it. The map saves walking, as long as the map is right.
:::

### In symbols {#formula}

$$\text{distribution model: } p(s', \rew{r} \mid s, a) \qquad\qquad \text{sample model: } (s', \rew{r}) \sim p(\cdot, \cdot \mid s, a)$$

### Examples {#examples}

| Model-based | Model-free |
| --- | --- |
| [[value-iteration]] (model given) | [[mc-prediction]] |
| [[dyna-q]] (model learned) | [[q-learning]], [[sarsa]] |
| [[mcts]] and AlphaZero | [[reinforce]], [[ppo]] |

### Why it matters {#why}

It decides what the agent needs: a model needs dynamics or a simulator, and buys sample efficiency at the risk of model bias. The labels *model-free* and *model-based* on each algorithm say which side it is on.

### Pitfalls

- Trusting a learned model where it has seen little data: the planner will find its errors.
- Calling a method model-free because the model was learned: if it plans with it, it is model-based.
- Assuming a simulator is a distribution model. Most simulators only sample.

### Check yourself {#check}

::: question
Is Q-learning with experience replay model-based?
---
Usually called model-free: it never predicts outcomes it has not seen. But the replay memory acts like a limited sample model, which is part of why replay helps.
:::

::: question
What is the main risk of planning with a learned model?
---
Model bias: the planner exploits the model's mistakes, and a plan that looks good in the model can fail in the real world.
:::

::: question
Dyna-Q learns a model and plans with it, and also makes ordinary Q-learning updates. Is it model-based or model-free?
---
Model-based, as the map labels it, because its planning updates come from a learned model. But it keeps the model-free update on real experience as well: it is the standard example of using both, so a wrong model slows it down rather than ruining it.
:::
