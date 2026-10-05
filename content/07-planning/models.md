+++
summary = "A model is anything the agent can ask “what would happen if I did this here?” Planning is learning from the model's answers instead of from the world: the same updates, applied to imagined experience."
prereqs = ["model-based-free", "q-learning", "policy-evaluation"]
lab = "dyna-maze"
sources = [
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., §8.1, §8.5–8.6 and §8.8–8.11", url = "http://incompleteideas.net/book/the-book-2nd.html" },
  { text = "Sutton (1990), Integrated architectures for learning, planning, and reacting based on approximating dynamic programming, Proceedings of the 7th International Conference on Machine Learning", url = "https://doi.org/10.1016/B978-1-55860-141-3.50030-4" },
  { text = "Barto, Bradtke & Singh (1995), Learning to act using real-time dynamic programming, Artificial Intelligence 72", url = "https://doi.org/10.1016/0004-3702(94)00011-O" },
  { text = "Tesauro & Galperin (1997), On-line policy improvement using Monte-Carlo search, Advances in Neural Information Processing Systems 9" },
  { text = "Craik (1943), The Nature of Explanation, Cambridge University Press" },
]
+++

## Textbook

### What a model is {#idea}

A **model** of the environment is anything an agent can use to predict how the environment will respond to its actions: given a state and an action, it produces a next state and a reward. Models come in two kinds.

- A **distribution model** gives every possible next state and reward with its probability, $p(s', r \mid s, a)$. Dynamic programming needs this kind ([[policy-evaluation]]).
- A **sample model** gives one next state and reward, drawn with the right probabilities. A simulator of a game of dice is a sample model: it is easy to roll the dice, and tedious to list every outcome with its probability.

A distribution model can always produce samples; a sample model can only estimate the distribution. A model may be given in advance, like the rules of a game, or **learned** from experience, by recording what happened after each action. Either way, a model lets the agent produce *simulated experience*: transitions that did not happen, but could have.

### Planning {#planning}

**Planning** is any computation that takes a model and produces or improves a policy. In reinforcement learning it usually means *state-space planning*: searching through states and actions for a good policy, by computing value functions as intermediate steps, and computing those values with updates applied to simulated experience. Seen this way, planning and learning are the same activity with different inputs ([[model-based-free]]):

$$\text{model} \;\longrightarrow\; \text{simulated experience} \;\xrightarrow{\ \text{updates}\ }\; \text{values} \;\longrightarrow\; \text{policy}. \label{eq-planning}$$

Learning methods use real experience; planning methods use simulated experience. Many ideas transfer between them, and any learning method can be turned into a planning method by feeding it the model's output. The simplest example turns [[q-learning]] into a planner:

::: algorithm {#alg-qplan} Random-sample one-step tabular Q-planning
Input: a sample model, a step size $\alp$ and a discount $\gam$
Repeat forever:
  Pick a state $S$ and an action $A$ at random
  Ask the sample model for a next state $S'$ and reward $\rew{R}$
  $\val{Q(S, A)} \leftarrow \val{Q(S, A)} + \alp\,[\rew{R} + \gam \max_a \val{Q(S', a)} - \val{Q(S, A)}]$
:::

Under the same conditions as Q-learning, this converges to the optimal values of the model. Planning in small steps like this has two advantages: it can be interrupted at any moment with a useful answer, and it mixes easily with acting and learning, which is the idea of [[dyna-q]].

### Expected and sample updates {#updates}

An update can **average** over all possible next states, using a distribution model, or use one **sample**. For action values, the two versions are

$$\val{Q(s, a)} \leftarrow \sum_{s', r} p(s', r \mid s, a)\,\Big[\rew{r} + \gam \max_{a'} \val{Q(s', a')}\Big] \qquad \text{and} \qquad \val{Q(s, a)} \leftarrow \val{Q(s, a)} + \alp\,\Big[\rew{R} + \gam \max_{a'} \val{Q(S', a')} - \val{Q(s, a)}\Big]. \label{eq-updates}$$

The expected update is exact (given the successors' values) but costs as many computations as there are successors, the **branching factor** $b$. The sample update costs one computation, is noisy, and needs many samples to average the noise away. Which is better for a fixed amount of computation? If the successors' values are already right, an expected update removes the whole error after $b$ computations; $t$ sample updates with step sizes $1/t$ leave an error of $\sqrt{(b-1)/(bt)}$ of the original.

::: figure {#fig-updates}
{{expected-vs-sample}}
Error left in one estimate after expected and sample updates, against the number of computations, for branching factors 2, 10, 100 and 1000; the estimate starts with an error of 1 and the successors' values are exact. Worked out exactly. After Sutton & Barto, Figure 8.7.
:::

For large $b$, a small fraction of the expected update's computation already removes most of the error with samples (\ref{fig-updates}): with $b = 1000$, a tenth of the computation leaves about 10% of the error. In real problems the successors' values are themselves estimates being improved by other updates, which favors samples further: a sample update spreads computation more evenly and its values are fresher. Expected updates win when $b$ is small or computation is cheap relative to precision.

### Where to spend the updates {#distribution}

Dynamic programming sweeps the whole state space, giving every state the same attention. Most states may never matter: they are unreachable, or reached only by poor policies. Two ways of focusing computation follow.

- **Trajectory sampling**: simulate episodes with the current policy, and update the states and actions they visit, so that the updates follow the on-policy distribution. Early in planning this focuses on the states that matter and gives faster progress; in the long run, it can neglect states whose values would still change.
- **Real-time dynamic programming** (Barto, Bradtke & Singh, 1995) does value-iteration updates only on the states visited in real or simulated trajectories, and for many problems finds an optimal policy on the relevant states while ignoring large parts of the state space.

Within a fixed set of states, the order of the updates matters too: updates that change nothing waste computation. [[prioritized-sweeping]] works backward from the states whose values just changed.

### Planning at decision time {#decision-time}

So far planning improves a table of values or a policy in the background, for all states. **Decision-time planning** instead plans for the current state only, when an action must be chosen, and usually discards the result afterward.

- **Heuristic search** looks ahead from the current state through a tree of possible continuations, evaluates the leaves with a value function, and backs the values up to choose an action. Deeper search compensates for an imperfect value function.
- **Rollout algorithms** estimate each action's value by averaging the returns of many simulated episodes that start with that action and then follow a fixed rollout policy; acting greedily on those estimates improves on the rollout policy (Tesauro & Galperin, 1997).
- **Monte Carlo tree search** grows a search tree selectively, using rollouts to evaluate new nodes and statistics in the tree to decide where to look next. It is at the heart of game-playing programs such as AlphaGo and AlphaZero ([[mcts]]).

Background planning pays off when the same states recur and computation can be spread over time; decision-time planning when fast responses are not needed and the current state deserves all the computation, as in games.

### Historical remarks {#history}

The idea that an agent can carry a small-scale model of the world in its head, and use it to try out alternatives before acting, goes back to Craik (1943). The view of planning as learning from simulated experience, and its integration with acting in the Dyna architecture, is due to Sutton (1990). Real-time dynamic programming is due to Barto, Bradtke and Singh (1995); rollout algorithms in this form to Tesauro and Galperin (1997).

## Card

### Idea

A model answers “what would happen if I did this here?”. Planning is learning from those answers instead of from the world: the same updates as learning, fed with imagined experience. A learned model turns every real step into many imagined ones.

::: analogy
A chess player thinking before moving: the rules of the game are the model, and the moves tried in the head are simulated experience. Nothing on the board changes until the decision is made.
:::

### Two kinds of model {#kinds}

| | gives | needed by |
| --- | --- | --- |
| distribution model | every next state and reward, with its probability | dynamic programming, expected updates |
| sample model | one next state and reward, drawn at random | rollouts, sample updates, Dyna |

### Planning as learning {#formula}

$$\text{model} \;\to\; \text{simulated experience} \;\to\; \text{values} \;\to\; \text{policy}$$

Real experience can improve the model (model learning) and the values directly (direct RL); simulated experience improves the values (planning). See [[dyna-q]].

### Why it matters {#why}

- Real experience can be expensive or slow; computation is often cheap. A model turns each real step into many updates. [See it](lab:dyna-maze)
- Sample updates usually beat expected updates for the same computation when the branching factor is large.
- Focusing planning (trajectory sampling, [[prioritized-sweeping]], search from the current state) beats uniform sweeps.

### Pitfalls

- Trusting a wrong model: planning amplifies its errors ([[dyna-q-plus]]).
- Planning on states that never matter: uniform sweeps waste most of their effort in large problems.

### Check yourself {#check}

::: question
What is the difference between a distribution model and a sample model, and which one does dynamic programming need?
---
A distribution model gives every possible outcome with its probability; a sample model gives one outcome drawn at random. Dynamic programming needs a distribution model, because its updates average over all outcomes.
:::

::: question
How does random-sample Q-planning differ from Q-learning?
---
Only in where its transitions come from: a model queried at random state–action pairs instead of the real environment. The update is the same.
:::

::: question
With a branching factor of 1000, why might sample updates beat expected updates for the same computation?
---
One expected update costs 1000 computations. In the same budget, 1000 sample updates can each be spent on a different state, and even a hundred samples on one state already remove about 90% of its error.
:::
