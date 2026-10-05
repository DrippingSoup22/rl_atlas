+++
summary = "Dynamic programming needs a perfect model and touches every state in every sweep. Real problems have no model and astronomically many states: the rest of reinforcement learning is about getting around both."
prereqs = ["value-iteration", "policy-iteration", "model-based-free"]
sources = [
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., §4.5, §4.7–4.8 and §8.5", url = "http://incompleteideas.net/book/the-book-2nd.html" },
  { text = "Bellman (1957), Dynamic Programming, Princeton University Press" },
  { text = "Papadimitriou & Tsitsiklis (1987), The complexity of Markov decision processes, Mathematics of Operations Research 12", url = "https://doi.org/10.1287/moor.12.3.441" },
  { text = "Barto, Bradtke & Singh (1995), Learning to act using real-time dynamic programming, Artificial Intelligence 72", url = "https://doi.org/10.1016/0004-3702(94)00011-O" },
  { text = "Tesauro (1995), Temporal difference learning and TD-Gammon, Communications of the ACM 38", url = "https://doi.org/10.1145/203330.203343" },
]
+++

## Textbook

### What dynamic programming needs {#needs}

The algorithms of this part, [[policy-evaluation]], [[policy-iteration]] and [[value-iteration]], solve finite Markov decision processes exactly. They need three things.

1. **A complete and accurate model**: the probability $p(s', r \mid s, a)$ of every outcome of every action in every state, used in every update.
2. **A sweep over all states**: each iteration updates every state, and each update sums over every action and every possible outcome.
3. **A table**: one stored value, and for policy iteration one stored action, per state.

Each requirement fails in most problems of interest. Most of the atlas is about doing without them.

### No model {#model}

For a board game, the rules are a perfect model of one move; but the opponent's reply is part of the dynamics, and it is not known. For a robot, the physics of its motors, its floor and its sensors can be simulated at best approximately. For a customer, a market or a patient, there is no model at all, only experience. Even when a model exists, it may be a *sample model*, a simulator that can produce one outcome at a time, rather than the *distribution model* of all outcomes with their probabilities that dynamic programming needs ([[model-based-free]]).

Learning from experience removes the first requirement. Monte Carlo methods estimate values from the returns of sampled episodes ([[mc-prediction]]); temporal-difference methods from sampled transitions ([[td0]]). Both keep the shape of the dynamic programming updates and replace the expectation over $p$ by samples.

### Too many states {#curse}

The number of states usually grows exponentially with the number of variables that describe a state. A problem described by $n$ variables, each with $m$ values, has $m^n$ states: ten variables with ten values each give $10^{10}$ states. Bellman called this the **curse of dimensionality**. Backgammon has about $10^{20}$ states, and Go about $2 \times 10^{170}$ legal positions. At a billion state updates per second, a single sweep through the states of backgammon would take thousands of years.

The blame lies with the problems, not with dynamic programming, which is efficient by the standards of exact methods. Searching the space of policies directly would mean comparing $|\mathcal{A}|^{|\mathcal{S}|}$ deterministic policies; dynamic programming finds an optimal one in time polynomial in $|\mathcal{S}|$ and $|\mathcal{A}|$. Linear programming solves MDPs in polynomial time too, with better worst-case guarantees, but in practice it runs out of steam on problems far smaller than those dynamic programming handles. In complexity terms, solving an MDP is P-complete: polynomial, but hard to parallelize well (Papadimitriou & Tsitsiklis, 1987). Problems with millions of states are within reach of an ordinary computer.

### Asynchronous dynamic programming {#async}

Nothing forces the updates to march through the states in order. **Asynchronous** algorithms update one state's value at a time, in place, picking the states in whatever order they like and using the latest values of the others: a busy region may get dozens of updates while a quiet corner waits. The one condition for convergence is that no state is abandoned for good; every state must keep being updated now and then. Within that rule the algorithm decides where the computation goes, which gives two kinds of freedom:

- **Focus.** Updates can go where they matter most: to states whose values are changing, as in [[prioritized-sweeping]], or to states that are relevant to the task at hand. In a problem where most states are never visited under good policies, sweeping them is wasted work.
- **Interleaving with experience.** An agent can run updates on the states it is actually visiting, while it visits them. Real-time dynamic programming (Barto, Bradtke & Singh, 1995) updates only the states along the agent's trajectories, and under reasonable conditions converges to optimal behavior on the relevant states without ever visiting most of the others.

### Sample updates against expected updates {#sample}

An expected update looks at every possible next state: with a branching factor $b$, the number of possible next states, it costs about $b$ times as much as an update from a single sampled next state. Which is better for a fixed amount of computation? An expected update removes all the sampling error of its target at once; $b$ sample updates, each with a step size like $1/n$, reduce the error almost as well when $b$ is large, and long before the expected update is finished, a fraction of them has already done most of the work. For large branching factors, many sample updates generally beat a few expected ones (Sutton & Barto, §8.5). This is one reason why methods built on samples are preferred even when a model is available ([[models]]).

### What the rest of the atlas changes {#later}

Each later part removes one of dynamic programming's requirements, and each brings a new difficulty:

| Requirement | Replaced by | Where | New difficulty |
| --- | --- | --- | --- |
| the model, in every update | sampled episodes or transitions | [[mc-prediction]], [[td0]], [[sarsa]], [[q-learning]] | exploration; noisy targets |
| sweeps over all states | updates along experienced trajectories | the same, and [[dyna-q]] | some states rarely visited |
| one expected backup per state | many cheap sample backups | [[models]], [[dyna-q]] | variance |
| a table of values | a parameterized function that generalizes | [[why-approximate]], [[semi-gradient-td]], [[dqn]] | instability ([[deadly-triad]]) |

TD-Gammon, which learned backgammon at a world-class level, combined all of these: values learned from sampled self-play games, by temporal-difference updates, stored in a neural network rather than a table of $10^{20}$ entries (Tesauro, 1995).

### Historical remarks {#history}

Bellman (1957) coined the phrase *curse of dimensionality* in the preface of the book that introduced dynamic programming. Papadimitriou and Tsitsiklis (1987) classified the complexity of solving MDPs. Asynchronous dynamic programming was analyzed by Bertsekas and Tsitsiklis in the 1980s, and Barto, Bradtke and Singh (1995) connected it to learning in real time. Sutton and Barto (§8.5) analyze the trade-off between expected and sample updates.

## Card

### Idea

Dynamic programming solves an MDP exactly, but only with a perfect model, and only by touching every state in every sweep. Real problems have no model and far too many states. Learning from samples removes the need for a model; approximation removes the need for a table.

::: analogy
A perfect road atlas of the country lets you compute the best route between any two towns, if you have the atlas, and the time to check every town. Most travelers have neither: they learn the roads by driving them, and remember rules of thumb instead of every town.
:::

### In numbers {#numbers}

| | |
| --- | --- |
| one sweep | every state × every action × every outcome |
| states of backgammon | about $10^{20}$ |
| legal positions of Go | about $2 \times 10^{170}$ |
| deterministic policies | $|\mathcal{A}|^{|\mathcal{S}|}$: what DP avoids enumerating |

### Why it matters {#why}

It explains the shape of the rest of the atlas. Monte Carlo and TD methods drop the model; planning methods use samples from a model; function approximation drops the table.

### Pitfalls

- Blaming dynamic programming for the curse of dimensionality: the number of states is the problem's; DP is polynomial in it.
- Assuming a simulator is a model in the DP sense: DP needs the probabilities of all outcomes, not one sample at a time.
- Sweeping states that never matter: asynchronous updates can focus on the relevant ones.

### Check yourself {#check}

::: question
A world is described by 20 yes-or-no variables. How many states does it have?
---
$2^{20}$, about a million: still within reach of dynamic programming. With 60 such variables, $2^{60} \approx 10^{18}$, it is not.
:::

::: question
What does asynchronous dynamic programming give up, and what does it gain?
---
It gives up the orderly sweep. It gains the freedom to update states in any order and as often as useful, focusing on relevant states or on the ones the agent is visiting, while still converging if every state keeps being updated.
:::
