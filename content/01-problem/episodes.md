+++
summary = "Some tasks end and start again (a game, a maze); others go on forever (a thermostat). The difference decides how rewards are added up."
prereqs = ["agent-environment", "state-action-reward"]
sources = [
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., §3.3 and §3.4", url = "http://incompleteideas.net/book/the-book-2nd.html" },
  { text = "Pardo, Tavakoli, Levdik & Kormushev (2018), Time limits in reinforcement learning, Proceedings of the 35th International Conference on Machine Learning", url = "https://proceedings.mlr.press/v80/pardo18a.html" },
]
+++

## Textbook

### Episodic tasks {#episodic}

In many problems the interaction breaks naturally into separate pieces: a game is played to the end, a maze is solved, a robot arm completes a grasp. Each piece is an **episode**. It ends in a special **terminal state**, after which the environment is reset to a starting state, drawn from some fixed distribution, and a new episode begins independently of how the last one ended.

Tasks of this kind are **episodic**. The set of all states including the terminal one is written $\mathcal{S}^+$, and the nonterminal states $\mathcal{S}$. The time at which an episode ends, $T$, is a random variable that normally differs from episode to episode: a game can be won quickly or slowly. Within an episode the trajectory is

$$S_0,\ A_0,\ \rew{R_1},\ S_1,\ \ldots,\ S_{T-1},\ A_{T-1},\ \rew{R_T},\ S_T, \label{episode}$$

with $S_T$ terminal. Nothing happens after $S_T$: its value is zero by definition, and every update that bootstraps from a terminal state uses 0 in place of its value.

### Continuing tasks {#continuing}

Other problems never end: a thermostat keeps regulating, a process-control system keeps running, a trading agent keeps trading. These **continuing** tasks have $T = \infty$. They raise a difficulty that episodic tasks do not: the total reward over an infinite future can be infinite, so "maximize the total" is no longer a well-defined goal. Discounting solves this ([[discount]]); an alternative, the average reward per step, is used in some continuing problems.

### One notation for both {#unified}

The two cases can be written with one formula by treating the end of an episode as entry into an **absorbing state** that transitions only to itself and gives zero reward forever. After the end, the trajectory continues as $S_T, 0, S_T, 0, \ldots$, and summing to infinity gives the same total as summing to $T$. The return can then be written

$$\rew{G_t} = \sum_{k=t+1}^{T} \gam^{\,k-t-1}\, \rew{R_k}, \label{unified-return}$$

where either $T = \infty$ or $\gam = 1$ is allowed, but not both ([[return]]). This is the convention of Sutton and Barto, and of the atlas.

### Time limits are not terminal states {#time-limits}

In practice, episodes are often cut off after a fixed number of steps: a simulated robot is reset after 1000 steps whether or not anything has ended. Such a **truncation** is not a terminal state. The robot could have continued, and the state at the cut-off still has a value: the reward it would have collected afterwards.

The distinction matters for every method that bootstraps. At a true terminal state, the target is just the last reward. At a truncation, the target should still include the estimated value of the last state,

$$\text{terminal: } \rew{R_{t+1}} \qquad\qquad \text{truncated: } \rew{R_{t+1}} + \gam\,\val{V(S_{t+1})},$$

or else the agent learns that the world ends at the time limit, which it does not. Treating time-outs as terminal is a common source of subtle errors (Pardo et al., 2018). Modern environment interfaces such as Gymnasium report the two cases separately, as `terminated` and `truncated`, for this reason.

### Examples {#examples}

| Task | Kind | Ends when |
| --- | --- | --- |
| A game of chess | episodic | checkmate, stalemate or resignation |
| The cliff world | episodic | the agent reaches the goal |
| The 5 × 5 gridworld ([[value-functions]]) | continuing | never |
| Balancing a pole on a cart | either | episodic if the pole falling ends it; continuing if it is reset and the task goes on |
| Regulating the temperature of a building | continuing | never |

The same physical problem can be posed either way: the choice depends on what is to be optimized.

## Card

### Idea

An **episodic** task breaks into episodes that end in a terminal state and start afresh: a game, a maze. A **continuing** task goes on forever: a thermostat. In an episode, everything after the terminal state is worth 0; in a continuing task, discounting keeps the total finite.

### In symbols {#formula}

$$\rew{G_t} = \sum_{k=t+1}^{T} \gam^{\,k-t-1}\, \rew{R_k}, \qquad T = \infty \text{ or } \gam = 1, \text{ not both}$$

### Terminal or truncated? {#truncation}

| | the episode stops because | the target uses |
| --- | --- | --- |
| terminated | the task really ended | $\rew{R_{t+1}}$ alone |
| truncated | a time limit cut it off | $\rew{R_{t+1}} + \gam\,\val{V(S_{t+1})}$ |

### Why it matters {#why}

Monte Carlo methods need episodes that end: they wait for the return ([[mc-prediction]]). Temporal-difference methods work in both kinds of task, because they learn from each step ([[td0]]).

### Pitfalls

- Treating a time limit as the end of the world: the agent learns that states near the limit are worth nothing.
- Forgetting that the value of a terminal state is 0, and bootstrapping from whatever the table holds there.
- Using $\gam = 1$ in a continuing task: the total can grow without bound.

### Check yourself {#check}

::: question
A simulated robot is reset every 1000 steps. Is step 1000 a terminal state?
---
No. It is a truncation: the robot could have gone on, so the last state still has a value and the target should include it.
:::

::: question
Why can't Monte Carlo methods learn in a continuing task?
---
They need the full return, which only exists once an episode ends. In a continuing task it never does.
:::
