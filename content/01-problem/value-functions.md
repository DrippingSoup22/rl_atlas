+++
summary = "How good it is to be in a state, or to take an action there, when you keep acting the same way: the expected return, as a number for every state."
prereqs = ["return", "policy", "mdp"]
sources = [
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., §3.5, Example 3.5 and Figure 3.2", url = "http://incompleteideas.net/book/the-book-2nd.html" },
  { text = "Baird (1994), Reinforcement learning in continuous time: advantage updating, Proceedings of the IEEE International Conference on Neural Networks" },
  { text = "OpenAI (2018), Spinning Up in Deep RL, Part 1: Key Concepts in RL (value functions, advantage)", url = "https://spinningup.openai.com/en/latest/spinningup/rl_intro.html" },
]

[story]
scene = "grid"
env = "gridworld"
gamma = 0.9
digits = 1
seed = 3
formula = '\begin{aligned} \step{1}{\val{v_\pi(s)}} &\step{1}{= \mathbb{E}_\pi[\,\rew{G_t} \mid S_t = s\,]} \\ \step{2}{\val{q_\pi(s,a)}} &\step{2}{= \mathbb{E}_\pi[\,\rew{G_t} \mid S_t = s,\ A_t = a\,]} \\ \step{3}{\err{A_\pi(s,a)}} &\step{3}{= \val{q_\pi(s,a)} - \val{v_\pi(s)}} \end{aligned}'
+++

## Story

::: step {agent = "none"}
**The gridworld** of Sutton and Barto. Each move goes one cell in its direction. A move into the edge costs $\rew{-1}$ and leaves the agent where it is. Every move from $A$ lands on $A'$ with $\rew{+10}$, every move from $B$ on $B'$ with $\rew{+5}$. The task never ends, and $\gam = 0.9$.
:::

::: step {policy = "random", agent = "none"}
Take the simplest policy: in every cell, each of the four directions with probability ¼. How good is each cell for an agent that wanders like this?
:::

::: step {policy = "random", focus = [0, 0], wander = "random"}
Watch it wander from the outlined cell. Each walk collects a different discounted return, and the note below keeps count. The average of many walks estimates how good the cell is.
:::

::: step {v = "random", agent = "none", formula = 1}
That average, taken over all possible walks, is the **value** of the state, $\val{v_\pi(s)}$: the expected return from $s$ when following $\pol{\pi}$. Here are all 25 values at once, computed exactly: blue is good, orange is bad.
:::

::: step {v = "random", values = true, agent = "none", formula = 1}
The numbers, as in the book's Figure 3.2. $A$ is the best cell but worth less than its $\rew{+10}$: from $A'$, by the bottom edge, the wandering agent often bumps into the wall. $B$ is worth more than its $\rew{+5}$: from $B'$ it can still stumble onto $A$ or $B$.
:::

::: step {q = "random", agent = "none", formula = 2}
The **action value** $\val{q_\pi(s,a)}$ is the expected return if the agent first takes $a$, then follows $\pol{\pi}$. Each cell now shows four triangles, one per move. Next to an edge, the move into the wall looks clearly worse.
:::

::: step {q = "advantage", agent = "none", formula = 3, range = 2}
The **advantage** $\err{A_\pi(s,a)} = \val{q_\pi(s,a)} - \val{v_\pi(s)}$ says how much better a move is than what the policy does on average in that cell. Blue triangles beat the average, orange ones fall short.
:::

::: step {q = "advantage", policy = "improved", agent = "none", formula = 3, range = 2}
Taking the best triangle in every cell gives a new policy, already far better than wandering. “Evaluate, then improve, then evaluate again” is the plan of [[policy-iteration]]; where it ends is the subject of [[optimality]].
:::

## Textbook

### State values {#v}

The return is random ([[return]]), so the quality of a state under a policy is measured by its expectation. The **state-value function** of a policy $\pol{\pi}$ gives, for every state, the expected return when starting there and following $\pol{\pi}$:

::: definition {#def-v} State-value function
$$\val{v_\pi(s)} = \mathbb{E}_\pi[\,\rew{G_t} \mid S_t = s\,] = \mathbb{E}_\pi\Big[\sum_{k=0}^{\infty} \gam^k\,\rew{R_{t+k+1}} \;\Big|\; S_t = s\Big] \qquad \text{for all } s \in \mathcal{S}. \label{v-def}$$
The value of a terminal state, if there is one, is always 0.
:::

The subscript $\pi$ matters: a value is always the value *of a policy*. The same state can be excellent under one policy and poor under another. The expectation is over everything random: the agent's choices, drawn from $\pol{\pi}$, and the environment's responses, drawn from $p$. Because the state is Markov ([[mdp]]) and the policy is stationary, the expectation does not depend on $t$.

### Action values {#q}

Choosing among actions requires a finer quantity: how good it is to take a particular action in a state.

::: definition {#def-q} Action-value function
$$\val{q_\pi(s,a)} = \mathbb{E}_\pi[\,\rew{G_t} \mid S_t = s,\ A_t = a\,], \label{q-def}$$
the expected return when starting in $s$, taking action $a$, and following $\pol{\pi}$ afterwards.
:::

The first action is fixed, whatever the policy would have done; from the next step on, the policy takes over. With action values, a better action can be read off directly, without a model of the environment, which is why most model-free control methods learn $\val{q}$ ([[sarsa]], [[q-learning]]).

### How the two relate {#relations}

The two functions determine each other, given the policy and the dynamics. A state's value is the average of its action values, weighted by how often the policy chooses each action:

$$\val{v_\pi(s)} = \sum_{a} \pol{\pi(a \mid s)}\, \val{q_\pi(s,a)}. \label{v-from-q}$$

An action's value is one step of the environment followed by the value of wherever it leads. Using $\rew{G_t} = \rew{R_{t+1}} + \gam\,\rew{G_{t+1}}$ and the Markov property,

$$\val{q_\pi(s,a)} = \mathbb{E}[\,\rew{R_{t+1}} + \gam\,\val{v_\pi(S_{t+1})} \mid S_t = s,\ A_t = a\,] = \sum_{s',\,r} p(s', \rew{r} \mid s, a)\,\big[\,\rew{r} + \gam\,\val{v_\pi(s')}\,\big]. \label{q-from-v}$$

Substituting either relation into the other gives an equation in one function alone, the Bellman equation ([[bellman]]).

### The advantage {#advantage}

The **advantage** of an action compares it with the policy's own average in that state:

$$\err{A_\pi(s,a)} = \val{q_\pi(s,a)} - \val{v_\pi(s)}. \label{advantage-def}$$

A positive advantage means that taking $a$ once, and then going back to $\pol{\pi}$, does better than following $\pol{\pi}$ from the start.

::: lemma {#lem-zero} The advantage averages to zero
For every state $s$, $\sum_a \pol{\pi(a \mid s)}\, \err{A_\pi(s,a)} = 0$.
:::

::: proof
By \ref{v-from-q}, $\sum_a \pol{\pi(a \mid s)}\,\val{q_\pi(s,a)} - \val{v_\pi(s)} \sum_a \pol{\pi(a \mid s)} = \val{v_\pi(s)} - \val{v_\pi(s)} = 0$.
:::

So under its own policy, the advantage is centered: some actions are better than average and some worse. Shifting probability toward actions with positive advantage improves the policy; this is the policy improvement theorem ([[policy-improvement]]). The advantage also plays a central role in policy-gradient methods, where subtracting $\val{v_\pi}$ from the return reduces variance without introducing bias ([[baseline]], [[actor-critic]], [[gae]]). The name goes back to Baird (1994).

### Example: the gridworld {#example}

::: example {#ex-gridworld} Gridworld (Sutton & Barto, Example 3.5)
The cells of a $5 \times 5$ grid are the states; the actions move one cell up, down, left or right. An action that would leave the grid leaves the agent in place and gives $\rew{-1}$. Every action in cell $A$ moves the agent to $A'$ with $\rew{+10}$, every action in cell $B$ moves it to $B'$ with $\rew{+5}$. All other actions give 0. The task is continuing, with $\gam = 0.9$.
:::

::: figure {#fig-random}
{{gridworld random}}
The gridworld (left) and the state values of the equiprobable random policy (right), computed by the Lab by iterative policy evaluation. After Sutton & Barto, Figure 3.2.
:::

\ref{fig-random} shows $\val{v_\pi}$ for the policy that picks each direction with probability $1/4$. The values near the bottom edge are negative: there, random moves often run into the wall. $A$ is the best state, yet its value, 8.8, is less than its immediate reward: every action from $A$ leads to $A'$, whose value is $-1.3$, and indeed $10 + 0.9 \times (-1.3) \approx 8.8$, as \ref{q-from-v} requires. $B$, on the contrary, is worth more than its reward of 5, because $B'$ has a positive value: from $B'$, the expected cost of bumping into the edge is more than made up by the chance of wandering onto $A$ or $B$.

Action values follow from \ref{q-from-v}. In the center cell, where $\val{v_\pi} = 0.67$, the four moves lead to cells worth 2.25 (up), 0.36 (right), $-0.35$ (down) and 0.74 (left), so

| move | up | right | down | left |
| --- | --- | --- | --- | --- |
| $\val{q_\pi(s,a)} = 0.9\,\val{v_\pi(s')}$ | 2.03 | 0.32 | $-0.32$ | 0.66 |
| $\err{A_\pi(s,a)}$ | 1.35 | $-0.35$ | $-0.99$ | $-0.01$ |

The advantages, weighted by ¼ each, add up to zero, as \ref{lem-zero} says.

### Estimating values {#estimating}

The definitions say what values are, not how to obtain them. There are three routes, and the rest of the atlas follows all of them.

- **Average sampled returns.** By the law of large numbers, the average of the returns observed after visits to a state converges to its value. This is Monte Carlo prediction ([[mc-prediction]]), and it is what the wandering agent of the story does.
- **Solve the Bellman equations** with the model, as dynamic programming does ([[policy-evaluation]]). The values in \ref{fig-random} were computed this way.
- **Sample the Bellman equations**, combining the two: temporal-difference learning ([[td0]]).

When there are too many states for a table, values are approximated by a parameterized function $\hat v(s, \mathbf{w})$, and the question becomes which weights fit best ([[features]]).

### Rewards and values {#remarks}

Rewards are primary: they are what the environment gives and what defines the goal. Values are secondary, predictions of future reward, and they matter only because they lead to more reward. Yet decisions are made on values, because the action that is best in the long run is rarely the one with the best immediate reward. Most of the work in reinforcement learning is estimating values well.

## Card

### Idea

The **value** $\val{v_\pi(s)}$ of a state is the return you can expect from it if you keep following the policy $\pol{\pi}$. The **action value** $\val{q_\pi(s,a)}$ is the same, but you first take action $a$. The **advantage** $\err{A_\pi(s,a)} = \val{q_\pi} - \val{v_\pi}$ says how much better that action is than the policy's average.

::: analogy
The value of a chess position: not what you have captured so far, but how likely you are to win from here, given how you play.
:::

### In symbols {#formula}

$$\val{v_\pi(s)} = \mathbb{E}_\pi[\rew{G_t} \mid S_t = s], \qquad \val{q_\pi(s,a)} = \mathbb{E}_\pi[\rew{G_t} \mid S_t = s, A_t = a]$$

$$\val{v_\pi(s)} = \sum_a \pol{\pi(a \mid s)}\,\val{q_\pi(s,a)}, \qquad \val{q_\pi(s,a)} = \sum_{s', r} p(s', \rew{r} \mid s, a)\,[\rew{r} + \gam\,\val{v_\pi(s')}]$$

### In the gridworld {#example}

| | value under the random policy |
| --- | --- |
| cell $A$ | 8.8: less than its $\rew{+10}$, because $A'$ is worth $-1.3$ |
| cell $B$ | 5.3: more than its $\rew{+5}$, because $B'$ is worth 0.4 |
| bottom corners | about $-2$: the walls are close |

### Why it matters {#why}

Values turn a long-term goal into numbers attached to states, so that a good move can be recognized now. Every value-based method learns them ([[td0]], [[q-learning]]); policy-gradient methods use them to judge their own choices ([[actor-critic]]).

### Pitfalls

- A value belongs to a policy. Change the policy and every value changes.
- $\val{v}$ alone does not tell which action to take unless the model is known; $\val{q}$ does.
- The value of a terminal state is 0, whatever the table holds.

### Check yourself {#check}

::: question
Why is cell $A$ worth less than its reward of 10 under the random policy?
---
Every move from $A$ lands on $A'$, which is worth $-1.3$: it sits by the bottom edge, where random moves often hit the wall. $10 + 0.9 \times (-1.3) \approx 8.8$.
:::

::: question
Under its own policy, what is the average advantage of the actions in a state?
---
Zero. The policy's average of its action values is exactly the state's value.
:::

::: question
Which of $\val{v_\pi}$ and $\val{q_\pi}$ lets an agent without a model pick a better action?
---
$\val{q_\pi}$: compare the numbers and take the largest. With $\val{v_\pi}$ the agent would need to know where each action leads.
:::
