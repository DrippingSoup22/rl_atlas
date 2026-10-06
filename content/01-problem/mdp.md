+++
summary = "States, actions, rewards and the odds of what happens next: the formal description of the problem, and the promise that the present is enough to predict the future."
prereqs = ["agent-environment", "state-action-reward"]
sources = [
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., §3.1 and Example 3.3", url = "http://incompleteideas.net/book/the-book-2nd.html" },
  { text = "Bellman (1957), A Markovian decision process, Journal of Mathematics and Mechanics 6" },
  { text = "Howard (1960), Dynamic Programming and Markov Processes, MIT Press" },
  { text = "Kaelbling, Littman & Cassandra (1998), Planning and acting in partially observable stochastic domains, Artificial Intelligence 101", url = "https://doi.org/10.1016/S0004-3702(98)00023-X" },
]

[story]
scene = "mdp"
mdp = "robot"
formula = '\step{1}{p(s^\prime, \rew{r} \mid s, a) = \Pr\{S_{t+1} = s^\prime,\ \rew{R_{t+1}} = \rew{r} \mid S_t = s,\ A_t = a\}}'
+++

## Story

::: step {level = 1}
A small robot collects empty cans in an office. For deciding what to do, only one thing about its situation matters: whether its battery is **high** or **low**. These are its two **states**.
:::

::: step {level = 2}
In each state it can choose an **action**, drawn as a dot: **search** for cans, or **wait** for someone to bring one. When the battery is low it can also go and **recharge**.
:::

::: step {level = 3}
Outcomes are uncertain. Searching with a high battery leaves it high with probability 0.7 and drains it to low with probability 0.3. An arrow leads from each action to every place it may end up.
:::

::: step {level = 4}
Each outcome also brings a **reward**: the cans collected, $\rew{+2}$ when searching and $\rew{+1}$ when waiting, and $\rew{-3}$ if the battery runs flat during a search and someone has to carry the robot back.
:::

::: step {level = 4, highlight = "low-search", table = true, formula = 1}
Everything about this environment fits in one table, its **dynamics** $p(s', \rew{r} \mid s, a)$: for every state and action, the probability of each next state and reward. The outcomes of an action always add up to 1: searching with a low battery gives $0.6 + 0.4$.
:::

::: step {level = 4, walk = true}
Now watch the robot act at random. Where it goes next depends only on where it is and what it does now, not on how it got there. That is the **Markov property**: the present state is enough.
:::

::: step {level = 4, walk = true, formula = 1}
States, actions, rewards and dynamics together form a **Markov decision process**, an MDP. It is the formal description of the problem behind every method in the atlas. The agent usually does not know $p$; it only lives through samples of it.
:::

## Textbook

### Finite Markov decision processes {#definition}

The agent–environment interaction ([[agent-environment]]) is formalized as a Markov decision process. In the finite case it is defined as follows.

::: definition {#def-mdp} Finite Markov decision process
A finite MDP consists of a finite set of states $\mathcal{S}$, for each state a finite set of actions $\mathcal{A}(s)$, a finite set of rewards $\mathcal{R} \subset \mathbb{R}$, and a **dynamics function** that gives, for every state $s$ and action $a$, the probability of each next state $s'$ and reward $r$:
$$p(s', \rew{r} \mid s, a) = \Pr\{S_{t+1} = s',\ \rew{R_{t+1}} = \rew{r} \mid S_t = s,\ A_t = a\}. \label{dynamics}$$
For every $s$ and $a$ these probabilities add up to one: $\sum_{s' \in \mathcal{S}} \sum_{r \in \mathcal{R}} p(s', \rew{r} \mid s, a) = 1$.
:::

Together with a discount rate $\gam$ ([[discount]]), an MDP specifies a reinforcement learning problem completely. The function $p$ is the **model** of the environment. Dynamic programming assumes it is known ([[policy-evaluation]]); most of the atlas assumes it is not, and learns from samples of it instead ([[model-based-free]]).

The finite case keeps the mathematics simple, but nothing essential depends on it. With continuous states or actions, such as a car's position and speed or a steering angle, sums over $s'$ become integrals and $p$ becomes a density; the definitions, the Markov property and the Bellman equations carry over, and the hard part becomes representing values and policies, which is what Parts 8 to 11 are about.

### The Markov property {#markov}

Definition \ref{def-mdp} contains a strong assumption: the probabilities in \ref{dynamics} depend only on the current state and action. In full,

$$\Pr\{S_{t+1}, \rew{R_{t+1}} \mid S_t, A_t, \rew{R_t}, S_{t-1}, A_{t-1}, \ldots, S_0, A_0\} = \Pr\{S_{t+1}, \rew{R_{t+1}} \mid S_t, A_t\}. \label{markov-eq}$$

Once the present state and action are known, the earlier history adds nothing to the prediction of what happens next. This is the **Markov property**. It is a property of the *state*, not of the world: a state has it when it includes every aspect of the past that makes a difference for the future.

A chess position nearly has it, but not quite: whether castling is still allowed, whether a pawn may be captured en passant, and how often the position has occurred before (a threefold repetition can end the game) all depend on the history. The board alone is therefore not a Markov state, while the board together with these few facts is. A single photograph of a moving ball is not Markov, because it shows the position but not the velocity; position and velocity together are.

The Markov property is what makes values well defined as functions of the state alone, and what lets the Bellman equations relate the value of a state to the values of its successors ([[bellman]]).

### Quantities derived from the dynamics {#derived}

Everything else about the environment can be computed from $p$. The **state-transition probabilities** forget the reward,

$$p(s' \mid s, a) = \sum_{r \in \mathcal{R}} p(s', \rew{r} \mid s, a), \label{transition}$$

the **expected reward** of a state–action pair averages over all outcomes,

$$r(s, a) = \mathbb{E}[\rew{R_{t+1}} \mid S_t = s, A_t = a] = \sum_{r \in \mathcal{R}} \rew{r} \sum_{s' \in \mathcal{S}} p(s', \rew{r} \mid s, a), \label{expected-reward}$$

and the expected reward of a particular transition is $r(s, a, s') = \sum_{r} \rew{r}\, p(s', \rew{r} \mid s, a) / p(s' \mid s, a)$. For computing values, $p(s' \mid s, a)$ and $r(s, a)$ are all that is needed; the full joint distribution matters only when rewards and next states are correlated in ways that must be reproduced, as when simulating the environment.

### Example: the recycling robot {#robot}

::: example {#ex-robot} The recycling robot (after Sutton & Barto, Example 3.3)
A mobile robot collects empty cans in an office. Its state is the charge of its battery, $\mathcal{S} = \{\text{high}, \text{low}\}$. In each state it can **search** for cans or **wait** for someone to bring one; with a low battery it can also **recharge**, which brings the battery back to high. Searching with a high battery leaves it high with probability 0.7 and lowers it with probability 0.3; searching with a low battery leaves it low with probability 0.6, and with probability 0.4 drains it completely, in which case the robot is rescued and recharged, which costs $\rew{-3}$. Searching collects $\rew{+2}$ cans on average, waiting $\rew{+1}$, recharging nothing. Waiting never changes the battery.
:::

Sutton and Barto keep the probabilities and rewards symbolic; the atlas fixes numbers so that values can be computed exactly ([[optimality]]). The whole dynamics fits in a table:

| $s$ | $a$ | $s'$ | $\rew{r}$ | $p(s', \rew{r} \mid s, a)$ |
| --- | --- | --- | --- | --- |
| high | search | high | $\rew{+2}$ | 0.7 |
| high | search | low | $\rew{+2}$ | 0.3 |
| high | wait | high | $\rew{+1}$ | 1 |
| low | search | low | $\rew{+2}$ | 0.6 |
| low | search | high | $\rew{-3}$ | 0.4 |
| low | wait | low | $\rew{+1}$ | 1 |
| low | recharge | high | 0 | 1 |

::: figure {#fig-robot}
{{mdp-graph robot}}
The transition graph of the recycling robot. Large circles are states, small dots are actions; each arrow from an action is labeled with the probability of that outcome and its reward.
:::

\ref{fig-robot} draws the same table as a **transition graph**. It has a node for each state and a smaller node for each state–action pair. An edge from a state to one of its action nodes stands for choosing that action; the edges from an action node to the possible next states carry the probabilities and rewards, and the probabilities leaving any action node add up to 1.

### When the state is not Markov {#beyond}

In many real problems the agent does not observe a Markov state. A robot sees through cameras with a limited view; a poker player does not see the other hands. Such problems are **partially observable**: the agent receives observations $O_t$ that depend on a hidden state. They can be formalized as partially observable MDPs (Kaelbling, Littman & Cassandra, 1998), whose exact solution is far harder.

The practical remedy is to build an approximately Markov state from the history of observations: the last few frames of a video game, a running summary computed by a recurrent network, or, in the theory, the probability distribution over hidden states given everything seen so far, the *belief state*. The methods of the atlas assume a Markov state; many still work when the property holds only approximately, but their guarantees no longer apply.

### Historical remarks {#history}

Markov decision processes come from optimal control and operations research. The Markov property is named after Andrey Markov, who studied chains of random events whose next step depends only on the present one (1906). Bellman (1957) formulated the discrete stochastic version and the dynamic programming methods that solve it; Howard (1960) introduced policy iteration. Reinforcement learning took the framework over in the 1980s, with the difference that the dynamics are not assumed to be known. Sutton and Barto's account in their Chapter 3 is the one followed here.

## Card

### Idea

A **Markov decision process** describes the problem: states, actions, rewards, and the dynamics $p(s', \rew{r} \mid s, a)$, the odds of each next state and reward after each action. The **Markov property** says the present state is enough: knowing how the agent got there adds nothing.

::: analogy
A board game whose next move depends only on the board, not on the moves that led to it. Look at the board, and you know everything that matters.
:::

### In symbols {#formula}

$$p(s', \rew{r} \mid s, a) = \Pr\{S_{t+1} = s',\ \rew{R_{t+1}} = \rew{r} \mid S_t = s,\ A_t = a\}, \qquad \sum_{s', r} p(s', \rew{r} \mid s, a) = 1$$

### The pieces {#pieces}

| | in the recycling robot |
| --- | --- |
| states $\mathcal{S}$ | battery high, battery low |
| actions $\mathcal{A}(s)$ | search, wait, recharge (only when low) |
| rewards | $\rew{+2}$ search, $\rew{+1}$ wait, $\rew{-3}$ rescue |
| dynamics $p$ | e.g. search when high: stays high 0.7, drops to low 0.3 |

### Why it matters {#why}

Value functions, Bellman equations and optimal policies are all defined on an MDP. Model-based methods use $p$ directly ([[policy-evaluation]]); model-free methods learn from samples of it ([[td0]]).

### Pitfalls

- A state that leaves out what matters is not Markov: one video frame shows where a ball is, not where it goes.
- The agent rarely knows $p$. Writing an MDP down describes the problem, not what the agent knows.
- The probabilities of an action's outcomes must add up to 1.

### Check yourself {#check}

::: question
Is the position of the pieces alone a Markov state in chess?
---
Almost, but not quite: castling rights, en passant and repetitions depend on the history. Adding those few facts makes it Markov.
:::

::: question
In the robot, what is the expected reward of searching with a low battery?
---
$0.6 \times 2 + 0.4 \times (-3) = 0$. Searching on a low battery earns nothing on average.
:::

::: question
What does the Markov property buy us?
---
Values and policies can depend on the current state alone, and the value of a state can be written in terms of the values of the states that follow it.
:::
