+++
summary = "The agent's way of choosing: for every state, a probability for every action. It is the thing reinforcement learning tries to improve."
story_in = ["value-functions", "policy-iteration"]
prereqs = ["agent-environment", "mdp"]
sources = [
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., §3.5", url = "http://incompleteideas.net/book/the-book-2nd.html" },
  { text = "Puterman (1994), Markov Decision Processes: Discrete Stochastic Dynamic Programming, Wiley, §6.2" },
  { text = "OpenAI (2018), Spinning Up in Deep RL, Part 1: Key Concepts in RL (policies)", url = "https://spinningup.openai.com/en/latest/spinningup/rl_intro.html" },
]
+++

## Textbook

### Definition {#definition}

A **policy** is a rule for choosing actions. Formally it is a mapping from states to probabilities of selecting each possible action,

$$\pol{\pi(a \mid s)} = \Pr\{A_t = a \mid S_t = s\}, \qquad \sum_{a \in \mathcal{A}(s)} \pol{\pi(a \mid s)} = 1, \label{policy}$$

for every state $s$. A **deterministic** policy picks one action in each state and is written $\pol{\pi(s)}$; it is the special case in which one action has probability 1. A policy that gives every action some positive probability is called *soft*.

Reinforcement learning methods specify how the agent's policy changes with experience. Everything else in the atlas, values, models and gradients, exists in the service of this one object.

### A policy and an MDP make a process {#process}

Once a policy is fixed, the interaction with an MDP ([[mdp]]) becomes a well-defined random process. The probability of an episode is a product of factors, alternately chosen by the agent and by the environment:

$$\Pr\{S_0, A_0, \rew{R_1}, S_1, \ldots, S_T\} = \mu(S_0) \prod_{t=0}^{T-1} \pol{\pi(A_t \mid S_t)}\; p(S_{t+1}, \rew{R_{t+1}} \mid S_t, A_t), \label{trajectory}$$

where $\mu$ is the distribution of starting states. The policy controls the action factors and nothing else. Under a fixed policy the sequence of states is a Markov chain, with transition probabilities

$$p_\pi(s' \mid s) = \sum_{a} \pol{\pi(a \mid s)}\, p(s' \mid s, a). \label{chain}$$

This is why a fixed policy can be *evaluated*: its expected returns are well-defined quantities of that chain ([[value-functions]]).

### Which policies are needed {#which}

A policy could in principle depend on the whole history of the interaction and on the time step, not only on the current state. For the problems in the atlas this generality is unnecessary.

::: theorem {#thm-stationary} Puterman, 1994, §6.2
In a finite MDP with discount $\gam < 1$, there is an optimal policy that is **stationary** (it does not depend on $t$), **Markov** (it depends only on the current state) and **deterministic**.
:::

The Markov property of the state is what makes the history irrelevant, and the infinite, discounted horizon is what makes time irrelevant: from every state, the future looks the same at every step. In a problem with a fixed, finite number of steps this is no longer true, and the best action can depend on how many steps remain. In [[be-the-agent]], with only one press left, collecting $\rew{+2}$ beats a forward step that pays nothing, unless the agent is already in the last room.

Stochastic policies are nevertheless useful, for four reasons: to explore ([[explore-exploit]]); against an adversary, where being predictable is exploitable, as in rock–paper–scissors; under partial observability, where randomizing can beat every deterministic choice; and for policy-gradient methods, which need the policy to change smoothly with its parameters ([[why-policy]]).

### Representing a policy {#representing}

There are three common ways to write a policy down.

- **A table** with one probability for each state and action. Possible only when both are few.
- **Derived from action values.** The greedy policy takes $\operatorname*{arg\,max}_a \val{Q(s,a)}$; ε-greedy takes it with probability $1-\eps$ and a uniformly random action otherwise ([[epsilon-greedy]]); the **softmax** or Boltzmann policy chooses with probabilities $\pol{\pi(a \mid s)} = e^{\val{Q(s,a)}/\tau} / \sum_{b} e^{\val{Q(s,b)}/\tau}$, where the temperature $\tau > 0$ sets how sharply it prefers the best action.
- **Parameterized directly**, as $\pol{\pi(a \mid s, \boldsymbol\theta)}$ with weights $\boldsymbol\theta$: a softmax over learned preferences for discrete actions, or a Gaussian whose mean and spread are computed from the state for continuous actions ([[policy-parameterization]]).

Value-based methods use the second form: they learn values, and the policy follows from them. Policy-gradient methods use the third ([[reinforce]]).

### Example: two policies on the gridworld {#example}

In the $5 \times 5$ gridworld of Sutton and Barto (Example 3.5), the **random policy** chooses each of the four directions with probability $1/4$ in every cell. \ref{fig-optimal} shows a very different policy: in each cell, the moves that are best in the long run, found by the methods of [[optimality]]. Where several arrows appear, any mixture of them is equally good.

::: figure {#fig-optimal}
{{gridworld optimal}}
The gridworld (left), its optimal values (middle) and an optimal policy (right): from every cell, head for $A$, whose moves all jump to $A'$ with $\rew{+10}$. Computed by the Lab with value iteration, $\gam = 0.9$; after Sutton & Barto, Figure 3.5.
:::

## Card

### Idea

A **policy** $\pol{\pi(a \mid s)}$ is the agent's way of choosing: for every state, a probability for every action. Deterministic policies pick one action per state; soft ones keep some chance for every action. Learning to act means improving the policy.

::: analogy
A recipe that says, in every situation, what to do, or how likely you are to do each thing.
:::

### In symbols {#formula}

$$\pol{\pi(a \mid s)} = \Pr\{A_t = a \mid S_t = s\}, \qquad \sum_a \pol{\pi(a \mid s)} = 1$$

### Kinds {#kinds}

| Policy | Chooses |
| --- | --- |
| deterministic | one action per state, $\pol{\pi(s)}$ |
| greedy | the action with the highest value |
| ε-greedy | greedy, except a random action with probability $\eps$ |
| softmax | each action with probability $\propto e^{\val{Q}/\tau}$ |
| Gaussian | a real number around a learned mean (continuous actions) |

### Why it matters {#why}

Values are always the values *of a policy* ([[value-functions]]); control methods improve a policy ([[prediction-control]]); policy-gradient methods learn it directly ([[reinforce]]).

### Pitfalls

- A greedy policy never explores: what looks best early can stay unchallenged forever ([[explore-exploit]]).
- In finite-horizon problems the best action can depend on the steps left; a policy of the state alone then needs the time in the state.
- The probabilities in each state must add up to 1.

### Check yourself {#check}

::: question
With four actions and $\eps = 0.2$, what probability does ε-greedy give the greedy action?
---
$1 - 0.2 + 0.2/4 = 0.85$: the greedy part, plus its share of the random part.
:::

::: question
Why can an optimal policy for a discounted MDP ignore the history?
---
The state is Markov, so the history adds nothing about the future; and with an infinite horizon, the future looks the same at every step.
:::
