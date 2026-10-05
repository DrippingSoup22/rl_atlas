+++
summary = "The best any policy can do from each state, and the policies that do it: v*, q* and π*, tied together by the Bellman optimality equation."
prereqs = ["bellman", "value-functions", "policy"]
sources = [
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., §3.6, Examples 3.8 and 3.9, Figures 3.5 and 3.4", url = "http://incompleteideas.net/book/the-book-2nd.html" },
  { text = "Bellman (1957), Dynamic Programming, Princeton University Press" },
  { text = "Puterman (1994), Markov Decision Processes: Discrete Stochastic Dynamic Programming, Wiley, §6.2" },
]

[story]
scene = "grid"
env = "gridworld"
gamma = 0.9
digits = 1
seed = 5
formula = '\begin{aligned} \step{1}{\val{v_*(s)}} &\step{1}{= \max_\pi \val{v_\pi(s)}} \\ \step{4}{\val{v_*(s)}} &\step{4}{= \max_a \sum_{s^\prime\!,\,r} p(s^\prime\!, \rew{r} \mid s, a)\,\big[\,\rew{r} + \gam\,\val{v_*(s^\prime)}\,\big]} \\ \step{5}{\pol{\pi_*(s)}} &\step{5}{= \operatorname*{arg\,max}_a \val{q_*(s,a)}} \end{aligned}'

[story.numbers]
cycle = '\val{v_*(A)} = \rew{10} + \gam^5\,\rew{10} + \gam^{10}\,\rew{10} + \cdots = \frac{\rew{10}}{1 - 0.9^5} \approx \val{24.4}'
center = '\val{v_*(\text{center})} = \max\{\,0.9 \cdot \val{19.8},\ 0.9 \cdot \val{16.0},\ 0.9 \cdot \val{16.0},\ 0.9 \cdot \val{19.8}\,\} \approx \val{17.8}'
+++

## Story

::: step {v = "random", values = true, policy = "random", agent = "none"}
The values of the random policy, once more. Can some policy do better, and how much better? There is always a policy that does at least as well as every other one, in every cell at once: an **optimal policy**.
:::

::: step {v = "optimal", values = true, agent = "none", range = 25, formula = 1}
Its values are the **optimal values** $\val{v_*}$: the most any policy can expect from each cell. Every cell is worth more than before, and $A$ is worth 24.4.
:::

::: step {v = "optimal", values = true, policy = "optimal", agent = "none", range = 25, formula = 1}
An optimal policy heads for $A$ from everywhere. Where a cell shows two arrows, both moves are equally good.
:::

::: step {v = "optimal", values = true, policy = "optimal", focus = [0, 1], wander = "optimal", range = 25, formula = 1, numbers = "cycle"}
Why 24.4? From $A$ the agent jumps to $A'$ for $\rew{+10}$, walks four steps back up to $A$, and jumps again: $\rew{+10}$ every five steps, forever. Discounted, that adds up to $10/(1-0.9^5)$. Watch it go round.
:::

::: step {v = "optimal", values = true, focus = [2, 2], next = true, agent = "none", range = 25, formula = 4, numbers = "center"}
The optimal values satisfy a Bellman equation of their own, with a **max** in place of the policy's average: the value of a cell is that of its best move. In the center the best neighbors are worth 19.8, and $0.9 \times 19.8 \approx 17.8$.
:::

::: step {q = "optimal", agent = "none", range = 25, formula = 5}
With the optimal action values $\val{q_*}$, one triangle per move, acting optimally needs no model at all: in each cell, take the move with the bluest triangle. This is what [[q-learning]] learns.
:::

::: step {v = "optimal", values = true, policy = "optimal", agent = "none", range = 25, formula = 5}
Much of the atlas aims at these numbers. Dynamic programming computes them from the model ([[value-iteration]]), Q-learning learns them from experience, and when a table is too small for the world, function approximation estimates them ([[dp-limits]]).
:::

## Textbook

### Comparing policies {#order}

Value functions order policies. A policy $\pol{\pi}$ is **at least as good as** $\pol{\pi'}$ if its expected return is at least as large from every state:

$$\pol{\pi} \ge \pol{\pi'} \iff \val{v_\pi(s)} \ge \val{v_{\pi'}(s)} \quad \text{for all } s \in \mathcal{S}. \label{order-eq}$$

This is only a partial order: of two policies, each may be better in some states. It is a remarkable fact that, in a finite MDP, there is always a policy that is at least as good as every other policy in every state at once. Such a policy is **optimal**, written $\pol{\pi_*}$. There may be several, but they all share the same values.

### Optimal value functions {#values}

::: definition {#def-optimal} Optimal value functions
$$\val{v_*(s)} = \max_{\pi}\, \val{v_\pi(s)}, \qquad \val{q_*(s,a)} = \max_{\pi}\, \val{q_\pi(s,a)}, \qquad \text{for all } s \text{ and } a. \label{optimal}$$
:::

$\val{q_*(s,a)}$ is the expected return of taking $a$ in $s$ and behaving optimally afterwards. The two are related as action values and state values always are, with an optimal policy in the role of $\pol{\pi}$:

$$\val{q_*(s,a)} = \mathbb{E}[\,\rew{R_{t+1}} + \gam\,\val{v_*(S_{t+1})} \mid S_t = s,\ A_t = a\,], \qquad \val{v_*(s)} = \max_a \val{q_*(s,a)}. \label{relations}$$

The second relation holds because an optimal policy puts all its probability on actions with the highest value: averaging over such actions gives their maximum.

### The Bellman optimality equations {#equations}

Substituting each relation in \ref{relations} into the other gives an equation in one function alone. For $\val{v_*}$,

$$\val{v_*(s)} = \max_a \sum_{s',\,r} p(s', \rew{r} \mid s, a)\,\big[\,\rew{r} + \gam\,\val{v_*(s')}\,\big], \label{bellman-v}$$

and for $\val{q_*}$,

$$\val{q_*(s,a)} = \sum_{s',\,r} p(s', \rew{r} \mid s, a)\,\Big[\,\rew{r} + \gam \max_{a'} \val{q_*(s',a')}\Big]. \label{bellman-q}$$

These are the **Bellman optimality equations**. They say that the value of a state under an optimal policy equals the expected return of the best action from that state. Compared with the Bellman equations of a fixed policy ([[bellman]]), the average over the policy's actions is replaced by a maximum, which makes the system nonlinear. Their backup diagrams (\ref{fig-backup}) show the maximum as an arc.

::: figure {#fig-backup}
{{backup v-star}}
The backup diagram of $\val{v_*}$: the arc across the actions means that the best one is taken, not an average. After Sutton & Barto, Figure 3.4.
:::

For a finite MDP with $\gam < 1$, \ref{bellman-v} has a unique solution, $\val{v_*}$: the operator on its right-hand side is a $\gam$-contraction in the max norm, exactly as for action values ([[q-learning]]).

### From optimal values to an optimal policy {#greedy}

Once $\val{v_*}$ is known, finding an optimal policy is easy: in every state, choose an action that achieves the maximum in \ref{bellman-v}.

::: theorem {#thm-greedy} Greedy with respect to v* is optimal
Let $\pol{\pi}$ be any policy that, in every state, chooses only actions achieving the maximum in \ref{bellman-v}. Then $\val{v_\pi} = \val{v_*}$, so $\pol{\pi}$ is optimal.
:::

::: proof
Because $\pol{\pi}$ only takes maximizing actions, the Bellman equation of $\pol{\pi}$, applied to $\val{v_*}$, gives the same result as the right-hand side of \ref{bellman-v}: $\mathcal{T}_\pi \val{v_*} = \val{v_*}$. So $\val{v_*}$ is a fixed point of $\mathcal{T}_\pi$. That operator has only one fixed point, $\val{v_\pi}$ ([[bellman]]). Hence $\val{v_\pi} = \val{v_*}$.
:::

A one-step look ahead is enough because $\val{v_*}$ already accounts for all future consequences: an action that looks best for one step, by the value of where it leads, is best in the long run. With $\val{q_*}$ even the look ahead disappears, and no model is needed:

$$\pol{\pi_*(s)} = \operatorname*{arg\,max}_a \val{q_*(s,a)}. \label{argmax}$$

This is why so many methods learn action values: the best action can be read off directly.

### Example: the gridworld {#gridworld}

::: figure {#fig-optimal}
{{gridworld optimal}}
The gridworld (left), its optimal values $\val{v_*}$ (middle) and an optimal policy (right), computed by the Lab with value iteration, $\gam = 0.9$. After Sutton & Barto, Figure 3.5.
:::

\ref{fig-optimal} shows the solution of \ref{bellman-v} for the gridworld of [[value-functions]]. The value of $A$ can be checked by hand. An optimal agent in $A$ jumps to $A'$ for $\rew{+10}$, then walks four cells up, which costs nothing, and jumps again, collecting $\rew{+10}$ every five steps:

$$\val{v_*(A)} = 10 + \gam^5 \cdot 10 + \gam^{10} \cdot 10 + \cdots = \frac{10}{1 - \gam^5} = \frac{10}{1 - 0.59049} \approx 24.4.$$

Every other value follows from the equation. In the center cell all moves pay 0, and the best neighbors, above and to the left, are worth 19.8, so $\val{v_*(\text{center})} = 0.9 \times 19.8 \approx 17.8$. Both of those moves achieve the maximum, which is why the optimal policy shows two arrows there.

### Example: the recycling robot {#robot}

For the recycling robot ([[mdp]]) with $\gam = 0.9$, the equations \ref{bellman-v} for the two states read

$$\begin{aligned} \val{v_*(\text{h})} &= \max\big\{\, 2 + 0.9\,[\,0.7\,\val{v_*(\text{h})} + 0.3\,\val{v_*(\text{l})}\,],\ \ 1 + 0.9\,\val{v_*(\text{h})} \,\big\}, \\ \val{v_*(\text{l})} &= \max\big\{\, 0.6\,[\,2 + 0.9\,\val{v_*(\text{l})}\,] + 0.4\,[\,{-3} + 0.9\,\val{v_*(\text{h})}\,],\ \ 1 + 0.9\,\val{v_*(\text{l})},\ \ 0.9\,\val{v_*(\text{h})} \,\big\}, \end{aligned}$$

where the terms are search and wait in the high state, and search, wait and recharge in the low one. Their solution, computed by the Lab, is $\val{v_*(\text{high})} \approx 15.75$ and $\val{v_*(\text{low})} \approx 14.17$. With a high battery the robot should search (worth 15.75, against 15.17 for waiting); with a low one it should recharge (14.17, against 13.76 for waiting and 13.32 for searching). With other probabilities or rewards the answer changes; Sutton and Barto keep them symbolic for this reason (Example 3.9).

### Solving the optimality equations {#solving}

The optimality equations are nonlinear, because of the maximum, and in general have no closed-form solution. They can be solved by iteration, as in value iteration ([[value-iteration]]), by alternating evaluation and improvement, as in policy iteration ([[policy-iteration]]), or as a linear program.

Solving them exactly rests on three assumptions that are rarely all true: the dynamics are known; there is enough computation to handle every state; and the states are Markov. A game like backgammon has about $10^{20}$ states, far too many to visit, let alone sweep. Most of reinforcement learning can therefore be read as ways of solving the Bellman optimality equation approximately: from samples instead of the model, for the states that matter most, and with functions that generalize instead of tables ([[dp-limits]]).

### Historical remarks {#history}

Bellman's *principle of optimality* (1957) is the idea behind \ref{bellman-v}: whatever the first decision, the remaining decisions of an optimal policy must be optimal for the state that results. The existence of optimal policies for finite discounted MDPs, and the contraction arguments, are treated in Puterman (1994).

## Card

### Idea

$\val{v_*(s)}$ is the most any policy can expect from state $s$; $\val{q_*(s,a)}$ the most after first taking $a$. An optimal policy $\pol{\pi_*}$ achieves them everywhere at once. Once $\val{q_*}$ is known, acting optimally is easy: take the action with the highest value.

::: analogy
A sat-nav that knows, from every junction, the shortest time to your destination. To drive optimally, just take the exit that leads to the junction with the best time.
:::

### In symbols {#formula}

$$\val{v_*(s)} = \max_a \sum_{s', r} p(s', \rew{r} \mid s, a)\,\big[\rew{r} + \gam\,\val{v_*(s')}\big]$$

$$\val{q_*(s,a)} = \sum_{s', r} p(s', \rew{r} \mid s, a)\,\big[\rew{r} + \gam \max_{a'} \val{q_*(s',a')}\big], \qquad \pol{\pi_*(s)} = \operatorname*{arg\,max}_a \val{q_*(s,a)}$$

### Backup diagram {#backup}

{{backup q-star}}

The arcs mean “take the best”: a maximum instead of an average.

### Why it matters {#why}

It is the target of control: [[value-iteration]] computes $\val{v_*}$, [[q-learning]] learns $\val{q_*}$, and every greedy policy built on them is optimal.

### Pitfalls

- Optimal values are unique; optimal policies need not be. Ties are equally good.
- Greedy with respect to $\val{v}$ needs the model to look one step ahead; greedy with respect to $\val{q}$ does not.
- An optimal policy for one $\gam$ need not be optimal for another ([[discount]]).

### Check yourself {#check}

::: question
Why is $\val{v_*(A)} = 10/(1 - 0.9^5)$ in the gridworld?
---
The optimal agent collects $\rew{+10}$ every five steps: one jump and four steps back up. The discounted sum of that endless stream is $10/(1-\gam^5)$.
:::

::: question
What is the difference between the Bellman equation of a policy and the optimality equation?
---
The policy's equation averages over the actions it chooses; the optimality equation takes the best action.
:::

::: question
In the recycling robot, why recharge with a low battery rather than search?
---
Searching risks a flat battery and the $\rew{-3}$ rescue. Recharging gives nothing now but puts the robot back in the high state, worth more in the long run.
:::
