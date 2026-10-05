+++
summary = "Skip the separate evaluations: in every sweep, give each state the value of its best move, one step ahead. The values converge to the optimal ones, and acting greedily on them is optimal."
change = "Stop each evaluation after a single sweep, and fold the improvement into it: every update takes the best action's backup, a max over actions, instead of the current policy's average."
prereqs = ["policy-iteration", "optimality", "bellman"]
lab = "dp-iteration"
sources = [
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., §4.4–4.5", url = "http://incompleteideas.net/book/the-book-2nd.html" },
  { text = "Bellman (1957), Dynamic Programming, Princeton University Press" },
  { text = "Singh & Yee (1994), An upper bound on the loss from approximate optimal-value functions, Machine Learning 16", url = "https://doi.org/10.1007/BF00993308" },
  { text = "Barto, Bradtke & Singh (1995), Learning to act using real-time dynamic programming, Artificial Intelligence 72", url = "https://doi.org/10.1016/0004-3702(94)00011-O" },
  { text = "Puterman (1994), Markov Decision Processes: Discrete Stochastic Dynamic Programming, Wiley, §6.3" },
]

[story]
scene = "grid"
env = "frozen-lake"
digits = 2
formula = '\step{1}{\val{V(s)} \leftarrow \max_a} \step{2}{\sum_{s^\prime\!,\,r} p(s^\prime\!, \rew{r} \mid s, a)\,\big[\,\rew{r} + \gam\,\val{V(s^\prime)}\,\big]}'
numbers = '\val{V(s)} \leftarrow \tfrac13\,\big(\rew{1} + 0.99 \cdot \val{0}\big) + \tfrac23\,\big(\rew{0} + 0.99 \cdot \val{0}\big) = \val{0.33}'

[story.runs]
vi = { algorithm = "value-iteration", gamma = 0.99, theta = 0.001, units = 120 }
+++

## Story

::: step {run = "vi", at = 0, values = true, formula = 1}
**The same frozen lake**, the same slippery rules. Value iteration does not evaluate any policy. In each sweep, every state simply takes the value of its **best** move, looking one step ahead. The arrows show the greedy policy for the current values; with every value at 0, all four moves tie.
:::

::: step {run = "vi", at = 0, values = true, play = 1, pace = 320, fine = true, numbers = true, formula = 2}
Watch the first sweep. Only one cell gains a value: the one left of the gem. Its best move is right, which reaches the gem one time in three, so its value becomes $\val{0.33}$. From every other cell, each move still leads to cells worth 0.
:::

::: step {run = "vi", at = 3, values = true, formula = 2}
After 3 sweeps, value has flowed three cells back from the gem, like a wave. Each sweep carries it one step further, because each update looks one step ahead.
:::

::: step {run = "vi", at = 20, values = true, formula = 2}
After 20 sweeps every safe cell has a value, and the arrows already form a route to the gem. The values are still too low: they keep rising, because longer and longer successful walks get counted.
:::

::: step {run = "vi", at = 84, values = true, focus = [1, 2]}
After 84 sweeps the largest change drops below 0.001. These are the optimal values: $\val{0.53}$ at the start. Look at the outlined cell, between two holes. Its best moves point left and right, *at the holes*. Moving up or down would slide into a hole two times in three; moving toward one falls in only one time in three.
:::

::: step {run = "vi", at = 84, values = true}
Policy iteration took 112 sweeps on this lake; value iteration needed 84. Its greedy policy was optimal after only 34, long before its values had settled. [Race the two in the Lab](lab:dp-iteration).
:::

## Textbook

### One sweep of evaluation is enough {#truncate}

Each iteration of policy iteration waits for a full policy evaluation, which may take many sweeps ([[policy-iteration]]). The evaluation step can be truncated without losing the guarantee of convergence. The extreme case stops it after a single sweep, one update of each state, and improves immediately. The two steps then combine into one simple update:

$$v_{k+1}(s) = \max_a \sum_{s',\,r} p(s', r \mid s, a)\,\big[\,\rew{r} + \gam\,v_k(s')\,\big] \quad \text{for all } s \in \mathcal{S}. \label{vi-update}$$

This is **value iteration**. It is the Bellman optimality equation ([[optimality]]) turned into an update rule, exactly as policy evaluation is the Bellman equation of a policy turned into one ([[policy-evaluation]]). The only difference from a policy-evaluation sweep is the max over actions in place of the policy's average; its backup diagram is that of $\val{v_*}$ (\ref{fig-backup}).

::: figure {#fig-backup}
{{backup v-star}}
The backup diagram of value iteration: from the state, the best of its actions (the arc means max), each averaged over the states it may lead to.
:::

### Convergence {#convergence}

Write the right-hand side of \ref{vi-update} as an operator, $(\mathcal{T} v)(s) = \max_a \sum_{s', r} p(s', r \mid s, a)\,[r + \gam\,v(s')]$.

::: lemma {#lem-contraction} The optimality operator is a contraction
If $\gam < 1$, then $\lVert \mathcal{T} u - \mathcal{T} v \rVert_\infty \le \gam\,\lVert u - v \rVert_\infty$ for any two value functions $u$ and $v$.
:::

::: proof
For each state, $|\max_a f(a) - \max_a g(a)| \le \max_a |f(a) - g(a)|$, and for each action the rewards cancel, leaving $\gam \sum_{s'} p(s' \mid s, a)\,|u(s') - v(s')| \le \gam\,\lVert u - v \rVert_\infty$.
:::

::: theorem {#thm-vi} Value iteration converges to $\val{v_*}$
If $\gam < 1$, value iteration converges to the optimal value function from any starting point, and $\lVert v_k - \val{v_*} \rVert_\infty \le \gam^k\,\lVert v_0 - \val{v_*} \rVert_\infty$. The same holds for $\gam = 1$ in episodic tasks where termination is guaranteed under every policy.
:::

::: proof
$\val{v_*}$ satisfies the Bellman optimality equation, so it is a fixed point of $\mathcal{T}$; by \ref{lem-contraction} and the Banach fixed-point theorem it is the only one, and each sweep shrinks the distance to it by the factor $\gam$.
:::

### Stopping, and extracting a policy {#stopping}

Value iteration converges only in the limit, so in practice it stops when a sweep changes no value by more than a small threshold $\theta$. Its output is then a policy, the greedy one with respect to the final values:

$$\pol{\pi(s)} = \operatorname*{arg\,max}_a \sum_{s',\,r} p(s', r \mid s, a)\,\big[\,\rew{r} + \gam\,\val{V(s')}\,\big]. \label{extract}$$

How good is that policy, if the values are only close to $\val{v_*}$? Not much worse, and the loss is bounded.

::: theorem {#thm-loss} Singh & Yee, 1994
If $\lVert \val{V} - \val{v_*} \rVert_\infty \le \epsilon$ and $\pol{\pi}$ is greedy with respect to $\val{V}$, then $\val{v_\pi(s)} \ge \val{v_*(s)} - \dfrac{2\gam\epsilon}{1 - \gam}$ for every state $s$.
:::

In practice the greedy policy often becomes optimal long before the values converge, because the ranking of the actions settles before their exact values do. On the lake below, value iteration's greedy policy is optimal after 34 sweeps; the values need 84 to settle within $\theta = 0.001$.

### The algorithm {#algorithm}

::: algorithm {#alg-vi} Value iteration, for estimating $\pol{\pi} \approx \pol{\pi_*}$
Parameter: a small threshold $\theta > 0$ determining the accuracy of estimation
Initialize $\val{V(s)}$ for all $s \in \mathcal{S}^+$ arbitrarily, except that $\val{V(\textit{terminal})} = 0$
Loop:
  $\Delta \leftarrow 0$
  Loop for each $s \in \mathcal{S}$:
    $v \leftarrow \val{V(s)}$
    $\val{V(s)} \leftarrow \max_a \sum_{s',r} p(s', r \mid s, a)\,[\rew{r} + \gam\,\val{V(s')}]$
    $\Delta \leftarrow \max(\Delta, |v - \val{V(s)}|)$
until $\Delta < \theta$
Output a deterministic policy, $\pol{\pi} \approx \pol{\pi_*}$, such that $\pol{\pi(s)} = \operatorname*{arg\,max}_a \sum_{s',r} p(s', r \mid s, a)\,[\rew{r} + \gam\,\val{V(s')}]$
:::

A sweep costs the same as a sweep of policy evaluation, with the max replacing the policy's weights: $O(|\mathcal{S}|^2 |\mathcal{A}|)$ in the worst case. Like policy evaluation, the in-place version shown here usually converges faster than one with two arrays.

### Example: Frozen Lake {#example}

On Frozen Lake ([[policy-iteration]], with $\gam = 0.99$), the first sweep gives a value only to the cell left of the gem, whose best move reaches the gem with probability $\tfrac13$: $v_1 = \tfrac13$. Each further sweep carries value one cell further back. Because the values of longer and longer successful walks keep adding up, they rise for many sweeps, and with $\theta = 0.001$ the iteration stops after 84 sweeps, with $\val{V(\textit{start})} \approx 0.530$ against the exact $\val{v_*(\textit{start})} = 0.542$. \ref{fig-optimal} shows the optimal values and policy: undiscounted, the optimal policy reaches the gem with probability 0.82 from the start.

::: figure {#fig-optimal}
{{frozen optimal}}
Frozen Lake solved: the optimal values for $\gam = 0.99$, an optimal policy, and, under that policy, the chance of reaching the gem from each cell. Where two arrows appear, both moves are optimal. Computed by the Lab.
:::

\ref{fig-race} compares value iteration with policy iteration, sweep by sweep. Policy iteration spends its first sweeps evaluating the random policy, which helps little; its first improvement then jumps far ahead. Value iteration improves a little with every sweep. Which one wins depends on the problem and on the tolerance: here value iteration reaches $\theta$ first, while policy iteration's policy is optimal after its second improvement, at sweep 83.

::: figure {#fig-race}
{{dp-race}}
The distance between each method's values and the optimal values on Frozen Lake, after each sweep through the states, on a logarithmic scale. Policy iteration's improvement steps are marked. Computed by the Lab.
:::

### Between value and policy iteration {#between}

Policy iteration evaluates fully between improvements; value iteration evaluates for one sweep. In between lies **modified policy iteration**, which performs a fixed number $m$ of evaluation sweeps before each improvement. All of these are instances of the same pattern, alternating some evaluation with some improvement, and all converge to an optimal policy for discounted finite MDPs ([[gpi]]). In practice a few evaluation sweeps between improvements are often faster than either extreme.

### Asynchronous value iteration {#async}

Nothing requires sweeping the states in order, or at all evenly. **Asynchronous** value iteration updates the values in place, in any order, using whatever values are available; it still converges to $\val{v_*}$ provided every state keeps being updated. This freedom makes it possible to focus the computation where it matters: on the states an agent actually visits, as real-time dynamic programming does (Barto, Bradtke & Singh, 1995), or on the states whose values are changing most, as [[prioritized-sweeping]] does. It also makes it possible to interleave computation with real interaction ([[dp-limits]]).

### Value iteration on action values {#q}

The same update can be written for action values, which makes acting on the result easier, since the greedy action can then be read off without the model:

$$Q_{k+1}(s,a) = \sum_{s',\,r} p(s', r \mid s, a)\,\big[\,\rew{r} + \gam \max_{a'} Q_k(s', a')\,\big]. \label{q-vi}$$

[[q-learning]] is what remains of this update when the expectation over $p$ is replaced by a single sampled transition: a stochastic, asynchronous value iteration.

### Historical remarks {#history}

Value iteration, the method of successive approximation for the Bellman optimality equation, is the oldest dynamic programming algorithm (Bellman, 1957). Singh and Yee (1994) bounded the loss of policies that are greedy with respect to approximate values. Barto, Bradtke and Singh (1995) connected asynchronous dynamic programming with learning in real time, a bridge between the methods of this part and those that learn from experience.

## Card

### Idea

In every sweep, give each state the value of its best move one step ahead: the max over actions of reward plus discounted value of where the move leads. The values converge to the optimal ones, and the greedy policy with respect to them is optimal. It is policy iteration with every evaluation cut to a single sweep.

::: analogy
Working out the shortest drive home from every town, by repeatedly setting each town's distance to "the best neighbor's distance plus the road to it". Distances spread outward from home, and after enough rounds every town knows its shortest route.
:::

### The update {#update}

$$\val{V(s)} \leftarrow \max_a \sum_{s', r} p(s', \rew{r} \mid s, a)\,\big[\,\rew{r} + \gam\,\val{V(s')}\,\big]$$

Then act greedily: $\pol{\pi(s)} = \operatorname*{arg\,max}_a$ of the same expression.

### One change from policy iteration {#change}

| | between improvements | update |
| --- | --- | --- |
| [[policy-iteration]] | evaluates the policy until it settles | the policy's average over actions |
| value iteration | one sweep only | the best action's value: a max |

### Backup diagram {#backup}

{{backup v-star}}

From the state, through every action, keeping the best (the arc means max), each action averaged over the states it may lead to.

### Pseudocode

::: pseudocode
Parameters: the model $p$, a tolerance $\theta > 0$, discount $\gam$
Set $\val{V(s)} = 0$ for every state {#init}
Repeat:
  $\Delta \leftarrow 0$
  For each state $s$: {#sweep}
    $\val{V(s)} \leftarrow \max_a \sum_{s',r} p(s',r \mid s,a)\,[\rew{r} + \gam\,\val{V(s')}]$, and $\Delta \leftarrow \max(\Delta, \text{change})$ {#update}
  until $\Delta < \theta$ {#delta}
Act greedily: $\pol{\pi(s)} = \operatorname*{arg\,max}_a \sum_{s',r} p(s',r \mid s,a)\,[\rew{r} + \gam\,\val{V(s')}]$
:::

### Perks

- One simple update, no separate evaluations, no policy to store while it runs. [See it](lab:dp-iteration)
- Converges to the optimal values from any start, geometrically when $\gam < 1$.
- Its greedy policy is usually optimal long before the values settle.

### Flaws

- Converges only in the limit: when to stop, and how good the policy is, rests on a tolerance.
- Needs a complete model of the world.
- Every sweep touches every state, unless updates are focused asynchronously ([[dp-limits]]).

### Knobs

| Knob | Too low | Too high |
| --- | --- | --- |
| $\theta$ tolerance | sweeps long after the policy stopped changing | stops before the greedy policy is optimal |
| $\gam$ discount | short-sighted: may prefer a quick but risky route | many sweeps; the values rise slowly |

### Pitfalls

- Reading the values as those of a policy: until convergence they are not the values of any policy.
- Stopping on a loose tolerance with $\gam$ close to 1: the error bound grows like $1/(1 - \gam)$.
- Forgetting to extract the policy: value iteration's output is the greedy policy, computed with the model.

### Check yourself {#check}

::: question
What is the only difference between a sweep of value iteration and a sweep of policy evaluation?
---
The max over actions: value iteration backs up the best action, policy evaluation the average over the policy's actions.
:::

::: question
On Frozen Lake, why does only one cell change in the first sweep?
---
All values start at 0 and only the gem pays. The only cell from which a move can reach the gem in one step is the one next to it.
:::

::: question
Why can the greedy policy be optimal before the values have converged?
---
Acting only needs the right ranking of the actions in each state, which settles before their exact values do.
:::
