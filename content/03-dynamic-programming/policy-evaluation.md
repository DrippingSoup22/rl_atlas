+++
summary = "Compute how good a policy is, when you know the rules of the world: sweep through the states, again and again, replacing each value by the average of what one step leads to."
prereqs = ["bellman", "value-functions", "mdp"]
lab = "dp-evaluation"
sources = [
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., §4.1, Example 4.1 and Figure 4.1", url = "http://incompleteideas.net/book/the-book-2nd.html" },
  { text = "Bellman (1957), Dynamic Programming, Princeton University Press" },
  { text = "Puterman (1994), Markov Decision Processes: Discrete Stochastic Dynamic Programming, Wiley, §6.3" },
  { text = "Bertsekas & Tsitsiklis (1989), Parallel and Distributed Computation: Numerical Methods, Prentice Hall" },
]

[story]
scene = "grid"
env = "small-gridworld"
digits = 1
formula = '\step{1}{\val{V(s)}} \step{2}{\leftarrow \sum_a \pol{\pi(a \mid s)}} \step{3}{\sum_{s^\prime\!,\,r} p(s^\prime\!, \rew{r} \mid s, a)} \step{4}{\big[\,\rew{r} + \gam\,\val{V(s^\prime)}\,\big]}'
numbers = '\val{V(s)} \leftarrow \tfrac14\,\big(\rew{-1} + \val{0}\big) + 3 \cdot \tfrac14\,\big(\rew{-1} + \val{(-1)}\big) = \val{-1.75}'

[story.runs]
sweeps = { algorithm = "policy-evaluation", policy = "random", gamma = 1.0, theta = 0.000001, sync = true, units = 400 }
+++

## Story

::: step {run = "sweeps", at = 0, values = true}
**A 4 × 4 grid.** The two shaded corners end the episode. Every move costs $\rew{-1}$, and a move off the grid leaves the agent where it is. The policy to judge moves at random, each direction with probability ¼ (the arrows). How good is that? Its value $\val{v_\pi(s)}$ is minus the average number of steps it takes to reach a corner from $s$.
:::

::: step {run = "sweeps", at = 0, values = true, play = 1, pace = 300, fine = true, formula = 4}
We know the rules of this world, so we can compute the values without playing a single episode. **Policy evaluation** turns the Bellman equation into an update: each state's value becomes the average, over the moves, of the reward plus the current value of where the move lands. Watch one sweep through the 14 states. Every value becomes $\val{-1}$, since every neighbor is still 0.
:::

::: step {run = "sweeps", at = 1, values = true, focus = [0, 1], next = true, numbers = true, formula = 4}
The second sweep, at the state next to the top-left corner. One move ends the episode, at value 0. The other three, including the bump into the wall that stays put, land on states now worth $\val{-1}$. So the new value is $\val{-1.75}$.
:::

::: step {run = "sweeps", at = 3, values = true}
After 3 sweeps: $\val{-2.4}$ next to the corners, down to $\val{-3.0}$ far from them. Each sweep looks one step further ahead: after $k$ sweeps, the values count the rewards of the first $k$ steps of every walk.
:::

::: step {run = "sweeps", at = 10, values = true}
After 10 sweeps the values range from $\val{-6.1}$ to $\val{-9.0}$. They keep falling, by less each time: the longer walks, which these values do not yet count, are rarer and rarer.
:::

::: step {run = "sweeps", at = 400, values = true}
After a few hundred sweeps nothing changes any more. A random walk from a cell next to a corner takes 14 steps on average to end; from the far corners, 22. These numbers solve the Bellman equations of the random policy. [Sweep it in the Lab](lab:dp-evaluation), where each new value is used as soon as it is computed, which settles sooner.
:::

## Textbook

### Prediction with a model {#problem}

Suppose the agent were handed the rules of its world: the states, the actions and the full dynamics $p(s', r \mid s, a)$ of a finite Markov decision process ([[mdp]]). Finding good behavior would then be a matter of computation rather than experience, and **dynamic programming** (DP) is the family of algorithms that carry out that computation. Real agents rarely get such a model, and even with one the work grows with the number of states ([[dp-limits]]), so DP is seldom what runs in practice. It is the foundation all the same: the learning methods later in the atlas read best as ways to approximate what DP computes, from samples instead of a model and with far less work per step ([[model-based-free]]).

The first problem is **prediction**, or policy evaluation ([[prediction-control]]): given a policy $\pol{\pi}$, compute its state-value function $\val{v_\pi}$. By the Bellman equation ([[bellman]]), for every state $s$,

$$\val{v_\pi(s)} = \sum_a \pol{\pi(a \mid s)} \sum_{s',\,r} p(s', r \mid s, a)\,\big[\,\rew{r} + \gam\,\val{v_\pi(s')}\,\big]. \label{bellman-v}$$

With the dynamics known, these are linear equations, one per state, in as many unknown values, and a linear solver could produce the answer in one go, at a cost that grows with the cube of the number of states. Repeating a cheap update until the values settle scales better, and it is the version whose ideas carry over to learning.

### Iterative policy evaluation {#iteration}

Turn the equation into an assignment. Begin with any guess $v_0$ (a terminal state, if there is one, starts and stays at 0), and compute each new guess from the previous one by evaluating the right-hand side of the Bellman equation:

$$v_{k+1}(s) = \sum_a \pol{\pi(a \mid s)} \sum_{s',\,r} p(s', r \mid s, a)\,\big[\,\rew{r} + \gam\,v_k(s')\,\big] \quad \text{for all } s \in \mathcal{S}. \label{update-rule}$$

$v_k = \val{v_\pi}$ is a fixed point of this rule, because \ref{bellman-v} holds for it. In operator form, $v_{k+1} = \mathcal{T}_\pi v_k$, where $\mathcal{T}_\pi$ is the Bellman operator of $\pol{\pi}$.

::: theorem {#thm-converge} Convergence of iterative policy evaluation
If $\gam < 1$, or if termination is guaranteed from every state under $\pol{\pi}$, the sequence $v_k$ converges to $\val{v_\pi}$ from any starting point. With $\gam < 1$ the error shrinks at least geometrically: $\lVert v_k - \val{v_\pi} \rVert_\infty \le \gam^k\, \lVert v_0 - \val{v_\pi} \rVert_\infty$.
:::

::: proof
For $\gam < 1$, $\mathcal{T}_\pi$ is a $\gam$-contraction in the max norm, and $\val{v_\pi}$ is its unique fixed point ([[bellman]]), so each application brings $v_k$ closer to it by a factor $\gam$. When $\gam = 1$ and every state reaches termination with probability 1, some power of the policy's transition matrix restricted to the nonterminal states is a contraction, and the same argument applies to blocks of sweeps.
:::

When $v_0 = 0$ the iterates have a direct meaning: $v_k(s)$ is the expected sum of the first $k$ rewards, discounted, of an episode that starts in $s$ and follows $\pol{\pi}$. Each sweep extends the horizon by one step.

### Expected updates {#expected}

Look at what one application of \ref{update-rule} does to a single state. It considers every action the policy might take there and every reward and next state each action might produce, weighs each outcome by its probability, and sets the state's value to the weighted average of reward plus discounted successor value. Nothing is sampled: the whole distribution of one step is averaged at once, which is why this is called an **expected update**. Its backup diagram is that of $\val{v_\pi}$: from the state, through every action the policy may take, to every next state (\ref{fig-backup}).

::: figure {#fig-backup}
{{backup v-pi}}
The backup diagram of iterative policy evaluation: an expected update looks at every action the policy may take and every state each action may lead to.
:::

The contrast with what comes later is the whole point. Monte Carlo methods replace the expectation by the return of one sampled episode ([[mc-prediction]]); temporal-difference methods replace it by one sampled transition, $\rew{R_{t+1}} + \gam\,\val{V(S_{t+1})}$ ([[td0]]). Both keep the shape of \ref{update-rule} and give up the need for $p$.

### In place or two arrays {#in-place}

A sweep can be organized in two ways. The **two-array** version computes all the new values $v_{k+1}(s)$ from the old values $v_k$, kept in a separate array. The **in-place** version uses one array and overwrites each value as soon as it is computed, so states later in the sweep already use the new values of the states before them. In-place sweeps converge to $\val{v_\pi}$ as well, and usually in fewer sweeps: a value improved early in a sweep immediately improves every later update that depends on it, instead of waiting for the next sweep. On the 4 × 4 gridworld below, they settle to a tolerance of $10^{-4}$ in 114 sweeps instead of 173. Visiting the states in a good order, for instance outward from the terminal states, can speed things up further. Numerical analysts know the two versions as the Jacobi and Gauss–Seidel iterations.

### When to stop {#stopping}

The values reach $\val{v_\pi}$ exactly only after infinitely many sweeps, so in practice the loop ends once a sweep changes nothing by much: when the largest change, $\Delta = \max_s |v_{k+1}(s) - v_k(s)|$, drops below a small threshold $\theta$. For $\gam < 1$ a small last change also means a small remaining error.

::: lemma {#lem-stop} A stopping rule with a guarantee
If $\gam < 1$ and $\lVert v_{k+1} - v_k \rVert_\infty < \theta$ for two-array sweeps, then $\lVert v_{k+1} - \val{v_\pi} \rVert_\infty \le \dfrac{\gam\,\theta}{1 - \gam}$.
:::

::: proof
By the contraction property, $\lVert v_{k+1} - v_\pi \rVert \le \gam\, \lVert v_k - v_\pi \rVert \le \gam\, \big(\lVert v_k - v_{k+1} \rVert + \lVert v_{k+1} - v_\pi \rVert\big)$. Rearranging gives $(1 - \gam)\,\lVert v_{k+1} - v_\pi \rVert \le \gam\,\theta$.
:::

For $\gam$ close to 1 the factor $\gam / (1 - \gam)$ is large: with $\gam = 0.99$, a change below $10^{-3}$ guarantees an error of at most about $0.1$.

### Example: the 4 × 4 gridworld {#example}

::: example {#ex-grid} The 4 × 4 gridworld (Sutton & Barto, Example 4.1)
The nonterminal states are the 14 cells of a $4 \times 4$ grid other than two opposite corners, which are terminal. In each state there are four actions, which move one cell up, down, left or right deterministically, except that a move off the grid leaves the state unchanged. Every transition has reward $\rew{-1}$ until a terminal state is reached. The task is undiscounted, $\gam = 1$, and the policy to evaluate is equiprobable random.
:::

::: figure {#fig-sweeps}
{{dp-sweeps}}
The values of the random policy after $k$ two-array sweeps of iterative policy evaluation, from $v_0 = 0$ (the numbers), and the greedy policy with respect to each of them (the arrows, discussed in [[policy-improvement]]). Computed by the Lab. After Sutton & Barto, Figure 4.1.
:::

After one sweep every nonterminal value is $-1$: one step costs 1 whatever happens. In the second sweep, a cell next to a corner has one move into the corner, worth $0$, and three moves to cells worth $-1$, so its value becomes $\tfrac14(-1 + 0) + \tfrac34(-1 - 1) = -1.75$; cells farther away get $-2$. The final values, $-14$ next to the corners and $-22$ in the far corners, are the negated expected numbers of steps a random walk takes to end (\ref{fig-sweeps}).

### The algorithm {#algorithm}

::: algorithm {#alg-pe} Iterative policy evaluation, for estimating $V \approx \val{v_\pi}$
Input: the policy $\pol{\pi}$ to evaluate, and a tolerance $\theta > 0$ (smaller is more accurate)
Set $\val{V(s)}$ to any value for every nonterminal state, and $\val{V(\textit{terminal})} = 0$
Loop:
  $\Delta \leftarrow 0$
  Loop for each $s \in \mathcal{S}$:
    $v \leftarrow \val{V(s)}$
    $\val{V(s)} \leftarrow \sum_a \pol{\pi(a \mid s)} \sum_{s',r} p(s', r \mid s, a)\,[\rew{r} + \gam\,\val{V(s')}]$
    $\Delta \leftarrow \max(\Delta, |v - \val{V(s)}|)$
until $\Delta < \theta$
:::

This is the in-place version. One sweep costs one expected update per state, each summing over the actions and their outcomes: $O(|\mathcal{S}|^2 |\mathcal{A}|)$ in the worst case, $O(|\mathcal{S}|\,|\mathcal{A}|\,b)$ when each action leads to at most $b$ states. With $\gam < 1$, reaching an accuracy $\epsilon$ from an error $e_0$ takes about $\ln(e_0/\epsilon) / \ln(1/\gam)$ sweeps: the closer $\gam$ is to 1, the more sweeps.

### Properties {#properties}

- **It needs the model.** Every update sums over $p(s', r \mid s, a)$. Without the model, the same update must be estimated from experience ([[mc-prediction]], [[td0]]).
- **It bootstraps.** Each value is updated from the current estimates of other values. The iteration still converges, because the update is a contraction ([[bootstrapping]]).
- **It is exact up to $\theta$.** There is no sampling noise: the only error is the one left by stopping early.
- **It is the evaluation half of policy iteration.** Alternated with greedy improvement, it finds optimal policies ([[policy-iteration]]); truncated to a single sweep, it becomes [[value-iteration]].
- **It can update states in any order.** Asynchronous versions update whichever states they like, as long as every state keeps being updated, and still converge ([[dp-limits]]).

### Historical remarks {#history}

Dynamic programming and the Bellman equations are due to Bellman (1957). Iterative policy evaluation is the method of successive approximation applied to them; Puterman (1994) gives the standard treatment. In-place and asynchronous versions, in which values are updated in any order and even by different processors at different times, were analyzed by Bertsekas and Tsitsiklis (1989). The gridworld example is Sutton and Barto's Example 4.1.

## Card

### Idea

To find out how good a policy is when you know the rules, sweep through the states and replace each value by the average, over the policy's moves and their outcomes, of the reward plus the value of where the move lands. Repeat until nothing changes. It is the Bellman equation used as an update.

::: analogy
Working out house prices street by street from the prices of neighboring streets. Each round of revisions uses the latest neighbors' prices, and after enough rounds every price agrees with its neighbors.
:::

### The update {#update}

$$\val{V(s)} \leftarrow \sum_a \pol{\pi(a \mid s)} \sum_{s', r} p(s', \rew{r} \mid s, a)\,\big[\,\rew{r} + \gam\,\val{V(s')}\,\big]$$

Applied to every state, sweep after sweep, until the largest change $\Delta$ is below $\theta$.

### One change from the Bellman equation {#change}

| | |
| --- | --- |
| [[bellman]] equation | $\val{v_\pi} = \mathcal{T}_\pi \val{v_\pi}$: a condition the true values satisfy |
| policy evaluation | $\val{V} \leftarrow \mathcal{T}_\pi \val{V}$: the same right-hand side, used as an assignment, again and again |

### Backup diagram {#backup}

{{backup v-pi}}

From the state, through every action the policy may take, to every state each action may lead to: an expected update over all of them.

### Pseudocode

::: pseudocode
Parameters: the policy $\pol{\pi}$, the model $p$, a tolerance $\theta > 0$, discount $\gam$
Set $\val{V(s)} = 0$ for every state; a terminal state stays at 0 {#init}
Repeat:
  $\Delta \leftarrow 0$
  For each state $s$: {#sweep}
    $\val{V(s)} \leftarrow \sum_a \pol{\pi(a \mid s)} \sum_{s',r} p(s',r \mid s,a)\,[\rew{r} + \gam\,\val{V(s')}]$, and $\Delta \leftarrow \max(\Delta, \text{change})$ {#update}
  until $\Delta < \theta$ {#delta}
:::

### Perks

- Exact: no sampling noise, only the error left by stopping early. [See it](lab:dp-evaluation)
- Simple and guaranteed to converge, from any starting values.
- Works in place and in any order, which saves memory and often time.

### Flaws

- Needs a complete model of the world: every $p(s', r \mid s, a)$.
- Every sweep touches every state: hopeless for huge state spaces ([[dp-limits]]).
- Slow when $\gam$ is close to 1 or episodes are long: many sweeps are needed.

### Knobs

| Knob | Too low | Too high |
| --- | --- | --- |
| $\theta$ tolerance | many sweeps for tiny changes | stops while values are still off |
| $\gam$ discount | short-sighted values | slow convergence (more sweeps) |

### Pitfalls

- Forgetting the terminal state: its value must stay 0.
- Two arrays when one would do: in-place sweeps use less memory and usually converge faster.
- Reading the values as the agent's behavior: they describe the given policy, not the best one.

### Check yourself {#check}

::: question
In the 4 × 4 gridworld, all values start at 0. What does every value become after the first sweep, and why?
---
$-1$: whatever the move, it costs 1, and every state it can lead to is still worth 0.
:::

::: question
What does $v_k(s)$ mean after $k$ two-array sweeps from $v_0 = 0$?
---
The expected sum of the first $k$ rewards of an episode that starts in $s$ and follows the policy.
:::

::: question
Why do in-place sweeps usually converge faster?
---
Each update already uses the new values of the states updated before it in the same sweep, so information spreads further in one sweep.
:::
