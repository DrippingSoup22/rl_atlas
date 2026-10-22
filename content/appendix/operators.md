+++
title = "Operators"
summary = "A function whose input and output are whole functions: it takes a value function and returns a new one. The Bellman operator turns “one equation per state” into a single object that can be applied, repeated and analyzed."
group = "Analysis"
sources = [
  { text = "Puterman (1994), Markov Decision Processes: Discrete Stochastic Dynamic Programming, Wiley, §6.2" },
]
+++

An ordinary function takes a number and returns a number. An **operator** takes a whole function and returns a whole function. The word is only a name for that: a function one level up.

### An example {#example}

Take any value function $v$, which assigns a number $v(s)$ to every state ([[a:sets-functions|functions]]). From it, build a new value function by doing, in every state, one step of look-ahead under a policy $\pol{\pi}$: the expected reward of the next step, plus $\gam$ times the expected value of $v$ at the next state. Call the result $\mathcal{T}_\pi v$. Its value at $s$ is

$$(\mathcal{T}_\pi v)(s) = \sum_a \pol{\pi(a \mid s)} \sum_{s',\,r} p(s', \rew{r} \mid s, a)\,\big[\,\rew{r} + \gam\,v(s')\,\big]. \label{bellman-op}$$

$\mathcal{T}_\pi$ is the **Bellman operator** of the policy ([[bellman]]). It takes in the whole table $v$ and gives back a whole new table $\mathcal{T}_\pi v$: one new number for every state, each computed from the old numbers of the states that can follow it.

::: example {#ex-numbers}
In a world with two states, suppose $\mathcal{T}_\pi$ is $(\mathcal{T}_\pi v)(1) = 1 + 0.9\,v(2)$ and $(\mathcal{T}_\pi v)(2) = 0.9\,v(1)$. Applied to $v = (0, 0)$, it gives $(1, 0)$; applied to $(1, 0)$, it gives $(1, 0.9)$; applied to $(1, 0.9)$, it gives $(1.81, 0.9)$. Each application turns one table into the next.
:::

For a finite MDP, a value function is a vector with one entry per state ([[a:vectors-matrices|vectors]]), and the Bellman operator of a policy is the map $v \mapsto r_\pi + \gam P_\pi v$: multiply by the transition matrix, scale by $\gam$, add the expected rewards.

### Why talk about operators {#why}

Writing the many Bellman equations as one operator does three things.

- **The equations become one equation.** “$v(s)$ equals the right-hand side of \ref{bellman-op}, for every $s$” is the single statement $v = \mathcal{T}_\pi v$: the value function is a function that the operator leaves unchanged, a **fixed point** ([[a:fixed-points|fixed points]]).
- **Algorithms become repetition.** Applying the operator again and again, $v_{k+1} = \mathcal{T}_\pi v_k$, is iterative policy evaluation ([[policy-evaluation]]). Writing $\mathcal{T}_\pi^k$ for $k$ applications in a row, the $k$-th estimate is $\mathcal{T}_\pi^k v_0$.
- **Properties can be proved once.** Whether the repetition converges, and how fast, depends on one property of the operator: whether it brings any two inputs closer together ([[a:contraction|contractions]]).

The optimality equation has its operator too, with a maximum in place of the policy's average,

$$(\mathcal{T}_* v)(s) = \max_a \sum_{s',\,r} p(s', \rew{r} \mid s, a)\,\big[\,\rew{r} + \gam\,v(s')\,\big],$$

and value iteration is its repetition ([[optimality]]). $\mathcal{T}_\pi$ is linear, $\mathcal{T}_*$ is not (because of the max), but both bring value functions closer together, which is all the theory needs.
