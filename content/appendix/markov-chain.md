+++
title = "Markov chains"
summary = "A random process that hops between states, where the next hop depends only on the current state. Its transition probabilities form a matrix whose rows add up to 1; an MDP with a fixed policy is one."
group = "Probability"
sources = [
  { text = "Norris (1997), Markov Chains, Cambridge University Press, chapter 1" },
  { text = "Blitzstein & Hwang (2019), Introduction to Probability, 2nd ed., CRC Press, chapter 11", url = "https://projects.iq.harvard.edu/stat110" },
]
+++

A **Markov chain** is a sequence of random states $X_0, X_1, X_2, \ldots$ in which the next state depends only on the current one, not on the earlier history:

$$\Pr\{X_{t+1} = j \mid X_t = i,\ X_{t-1}, \ldots, X_0\} = \Pr\{X_{t+1} = j \mid X_t = i\} = P(i, j). \label{markov}$$

The numbers $P(i, j)$ are the **transition probabilities**: the probability of hopping from state $i$ to state $j$ in one step. The chain has no decisions and no rewards; it just moves. An MDP adds both. Once a policy is fixed, the decisions are made by chance too, and the states of the MDP form a Markov chain ([[policy]]).

### The transition matrix {#matrix}

For a finite set of states, the transition probabilities fill a square table, the **transition matrix** $P$, with one row and one column per state ([[a:vectors-matrices|matrices]]). Row $i$ lists where the chain may go from $i$, so every row is a probability distribution: its entries are nonnegative and add up to 1. A matrix with this property is called **stochastic**.

::: example {#ex-weather}
A crude weather model with states sunny and rainy: after a sunny day, sunny again with probability 0.8; after a rainy day, rainy again with probability 0.6.

| today ↓, tomorrow → | sunny | rainy |
| --- | --- | --- |
| sunny | 0.8 | 0.2 |
| rainy | 0.4 | 0.6 |

Each row adds up to 1.
:::

The recycling robot with the policy “always search when high, always recharge when low” is the chain with $P(\text{high}, \text{high}) = 0.7$, $P(\text{high}, \text{low}) = 0.3$ and $P(\text{low}, \text{high}) = 1$ ([[mdp]]).

### Where the chain will be {#distribution}

If $\mu_t$ is the distribution of the state at time $t$, written as a row of probabilities, the distribution one step later is found by the law of total probability ([[a:conditional-probability|conditional probability]]): $\mu_{t+1}(j) = \sum_i \mu_t(i)\,P(i, j)$, which is the matrix product $\mu_{t+1} = \mu_t P$. After $k$ steps, $\mu_{t+k} = \mu_t P^k$.

Applying $P$ to a column of numbers $v$, one per state, does something different and equally useful: $(Pv)(i) = \sum_j P(i, j)\,v(j)$ is the **average of $v$ over the next state**, starting from $i$. Each entry of $Pv$ is a weighted average of the entries of $v$, so it lies between the smallest and the largest of them ([[a:max-argmax|max and argmax]]). This is the property that makes the Bellman equation a contraction ([[bellman]]).

### Long-run behavior {#long-run}

Many chains forget where they started: whatever $\mu_0$, the distribution $\mu_t$ settles on a fixed **stationary distribution** $d$, one that the chain's step leaves unchanged, $d = dP$. For the weather chain, $d = (\tfrac23, \tfrac13)$: in the long run, two days in three are sunny. Stationary distributions say how often a policy visits each state, which matters when some states must be weighted more than others, as in learning with function approximation.

A state that, once entered, is never left, $P(i, i) = 1$, is **absorbing**. The end of an episode is treated as an absorbing state that gives no reward ([[episodes]]).
