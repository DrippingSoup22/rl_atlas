+++
title = "Growth rates and big-O"
summary = "How fast a cost grows with the size of a problem: logarithmic, linear, polynomial or exponential. The difference decides what can be computed and how fast an agent stops making mistakes."
group = "Analysis"
sources = [
  { text = "Cormen, Leiserson, Rivest & Stein (2022), Introduction to Algorithms, 4th ed., MIT Press, chapter 3" },
]
+++

When the size of a problem grows, the work needed to solve it grows too. What matters is usually not the exact count but its **rate of growth**: whether doubling the problem doubles the work, multiplies it by eight, or squares it.

### Big-O notation {#big-o}

$O(g(n))$, read “order $g(n)$” or “big-O of $g(n)$”, describes a quantity that grows at most like $g(n)$, up to a constant factor, once $n$ is large. Solving $n$ linear equations takes $O(n^3)$ operations: perhaps $\tfrac13 n^3$, perhaps $2n^3 + 5n^2$, but roughly proportional to $n^3$ for large $n$ ([[a:linear-systems|linear systems]]). Constants and smaller terms are dropped because, for large problems, the dominant term decides everything.

### The common rates {#rates}

| Rate | Name | When $n$ doubles, the cost | Example in the atlas |
| --- | --- | --- | --- |
| $O(\log n)$ | logarithmic | adds a constant | the regret of a good bandit strategy after $n$ steps |
| $O(n)$ | linear | doubles the cost | one sweep over $n$ states; the regret of a constant ε-greedy |
| $O(n^2)$ | quadratic | multiplies by 4 | a table of transition probabilities between $n$ states |
| $O(n^3)$ | cubic | multiplies by 8 | solving the Bellman equations directly |
| $O(2^n)$ | exponential | squares the cost | the number of action sequences of length $n$ with two actions |

The gap between the rows is enormous. For $n = 100$: $\log_2 n \approx 7$, $n^3 = 10^6$, and $2^n \approx 10^{30}$, more than the number of microseconds since the beginning of the universe. Polynomial costs are usually affordable; exponential ones are not, beyond small sizes.

### Exponential in reinforcement learning {#exponential}

Exponential growth is the enemy behind several difficulties of the atlas.

- **Too many states.** A state described by $d$ features, each with 10 possible values, can take $10^d$ values. Backgammon has about $10^{20}$ positions; no table can hold them, and no sweep can visit them ([[optimality]]). This is the *curse of dimensionality*.
- **Random exploration.** In a chain of $n$ states where each wrong action sends the agent back to the start, an agent choosing at random between two actions reaches the end with probability $2^{-n}$ per attempt, so the expected number of attempts grows like $2^n$ ([[explore-exploit]]).

### Logarithmic in reinforcement learning {#log}

A logarithm grows very slowly ([[a:exponential-logarithm|logarithms]]): $\ln n$ is about 7 for $n = 1000$ and about 14 for $n = 1\,000\,000$. In the bandit problem, the **regret** of a strategy, the reward lost compared with always choosing the best action, can grow like $\ln n$ but no slower; good strategies such as UCB achieve this ([[explore-exploit]]). A constant-rate ε-greedy keeps choosing at random a fixed fraction of the time, so its regret grows like $n$. After a million steps, the difference between the two is the difference between a few dozen mistakes and tens of thousands.
