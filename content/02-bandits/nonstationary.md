+++
summary = "When the values themselves change over time, averaging everything ever seen falls behind. A constant step size keeps the estimates current, and exploration must never stop."
prereqs = ["step-size", "epsilon-greedy"]
lab = "bandit-drift"
sources = [
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., §2.5 and Exercise 2.5", url = "http://incompleteideas.net/book/the-book-2nd.html" },
  { text = "Kalman (1960), A new approach to linear filtering and prediction problems, Journal of Basic Engineering 82", url = "https://doi.org/10.1115/1.3662552" },
  { text = "Garivier & Moulines (2011), On upper-confidence bound policies for switching bandit problems, Algorithmic Learning Theory" },
  { text = "Besbes, Gur & Zeevi (2014), Stochastic multi-armed-bandit problem with non-stationary rewards, Advances in Neural Information Processing Systems 27" },
]

[story]
scene = "bandit"
env = "drifting"
seed = 67
units = 10000
average = 100
formula = '\step{1}{\val{q_*(a)} \leftarrow \val{q_*(a)} + \text{a small random step} \qquad} \step{2}{\val{Q(A)} \leftarrow \val{Q(A)} + \alp\,\big[\rew{R} - \val{Q(A)}\big]}'

[story.runs]
average = { algorithm = "epsilon-greedy", epsilon = 0.1, alpha = 0.0, name = "sample averages (α = 1/n)" }
recent = { algorithm = "epsilon-greedy", epsilon = 0.1, alpha = 0.1, name = "constant α = 0.1" }
+++

## Story

::: step {run = "average", at = 0, truth = true, learner = false, formula = 1}
**Machines that change.** These ten arms all start out equal, paying 0 on average. But after every pull, every arm's mean takes a small random step up or down. Over thousands of pulls the arms drift apart, and which arm is best keeps changing.
:::

::: step {run = "average", at = 5000, truth = true, numbers = true}
After 5000 pulls, arm 3 has pulled ahead, at $\rew{0.88}$. This player keeps plain averages and explores with $\eps = 0.1$. It has pulled arm 3 476 times, yet estimates it at only $\val{0.11}$: all 476 rewards count equally, and most of them came long ago, when arm 3 paid less.
:::

::: step {run = "average", at = 10000, truth = true, numbers = true, focus = 3}
After 10,000 pulls arm 3 pays $\rew{2.67}$ on average, far ahead of every other arm. The averaging player still estimates it at $\val{0.27}$, and spends its pulls on arm 5, worth $\rew{0.39}$. A new reward moves an average of hundreds of rewards by a fraction of a percent: the average can no longer follow.
:::

::: step {run = "recent", at = 10000, truth = true, numbers = true, focus = 3, formula = 2}
The same player with a **constant step size** $\alp = 0.1$: each reward moves the estimate a tenth of the way toward it, so the estimate is made mostly of the last few dozen rewards. It estimates arm 3 at $\val{2.46}$ and pulls it almost every time.
:::

::: step {run = "recent", at = 10000, truth = true, curves = ["average", "recent"], metric = "optimal"}
Averaged over 100 drifting problems, the averaging player stalls: it picks the current best arm only about 40% of the time, while the constant step size keeps up, at about 75%. [Race them in the Lab](lab:bandit-drift).
:::

## Textbook

### When values change {#problem}

The bandit methods so far assume a **stationary** problem: each action's reward distribution, and so its value $\val{q_*(a)}$, stays the same forever ([[k-armed-bandit]]). In a **nonstationary** problem the values change over time, and we write $\val{q_t(a)}$ for the value of $a$ at step $t$. The best action today may be a poor one tomorrow, so an agent must keep estimating values it has already learned, and keep checking actions it has already dismissed.

Nonstationarity is the rule rather than the exception. Customers' tastes change, markets move, opponents adapt. And it arises inside reinforcement learning even when the world itself is fixed. An agent that improves its policy changes the values it is trying to estimate, since a value always belongs to a policy ([[gpi]]); and methods that bootstrap learn from targets that contain their own changing estimates ([[bootstrapping]]). So the questions of this station come back in every later method.

### Averages fall behind {#averages}

The sample average gives every reward the same weight. After $n$ rewards, a new reward moves the estimate by only $1/n$ of its error ([[incremental-mean]]). Suppose an action's value jumps by $\Delta$ after its $n_0$-th reward, and $m$ more rewards follow. The average then mixes $n_0$ rewards from before with $m$ from after, and its expected error is

$$\Delta\,\frac{n_0}{n_0 + m}. \label{lag}$$

The error decays only like $1/m$: to close 90% of the gap the average needs $9 n_0$ new rewards, and the longer the agent has been learning, the slower it reacts. A constant step size forgets geometrically instead: after $m$ rewards the remaining error is $\Delta\,(1 - \alp)^m$, about $0.9^{22} \approx 10\%$ after 22 rewards for $\alp = 0.1$, however long the agent has been learning ([[step-size]]).

### Constant step sizes track {#tracking}

With a constant step size, the estimate is an exponential recency-weighted average that tracks a moving value. How well it tracks depends on how fast the value moves and how noisy the rewards are. A simple model makes this exact.

::: theorem {#thm-tracking} Tracking a random walk
Let the value of an action follow a random walk, $\val{q_{t+1}} = \val{q_t} + \eta_t$, and let each reward be $\rew{R_t} = \val{q_t} + \epsilon_t$, with independent noises $\eta_t$ of variance $\tau^2$ and $\epsilon_t$ of variance $\sigma^2$. If the estimate is updated at every step with a constant $\alp \in (0, 1)$, its mean squared error converges to
$$\mathbb{E}\big[(\val{Q_t} - \val{q_t})^2\big] \;\to\; \frac{\alp^2 \sigma^2 + \tau^2}{\alp\,(2 - \alp)}. \label{tracking-error}$$
:::

::: proof
The error $e_t = Q_t - q_t$ evolves as $e_{t+1} = Q_t + \alpha (q_t + \epsilon_t - Q_t) - q_t - \eta_t = (1 - \alpha)\, e_t + \alpha\, \epsilon_t - \eta_t$. The three terms are independent, so the variance $v_t$ of the error obeys $v_{t+1} = (1 - \alpha)^2 v_t + \alpha^2 \sigma^2 + \tau^2$. Since $(1 - \alpha)^2 < 1$, it converges to the fixed point $v = (\alpha^2 \sigma^2 + \tau^2) / (1 - (1 - \alpha)^2)$, and $1 - (1 - \alpha)^2 = \alpha (2 - \alpha)$. The mean of the error tends to 0 for the same reason, so the variance is the mean squared error.
:::

The two terms pull in opposite directions. The noise term, about $\alp \sigma^2 / 2$ for small $\alp$, asks for a small step size; the drift term, about $\tau^2 / (2\alp)$, for a large one. Their sum is smallest near $\alp \approx \tau / \sigma$: the faster the value moves relative to the noise, the larger the step size should be. Sample averages are the limit for $\tau = 0$, where the drift term vanishes and the best step size shrinks to zero. The Kalman filter (Kalman, 1960) solves this tracking problem optimally, and its gain converges to exactly such a constant.

### Exploration never ends {#exploration}

In a stationary problem, exploration can fade once the estimates are good. In a nonstationary one, an action that was rightly dismissed may later become the best, and the agent only finds out by trying it again. This changes which methods work.

- **ε-greedy** with a constant $\eps$ keeps exploring at the same rate forever, which here is a virtue ([[epsilon-greedy]]).
- **Optimistic initial values** fail: their drive to explore is spent in the first steps and never returns ([[optimistic-init]]).
- **UCB** with plain counts fails: counts accumulate confidence that is no longer warranted. Variants discount old pulls or count only a recent window (Garivier & Moulines, 2011; [[ucb]]).
- **The gradient bandit** works with a constant step size and a recency-weighted baseline ([[gradient-bandit]]).

How hard the problem is depends on how much the values can change. If the total variation of the values over $T$ steps is bounded by $V_T$, the best achievable regret grows like $(V_T\,k)^{1/3}\,T^{2/3}$, far faster than the $\ln T$ of the stationary case (Besbes, Gur & Zeevi, 2014).

### The drifting testbed {#testbed}

Sutton and Barto's Exercise 2.5 makes the testbed nonstationary: all values start equal at 0, and after every step each takes an independent random step drawn from a normal distribution with standard deviation 0.01.

::: figure {#fig-drift}
{{bandit-curves drift}}
Average reward (top) and the fraction of steps on which the current best arm was chosen (bottom), on the drifting testbed, for ε-greedy ($\eps = 0.1$) with sample averages and with a constant step size $\alp = 0.1$. Each curve averages 100 runs of 10,000 steps, smoothed over 100 steps; the Lab computes them when the figure comes into view. After Sutton & Barto, Exercise 2.5.
:::

Over the last thousand steps of the runs above, the player with a constant step size earns about 1.37 per step and picks the current best arm about 76% of the time; the player with sample averages earns about 1.06 and picks it about 41% of the time (\ref{fig-drift}). Both earn more as time goes on, because the arms drift apart and the best arm pays more and more. But only the constant step size keeps track of which arm that is; the averaging player stalls early, at around 40%, because the older its averages get, the less they move.

### Historical remarks {#history}

Sutton and Barto (§2.5) introduce constant step sizes for tracking nonstationary problems, and their Exercise 2.5 defines the drifting testbed. Tracking a drifting quantity is the problem the Kalman filter was invented for (Kalman, 1960). For bandits, Garivier and Moulines (2011) analyzed discounted and sliding-window versions of UCB, and Besbes, Gur and Zeevi (2014) related the achievable regret to the total amount of change.

## Card

### Idea

If the values change over time, the past becomes misleading. Averages give a reward from long ago as much weight as the latest one, so they fall behind. A constant step size forgets old rewards and follows the change, and exploration must continue, because a rejected arm may become the best.

::: analogy
Last year's restaurant reviews in a city where chefs keep moving. An average over all the reviews ever written still praises the place whose chef left, and ignores the one that just got good.
:::

### In symbols {#formula}

$$\text{averages: } \val{Q_{n+1}} = \val{Q_n} + \tfrac{1}{n}\big[\rew{R_n} - \val{Q_n}\big] \qquad \text{tracking: } \val{Q_{n+1}} = \val{Q_n} + \alp\,\big[\rew{R_n} - \val{Q_n}\big]$$

After a change of size $\Delta$: averages are still off by $\Delta\,n_0 / (n_0 + m)$ after $m$ rewards, a constant step size by $\Delta\,(1 - \alp)^m$.

### Why it matters {#why}

Reinforcement learning is nonstationary at its core: as the policy improves, the values it must estimate change, and bootstrapped targets move with the estimates. That is why most methods use a constant step size. [See it](lab:bandit-drift)

### Pitfalls

- Sample averages in a changing world: they converge to the average of the past, not to the present.
- Exploration that fades: optimistic starts and decaying $\eps$ stop looking for changes.
- A step size too large for noisy rewards: tracking turns into chasing noise.

### Check yourself {#check}

::: question
An arm has paid for 1000 pulls with sample averages, and then its value jumps by 1. Roughly how far has the estimate moved after 100 more pulls?
---
About $100/1100 \approx 0.09$ of the jump. With $\alp = 0.1$ it would have closed all but $0.9^{100} \approx 0.003\%$ of the gap.
:::

::: question
Why do optimistic initial values fail on a nonstationary problem?
---
Their push to explore comes from the initial estimates and is used up in the first pulls. Nothing makes the agent revisit arms later, when another arm becomes better.
:::

::: question
Why is reinforcement learning nonstationary even when the world never changes?
---
The values being estimated belong to the agent's policy, which keeps improving, and many targets contain the agent's own estimates, which keep changing.
:::
