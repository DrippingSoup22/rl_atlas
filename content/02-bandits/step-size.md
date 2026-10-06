+++
summary = "How far each update moves an estimate toward its target. A step size that shrinks like 1/n averages everything; a constant one forgets the past and follows the present."
prereqs = ["incremental-mean"]
lab = "bandit-drift"
sources = [
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., §2.4–2.5 and Exercise 2.7", url = "http://incompleteideas.net/book/the-book-2nd.html" },
  { text = "Robbins & Monro (1951), A stochastic approximation method, Annals of Mathematical Statistics 22", url = "https://doi.org/10.1214/aoms/1177729586" },
  { text = "Blum (1954), Approximation methods which converge with probability one, Annals of Mathematical Statistics 25" },
  { text = "Kushner & Yin (2003), Stochastic Approximation and Recursive Algorithms and Applications, 2nd ed., Springer" },
  { text = "Bertsekas & Tsitsiklis (1996), Neuro-Dynamic Programming, Athena Scientific, chapter 4" },
]

[story]
scene = "stream"
source = { kind = "normal", mean = 1.0, sd = 1.0, drift = 0.08, n = 400, seed = 2 }
domain = [-2.0, 5.5]
formula = '''\step{1}{\val{Q_{n+1}} = \val{Q_n} + \alp\big[\rew{R_n} - \val{Q_n}\big]} \step{2}{\qquad \val{Q_{n+1}} = (1 - \alp)^n \val{Q_1} + \sum_{i=1}^{n} \alp (1 - \alp)^{n-i} \rew{R_i}}'''

[story.rules]
avg = { rule = "average", label = "average (1/n)" }
slow = { rule = "constant", alpha = 0.1, q0 = 5.0, label = "α = 0.1, from 5" }
fast = { rule = "constant", alpha = 0.5, q0 = 5.0, label = "α = 0.5, from 5" }
fair = { rule = "unbiased", alpha = 0.1, q0 = 5.0, label = "α = 0.1 without the bias" }
+++

## Story

::: step {upto = 10, rules = ["avg", "slow"], formula = 1}
**An arm whose mean wanders** (the dashed line), and two learners that both start from a wild first guess, 5. The average's first step is $1/1$: it jumps onto the first reward and forgets the 5 at once. A constant step of 0.1 keeps a share of it: after 10 pulls, $0.9^{10}$, about a third, of the gap it started with. It says 2.61 where the mean is 1.13.
:::

::: step {upto = 40, rules = ["avg", "slow"], formula = 2}
**The bias fades.** Unrolled, a constant step size is a weighted sum: the first estimate keeps the weight $(1 - \alp)^n$, and each reward gets $\alp(1 - \alp)^{n-i}$, more the more recent it is. After 40 pulls, the 5 weighs $0.9^{40}$, under 2%.
:::

::: step {upto = 300, rules = ["avg", "slow"], formula = 2}
**Now the arm changes.** Its mean climbed to about 2.4 by pull 40, then fell back to 1.1 by pull 300. The average weighs all 300 pulls equally and still says 2.06. The constant step weighs the last 10 pulls at 65% between them, and says 1.20. For a world that changes, forgetting the old is the point.
:::

::: step {upto = 300, rules = ["slow", "fast"]}
**A bigger step tracks faster, and never settles.** With $\alp = 0.5$, each reward pulls the estimate halfway to it, so the estimate is mostly the last two or three rewards: at pull 300 it says 0.67, at pull 400 1.61, while the mean drifts from 1.1 to 0.8. The noise of a constant step size never fades: it trades steadiness for speed of tracking.
:::

::: step {upto = 10, rules = ["slow", "fair"]}
**Both, without the bias.** A step size that starts at 1 and slides down to $\alp$ takes the first reward whole, as the average does, and then forgets at the constant rate: $\beta_n = \alp / \bar o_n$, with $\bar o_n$ creeping from 0 toward 1. From the same wild 5, it says 1.34 after 10 pulls, against 2.61, and from then on it moves exactly as the constant step does. [Tracking a moving arm, in the Lab](lab:bandit-drift).
:::

## Textbook

### The step size {#step}

The incremental mean moves an estimate a fraction $1/n$ of the way toward each new reward ([[incremental-mean]]). The fraction is called the **step size**, and nothing forces it to be $1/n$. Write $\alpha_n$ for the step size used with the $n$-th reward:

$$\val{Q_{n+1}} = \val{Q_n} + \alpha_n\,\big[\,\rew{R_n} - \val{Q_n}\,\big]. \label{update}$$

With $\alpha_n = 1/n$ the estimate is the sample average. The other common choice is a constant, $\alpha_n = \alp \in (0, 1]$, used by most of the methods in the atlas. In the atlas $\alp$ always denotes a step size, drawn in ink like every setting the user chooses. The two choices answer different questions: the average of everything seen, or the current value of something that may change.

### Constant step sizes weight the recent past {#weights}

Unrolling \ref{update} with a constant $\alp$ shows what the estimate is made of:

$$\begin{aligned} \val{Q_{n+1}} &= \alp\,\rew{R_n} + (1 - \alp)\,\val{Q_n} = \alp\,\rew{R_n} + (1 - \alp)\,\alp\,\rew{R_{n-1}} + (1 - \alp)^2\,\val{Q_{n-1}} \\ &= (1 - \alp)^n\,\val{Q_1} + \sum_{i=1}^{n} \alp\,(1 - \alp)^{n-i}\,\rew{R_i}. \end{aligned} \label{unrolled}$$

The weights add up to one, since $(1 - \alp)^n + \sum_{i=1}^{n} \alp (1 - \alp)^{n-i} = (1-\alp)^n + 1 - (1 - \alp)^n = 1$, so the estimate is a weighted average of the initial estimate and the rewards. But the weights are not equal: reward $\rew{R_i}$ counts with $\alp (1 - \alp)^{n-i}$, which decays exponentially with its age. This is an **exponential recency-weighted average**. A reward's weight falls to about a third, a factor $e^{-1}$, after $1/\alp$ newer rewards: 10 rewards for $\alp = 0.1$, 2 for $\alp = 0.5$ (\ref{fig-weights}). With $\alp = 1$ the estimate is simply the last reward.

::: figure {#fig-weights}
{{step-weights}}
The weight of each of the last 20 rewards in the estimate $\val{Q_{21}}$. Sample averages ($1/n$) weigh all rewards equally. A constant step size weighs recent rewards most: $\alp = 0.1$ remembers about the last 10, $\alp = 0.5$ about the last 2.
:::

### Bias from the first estimate {#bias}

Taking expectations in \ref{unrolled}, for rewards with mean $\val{q_*}$,

$$\mathbb{E}\big[\val{Q_{n+1}}\big] = (1 - \alp)^n\,\val{Q_1} + \big(1 - (1 - \alp)^n\big)\,\val{q_*}. \label{bias-eq}$$

The estimate is **biased** toward the initial value $\val{Q_1}$, and the bias shrinks by a factor $(1 - \alp)$ with each reward. With sample averages it disappears at the first reward, since then $\val{Q_2} = \rew{R_1}$. In practice the bias is often harmless, and it can even be useful: the initial values are a way to supply prior knowledge, or to encourage exploration ([[optimistic-init]]). It can also be removed (\ref{unbiased}).

### Noise that never fades {#noise}

A sample average becomes ever more precise, with variance $\sigma^2 / n$. A constant step size does not. With independent rewards of variance $\sigma^2$, \ref{unrolled} gives

$$\operatorname{Var}\big[\val{Q_{n+1}}\big] = \sum_{i=1}^{n} \alp^2 (1 - \alp)^{2(n-i)}\,\sigma^2 \;\longrightarrow\; \frac{\alp^2\,\sigma^2}{1 - (1 - \alp)^2} = \frac{\alp}{2 - \alp}\,\sigma^2 \quad (n \to \infty). \label{noise-eq}$$

The estimate keeps fluctuating around the true value forever, with a standard deviation of about $0.23\,\sigma$ for $\alp = 0.1$ and $0.58\,\sigma$ for $\alp = 0.5$: it effectively averages only the last few rewards. This is the price of a constant step size, and its benefit is the same property seen from the other side (\ref{tracking}).

### When estimates converge {#robbins-monro}

Which sequences of step sizes make the estimates converge? Stochastic approximation gives the classical answer.

::: theorem {#thm-rm} Stochastic approximation
Let the rewards be independent with mean $\val{q_*}$ and finite variance. Then the estimates of \ref{update} converge to $\val{q_*}$ with probability 1 if
$$\sum_{n=1}^{\infty} \alpha_n = \infty \qquad\text{and}\qquad \sum_{n=1}^{\infty} \alpha_n^2 < \infty. \label{conditions}$$
:::

Robbins and Monro (1951) proved convergence in probability under these conditions, and Blum (1954) strengthened it to convergence with probability 1. The first condition makes the steps large enough, in total, to overcome any initial error and any run of bad luck. The second makes them small enough, eventually, for the noise to average out. The sample-average choice $\alpha_n = 1/n$ meets both, since the harmonic series diverges and $\sum 1/n^2 = \pi^2/6$. So does $\alpha_n = 1/n^p$ for $1/2 < p \le 1$. A constant step size meets the first but not the second, which is why its estimates never settle: they converge in distribution, to a spread around $\val{q_*}$ of the size given by \ref{noise-eq}, but not to a point.

The same two conditions reappear in the convergence theorems of [[q-learning]], [[sarsa]] and [[td0]], with one step-size sequence for each state or state–action pair. In practice, sequences that meet them often learn slowly and need careful tuning, so they serve theory more than applications, where constant step sizes are the rule.

### Tracking a moving target {#tracking}

If the true value changes over time, the problem is **nonstationary** ([[nonstationary]]), and the sample average is the wrong tool: after a thousand rewards, a new reward moves it by a thousandth of its error, so it barely follows a change. A constant step size never stops listening. After the value changes, the estimate forgets the old value geometrically, at rate $(1 - \alp)$ per reward, and catches up within a few times $1/\alp$ rewards.

::: figure {#fig-steps}
{{step-sizes}}
Sample averages and a constant step size on two arms with standard-normal noise. Top, an arm whose mean stays at 1: the sample average ($1/n$) settles, while $\alp = 0.1$ keeps wobbling around the mean. Bottom, an arm whose mean wanders (the thin line): the sample average falls behind, while $\alp = 0.1$ follows with a short lag.
:::

\ref{fig-steps} shows the trade-off. A small step size averages more rewards: low noise but a slow reaction. A large step size reacts quickly but believes every lucky reward; with $\alp = 0.5$ the estimate's standard deviation around the true value would be $0.58$, by \ref{noise-eq}. The best $\alp$ balances how fast the target moves against how noisy the rewards are. On the drifting testbed, ε-greedy with $\alp = 0.1$ clearly beats sample averages ([[nonstationary]]).

### A constant step size without the bias {#unbiased}

The bias of \ref{bias-eq} can be removed while keeping the recency weighting (Sutton & Barto, Exercise 2.7). Keep a second number $\bar{o}$, starting at $\bar{o}_0 = 0$, and use the step size

$$\beta_n = \frac{\alp}{\bar{o}_n}, \qquad \text{where} \quad \bar{o}_n = \bar{o}_{n-1} + \alp\,(1 - \bar{o}_{n-1}). \label{beta}$$

Then $\bar{o}_n = 1 - (1 - \alp)^n$, so $\beta_1 = 1$ and the first reward replaces $\val{Q_1}$ entirely, while $\beta_n \to \alp$ as $n$ grows. The resulting estimate is $\sum_{i=1}^{n} \alp (1 - \alp)^{n-i} \rew{R_i} / \bar{o}_n$: the recency-weighted average of the rewards alone, with weights renormalized to add up to one. The same correction appears in the Adam optimizer, which divides its running averages of gradients by $1 - \beta^n$ for exactly this reason.

### Step sizes beyond bandits {#beyond}

Every learning method in the atlas has a step size. In Monte Carlo and temporal-difference methods it decides how far a state's estimate moves toward its target, and the same trade-off holds: small values learn slowly and steadily, large ones quickly and noisily, and too large a value makes estimates oscillate. With function approximation the step size becomes the learning rate of stochastic gradient descent ([[value-error]]), where too large a value can make the estimates diverge instead of merely wobble.

## Card

### Idea

The step size $\alp$ says how far an estimate moves toward each new target. With $1/n$ the estimate is the average of everything seen, steadier with every reward. With a constant $\alp$ it is an average that forgets: recent rewards count most, so it keeps up when things change, at the cost of never settling.

::: analogy
A thermostat reading. One that averages every reading since it was installed is steady but useless when the weather turns. One that trusts only the last few readings follows the weather, and also every draft from an open door.
:::

### In symbols {#formula}

$$\val{Q_{n+1}} = \val{Q_n} + \alp\,\big[\,\rew{R_n} - \val{Q_n}\,\big] = (1 - \alp)^n\,\val{Q_1} + \sum_{i=1}^{n} \alp\,(1 - \alp)^{n-i}\,\rew{R_i}$$

Converges to the true value if $\sum_n \alpha_n = \infty$ and $\sum_n \alpha_n^2 < \infty$ (true for $1/n$, false for a constant).

### Choosing it {#choosing}

| Step size | Estimate | Good for |
| --- | --- | --- |
| $1/n$ | the plain average, converges | values that never change |
| small constant, such as 0.01 | slow, steady, long memory | slowly changing values, noisy rewards |
| large constant, such as 0.5 | fast, jumpy, short memory | quickly changing values, quiet rewards |

### Why it matters {#why}

It is the most common knob in reinforcement learning. Every method has one, and the same trade-off between noise and reaction speed decides how it should be set. [See it](lab:bandit-drift)

### Pitfalls

- Expecting a constant step size to converge: its estimates keep moving, with a spread that grows with $\alp$.
- Using sample averages when the world can change: after many rewards they hardly move at all.
- Too large a step size with function approximation: estimates can oscillate or diverge.

### Check yourself {#check}

::: question
With a constant $\alp = 0.1$, what weight does the reward from 10 pulls ago have, compared with the latest one?
---
$0.9^{10} \approx 0.35$ times as much: the weights decay by a factor $1 - \alp$ per reward.
:::

::: question
Why does a constant step size violate the Robbins–Monro conditions?
---
$\sum_n \alp^2$ is infinite when $\alp$ is constant, so the noise never averages out and the estimate never settles on a single value.
:::

::: question
When is a constant step size better than a sample average?
---
When the true values change over time. The average gives a reward from long ago as much weight as the latest one and falls behind; a constant step size follows the change.
:::
