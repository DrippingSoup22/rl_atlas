+++
summary = "Keep an average up to date with one number and one line of arithmetic: move the old average a fraction 1/n of the way toward the new value. The shape of almost every update in reinforcement learning."
prereqs = ["k-armed-bandit"]
lab = "bandit-epsilon"
sources = [
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., §2.4", url = "http://incompleteideas.net/book/the-book-2nd.html" },
  { text = "Robbins & Monro (1951), A stochastic approximation method, Annals of Mathematical Statistics 22", url = "https://doi.org/10.1214/aoms/1177729586" },
  { text = "Welford (1962), Note on a method for calculating corrected sums of squares and products, Technometrics 4", url = "https://doi.org/10.1080/00401706.1962.10490022" },
]
+++

## Textbook

### Averages without a memory {#incremental}

A bandit agent estimates the value of an action by the average of the rewards it has paid ([[k-armed-bandit]]). Computed directly, the average needs every past reward: memory and work that grow with each pull. It does not have to be computed that way.

Fix one action and write $\rew{R_i}$ for the reward it paid the $i$-th time it was chosen, and $\val{Q_n}$ for the estimate after it has been chosen $n - 1$ times:

$$\val{Q_n} = \frac{\rew{R_1} + \rew{R_2} + \cdots + \rew{R_{n-1}}}{n - 1}. \label{average}$$

When the $n$-th reward arrives, the new average can be written in terms of the old one:

$$\begin{aligned} \val{Q_{n+1}} &= \frac{1}{n} \sum_{i=1}^{n} \rew{R_i} = \frac{1}{n}\Big(\rew{R_n} + \sum_{i=1}^{n-1} \rew{R_i}\Big) = \frac{1}{n}\big(\rew{R_n} + (n-1)\,\val{Q_n}\big) \\ &= \val{Q_n} + \frac{1}{n}\,\big[\,\rew{R_n} - \val{Q_n}\,\big]. \end{aligned} \label{inc-mean}$$

The last line is the **incremental mean**. It needs only the current estimate and the count $n$, and one subtraction, one division and one addition per reward. It holds even for $n = 1$, whatever $\val{Q_1}$ is: the step $1/1$ moves the estimate all the way to the first reward, $\val{Q_2} = \rew{R_1}$. So the initial estimate is forgotten after one reward, a fact that matters for [[optimistic-init]].

### The general form {#form}

The incremental mean has a shape that recurs throughout reinforcement learning:

$$\textit{NewEstimate} \leftarrow \textit{OldEstimate} + \textit{StepSize}\,\big[\,\textit{Target} - \textit{OldEstimate}\,\big]. \label{general}$$

The bracket is the **error** of the estimate, the difference between a target and the current guess. Each update reduces the error by a fraction, the **step size**. In the incremental mean the target is the latest reward and the step size is $1/n$; it shrinks as evidence accumulates, so each new reward counts less than the one before. Other choices of step size are studied in [[step-size]].

Almost every method in the atlas is this update with a different target:

| Method | Estimate | Target | Step size |
| --- | --- | --- | --- |
| sample average | $\val{Q(a)}$ | the reward $\rew{R}$ | $1/N(a)$ |
| [[mc-prediction]] | $\val{V(S_t)}$ | the return $\rew{G_t}$ | $1/N(S_t)$ or $\alp$ |
| [[td0]] | $\val{V(S_t)}$ | $\rew{R_{t+1}} + \gam\,\val{V(S_{t+1})}$ | $\alp$ |
| [[sarsa]] | $\val{Q(S_t, A_t)}$ | $\rew{R_{t+1}} + \gam\,\val{Q(S_{t+1}, A_{t+1})}$ | $\alp$ |
| [[q-learning]] | $\val{Q(S_t, A_t)}$ | $\rew{R_{t+1}} + \gam \max_a \val{Q(S_{t+1}, a)}$ | $\alp$ |

The target is the thing that changes. A reward is a noisy but unbiased sample of the value. Later targets mix rewards with other estimates, which brings new questions, but the update itself stays the same.

### Why the average converges {#convergence}

If the rewards are independent draws from a distribution with mean $\val{q_*}$ and variance $\sigma^2$, the sample average $\val{Q_{n+1}}$ has mean $\val{q_*}$ and variance $\sigma^2 / n$:

$$\mathbb{E}\big[\val{Q_{n+1}}\big] = \val{q_*}, \qquad \operatorname{Var}\big[\val{Q_{n+1}}\big] = \frac{1}{n^2} \sum_{i=1}^{n} \operatorname{Var}\big[\rew{R_i}\big] = \frac{\sigma^2}{n}. \label{variance}$$

So the estimate is unbiased after the first reward, and its typical error, the standard deviation $\sigma / \sqrt{n}$, shrinks with the square root of the number of rewards: four times as many rewards halve the error. By the strong law of large numbers, $\val{Q_n} \to \val{q_*}$ with probability 1 (\ref{fig-average}).

::: figure {#fig-average}
{{sample-average}}
Three runs of the incremental mean on the same arm, whose rewards are normal with mean $\val{q_*} = 1$ and standard deviation 1. Early estimates disagree wildly; after $n$ rewards they mostly lie within $1/\sqrt{n}$ of the true value (the thin lines). Note the logarithmic scale of the horizontal axis.
:::

### The update as a gradient step {#gradient}

The average is also the number that best fits the rewards in the least-squares sense: $\val{Q_{n+1}}$ minimizes $\sum_{i=1}^{n} \tfrac12 (\rew{R_i} - Q)^2$ over $Q$. The incremental form is a step of *stochastic gradient descent* on that loss. The loss of one reward, $\tfrac12 (\rew{R_n} - Q)^2$, has gradient $-(\rew{R_n} - Q)$ with respect to $Q$, and a step of size $\alpha$ against the gradient gives

$$Q \leftarrow Q + \alpha\,\big[\,\rew{R_n} - Q\,\big], \label{sgd}$$

which with $\alpha = 1/n$ is exactly \ref{inc-mean}. This is a first glimpse of how reinforcement learning methods turn into gradient methods when the table of estimates is replaced by a parameterized function ([[value-error]]).

Seen as a sequence of noisy steps toward an unknown mean, the incremental mean is the simplest instance of **stochastic approximation** (Robbins & Monro, 1951): find the point where an expected error is zero, using only noisy samples of that error. The conditions under which such steps converge, for step sizes other than $1/n$, are the subject of [[step-size]].

### Example: the first rewards of an arm {#example}

In the testbed problem of the stories, arm 7 paid $\rew{1.29}$, $\rew{-0.86}$ and $\rew{0.70}$ on its first three pulls. Starting from $\val{Q_1} = 0$:

$$\val{Q_2} = 0 + \tfrac11\,(\rew{1.29} - 0) = 1.29, \qquad \val{Q_3} = 1.29 + \tfrac12\,(\rew{-0.86} - 1.29) \approx 0.215, \qquad \val{Q_4} = 0.215 + \tfrac13\,(\rew{0.70} - 0.215) \approx 0.377,$$

which is $(1.29 - 0.86 + 0.70)/3 \approx 0.377$, the average of the three rewards, as it should be. Its true value is $\val{q_*(7)} = 0.38$.

### Spread as well as mean {#spread}

The same idea keeps other statistics current. Welford's method (Welford, 1962) maintains the mean and the sum of squared deviations $M$ of a stream $x_1, x_2, \ldots$ with one pass and no stored values:

$$\delta = x_n - \bar{x}_{n-1}, \qquad \bar{x}_n = \bar{x}_{n-1} + \frac{\delta}{n}, \qquad M_n = M_{n-1} + \delta\,(x_n - \bar{x}_n), \label{welford}$$

and the variance is $M_n / (n - 1)$. It is numerically far more stable than subtracting the square of the mean from the mean of the squares. Deep reinforcement learning uses running statistics of this kind to normalize observations and rewards ([[normalization]]).

## Card

### Idea

To keep an average current, you do not need the list of values. Move the old average a fraction $1/n$ of the way toward the newest value. Every estimate in reinforcement learning is updated this way, with different targets.

::: analogy
Your average grade after a new exam: if it is the fifth exam, the new grade pulls your average a fifth of the way toward itself. You never need to look up the four old grades.
:::

### In symbols {#formula}

$$\val{Q_{n+1}} = \val{Q_n} + \frac{1}{n}\,\big[\,\rew{R_n} - \val{Q_n}\,\big]$$

$\textit{new estimate} = \textit{old estimate} + \textit{step size} \times (\textit{target} - \textit{old estimate})$: here the target is the reward and the step size is $1/n$.

### Why it matters {#why}

It is the template of every update in the atlas. Monte Carlo methods put the return in place of the reward, TD methods a reward plus an estimate. Learning the template once makes every later algorithm a variation.

### Pitfalls

- Off by one: $\val{Q_{n+1}}$ is the estimate *after* $n$ rewards, and its step size is $1/n$.
- $n$ counts the pulls of this action, not all steps.
- The first reward replaces the initial estimate completely ($1/1 = 1$), so with averages the starting value has no lasting effect.

### Check yourself {#check}

::: question
An arm's average after 9 rewards is 2.0. The 10th reward is 3.0. What is the new average?
---
$2.0 + \tfrac{1}{10}(3.0 - 2.0) = 2.1$.
:::

::: question
How much memory does the incremental mean need per action?
---
Two numbers: the current estimate and the count $n$. The past rewards are never needed again.
:::

::: question
How many times as many rewards are needed to halve the typical error of a sample average?
---
Four times as many: the error shrinks like $1/\sqrt{n}$.
:::
