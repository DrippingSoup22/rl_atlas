+++
title = "Limits and convergence"
summary = "A sequence converges to a limit when its terms end up as close to it as one likes and stay there. Infinite sums, the convergence of learning algorithms and the long run are all defined this way."
group = "Analysis"
sources = [
  { text = "Stewart, Calculus: Early Transcendentals, §2.2 and §11.1 (sequences)" },
]
+++

Many statements in the atlas are about what happens “in the long run” or “eventually”: an infinite sum of rewards, an estimate that settles on the true value, an iteration that reaches a fixed point. All of them rest on one idea, the limit of a sequence.

### Sequences {#sequences}

A **sequence** is an endless list of numbers $x_1, x_2, x_3, \ldots$, one for each step $n$: the estimate of a value after $n$ updates, the sum of the first $n$ rewards, the error after $n$ iterations.

### Convergence {#convergence}

A sequence **converges** to a number $L$, its **limit**, if its terms end up arbitrarily close to $L$ and stay there. Precisely: for every tolerance $\varepsilon > 0$, however small, there is a step $N$ after which every term is within $\varepsilon$ of $L$,

$$\lvert x_n - L \rvert < \varepsilon \quad \text{for all } n \ge N. \label{def}$$

This is written $\lim_{n \to \infty} x_n = L$, or $x_n \to L$, read “$x_n$ tends to $L$”. The definition does not care about the first terms, however wild; it only asks what the sequence does eventually. A sequence that does not converge **diverges**: it may grow without bound, like $1, 2, 3, \ldots$, or keep jumping about, like $1, -1, 1, -1, \ldots$

### The basic example {#powers}

The powers of a number between $-1$ and $1$ tend to zero:

$$x^n \to 0 \qquad \text{when } \lvert x \rvert < 1. \label{powers}$$

Each power is a fixed fraction $\lvert x \rvert$ of the one before, so the terms keep shrinking by the same factor and pass below any tolerance. With $x = 0.9$: $0.9^{10} \approx 0.35$, $0.9^{50} \approx 0.005$, $0.9^{100} \approx 0.00003$. This simple fact is behind the finiteness of discounted returns ([[a:geometric-series|geometric series]]) and the convergence of dynamic programming ([[a:contraction|contractions]]): an error multiplied by $\gam < 1$ at every step tends to zero.

### Infinite sums {#series}

An infinite sum $\sum_{k=0}^{\infty} a_k$ is defined as the limit of its **partial sums** $S_n = a_0 + a_1 + \cdots + a_n$. If the partial sums converge, the sum is their limit; otherwise the sum has no value. The discounted return $\sum_k \gam^k \rew{R_{t+k+1}}$ is such a sum: it has a value whenever the rewards are bounded and $\gam < 1$ ([[return]]).

### Averages in the long run {#averages}

Limits also define long-run averages. The average reward per step of a continuing task is

$$\lim_{h \to \infty} \frac{1}{h}\,\mathbb{E}\Big[\sum_{t=1}^{h} \rew{R_t}\Big],$$

the limit, as the horizon $h$ grows, of the expected reward per step over the first $h$ steps ([[discount]]).

### Convergence of random quantities {#random}

When the terms are random, as the estimates of a learning agent are, “converges” needs a qualifier. The atlas's convergence theorems use convergence **with probability 1**: apart from a set of runs whose total probability is zero, every run's sequence of estimates converges in the sense above. The law of large numbers is the first example ([[a:law-of-large-numbers|sample averages]]).
