+++
title = "Geometric series"
summary = "The sum 1 + x + x² + x³ + … of the powers of a number. It is finite exactly when the number lies strictly between −1 and 1, and then it equals 1/(1 − x)."
group = "Sums and series"
sources = [
  { text = "Stewart, Calculus: Early Transcendentals, §11.2 (Series)" },
]
+++

A **geometric series** is a sum in which each term is the previous one multiplied by the same number $x$, the *ratio*:

$$1 + x + x^2 + x^3 + \cdots = \sum_{k=0}^{\infty} x^k.$$

It comes up whenever something is scaled down by the same factor at every step. In reinforcement learning, that something is a reward that lies one more step in the future and is therefore multiplied by one more factor $\gam$.

### The finite sum {#finite}

Start with the first $n$ terms, $S_n = 1 + x + x^2 + \cdots + x^{n-1}$. Multiplying by $x$ shifts every term one place:

$$x\,S_n = x + x^2 + \cdots + x^{n-1} + x^n.$$

The two sums share all their terms except the first of $S_n$ and the last of $x\,S_n$, so subtracting one from the other leaves only those two: $S_n - x\,S_n = 1 - x^n$. For $x \ne 1$, dividing by $1 - x$ gives

$$S_n = \sum_{k=0}^{n-1} x^k = \frac{1 - x^n}{1 - x}. \label{finite}$$

For $x = 1$ every term is 1, and the sum is simply $n$.

### The infinite sum {#infinite}

An infinite sum is defined as the [[a:limits|limit]] of its finite sums as $n$ grows. In \ref{finite} only $x^n$ depends on $n$. If $|x| < 1$, the powers $x^n$ shrink towards 0 (each one is a fixed fraction of the previous one), so

$$\sum_{k=0}^{\infty} x^k = \frac{1}{1 - x} \qquad \text{for } |x| < 1. \label{infinite}$$

If $|x| \ge 1$, the terms do not shrink, and the sum has no finite value: for $x = 1$ it grows without bound, for $x = -1$ it jumps between 1 and 0 forever.

A sum that starts later, or with another first term $a$, follows by taking out the common factor:

$$\sum_{k=m}^{\infty} a\,x^k = a\,x^m \sum_{j=0}^{\infty} x^j = \frac{a\,x^m}{1 - x}.$$

::: example {#ex-reward}
A reward of 1 at every step, discounted by $\gam = 0.9$, adds up to

$$1 + 0.9 + 0.9^2 + \cdots = \frac{1}{1 - 0.9} = 10.$$

With $\gam = 0.99$ the same rewards add up to 100. The number $1/(1 - \gam)$ is often read as the *effective horizon*: discounting with $\gam$ counts roughly the next $1/(1-\gam)$ rewards.
:::

### Where it is used {#atlas}

- The discounted return $\rew{G_t} = \sum_k \gam^k \rew{R_{t+k+1}}$ is finite when $\gam < 1$ and the rewards are bounded by some $R_{\max}$: it is at most $R_{\max}/(1 - \gam)$, by \ref{infinite} ([[discount]]).
- The error of an iteration that shrinks by a factor $\gam$ at every step falls like $\gam^k$, a geometric sequence ([[a:contraction|contractions]]).
