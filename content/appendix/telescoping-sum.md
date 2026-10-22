+++
title = "Telescoping sums"
summary = "A sum of differences of consecutive terms, in which everything cancels except the first and the last. It is why potential-based reward shaping cannot change the optimal policy."
group = "Sums and series"
+++

A sum **telescopes** when each term is the difference of two consecutive values of the same sequence. Writing it out, every value appears once with a plus sign and once with a minus sign, except the two at the ends, so the long sum collapses, like a telescope folding up, to a single difference:

$$\sum_{k=0}^{n-1} \big(a_{k+1} - a_k\big) = (a_1 - a_0) + (a_2 - a_1) + \cdots + (a_n - a_{n-1}) = a_n - a_0. \label{telescope}$$

::: example {#ex-height}
Climbing a hill and coming back down: the changes of altitude along the way, ups and downs, always add up to the final altitude minus the starting one. A walk that ends where it began has a total change of zero, however winding it was.
:::

### With discounting {#discounted}

The same cancellation works when each step is weighted by $\gam^k$ and the later value is multiplied by $\gam$:

$$\sum_{k=0}^{n-1} \gam^k \big(\gam\,a_{k+1} - a_k\big) = \gam^n a_n - a_0. \label{discounted}$$

To see it, write out the terms: $(\gam a_1 - a_0) + (\gam^2 a_2 - \gam a_1) + (\gam^3 a_3 - \gam^2 a_2) + \cdots$. Each $\gam^k a_k$ appears once with each sign and cancels, leaving the first $-a_0$ and the last $\gam^n a_n$.

If the values $a_k$ stay bounded and $\gam < 1$, then $\gam^n a_n$ shrinks to 0 as $n$ grows ([[a:limits|limits]]), and the infinite sum is simply $-a_0$.

### Where it is used {#atlas}

**Potential-based reward shaping** ([[reward-design]]) adds $\gam\,\Phi(s') - \Phi(s)$ to the reward of every move, for some function $\Phi$ of the state. Along any trajectory these extra rewards are exactly the terms of \ref{discounted} with $a_k = \Phi(S_{t+k})$, so in total they add $-\Phi(S_t)$ to the return, whatever path is taken. Every action from the same state is shifted by the same amount, so the best action stays the best. A loop that returns to its starting state collects $\gam^n \Phi(s) - \Phi(s)$, which is zero without discounting and slightly negative with it: shaping of this form can never be farmed by going round in circles.

Later in the atlas, the same cancellation shows that the error of a Monte Carlo estimate is the discounted sum of the temporal-difference errors along the episode.
