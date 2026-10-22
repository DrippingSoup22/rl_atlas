+++
title = "Variance and standard deviation"
summary = "How far a random quantity typically lands from its average. Variances of independent pieces add up, which is why long sums of rewards are noisy and why averages of many samples are not."
group = "Probability"
sources = [
  { text = "Blitzstein & Hwang (2019), Introduction to Probability, 2nd ed., CRC Press, chapter 4", url = "https://projects.iq.harvard.edu/stat110" },
]
+++

Two random quantities can have the same average and behave very differently: a sure reward of 5, and a coin flip between 0 and 10. The **variance** measures the difference: how widely the values spread around their average.

### Definition {#definition}

For a random variable $X$ with mean $\mu = \mathbb{E}[X]$ ([[a:expectation|expectation]]), the variance is the expected squared distance from the mean:

$$\operatorname{Var}[X] = \mathbb{E}\big[(X - \mu)^2\big] = \mathbb{E}[X^2] - \mu^2. \label{var}$$

Squaring makes every deviation count positively, and large deviations count much more than small ones. The second form, which follows by expanding the square and using linearity, is often quicker to compute.

The variance is in squared units (squared rewards, for instance), so the spread is usually reported by its square root, the **standard deviation** $\sigma = \sqrt{\operatorname{Var}[X]}$, which is in the same units as $X$. Most values of a random variable fall within a couple of standard deviations of its mean.

::: example {#ex-coin}
The sure reward of 5 has variance 0. The coin flip between 0 and 10 has mean 5 and variance $\tfrac12 (0 - 5)^2 + \tfrac12 (10 - 5)^2 = 25$, standard deviation 5: a typical outcome misses the mean by 5.
:::

### Rules {#rules}

- A constant shift changes nothing, and a factor comes out squared: $\operatorname{Var}[aX + b] = a^2 \operatorname{Var}[X]$.
- For **independent** random variables ([[a:conditional-probability|independence]]), variances add: $\operatorname{Var}[X + Y] = \operatorname{Var}[X] + \operatorname{Var}[Y]$. Without independence a correction term appears, the covariance.

### Why long returns are noisy {#returns}

A return adds up many random rewards ([[return]]). Even if every reward had a modest variance and they were independent, the variance of their sum would grow with every term added; with discounting, the $k$-th reward's contribution is scaled by $\gam^{2k}$, so a $\gam$ close to 1 lets many terms contribute ([[discount]]). In practice the rewards of one episode are not independent, but the conclusion stands: the longer the stretch of future a return covers, the more it varies from one episode to the next. Methods that estimate values from whole returns need many episodes to settle; methods that bootstrap from one step trade some bias for much less variance.

### Why averages help {#averages}

Averaging works the other way. The mean of $n$ independent samples of $X$, each with variance $\sigma^2$, has variance

$$\operatorname{Var}\Big[\frac{1}{n} \sum_{i=1}^{n} X_i\Big] = \frac{1}{n^2} \cdot n\,\sigma^2 = \frac{\sigma^2}{n}, \label{mean-var}$$

so its standard deviation is $\sigma / \sqrt{n}$. Four times as many samples halve the typical error. This is the quantitative side of the law of large numbers ([[a:law-of-large-numbers|sample averages]]), and the reason a single episode is a poor judge of a policy while a hundred are a fair one.
