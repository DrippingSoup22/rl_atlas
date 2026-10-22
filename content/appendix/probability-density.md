+++
title = "Densities and the normal distribution"
summary = "For quantities that can take any value in a range, such as a torque, probabilities come from a density: the area under its curve. The bell-shaped normal distribution is the most common."
group = "Probability"
sources = [
  { text = "Blitzstein & Hwang (2019), Introduction to Probability, 2nd ed., CRC Press, chapter 5", url = "https://projects.iq.harvard.edu/stat110" },
]
+++

A die has six outcomes, and each has a probability. A robot's joint torque or a car's position can take any value in a range, and there are infinitely many such values. Each single value then has probability 0, and probabilities are given to *ranges* of values through a density.

### Densities {#density}

A **probability density** is a function $f(x) \ge 0$ whose curve encloses an area of 1 over all values. The probability that the random quantity $X$ falls between $a$ and $b$ is the area under the curve between them:

$$\Pr\{a \le X \le b\} = \int_a^b f(x)\,dx. \label{area}$$

The integral sign is the continuous version of a sum ([[a:sums-products|sums]]): it adds up $f(x)$ over a continuum of tiny intervals of width $dx$. A density is not itself a probability, and it can be larger than 1; what it says is where values are concentrated. Where the curve is high, values are likely; where it is low, rare.

Everything said about distributions over finite sets carries over with integrals in place of sums. The total is 1, $\int f(x)\,dx = 1$; the expectation is $\mathbb{E}[X] = \int x\,f(x)\,dx$ ([[a:expectation|expectation]]); conditional densities and marginals work the same way ([[a:probability|probability]]). This is why the theory of the atlas, written for finite MDPs with sums, extends to continuous states and actions: the sums over next states become integrals against a density $p(s' \mid s, a)$ ([[mdp]]).

### The normal distribution {#normal}

The most used density is the bell-shaped **normal** or **Gaussian** distribution, written $\mathcal{N}(\mu, \sigma^2)$:

$$f(x) = \frac{1}{\sigma\sqrt{2\pi}}\,\exp\Big(-\frac{(x - \mu)^2}{2\sigma^2}\Big). \label{normal}$$

It has two parameters. The **mean** $\mu$ is where the bell is centered; the **standard deviation** $\sigma$ is how wide it is ([[a:variance|variance]]). About 68% of the area lies within $\sigma$ of the mean and about 95% within $2\sigma$. The formula looks forbidding, but only the exponent matters for its shape: the density falls off with the square of the distance from the mean, measured in units of $\sigma$ ([[a:exponential-logarithm|exponentials]]).

The normal distribution appears so often for a reason: a sum of many small independent random effects is approximately normal, whatever the effects are (the central limit theorem). In the atlas it has two main uses:

- as a **policy for continuous actions**: the policy computes a mean and a spread from the state and samples the action from the corresponding normal distribution ([[policy]]);
- as **exploration noise** added to an action, and as a model of noisy rewards, as in the 10-armed testbed of the bandit chapters.
