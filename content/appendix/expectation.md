+++
title = "Expected value"
summary = "The average of a random quantity, each value weighted by its probability: what you would get per try in the long run. Expectations of sums are sums of expectations."
group = "Probability"
sources = [
  { text = "Blitzstein & Hwang (2019), Introduction to Probability, 2nd ed., CRC Press, chapter 4", url = "https://projects.iq.harvard.edu/stat110" },
]
+++

The **expected value** of a random variable ([[a:random-variable|random variables]]) is its average, with each possible value weighted by its probability. It is written $\mathbb{E}[X]$, and read “the expectation of $X$”.

### Definition {#definition}

For a random variable $X$ that takes the values $x$ with probabilities $p(x)$,

$$\mathbb{E}[X] = \sum_x x\,p(x). \label{exp}$$

::: example {#ex-die}
A fair die: $\mathbb{E}[X] = 1 \cdot \tfrac16 + 2 \cdot \tfrac16 + \cdots + 6 \cdot \tfrac16 = 3.5$.
:::

::: example {#ex-robot}
The reward of searching with a low battery in the recycling robot ([[mdp]]) is $\rew{+2}$ with probability 0.6 and $\rew{-3}$ with probability 0.4, so its expectation is $0.6 \times 2 + 0.4 \times (-3) = 0$.
:::

The second example shows that the expected value need not be a value the variable can take: no single search pays 0. It is not the typical outcome but the **long-run average per try**: over many searches, the rewards average out to about 0 per search ([[a:law-of-large-numbers|the law of large numbers]]). This is why a quantity that is random, like the return, is judged by its expectation ([[return]]).

### Expectations of functions {#functions}

To find the expectation of a quantity computed from $X$, weight the computed values by the same probabilities:

$$\mathbb{E}[f(X)] = \sum_x f(x)\,p(x). \label{lotus}$$

For a die, $\mathbb{E}[X^2] = (1 + 4 + 9 + 16 + 25 + 36)/6 \approx 15.17$, which is not $3.5^2 = 12.25$: in general $\mathbb{E}[f(X)]$ is not $f(\mathbb{E}[X])$. Choosing the action whose *expected* outcome is best is therefore not the same as choosing the action that is best for the *expected* state.

### Linearity {#linearity}

The most useful property of expectations: for any random variables $X$ and $Y$ and numbers $a$ and $b$,

$$\mathbb{E}[aX + bY] = a\,\mathbb{E}[X] + b\,\mathbb{E}[Y]. \label{linear}$$

It holds whether or not $X$ and $Y$ are independent, and it extends to sums of any number of terms. It is what lets the expectation of the return be taken term by term:

$$\mathbb{E}[\rew{G_t}] = \mathbb{E}[\rew{R_{t+1}}] + \gam\,\mathbb{E}[\rew{R_{t+2}}] + \gam^2\,\mathbb{E}[\rew{R_{t+3}}] + \cdots$$

and the Bellman equation be split into an expected reward and an expected next value ([[bellman]]). A constant is its own expectation, $\mathbb{E}[c] = c$, and expectations respect order: if $X \le Y$ always, then $\mathbb{E}[X] \le \mathbb{E}[Y]$.

### Subscripts and conditions {#notation}

Two decorations appear in RL formulas.

- A **subscript** names whose choices the randomness comes from. $\mathbb{E}_\pi[\ldots]$ means the expectation when actions are drawn from the policy $\pol{\pi}$ and outcomes from the environment's dynamics. Changing the policy changes the probabilities, and so the expectation.
- A **condition** after a bar restricts the average to the cases where something is known: $\mathbb{E}_\pi[\rew{G_t} \mid S_t = s]$ averages the return only over the times the agent is in $s$ ([[a:conditional-expectation|conditional expectation]]). That is exactly the value of $s$ ([[value-functions]]).

### Continuous variables {#continuous}

For a continuous random variable with density $f$, the sum becomes an integral, $\mathbb{E}[X] = \int x\,f(x)\,dx$, with the same meaning and the same properties ([[a:probability-density|densities]]).
