+++
title = "Sample averages"
summary = "The average of many independent tries settles on the expected value: the law of large numbers. It is why an agent can learn an expectation from experience, and the incremental form shows how to update an average one sample at a time."
group = "Probability"
sources = [
  { text = "Blitzstein & Hwang (2019), Introduction to Probability, 2nd ed., CRC Press, chapter 10", url = "https://projects.iq.harvard.edu/stat110" },
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., §2.4 (incremental implementation)", url = "http://incompleteideas.net/book/the-book-2nd.html" },
]
+++

An expected value ([[a:expectation|expectation]]) is defined with probabilities, which an agent does not know. What an agent can do is try, observe and average. The law of large numbers says that this is enough.

### The law of large numbers {#law}

Let $X_1, X_2, \ldots$ be independent samples of the same random quantity, with expected value $\mu$. Their **sample average** after $n$ tries is

$$\bar X_n = \frac{1}{n} \sum_{i=1}^{n} X_i. \label{mean}$$

The **law of large numbers** says that $\bar X_n$ converges to $\mu$ as $n$ grows ([[a:limits|limits]]): with enough samples, the average is as close to the expectation as one likes, with probability 1.

How close, after $n$ samples? The standard deviation of $\bar X_n$ is $\sigma/\sqrt{n}$, where $\sigma$ is the standard deviation of one sample ([[a:variance|variance]]). So the error shrinks like $1/\sqrt{n}$: a hundred samples are ten times more precise than one, and another ten times needs ten thousand. Averages converge, but slowly, and more slowly the noisier the samples.

In reinforcement learning, this law is the justification of the simplest way to estimate a value: average the returns observed after visits to a state ([[value-functions]]). It is also why one try is not enough to judge an action ([[be-the-agent]]).

### Updating an average one sample at a time {#incremental}

An agent receives its samples one by one and should not have to store them all. The average can be updated in place. Writing $Q_n$ for the average of the first $n - 1$ samples and $X_n$ for the new one,

$$Q_{n+1} = \frac{1}{n} \sum_{i=1}^{n} X_i = \frac{1}{n}\big(X_n + (n-1)\,Q_n\big) = Q_n + \frac{1}{n}\big(X_n - Q_n\big). \label{incremental}$$

Read the last form as a rule: **new estimate = old estimate + step size × (sample − old estimate)**. The difference $X_n - Q_n$ is the error of the old estimate on the new sample; the estimate moves a fraction $1/n$ of the way toward the sample. This one line is the template of almost every update rule in the atlas, with other step sizes and other targets in place of $1/n$ and $X_n$.

::: example {#ex-update}
Samples 4, 8, 6. Start from $Q_1 = 0$ (the value before any sample does not matter, since the first step size is 1): $Q_2 = 0 + 1 \cdot (4 - 0) = 4$, then $Q_3 = 4 + \tfrac12 (8 - 4) = 6$, then $Q_4 = 6 + \tfrac13 (6 - 6) = 6$, the average of the three.
:::

### A constant step size {#constant}

Replacing $1/n$ by a constant $\alp$ between 0 and 1 gives $Q_{n+1} = Q_n + \alp\,(X_n - Q_n)$. The estimate then no longer equals the plain average: it is a weighted average in which the sample from $k$ steps ago has weight $\alp(1 - \alp)^k$, so recent samples count more ([[a:geometric-series|geometric series]]). It never settles exactly, since each new sample keeps moving it, but it follows a target that changes over time, which the plain average does not.
