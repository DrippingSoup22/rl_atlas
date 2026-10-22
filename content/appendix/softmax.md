+++
title = "The softmax function"
summary = "Turns a list of scores into probabilities that add up to 1, giving higher scores more weight. A temperature sets how strongly: from always picking the best to picking uniformly at random."
group = "Calculus and optimization"
sources = [
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., §2.8 and §13.1", url = "http://incompleteideas.net/book/the-book-2nd.html" },
]
+++

An agent that has a score for each action, such as an action value or a learned preference, often needs probabilities instead: to choose at random, but favoring the better actions. The **softmax** function does this conversion.

### Definition {#definition}

Given scores $x_1, \ldots, x_n$ and a **temperature** $\tau > 0$, the softmax gives option $i$ the probability

$$\sigma_i = \frac{e^{x_i/\tau}}{\sum_{j=1}^{n} e^{x_j/\tau}}. \label{softmax}$$

Each score is turned into a positive weight by the exponential function ([[a:exponential-logarithm|exponentials]]), and the weights are divided by their total so that they add up to 1 ([[a:probability|probability]]). A higher score always gets a higher probability, but every option keeps some.

::: example {#ex-values}
Scores 1, 2 and 3, with $\tau = 1$: the weights are $e^1 \approx 2.72$, $e^2 \approx 7.39$ and $e^3 \approx 20.09$, with total 30.19, so the probabilities are about 0.09, 0.24 and 0.67. With $\tau = 0.5$ the scores are effectively doubled and the probabilities become 0.02, 0.12 and 0.87. With $\tau = 5$ they become 0.27, 0.33 and 0.40.
:::

### The temperature {#temperature}

The temperature sets how sharply the softmax prefers the best option.

- As $\tau \to 0$, the largest score dominates completely, and the softmax puts all its probability on the best option: it becomes the **greedy** choice, the argmax ([[a:max-argmax|max and argmax]]).
- As $\tau \to \infty$, all the weights approach $e^0 = 1$, and the softmax becomes **uniform**: every option equally likely.

Between the two, the softmax explores in proportion to how good options look: a nearly-best option is tried often, a clearly bad one rarely. That is its advantage over ε-greedy, which gives the worst option as much of its random share as the second best ([[explore-exploit]]).

### Only differences matter {#shift}

Adding the same constant $c$ to every score changes nothing: every weight is multiplied by $e^{c/\tau}$, and the factor cancels between the top and the bottom of \ref{softmax}. The probabilities depend only on how much the scores differ, divided by the temperature. A softmax over action values therefore behaves the same whatever the overall level of the values, but it depends on their scale: doubling all the values has the same effect as halving the temperature. (Computer programs use the shift to avoid overflow: they subtract the largest score from all of them before taking exponentials.)

### Where it is used {#atlas}

In the atlas the softmax turns action values into an exploring policy, the Boltzmann policy ([[policy]]); turns learned preferences into the policy of a gradient bandit and of policy-gradient methods for discrete actions; and, as the name *softmax* suggests, serves as a smooth stand-in for the max wherever a function must change gradually rather than jump.
