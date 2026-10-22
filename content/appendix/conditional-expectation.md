+++
title = "Conditional expectation"
summary = "The average of a random quantity over only the cases where something is known. Averaging the conditional averages gives back the overall average: the law of total expectation, the step at the heart of every Bellman equation."
group = "Probability"
sources = [
  { text = "Blitzstein & Hwang (2019), Introduction to Probability, 2nd ed., CRC Press, chapter 9", url = "https://projects.iq.harvard.edu/stat110" },
]
+++

The value of a state is an average of returns, but not of all returns: only of those that start in that state. Averages restricted to the cases where something is known are **conditional expectations**.

### Definition {#definition}

The expectation of $X$ given that $Y = y$ uses the conditional probabilities of $X$ given $Y = y$ ([[a:conditional-probability|conditional probability]]) in place of the plain ones ([[a:expectation|expectation]]):

$$\mathbb{E}[X \mid Y = y] = \sum_x x\,\Pr\{X = x \mid Y = y\}. \label{cond-exp}$$

It is a number for each value $y$: the long-run average of $X$ over the occasions on which $Y$ turns out to be $y$.

::: example {#ex-die}
Roll a fair die. Overall, $\mathbb{E}[X] = 3.5$. Given that the roll is even, the possible values are 2, 4 and 6, each with probability $\tfrac13$, so $\mathbb{E}[X \mid \text{even}] = 4$; given that it is odd, $\mathbb{E}[X \mid \text{odd}] = 3$.
:::

The state-value and action-value functions are conditional expectations of the return:

$$\val{v_\pi(s)} = \mathbb{E}_\pi[\rew{G_t} \mid S_t = s], \qquad \val{q_\pi(s,a)} = \mathbb{E}_\pi[\rew{G_t} \mid S_t = s,\ A_t = a].$$

Conditional expectations keep all the properties of ordinary ones, linearity in particular: $\mathbb{E}[X + Y \mid B] = \mathbb{E}[X \mid B] + \mathbb{E}[Y \mid B]$.

### The law of total expectation {#total}

In the example, the overall average 3.5 is the average of the two conditional averages, each weighted by its probability: $\tfrac12 \cdot 4 + \tfrac12 \cdot 3 = 3.5$. This always works. If $Y$ takes the values $y$ with probabilities $\Pr\{Y = y\}$,

$$\mathbb{E}[X] = \sum_y \Pr\{Y = y\}\,\mathbb{E}[X \mid Y = y]. \label{total}$$

In words: to average over everything, first average within each case, then average the cases. The same holds when everything is already conditioned on some event $B$:

$$\mathbb{E}[X \mid B] = \sum_y \Pr\{Y = y \mid B\}\,\mathbb{E}[X \mid B,\ Y = y]. \label{total-cond}$$

This law is the main tool of the atlas's derivations.

- With $X$ the return, $B$ the event $S_t = s$ and $Y$ the action $A_t$, it gives $\val{v_\pi(s)} = \sum_a \pol{\pi(a \mid s)}\,\val{q_\pi(s,a)}$: a state's value is the policy's average of its action values ([[value-functions]]).
- With $Y$ the pair (next state, reward), it splits the expected return after an action into the possible outcomes of that action, weighted by their probabilities. That is the step from the return's recursion to the Bellman equation ([[bellman]]).

### Conditioning on what does not matter {#markov}

Sometimes extra information does not change an average. If $X$ depends on the past only through the present state, then adding facts about the past to the condition leaves the conditional expectation as it was:

$$\mathbb{E}[\rew{G_{t+1}} \mid S_{t+1} = s',\ S_t = s,\ A_t = a] = \mathbb{E}[\rew{G_{t+1}} \mid S_{t+1} = s'].$$

This is the Markov property ([[mdp]]) at work: once the next state is known, how the agent got there tells nothing more about the return to come. Together with the law of total expectation, it is all the Bellman equation needs.
