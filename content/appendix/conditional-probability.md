+++
title = "Conditional probability"
summary = "How likely something is once something else is known: the bar in Pr{A |⁠ B} reads “given”. With it come independence, the chain rule for sequences, and Bayes' rule."
group = "Probability"
sources = [
  { text = "Blitzstein & Hwang (2019), Introduction to Probability, 2nd ed., CRC Press, chapter 2", url = "https://projects.iq.harvard.edu/stat110" },
]
+++

Knowing something changes the odds of other things. A **conditional probability** is the probability of an event computed as if another event were known to have happened.

### Definition {#definition}

For events $A$ and $B$ with $\Pr\{B\} > 0$,

$$\Pr\{A \mid B\} = \frac{\Pr\{A \text{ and } B\}}{\Pr\{B\}}. \label{cond}$$

The vertical bar reads “given”. Knowing that $B$ happened shrinks the world to the outcomes in $B$; the formula measures what share of that smaller world also lies in $A$.

::: example {#ex-die}
Roll a fair die. The probability of a 6 is $\tfrac16$. Given that the roll is even, it is

$$\Pr\{6 \mid \text{even}\} = \frac{\Pr\{6 \text{ and even}\}}{\Pr\{\text{even}\}} = \frac{1/6}{1/2} = \frac13.$$
:::

In the atlas, conditional probabilities are everywhere, usually conditioned on the present: $\pol{\pi(a \mid s)} = \Pr\{A_t = a \mid S_t = s\}$ is the probability of an action *given* the state ([[policy]]), and $p(s', r \mid s, a)$ the probability of an outcome *given* the state and the action ([[mdp]]). For each fixed condition, a conditional probability is an ordinary distribution: $\sum_a \pol{\pi(a \mid s)} = 1$ for every $s$.

### The product rule {#product}

Turned around, \ref{cond} says how to compute the probability that two things both happen:

$$\Pr\{A \text{ and } B\} = \Pr\{B\}\,\Pr\{A \mid B\}. \label{product}$$

Applied again and again, it gives the **chain rule** for a sequence of events: the probability of the whole sequence is the probability of the first, times the probability of the second given the first, times that of the third given the first two, and so on. For a trajectory of an MDP, the Markov property ([[mdp]]) says each factor depends only on the step just before, which leaves the short product

$$\begin{aligned} \Pr\{S_0, A_0, S_1, A_1, S_2\} = {} & \mu(S_0)\,\pol{\pi(A_0 \mid S_0)}\,p(S_1 \mid S_0, A_0) \\ & \times \pol{\pi(A_1 \mid S_1)}\,p(S_2 \mid S_1, A_1), \end{aligned}$$

with $\mu$ the distribution of the first state ([[a:sums-products|products]]).

### Total probability {#total}

If the events $B_1, B_2, \ldots$ split the outcomes into pieces that cannot happen together and between them cover everything, then any event $A$ can be computed piece by piece:

$$\Pr\{A\} = \sum_i \Pr\{B_i\}\,\Pr\{A \mid B_i\}. \label{total}$$

This is how the transition probabilities of a fixed policy are found: the probability of reaching $s'$ from $s$ averages, over the actions, the probability of reaching $s'$ with each action, $p_\pi(s' \mid s) = \sum_a \pol{\pi(a \mid s)}\,p(s' \mid s, a)$ ([[policy]]). The same idea for averages instead of probabilities is the law of total expectation ([[a:conditional-expectation|conditional expectation]]).

### Independence {#independence}

Two events are **independent** if knowing one says nothing about the other: $\Pr\{A \mid B\} = \Pr\{A\}$, or equivalently $\Pr\{A \text{ and } B\} = \Pr\{A\}\,\Pr\{B\}$. Two rolls of a die are independent; the battery level now and the battery level after a search are not. Independence is what lets probabilities of separate events simply be multiplied: the chance that two independent fair coins both land heads is $\tfrac12 \cdot \tfrac12 = \tfrac14$, and the chance that $n$ independent random choices between two actions are all right is $2^{-n}$ ([[explore-exploit]]).

### Bayes' rule {#bayes}

Writing the product rule \ref{product} both ways round, $\Pr\{B\}\Pr\{A \mid B\} = \Pr\{A\}\Pr\{B \mid A\}$, and dividing gives **Bayes' rule**:

$$\Pr\{A \mid B\} = \frac{\Pr\{B \mid A\}\,\Pr\{A\}}{\Pr\{B\}}. \label{bayes}$$

It turns the probability of evidence given a hypothesis into the probability of the hypothesis given the evidence. Read with $A$ = “this action is the best” and $B$ = “the rewards observed so far”, it is how a learner updates its beliefs after each reward: the belief before the evidence, $\Pr\{A\}$, called the *prior*, becomes the belief after it, $\Pr\{A \mid B\}$, the *posterior*. Thompson sampling acts on such posteriors ([[explore-exploit]]).
