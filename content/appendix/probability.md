+++
title = "Probability and distributions"
summary = "A probability is a number between 0 and 1 that says how likely something is; a distribution gives one to every possible outcome, and they add up to 1."
group = "Probability"
sources = [
  { text = "Blitzstein & Hwang (2019), Introduction to Probability, 2nd ed., CRC Press, chapters 1 and 3", url = "https://projects.iq.harvard.edu/stat110" },
]
+++

Reinforcement learning deals with chance everywhere: a policy may choose at random, the world may answer at random, and the return is the result of many such draws. Probability is the arithmetic of chance.

### Outcomes and events {#outcomes}

A random experiment, such as rolling a die or taking one step in a stochastic world, has a set of possible **outcomes**: $\{1, 2, 3, 4, 5, 6\}$ for the die. An **event** is a set of outcomes, described in words or by a condition: “the roll is even” is the event $\{2, 4, 6\}$.

A **probability** assigns to every event a number between 0 and 1: 0 for impossible, 1 for certain. It must obey two rules:

- the probability of the set of all outcomes is 1;
- the probability of an event made of separate pieces that cannot happen together is the sum of their probabilities.

From these rules, $\Pr\{\text{even}\} = \Pr\{2\} + \Pr\{4\} + \Pr\{6\} = 3 \times \tfrac16 = \tfrac12$ for a fair die, and the probability that something does *not* happen is one minus the probability that it does.

The notation $\Pr\{\ldots\}$ is read “the probability that …”. Inside the braces is a condition on random quantities: $\Pr\{S_{t+1} = \text{low}\}$ is the probability that the next state is “low”.

### Distributions {#distributions}

For a finite set of outcomes, it is enough to give the probability $p(x)$ of each single outcome $x$. Such a list is a **probability distribution**: numbers with

$$p(x) \ge 0 \text{ for every } x, \qquad \sum_x p(x) = 1. \label{dist}$$

The **uniform** distribution gives every outcome the same probability: $1/n$ for $n$ outcomes. The random policy of the gridworld is uniform over the four moves, $\tfrac14$ each. A distribution that puts probability 1 on a single outcome is **deterministic**: there is no chance left.

Every row of the dynamics table of an MDP is a distribution over the outcomes of one action ([[mdp]]); a policy gives one distribution over actions per state ([[policy]]). Checking that each one adds up to 1 is the first test of any model written by hand.

### Joint and marginal distributions {#joint}

When an experiment produces two things at once, such as the next state and the reward, their **joint** distribution gives the probability of every pair: $p(s', r)$ is the probability that the next state is $s'$ *and* the reward is $r$.

The distribution of one of them alone, its **marginal**, is found by adding up over the other: the probability that the next state is $s'$, whatever the reward, is

$$p(s') = \sum_{r} p(s', r). \label{marginal}$$

This is the “sum out what you don't care about” move by which the state-transition probabilities $p(s' \mid s, a)$ are computed from the dynamics $p(s', r \mid s, a)$ ([[mdp]]). Probabilities that depend on something known, as these depend on $s$ and $a$, are conditional probabilities ([[a:conditional-probability|conditional probability]]).

### Where the numbers come from {#meaning}

A probability can be read as a long-run frequency: if an event has probability 0.3, then in many independent repetitions it happens about 30% of the time ([[a:law-of-large-numbers|the law of large numbers]] makes this precise). It can also be read as a degree of belief, as in Thompson sampling, which keeps a distribution over what each action is worth ([[explore-exploit]]). The arithmetic is the same either way.
