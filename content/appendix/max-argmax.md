+++
title = "Max and argmax"
summary = "max gives the largest value a function takes; argmax gives where it takes it. An average can never exceed the maximum, a fact behind policy improvement and the convergence of value iteration."
group = "Notation"
+++

Choosing the best option is written with two related operators. Given a function $f$ on a set $X$:

- $\max_{x \in X} f(x)$ is the **largest value** that $f$ takes on $X$;
- $\operatorname*{arg\,max}_{x \in X} f(x)$ is **the element** (or the elements) at which it takes it: the *argument* of the maximum.

For the action values $\val{Q(s, \text{left})} = 2$, $\val{Q(s, \text{right})} = 5$ and $\val{Q(s, \text{up})} = 3$, the maximum is $\max_a \val{Q(s,a)} = 5$ and the argmax is $\operatorname*{arg\,max}_a \val{Q(s,a)} = \text{right}$. The max answers “how good is the best choice?”; the argmax answers “which choice is it?”. The greedy policy takes the argmax of the action values, and the optimal value of a state is their max ([[optimality]]).

The minimum, $\min$, and $\operatorname*{arg\,min}$ are defined the same way with the smallest value.

### Ties {#ties}

Several elements can share the largest value. Strictly, the argmax is then the *set* of all of them; in the gridworld, the optimal policy shows two arrows where two moves tie. An algorithm that must pick one usually breaks ties at random. Always taking the first in some fixed order also gives a correct answer, but it can bias an agent that is still learning toward the actions listed first.

### The average is at most the maximum {#average}

One inequality does a great deal of work in the atlas. If $p$ is any probability distribution over $X$ ([[a:probability|probability]]), so that the weights $p(x)$ are nonnegative and add up to 1, then

$$\sum_{x} p(x)\,f(x) \le \max_{x} f(x). \label{avg-max}$$

The reason: every term $f(x)$ is at most the maximum, so replacing each $f(x)$ by the maximum can only increase the sum, and $\sum_x p(x) \max f = \max f$ because the weights add up to 1. Likewise, an average is at least the minimum. A weighted average always lies between the smallest and the largest of the things averaged.

Two consequences:

- Acting greedily never makes a policy worse. In each state, the greedy action is worth $\max_a \val{q_\pi(s,a)}$, at least the policy's own average $\sum_a \pol{\pi(a \mid s)}\,\val{q_\pi(s,a)} = \val{v_\pi(s)}$ ([[prediction-control]]).
- An average of differences is at most the largest difference. This is why the Bellman operator, which averages, shrinks the gap between two value functions ([[a:contraction|contractions]]).

### Two more facts {#facts}

The maximum of a sum is at most the sum of the maxima, $\max_x \big(f(x) + g(x)\big) \le \max_x f(x) + \max_x g(x)$, because the two maxima may be reached at different places. And two maxima of functions that are close are close: the gap between $\max_x f(x)$ and $\max_x g(x)$ is at most the largest gap $\max_x \lvert f(x) - g(x) \rvert$. The second fact is what makes value iteration, which takes a max at every step, a contraction just like policy evaluation.
