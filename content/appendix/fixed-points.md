+++
title = "Fixed points"
summary = "An input that a function returns unchanged, f(x) = x. Many equations, the Bellman equations among them, say exactly that something is a fixed point, and repeating the function is a natural way to look for one."
group = "Analysis"
+++

A **fixed point** of a function $f$ is an input that $f$ leaves unchanged:

$$f(x^*) = x^*. \label{fixed}$$

The function $f(x) = x^2$ has two fixed points, 0 and 1. The function $f(x) = x + 1$ has none. The function $f(x) = x$ has every number as a fixed point.

### Equations as fixed points {#equations}

Many equations can be written in the form “$x$ equals something computed from $x$”, and solving them means finding a fixed point. The simplest example in the atlas is a state that always gives a reward of 1 and stays where it is, with $\gam = 0.9$. Its value must satisfy

$$v = 1 + 0.9\,v,$$

so $v$ is a fixed point of $f(v) = 1 + 0.9\,v$. Here it is easy to solve directly: $0.1\,v = 1$, so $v = 10$, which is the discounted sum $1 + 0.9 + 0.9^2 + \cdots$ ([[a:geometric-series|geometric series]]).

The Bellman equation says the same thing for a whole value function at once: $\val{v_\pi}$ is a fixed point of the Bellman operator, $\val{v_\pi} = \mathcal{T}_\pi \val{v_\pi}$ ([[a:operators|operators]]). Applying one step of look-ahead to the true values gives back the true values. Any other value function is changed by the operator, and that change measures how inconsistent it is.

### Finding a fixed point by repetition {#iteration}

A natural way to search for a fixed point is to start anywhere and apply the function again and again: $x_{k+1} = f(x_k)$. If the sequence settles on some $x^*$, and $f$ is continuous, then $x^*$ is a fixed point, since applying $f$ to the limit must give the limit again.

::: example {#ex-iterate}
Iterating $f(v) = 1 + 0.9\,v$ from $v_0 = 0$ gives $1, 1.9, 2.71, 3.44, \ldots$ The gap to 10 is $10, 9, 8.1, 7.29, \ldots$: it shrinks by the factor 0.9 at every step, so the iteration converges to 10.
:::

Repetition does not always work. Iterating $f(x) = 2x$ from any $x_0 \ne 0$ runs away from its fixed point 0, and iterating $f(x) = 1 - x$ from 0 jumps $0, 1, 0, 1, \ldots$ forever without reaching its fixed point $\tfrac12$. What makes the example work is that $f$ shrinks distances: $\lvert f(u) - f(v) \rvert = 0.9\,\lvert u - v \rvert$. Functions with this property are contractions, and for them repetition always finds the fixed point, and it is the only one ([[a:contraction|contractions]]).

### In reinforcement learning {#atlas}

Dynamic programming is fixed-point iteration: policy evaluation repeats the Bellman operator of a policy ([[policy-evaluation]]), value iteration repeats the optimality operator ([[optimality]]). Temporal-difference learning is a noisy version of the same repetition, in which each step applies the operator to one sampled transition instead of computing it exactly ([[bellman]]).
