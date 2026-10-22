+++
title = "Partial orders"
summary = "A way of comparing things in which some pairs cannot be ranked: one value function is better than another only if it is at least as large in every state."
group = "Notation"
+++

Numbers can always be compared: of two different numbers, one is the larger. Many other things cannot be ranked so simply. An **order** is a rule that says, for some pairs $x$ and $y$, that $x \le y$. It must behave the way “at most” behaves for numbers:

- every $x$ satisfies $x \le x$;
- if $x \le y$ and $y \le x$, then $x = y$;
- if $x \le y$ and $y \le z$, then $x \le z$.

If every pair can be compared, one way or the other, the order is **total**, like the order of the real numbers. If some pairs cannot, it is **partial**.

### Comparing lists of numbers {#vectors}

The standard example compares lists of numbers entry by entry: $u \le v$ when $u(i) \le v(i)$ for every position $i$. Then

$$(1, 2) \le (3, 4), \qquad \text{but neither } (1, 4) \le (3, 2) \text{ nor } (3, 2) \le (1, 4).$$

The second pair cannot be ranked: each list wins in one position. This is a partial order.

### Comparing policies {#policies}

Policies are compared in exactly this way, through their value functions ([[optimality]]): $\pol{\pi} \ge \pol{\pi'}$ when $\val{v_\pi(s)} \ge \val{v_{\pi'}(s)}$ in every state $s$. A value function is a list of numbers, one per state, so two policies that each do better in some states are simply not comparable.

What makes MDPs pleasant is a fact that a partial order does not guarantee in general: among all policies there is one that is at least as good as every other, in every state at once, a **greatest element**. In a partial order chosen at random there may be several maximal elements, none above the others; in a finite MDP the policy improvement step, which can raise the values of every state together, rules this out. That greatest element is an optimal policy, and its value function $\val{v_*}$ is the largest value function, entry by entry.
