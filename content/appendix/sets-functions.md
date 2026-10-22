+++
title = "Sets and functions"
summary = "The vocabulary under every definition: a set is a collection of things, ∈ says that something belongs to one, and a function assigns to every element of one set an element of another."
group = "Notation"
+++

Most definitions in the atlas are written in the language of sets and functions. It takes only a handful of symbols.

### Sets {#sets}

A **set** is a collection of distinct things, its **elements**. A small set can be listed between braces: the states of the recycling robot form the set $\{\text{high}, \text{low}\}$. The order of the list does not matter, and nothing appears twice.

| Symbol | Read | Example |
| --- | --- | --- |
| $x \in A$ | $x$ is an element of $A$ | $\text{low} \in \{\text{high}, \text{low}\}$ |
| $x \notin A$ | $x$ is not an element of $A$ | $\text{empty} \notin \{\text{high}, \text{low}\}$ |
| $A \subset B$ | every element of $A$ is also in $B$ | $\mathcal{S} \subset \mathcal{S}^+$ |
| $\lvert A \rvert$ | the number of elements of $A$ | $\lvert \{\text{high}, \text{low}\} \rvert = 2$ |
| $\{x \in A : \ldots\}$ | the elements of $A$ that satisfy a condition | $\{n \in \mathbb{N} : n \text{ even}\}$ |
| $A \times B$ | all pairs $(a, b)$ with $a \in A$ and $b \in B$ | $\mathcal{S} \times \mathcal{A}$, all state–action pairs |

Calligraphic letters are the custom for the sets of an MDP: $\mathcal{S}$ for the states, $\mathcal{A}$ for the actions, $\mathcal{R}$ for the possible rewards. When the actions available depend on the state, $\mathcal{A}(s)$ is the set of actions available in $s$.

### Numbers {#numbers}

A few sets of numbers have their own letters.

- $\mathbb{N} = \{0, 1, 2, \ldots\}$, the natural numbers: time steps, counts.
- $\mathbb{R}$, the real numbers: every point of the number line, such as $-3$, $0.9$ or $\pi$. A reward is a real number, $\rew{R_{t+1}} \in \mathbb{R}$.
- $\mathbb{R}^d$, the lists of $d$ real numbers $(x_1, \ldots, x_d)$, also called vectors ([[a:vectors-matrices|vectors]]): a robot's state might be a list of joint angles and velocities.
- An **interval** is a stretch of the line. Square brackets include the end, round ones exclude it: $[0, 1]$ is every number from 0 to 1, ends included, and $[0, 1)$ leaves out the 1. The discount rate lives in $[0, 1]$.

A set is **finite** if its elements can be counted to an end, like the 25 cells of a gridworld, and **infinite** otherwise, like $\mathbb{R}$. Finite sets allow sums over all their elements; that is why so much of the theory first assumes a finite MDP.

### Functions {#functions}

A **function** $f : X \to Y$ assigns to every element $x$ of the set $X$ exactly one element $f(x)$ of the set $Y$. $X$ is its domain, $Y$ where its values lie. A function can be given by a formula, $f(x) = x^2$, or by a table that lists $f(x)$ for every $x$. For a finite domain the two are the same thing: a table *is* a function.

The atlas is full of functions given by tables.

- A state-value function $\val{v} : \mathcal{S} \to \mathbb{R}$ attaches a number to every state: one entry per cell of the gridworld.
- An action-value function $\val{q} : \mathcal{S} \times \mathcal{A} \to \mathbb{R}$ attaches a number to every state–action pair: four triangles per cell.
- A deterministic policy $\pol{\pi} : \mathcal{S} \to \mathcal{A}$ attaches an action to every state: one arrow per cell.

A function with several inputs, such as $p(s', r \mid s, a)$, is a function on a set of tuples; the bar inside the parentheses is only a way of grouping the inputs, read “given” ([[a:conditional-probability|conditional probability]]).

### For all, there exists {#quantifiers}

Two short phrases recur. “For all $s \in \mathcal{S}$”, sometimes written $\forall s \in \mathcal{S}$, means the statement holds for every state, one by one: the Bellman equation is one equation for each state. “There exists”, written $\exists$, means at least one element has the property: there exists an optimal policy.
