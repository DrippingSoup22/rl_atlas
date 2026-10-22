+++
title = "Absolute values, distances and norms"
summary = "How big a number or a list of numbers is, and how far apart two of them are. The max norm measures two value functions by their largest disagreement over all states."
group = "Linear algebra"
sources = [
  { text = "Strang (2016), Introduction to Linear Algebra, 5th ed., Wellesley-Cambridge Press, §1.2 and §11.2" },
]
+++

To say that an estimate is close to the truth, or that an iteration gets closer at every step, distances are needed: between numbers, and between whole value functions.

### Absolute value {#absolute}

The **absolute value** $\lvert x \rvert$ of a number is its size without its sign: $\lvert 3 \rvert = 3$ and $\lvert -3 \rvert = 3$. The distance between two numbers is the absolute value of their difference, $\lvert x - y \rvert$: the length of the stretch of number line between them.

The key property is the **triangle inequality**:

$$\lvert x + y \rvert \le \lvert x \rvert + \lvert y \rvert. \label{triangle}$$

When $x$ and $y$ have the same sign the two sides are equal; when they have opposite signs, part of one cancels part of the other and the left side is smaller. The inequality extends to any number of terms, $\lvert \sum_k x_k \rvert \le \sum_k \lvert x_k \rvert$, which is how a sum of discounted rewards, each at most $R_{\max}$ in size, is shown to be at most $R_{\max}/(1-\gam)$ in size ([[return]]).

### Norms {#norms}

A **norm** does for vectors ([[a:vectors-matrices|vectors]]) what the absolute value does for numbers: it gives each vector $v$ a size $\lVert v \rVert \ge 0$, zero only for the zero vector, such that scaling scales the size, $\lVert c\,v \rVert = \lvert c \rvert\,\lVert v \rVert$, and the triangle inequality holds, $\lVert u + v \rVert \le \lVert u \rVert + \lVert v \rVert$. The distance between two vectors is then the norm of their difference, $\lVert u - v \rVert$.

There are several norms, each measuring size in its own way. For a vector with entries $v(1), \ldots, v(n)$:

| Norm | Definition | Measures |
| --- | --- | --- |
| max norm $\lVert v \rVert_\infty$ | $\max_i \lvert v(i) \rvert$ | the largest entry in size |
| Euclidean norm $\lVert v \rVert_2$ | $\sqrt{\sum_i v(i)^2}$ | the ordinary length of an arrow |
| sum norm $\lVert v \rVert_1$ | $\sum_i \lvert v(i) \rvert$ | the total size of all entries |

### The max norm {#max}

The theory of dynamic programming uses the **max norm**, also called the infinity norm, written with a small $\infty$:

$$\lVert v \rVert_\infty = \max_{s} \lvert v(s) \rvert. \label{max-norm}$$

Applied to the difference of two value functions $u$ and $v$, it gives their **largest disagreement over all states**:

$$\lVert u - v \rVert_\infty = \max_s \lvert u(s) - v(s) \rvert.$$

::: example {#ex-max}
Two estimates of a three-state value function, $u = (1.0, 2.0, 5.0)$ and $v = (1.5, 2.0, 3.0)$. Their differences are $(-0.5, 0, 2.0)$, so $\lVert u - v \rVert_\infty = 2.0$: the estimates agree to within 2 everywhere, and no better than that somewhere.
:::

Saying $\lVert V - \val{v_\pi} \rVert_\infty \le 0.1$ is therefore a strong guarantee: the estimate is within 0.1 of the truth in **every** state, not just on average. This is the natural norm for averaging operations, because an average of numbers never exceeds the largest of them in size ([[a:max-argmax|max and argmax]]): applying a matrix of probabilities never increases the max norm, $\lVert P v \rVert_\infty \le \lVert v \rVert_\infty$. That one fact is why the Bellman operator is a contraction in this norm ([[a:contraction|contractions]]).

Other norms appear later in the atlas: a weighted Euclidean norm measures the error of a value function that cannot be exact everywhere, weighting each state by how often it is visited.
