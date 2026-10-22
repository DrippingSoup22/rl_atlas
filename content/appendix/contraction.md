+++
title = "Contractions and Banach's theorem"
summary = "A function that brings any two inputs closer together by a fixed factor. Banach's fixed-point theorem says it has exactly one fixed point, and that repeating the function from anywhere reaches it, with the error shrinking by that factor at every step."
group = "Analysis"
sources = [
  { text = "Puterman (1994), Markov Decision Processes: Discrete Stochastic Dynamic Programming, Wiley, §6.2" },
  { text = "Bertsekas & Tsitsiklis (1996), Neuro-Dynamic Programming, Athena Scientific, §2.2" },
]
+++

Why does repeating the Bellman operator always end at the true values, from any starting guess, and why is there only one solution? Both answers come from one property of the operator and one theorem about that property.

### Contractions {#definition}

Let $f$ map a set of objects to itself, and let $\lVert x - y \rVert$ measure the distance between two of them: the absolute difference for numbers, a norm for vectors or value functions ([[a:norms|norms]]).

::: definition {#def-contraction} Contraction
$f$ is a **contraction** with modulus $\gam$, where $0 \le \gam < 1$, if for every two inputs $x$ and $y$
$$\lVert f(x) - f(y) \rVert \le \gam\,\lVert x - y \rVert. \label{contraction}$$
:::

In words: whatever two inputs you pick, their outputs are closer together than they were, by at least the factor $\gam$. Applying the function squeezes the whole space.

::: example {#ex-line}
$f(x) = 1 + 0.9\,x$ on the real numbers is a contraction with modulus 0.9: $\lvert f(x) - f(y) \rvert = 0.9\,\lvert x - y \rvert$. The function $f(x) = 2x$ is not; it doubles distances. Nor is $f(x) = x + 1$, which keeps them the same: a factor of exactly 1 is not enough.
:::

### Banach's fixed-point theorem {#banach}

::: theorem {#thm-banach} Banach, 1922
Let $f$ be a contraction with modulus $\gam < 1$ on a complete space (for example the real numbers, or vectors of a fixed length with any norm). Then

1. $f$ has exactly one fixed point $x^*$, with $f(x^*) = x^*$;
2. from any starting point $x_0$, the sequence $x_{k+1} = f(x_k)$ converges to $x^*$;
3. the error shrinks at least geometrically: $\lVert x_k - x^* \rVert \le \gam^k\,\lVert x_0 - x^* \rVert$.
:::

“Complete” is a technical condition that rules out spaces with holes in them; every space used in the atlas has it, and it can be ignored here.

::: proof
*At most one fixed point.* If $x^*$ and $y^*$ were both fixed points, then $\lVert x^* - y^* \rVert = \lVert f(x^*) - f(y^*) \rVert \le \gam\,\lVert x^* - y^* \rVert$. A non-negative number that is at most $\gam < 1$ times itself must be 0, so $x^* = y^*$.

*The error shrinks.* Suppose a fixed point $x^*$ exists. Since $x^* = f(x^*)$, each step gives $\lVert x_{k+1} - x^* \rVert = \lVert f(x_k) - f(x^*) \rVert \le \gam\,\lVert x_k - x^* \rVert$. Applied $k$ times, this is the bound in 3, and since $\gam^k \to 0$ ([[a:limits|limits]]), the sequence converges to $x^*$.

*A fixed point exists.* Successive steps also shrink: $\lVert x_{k+1} - x_k \rVert \le \gam^k\,\lVert x_1 - x_0 \rVert$. So the total distance the sequence can still travel after step $k$ is at most

$$\gam^k \lVert x_1 - x_0 \rVert\,(1 + \gam + \gam^2 + \cdots) = \frac{\gam^k}{1 - \gam}\,\lVert x_1 - x_0 \rVert$$

by the [[a:geometric-series|geometric series]], and this tends to 0. The terms therefore bunch together ever more tightly, and in a complete space such a sequence has a limit $x^*$. A contraction moves nearby points to nearby points, so $f(x_k)$ tends to $f(x^*)$; but $f(x_k) = x_{k+1}$ tends to $x^*$. The two limits agree: $f(x^*) = x^*$.
:::

::: example {#ex-iterate}
For $f(x) = 1 + 0.9\,x$ the fixed point is 10. From $x_0 = 0$ the errors are $10, 9, 8.1, 7.29, \ldots$: exactly $10 \times 0.9^k$, as the theorem allows. From $x_0 = 100$ they are $90, 81, 72.9, \ldots$, shrinking by the same factor toward the same point.
:::

### The Bellman operator is a contraction {#bellman}

The Bellman operator of a policy computes, in every state, the expected reward plus $\gam$ times the **average** of the next state's value ([[a:operators|operators]]):

$$(\mathcal{T}_\pi v)(s) = r_\pi(s) + \gam \sum_{s'} P_\pi(s, s')\,v(s').$$

Apply it to two value functions $u$ and $v$ and subtract. The expected rewards are the same for both and cancel:

$$(\mathcal{T}_\pi u)(s) - (\mathcal{T}_\pi v)(s) = \gam \sum_{s'} P_\pi(s, s')\,\big(u(s') - v(s')\big).$$

The sum on the right is a weighted average of the differences $u(s') - v(s')$, with weights that are probabilities. An average is never larger in size than the largest of the things averaged ([[a:max-argmax|max and argmax]]), and the largest difference in size is $\lVert u - v \rVert_\infty$. So in every state $s$,

$$\big\lvert (\mathcal{T}_\pi u)(s) - (\mathcal{T}_\pi v)(s) \big\rvert \le \gam\,\lVert u - v \rVert_\infty,$$

and taking the largest over $s$ gives $\lVert \mathcal{T}_\pi u - \mathcal{T}_\pi v \rVert_\infty \le \gam\,\lVert u - v \rVert_\infty$. The operator is a contraction with modulus $\gam$ in the max norm.

Banach's theorem then delivers everything at once: the Bellman equation has exactly one solution, $\val{v_\pi}$; iterative policy evaluation reaches it from any start; and after $k$ sweeps the error in every state is at most $\gam^k$ times the initial error ([[bellman]], [[policy-evaluation]]). With $\gam = 0.9$, each sweep removes at least a tenth of the remaining error, and 44 sweeps divide it by about 100. The optimality operator, with a max in place of the average, is a contraction for the same reason, which is why value iteration converges ([[optimality]]).

The discount is doing the work. With $\gam = 1$ the argument gives only $\le 1$, which is not a contraction, and indeed the Bellman equation of a task that never ends can then fail to have a solution.
