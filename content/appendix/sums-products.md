+++
title = "Sums and products"
summary = "How to read Σ and Π: a compact way to write that many terms are added or multiplied, with the rules that let a long sum be split, reordered and simplified."
group = "Notation"
+++

The capital Greek letter sigma, $\Sigma$, abbreviates a sum of many terms that follow one pattern. Its cousin pi, $\Pi$, does the same for a product. Nearly every formula of the atlas uses one of them.

### Reading a sum {#reading}

$$\sum_{k=0}^{3} \gam^k\,\rew{R_{t+k+1}} = \rew{R_{t+1}} + \gam\,\rew{R_{t+2}} + \gam^2\,\rew{R_{t+3}} + \gam^3\,\rew{R_{t+4}}.$$

The letter under the sigma, here $k$, is the **index**. It starts at the value written below ($k = 0$) and goes up by one until the value written above ($3$). For each value, the expression to the right is written out with that value of $k$, and the results are added. The index is a placeholder: $\sum_{k=0}^{3} x_k$ and $\sum_{j=0}^{3} x_j$ are the same sum.

An $\infty$ on top means the terms go on forever; such a sum is defined as the [[a:limits|limit]] of its first $n$ terms as $n$ grows, and it may or may not be finite ([[a:geometric-series|geometric series]]).

### Sums over a set {#sets}

Instead of a range of numbers, the index can run over the elements of a set ([[a:sets-functions|sets]]):

$$\sum_{a \in \mathcal{A}(s)} \pol{\pi(a \mid s)}\,\val{q_\pi(s,a)}$$

adds one term for every action available in $s$. When the set is clear, it is often left out, $\sum_a$. Two indices under one sigma, as in $\sum_{s', r}$, mean a term for every combination: every next state paired with every reward. That is the same as two sums, one inside the other:

$$\sum_{s',\,r} p(s', \rew{r} \mid s, a)\,f(s', \rew{r}) = \sum_{s'} \sum_{r} p(s', \rew{r} \mid s, a)\,f(s', \rew{r}).$$

### The rules {#rules}

Sums obey the rules of ordinary addition, written in a new way.

- **A constant comes out.** A factor that does not depend on the index can be taken outside: $\sum_k c\,x_k = c \sum_k x_k$. In $\sum_{s'} p(s' \mid s, a)\,\gam\,v(s')$, the $\gam$ comes out.
- **A sum splits.** $\sum_k (x_k + y_k) = \sum_k x_k + \sum_k y_k$. This is how the Bellman equation separates the expected reward from the expected next value.
- **The order of two sums can be swapped** (for finite sums, and for infinite ones whose terms are all positive or that converge absolutely): $\sum_i \sum_j x_{ij} = \sum_j \sum_i x_{ij}$. Both add every $x_{ij}$ once.
- **The first term can be taken out**, which shifts the index: $\sum_{k=0}^{n} x_k = x_0 + \sum_{k=1}^{n} x_k = x_0 + \sum_{j=0}^{n-1} x_{j+1}$. Taking out the first reward of the return is exactly this move ([[return]]).

Two sums in which a constant is summed are worth remembering: $\sum_{k=1}^{n} 1 = n$, and a probability distribution always satisfies $\sum_x p(x) = 1$ ([[a:probability|probability]]), so $\sum_x p(x)\,c = c$.

### Products {#products}

$\prod$ works the same way with multiplication:

$$\prod_{t=0}^{2} \pol{\pi(A_t \mid S_t)} = \pol{\pi(A_0 \mid S_0)} \cdot \pol{\pi(A_1 \mid S_1)} \cdot \pol{\pi(A_2 \mid S_2)}.$$

Products appear when probabilities of successive independent choices are combined: the probability of a whole trajectory is the product of the probabilities of its steps ([[a:conditional-probability|conditional probability]]). A constant factor inside a product comes out with a power: $\prod_{k=1}^{n} c\,x_k = c^n \prod_{k=1}^{n} x_k$. A logarithm turns a product into a sum, $\ln \prod_k x_k = \sum_k \ln x_k$ ([[a:exponential-logarithm|logarithms]]), which is why long products of probabilities are usually handled as sums of their logarithms.
