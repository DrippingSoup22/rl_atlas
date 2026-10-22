+++
title = "Vectors and matrices"
summary = "A vector is a list of numbers, such as one value per state; a matrix is a table of numbers, such as the transition probabilities. Multiplying a matrix by a vector computes many weighted sums at once."
group = "Linear algebra"
sources = [
  { text = "Strang (2016), Introduction to Linear Algebra, 5th ed., Wellesley-Cambridge Press, chapters 1 and 2" },
]
+++

For a finite MDP, a value function is just a list of numbers, one per state, and the dynamics of a policy are a table of numbers, one per pair of states. Linear algebra is the language for working with whole lists and tables at once.

### Vectors {#vectors}

A **vector** is an ordered list of numbers. A value function on $n$ states can be written as the column

$$v = \begin{pmatrix} v(1) \\ v(2) \\ \vdots \\ v(n) \end{pmatrix} \in \mathbb{R}^n.$$

Vectors of the same length are added entry by entry, and multiplied by a number entry by entry: $(u + v)(i) = u(i) + v(i)$ and $(c\,v)(i) = c\,v(i)$. Comparing them entry by entry gives a partial order ([[a:partial-order|partial orders]]); measuring their size gives a norm ([[a:norms|norms]]).

### Matrices {#matrices}

A **matrix** is a rectangular table of numbers; $A(i, j)$ is the entry in row $i$ and column $j$. A square matrix with $n$ rows and $n$ columns has one entry for each pair of states, which makes it the natural home of transition probabilities: $P_\pi(s, s')$ is the probability of moving from $s$ to $s'$ in one step under the policy $\pol{\pi}$ ([[a:markov-chain|Markov chains]]).

The **identity matrix** $I$ has 1 on its diagonal, where row and column agree, and 0 everywhere else.

### Multiplying a matrix by a vector {#product}

The product $A v$ of an $n \times n$ matrix and a vector of length $n$ is a new vector of length $n$. Its entry $i$ is the sum of the products of row $i$ of $A$ with the entries of $v$:

$$(A v)(i) = \sum_{j=1}^{n} A(i, j)\,v(j). \label{matvec}$$

::: example {#ex-product}
$$\begin{pmatrix} 0.7 & 0.3 \\ 1 & 0 \end{pmatrix} \begin{pmatrix} 10 \\ 4 \end{pmatrix} = \begin{pmatrix} 0.7 \cdot 10 + 0.3 \cdot 4 \\ 1 \cdot 10 + 0 \cdot 4 \end{pmatrix} = \begin{pmatrix} 8.2 \\ 10 \end{pmatrix}.$$
:::

When $A$ is a transition matrix, each row is a probability distribution, and entry $i$ of $Av$ is the **expected value of $v$ at the next state**, starting from $i$. The whole Bellman equation of a policy, one equation per state, becomes one line in this notation, $v = r_\pi + \gam P_\pi v$ ([[bellman]]): the vector of values equals the vector of expected rewards plus $\gam$ times the vector of expected next values.

The identity leaves every vector unchanged, $Iv = v$, and the product respects sums and constants: $A(u + v) = Au + Av$ and $A(cv) = c\,Av$. A map with these two properties is called **linear**, which is where linear algebra gets its name.

### Multiplying matrices {#matmul}

Two matrices are multiplied by applying the same rule column by column: $(AB)(i, j) = \sum_k A(i, k)\,B(k, j)$. For transition matrices, $P^2 = PP$ gives the probabilities of moving between states in two steps, and $P^k$ in $k$ steps. The order matters: in general $AB \ne BA$.

### The inverse {#inverse}

Some square matrices $A$ have an **inverse** $A^{-1}$, the matrix that undoes them: $A^{-1} A = A A^{-1} = I$. When it exists, the system of equations $Ax = b$ has exactly one solution, $x = A^{-1} b$ ([[a:linear-systems|linear systems]]). The Bellman equation can be solved this way, because $I - \gam P_\pi$ always has an inverse when $\gam < 1$.
