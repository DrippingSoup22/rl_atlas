+++
title = "Linear programming"
summary = "Optimizing a weighted sum of unknowns subject to linear inequalities. The Bellman optimality equation, nonlinear because of its max, can be rewritten as a linear program and solved by standard software."
group = "Calculus and optimization"
sources = [
  { text = "Puterman (1994), Markov Decision Processes: Discrete Stochastic Dynamic Programming, Wiley, §6.9" },
  { text = "de Farias & Van Roy (2003), The linear programming approach to approximate dynamic programming, Operations Research 51", url = "https://doi.org/10.1287/opre.51.6.850.24925" },
]
+++

A **linear program** is an optimization problem in which both the quantity to optimize and the constraints are linear in the unknowns ([[a:linear-systems|linear equations]]):

$$\begin{aligned} \text{minimize } \ & c_1 x_1 + c_2 x_2 + \cdots + c_n x_n \\ \text{subject to } \ & a_{i1} x_1 + a_{i2} x_2 + \cdots + a_{in} x_n \ge b_i \quad \text{for } i = 1, \ldots, m. \end{aligned}$$

The constraints carve out a region of allowed points, a many-sided shape with flat faces, and the best point is always found at one of its corners. Algorithms such as the simplex method and interior-point methods solve linear programs with millions of unknowns, and the field has been a workhorse of operations research since the 1940s.

### A small example {#example}

Minimize $x + y$ subject to $x \ge 1$, $y \ge 2$ and $x + 2y \ge 6$. The region is everything up and to the right of the three lines. Its corners are $(1, 2.5)$ and $(2, 2)$, where $x + y$ is 3.5 and 4, so the minimum is 3.5, at $x = 1$, $y = 2.5$.

### The Bellman optimality equation as a linear program {#bellman}

The optimality equation ([[optimality]]) is not linear, because of its maximum over actions. But a maximum can be expressed with inequalities: a number $v(s)$ is at least the largest of several quantities exactly when it is at least each of them. So consider the value functions $v$ that satisfy, in every state $s$ and for every action $a$,

$$v(s) \ge \sum_{s',\,r} p(s', \rew{r} \mid s, a)\,\big[\,\rew{r} + \gam\,v(s')\,\big]. \label{constraints}$$

These are linear inequalities in the unknowns $v(s)$. The optimal values $\val{v_*}$ satisfy them, with equality for the best action. One can show that every other $v$ satisfying them lies above $\val{v_*}$ in every state. So $\val{v_*}$ is the solution of the linear program

$$\begin{aligned} \text{minimize } \ & \textstyle\sum_s v(s) \\ \text{subject to } \ & \text{the inequalities above, for every } s \text{ and } a. \end{aligned}$$

This gives a third way to solve a known MDP exactly, besides value iteration and policy iteration. It has one unknown per state and one constraint per state–action pair, so it shares their limits: it needs the model, and it is only practical when the states can be listed. Approximate versions, with far fewer unknowns than states, have been studied as an alternative route to large problems (de Farias and Van Roy, 2003).
