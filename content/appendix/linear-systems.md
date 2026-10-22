+++
title = "Systems of linear equations"
summary = "Several equations in several unknowns, each a weighted sum of the unknowns. A system with as many independent equations as unknowns has exactly one solution, found by elimination at a cost that grows like the cube of its size."
group = "Linear algebra"
sources = [
  { text = "Strang (2016), Introduction to Linear Algebra, 5th ed., Wellesley-Cambridge Press, chapter 2" },
]
+++

An equation is **linear** in its unknowns if each unknown appears only multiplied by a number and added: $2x + 3y = 7$ is linear, $xy = 7$ and $\max(x, y) = 7$ are not. A **system** of linear equations asks for values of the unknowns that satisfy several such equations at once.

### A small example {#example}

The Bellman equations of a policy are a linear system: one unknown value per state, one equation per state ([[bellman]]). Take a two-state version of the recycling robot ([[mdp]]) with the policy “search when high, recharge when low”, $\gam = 0.9$, and write $h$ and $l$ for the unknown values of the two states. Searching when high earns 2 and stays high with probability 0.7; recharging when low earns 0 and always leads to high:

$$\begin{aligned} h &= 2 + 0.9\,(0.7\,h + 0.3\,l), \\ l &= 0 + 0.9\,h. \end{aligned}$$

The second equation gives $l$ in terms of $h$. **Substituting** it into the first leaves one equation in one unknown, $h = 2 + 0.63\,h + 0.243\,h$, so $0.127\,h = 2$ and $h \approx 15.75$. Then $l = 0.9\,h \approx 14.17$. These are the values of this policy, and in fact the optimal values of the robot ([[optimality]]).

### Matrix form {#matrix}

Collect the unknowns in a vector $x$ ([[a:vectors-matrices|vectors and matrices]]). A system of $n$ linear equations in $n$ unknowns is then written $Ax = b$: row $i$ of the matrix $A$ holds the coefficients of equation $i$, and $b$ holds the constants. The robot's equations, with the unknowns moved to the left, read

$$\begin{pmatrix} 1 - 0.63 & -0.27 \\ -0.9 & 1 \end{pmatrix} \begin{pmatrix} h \\ l \end{pmatrix} = \begin{pmatrix} 2 \\ 0 \end{pmatrix}.$$

For any policy, $v = r_\pi + \gam P_\pi v$ rearranges the same way into $(I - \gam P_\pi)\,v = r_\pi$.

### When is there exactly one solution? {#unique}

A linear system can have one solution, none, or infinitely many: $x + y = 1$ and $x + y = 2$ contradict each other; $x + y = 1$ and $2x + 2y = 2$ say the same thing twice. There is exactly one solution, for every right-hand side $b$, precisely when the matrix $A$ has an inverse, and then $x = A^{-1} b$.

For the Bellman equations with $\gam < 1$ this is always the case: $I - \gam P_\pi$ has an inverse. One way to see it is through contractions ([[a:contraction|contractions]]); another is the series $(I - \gam P_\pi)^{-1} = I + \gam P_\pi + \gam^2 P_\pi^2 + \cdots$, the matrix version of the geometric series ([[a:geometric-series|geometric series]]), which converges because every power of $P_\pi$ is again a matrix of probabilities. Read term by term, it says the value is the expected reward now, plus $\gam$ times the expected reward one step later, plus $\gam^2$ times the one after: the expected discounted return.

### The cost of solving {#cost}

The standard method, **Gaussian elimination**, is the substitution of the example done systematically: use the first equation to remove the first unknown from all the others, then the second to remove the second unknown, and so on, and solve backwards. For $n$ unknowns it takes on the order of $n^3$ arithmetic operations ([[a:growth-rates|growth rates]]). That is nothing for the 25 states of a gridworld (about 15 000 operations) and impossible for a game with $10^{20}$ states. This cost, and the need to know $P_\pi$ at all, is why reinforcement learning prefers iterative methods that approach the solution step by step.
