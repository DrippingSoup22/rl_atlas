+++
summary = "The value of a state equals what you expect to get on the next step plus the discounted value of where you land: one equation per state that ties all the values together."
prereqs = ["value-functions", "return", "mdp"]
sources = [
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., §3.5 and Figure 3.4", url = "http://incompleteideas.net/book/the-book-2nd.html" },
  { text = "Bellman (1957), Dynamic Programming, Princeton University Press" },
  { text = "Puterman (1994), Markov Decision Processes: Discrete Stochastic Dynamic Programming, Wiley, §6.1" },
]

[story]
scene = "grid"
env = "gridworld"
gamma = 0.9
digits = 2
formula = '\step{1}{\val{v_\pi(s)}} \step{2}{= \sum_a \pol{\pi(a \mid s)}} \step{3}{\sum_{s^\prime\!,\,r} p(s^\prime\!, \rew{r} \mid s, a)} \step{4}{\big[\,\rew{r} + \gam\,\val{v_\pi(s^\prime)}\,\big]}'

[story.numbers]
center = '\val{0.67} \approx \tfrac14(0 + 0.9 \cdot \val{2.25}) + \tfrac14(0 + 0.9 \cdot \val{0.36}) + \tfrac14(0 + 0.9 \cdot \val{(-0.35)}) + \tfrac14(0 + 0.9 \cdot \val{0.74})'
jump = '\val{v_\pi(A)} = \rew{10} + 0.9 \cdot \val{v_\pi(A^\prime)} = \rew{10} + 0.9 \cdot \val{(-1.35)} \approx \val{8.79}'
corner = '\val{-1.86} \approx \tfrac14(0 + 0.9 \cdot \val{(-0.97)}) + \tfrac14(0 + 0.9 \cdot \val{(-1.35)}) + \tfrac12(\rew{-1} + 0.9 \cdot \val{(-1.86)})'
+++

## Story

::: step {v = "random", values = true, agent = "none"}
Here are the values of the random policy in the gridworld once more. They are not 25 unrelated numbers: each one is tied to the values of its neighbors by one simple equation.
:::

::: step {v = "random", values = true, focus = [2, 2], agent = "none", formula = 1}
Take the center cell $s$. Its value, 0.67, is the expected return from here when moving at random.
:::

::: step {v = "random", values = true, focus = [2, 2], next = true, policy = "random", agent = "none", formula = 3}
From $s$ the policy picks each of the four moves with probability ¼. Each move earns a reward, 0 here, and lands on a neighbor $s'$, outlined, whose value is the expected return from there.
:::

::: step {v = "random", values = true, focus = [2, 2], next = true, policy = "random", agent = "none", formula = 4}
So the expected return from $s$ is the average, over the moves, of the reward plus $\gam$ times the value of where the move lands. That is the **Bellman equation** for $\val{v_\pi}$.
:::

::: step {v = "random", values = true, focus = [2, 2], next = true, policy = "random", agent = "none", formula = 4, numbers = "center"}
With numbers: the four neighbors are worth 2.25, 0.36, $-0.35$ and 0.74, and the equation gives back 0.67. The center cell agrees with its neighbors.
:::

::: step {v = "random", values = true, focus = [0, 1], next = true, agent = "none", formula = 4, numbers = "jump"}
It holds for the special cells too. Every move from $A$ jumps to $A'$ with $\rew{+10}$, so $\val{v_\pi(A)} = 10 + 0.9\,\val{v_\pi(A')}$: 8.79.
:::

::: step {v = "random", values = true, focus = [4, 0], next = true, agent = "none", formula = 4, numbers = "corner"}
And in a corner. Two of the four moves hit a wall: they cost $\rew{-1}$ and leave the agent where it is. The corner's own value appears on both sides of its equation.
:::

::: step {v = "random", values = true, agent = "none", formula = 4}
One such equation holds in every cell: 25 linear equations in 25 unknowns, whose only solution is this table. Dynamic programming solves them with the model ([[policy-evaluation]]); temporal-difference learning samples them one move at a time ([[td0]]).
:::

## Textbook

### From the return to an equation {#derivation}

The return obeys the recursion $\rew{G_t} = \rew{R_{t+1}} + \gam\,\rew{G_{t+1}}$ ([[return]]). Taking expectations turns it into a relation between values. For any policy $\pol{\pi}$ and state $s$,

$$\begin{aligned} \val{v_\pi(s)} &= \mathbb{E}_\pi[\,\rew{G_t} \mid S_t = s\,] = \mathbb{E}_\pi[\,\rew{R_{t+1}} + \gam\,\rew{G_{t+1}} \mid S_t = s\,] \\ &= \sum_a \pol{\pi(a \mid s)} \sum_{s',\,r} p(s', \rew{r} \mid s, a)\,\Big[\,\rew{r} + \gam\,\mathbb{E}_\pi[\,\rew{G_{t+1}} \mid S_{t+1} = s'\,]\Big] \\ &= \sum_a \pol{\pi(a \mid s)} \sum_{s',\,r} p(s', \rew{r} \mid s, a)\,\big[\,\rew{r} + \gam\,\val{v_\pi(s')}\,\big]. \end{aligned} \label{bellman-v}$$

The second line conditions on the first action, next state and reward. The step that makes it work is the Markov property ([[mdp]]): once $S_{t+1} = s'$ is known, the expected return from there does not depend on how the agent got to $s'$, so it is simply $\val{v_\pi(s')}$. Equation \ref{bellman-v} is the **Bellman equation** for $\val{v_\pi}$. Many texts call it the Bellman *expectation* equation, to set it apart from the optimality equation of [[optimality]], which has a max where this one has an average. It holds for every state, and it expresses each value through the values of the states that can follow it.

Read from right to left, \ref{bellman-v} is an average over everything that can happen in one step: the policy chooses $a$, the environment chooses $s'$ and $r$, and each possibility contributes its reward plus the discounted value of where it leads.

### For action values {#q}

The same argument, starting from a fixed first action, gives the Bellman equation for $\val{q_\pi}$:

$$\val{q_\pi(s,a)} = \sum_{s',\,r} p(s', \rew{r} \mid s, a)\,\Big[\,\rew{r} + \gam \sum_{a'} \pol{\pi(a' \mid s')}\,\val{q_\pi(s',a')}\Big]. \label{bellman-q}$$

The two equations are the two halves of one step, glued in different orders: $\val{v_\pi}$ averages over actions and then over outcomes, $\val{q_\pi}$ over outcomes and then over the next action ([[value-functions]]).

### Backup diagrams {#diagrams}

The equations are pictured by **backup diagrams** (\ref{fig-backup}). An open circle is a state, a solid dot a state–action pair. Read from the top: from $s$, the policy may take any of several actions; from each action, the environment may lead to several states, with a reward on the way. The value at the top is computed from the values at the bottom: it is *backed up* from the successors. Every update in the atlas has a backup diagram, and comparing diagrams is the quickest way to compare methods.

::: figure {#fig-backup}
{{backup v-pi}}
The backup diagram of $\val{v_\pi}$: from the state $s$, through each action $a$ chosen by $\pol{\pi}$, to each next state $s'$ with its reward $\rew{r}$. After Sutton & Barto, Figure 3.4.
:::

::: figure {#fig-backup-q}
{{backup q-pi}}
The backup diagram of $\val{q_\pi}$: from the pair $(s, a)$, through each next state $s'$, to each next action $a'$.
:::

### A system of linear equations {#linear}

For a finite MDP, \ref{bellman-v} is a system of $|\mathcal{S}|$ linear equations in $|\mathcal{S}|$ unknowns. Write $r_\pi(s) = \sum_a \pol{\pi(a \mid s)}\, r(s, a)$ for the expected reward in $s$ and $P_\pi(s, s') = \sum_a \pol{\pi(a \mid s)}\, p(s' \mid s, a)$ for the transition matrix of the policy. Then \ref{bellman-v} reads

$$\val{v_\pi} = r_\pi + \gam\,P_\pi \val{v_\pi} \qquad\Longleftrightarrow\qquad \val{v_\pi} = (I - \gam P_\pi)^{-1}\, r_\pi. \label{matrix}$$

::: theorem {#thm-unique} Existence and uniqueness
In a finite MDP with $\gam < 1$, the Bellman equation \ref{bellman-v} has exactly one solution, and it is $\val{v_\pi}$.
:::

::: proof
Define the **Bellman operator** $(\mathcal{T}_\pi v)(s) = r_\pi(s) + \gam \sum_{s'} P_\pi(s, s')\, v(s')$. Each row of $P_\pi$ is a probability distribution, so for any two value functions $u$ and $v$,
$$\lVert \mathcal{T}_\pi u - \mathcal{T}_\pi v \rVert_\infty = \gam\, \lVert P_\pi (u - v) \rVert_\infty \le \gam\, \lVert u - v \rVert_\infty.$$
So $\mathcal{T}_\pi$ is a contraction, and by the Banach fixed-point theorem it has a unique fixed point. $\val{v_\pi}$ satisfies \ref{bellman-v}, so it is a fixed point, and therefore the only one.
:::

The theorem also holds in episodic tasks with $\gam = 1$, provided the policy reaches a terminal state with probability 1 from every state. Solving \ref{matrix} directly costs about $|\mathcal{S}|^3$ operations: trivial for the 25 cells of the gridworld, impossible for a game with $10^{20}$ positions.

### Solving by iteration {#iteration}

The contraction suggests a cheaper method: start from any $v_0$, for instance all zeros, and apply the operator again and again, $v_{k+1} = \mathcal{T}_\pi v_k$. Each application shrinks the distance to $\val{v_\pi}$ by at least a factor $\gam$:

$$\lVert v_k - \val{v_\pi} \rVert_\infty \le \gam^k\, \lVert v_0 - \val{v_\pi} \rVert_\infty. \label{rate}$$

This is iterative policy evaluation ([[policy-evaluation]]), the first algorithm of dynamic programming; the gridworld values of the story were computed this way. It needs the model $p$. Methods that learn without a model replace the sum over outcomes by one sampled transition: the target $\rew{R_{t+1}} + \gam\,\val{V(S_{t+1})}$ of TD(0) is \ref{bellman-v} with a single sample in place of the expectation ([[td0]]).

### Example: three cells of the gridworld {#example}

In the gridworld with the random policy ([[value-functions]]), every move from the center cell earns 0 and lands on one of four neighbors worth 2.25, 0.36, $-0.35$ and 0.74, so \ref{bellman-v} gives

$$\val{v_\pi(\text{center})} = \tfrac14 \cdot 0.9\,(2.25 + 0.36 - 0.35 + 0.74) \approx 0.67,$$

which is indeed the value of the center. From $A$, every move lands on $A'$ with $\rew{+10}$, so $\val{v_\pi(A)} = 10 + 0.9\,\val{v_\pi(A')} = 10 + 0.9 \times (-1.35) \approx 8.79$. In the bottom-left corner, two moves bump into a wall and stay, so the corner's value appears on both sides of its own equation:

$$\val{v} = \tfrac14 \cdot 0.9 \cdot (-0.97) + \tfrac14 \cdot 0.9 \cdot (-1.35) + \tfrac12\,(-1 + 0.9\,\val{v}) \quad\Longrightarrow\quad \val{v} \approx -1.86.$$

### Historical remarks {#history}

The equations are named after Richard Bellman, who made them the basis of dynamic programming (Bellman, 1957). His *principle of optimality*, that an optimal policy must remain optimal from whatever state it reaches, leads to the optimality version of the equation ([[optimality]]). The operator view, with contraction arguments, is the standard treatment of Puterman (1994).

## Card

### Idea

The value of a state equals the expected **next reward** plus $\gam$ times the expected **value of the next state**. One such equation holds for every state, so the values are tied together: knowing the neighbors' values tells you yours.

::: analogy
The price of a share today is the dividend it will pay plus what the share will be worth after the payment, discounted. Prices tomorrow determine prices today.
:::

### In symbols {#formula}

$$\val{v_\pi(s)} = \sum_a \pol{\pi(a \mid s)} \sum_{s', r} p(s', \rew{r} \mid s, a)\,\big[\rew{r} + \gam\,\val{v_\pi(s')}\big]$$

$$\val{q_\pi(s,a)} = \sum_{s', r} p(s', \rew{r} \mid s, a)\,\Big[\rew{r} + \gam \sum_{a'} \pol{\pi(a' \mid s')}\,\val{q_\pi(s', a')}\Big]$$

### Backup diagram {#backup}

{{backup v-pi}}

From the state, through each action the policy may take, to each state the environment may lead to.

### Why it matters {#why}

It is the equation everything else solves or samples. Dynamic programming iterates it with the model ([[policy-evaluation]]); TD learning samples it ([[td0]]); [[sarsa]] samples its $\val{q}$ version.

### Pitfalls

- The equation holds for the values of the policy being followed. Mixing the values of one policy with the actions of another breaks it.
- The value of a terminal state is 0 in the sum, always.
- A wall move that leaves the agent in place puts the state's own value on the right-hand side.

### Check yourself {#check}

::: question
What property of the state makes the Bellman equation possible?
---
The Markov property: once the next state is known, the expected return from it does not depend on how the agent got there.
:::

::: question
With one action that always gives $\rew{+1}$ and stays in the same state, what does the Bellman equation say for $\gam = 0.9$?
---
$v = 1 + 0.9\,v$, so $v = 10$: the same as summing the discounted rewards.
:::

::: question
Why does repeatedly applying the Bellman equation converge?
---
It is a contraction: each application shrinks the largest error by at least a factor $\gam$.
:::
