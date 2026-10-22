+++
summary = "The value of a state equals what you expect to get on the next step plus the discounted value of where you land: one equation per state that ties all the values together."
prereqs = ["value-functions", "return", "mdp"]
sources = [
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., §3.5 and Figure 3.4", url = "http://incompleteideas.net/book/the-book-2nd.html" },
  { text = "Bellman (1957), Dynamic Programming, Princeton University Press" },
  { text = "Puterman (1994), Markov Decision Processes: Discrete Stochastic Dynamic Programming, Wiley, §6.1 and §6.2" },
]

[story]
scene = "grid"
env = "gridworld"
gamma = 0.9
digits = 2
formula = '\begin{aligned} \step{1}{\rew{G_t}} &\step{1}{= \rew{R_{t+1}} + \gam\,\rew{G_{t+1}}} \\ \step{2}{\val{v_\pi(s)}} &\step{2}{= \sum_a \pol{\pi(a \mid s)}} \step{3}{\sum_{s^\prime\!,\,r} p(s^\prime\!, \rew{r} \mid s, a)} \step{4}{\big[\,\rew{r} + \gam\,\val{v_\pi(s^\prime)}\,\big]} \end{aligned}'

[story.numbers]
center = '\val{0.67} = \tfrac14\,\big(\val{2.03} + \val{0.32} + \val{(-0.32)} + \val{0.66}\big)'
jump = '\val{8.79} = \rew{10} + 0.9 \times \val{(-1.35)}'
corner = '\val{-1.86} = \tfrac14\,\big(\val{(-0.88)} + \val{(-1.21)} + 2 \times \val{(-2.67)}\big)'

[story.runs]
sweeps = { algorithm = "policy-evaluation", policy = "random", gamma = 0.9, theta = 0.000001, sync = true, units = 100 }
+++

## Story

::: step {v = "random", values = true, agent = "none"}
**Twenty-five numbers.** These are the values of the random walker of [[value-functions]], who picks each of the four moves with probability ¼, with $\gam = 0.9$. Each number is the average discounted return of endless walks from its cell. They look like 25 separate facts, but each one can be checked from its neighbors alone.
:::

::: step {v = "random", values = true, agent = "none", formula = 1}
**Why the neighbors are enough.** A return splits into the first reward plus $\gam$ times the return from the next cell on ([[return]]). Average both sides over many walks: the value of a cell is the average first reward, plus $\gam$ times the average value of the cell the first move lands on. Everything after the first move is already summed up in that neighbor's value.
:::

::: step {v = "random", values = true, focus = [2, 2], next = true, policy = "random", agent = "none", formula = 1}
**The center cell is worth 0.67.** From here the walker has four moves, each with probability ¼. Every one of them pays 0 and lands on a neighbor, outlined, whose value can be read off the grid.
:::

::: step {v = "random", values = true, focus = [2, 2], next = true, policy = "random", agent = "none", formula = 1, numbers = "center"}
Each move is worth its reward plus 0.9 times the value of where it lands:

| move | lands on a cell worth | 0 + 0.9 × that |
| --- | --- | --- |
| up | 2.25 | 2.03 |
| right | 0.36 | 0.32 |
| down | −0.35 | −0.32 |
| left | 0.74 | 0.66 |

The walker takes each move a quarter of the time, so the center is worth the average of the last column: **0.67**, the number in the grid.
:::

::: step {v = "random", values = true, focus = [0, 1], next = true, agent = "none", formula = 1, numbers = "jump"}
**Cell $A$ is worth 8.79.** Every move from $A$ jumps to $A'$ and pays $\rew{+10}$, so all four rows of its table are the same: 10 plus 0.9 times the value of $A'$, which is $-1.35$. $A$ is worth less than its jackpot because $A'$, by the bottom wall, is a poor place to land.
:::

::: step {v = "random", values = true, focus = [4, 0], next = true, agent = "none", formula = 1, numbers = "corner"}
**The corner is worth −1.86.** Up and right lead to cells worth $-0.97$ and $-1.35$. Down and left hit the wall: each costs $\rew{-1}$ and leaves the walker in the corner, so the corner's own value appears in its own table, $-1 + 0.9 \times (-1.86) = -2.67$. The average of the four rows gives $-1.86$ back. The corner agrees with itself too.
:::

::: step {v = "random", values = true, agent = "none", formula = 4}
**This check is the Bellman equation.** All 25 cells pass it. Written once for any state $s$, it reads, piece by piece:

- $\sum_a \pol{\pi(a \mid s)}$: average over the moves the policy makes (¼ each here);
- $\sum_{s', r} p(s', \rew{r} \mid s, a)$: average over what the world does after the move (here each move has one sure outcome);
- $\rew{r} + \gam\,\val{v_\pi(s')}$: one row of the tables above.
:::

::: step {run = "sweeps", at = 0, values = true, play = 1, pace = 120, fine = true, formula = 4}
**From a check to a recipe.** The equation can also *produce* the values. Start with every cell at 0, and replace each value by the right-hand side, computed from the current numbers. Watch one sweep: only $A$, $B$ and the cells along the walls change, because so far only the first reward counts.
:::

::: step {run = "sweeps", at = 5, values = true, formula = 4}
After 5 sweeps, each value counts the first 5 rewards of the walks from its cell. The numbers keep moving toward the table we started with, by less and less each time.
:::

::: step {run = "sweeps", at = 100, values = true, formula = 4}
After about 60 sweeps the numbers stop moving: the grid shows the same 25 numbers as at the start. From any starting numbers it would end here, because each sweep shrinks the largest error by at least a factor $\gam$. So exactly one table agrees with itself, and repeating the equation finds it. This is [[policy-evaluation]]; [[td0]] does the same from experience, one move at a time.
:::

## Textbook

### From the return to an equation {#derivation}

The value of a state is the expected return from it ([[value-functions]]):

$$\val{v_\pi(s)} = \mathbb{E}_\pi[\,\rew{G_t} \mid S_t = s\,]. \label{v-def}$$

The return, in turn, obeys a one-step recursion ([[return]]): what is collected from time $t$ on is the next reward plus $\gam$ times what is collected from time $t+1$ on,

$$\rew{G_t} = \rew{R_{t+1}} + \gam\,\rew{G_{t+1}}. \label{recursion}$$

The Bellman equation is what \ref{recursion} becomes after taking expectations. We derive it in three short steps, each using one fact about averages.

**Step 1: split the expectation.** Put \ref{recursion} into \ref{v-def}. The expectation of a sum is the sum of the expectations, and a constant factor comes out ([[a:expectation|linearity of expectation]]):

$$\val{v_\pi(s)} = \mathbb{E}_\pi[\,\rew{R_{t+1}} \mid S_t = s\,] + \gam\,\mathbb{E}_\pi[\,\rew{G_{t+1}} \mid S_t = s\,]. \label{split}$$

The first term is the expected next reward. The second is the expected return from the next step on, seen from $s$: it depends on where the agent lands.

**Step 2: average over where the agent lands.** The law of total expectation ([[a:conditional-expectation|conditional expectation]]) computes an average case by case: first the average within each case, then the average of the cases, weighted by their probabilities. Taking as cases the possible next states $s'$,

$$\begin{aligned} \mathbb{E}_\pi[\,\rew{G_{t+1}} \mid S_t = s\,] = \sum_{s'} \; & \Pr\{S_{t+1} = s' \mid S_t = s\} \\ & \times \mathbb{E}_\pi[\,\rew{G_{t+1}} \mid S_t = s,\ S_{t+1} = s'\,]. \end{aligned} \label{total}$$

**Step 3: forget the past.** Inside the sum, the agent is known to be in $s'$ at time $t+1$. By the Markov property ([[mdp]]), the future from there does not depend on how it got to $s'$, so knowing that it came from $s$ changes nothing:

$$\mathbb{E}_\pi[\,\rew{G_{t+1}} \mid S_t = s,\ S_{t+1} = s'\,] = \mathbb{E}_\pi[\,\rew{G_{t+1}} \mid S_{t+1} = s'\,] = \val{v_\pi(s')}.$$

The last equality is the definition \ref{v-def} again, one step later: the policy is stationary, so the value of $s'$ does not depend on the time at which the agent is there. Putting the three steps together, the expected return from the next step is the expected value of the next state, and \ref{split} becomes

$$\val{v_\pi(s)} = \mathbb{E}_\pi[\,\rew{R_{t+1}} + \gam\,\val{v_\pi(S_{t+1})} \mid S_t = s\,]. \label{bellman-compact}$$

This is the **Bellman equation** in its shortest form. The value of a state is the expected next reward, plus $\gam$ times the expected value of the next state. The infinite sum hidden in the return has disappeared: one step of the future, and the values of the states it leads to, are enough.

### Writing the expectation out {#explicit}

The expectation in \ref{bellman-compact} is over everything random in one step: which action the policy picks, and how the environment answers. It is easiest to write out in stages.

**A deterministic policy in a deterministic world.** If the policy always takes the action $\pol{\pi(s)}$, and that action always gives the reward $r$ and leads to the state $s'$, nothing is random and the expectation has a single term:

$$\val{v_\pi(s)} = \rew{r} + \gam\,\val{v_\pi(s')}.$$

**A random policy in a deterministic world.** If the policy picks action $a$ with probability $\pol{\pi(a \mid s)}$, and each action $a$ gives a sure reward $r_a$ and leads to a sure state $s'_a$, the expectation is a weighted average over the actions:

$$\val{v_\pi(s)} = \sum_a \pol{\pi(a \mid s)}\,\big[\,\rew{r_a} + \gam\,\val{v_\pi(s'_a)}\,\big]. \label{bellman-det}$$

This is the gridworld of the story: four actions, each with probability $\tfrac14$, each with one outcome.

**A random policy in a random world.** In general an action can lead to several outcomes. The dynamics $p(s', \rew{r} \mid s, a)$ give the probability that action $a$ in state $s$ yields reward $r$ and next state $s'$ ([[mdp]]). The probability that the policy picks $a$ *and* the world then answers with $(s', r)$ is the product $\pol{\pi(a \mid s)}\,p(s', \rew{r} \mid s, a)$ ([[a:conditional-probability|conditional probability]]). Averaging over all actions and all outcomes, each weighted by this probability ([[a:sums-products|sums]]):

$$\val{v_\pi(s)} = \sum_a \pol{\pi(a \mid s)} \sum_{s',\,r} p(s', \rew{r} \mid s, a)\,\big[\,\rew{r} + \gam\,\val{v_\pi(s')}\,\big]. \label{bellman-v}$$

Equation \ref{bellman-v} is the Bellman equation as it is usually written. Read it from the inside out:

- $\rew{r} + \gam\,\val{v_\pi(s')}$ is what one particular outcome is worth: its reward, plus the discounted value of where it leads;
- $\sum_{s', r} p(s', \rew{r} \mid s, a)\,[\ldots]$ is what the action $a$ is worth: the average over its outcomes;
- $\sum_a \pol{\pi(a \mid s)}\,[\ldots]$ is what the state is worth: the average over the actions the policy picks.

The first two pieces together are the action value $\val{q_\pi(s,a)}$, so \ref{bellman-v} also says that a state's value is the policy's average of its action values ([[value-functions]]). Many texts call \ref{bellman-v} the Bellman *expectation* equation, to set it apart from the optimality equation of [[optimality]], which has a maximum where this one has an average. A terminal state, if there is one, has value 0 by definition, and that 0 is used wherever $\val{v_\pi(s')}$ refers to it.

### Checking it on the gridworld {#example}

The equation can be checked by hand on the values of the random policy in the gridworld ([[value-functions]]), with $\gam = 0.9$. In the center cell, worth 0.67, every move pays 0 and lands on a neighbor, so \ref{bellman-det} has four terms, each with weight $\tfrac14$:

| move | neighbor's value | $0 + 0.9 \times$ value |
| --- | --- | --- |
| up | 2.25 | 2.03 |
| right | 0.36 | 0.32 |
| down | $-0.35$ | $-0.32$ |
| left | 0.74 | 0.66 |

The average of the last column is $2.69 / 4 \approx 0.67$, the center's value.

From cell $A$, every move jumps to $A'$, worth $-1.35$, and pays $\rew{+10}$, so all four terms are equal and $\val{v_\pi(A)} = 10 + 0.9 \times (-1.35) \approx 8.79$.

In the bottom-left corner two moves bump into the walls: they pay $\rew{-1}$ and leave the agent where it is, so the corner's own value $\val{v}$ appears on the right-hand side. The other two moves lead to cells worth $-0.97$ and $-1.35$:

$$\val{v} = \tfrac14 \big(0.9 \times (-0.97)\big) + \tfrac14 \big(0.9 \times (-1.35)\big) + \tfrac12 \big({-1} + 0.9\,\val{v}\big).$$

This is one equation with one unknown. Collecting the terms in $\val{v}$ gives $0.55\,\val{v} \approx -1.02$, so $\val{v} \approx -1.86$, the corner's value. The equation does not require a state's value to be known before it is used: it is a condition that all the values must satisfy together.

### For action values {#q}

The same reasoning, started from a fixed first action, gives an equation for $\val{q_\pi}$. Two relations from [[value-functions]] are enough. An action's value is the average over its outcomes of the reward plus the discounted value of the next state, and a state's value is the policy's average of its action values:

$$\val{q_\pi(s,a)} = \sum_{s',\,r} p(s', \rew{r} \mid s, a)\,\big[\,\rew{r} + \gam\,\val{v_\pi(s')}\,\big], \qquad \val{v_\pi(s')} = \sum_{a'} \pol{\pi(a' \mid s')}\,\val{q_\pi(s',a')}.$$

Substituting the second into the first:

$$\val{q_\pi(s,a)} = \sum_{s',\,r} p(s', \rew{r} \mid s, a)\,\Big[\,\rew{r} + \gam \sum_{a'} \pol{\pi(a' \mid s')}\,\val{q_\pi(s',a')}\Big]. \label{bellman-q}$$

The two equations are the two halves of one step glued in different orders: $\val{v_\pi}$ averages over the action and then over the outcome, $\val{q_\pi}$ over the outcome and then over the next action.

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

### One equation per state {#linear}

Equation \ref{bellman-v} holds for every state. In a finite MDP with $n$ states, the values $\val{v_\pi(s)}$ are therefore $n$ unknowns tied together by $n$ equations. Each equation is **linear** in the unknowns: they appear only multiplied by numbers (probabilities times $\gam$) and added, never multiplied by one another or put inside a maximum. Finding the values is solving a system of linear equations ([[a:linear-systems|linear systems]]).

::: example {#ex-robot} Two states
Take the recycling robot ([[mdp]]) with the policy “search when the battery is high, recharge when it is low”, and $\gam = 0.9$. Searching earns 2 and keeps the battery high with probability 0.7; recharging earns 0 and always leads to high. Writing $h$ and $l$ for the two unknown values, \ref{bellman-v} gives two equations:

$$\begin{aligned} h &= 2 + 0.9\,(0.7\,h + 0.3\,l), \\ l &= 0 + 0.9\,h. \end{aligned}$$

Substituting the second into the first leaves $h = 2 + 0.873\,h$, so $h \approx 15.75$ and $l = 0.9\,h \approx 14.17$.
:::

For many states, matrix notation keeps the bookkeeping short ([[a:vectors-matrices|vectors and matrices]]). Collect the values in a vector $\val{v_\pi}$ with one entry per state. Two quantities summarize what the policy does in one step:

- the **expected reward** in each state, $r_\pi(s) = \sum_a \pol{\pi(a \mid s)} \sum_{s', r} \rew{r}\,p(s', \rew{r} \mid s, a)$, a vector;
- the **transition matrix** of the policy, $P_\pi(s, s') = \sum_a \pol{\pi(a \mid s)}\,p(s' \mid s, a)$, the probability of moving from $s$ to $s'$ in one step. Each row is a probability distribution: its entries are nonnegative and add up to 1 ([[a:markov-chain|Markov chains]]).

Entry $s$ of the product $P_\pi \val{v_\pi}$ is $\sum_{s'} P_\pi(s, s')\,\val{v_\pi(s')}$, the expected value of the next state. So \ref{bellman-v}, for all states at once, reads

$$\val{v_\pi} = r_\pi + \gam\,P_\pi\,\val{v_\pi}. \label{matrix}$$

Moving the unknowns to the left, $(I - \gam P_\pi)\,\val{v_\pi} = r_\pi$, where $I$ is the identity matrix. When $\gam < 1$ the matrix $I - \gam P_\pi$ has an inverse, and

$$\val{v_\pi} = (I - \gam P_\pi)^{-1}\, r_\pi = \big(I + \gam P_\pi + \gam^2 P_\pi^2 + \cdots\big)\, r_\pi. \label{solution}$$

The series on the right, the matrix version of the geometric series ([[a:geometric-series|geometric series]]), gives the solution a direct meaning: $P_\pi^k r_\pi$ is the expected reward $k$ steps from now, so the value is the expected reward now, plus $\gam$ times the expected reward one step later, plus $\gam^2$ times the one after: the expected discounted return, as it should be.

Solving \ref{matrix} directly, by elimination, takes on the order of $n^3$ operations ([[a:growth-rates|growth rates]]): trivial for the 25 cells of the gridworld, impossible for a game with $10^{20}$ positions. That is the reason for the iterative method of the next sections.

### Why there is exactly one solution {#unique}

Does \ref{bellman-v} always have a solution, and could it have several? The answer comes from looking at the right-hand side of the equation as a machine that improves guesses.

**The Bellman operator.** Take *any* table of numbers $v$, one per state; it need not be the true values. Compute, in every state, the right-hand side of \ref{bellman-v} with $v$ in place of $\val{v_\pi}$:

$$(\mathcal{T}_\pi v)(s) = \sum_a \pol{\pi(a \mid s)} \sum_{s',\,r} p(s', \rew{r} \mid s, a)\,\big[\,\rew{r} + \gam\,v(s')\,\big]. \label{operator}$$

The result is a new table, written $\mathcal{T}_\pi v$. The rule that turns the table $v$ into the table $\mathcal{T}_\pi v$ is the **Bellman operator** of the policy: “operator” is the name for a function whose input and output are whole functions ([[a:operators|operators]]). In matrix notation, $\mathcal{T}_\pi v = r_\pi + \gam P_\pi v$.

The Bellman equation now says something short: $\val{v_\pi} = \mathcal{T}_\pi \val{v_\pi}$. The true values are a table that the operator leaves unchanged, a **fixed point** of the operator ([[a:fixed-points|fixed points]]). A wrong table is changed by the operator, as the story's sweeps show.

**Measuring how far apart two tables are.** To compare two tables $u$ and $v$, take their largest disagreement over all states, the **max norm** of their difference ([[a:norms|norms]]):

$$\lVert u - v \rVert_\infty = \max_s\, \lvert u(s) - v(s) \rvert.$$

If $\lVert u - v \rVert_\infty = 0.5$, the two tables agree to within 0.5 in every state, and no better somewhere.

::: lemma {#lem-contraction} The Bellman operator is a contraction
For any two tables $u$ and $v$, $\lVert \mathcal{T}_\pi u - \mathcal{T}_\pi v \rVert_\infty \le \gam\,\lVert u - v \rVert_\infty$.
:::

::: proof
Fix a state $s$ and subtract the two right-hand sides of \ref{operator}. The rewards are the same in both and cancel, leaving

$$(\mathcal{T}_\pi u)(s) - (\mathcal{T}_\pi v)(s) = \gam \sum_{s'} P_\pi(s, s')\,\big(u(s') - v(s')\big).$$

The sum is a weighted average of the differences $u(s') - v(s')$: the weights $P_\pi(s, s')$ are nonnegative and add up to 1. A weighted average can never be larger in size than the largest of the numbers averaged ([[a:max-argmax|max and argmax]]), and the largest difference in size is $\lVert u - v \rVert_\infty$. So $\lvert (\mathcal{T}_\pi u)(s) - (\mathcal{T}_\pi v)(s) \rvert \le \gam\,\lVert u - v \rVert_\infty$. This holds in every state $s$, so it holds for the largest of them, which is the claim.
:::

In words: whatever two tables go in, the tables that come out are closer together, by at least the factor $\gam$. A function with this property is called a **contraction** ([[a:contraction|contractions]]). The averaging over next states is what makes it work, and the discount $\gam < 1$ is what makes the shrinking strict.

::: theorem {#thm-unique} Existence and uniqueness
In a finite MDP with $\gam < 1$, the Bellman equation \ref{bellman-v} has exactly one solution, and it is $\val{v_\pi}$.
:::

::: proof
*At most one solution.* Suppose $u$ and $v$ were two different solutions, so that $u = \mathcal{T}_\pi u$ and $v = \mathcal{T}_\pi v$. The lemma would then give

$$\lVert u - v \rVert_\infty = \lVert \mathcal{T}_\pi u - \mathcal{T}_\pi v \rVert_\infty \le \gam\,\lVert u - v \rVert_\infty.$$

A positive number cannot be at most $\gam < 1$ times itself, so $\lVert u - v \rVert_\infty = 0$, and $u = v$.

*At least one solution.* $\val{v_\pi}$ is a solution: \ref{bellman-v} was derived from its definition in \ref{derivation}. (For a contraction, a solution also exists without that derivation: this is Banach's fixed-point theorem, and the iteration of the next section constructs it.)
:::

The theorem also holds in episodic tasks with $\gam = 1$, provided the policy reaches a terminal state with probability 1 from every state; the proof is the same, applied to several steps at once. Without either condition it can fail: with $\gam = 1$, a state that pays $\rew{+1}$ forever would need $v = 1 + v$, which no number satisfies.

### Solving by iteration {#iteration}

The contraction gives a cheaper way to find $\val{v_\pi}$ than solving \ref{matrix}. Start from any table $v_0$, all zeros for instance, and apply the operator again and again:

$$v_{k+1} = \mathcal{T}_\pi v_k.$$

Since $\val{v_\pi} = \mathcal{T}_\pi \val{v_\pi}$, the lemma applied to $v_k$ and $\val{v_\pi}$ gives $\lVert v_{k+1} - \val{v_\pi} \rVert_\infty \le \gam\,\lVert v_k - \val{v_\pi} \rVert_\infty$: every application shrinks the largest error by at least the factor $\gam$. After $k$ applications,

$$\lVert v_k - \val{v_\pi} \rVert_\infty \le \gam^k\, \lVert v_0 - \val{v_\pi} \rVert_\infty. \label{rate}$$

With $\gam = 0.9$, each sweep removes at least a tenth of the remaining error, and 44 sweeps divide it by at least 100, whatever the start. This is **iterative policy evaluation** ([[policy-evaluation]]), the first algorithm of dynamic programming, and the sweeps at the end of the story. Each sweep costs on the order of $n^2$ operations, or far fewer when each state has only a few successors, as in the gridworld.

The iteration needs the model $p$. Methods that learn without a model replace the average over outcomes by one sampled transition: the target $\rew{R_{t+1}} + \gam\,\val{V(S_{t+1})}$ of TD(0) is the inside of \ref{bellman-compact}, observed once instead of averaged ([[td0]]). The rest of the atlas is, to a large extent, a collection of ways to solve the Bellman equations approximately: from samples, for the states that matter, with functions that generalize.

### Historical remarks {#history}

The equations are named after Richard Bellman, who made them the basis of dynamic programming (Bellman, 1957). His *principle of optimality*, that an optimal policy must remain optimal from whatever state it reaches, leads to the optimality version of the equation ([[optimality]]). The operator view, with contraction arguments, is the standard treatment of Puterman (1994).

## Card

### Idea

The value of a state equals the expected **next reward** plus $\gam$ times the expected **value of the next state**. One such equation holds for every state, so the values are tied together: knowing the neighbors' values tells you yours. Repeating the equation from any starting guess finds the values.

::: analogy
The price of a share today is the dividend it will pay plus what the share will be worth after the payment, discounted. Prices tomorrow determine prices today.
:::

### In symbols {#formula}

The value of a state: average over the policy's actions and the world's answers of the reward plus the discounted value of where it leads.

$$\val{v_\pi(s)} = \sum_a \pol{\pi(a \mid s)} \sum_{s', r} p(s', \rew{r} \mid s, a)\,\big[\rew{r} + \gam\,\val{v_\pi(s')}\big]$$

The value of an action: the same step, with the next action averaged at the end.

$$\val{q_\pi(s,a)} = \sum_{s', r} p(s', \rew{r} \mid s, a)\,\Big[\rew{r} + \gam \sum_{a'} \pol{\pi(a' \mid s')}\,\val{q_\pi(s', a')}\Big]$$

All states at once: expected rewards plus the discounted expected next values.

$$\val{v_\pi} = r_\pi + \gam\,P_\pi\,\val{v_\pi}$$

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
Each application replaces every value by an average of next values, scaled by $\gam$. An average never strays further than the largest error, so the largest error shrinks by at least the factor $\gam$ each time.
:::
