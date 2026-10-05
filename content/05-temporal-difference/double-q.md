+++
summary = "Keep two independent tables: one picks the best next move, the other says what it is worth. The max over noisy estimates no longer fools itself."
change = "Keep two sets of estimates and use one to choose the best next action and the other to value it, instead of letting the same noisy estimates both choose and evaluate."
prereqs = ["q-learning", "explore-exploit"]
lab = "max-bias"
sources = [
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., §6.7, Example 6.7 and Figure 6.5", url = "http://incompleteideas.net/book/the-book-2nd.html" },
  { text = "van Hasselt (2010), Double Q-learning, Advances in Neural Information Processing Systems 23" },
  { text = "Thrun & Schwartz (1993), Issues in using function approximation for reinforcement learning, Proceedings of the 4th Connectionist Models Summer School" },
  { text = "Smith & Winkler (2006), The optimizer's curse: skepticism and postdecision surprise in decision analysis, Management Science 52", url = "https://doi.org/10.1287/mnsc.1050.0451" },
  { text = "van Hasselt, Guez & Silver (2016), Deep reinforcement learning with double Q-learning, Proceedings of the 30th AAAI Conference on Artificial Intelligence", url = "https://arxiv.org/abs/1509.06461" },
]

[story]
scene = "graph"
env = "max-bias"
seed = 1
average = 1000
formula = '\step{1}{\val{Q_1(S,A)} \leftarrow \val{Q_1(S,A)} + \alp\,\big[\rew{R} + \gam\,} \step{2}{\val{Q_2\big(S^\prime\!, \operatorname*{arg\,max}_a Q_1(S^\prime\!,a)\big)}} \step{3}{- \val{Q_1(S,A)}\,\big]}'

[story.runs]
q = { algorithm = "q-learning", alpha = 0.1, epsilon = 0.1, gamma = 1.0, units = 300, name = "Q-learning" }
dq = { algorithm = "double-q", alpha = 0.1, epsilon = 0.1, gamma = 1.0, units = 300, name = "Double Q-learning" }
+++

## Story

::: step {run = "q", at = 0}
**A trap built from noise.** The agent starts in A. Going right ends the episode at once, with nothing gained or lost. Going left leads to B, where ten different moves all end the episode with a random reward that averages $\rew{-0.1}$. On average, left is a bad idea. With $\eps = 0.1$, the best an exploring agent can do is go left 5% of the time.
:::

::: step {run = "q", at = 0, play = 4, pace = 380}
Q-learning starts exploring. In B, each of the ten moves gets a few noisy rewards. Some happen to come out positive, so their estimates, the bars above B, rise above 0.
:::

::: step {run = "q", at = 30}
After 30 episodes, the luckiest of B's ten estimates is $\val{+0.31}$. Q-learning's target for going left from A uses the **largest** estimate in B, so going left now looks worth $\val{+0.20}$, more than going right, worth 0. Q-learning went left in 9 of its first 10 episodes... and in 27 of its first 30. Every one of B's moves loses on average, yet the max of ten noisy estimates is almost always positive.
:::

::: step {run = "dq", at = 30, formula = 2}
**Double Q-learning** keeps two independent tables. To update one, it lets that table pick the best-looking move in B, but takes the *other* table's estimate of that move. A move that looks good in one table only by luck is unlikely to be lucky in the other too, so the luck does not add up. After the same 30 episodes, going left is worth about 0, and the agent went left in only 6 of them.
:::

::: step {run = "dq", at = 300}
After 300 episodes, Double Q-learning goes left rarely, close to the 5% that exploration forces on it. Q-learning gets there too, eventually, as every move in B is tried often enough for its noise to average out.
:::

::: step {run = "dq", at = 300, curves = ["q", "dq"], metric = "left"}
Averaged over 1000 runs: early on, Q-learning goes left from A far more often than any sensible agent should, peaking above 90%. Double Q-learning never leaves the neighborhood of 50% at the start and quickly settles near the optimum. [Race them in the Lab](lab:max-bias).
:::

## Textbook

### Maximization bias {#bias}

All the control algorithms seen so far involve a maximization in the construction of their target policies: [[q-learning]] takes the max over the next action values, and [[sarsa]] usually follows an ε-greedy policy, which also involves a max. Using the maximum of *estimated* values as an estimate of the maximum *value* introduces a positive bias. Suppose that in some state every action has true value 0, but the estimates are uncertain, some above 0 and some below. The maximum of the true values is 0, yet the maximum of the estimates is positive. This is **maximization bias**. It follows from Jensen's inequality, since the max is a convex function:

$$\mathbb{E}\Big[\max_a \val{Q(s,a)}\Big] \;\ge\; \max_a\, \mathbb{E}\big[\val{Q(s,a)}\big]. \label{jensen}$$

With bootstrapping, the bias travels: the inflated max becomes part of the targets of the states that lead to $s$. Decision analysts know the same effect as the *optimizer's curse*: the option chosen because its estimate is highest is, on average, worse than its estimate (Smith & Winkler, 2006).

::: example {#ex-bias} Maximization bias (Sutton & Barto, Example 6.7)
An episode starts in state A, with two actions. *Right* ends the episode immediately with reward 0. *Left* leads, with reward 0, to state B, which has many actions, here ten, each ending the episode with a reward drawn from a normal distribution with mean $-0.1$ and variance 1. The expected return of *left* is $-0.1$, so *right* is better; with $\gam = 1$ and ε-greedy exploration at $\eps = 0.1$, the best behavior goes left $\eps/2 = 5\%$ of the time.
:::

Q-learning initially learns to take *left* much more often than *right*, and still does so more than optimal after hundreds of episodes (\ref{fig-bias}). Early on, the largest of B's ten estimates is almost always positive, so the target for *left* in A is too.

### Two estimates {#two}

The problem is that the same samples are used both to determine the maximizing action and to estimate its value. Divide the experience into two sets and learn two independent estimates, $\val{Q_1}$ and $\val{Q_2}$, of the true values $\val{q(a)}$. Use one to *choose*, $A^* = \operatorname*{arg\,max}_a \val{Q_1(a)}$, and the other to *evaluate*: $\val{Q_2(A^*)} = \val{Q_2(\operatorname*{arg\,max}_a Q_1(a))}$.

::: lemma {#lem-unbiased} Choosing with one estimate, valuing with another
If $\val{Q_2}$ is independent of $\val{Q_1}$ and unbiased, $\mathbb{E}[\val{Q_2(a)}] = \val{q(a)}$ for every $a$, then $\mathbb{E}\big[\val{Q_2(A^*)}\big] = \mathbb{E}\big[\val{q(A^*)}\big] \le \max_a \val{q(a)}$, where $A^* = \operatorname*{arg\,max}_a \val{Q_1(a)}$.
:::

::: proof
Condition on $\val{Q_1}$: it fixes $A^*$, and by independence $\val{Q_2(A^*)}$ then has mean $\val{q(A^*)}$. Averaging over $\val{Q_1}$ gives $\mathbb{E}[\val{q(A^*)}]$, the true value of an action, which is at most the best true value.
:::

So the double estimate is never optimistic in expectation; if anything it is pessimistic, since $\val{q(A^*)}$ falls short of the maximum whenever $\val{Q_1}$ picks the wrong action. Swapping the roles gives a second estimate, $\val{Q_1(\operatorname*{arg\,max}_a Q_2(a))}$. This is **double learning**. Two estimates are learned, but each is updated with only half of the experience, so the memory doubles while the computation per step does not.

### The update {#update}

**Double Q-learning** applies the idea to Q-learning. At each step, a coin flip decides which table is updated. With probability $\tfrac12$,

$$\begin{gathered} \val{Q_1(S_t,A_t)} \leftarrow \val{Q_1(S_t,A_t)} + \alp\,\big[\,\rew{R_{t+1}} + \gam\,\val{Q_2(S_{t+1}, A^*)} - \val{Q_1(S_t,A_t)}\,\big], \\ \text{where } A^* = \operatorname*{arg\,max}_a \val{Q_1(S_{t+1},a)}, \end{gathered} \label{update-rule}$$

and otherwise the same update with $\val{Q_1}$ and $\val{Q_2}$ swapped. The two tables are treated completely symmetrically. The behavior policy can use both, for instance ε-greedy with respect to their sum or average.

### The algorithm {#algorithm}

::: algorithm {#alg-dq} Double Q-learning, for estimating $\val{Q_1} \approx \val{Q_2} \approx \val{q_*}$
Parameters: step size $\alp \in (0, 1]$, small $\eps > 0$
Initialize $\val{Q_1(s,a)}$ and $\val{Q_2(s,a)}$, for all $s \in \mathcal{S}^+$, $a \in \mathcal{A}(s)$, such that $\val{Q(\textit{terminal}, \cdot)} = 0$
Loop for each episode:
  Initialize $S$
  Loop for each step of the episode, until $S$ is terminal:
    Choose $A$ from $S$ using the policy ε-greedy in $\val{Q_1} + \val{Q_2}$
    Take action $A$, observe $\rew{R}$, $S'$
    With probability 0.5:
      $\val{Q_1(S,A)} \leftarrow \val{Q_1(S,A)} + \alp\,\big[\rew{R} + \gam\,\val{Q_2(S', \operatorname*{arg\,max}_a Q_1(S',a))} - \val{Q_1(S,A)}\big]$
    else:
      $\val{Q_2(S,A)} \leftarrow \val{Q_2(S,A)} + \alp\,\big[\rew{R} + \gam\,\val{Q_1(S', \operatorname*{arg\,max}_a Q_2(S',a))} - \val{Q_2(S,A)}\big]$
    $S \leftarrow S'$
:::

Van Hasselt (2010) proved that Double Q-learning converges to the optimal action values under the same conditions as Q-learning. The atlas's Lab acts ε-greedily on the average of the two tables, which picks the same actions as their sum.

### Example: escaping the trap {#example}

::: figure {#fig-bias}
{{max-bias}}
The fraction of episodes in which each method goes left from A, the worse action, in the maximization-bias example, with $\alp = 0.1$, $\eps = 0.1$ and $\gam = 1$. Each curve averages 1000 runs; the line marks the best possible behavior with $\eps = 0.1$. The Lab computes the runs when the figure comes into view. After Sutton & Barto, Figure 6.5.
:::

In the runs above, Q-learning goes left in more than 90% of the episodes around episode 20, and still in about 12% after 300 episodes. Double Q-learning starts at the even split that ties produce, goes left less and less, and is close to the optimal 5% after a few hundred episodes (\ref{fig-bias}). It is essentially unaffected by maximization bias.

### Beyond tables {#beyond}

Overestimation is worse with function approximation, where the errors of a network act like noise in the max, and it was identified there early on (Thrun & Schwartz, 1993). DQN computes its target with the max of a *target network*, a periodic copy of the online network ([[target-network]]). Double DQN reuses that second network as the evaluator: the online network chooses $\operatorname*{arg\,max}_a Q(S_{t+1}, a; \mathbf{w})$, the target network values it, $Q(S_{t+1}, \cdot\,; \mathbf{w}^-)$, and the change reduces DQN's overestimation and improves its scores on Atari games (van Hasselt, Guez & Silver, 2016; [[dqn-extensions]]). In continuous control, TD3 takes the minimum of two critics for the same reason ([[td3]]).

### Properties {#properties}

- **No maximization bias.** The choice of the best action and the estimate of its value use independent experience.
- **A slight pessimism instead.** The double estimate can underestimate when the choosing table picks a worse action; this is usually harmless.
- **Twice the memory, half the data per table.** Each table learns from about half the transitions, which can slow learning when the bias was not a problem.
- **Same cost per step** as Q-learning: one max, one update.
- **Off-policy, like Q-learning,** and with the same convergence guarantees in the tabular case.

### Historical remarks {#history}

Overestimation in Q-learning with function approximation was analyzed by Thrun and Schwartz (1993). Van Hasselt (2010) introduced double estimators into reinforcement learning as Double Q-learning and proved its convergence; van Hasselt, Guez and Silver (2016) carried the idea into deep reinforcement learning as Double DQN. The maximization-bias example follows Sutton and Barto (Example 6.7). The same phenomenon is known in decision analysis as the optimizer's curse (Smith & Winkler, 2006).

## Card

### Idea

Q-learning's target takes the max over noisy estimates, which is biased upward: some estimate is always lucky. Double Q-learning keeps two independent tables; one picks the best next move, the other says what it is worth. Luck in one table is not luck in the other, so the bias disappears.

::: analogy
Choosing a restaurant from one critic's ratings, then judging your choice by a second critic's rating of it. If the first critic overrated a place by chance, the second critic is unlikely to have made the same mistake.
:::

### The update {#update}

$$\begin{gathered} \val{Q_1(S,A)} \leftarrow \val{Q_1(S,A)} + \alp\,\big[\,\rew{R} + \gam\,\val{Q_2(S', A^*)} - \val{Q_1(S,A)}\,\big] \\ \text{where } A^* = \operatorname*{arg\,max}_a \val{Q_1(S',a)} \end{gathered}$$

Or, with probability ½, the same with the two tables swapped. Act ε-greedily on their sum.

### One change from Q-learning {#change}

| | target uses |
| --- | --- |
| [[q-learning]] | $\max_a \val{Q(S',a)}$: the same table chooses and evaluates |
| Double Q-learning | $\val{Q_2(S', \operatorname*{arg\,max}_a Q_1(S',a))}$: one table chooses, the other evaluates |

### Backup diagram {#backup}

{{backup double-q}}

The shape of Q-learning's diagram, but the best next move is chosen by one table and valued by the other.

### Pseudocode

::: pseudocode
Parameters: step size $\alp \in (0, 1]$, exploration rate $\eps > 0$, discount $\gam$
Set $\val{Q_1(s,a)} = \val{Q_2(s,a)} = 0$ for every state and action {#init}
Repeat for each episode:
  Start: $S$ {#start}
  Repeat until $S$ is terminal:
    Choose $A$ in $S$ ε-greedily on the average of $\val{Q_1}$ and $\val{Q_2}$ {#choose}
    Take $A$, observe $\rew{R}$ and $S'$ {#act}
    With probability ½: $\val{Q_1(S,A)} \leftarrow \val{Q_1(S,A)} + \alp\,[\rew{R} + \gam\,\val{Q_2(S', \operatorname*{arg\,max}_a Q_1(S',a))} - \val{Q_1(S,A)}]$ {#update-1}
    Otherwise: $\val{Q_2(S,A)} \leftarrow \val{Q_2(S,A)} + \alp\,[\rew{R} + \gam\,\val{Q_1(S', \operatorname*{arg\,max}_a Q_2(S',a))} - \val{Q_2(S,A)}]$ {#update-2}
    $S \leftarrow S'$ {#next}
:::

### Perks

- Removes maximization bias: no more preferring actions that only look good by luck. [See it](lab:max-bias)
- Costs the same per step as Q-learning.
- The same idea fixes overestimation in deep RL (Double DQN, TD3).

### Flaws

- Twice the memory, and each table learns from only half the experience.
- Can underestimate slightly instead.
- When noise is small, it adds little and may learn a bit more slowly than Q-learning.

### Knobs

| Knob | Too low | Too high |
| --- | --- | --- |
| $\alp$ step size | learns slowly | estimates follow the latest rewards; the tables become less independent of luck |
| $\eps$ exploration | some moves rarely tried, their estimates stay noisy | more random behavior; the optimum itself goes left more often |

### Pitfalls

- Using the same table to pick and to evaluate: that is plain Q-learning again.
- Updating both tables from the same transition: they stop being independent.
- Acting on only one table: the behavior should use both, for instance their sum.

### Check yourself {#check}

::: question
Ten actions all have true value 0 and noisy estimates. Why is the largest estimate positive on average?
---
The maximum picks whichever estimate happens to be highest, so it collects the lucky noise. The max is convex, and by Jensen's inequality the expected maximum is at least the maximum of the expectations.
:::

::: question
Why doesn't Double Q-learning inherit that bias?
---
The table that picks the best action is not the one that values it. Given the choice, the other table's estimate is an unbiased estimate of that action's true value, which is at most the best true value.
:::

::: question
In deep RL, where does Double DQN get its second estimate?
---
From the target network: the online network chooses the best next action, and the target network supplies its value.
:::
