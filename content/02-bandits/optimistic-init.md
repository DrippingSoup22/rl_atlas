+++
summary = "Start every estimate far too high and simply act greedily: each pull disappoints, so the agent moves on and tries every arm before it settles."
change = "Start every estimate far above any real payout and set ε to 0: the exploring is done by disappointment instead of by random pulls."
prereqs = ["epsilon-greedy", "step-size"]
lab = "bandit-optimistic"
sources = [
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., §2.6, Figure 2.3 and Exercise 2.6", url = "http://incompleteideas.net/book/the-book-2nd.html" },
  { text = "Kaelbling (1993), Learning in Embedded Systems, MIT Press" },
  { text = "Brafman & Tennenholtz (2002), R-max: a general polynomial time algorithm for near-optimal reinforcement learning, Journal of Machine Learning Research 3", url = "https://www.jmlr.org/papers/v3/brafman02a.html" },
  { text = "Szita & Lőrincz (2008), The many faces of optimism: a unifying approach, Proceedings of the 25th International Conference on Machine Learning" },
]

[story]
scene = "bandit"
env = "testbed"
seed = 58
units = 1000
average = 200
formula = '\step{1}{\val{Q_1(a)} = 5 \qquad} \step{2}{\val{Q(A)} \leftarrow \val{Q(A)} + \alp\,\big[\rew{R} - \val{Q(A)}\big]}'

[story.numbers]
first = '\val{Q(6)} \leftarrow 5 + 0.1\,\big[\rew{-0.32} - 5\big] = \val{4.47}'

[story.runs]
optimistic = { algorithm = "optimistic-init", q0 = 5.0, epsilon = 0.0, alpha = 0.1, name = "optimistic, greedy (Q₁ = 5)" }
realistic = { algorithm = "epsilon-greedy", q0 = 0.0, epsilon = 0.1, alpha = 0.1, name = "realistic, ε = 0.1 (Q₁ = 0)" }
+++

## Story

::: step {run = "optimistic", at = 0, truth = false, numbers = true, formula = 1}
**The same ten machines**, and a purely greedy player, with one twist: every estimate starts at $\val{+5}$. On these machines no arm pays anywhere near that on average, so every estimate is wildly optimistic. The bars sit at the top of the scale; the numbers say 5.00.
:::

::: step {run = "optimistic", at = 0, truth = false, numbers = true, play = 10, pace = 700, formula = 2}
Watch the first ten pulls. Arm 6 pays $\rew{-0.32}$, and with step size $\alp = 0.1$ its estimate drops from 5 to 4.47. Now every other arm looks better, still at 5, so the greedy player moves on. Each pull disappoints in turn, and in ten pulls every arm has been tried once.
:::

::: step {run = "optimistic", at = 10, truth = false, numbers = true, formula = 2}
All ten estimates now sit between 4.26 and 4.64, ordered by how well each arm paid on its one pull. The player keeps going round: whatever it pulls drops a little, and another arm takes the lead. It is exploring without ever choosing at random.
:::

::: step {run = "optimistic", at = 100, truth = true, numbers = true}
After 100 pulls the estimates have come down to between 1.47 and 1.80, still above every true value, and the best arm, arm 2, has already been pulled more often than any other: 20 times.
:::

::: step {run = "optimistic", at = 1000, truth = true, numbers = true, focus = 2}
After 1000 pulls the player has settled: 842 pulls on arm 2. Look at the other estimates: they are still far too high. Arm 3's estimate is 0.86 while its value is $\rew{-1.99}$. The player stopped pulling those arms as soon as they fell below arm 2, not when they reached the truth. To act well, it only needed the top of the ranking to be right.
:::

::: step {run = "optimistic", at = 1000, truth = true, curves = ["optimistic", "realistic"], metric = "optimal"}
Averaged over 200 problems, against a realistic ε-greedy player ($\val{Q_1} = 0$, $\eps = 0.1$, same $\alp$). The optimistic player is worse at first, because it explores more, and better later, because once its optimism has worn off it stops exploring altogether. Note the spike near step 11. [Race them in the Lab](lab:bandit-optimistic).
:::

## Textbook

### Initial estimates are a bias {#bias}

Every estimate has to start somewhere. Write $\val{Q_1(a)}$ for the initial estimate of action $a$. With sample averages the initial value disappears at the first pull, because the step size $1/1$ replaces it by the reward ([[incremental-mean]]). With a constant step size $\alp$ it never disappears completely: after $n$ pulls of $a$ the estimate is

$$\val{Q_{n+1}(a)} = (1 - \alp)^n\,\val{Q_1(a)} + \sum_{i=1}^{n} \alp\,(1 - \alp)^{n-i}\,\rew{R_i}, \label{unrolled}$$

so the initial value keeps a weight $(1 - \alp)^n$ that shrinks geometrically ([[step-size]]). The initial values are a bias, but a useful one. They are an easy way to supply prior knowledge about which rewards to expect, and, set in the right way, a way to make the agent explore.

### Optimism makes a greedy agent explore {#optimism}

Set every initial estimate higher than any value the agent could plausibly meet, and act greedily, with $\eps = 0$. On the 10-armed testbed, where the values $\val{q_*(a)}$ are drawn from a standard normal distribution, $\val{Q_1(a)} = 5$ is wildly optimistic. Whichever arm the agent pulls first, its reward is almost surely far below 5, so the update lowers that arm's estimate, and some arm still at 5 now looks best. The agent is *disappointed* by every arm it tries, and moves on. Taking expectations in \ref{unrolled}, the estimate of an arm pulled $n$ times is

$$\mathbb{E}\big[\val{Q_{n+1}(a)}\big] = \val{q_*(a)} + (1 - \alp)^n\,\big(\val{Q_1(a)} - \val{q_*(a)}\big), \label{decay}$$

an optimism that wears off geometrically with the number of pulls. Because the greedy agent always pulls the arm whose estimate is currently highest, it keeps pulling arms until all of their optimism has worn down to roughly the same level, and only then do the true values start to decide the ranking. In this way all arms are tried, several times each, before the estimates settle. The exploring is done by the initial values; no random choice is needed.

### The first steps {#spike}

The first ten steps on the testbed are predictable. After one pull, an arm's estimate is $5 + 0.1\,(\rew{R} - 5) = 4.5 + 0.1\,\rew{R}$, which stays below the untouched value 5 unless the reward exceeds 5, which practically never happens. So the first ten pulls try each arm once. At step 11 the greedy choice is the arm whose single reward was the largest, and that is the best arm more often than any other. This is the spike in \ref{fig-curves} (Sutton & Barto, Exercise 2.6). It is short-lived: the second pull of that arm lowers its estimate again, below the arms pulled only once, and the agent goes back to cycling through them. Only after many rounds does the best arm pull ahead for good.

### Results on the testbed {#testbed}

::: figure {#fig-curves}
{{bandit-curves optimistic}}
The fraction of steps on which a best arm was chosen, for a greedy agent with optimistic initial estimates ($\val{Q_1} = 5$, $\eps = 0$) and an ε-greedy agent with realistic ones ($\val{Q_1} = 0$, $\eps = 0.1$), both with $\alp = 0.1$. Each curve averages 1000 runs; the Lab computes them when the figure comes into view. After Sutton & Barto, Figure 2.3.
:::

The optimistic method starts worse, because it spends the early steps working through every arm, and ends better: after a few hundred steps it chooses the best arm more often than ε-greedy, about 84% of the time against 76% over the last hundred steps. Its advantage is that its exploration ends by itself. ε-greedy keeps pulling random arms forever, at the rate $\eps$.

### Limits {#limits}

On a fixed testbed the trick works well. As a general answer to exploration it falls short, for four reasons.

- **The drive to explore is temporary.** It is spent in the first rounds of pulls. If the values change later, nothing sends the agent exploring again ([[nonstationary]]): all its curiosity was front-loaded, and a run has only one start.
- **It needs a sense of scale.** “Far above any plausible value” requires knowing what values are plausible. Too little optimism explores too little; too much wastes many pulls on arms whose estimates must first come down from the sky.
- **It needs a constant step size**, or some way of counting the initial estimate as evidence. With sample averages the first pull of each arm erases its optimism, and the agent is only greedy.
- **It does not keep the agent honest later.** Once the optimism has worn off, the agent is greedy, and an unlucky streak on the best arm can still leave it stuck on a worse one.

### Optimism in the face of uncertainty {#principle}

The trick is one instance of a principle that runs through the study of exploration: when unsure, act as if the world were as good as it plausibly could be. If the optimism was justified, the agent collects a high reward; if not, it learns something and its estimate comes down. Either way it is not stuck. [[ucb]] turns the principle into a rule, with a bonus that shrinks as an action is tried more. In problems with states, initializing $\val{Q(s,a)}$ optimistically makes [[q-learning]] and [[sarsa]] explore systematically, because every unvisited state–action pair looks attractive; R-max (Brafman & Tennenholtz, 2002) builds a whole algorithm on it, with guarantees, by treating every insufficiently explored state as maximally rewarding.

### Historical remarks {#history}

Sutton and Barto (§2.6) compare optimistic initial values with ε-greedy on the testbed. The principle of optimism in the face of uncertainty goes back at least to Kaelbling's interval estimation (1993), which chooses the action whose confidence interval reaches highest. R-max (Brafman & Tennenholtz, 2002) brought it to Markov decision processes with polynomial guarantees, and Szita and Lőrincz (2008) showed how far a carefully designed optimistic initialization alone can go.

## Card

### Idea

Start every estimate far above any real payout and always pull the best-looking arm. Every pull disappoints, so the pulled arm drops in the ranking and another arm gets tried. The agent explores every arm before settling, without a single random choice.

::: analogy
A new town where you assume every restaurant is excellent. Each visit brings you down to earth about that one place, so you keep trying the ones you have not been disappointed by yet, until the good ones stand out.
:::

### The update {#update}

$$\val{Q_1(a)} = \text{high, e.g. } 5 \qquad A \leftarrow \operatorname*{arg\,max}_a \val{Q(a)} \qquad \val{Q(A)} \leftarrow \val{Q(A)} + \alp\,\big[\rew{R} - \val{Q(A)}\big]$$

The optimism of an arm pulled $n$ times has shrunk by the factor $(1 - \alp)^n$.

### One change from [[epsilon-greedy]] {#change}

| | starts at | explores by |
| --- | --- | --- |
| [[epsilon-greedy]] | $\val{Q_1(a)} = 0$ | random pulls, with probability $\eps$, forever |
| optimistic | $\val{Q_1(a)} = 5$, with $\eps = 0$ | disappointment, until the optimism wears off |

### Backup diagram {#backup}

{{backup bandit}}

The same as for any bandit method: from the arm pulled to its reward. What changes is where the estimates start, not what they are updated from.

### Pseudocode

::: pseudocode
Parameters: optimistic first estimate $\val{Q_1}$ (e.g. 5), step size $\alp$ (e.g. 0.1)
Set $\val{Q(a)} = \val{Q_1}$ for every arm $a$ {#init}
Repeat:
  Choose $A$ with the highest $\val{Q}$, breaking ties at random {#choose}
  Pull $A$, observe $\rew{R}$ {#act}
  $\val{Q(A)} \leftarrow \val{Q(A)} + \alp\,[\rew{R} - \val{Q(A)}]$ {#update}
:::

### Perks

- Exploration for free: no randomness, no extra parameter beyond the initial value. [See it](lab:bandit-optimistic)
- The exploring stops by itself, so in the long run it beats ε-greedy on stationary problems.
- Tries every arm early and systematically, not by chance.

### Flaws

- Explores only at the start: useless when the values change later ([[nonstationary]]).
- Needs to know how high “optimistic” is; the scale of the rewards must be guessed.
- Needs a constant step size: with sample averages the first pull erases the optimism.

### Knobs

| Knob | Too low | Too high |
| --- | --- | --- |
| $\val{Q_1}$ first estimate | little or no exploration: acts like greedy | many pulls wasted bringing estimates down |
| $\alp$ step size | optimism wears off slowly: long exploration | optimism vanishes after a pull or two: little exploration |

### Pitfalls

- Using sample averages ($\alp = 1/n$): the first reward replaces the optimistic value, and nothing is left of the trick.
- Comparing methods only by their early curves: optimism looks bad at first, by design.
- Using it on a problem that changes: it will not notice when another arm becomes better.

### Check yourself {#check}

::: question
On the testbed with $\val{Q_1} = 5$ and $\alp = 0.1$, why does the greedy agent try every arm in its first ten pulls?
---
After one pull an arm's estimate is $4.5 + 0.1\,R$, below 5 unless the reward exceeds 5, which practically never happens. So each pull makes an untried arm, still at 5, the greedy choice.
:::

::: question
Why does the trick need a constant step size?
---
With sample averages the first update uses step size 1 and replaces the initial estimate by the reward. The optimism is gone after one pull.
:::

::: question
After 1000 pulls the optimistic agent's estimates of its abandoned arms are still far too high. Is that a problem?
---
Not for acting: it only needs the best arm's estimate to be higher than the others. It stopped pulling the other arms once they fell below the best one, so their estimates never came down further.
:::
