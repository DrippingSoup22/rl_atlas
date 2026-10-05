+++
summary = "Pull the arm that looks best, except that now and then, with probability ε, pull one at random: just enough exploration to keep every estimate honest."
prereqs = ["k-armed-bandit", "incremental-mean", "explore-exploit"]
lab = "bandit-epsilon"
sources = [
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., §2.2–2.4 and Figure 2.2", url = "http://incompleteideas.net/book/the-book-2nd.html" },
  { text = "Auer, Cesa-Bianchi & Fischer (2002), Finite-time analysis of the multiarmed bandit problem, Machine Learning 47", url = "https://doi.org/10.1023/A:1013689704352" },
  { text = "Vermorel & Mohri (2005), Multi-armed bandit algorithms and empirical evaluation, European Conference on Machine Learning" },
  { text = "Mnih et al. (2015), Human-level control through deep reinforcement learning, Nature 518", url = "https://doi.org/10.1038/nature14236" },
]

[story]
scene = "bandit"
env = "testbed"
seed = 58
units = 1000
average = 200
formula = 'A \leftarrow \begin{cases} \step{1}{\operatorname*{arg\,max}_a \val{Q(a)}} & \step{2}{\text{with probability } 1 - \eps} \\ \step{2}{\text{an arm chosen at random}} & \step{2}{\text{with probability } \eps} \end{cases}'

[story.numbers]
stuck = '\val{Q(7)} = \rew{1.29} \;>\; 0 = \val{Q(a)} \ \text{for every untried arm } a'
found = '\val{Q(2)} = \rew{0.63} \;>\; \val{Q(7)} = \val{0.22}'

[story.runs]
greedy = { algorithm = "epsilon-greedy", epsilon = 0.0, alpha = 0.0, name = "greedy (ε = 0)" }
rare = { algorithm = "epsilon-greedy", epsilon = 0.01, alpha = 0.0, name = "ε = 0.01" }
explore = { algorithm = "epsilon-greedy", epsilon = 0.1, alpha = 0.0, name = "ε = 0.1" }
+++

## Story

::: step {run = "greedy", at = 0, truth = false, formula = 1}
**The ten machines again**, and a player that is purely **greedy**: it always pulls an arm with the highest estimate $\val{Q(a)}$, the average of what the arm has paid so far. At the start every estimate is 0, so the first choice is a tie, broken at random.
:::

::: step {run = "greedy", at = 0, truth = false, play = 3, pace = 900, formula = 1}
Watch its first three pulls. Arm 6 pays $\rew{-0.32}$, so its estimate drops below 0. Arm 3 pays $\rew{-1.93}$. Then arm 7 pays $\rew{+1.29}$.
:::

::: step {run = "greedy", at = 3, truth = false, focus = 7, numbers = "stuck", formula = 1}
Now arm 7 is the only arm whose estimate is above 0, so the greedy player pulls it again. And again: its average wobbles but stays above 0, the estimate of every arm never tried. The bar under arm 7 says it: a 100% chance of being pulled next.
:::

::: step {run = "greedy", at = 1000, truth = true, focus = 7}
After 1000 pulls the truth is revealed. 998 pulls went to arm 7, whose value is $\rew{0.38}$. Arm 2, worth $\rew{1.57}$, was never tried. More time would not have helped: an arm that looks bad is never pulled again, so a bad estimate is never corrected.
:::

::: step {run = "explore", at = 0, truth = false, formula = 2}
**ε-greedy** changes one thing. With probability $\eps$, here 0.1, it ignores its estimates and pulls an arm at random. Every arm keeps a chance of being tried, so every estimate can be corrected.
:::

::: step {run = "explore", at = 30, truth = false, play = 10, pace = 500, numbers = "found", formula = 2}
Same machines, same first pulls. Its random pulls go nowhere useful at first, but at pull 32 one lands on arm 2 for the first time. It pays $\rew{0.63}$, more than arm 7's average of $\val{0.22}$, and arm 2 becomes the greedy choice. Its next rewards confirm it.
:::

::: step {run = "explore", at = 1000, truth = true}
After 1000 pulls: 878 went to arm 2, the best arm. The random pulls cost something, about one pull in ten goes to an arm known to be worse, but they also keep the estimates of the other arms roughly right.
:::

::: step {run = "explore", at = 1000, truth = true, curves = ["greedy", "rare", "explore"], metric = "optimal"}
One problem proves little, so here are 200 of them. The greedy player gets stuck on a worse arm in most problems. With $\eps = 0.1$ the best arm is found quickly, but it can never be pulled more than 91% of the time. $\eps = 0.01$ learns more slowly, and would win in the end. [Race them in the Lab](lab:bandit-epsilon).
:::

## Textbook

### Greedy action selection {#greedy}

In a $k$-armed bandit ([[k-armed-bandit]]), the agent keeps an estimate $\val{Q_t(a)}$ of the value of each action, such as the sample average of its rewards ([[incremental-mean]]). The simplest way to act on the estimates is to always choose an action with the largest one:

$$A_t = \operatorname*{arg\,max}_a \val{Q_t(a)}, \label{greedy-rule}$$

with ties broken at random. This **greedy** rule exploits its current knowledge to maximize the immediate reward, and spends no time at all trying actions that look worse.

That is its flaw. An estimate changes only when its action is chosen. If the estimate of the best action is ever below that of another action, perhaps because its first reward was unlucky or because it was never tried, the greedy rule may never choose it again, and the error is never corrected. In the story, an arm with value 0.38 pays 1.29 on its first pull; from then on its average stays above the default estimate 0 of the untried arms, and the greedy player never tries the best arm, worth 1.57. The failure does not wear off with time: it is permanent.

### ε-greedy {#rule}

A simple remedy is to behave greedily most of the time and, every once in a while, choose an action at random.

::: definition {#def-eps} ε-greedy action selection
Let $0 \le \eps \le 1$. At each step, with probability $1 - \eps$ choose a greedy action, $\operatorname*{arg\,max}_a \val{Q_t(a)}$, breaking ties at random; otherwise, with probability $\eps$, choose an action uniformly at random from all $k$ actions.
:::

Because the random choice includes the greedy action, a unique greedy action is chosen with probability $1 - \eps + \eps / k$ and every other action with probability $\eps / k$:

$$\pol{\pi_t(a)} = \begin{cases} 1 - \eps + \dfrac{\eps}{k} & \text{if } a = \operatorname*{arg\,max}_{a'} \val{Q_t(a')}, \\[1ex] \dfrac{\eps}{k} & \text{otherwise.} \end{cases} \label{probs}$$

With $\eps = 0$ the rule is greedy; with $\eps = 1$ it is uniformly random. Combined with sample averages, it gives Sutton and Barto's *simple bandit algorithm* (\ref{alg-eps}).

### What happens in the limit {#limit}

::: theorem {#thm-limit} ε-greedy in a stationary bandit
Let $\eps > 0$ be constant and let the estimates be sample averages. Then with probability 1 every action is chosen infinitely often and $\val{Q_t(a)} \to \val{q_*(a)}$ for every $a$. If the best action is unique, the probability of choosing it converges to $1 - \eps + \eps / k$.
:::

::: proof
At every step, independently of the past, the rule makes a random choice with probability $\eps$, and that choice is action $a$ with probability $1/k$. So each step chooses $a$ through a random choice with probability $\eps / k$, independently of all other steps, and by the second Borel–Cantelli lemma this happens infinitely often with probability 1. Each action's estimate is then the average of infinitely many independent rewards, so it converges to its value by the strong law of large numbers. Once every estimate is within half of the smallest gap of its value, the greedy action is the best action, and the rule chooses it with probability $1 - \eps + \eps / k$.
:::

The guarantee is about the estimates, not about the behavior: with a constant $\eps$, the agent keeps choosing worse actions a fraction $\eps (k - 1)/k$ of the time forever. Its regret, the reward lost by not always choosing the best action, grows linearly with time.

### Decaying exploration {#decay}

Exploration is most valuable early, when the estimates are poor. Reducing $\eps$ over time keeps the benefits of exploring while letting the behavior become greedy in the limit. If $\eps_t$ decays like $1/t$, with a constant tuned to the gaps between the values, ε-greedy achieves regret that grows only logarithmically, the best possible rate (Auer, Cesa-Bianchi & Fischer, 2002). The catch is the tuning: the right constant depends on the gaps, which the agent does not know, and a schedule that decays too fast stops exploring before the best action is found. In full reinforcement learning, schedules that decay to zero slowly enough are what the convergence theorems of control methods require ([[sarsa]]); in practice $\eps$ is often decayed from 1 to a small constant. DQN, for example, lowered it linearly from 1 to 0.1 over its first million frames (Mnih et al., 2015).

### Results on the testbed {#testbed}

\ref{fig-curves} compares the greedy rule with ε-greedy for $\eps = 0.01$ and $\eps = 0.1$ on the 10-armed testbed, all with sample averages.

::: figure {#fig-curves}
{{bandit-curves epsilon}}
Average reward (top) and the fraction of steps on which a best arm was chosen (bottom), for ε-greedy with $\eps = 0.1$, $\eps = 0.01$ and $\eps = 0$ (greedy). Each curve averages 1000 runs, each on a new testbed problem; the Lab computes them when the figure comes into view. After Sutton & Barto, Figure 2.2.
:::

The greedy method improves fastest at the very start, then levels off at a lower reward: in most problems it settles on a worse arm, and in the end it chooses the best one on only about a third of the steps. With $\eps = 0.1$ the agent explores more, finds the best arm sooner, and chooses it about 80% of the time after 1000 steps; it can never exceed $1 - 0.1 + 0.01 = 91\%$. With $\eps = 0.01$ it improves more slowly, but its ceiling is 99.1%, so in the long run it overtakes $\eps = 0.1$.

How much exploration pays depends on the task. With noisier rewards, more exploration is needed to find the best arm. With deterministic rewards, one pull of each arm would reveal its value, and the greedy method would do best once every arm had been tried. And if the values change over time, exploration never stops being necessary, because the best arm today may not be the best tomorrow ([[nonstationary]]).

### The algorithm {#algorithm}

::: algorithm {#alg-eps} A simple bandit algorithm (ε-greedy with sample averages)
Initialize, for $a = 1$ to $k$: $\val{Q(a)} \leftarrow 0$, $N(a) \leftarrow 0$
Loop forever:
  $A \leftarrow \operatorname*{arg\,max}_a \val{Q(a)}$ with probability $1 - \eps$ (ties broken at random), or a random action with probability $\eps$
  $\rew{R} \leftarrow \textit{bandit}(A)$
  $N(A) \leftarrow N(A) + 1$
  $\val{Q(A)} \leftarrow \val{Q(A)} + \tfrac{1}{N(A)}\,[\rew{R} - \val{Q(A)}]$
:::

Each step costs $O(k)$ for the arg max and $O(1)$ for the update; memory is two numbers per action. For a bandit whose values change, replace $1/N(A)$ by a constant step size $\alp$ ([[step-size]]).

### ε-greedy beyond bandits {#beyond}

With states, the same rule is applied in each state to the action values $\val{Q(s, \cdot)}$: with probability $\eps$ a random action, otherwise a greedy one. It is the default exploration of [[sarsa]], [[q-learning]], [[mc-control]] and [[dqn]]. Two properties carry over from the bandit. ε-greedy is **ε-soft**, giving every action a probability of at least $\eps / |\mathcal{A}(s)|$, which on-policy methods need to keep exploring ([[mc-control]]). And it explores **blindly**: the worst action is tried as often as the second best, and a long sequence of lucky random choices is needed to reach a distant part of a large world ([[explore-exploit]]). Methods that direct exploration toward uncertainty, such as [[ucb]], address the first problem; the second remains an open research question ([[exploration-strategies]]).

### Historical remarks {#history}

ε-greedy is folklore: a rule simple enough to have been used long before it had a standard name. The comparison on the 10-armed testbed is from Sutton and Barto (§2.3). Auer, Cesa-Bianchi and Fischer (2002) proved logarithmic regret for a version with decaying $\eps$, and empirical comparisons have repeatedly found well-tuned ε-greedy hard to beat on simple problems (Vermorel & Mohri, 2005). With an annealed $\eps$ it was the exploration strategy of DQN (Mnih et al., 2015).

## Card

### Idea

Exploit almost always, explore a little. With probability $1 - \eps$ pull the arm with the highest estimate; with probability $\eps$ pull a random one. The random pulls make sure no arm is written off forever because of one unlucky result.

::: analogy
Ordering at your favorite restaurant, except that one night in ten you roll a die and try something else on the menu. Most nights are good, and you never stop learning the menu.
:::

### The rule {#update}

$$A \leftarrow \begin{cases} \operatorname*{arg\,max}_a \val{Q(a)} & \text{with probability } 1 - \eps \\ \text{a random arm} & \text{with probability } \eps \end{cases} \qquad \val{Q(A)} \leftarrow \val{Q(A)} + \tfrac{1}{N(A)}\big[\rew{R} - \val{Q(A)}\big]$$

The estimate is the arm's average reward, updated incrementally; for changing problems use a constant $\alp$ instead of $1/N(A)$.

### One change from greedy {#change}

| | chooses |
| --- | --- |
| greedy | always $\operatorname*{arg\,max}_a \val{Q(a)}$: a low estimate is never revisited |
| ε-greedy | a random arm with probability $\eps$: every arm keeps being tried |

### Backup diagram {#backup}

{{backup bandit}}

From the arm pulled (dot) to its reward. Nothing follows a pull, so the target is the reward itself: the shortest backup there is.

### Pseudocode

::: pseudocode
Parameters: exploration rate $\eps$; step size $\alp$, or $1/N(A)$ for sample averages
Set $\val{Q(a)} = 0$ and $N(a) = 0$ for every arm $a$ {#init}
Repeat:
  With probability $1 - \eps$ choose $A$ with the highest $\val{Q}$ (ties at random), otherwise a random arm {#choose}
  Pull $A$, observe $\rew{R}$ {#act}
  $N(A) \leftarrow N(A) + 1$ and $\val{Q(A)} \leftarrow \val{Q(A)} + \tfrac{1}{N(A)}\,[\rew{R} - \val{Q(A)}]$ {#update}
:::

### Perks

- One line of logic, no assumptions about the rewards. It works with any estimate and in any world.
- Every arm is tried infinitely often, so every estimate converges to the truth. [See it](lab:bandit-epsilon)
- Hard to beat on simple problems when $\eps$ is well tuned.

### Flaws

- Explores blindly: the worst arm is tried as often as the second best.
- With a constant $\eps$ it never stops paying for exploration: at most $1 - \eps + \eps/k$ of its pulls go to the best arm.
- The best $\eps$ depends on the problem (noise, number of arms, horizon) and must be tuned.

### Knobs

| Knob | Too low | Too high |
| --- | --- | --- |
| $\eps$ exploration | gets stuck on the first arm that looks good | wastes many pulls on arms known to be bad |
| $\alp$ step size (if not averaging) | slow to learn | estimates follow the latest lucky rewards |

### Pitfalls

- Always breaking ties toward the first arm: early on, when all estimates are equal, the “random” behavior then always starts with arm 1. Break ties at random.
- Judging what was learned from the ε-greedy behavior: the greedy policy is better than the behavior suggests.
- Forgetting that the random pull can pick the greedy arm too: the greedy arm's probability is $1 - \eps + \eps/k$, not $1 - \eps$.

### Check yourself {#check}

::: question
With $k = 10$ and $\eps = 0.1$, what is the largest fraction of pulls that can go to the best arm?
---
$1 - 0.1 + 0.1/10 = 0.91$: the greedy choice, plus the random choices that happen to land on it.
:::

::: question
Why can a greedy player stay stuck on a worse arm forever?
---
Estimates only change when their arm is pulled. An arm that looks worse is never pulled, so its estimate, possibly too low, is never corrected.
:::

::: question
On the testbed, ε = 0.1 finds the best arm faster than ε = 0.01. Why does ε = 0.01 do better in the long run?
---
Once both have found the best arm, ε = 0.01 wastes 1% of its pulls on random arms and ε = 0.1 wastes 10%. Its ceiling is 99.1% best-arm pulls against 91%.
:::
