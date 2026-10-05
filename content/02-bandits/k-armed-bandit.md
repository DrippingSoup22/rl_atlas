+++
summary = "One situation, k choices, each paying a random reward with an unknown average: collect as much as you can. The simplest problem where an agent must explore."
prereqs = ["explore-exploit", "state-action-reward"]
lab = "bandit-epsilon"
sources = [
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., §2.1–2.3, §2.9", url = "http://incompleteideas.net/book/the-book-2nd.html" },
  { text = "Thompson (1933), On the likelihood that one unknown probability exceeds another in view of the evidence of two samples, Biometrika 25" },
  { text = "Robbins (1952), Some aspects of the sequential design of experiments, Bulletin of the American Mathematical Society 58", url = "https://doi.org/10.1090/S0002-9904-1952-09620-8" },
  { text = "Gittins (1979), Bandit processes and dynamic allocation indices, Journal of the Royal Statistical Society B 41" },
  { text = "Lai & Robbins (1985), Asymptotically efficient adaptive allocation rules, Advances in Applied Mathematics 6", url = "https://doi.org/10.1016/0196-8858(85)90002-8" },
  { text = "Lattimore & Szepesvári (2020), Bandit Algorithms, Cambridge University Press", url = "https://tor-lattimore.com/downloads/book/book.pdf" },
]

[story]
scene = "bandit"
env = "testbed"
seed = 58
units = 1000
average = 200
formula = '\step{1}{\val{q_*(a)} = \mathbb{E}\big[\rew{R_t} \mid A_t = a\big]} \step{2}{\qquad \val{Q_t(a)} = \frac{\text{sum of the rewards } a \text{ has paid}}{\text{number of times } a \text{ was pulled}}}'
numbers = '\val{Q(7)} = \tfrac{1}{9}\big(\rew{1.29 - 0.86 + 0.70 + 1.01 + 0.40 + 0.68 + 0.53 - 0.62 - 1.41}\big) = \val{0.19}'

[story.runs]
learner = { algorithm = "epsilon-greedy", epsilon = 0.1, alpha = 0.0, name = "a learner that explores 1 pull in 10" }
+++

## Story

::: step {at = 0, truth = false, learner = false}
**Ten slot machines.** Each pull of a lever pays a random amount, sometimes more, sometimes less. Some machines pay more *on average* than others, but nothing tells you which. You have 1000 pulls. How do you collect as much as possible?
:::

::: step {at = 0, truth = true, learner = false, formula = 1}
Here is what the player never gets to see. Each machine $a$ pays from its own bell curve around a mean $\val{q_*(a)}$, the **value** of that arm: the reward it pays on average. Arm 2 is the best, at $\rew{+1.57}$. Arm 3 is the worst, at $\rew{-1.99}$.
:::

::: step {at = 0, truth = false, play = 12, pace = 600, formula = 2}
The player can only learn the values by pulling. So it keeps an **estimate** $\val{Q(a)}$ per arm: the average of what that arm has paid so far, drawn as a bar. Each reward (a dot) moves its own arm's bar. Untried arms stay at 0.
:::

::: step {at = 12, truth = false, numbers = true, focus = 7, formula = 2}
After 12 pulls, arm 7 has been pulled 9 times and averages $\val{0.19}$. Arms 3 and 6 paid badly, and seven arms have never been tried. Now comes the dilemma. Pull arm 7 again, the best-looking arm (**exploit**)? Or try an untried arm that might be better, or might be worse (**explore**)?
:::

::: step {at = 1000, truth = true}
This player mostly exploits, but one pull in ten goes to a random arm. After 1000 pulls its estimates of the arms it tried often are close to the true means, and it has found arm 2: 878 of its 1000 pulls went there (the counts are under the machines).
:::

::: step {at = 1000, truth = true, curves = ["learner"], metric = "optimal"}
One run proves little: these machines might just be easy. To judge a method, run it on many different problems, each with new means drawn at random, and average. This is the **10-armed testbed**. The chart averages 200 problems: how often the player pulls the best arm, step by step.
:::

::: step {at = 1000, truth = true, curves = ["learner"], metric = "return"}
And the reward per step. If you knew the true means, you would pull the best arm every time and earn about $\rew{1.54}$ per step on average over all testbed problems. Every bandit method is a way of closing that gap faster: [[epsilon-greedy]], [[optimistic-init]], [[ucb]] and [[gradient-bandit]].
:::

## Textbook

### The problem {#problem}

The bandit problem is reinforcement learning with the states taken away. There is a single situation, which repeats forever. At each time step $t = 1, 2, \ldots$ the agent chooses one of $k$ actions, $A_t \in \{1, \ldots, k\}$, and receives a reward $\rew{R_t}$ drawn from a probability distribution that depends only on the action chosen. The name comes from slot machines, nicknamed *one-armed bandits*: a $k$-armed bandit is a machine with $k$ levers, each paying out at random according to its own odds.

::: definition {#def-bandit} The $k$-armed bandit
A $k$-armed bandit is a set of $k$ reward distributions, one per action. At each step $t$ the agent chooses an action $A_t$ and receives a reward $\rew{R_t}$ drawn from the distribution of $A_t$, independently of all earlier steps. The agent does not know the distributions. Its goal is to maximize the expected total reward $\mathbb{E}\big[\sum_{t=1}^{T} \rew{R_t}\big]$ over some number of steps $T$.
:::

Two features set the problem apart from supervised learning. The feedback is **evaluative**: the reward says how good the chosen action was, not which action would have been best, and nothing at all about the actions not chosen. And the agent itself decides which data it gets: an action never chosen is never observed.

### Action values {#values}

The **value** of an action is the reward it pays on average:

$$\val{q_*(a)} = \mathbb{E}\big[\,\rew{R_t} \mid A_t = a\,\big]. \label{value}$$

If the values were known, the problem would be trivial: always choose an action with the largest value. They are not, so the agent keeps estimates. Write $\val{Q_t(a)}$ for the estimate of $\val{q_*(a)}$ before step $t$, and $N_t(a)$ for the number of times $a$ was chosen before step $t$. The most natural estimate is the **sample average** of the rewards that $a$ has paid:

$$\val{Q_t(a)} = \frac{\sum_{i=1}^{t-1} \rew{R_i}\, \mathbb{1}_{A_i = a}}{\sum_{i=1}^{t-1} \mathbb{1}_{A_i = a}} = \frac{\text{sum of the rewards when } a \text{ was taken before } t}{N_t(a)}, \label{sample-average}$$

where $\mathbb{1}_{A_i = a}$ is 1 if $A_i = a$ and 0 otherwise. If $N_t(a) = 0$ the estimate is set to a default value, such as 0. By the law of large numbers, $\val{Q_t(a)} \to \val{q_*(a)}$ as $N_t(a) \to \infty$. Computing the average needs neither the list of past rewards nor a growing amount of work: [[incremental-mean]] updates it in constant time and memory, and [[step-size]] generalizes the update.

### Exploiting and exploring {#dilemma}

At each step, at least one action has the largest estimate. Choosing one of these **greedy** actions,

$$A_t = \operatorname*{arg\,max}_a \val{Q_t(a)}, \label{greedy}$$

is called **exploiting**: it uses what the agent knows to maximize the next reward. Choosing any other action is **exploring**: it gives up some expected reward now to improve the estimate of that action. Exploiting is right for the next step, but exploring may be better in the long run, because an action whose estimate is low may only be unlucky ([[explore-exploit]]).

Whether to explore at a given step depends on the estimates, on how uncertain they are, and on how many steps remain: with many steps left, information is worth more. Optimal solutions exist for special cases. With a prior distribution over the values and discounted rewards, the **Gittins index** gives each arm a number computed from its own history alone, and choosing the arm with the largest index is optimal (Gittins, 1979). But such solutions rest on assumptions, a known prior and an unchanging problem, that rarely hold in the full reinforcement learning problem. The methods of this part make no such assumptions. They are simple rules that balance the two reasonably well, and they carry over to problems with states.

### Measuring performance: regret {#regret}

Let $\val{q_*} = \max_a \val{q_*(a)}$ be the best value and $\Delta_a = \val{q_*} - \val{q_*(a)}$ the **gap** of action $a$: how much is lost, on average, each time $a$ is chosen instead of a best action. The **regret** after $T$ steps is the reward lost by not always choosing a best action:

$$L_T = T\,\val{q_*} - \mathbb{E}\Big[\sum_{t=1}^{T} \rew{R_t}\Big]. \label{regret-def}$$

::: lemma {#lem-regret} Regret decomposition
$L_T = \sum_a \Delta_a\, \mathbb{E}[N_{T+1}(a)]$, where $N_{T+1}(a)$ is the number of times $a$ was chosen in the first $T$ steps.
:::

::: proof
The reward at step $t$ has expectation $\val{q_*(A_t)}$ given $A_t$, so $\mathbb{E}\big[\sum_t \rew{R_t}\big] = \mathbb{E}\big[\sum_t \val{q_*(A_t)}\big] = \sum_a \val{q_*(a)}\, \mathbb{E}[N_{T+1}(a)]$. Since $\sum_a N_{T+1}(a) = T$, we have $T\,\val{q_*} = \sum_a \val{q_*}\, \mathbb{E}[N_{T+1}(a)]$. Subtracting gives the claim.
:::

So regret grows only through pulls of worse arms, each weighted by its gap. An agent that keeps choosing every arm a fixed fraction of the time, as ε-greedy with a constant $\eps$ does, has regret that grows linearly in $T$. Lai and Robbins (1985) showed that no reasonable method can do better than logarithmic growth: every worse arm must be tried at least on the order of $\ln T$ times, and more often the harder it is to tell apart from the best one. Methods that achieve this rate explore by uncertainty rather than at random ([[ucb]]).

### The 10-armed testbed {#testbed}

To compare methods, Sutton and Barto use a suite of randomly generated problems, which the atlas uses too.

::: example {#ex-testbed} The 10-armed testbed (Sutton & Barto, §2.3)
A problem has $k = 10$ arms. Each true value $\val{q_*(a)}$ is drawn once from a normal distribution with mean 0 and variance 1. When arm $a$ is chosen, the reward is drawn from a normal distribution with mean $\val{q_*(a)}$ and variance 1. A *run* applies a method to one problem for 1000 steps. Its performance is measured by the reward at each step and by whether the step chose a best arm, both averaged over many runs, each on a new problem.
:::

::: figure {#fig-testbed}
{{testbed}}
The reward distributions of one testbed problem, the one the stories of this part play on. Arm $a$ pays rewards from a normal distribution around its value $\val{q_*(a)}$ (the line); the values themselves were drawn from a standard normal distribution. Here arm 2 is the best, with $\val{q_*(2)} = 1.57$. After Sutton & Barto, Figure 2.1.
:::

Averaging over problems matters. On a single problem, a method may do well because the best arm happened to pay well early, and a comparison would mostly measure luck. Over many problems, the luck averages out. A useful reference point: a player who knew the values would always choose the best arm, and the expected maximum of ten independent standard normal values is about 1.54. That is the best average reward per step any method can reach on the testbed.

### Beyond the bandit {#beyond}

The bandit is the full problem stripped to one feature: it has evaluative feedback, but a single situation, and actions that affect only the immediate reward. Adding features back one at a time leads to the rest of the atlas.

- **Associative search.** Suppose there are several bandits, and at each step the agent is told which one it faces, by a signal such as the color of the machine. Now it must learn a policy: the best action *for each situation*. This is a **contextual bandit**, the setting of many recommendation systems. Each action still affects only the immediate reward.
- **Full reinforcement learning.** If actions also affect the next situation, and through it later rewards, the problem becomes a Markov decision process ([[mdp]]). A bandit is the special case with one state, where every action leads back to it ([[agent-environment]]).

Everything learned here carries over: the incremental estimates, the role of the step size, and the strategies for exploration reappear in every method of the atlas, in the form *new estimate = old estimate + step size × (target − old estimate)*.

### Historical remarks {#history}

Thompson (1933) studied the two-armed case as a question about clinical trials: which of two treatments to give the next patient, when the evidence so far is limited. Robbins (1952) formulated the general problem of choosing between experiments sequentially, and the name *bandit* soon followed. Bellman and others studied it with dynamic programming in the 1950s, and Gittins (1979) found the index that solves the discounted Bayesian version. Lai and Robbins (1985) proved the logarithmic lower bound on regret. Sutton and Barto's testbed, used throughout this part, appears in their chapter 2. Lattimore and Szepesvári (2020) give a modern account of the theory.

## Card

### Idea

You choose again and again among $k$ actions. Each pays a random reward with its own unknown average $\val{q_*(a)}$. To earn a lot you must pull the arm that looks best, and to find which arm is best you must try the others. A bandit is reinforcement learning with a single state.

::: analogy
A row of slot machines in a casino, each with its own hidden odds, and a fixed budget of coins. Every coin spent testing a machine is a coin not spent on the best one you know.
:::

### In symbols {#formula}

$$\val{q_*(a)} = \mathbb{E}\big[\rew{R_t} \mid A_t = a\big] \qquad \val{Q_t(a)} = \frac{\text{sum of the rewards from } a \text{ before } t}{N_t(a)} \qquad A_t^{\text{greedy}} = \operatorname*{arg\,max}_a \val{Q_t(a)}$$

Regret after $T$ steps: $L_T = T\,\val{q_*} - \mathbb{E}\big[\sum_{t \le T} \rew{R_t}\big] = \sum_a \Delta_a\,\mathbb{E}[N(a)]$.

### The testbed {#testbed}

| | |
| --- | --- |
| Arms | $k = 10$ |
| True values | $\val{q_*(a)}$ drawn from a normal distribution, mean 0, variance 1 |
| Rewards | drawn from a normal distribution, mean $\val{q_*(a)}$, variance 1 |
| A run | 1000 steps on one problem; results averaged over many problems |
| Best possible | about $\rew{1.54}$ per step on average |

### Why it matters {#why}

It isolates exploration. Every idea for balancing exploring and exploiting is first tried on bandits, and the estimates and step sizes introduced here are used by every later method.

### Pitfalls

- Judging a method on one problem: average over many, or you measure luck.
- Forgetting that untried arms have no information: an estimate of 0 for an arm never pulled is a default, not a measurement.
- Confusing the value $\val{q_*(a)}$, a fixed property of the arm, with the estimate $\val{Q_t(a)}$, which changes with every pull.

### Check yourself {#check}

::: question
Why can't a bandit agent simply compute which arm is best?
---
The values are averages of random rewards that it never sees directly. It can only estimate them by pulling the arms, and every pull spent on estimating is a pull not spent on the best arm.
:::

::: question
On the testbed, why average over many problems instead of running a method once?
---
On one problem a method can look good or bad by luck, for example when the best arm happens to pay well early. Averaging over many problems measures the method, not the luck.
:::

::: question
Which ingredient of full reinforcement learning is missing from a bandit?
---
States: actions do not change the situation, so they affect only the immediate reward. Nothing has delayed consequences.
:::
