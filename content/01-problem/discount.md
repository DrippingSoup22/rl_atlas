+++
summary = "One number between 0 and 1 that says how much a reward is worth if it comes one step later: how far ahead the agent looks."
prereqs = ["return"]
sources = [
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., §3.3, §10.3 and §10.4", url = "http://incompleteideas.net/book/the-book-2nd.html" },
  { text = "Puterman (1994), Markov Decision Processes: Discrete Stochastic Dynamic Programming, Wiley, §5.3 (discounting as a random horizon)" },
  { text = "Jiang, Kulesza, Singh & Lewis (2015), The dependence of effective planning horizon on model accuracy, Proceedings of the 14th International Conference on Autonomous Agents and Multiagent Systems" },
  { text = "Ainslie (1975), Specious reward: a behavioral theory of impulsiveness and impulse control, Psychological Bulletin 82", url = "https://doi.org/10.1037/h0076860" },
]

[story]
scene = "timeline"
formula = '\step{1}{\rew{G_t} = \rew{R_{t+1}} + \gam\,\rew{R_{t+2}} + \gam^2\,\rew{R_{t+3}} + \cdots = \sum_{k=0}^{\infty} \gam^k\,\rew{R_{t+k+1}}}'

[story.sequences]
steady = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
now = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
later = [0, 0, 0, 0, 0, 0, 0, 0, 0, 5]
+++

## Story

::: step {seq = "steady", gamma = 0.9, weights = true}
The **discount factor** $\gam$, a number between 0 and 1, says how much a reward is worth if it arrives one step later. A reward $k$ steps after the next one is multiplied by $\gam^k$. With $\gam = 0.9$ the weights shrink slowly: 1, 0.9, 0.81, …
:::

::: step {seq = "steady", gamma = 0.9, weights = true, discount = true, sum = true, formula = 1}
Each reward shrinks by its weight before it is added to the return. Ten rewards of $\rew{+1}$ are worth $\rew{6.51}$ seen from today.
:::

::: step {seq = "steady", gamma = 0.5, weights = true, discount = true, sum = true, formula = 1}
With $\gam = 0.5$ the weights halve at every step. The agent is short-sighted: the same ten rewards are worth only $\rew{2.00}$.
:::

::: step {seq = "steady", gamma = 0, weights = true, discount = true, sum = true, formula = 1}
With $\gam = 0$ only the next reward counts. This agent is **myopic**: it grabs whatever pays immediately and ignores the consequences.
:::

::: step {seq = "steady", gamma = 0.99, weights = true, discount = true, sum = true, formula = 1}
With $\gam = 0.99$ the weights hardly shrink: the agent is far-sighted. A reward 100 steps away still counts $0.99^{100} \approx 0.37$ of its size.
:::

::: step {seq = ["now", "later"], gamma = 0.9, weights = true, discount = true, sum = true}
$\gam$ changes decisions, not only numbers. Take $\rew{+1}$ now, or $\rew{+5}$ nine steps later? With $\gam = 0.9$ the later $\rew{+5}$ is worth $5 \times 0.9^9 \approx 1.94$ today, so waiting is better.
:::

::: step {seq = ["now", "later"], gamma = 0.5, weights = true, discount = true, sum = true}
With $\gam = 0.5$ the same $\rew{+5}$ is worth $5 \times 0.5^9 \approx 0.01$ today: take the $\rew{+1}$ now. Same world, different $\gam$, different best behavior.
:::

::: step {seq = "steady", gamma = 0.9, weights = true, discount = true, sum = true, infinite = true, formula = 1}
In a task that never ends, a $\gam$ below 1 keeps the total finite: endless $\rew{+1}$s add up to $1/(1-\gam) = 10$. That number is the agent's **effective horizon**, roughly how many steps ahead it looks. Try other values with the slider in the Card.
:::

## Textbook

### Definition {#definition}

The **discount rate** $\gam \in [0, 1]$ weighs future rewards in the return ([[return]]):

$$\rew{G_t} = \sum_{k=0}^{\infty} \gam^k\,\rew{R_{t+k+1}}. \label{return}$$

A reward received $k$ steps after the next one is worth $\gam^k$ times what it would be worth if it came immediately. In this sense $\gam$ is an exchange rate between the present and the future. It is part of the definition of the problem, chosen by the designer, not something the agent learns.

### Why discount {#why}

Three reasons are usually given, and they are of different kinds.

- **Mathematical.** In a continuing task, undiscounted sums of rewards can diverge. With $\gam < 1$ and bounded rewards every return is bounded by $R_{\max}/(1-\gam)$, so every policy has a finite value and policies can be compared ([[return]]). The operators of dynamic programming are then contractions with modulus $\gam$, which makes the theory and the algorithms well behaved ([[q-learning]]).
- **Uncertainty about the future.** Suppose the interaction may end at any step with probability $1 - \gam$, independently of everything else, and that nothing is collected afterwards. Then the probability of still collecting the reward $k$ steps after the next one is $\gam^k$, and the expected *undiscounted* total equals the discounted return of \ref{return} (Puterman, 1994). Discounting is equivalent to an uncertain horizon.
- **Preference for sooner rewards.** In economics, money now can be invested; in many tasks a reward now is simply preferred, as a battery charge now prevents a failure later.

The first reason makes $\gam$ convenient, the second gives it a meaning, and the third makes it a part of the goal. When the true goal is the undiscounted total, as in most episodic tasks, $\gam$ is best seen as a parameter of the method rather than of the problem.

### The effective horizon {#horizon}

The weights $\gam^k$ add up to

$$\sum_{k=0}^{\infty} \gam^k = \frac{1}{1-\gam}, \label{weights-sum}$$

so a discounted return behaves roughly like an undiscounted sum over the next $1/(1-\gam)$ steps. This **effective horizon** is 10 steps for $\gam = 0.9$, 100 for $\gam = 0.99$ and 1000 for $\gam = 0.999$. A related number is the half-life of the weights, the $k$ at which $\gam^k = 1/2$: $k = \ln 2 / \ln(1/\gam)$, about 6.6 steps for $\gam = 0.9$ and 69 for $\gam = 0.99$.

The horizon tells how far ahead the agent's decisions take consequences into account. A reward far beyond the horizon has almost no influence on what the agent does now.

::: figure {#fig-weights}
{{discount}}
The weight $\gam^k$ of a reward $k$ steps away, and the effective horizon $1/(1-\gam)$. Move the slider: the comparison at the bottom flips from waiting to grabbing when $\gam$ falls below $0.1^{1/20} \approx 0.891$.
:::

### γ changes the optimal policy {#policy}

Different discount rates can make different policies optimal. Compare two choices: $\rew{+1}$ immediately, or $\rew{+10}$ after twenty steps with nothing in between. Seen from now they are worth $1$ and $10\,\gam^{20}$. Waiting is better exactly when

$$10\,\gam^{20} > 1 \iff \gam > 0.1^{1/20} \approx 0.891.$$

So an agent with $\gam = 0.9$ waits, and an agent with $\gam = 0.88$ does not, although the world is the same. Choosing $\gam$ is therefore part of specifying the task. A low $\gam$ makes the agent impatient, and in tasks where reward comes late it may never discover the behavior the designer intended.

### γ as a knob in practice {#practice}

In practice $\gam$ also acts as a knob of the learning method. Larger values make the agent far-sighted but make learning harder: the returns are longer sums, so their variance grows, and information has to travel back over more steps ([[mc-vs-td]]). Values between 0.9 and 0.999 are common, 0.99 being a frequent default in deep reinforcement learning. In episodic tasks whose goal is the undiscounted total, a $\gam$ slightly below 1 is often used anyway, because it makes learning more stable. A smaller $\gam$ can even give better behavior than the true one when experience is scarce: a short horizon asks the agent to predict less of the future, so the errors of an estimated model or of noisy values compound over fewer steps (Jiang et al., 2015). Discounting then acts as a form of regularization.

In continuing tasks there is an alternative to discounting: the **average reward per step**, $\lim_{h \to \infty} \frac{1}{h}\,\mathbb{E}\big[\sum_{t=1}^{h} \rew{R_t}\big]$. Sutton and Barto argue that with function approximation the discounted formulation becomes problematic for continuing tasks and that the average-reward setting is more appropriate there (§10.3 and §10.4). The atlas uses discounting throughout, as most of the literature does.

### Why geometric {#geometric}

The weights $\gam^k$ have a property that other ways of discounting lack: they are **time-consistent**. The ratio between the weights of two future rewards depends only on how far apart they are, so a choice that looks best today still looks best tomorrow, when both rewards are one step closer. People and animals do not discount this way. Their preferences are closer to hyperbolic, $1/(1 + \kappa k)$, which falls steeply at first and slowly later, so their choices reverse as a reward approaches: the larger, later reward preferred from afar loses to the smaller, sooner one when it is near (Ainslie, 1975). A geometric discount never changes its mind, which is one reason the theory of this atlas holds together: an optimal policy stays optimal at every step.

## Card

### Idea

$\gam$ says how much a reward one step later is worth. A reward $k$ steps after the next one counts $\gam^k$. Small $\gam$: the agent is impatient and short-sighted. $\gam$ close to 1: it plans far ahead. In a task that never ends, $\gam < 1$ keeps the total finite.

::: analogy
Interest rates in reverse. Money promised in ten years is worth less than money today; how much less depends on the rate.
:::

### Try it {#demo}

{{discount}}

### In symbols {#formula}

$$\rew{G_t} = \sum_{k=0}^{\infty} \gam^k\,\rew{R_{t+k+1}}, \qquad \text{horizon} \approx \frac{1}{1-\gam}$$

### Why it matters {#why}

It decides how far ahead consequences count, and so what the best behavior is. It also appears in every update in the atlas, from [[td0]] to [[ppo]].

### Pitfalls

- A $\gam$ too small for the task: rewards that come late never influence the policy, and the agent never learns to reach them.
- A $\gam$ of 1 in a task that never ends: returns can be infinite.
- Comparing values computed with different $\gam$: they are on different scales.

### Knobs

| $\gam$ | Horizon | Behavior |
| --- | --- | --- |
| 0 | 1 step | grabs the next reward, ignores consequences |
| 0.9 | about 10 steps | plans a little ahead |
| 0.99 | about 100 steps | plans far ahead; learning is slower and noisier |

### Check yourself {#check}

::: question
With $\gam = 0.95$, what is the effective horizon?
---
$1/(1-0.95) = 20$ steps.
:::

::: question
Which is better, $\rew{+2}$ now or $\rew{+3}$ in one step, with $\gam = 0.6$? And with $\gam = 0.7$?
---
With $\gam = 0.6$: $3 \times 0.6 = 1.8 < 2$, take the $\rew{+2}$. With $\gam = 0.7$: $3 \times 0.7 = 2.1 > 2$, wait.
:::

::: question
How can discounting be read as a chance that the world ends?
---
If the interaction stops with probability $1-\gam$ at each step, a reward $k$ steps away is collected with probability $\gam^k$: exactly its discount weight.
:::
