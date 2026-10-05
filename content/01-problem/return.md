+++
summary = "Everything the agent collects from now on, added up: the quantity it really tries to make large."
prereqs = ["agent-environment", "episodes"]
sources = [
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., §3.3, §3.4 and Example 3.4", url = "http://incompleteideas.net/book/the-book-2nd.html" },
]

[story]
scene = "timeline"
formula = '\begin{aligned} \step{1}{\rew{G_t}} &\step{1}{= \rew{R_{t+1}} + \rew{R_{t+2}} + \cdots + \rew{R_T}} \\ \step{2}{\rew{G_t}} &\step{2}{= \rew{R_{t+1}} + \gam\,\rew{R_{t+2}} + \gam^2\,\rew{R_{t+3}} + \cdots} \\ \step{3}{\rew{G_t}} &\step{3}{= \rew{R_{t+1}} + \gam\,\rew{G_{t+1}}} \end{aligned}'

[story.sequences]
episode = [-1, -1, 0, -1, 2, -1, 0, -1, 5, 10]
another = [-1, 2, -1, -1, -1, 0, -1, -1, -1, 10]
forever = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
+++

## Story

::: step {seq = "episode", count = 1}
After time $t$, the agent gets a reward after every action: first $\rew{R_{t+1}}$, then $\rew{R_{t+2}}$, and so on. Here the first one is $\rew{-1}$.
:::

::: step {seq = "episode"}
Here is the whole stream until this episode ends, ten steps later, at time $T$. Some steps cost, a few pay, and the best reward comes last.
:::

::: step {seq = "episode", sum = true, formula = 1}
The **return** $\rew{G_t}$ is everything collected from $t$ on, added up. For this stream it is $\rew{12}$. This, and not the next reward, is what the agent tries to make large.
:::

::: step {seq = "another", sum = true, formula = 1}
The return is **random**. Starting from the same place, the same agent can meet a different stream: this time the return is $\rew{5}$. That is why the agent aims at the *expected* return, its average over all the ways things can go.
:::

::: step {seq = "forever", sum = true, infinite = true, formula = 1}
In a task that never ends, the stream goes on forever. A reward of $\rew{+1}$ at every step adds up to infinity, and when every way of acting is worth infinity, nothing can be compared.
:::

::: step {seq = "forever", sum = true, infinite = true, gamma = 0.9, weights = true, discount = true, formula = 2}
**Discounting** fixes this. A reward $k$ steps after the next one is multiplied by $\gam^k$, with $\gam$ below 1, so later rewards count less. With $\gam = 0.9$ the endless $\rew{+1}$s add up to $1/(1-\gam) = 10$.
:::

::: step {seq = "episode", sum = true, gamma = 0.9, weights = true, discount = true, formula = 2}
Back to the first episode with $\gam = 0.9$. The $\rew{+10}$ at the end now counts only $10 \times 0.9^9 \approx 3.9$, and the return is about $\rew{3.64}$.
:::

::: step {seq = "episode", sum = true, gamma = 0.9, weights = true, discount = true, recursion = true, formula = 3}
One last look. Everything after the first reward is itself a return, one step later and discounted once: $\rew{G_t} = \rew{R_{t+1}} + \gam\,\rew{G_{t+1}}$. This small identity is the seed of [[bellman]] and of every method that learns from one step at a time.
:::

## Textbook

### The return {#definition}

The agent's goal, informally, is to collect as much reward as possible over time. The **return** makes "over time" precise. In an episodic task that ends at time $T$ ([[episodes]]), the return from time $t$ is the sum of the rewards that follow:

$$\rew{G_t} = \rew{R_{t+1}} + \rew{R_{t+2}} + \rew{R_{t+3}} + \cdots + \rew{R_T}. \label{episodic}$$

The return is defined for every time step, not just the first: $\rew{G_t}$ is what remains to be collected from $t$ on. The agent's objective is to choose actions so as to maximize the **expected** return, for reasons explained in \ref{random}.

### Discounting {#discounting}

In a continuing task, $T = \infty$ and the sum in \ref{episodic} may diverge. The standard remedy is to discount: a reward received $k$ steps after the next one is multiplied by $\gam^k$, where the **discount rate** $\gam \in [0, 1]$ is a parameter of the problem ([[discount]]). The discounted return is

$$\rew{G_t} = \rew{R_{t+1}} + \gam\,\rew{R_{t+2}} + \gam^2\,\rew{R_{t+3}} + \cdots = \sum_{k=0}^{\infty} \gam^k\,\rew{R_{t+k+1}}. \label{discounted}$$

With $\gam = 0$ the agent is *myopic* and only the next reward counts; as $\gam$ approaches 1, later rewards count almost as much as immediate ones. One formula covers both kinds of task if an episode's end is treated as an absorbing state with zero reward, and either $T = \infty$ or $\gam = 1$ is allowed, but not both ([[episodes]]).

::: lemma {#lem-bound} Bounded returns
If every reward satisfies $|\rew{R_t}| \le R_{\max}$ and $\gam < 1$, then $|\rew{G_t}| \le R_{\max} / (1 - \gam)$ for every $t$.
:::

::: proof
By the triangle inequality and the geometric series,
$$|\rew{G_t}| \le \sum_{k=0}^{\infty} \gam^k\,|\rew{R_{t+k+1}}| \le R_{\max} \sum_{k=0}^{\infty} \gam^k = \frac{R_{\max}}{1 - \gam}.$$
:::

So with $\gam < 1$ and bounded rewards, every return is finite and every policy has a finite value. The bound is attained by a constant reward: $\rew{R} = \rew{+1}$ on every step gives $\rew{G_t} = 1/(1-\gam)$, which is 10 for $\gam = 0.9$ and 100 for $\gam = 0.99$.

### The recursion {#recursion}

Returns at successive time steps are related by a simple identity. Taking the first term out of \ref{discounted},

$$\rew{G_t} = \rew{R_{t+1}} + \gam\big(\rew{R_{t+2}} + \gam\,\rew{R_{t+3}} + \cdots\big) = \rew{R_{t+1}} + \gam\,\rew{G_{t+1}}, \label{recursion-eq}$$

which holds for all $t < T$, with $\rew{G_T} = 0$ at the end of an episode. It says that the return from now is the next reward plus the discounted return from the next step. Taking expectations of \ref{recursion-eq} gives the Bellman equations ([[bellman]]), and replacing $\rew{G_{t+1}}$ by an estimate gives the targets of temporal-difference learning ([[td0]]). Most of the atlas grows out of this line.

The recursion also gives an efficient way to compute every return of a finished episode: start with $\rew{G_T} = 0$ and work backwards, $\rew{G_{T-1}} = \rew{R_T}$, $\rew{G_{T-2}} = \rew{R_{T-1}} + \gam\,\rew{G_{T-1}}$, and so on. Monte Carlo methods do exactly this ([[mc-prediction]]).

### The return is random {#random}

Policies and environments are usually stochastic, so the same state can be followed by many different streams of reward, and $\rew{G_t}$ is a random variable. Two episodes from the same start can give returns of $\rew{12}$ and $\rew{5}$, as in the story. A random quantity cannot be maximized as such; what can be maximized is its expectation. The expected return from a state, under a given way of acting, is that state's **value** ([[value-functions]]).

The randomness of the return also has a practical side. The return depends on every action and every transition until the end of the episode, so its variance grows with the length of the episode. An estimate built from returns alone needs many episodes to settle; methods that bootstrap, through \ref{recursion-eq}, trade some bias for much less variance ([[mc-vs-td]]).

### Examples {#examples}

::: example {#ex-pole} Pole balancing (after Sutton & Barto, Example 3.4)
A cart moves along a track so as to keep a hinged pole from falling over. One formulation is episodic: every step before the pole falls gives $\rew{+1}$, so the return is the number of steps before failure. Another is continuing: the pole is put back up after each failure, a failure gives $\rew{-1}$ and every other step 0, and with discounting the return is $\rew{-\gam^{K}}$, where $K$ is the number of steps before the next failure. In both, the return is largest when the pole stays up as long as possible.
:::

In a maze where every step costs $\rew{-1}$, the return is minus the number of steps to the exit, so maximizing it means finding the shortest path. This is how the cliff world is set up ([[sarsa]]).

## Card

### Idea

The **return** $\rew{G_t}$ is everything the agent collects from time $t$ on, added up, with later rewards discounted by $\gam$. The agent tries to make its **expected** return large, not its next reward.

::: analogy
A salary over a career, not this month's pay. A job that pays less now but leads somewhere can be the better choice.
:::

### In symbols {#formula}

$$\rew{G_t} = \sum_{k=0}^{\infty} \gam^k\,\rew{R_{t+k+1}} \qquad\qquad \rew{G_t} = \rew{R_{t+1}} + \gam\,\rew{G_{t+1}}$$

In an episode, the sum stops at the end, and $\rew{G_T} = 0$.

### Numbers to remember {#numbers}

| Rewards | Return |
| --- | --- |
| $\rew{+1}$ forever, $\gam = 0.9$ | $1/(1-\gam) = 10$ |
| $\rew{+1}$ forever, $\gam = 0.99$ | $100$ |
| $\rew{-1}$ a step until the exit, $\gam = 1$ | minus the number of steps |

### Why it matters {#why}

Values are expected returns ([[value-functions]]), and the recursion $\rew{G_t} = \rew{R_{t+1}} + \gam\,\rew{G_{t+1}}$ is the starting point of the [[bellman]] equations and of temporal-difference learning.

### Pitfalls

- Maximizing the next reward instead of the return: the agent never invests in anything.
- Forgetting that the return is random: one lucky episode says little about a state's value.
- Summing undiscounted rewards in a task that never ends.

### Check yourself {#check}

::: question
Rewards $\rew{1}, \rew{2}, \rew{3}$ and then the episode ends. With $\gam = 0.5$, what is $\rew{G_0}$?
---
$1 + 0.5 \times 2 + 0.25 \times 3 = 2.75$. Or backwards: $\rew{G_2} = 3$, $\rew{G_1} = 2 + 0.5 \times 3 = 3.5$, $\rew{G_0} = 1 + 0.5 \times 3.5 = 2.75$.
:::

::: question
Why does the agent maximize the *expected* return?
---
The return is random: the same state can be followed by many different streams of reward. Only its average can be compared between ways of acting.
:::

::: question
With rewards between $-1$ and $1$ and $\gam = 0.9$, how large can a return be?
---
At most $1/(1-0.9) = 10$ in size, by the geometric series.
:::
