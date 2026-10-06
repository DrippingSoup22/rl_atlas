+++
summary = "Instead of learning how good each action is and then choosing, learn the way of choosing itself: a policy with adjustable weights. Then the best policy can be random, actions can be real numbers, and a small change of the weights is a small change of behavior."
prereqs = ["policy", "value-functions", "epsilon-greedy", "features"]
lab = "corridor-reinforce"
sources = [
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., §13.1 and Example 13.1", url = "http://incompleteideas.net/book/the-book-2nd.html" },
  { text = "Singh, Jaakkola & Jordan (1994), Learning without state-estimation in partially observable Markovian decision processes, Proceedings of the 11th International Conference on Machine Learning" },
  { text = "Williams (1992), Simple statistical gradient-following algorithms for connectionist reinforcement learning, Machine Learning 8", url = "https://doi.org/10.1007/BF00992696" },
  { text = "Sutton, McAllester, Singh & Mansour (2000), Policy gradient methods for reinforcement learning with function approximation, Advances in Neural Information Processing Systems 12" },
  { text = "Gordon (1996), Chattering in SARSA(λ), CMU Learning Lab Internal Report" },
  { text = "OpenAI (2018), Spinning Up in Deep RL, Part 3: Intro to Policy Optimization", url = "https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html" },
]

[story]
scene = "corridor"
env = "corridor"
seed = 3
average = 100
formula = '\step{1}{\pol{\pi(\text{right})} = p \qquad} \step{2}{J(p) = -\frac{2\,(2 - p)}{p\,(1 - p)}}'

[story.runs]
stuck = { algorithm = "fixed-policy", right0 = 0.999999, gamma = 1.0, features = "own", maxSteps = 40, units = 3, name = "always right" }
right = { algorithm = "fixed-policy", right0 = 0.95, gamma = 1.0, features = "own", maxSteps = 1000, units = 3, name = "ε-greedy, preferring right" }
left = { algorithm = "fixed-policy", right0 = 0.05, gamma = 1.0, features = "own", maxSteps = 1000, units = 3, name = "ε-greedy, preferring left" }
best = { algorithm = "fixed-policy", right0 = 0.5858, gamma = 1.0, features = "own", maxSteps = 1000, units = 3, name = "the best random policy" }
learn = { algorithm = "baseline", alpha = 0.001953125, alphaW = 0.015625, right0 = 0.05, gamma = 1.0, features = "own", maxSteps = 1000, units = 300, seed = 2, measures = ["right"], name = "a learner of p" }
+++

## Story

::: step {run = "best", at = 0, p = 0.5}
**A short corridor.** Three cells, then the goal, and every step costs $\rew{-1}$. Stepping left in the first cell bumps into the wall. In the second cell the controls are swapped: right goes left and left goes right. And the catch: the agent cannot tell the cells apart. It sees the same thing in all three, so whatever it does, it does in every cell.
:::

::: step {run = "stuck", at = 0, play = 1, pace = 140}
A learner that estimates the value of each action and takes the best one does the same thing everywhere. Say right is best: from cell 1 to cell 2, where right means left, and back to cell 1, again and again. This episode was cut short after 40 steps; it would never end. Always left is no better: the agent never leaves cell 1.
:::

::: step {run = "right", at = 2, formula = 1}
**ε-greedy** adds a little randomness: right 95% of the time, left 5%. Now the agent escapes, but only when the 5% comes up in cell 2: this episode took 102 steps. On average the policy is worth $\rew{-44.2}$ per episode, the hollow point on the landscape below.
:::

::: step {run = "left", at = 1}
Preferring left with the same 5% of randomness is worse: $\rew{-82.1}$ on average. This episode took 62 steps, most of them spent bumping into the wall.
:::

::: step {run = "best", at = 0, play = 1, pace = 120, formula = 2}
Now forget values, and choose the **chance** of stepping right directly. The landscape below is the value of the start for every chance $p$, worked out exactly. Its top is at $p = 2 - \sqrt 2 \approx 59\%$, worth $\rew{-11.7}$: a coin that lands right a little more often than left, far better than either ε-greedy policy. In this corridor the best policy is random, and no policy that commits to one action per cell can come close.
:::

::: step {run = "learn", checkpoints = [1, 2, 3, 4, 5, 20, 50], hold = 1300}
A learner whose weights *are* the policy can climb this landscape. It starts at 5%, the ε-greedy policy that prefers left, and after every episode nudges its chance in the direction that worked better than expected. Watch the dot: over the first five episodes it moves between 4.6% and 15.8%, most after the fourth, a long one of 269 steps. After 20 episodes it is at 17%, after 50 at 34%.
:::

::: step {run = "learn", at = 100}
After 100 episodes it steps right 65% of the time, worth $\rew{-11.9}$: close to the top. It got there without ever estimating the value of an action.
:::

::: step {run = "learn", at = 300, curves = ["learn"], metric = "right"}
Averaged over 100 runs, the chance of stepping right climbs to 58% within 250 episodes, just under the best 59%; single runs wander above and below it, where the landscape is flat. This learner is REINFORCE with a baseline ([[baseline]]); [watch plain REINFORCE climb in the Lab](lab:corridor-reinforce).
:::

## Textbook

### Two ways to choose {#two-ways}

Everything in the atlas so far learned **values** and derived the policy from them: greedy or ε-greedy with respect to $\val{Q}$ ([[q-learning]], [[semi-gradient-sarsa]]). The policy itself had no parameters of its own; it changed only because the values did. This part takes the other route. The policy gets weights of its own, $\boldsymbol\theta \in \mathbb R^{d'}$, and is written

$$\pol{\pi(a \mid s, \boldsymbol\theta)} = \Pr\{A_t = a \mid S_t = s, \boldsymbol\theta_t = \boldsymbol\theta\}, \label{eq-policy}$$

a probability for each action, differentiable in $\boldsymbol\theta$ ([[policy-parameterization]]). Learning means moving $\boldsymbol\theta$ to make the policy better, measured by a performance $J(\boldsymbol\theta)$, typically the value of the start state. The methods of this part climb $J$ by gradient ascent,

$$\boldsymbol\theta_{t+1} = \boldsymbol\theta_t + \alp\,\widehat{\nabla J(\boldsymbol\theta_t)}, \label{eq-ascent}$$

with a stochastic estimate of the gradient in place of the gradient itself; they are called **policy-gradient methods**. They may still learn values, but only to help estimate the gradient: a learned value function used this way is a *critic*, and the policy an *actor* ([[actor-critic]]). The choice of action never goes through a max over values.

### The best policy can be random {#random}

For a finite Markov decision process there is always an optimal policy that is deterministic ([[policy]]), so value-based methods lose nothing by aiming at one. That guarantee rests on the agent knowing the state. When it does not, because states it cannot tell apart call for different actions, a random policy can be strictly better than every deterministic one.

::: example {#ex-corridor} The short corridor (Sutton & Barto, Example 13.1)
Three non-terminal cells in a row and a goal at the right end; every step pays $\rew{-1}$. The actions move one cell left or right, except that left in the first cell leaves the agent in place, and in the second cell the effects are swapped. The agent's features are the same in all three cells, $\mathbf x(s, \text{right}) = (1, 0)^\top$ and $\mathbf x(s, \text{left}) = (0, 1)^\top$, so any policy it can represent steps right with the same probability $p$ in every cell.
:::

Writing $v_i$ for the value of cell $i$ under such a policy, the [[bellman|Bellman equations]] are three linear equations,

$$\begin{aligned} v_1 &= -1 + p\,v_2 + (1-p)\,v_1, \\ v_2 &= -1 + p\,v_1 + (1-p)\,v_3, \\ v_3 &= -1 + (1-p)\,v_2, \end{aligned} \label{eq-cells}$$

whose solution gives the value of the start

$$J(p) = v_1 = -\frac{2\,(2 - p)}{p\,(1 - p)}. \label{eq-J}$$

$J$ falls to $-\infty$ at both ends: a policy that always steps the same way never reaches the goal. Setting $J'(p) = 0$ gives $p^2 - 4p + 2 = 0$, so the best policy steps right with probability $p_* = 2 - \sqrt 2 \approx 0.586$ and is worth $J(p_*) = -(6 + 4\sqrt 2) \approx -11.66$.

::: figure {#fig-corridor}
{{corridor-values}}
The value of the start of the short corridor for every probability $p$ of stepping right, \ref{eq-J}. The two ε-greedy policies with $\eps = 0.1$ are worth $\rew{-44.2}$ (preferring right, $p = 0.95$) and $\rew{-82.1}$ (preferring left, $p = 0.05$); the best policy, $p \approx 0.59$, is worth $\rew{-11.66}$. After Sutton & Barto, Example 13.1.
:::

A value-based learner with ε-greedy action selection can reach only the two points $p = 1 - \eps/2$ and $p = \eps/2$ (\ref{fig-corridor}): its randomness is a fixed side effect of exploration, not something it can tune. A parameterized policy can put its probability anywhere, and gradient ascent on $J$ finds $p_*$. The same holds in games against an opponent, where being predictable is exploited (the best strategy in rock–paper–scissors is uniform), and in partially observable problems in general: Singh, Jaakkola and Jordan (1994) showed that the best memoryless policy of a partially observable problem may need to be stochastic, and can be arbitrarily better than the best deterministic one.

### Small changes, small effects {#smooth}

The second reason is about learning, not about the solution. With values in charge, the policy is a step function of the values: an arbitrarily small change of $\val{\hat q}$ can flip the greedy action in a state, and with function approximation a single weight update can flip it in many states at once. The data the learner sees then changes abruptly, and its weights may wander without settling (*chattering*; Gordon, 1996), which is one reason why, even with linear features, semi-gradient control comes with few guarantees ([[semi-gradient-sarsa]]).

A softmax or Gaussian policy changes *smoothly* with its weights: a small step of $\boldsymbol\theta$ is a small change of every action probability. That is what makes gradient ascent on $J$ meaningful, and it is why policy-gradient methods have convergence results, to a local optimum of $J$, that hold with function approximation (Sutton, McAllester, Singh & Mansour, 2000; [[pg-theorem]]).

### More reasons, and the costs {#tradeoffs}

- **Continuous actions.** Choosing $\operatorname*{arg\,max}_a \val{\hat q(s, a)}$ over a continuous set of actions is an optimization problem at every step. A policy that outputs a distribution, a Gaussian over a steering angle say, just samples from it ([[policy-parameterization]]).
- **Simpler targets.** In some problems the policy is a simpler function than the values: knowing *what* to do can be easier than knowing *how good* each option is, to the last decimal.
- **Built-in exploration.** A stochastic policy explores by itself, and learns how much to explore in each state, instead of using a fixed $\eps$.
- **Prior knowledge.** The form of the policy is a natural place to put what is known about good behavior.

The costs are as real. Gradient ascent finds a **local** optimum of $J$, not necessarily the best one. The gradient must be estimated from experience, and the estimates are **noisy**: most of this part is about reducing their variance ([[baseline]], [[actor-critic]], [[gae]]). Most policy-gradient methods are **on-policy**: they learn from experience generated by the current policy, so every change of the policy makes old experience stale, and learning needs much more data than an off-policy method that replays it ([[experience-replay]]). And a step of the gradient that is too long can wreck a good policy in one update ([[trpo]], [[ppo]]).

### A view from the corridor {#corridor}

Policy-gradient methods treat the corridor as a landscape: the performance $J(\boldsymbol\theta)$ over the weights, here a single number in effect, $p$. A value-based learner sees no landscape at all, only values, and its policy jumps between the two ends of \ref{fig-corridor}. A policy-gradient learner starts somewhere on the curve and climbs: the Story's learner starts at the ε-greedy policy that prefers left and, step by step, moves to the top. The rest of this part explains how to find the uphill direction from experience alone ([[pg-theorem]]), and how to climb fast without falling off ([[reinforce]] to [[ppo]]).

### Historical remarks {#history}

Learning a stochastic policy directly is as old as reinforcement learning: the learning automata of the 1960s and 1970s adjusted action probabilities directly, and the actor–critic architecture of Barto, Sutton and Anderson (1983) learned a policy alongside a critic. Williams (1992) gave the general family of gradient methods its theory and the name [[reinforce|REINFORCE]]. Sutton, McAllester, Singh and Mansour (2000) proved the policy gradient theorem with function approximation and argued, with the chattering of value-based methods in mind, for learning policies directly. Example 13.1, the short corridor, is from Sutton and Barto (2018).

## Card

### Idea

Value-based methods learn how good each action is and pick the best. Policy-gradient methods give the policy its own weights and adjust them directly, along the gradient of performance. The best policy can then be random, actions can be continuous, and behavior changes smoothly as the weights change.

::: analogy
A penalty taker who always shoots to the same side is easy to stop. The best strategy is to mix, and to choose how often to shoot left: something no ranking of “best side” can express.
:::

### Two ways {#compare}

| | value-based | policy-based |
| --- | --- | --- |
| learns | values $\val{\hat q(s, a, \mathbf w)}$ | a policy $\pol{\pi(a \mid s, \boldsymbol\theta)}$ |
| acts | best action, plus fixed exploration | samples from $\pol{\pi}$ |
| best policy | deterministic (or ε-random) | any mixture it can represent |
| continuous actions | a max at every step | sample a distribution |
| small update | can flip the action | changes probabilities a little |
| typical drawback | instability with approximation | noisy gradients, local optima |

### Why it matters {#why}

- When the agent cannot tell states apart, the best policy can be random: on the short corridor it is worth $\rew{-11.7}$, against $\rew{-44.2}$ for the best ε-greedy policy. [See it](lab:corridor-reinforce)
- All of modern deep policy optimization, from [[a2c]] to [[ppo]], starts here.

### Pitfalls

- Expecting a policy-gradient method to find the global optimum: it climbs to a local one.
- Forgetting that the policy's randomness is learned: it can shrink to near-determinism and stop exploring ([[entropy-bonus]]).
- Reusing old experience as if it came from the current policy: most policy-gradient methods are on-policy.

### Check yourself {#check}

::: question
Why can't ε-greedy find the best policy of the short corridor?
---
It can only step right with probability $1 - \eps/2$ or $\eps/2$, depending on which action looks best, and both are far from the best $p \approx 0.59$. Its randomness is fixed exploration, not something it learns.
:::

::: question
In a fully observable finite MDP, is a random policy ever strictly better than every deterministic one?
---
No: there is always an optimal deterministic policy. Random policies win when states cannot be told apart, or against an opponent who adapts.
:::

::: question
What does “a small change of the weights is a small change of behavior” buy?
---
A meaningful gradient: performance becomes a smooth function of the weights, so gradient ascent can climb it, with convergence results that value-based control lacks.
:::
