+++
summary = "The simplest policy-gradient method: play a whole episode with the current policy, then go back over it and make each action more likely in proportion to the return that followed it. The update is an unbiased sample of the policy gradient, so on average it climbs; one episode at a time it is very noisy, and the step size must be small."
change = "Learn from whole episodes of an MDP instead of single pulls: after each episode, move the policy's weights along γᵗ G_t ∇ ln π(A_t | S_t, θ) for every step, with the return from that step in place of the bandit's reward, any differentiable policy, and no baseline."
prereqs = ["pg-theorem", "policy-parameterization", "gradient-bandit", "mc-prediction"]
lab = "corridor-reinforce"
sources = [
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., §13.3 and Figure 13.1", url = "http://incompleteideas.net/book/the-book-2nd.html" },
  { text = "Williams (1992), Simple statistical gradient-following algorithms for connectionist reinforcement learning, Machine Learning 8", url = "https://doi.org/10.1007/BF00992696" },
  { text = "Williams (1987), A class of gradient-estimating algorithms for reinforcement learning in neural networks, Proceedings of the IEEE First International Conference on Neural Networks" },
  { text = "OpenAI (2018), Spinning Up in Deep RL, Vanilla Policy Gradient", url = "https://spinningup.openai.com/en/latest/algorithms/vpg.html" },
]

[story]
scene = "corridor"
env = "corridor"
seed = 2
average = 100
formula = '\step{1}{\boldsymbol\theta \leftarrow \boldsymbol\theta + \alp\,\gam^t\,\rew{G_t}\,\nabla \ln \pol{\pi(A_t \mid S_t, \boldsymbol\theta)}} \step{2}{\qquad \rew{G_t} = -(T - t)}'

[story.runs]
learn = { algorithm = "reinforce", alpha = 0.0001220703125, right0 = 0.05, gamma = 1.0, features = "own", maxSteps = 1000, units = 1000, measures = ["right"], name = "REINFORCE, α = 2⁻¹³" }
thrown = { algorithm = "reinforce", alpha = 0.000244140625, right0 = 0.05, gamma = 1.0, features = "own", maxSteps = 1000, units = 20, seed = 1027, name = "REINFORCE, α = 2⁻¹²" }
a12 = { algorithm = "reinforce", alpha = 0.000244140625, right0 = 0.05, gamma = 1.0, features = "own", maxSteps = 1000, units = 1000, name = "α = 2⁻¹²" }
a13 = { algorithm = "reinforce", alpha = 0.0001220703125, right0 = 0.05, gamma = 1.0, features = "own", maxSteps = 1000, units = 1000, name = "α = 2⁻¹³" }
a14 = { algorithm = "reinforce", alpha = 0.00006103515625, right0 = 0.05, gamma = 1.0, features = "own", maxSteps = 1000, units = 1000, name = "α = 2⁻¹⁴" }
+++

## Story

::: step {run = "learn", at = 0, formula = 1}
**REINFORCE** on the short corridor ([[why-policy]]): three cells that look alike, a swapped middle cell, $\rew{-1}$ per step. The policy starts by stepping right 5% of the time. It learns from whole episodes: play to the end, then go back over every step and nudge the weights so that the action taken there becomes more likely, by an amount proportional to the return that followed it.
:::

::: step {run = "learn", at = 0, play = 1, pace = 40, formula = 2}
The first episode: 29 steps. Eighteen of them bump into the wall of cell 1; then the agent bounces between cells 2 and 3 four times before stepping right into the goal. The return from each step is minus the number of steps still to go: $\rew{G_0} = -29$ down to $\rew{G_{28}} = -1$.
:::

::: step {run = "learn", at = 1}
Then 29 updates, one per step. Every return is negative, so every action taken becomes a little *less* likely. Each left step raises the chance of right a little; each right step lowers it more, $(1 - p)/p = 19$ times more for the same return, because it was the rarer action. The balance after this episode: from 5.000% to 5.011%. The step size is tiny, $\alp = 2^{-13}$, for a reason the end of this story shows.
:::

::: step {run = "learn", at = 1, play = 4, pace = 10}
Episodes 2 to 5. The first three move the chance by a few hundredths of a percent, one of them down. Then episode 5 takes 227 steps, 150 of them in cell 1, with only 4 steps right. Its left steps carry returns as large as $\rew{-227}$, and they push hard: the chance of right jumps from 5.01% to 6.58%. One bad episode taught more than all the others together.
:::

::: step {run = "learn", at = 100}
After 100 episodes: 14.5%. The climb is slow and uneven, because each episode's estimate of the gradient is noisy ([[pg-theorem]]): most episodes say little, a few long ones say a lot, and some point downhill.
:::

::: step {run = "learn", at = 1000}
After 1000 episodes the agent steps right 48.8% of the time, worth $\rew{-12.1}$ per episode, close to the top of the landscape ($\rew{-11.7}$ at 59%). It got there without a model and without any value function: returns and the gradient of $\ln \pol{\pi}$ were enough.
:::

::: step {run = "thrown", at = 11}
Why so small a step size? Here is another run with twice the step size, $\alp = 2^{-12}$. After 11 episodes it steps right 5.36% of the time, an ordinary start.
:::

::: step {run = "thrown", at = 11, play = 1, pace = 6}
Its 12th episode is long: 426 steps, 404 of them left. The updates go back over it, each left step with a large return, and push the chance of right up, past 50% by the 133rd update. By the end of the episode the policy steps right with probability $1 - 10^{-8}$.
:::

::: step {run = "thrown", at = 20}
Now it steps right in every cell, and from cell 2 that leads back to cell 1: it never reaches the goal, and every episode is cut after 1000 steps. It is stuck for good. Each update is scaled by the chance of the actions *not* taken, now almost zero, so nothing moves: out there the landscape is flat. Such a throw is rare: over 200 runs of 1000 episodes with $\alp = 2^{-12}$, 4 ended thrown off like this, and none with $\alp = 2^{-13}$.
:::

::: step {run = "learn", at = 1000, curves = ["a12", "a13", "a14"], metric = "steps"}
Averaged over 100 runs. With $\alp = 2^{-12}$, three runs out of 100 were thrown off like this one, all within their first 20 episodes, and their 1000-step episodes keep the average above 40 steps for good. With $2^{-13}$ the average ends near 12 steps, close to the best 11.7; $2^{-14}$ is safe but slower. [Try the step sizes in the Lab](lab:corridor-reinforce).
:::

## Textbook

### The update {#update}

The policy gradient theorem in its sampled form ([[pg-theorem]]) says that, for an episode generated by $\pol{\pi(\cdot \mid \cdot, \boldsymbol\theta)}$,

$$\nabla J(\boldsymbol\theta) = \mathbb E_\pi\Big[\sum_{t=0}^{T-1} \gam^t\,\rew{G_t}\,\nabla \ln \pol{\pi(A_t \mid S_t, \boldsymbol\theta)}\Big].$$

REINFORCE (Williams, 1992) is stochastic gradient ascent on this expectation: after each episode, for each of its steps,

$$\boldsymbol\theta \leftarrow \boldsymbol\theta + \alp\,\gam^t\,\rew{G_t}\,\nabla \ln \pol{\pi(A_t \mid S_t, \boldsymbol\theta)}, \label{eq-update}$$

where $\rew{G_t} = \rew{R_{t+1}} + \gam\,\rew{R_{t+2}} + \dots + \gam^{T-t-1}\,\rew{R_T}$ is the return from step $t$. The returns are complete, so REINFORCE is a [[mc-prediction|Monte Carlo]] method: it needs the end of the episode, and applies to episodic tasks only.

Written with $\nabla \ln \pol{\pi} = \nabla \pol{\pi} / \pol{\pi}$, the update has a plain reading. The vector $\nabla \pol{\pi(A_t \mid S_t, \boldsymbol\theta)}$ is the direction in weight space that most increases the probability of repeating $A_t$ in $S_t$. The update moves along it in proportion to the return, so actions followed by high returns become more likely, and inversely to the action's probability, which corrects for the frequent actions being updated more often: without the division, an action would gain simply by being chosen a lot.

### The algorithm {#algorithm}

::: algorithm {#alg-reinforce} REINFORCE: Monte Carlo policy-gradient control (episodic)
Input: a differentiable policy $\pol{\pi(a \mid s, \boldsymbol\theta)}$, a step size $\alp > 0$, a discount $\gam$
Set the policy weights $\boldsymbol\theta$ (for example, to 0)
Repeat for each episode:
  Generate an episode $S_0, A_0, \rew{R_1}, \dots, S_{T-1}, A_{T-1}, \rew{R_T}$ following $\pol{\pi(\cdot \mid \cdot, \boldsymbol\theta)}$
  For each step $t = 0, 1, \dots, T - 1$:
    $\rew{G} \leftarrow \sum_{k=t+1}^{T} \gam^{k-t-1}\,\rew{R_k}$
    $\boldsymbol\theta \leftarrow \boldsymbol\theta + \alp\,\gam^t\,\rew{G}\,\nabla \ln \pol{\pi(A_t \mid S_t, \boldsymbol\theta)}$
:::

As Sutton and Barto write it, and as the Lab runs it, the weights change after each step's update, so later steps of the same episode use the slightly updated policy for their gradient; adding up all the steps' terms first and then updating once is the other common choice. With a softmax policy, $\nabla \ln \pol{\pi}$ is the action's features minus their average under the policy; with a Gaussian, the two parts that pull the mean and stretch the spread ([[policy-parameterization]]).

### Convergence {#convergence}

Each episode's update is, on average, a step along $\nabla J(\boldsymbol\theta)$: an unbiased estimate of the gradient. For a small enough step size, the expected performance therefore improves, and with step sizes that decrease in the usual way ($\sum_t \alp_t = \infty$, $\sum_t \alp_t^2 < \infty$) the weights converge to a local optimum of $J$ under standard stochastic-approximation conditions. Nothing more is guaranteed: the optimum can be local, and the approach can be slow.

### Variance and the step size {#variance}

The estimate is unbiased but its variance is high, for three reasons visible on the short corridor. The return of a whole episode collects the randomness of every later action and transition. The size of each step's push is proportional to its return, so a few long episodes dominate. And when all returns have the same sign, as when every step costs, every action taken is pushed the same way, *less* likely, and learning rests on the balance between pushes on different actions: an action is favored only by being pushed down less than the others.

::: figure {#fig-alpha}
{{reinforce-alpha}}
REINFORCE on the short corridor with three step sizes, starting from the policy that steps right 5% of the time. The middle step size approaches the best average, $\rew{-11.7}$; the smallest is slower. With the largest, a few runs are thrown to a policy that always steps the same way and never recover. Computed live by the Lab; after Sutton & Barto, Figure 13.1.
:::

High variance forces a small step size, and the failure of a large one is not gentle. With a softmax policy, one long episode at $\alp = 2^{-12}$ can push the probability of an action to within $10^{-8}$ of 1 (the Story shows one). There the gradient of $J$ is practically zero: the score of the near-certain action, $1 - \pol{\pi(a \mid s)}$, vanishes, and the other action is never tried. The policy sits on a flat part of the landscape far from the top, and no number of further episodes moves it. Saturated softmaxes and collapsed Gaussians are the policy-gradient version of a learning rate that is too high; the remedies are smaller steps, a baseline ([[baseline]]), keeping the policy random ([[entropy-bonus]]), and limiting how far one update can move the policy ([[trpo]], [[ppo]]).

The same holds for continuous actions. On the throw ([[policy-parameterization]]), REINFORCE with a Gaussian policy needs a step size ten times smaller than with a baseline: with $\alp = 0.0003$ its aim averages only 37° after 1000 throws, against 45° with a baseline at $\alp = 0.003$; with $\alp = 0.001$, 26 runs out of 100 end with their aim outside 25°–65°.

### Example: the short corridor {#example}

On the short corridor (Example 13.1 of Sutton and Barto, [[why-policy]]), the policy has one feature per action and the same probability $p$ of stepping right in every cell. Starting from $p = 0.05$, REINFORCE with $\alp = 2^{-13}$ climbs toward the best probability, $p_* \approx 0.59$: averaged over 100 runs, $p$ is 15% after 100 episodes, 36% after 500 and 47.5% after 1000, and an episode takes 12 steps on average at the end, against 11.7 for the best policy. The landscape is steepest near the ends, where both the gradient and the risk are largest: from $p = 0.05$, a single long episode can move $p$ by more than a percentage point, a hundred times as far as a typical one.

### Historical remarks {#history}

Williams (1987, 1992) introduced REINFORCE as a family of algorithms for stochastic units in neural networks, and proved that their expected update follows the gradient of expected reward; the name is an acronym for the form of the update, *REward Increment = Nonnegative Factor × Offset Reinforcement × Characteristic Eligibility*, where the offset is a baseline (here zero) and the characteristic eligibility is $\nabla \ln \pol{\pi}$. Its use with returns over whole episodes, and the connection to the policy gradient theorem, are presented in Sutton and Barto (2018, §13.3), whose Figure 13.1 the Lab reproduces. In deep reinforcement learning the same algorithm, usually with a baseline, is often called *vanilla policy gradient*.

## Card

### Idea

Play a whole episode, then go back over it: at every step, make the action taken more likely in proportion to the return that followed it, divided by how likely it was. On average this follows the gradient of performance exactly; one episode at a time, it is very noisy.

::: analogy
A coach who watches the whole match before saying anything, then reviews every decision, praising those made before a good final score more and those before a bad one less. Fair on average; but one lucky or unlucky match sways a whole review.
:::

### The update {#update}

$$\boldsymbol\theta \leftarrow \boldsymbol\theta + \alp\,\gam^t\,\rew{G_t}\,\nabla \ln \pol{\pi(A_t \mid S_t, \boldsymbol\theta)}$$

for every step $t$ of the episode, after it ends.

### One change from the gradient bandit {#change}

| | learns from | update |
| --- | --- | --- |
| [[gradient-bandit]] | each pull, one state | $\pol{H(A)} \leftarrow \pol{H(A)} + \alp\,(\rew{R} - \rew{\bar R})\,(1 - \pol{\pi(A)})$, and the other arms down |
| REINFORCE | each episode, many states | $\boldsymbol\theta \leftarrow \boldsymbol\theta + \alp\,\gam^t\,\rew{G_t}\,\nabla \ln \pol{\pi(A_t \mid S_t, \boldsymbol\theta)}$ |

The bandit's preference update is the same rule for one state with a table of preferences ($\nabla \ln \pol{\pi}$ of a softmax is $1 - \pol{\pi(A)}$ for the arm pulled, $-\pol{\pi(b)}$ for the others), and it has a baseline; plain REINFORCE has none ([[baseline]]).

### Backup diagram {#backup}

{{backup reinforce}}

A whole episode, sampled to the end; the return of each step weights the gradient of the log-probability of its action.

### Pseudocode

::: pseudocode
Parameters: step size $\alp$, discount $\gam$; a differentiable policy $\pol{\pi(a \mid s, \boldsymbol\theta)}$
Set the policy weights $\boldsymbol\theta$ (here, stepping right 5% of the time)
Repeat for each episode:
  Start: $S_0$ {#start}
  Generate an episode $S_0, A_0, \rew{R_1}, \dots, S_{T-1}, A_{T-1}, \rew{R_T}$ following $\pol{\pi(\cdot \mid \cdot, \boldsymbol\theta)}$ {#generate}
  For each step $t = 0, 1, \dots, T - 1$:
    $\rew{G} \leftarrow \sum_{k=t+1}^{T} \gam^{k-t-1}\,\rew{R_k}$ {#return}
    $\boldsymbol\theta \leftarrow \boldsymbol\theta + \alp\,\gam^t\,\rew{G}\,\nabla \ln \pol{\pi(A_t \mid S_t, \boldsymbol\theta)}$ {#update}
:::

### Perks

- Unbiased: on average each update follows the true gradient. [See it climb](lab:corridor-reinforce)
- Simple: no value function, no model, any differentiable policy.
- Handles random optimal policies and continuous actions.

### Flaws

- High variance: slow learning and a small step size.
- Waits for the end of each episode; episodic tasks only.
- On-policy: each episode is used once and thrown away.
- A step that is too large can saturate the policy, where the gradient vanishes for good.

### Knobs

| Knob | Too low | Too high |
| --- | --- | --- |
| $\alp$ step size | slow climb | one long episode throws the policy to an extreme it never leaves |
| $\gam$ discount | late rewards ignored | more variance from long returns |

### Pitfalls

- Using a step size tuned for value methods: REINFORCE's gradients scale with the returns, which can be large.
- Rewards all of one sign: every action is pushed the same way, and learning rests on small differences; a baseline fixes this ([[baseline]]).
- Updating with an episode generated by an older policy: the estimate is then biased.
- Dropping $\gam^t$ without noticing: common in practice, but then the update is no longer the gradient of the discounted value ([[pg-theorem]]).

### Check yourself {#check}

::: question
On the short corridor every return is negative. How can REINFORCE make stepping right more likely?
---
Every action taken becomes less likely, but by different amounts: a left step raises the chance of right a little, a right step lowers it more. What counts is the balance over the episode, and on average it points uphill, toward the best probability of about 59%.
:::

::: question
Why is the update divided by the probability of the action (the $\ln$ in $\nabla \ln \pol{\pi}$)?
---
Frequent actions are updated more often. Without the division they would gain just by being chosen; with it, the expected update is the true gradient.
:::

::: question
A run with a large step size ends up stepping right with probability $1 - 10^{-8}$ and never recovers. Why?
---
The score of a near-certain action, $1 - \pol{\pi(a \mid s)}$, is almost zero, and the other action is never tried, so every update is almost zero: the policy sits on a flat part of the landscape.
:::
