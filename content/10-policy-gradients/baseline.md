+++
summary = "REINFORCE, but each return is judged against what was expected from that state: a learned estimate of its value, the baseline. Actions followed by better-than-expected returns become more likely, worse ones less likely. The gradient stays unbiased, its variance drops, and much larger steps become safe."
change = "Subtract a baseline from each return before using it: G_t − v̂(S_t, w), where v̂ is a state-value estimate learned alongside by Monte Carlo. The update stays unbiased and gets much less noisy."
prereqs = ["reinforce", "pg-theorem", "gradient-mc"]
lab = "corridor-baseline"
sources = [
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., §13.4 and Figure 13.2", url = "http://incompleteideas.net/book/the-book-2nd.html" },
  { text = "Williams (1992), Simple statistical gradient-following algorithms for connectionist reinforcement learning, Machine Learning 8", url = "https://doi.org/10.1007/BF00992696" },
  { text = "Sutton (1984), Temporal credit assignment in reinforcement learning, PhD thesis, University of Massachusetts Amherst" },
  { text = "Greensmith, Bartlett & Baxter (2004), Variance reduction techniques for gradient estimates in reinforcement learning, Journal of Machine Learning Research 5", url = "https://www.jmlr.org/papers/v5/greensmith04a.html" },
  { text = "OpenAI (2018), Spinning Up in Deep RL, Part 3: Intro to Policy Optimization (baselines)", url = "https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html" },
]

[story]
scene = "corridor"
env = "corridor"
seed = 3
average = 100
formula = '\step{1}{\boldsymbol\theta \leftarrow \boldsymbol\theta + \alp^{\boldsymbol\theta}\,\gam^t\,\big(\rew{G_t} - \val{\hat v(S_t, \mathbf w)}\big)\,\nabla \ln \pol{\pi(A_t \mid S_t, \boldsymbol\theta)}} \step{2}{\qquad \mathbf w \leftarrow \mathbf w + \alp^{\mathbf w}\,\big(\rew{G_t} - \val{\hat v(S_t, \mathbf w)}\big)\,\nabla \val{\hat v(S_t, \mathbf w)}}'

[story.runs]
learn = { algorithm = "baseline", alpha = 0.001953125, alphaW = 0.015625, right0 = 0.05, gamma = 1.0, features = "own", maxSteps = 1000, units = 1000, measures = ["right"], name = "with a baseline, α = 2⁻⁹" }
plain = { algorithm = "reinforce", alpha = 0.0001220703125, right0 = 0.05, gamma = 1.0, features = "own", maxSteps = 1000, units = 1000, name = "REINFORCE, α = 2⁻¹³" }
plain9 = { algorithm = "reinforce", alpha = 0.001953125, right0 = 0.05, gamma = 1.0, features = "own", maxSteps = 1000, units = 1000, name = "REINFORCE, α = 2⁻⁹" }
+++

## Story

::: step {run = "learn", at = 0, formula = 1}
The short corridor once more, from the same start: right 5% of the time. Plain [[reinforce|REINFORCE]] suffers because every return is negative: every action taken is pushed down, and learning rests on small differences. The fix is to judge each return against what was **expected** from that state, a **baseline**. Here it is a learned estimate of the value of the state, $\val{\hat v(s, \mathbf w)}$; the cells look alike, so it is a single number, the dashed line on the landscape.
:::

::: step {run = "learn", at = 0, play = 1, pace = 40, formula = 2}
The first episode: 62 steps, 57 of them bumping into the wall of cell 1. The baseline starts at 0 and learns from each return as it goes over the episode, like Monte Carlo prediction ([[gradient-mc]]); by the end it expects $\val{-16.5}$.
:::

::: step {run = "learn", at = 1}
The early steps, with returns down to $\rew{-62}$, did worse than the baseline expected: their actions become less likely. The last 18, near the end of the episode, did better: theirs become more likely. Better or worse than expected, not good or bad, decides the direction. On balance the chance of right rises from 5.0% to 6.9%.
:::

::: step {run = "learn", at = 1, play = 2, pace = 60}
Episode 3 takes only 9 steps, and every return, from $\rew{-9}$ to $\rew{-1}$, beats the baseline of about $\val{-15}$: all nine actions become more likely, the four steps right among them, and the chance of right rises from 7.2% to 8.0%. Short episodes now teach as much as long ones.
:::

::: step {run = "learn", at = 50}
After 50 episodes it steps right 54% of the time, and the baseline expects $\val{-9.2}$. Plain REINFORCE, on average, is not this far after 1000 episodes. This learner also takes steps 16 times larger, $\alp^{\boldsymbol\theta} = 2^{-9}$ instead of $2^{-13}$: centered on what was expected, its pushes go up as often as down, and a large step is no longer reckless.
:::

::: step {run = "learn", at = 1000}
After 1000 episodes: 59.0%, the top of the landscape. Larger steps have a cost, though: single runs wander around the top, where the landscape is flat. In this one, a single 54-step episode, number 255, knocked the chance of right from 51% to 31%; it climbed back within about 100 episodes.
:::

::: step {run = "learn", at = 1000, curves = ["learn", "plain", "plain9"], metric = "steps"}
Averaged over 100 runs. With a baseline, episodes take 12 steps after 200 episodes, as many as plain REINFORCE manages after 1000. And plain REINFORCE with the baseline's step size is thrown off the landscape in 94 runs of 100: the baseline is what makes that step size safe. [Race them in the Lab](lab:corridor-baseline).
:::

## Textbook

### Subtracting a baseline {#baseline}

The policy gradient theorem allows any function of the state to be subtracted from the action values ([[pg-theorem]]):

$$\nabla J(\boldsymbol\theta) \propto \sum_s \mu(s) \sum_a \big(\val{q_\pi(s, a)} - b(s)\big)\,\nabla \pol{\pi(a \mid s, \boldsymbol\theta)},$$

because $\sum_a b(s)\,\nabla \pol{\pi(a \mid s, \boldsymbol\theta)} = b(s)\,\nabla \sum_a \pol{\pi(a \mid s, \boldsymbol\theta)} = b(s)\,\nabla 1 = \mathbf 0$. The baseline may be any function, even a random variable, as long as it does not depend on the action. Following the same steps as for REINFORCE gives the update

$$\boldsymbol\theta \leftarrow \boldsymbol\theta + \alp^{\boldsymbol\theta}\,\gam^t\,\big(\rew{G_t} - b(S_t)\big)\,\nabla \ln \pol{\pi(A_t \mid S_t, \boldsymbol\theta)}, \label{eq-update}$$

whose expectation is the same gradient as before. What changes is its variance, and with it how fast and how safely the policy can learn.

### Why it helps {#why}

Without a baseline, the direction in which an action's probability moves is set by the sign of the return. When all returns have the same sign, as when every step costs, every action taken is pushed the same way, and the policy improves only through the difference between large pushes on different actions: a noisy difference of large numbers. With a baseline near the typical return, the weight $\rew{G_t} - b(S_t)$ is positive for actions followed by better-than-usual returns and negative for the others. The pushes are centered, smaller, and point the right way much more often.

The variance can be computed exactly where the problem is small. On the short corridor at $p = 0.3$, the single-episode estimate of the slope along the preference for right ([[pg-theorem]]) has a standard deviation of 38.8 without a baseline; with the baseline a learned single number settles on, the average value of the cells visited ($\val{-13.9}$), it drops to 24.4. Its average does not move.

::: figure {#fig-estimates}
{{pg-estimates}}
Single-episode estimates of the same slope on the short corridor at $p = 0.3$, without and with a baseline. Both average close to the true 8.48; the baseline cuts the spread by more than a third.
:::

Which baseline is best? For a constant, minimizing the variance of the estimate gives the average of the returns weighted by the squared size of the score, $b_* = \mathbb E[\rew{G}\,\|\nabla \ln \pol{\pi}\|^2] / \mathbb E[\|\nabla \ln \pol{\pi}\|^2]$, for a single decision (Greensmith, Bartlett and Baxter, 2004). The minimum is flat: on the corridor, the best constant for whole episodes, about $-16.9$, gives a spread of 23.9, against 24.4 for $-13.9$. In practice the natural choice is the state value itself, $b(s) = \val{\hat v(s, \mathbf w)}$, close to the best and different in each state. With $b = \val{v_\pi}$, the weight $\rew{G_t} - \val{v_\pi(S_t)}$ is an unbiased estimate of the **advantage** of $A_t$, how much better it was than the policy's average action in $S_t$.

### Learning the baseline {#learned}

The baseline $\val{\hat v(s, \mathbf w)}$ is learned from the same returns, by [[gradient-mc|gradient Monte Carlo]]:

$$\mathbf w \leftarrow \mathbf w + \alp^{\mathbf w}\,\big(\rew{G_t} - \val{\hat v(S_t, \mathbf w)}\big)\,\nabla \val{\hat v(S_t, \mathbf w)}. \label{eq-critic}$$

There are now two step sizes. The one for the values is the familiar one: for linear features, a rule of thumb is $\alp^{\mathbf w} = 0.1 / \mathbb E[\|\nabla \val{\hat v(S_t, \mathbf w)}\|^2]$. The one for the policy depends on the scale of the rewards and on the parameterization of the policy, and has to be tuned; but a baseline makes much larger values safe. On the short corridor, $\alp^{\boldsymbol\theta} = 2^{-9}$ with a baseline is 16 times the largest safe step size without one, and the same $2^{-9}$ without a baseline throws 94 runs of 100 off the landscape.

### The algorithm {#algorithm}

::: algorithm {#alg-baseline} REINFORCE with baseline (episodic)
Input: a differentiable policy $\pol{\pi(a \mid s, \boldsymbol\theta)}$, a differentiable state-value estimate $\val{\hat v(s, \mathbf w)}$, step sizes $\alp^{\boldsymbol\theta} > 0$ and $\alp^{\mathbf w} > 0$, a discount $\gam$
Set the weights $\boldsymbol\theta$ and $\mathbf w$ (for example, to 0)
Repeat for each episode:
  Generate an episode $S_0, A_0, \rew{R_1}, \dots, S_{T-1}, A_{T-1}, \rew{R_T}$ following $\pol{\pi(\cdot \mid \cdot, \boldsymbol\theta)}$
  For each step $t = 0, 1, \dots, T - 1$:
    $\rew{G} \leftarrow \sum_{k=t+1}^{T} \gam^{k-t-1}\,\rew{R_k}$
    $\del \leftarrow \rew{G} - \val{\hat v(S_t, \mathbf w)}$
    $\mathbf w \leftarrow \mathbf w + \alp^{\mathbf w}\,\del\,\nabla \val{\hat v(S_t, \mathbf w)}$
    $\boldsymbol\theta \leftarrow \boldsymbol\theta + \alp^{\boldsymbol\theta}\,\gam^t\,\del\,\nabla \ln \pol{\pi(A_t \mid S_t, \boldsymbol\theta)}$
:::

### A baseline is not a critic {#not-critic}

The learned $\val{\hat v}$ looks like a critic, but it is only used to judge the return of the state it is evaluated in, never to stand in for the rest of an episode. The method is still Monte Carlo: it waits for complete returns, its gradient estimate is unbiased whatever the quality of $\val{\hat v}$, and it inherits the convergence of REINFORCE. Using the value estimate to **bootstrap**, replacing the rest of the return by the estimate of the next state, is the step to [[actor-critic|actor–critic methods]], which learn online and with less variance, at the price of some bias.

### Examples {#examples}

On the short corridor, with $\alp^{\boldsymbol\theta} = 2^{-9}$ and $\alp^{\mathbf w} = 2^{-6}$, REINFORCE with baseline steps right 53% of the time after 100 episodes and 57% after 200, on average over 100 runs (the best is 59%); plain REINFORCE with $\alp = 2^{-13}$ is at 15% and 22%.

::: figure {#fig-curves}
{{baseline-curves}}
REINFORCE with and without a learned baseline on the short corridor, each with a good step size. Computed live by the Lab; after Sutton & Barto, Figure 13.2.
:::

The same holds with a Gaussian policy. On the throw ([[policy-parameterization]]), the baseline is the running average distance of the throws: with it and $\alp^{\boldsymbol\theta} = 0.003$, the aim is within 5° of 45° after 500 throws in 99 runs of 100, and after 1000 in all of them; without it, $\alp^{\boldsymbol\theta} = 0.0003$ leaves the aim at 37° on average after 1000 throws, and $\alp^{\boldsymbol\theta} = 0.001$ already scatters a quarter of the runs far from 45°.

### Historical remarks {#history}

Baselines are as old as policy-gradient methods. Sutton (1984) studied reinforcement comparison, in which a reward is judged against a running average of past rewards, the baseline of the [[gradient-bandit|gradient bandit]]; Williams (1992) included a baseline, the *offset reinforcement*, in REINFORCE, and showed that it leaves the expected update unchanged. Greensmith, Bartlett and Baxter (2004) analyzed which baselines minimize the variance of the gradient estimate. Learning the baseline as a state-value function, as here, is the form given by Sutton and Barto (2018, §13.4), whose Figure 13.2 the Lab reproduces.

## Card

### Idea

REINFORCE, with each return judged against what was expected from its state: a learned state value. Better than expected makes the action more likely, worse makes it less likely. The average update is unchanged, its noise much smaller.

::: analogy
A student who scores 70 on a hard test did well; on an easy one, badly. Grading against what was expected, not on the raw score, tells which study habits to keep.
:::

### The update {#update}

$$\boldsymbol\theta \leftarrow \boldsymbol\theta + \alp^{\boldsymbol\theta}\,\gam^t\,\big(\rew{G_t} - \val{\hat v(S_t, \mathbf w)}\big)\,\nabla \ln \pol{\pi(A_t \mid S_t, \boldsymbol\theta)}$$

with $\val{\hat v}$ learned from the same returns, $\mathbf w \leftarrow \mathbf w + \alp^{\mathbf w}\,\big(\rew{G_t} - \val{\hat v(S_t, \mathbf w)}\big)\,\nabla \val{\hat v(S_t, \mathbf w)}$.

### One change from REINFORCE {#change}

| | weight of $\nabla \ln \pol{\pi(A_t \mid S_t)}$ | learns besides the policy |
| --- | --- | --- |
| [[reinforce]] | $\rew{G_t}$ | nothing |
| REINFORCE with baseline | $\rew{G_t} - \val{\hat v(S_t, \mathbf w)}$ | a state value, by Monte Carlo |

### Backup diagram {#backup}

{{backup baseline}}

The same complete episode as REINFORCE; each return is compared with the value estimate of the state it starts from.

### Pseudocode

::: pseudocode
Parameters: step sizes $\alp^{\boldsymbol\theta}$ (policy) and $\alp^{\mathbf w}$ (baseline), discount $\gam$; a policy $\pol{\pi(a \mid s, \boldsymbol\theta)}$ and a value estimate $\val{\hat v(s, \mathbf w)}$
Set $\boldsymbol\theta$ (here, stepping right 5% of the time) and $\mathbf w$ (here 0)
Repeat for each episode:
  Start: $S_0$ {#start}
  Generate an episode $S_0, A_0, \rew{R_1}, \dots, S_{T-1}, A_{T-1}, \rew{R_T}$ following $\pol{\pi(\cdot \mid \cdot, \boldsymbol\theta)}$ {#generate}
  For each step $t = 0, 1, \dots, T - 1$:
    $\rew{G} \leftarrow \sum_{k=t+1}^{T} \gam^{k-t-1}\,\rew{R_k}$ {#return}
    $\del \leftarrow \rew{G} - \val{\hat v(S_t, \mathbf w)}$, then $\mathbf w \leftarrow \mathbf w + \alp^{\mathbf w}\,\del\,\nabla \val{\hat v(S_t, \mathbf w)}$ {#critic}
    $\boldsymbol\theta \leftarrow \boldsymbol\theta + \alp^{\boldsymbol\theta}\,\gam^t\,\del\,\nabla \ln \pol{\pi(A_t \mid S_t, \boldsymbol\theta)}$ {#update}
:::

### Perks

- Still unbiased, with much less variance: faster learning. [See the race](lab:corridor-baseline)
- Much larger policy step sizes become safe.
- The learned values are useful on their own, and lead to actor–critic.

### Flaws

- Still Monte Carlo: waits for the end of each episode.
- Two step sizes to tune.
- Variance remains: the returns themselves are still noisy.

### Knobs

| Knob | Too low | Too high |
| --- | --- | --- |
| $\alp^{\boldsymbol\theta}$ policy step size | slow climb | wanders around the top; at worst, saturates |
| $\alp^{\mathbf w}$ baseline step size | stale baseline: back to REINFORCE's noise | baseline chases the last return |
| $\gam$ discount | late rewards ignored | more variance from long returns |

### Pitfalls

- A baseline that depends on the action: the estimate is then biased.
- Expecting the baseline to fix bias or speed up credit assignment: it only reduces variance; bootstrapping is the next step ([[actor-critic]]).
- Using the step size of plain REINFORCE: with a baseline, much larger steps work.

### Check yourself {#check}

::: question
Why does subtracting a baseline not bias the gradient?
---
Because the score averages to zero in every state: $\sum_a \pol{\pi(a \mid s)}\,\nabla \ln \pol{\pi(a \mid s)} = \nabla \sum_a \pol{\pi(a \mid s)} = \mathbf 0$. Any $b(s)$ times it averages to zero too.
:::

::: question
On the short corridor, a 9-step episode comes after the baseline has learned to expect about $-15$. What happens to the actions of that episode?
---
Every return, from $-9$ to $-1$, beats the baseline, so every action taken becomes more likely. Without a baseline, all of them would have been made less likely.
:::

::: question
Is REINFORCE with baseline an actor–critic method?
---
No: its value estimate only judges complete returns, it never replaces the rest of an episode by an estimate (no bootstrapping). It stays unbiased and Monte Carlo.
:::
