+++
summary = "An off-policy actor–critic that is paid to stay random: every reward comes with a bonus for the policy's entropy, so the policy explores by itself and stays random wherever the choice does not matter. Twin critics from TD3, a stochastic actor trained through its own samples, and an entropy weight that tunes itself. On Pendulum it learns on all 20 seeds and ends with the tightest results of the three."
change = "Replace TD3's deterministic actor and added noise by a stochastic actor (a squashed Gaussian, trained through its samples), add an entropy bonus to every target, and tune its weight α automatically toward a target entropy."
prereqs = ["td3", "entropy-bonus"]
lab = "sac-pendulum"
sources = [
  { text = "Haarnoja, Zhou, Abbeel & Levine (2018), Soft actor-critic: off-policy maximum entropy deep reinforcement learning with a stochastic actor, ICML", url = "https://arxiv.org/abs/1801.01290" },
  { text = "Haarnoja et al. (2018), Soft actor-critic algorithms and applications", url = "https://arxiv.org/abs/1812.05905" },
  { text = "Ziebart, Maas, Bagnell & Dey (2008), Maximum entropy inverse reinforcement learning, AAAI", url = "https://cdn.aaai.org/AAAI/2008/AAAI08-227.pdf" },
  { text = "Fujimoto, van Hoof & Meger (2018), Addressing function approximation error in actor-critic methods, ICML", url = "https://arxiv.org/abs/1802.09477" },
]

[story]
scene = "pendulum"
env = "pendulum"
formula = '''\step{1}{y = \rew{r} + \gam\Big(\min_{i} \val{\hat q'_i(s', a')} - \alpha \ln \pol{\pi(a' \mid s')}\Big),\ a' \sim \pol{\pi(\cdot \mid s')} \qquad} \step{2}{\ln\alpha \leftarrow \ln\alpha - \lambda\big(-\ln \pol{\pi(a \mid s)} - \bar{\mathcal H}\big)}'''

[story.runs]
learn = { recording = "sac-pendulum", name = "SAC" }
ddpg = { recording = "ddpg-pendulum", name = "DDPG" }
td3 = { recording = "td3-pendulum", name = "TD3" }
+++

## Story

::: step {run = "learn", at = 0, map = "action"}
**Random on purpose.** SAC's actor does not name one torque: for each angle and spin it gives a mean and a spread, and the torque is drawn from them. It is also paid for staying random: each reward comes with a bonus of $\alpha$ times the policy's entropy. Before learning, the spread is very wide. The map shows the mean, which is what the test episodes play.
:::

::: step {run = "learn", at = 3, formula = 1}
**Soft values.** Two critics, as in [[td3]], learn the reward to come *plus* the entropy to come (formula 1). After 4,500 steps the test still fails. But the entropy weight has already moved on its own, from 0.2 to 0.07: the policy is more random than it needs to be, so the bonus shrinks.
:::

::: step {run = "learn", at = 4, map = "action"}
**After 6,000 steps:** one swing back, upright by step 25, and held to the end, at −124. Every seed of the 20 gets its first test at −250 or better after 3 or 4 blocks.
:::

::: step {run = "learn", at = 40, formula = 2}
**The entropy weight tunes itself** (formula 2): it rises when the policy is less random than a target, and falls when it is more. By the end, $\alpha$ is tiny and the policy narrow, but not evenly: it stays widest near the top, where any small torque holds the pendulum, and narrowest during the swing, where the torque must be right.
:::

::: step {run = "learn", at = 40, curves = ["ddpg", "td3", "learn"], metric = "test"}
**Three methods, 20 seeds each.** All three swing the pendulum up on every seed. SAC is nearly as quick as DDPG, without its overestimation, and its final tests are the tightest. [Try it in the Lab](lab:sac-pendulum).
:::

## Textbook

### Maximum entropy reinforcement learning {#maxent}

Ordinary RL maximizes the expected return. SAC maximizes the return plus the policy's entropy at every state it visits:
$$J(\pol\pi) = \sum_t \mathbb E\Big[\rew{r(S_t, A_t)} + \alpha\,\mathcal H\big(\pol{\pi(\cdot \mid S_t)}\big)\Big], \label{objective}$$
where $\mathcal H(\pol{\pi(\cdot \mid s)}) = -\mathbb E_{a \sim \pi}[\ln \pol{\pi(a \mid s)}]$ and $\alpha$ weighs the two. Compare the entropy bonus of [[entropy-bonus]]: there the bonus is a regularizer added to the policy's loss at the states just visited; here it is part of the objective, so the critic also values *future* entropy, and the agent seeks out states where it can afford to stay random. The optimal policy under this objective puts probability on every action, in proportion to $e^{\val{q}/\alpha}$ with the soft values $\val q$: most on the best, a little on the rest. A softened greedy policy.

Why want that? Exploration comes from the objective, not from added noise. The policy does not collapse onto one action as soon as it looks best. And among near-equal choices it stays spread, which makes it more robust to errors in the critic and to changes in the world.

### Soft policy iteration {#soft-pi}

The objective has its own Bellman equations. For a policy $\pol\pi$, the **soft values** add the entropy still to come:
$$\val{q_\pi(s, a)} = \rew{r(s, a)} + \gam\, \mathbb E_{s'}\big[\val{v_\pi(s')}\big], \qquad \val{v_\pi(s)} = \mathbb E_{a \sim \pi}\big[\val{q_\pi(s, a)} - \alpha \ln \pol{\pi(a \mid s)}\big]. \label{soft-bellman}$$
They can be computed exactly as in [[policy-evaluation]], by applying \ref{soft-bellman} as an update until it settles. Policy improvement then moves the policy toward the softened greedy policy of those values, staying within the family $\Pi$ the policy can represent (Gaussians, say):
$$\pol{\pi_{\text{new}}}(\cdot \mid s) = \operatorname*{arg\,min}_{\pi' \in \Pi} D_{\mathrm{KL}}\big(\pi'(\cdot \mid s)\, \big\|\, g(\cdot \mid s)\big) \qquad g(a \mid s) = \frac{\exp\big(\val{q_{\pi_{\text{old}}}(s, a)} / \alpha\big)}{Z(s)}, \label{soft-improve}$$
where $Z(s)$ makes $g$, the softened greedy policy, a distribution.

::: theorem {#thm-soft-pi} Soft policy improvement
For every state and action, $\val{q_{\pi_{\text{new}}}(s, a)} \ge \val{q_{\pi_{\text{old}}}(s, a)}$. Alternating soft evaluation and soft improvement in a finite MDP converges to the policy in $\Pi$ with the highest soft values.
:::

::: proof Proof idea
The old policy is itself a candidate in \ref{soft-improve}, so the new one has a smaller divergence. Written out, that says $\mathbb E_{a \sim \pi_{\text{new}}}[\val{q_{\pi_{\text{old}}}(s, a)} - \alpha \ln \pol{\pi_{\text{new}}(a \mid s)}] \ge \val{v_{\pi_{\text{old}}}(s)}$ in every state. Using this inequality at the first step and following the old policy's values after it, then unrolling the soft Bellman equation one step at a time, as in the proof of [[policy-improvement]], gives the result (Haarnoja et al., 2018, Lemma 2 and Theorem 1).
:::

SAC is the approximate, sampled version: the critics do soft evaluation with gradient steps, and the actor does soft improvement with gradient steps on the same divergence.

### Soft values and the critics {#critics}

The soft action value includes the entropy still to come, and its target is
$$y = \rew{r} + \gam\Big(\min_{i = 1, 2} \val{\hat q_i(s', a', \mathbf w_i^-)} - \alpha \ln \pol{\pi(a' \mid s')}\Big), \qquad a' \sim \pol{\pi(\cdot \mid s')}. \label{soft-target}$$
Like [[td3]], SAC trains twin critics and takes the min, against overestimation, with soft target copies. Unlike TD3 it has no target actor: the next action is drawn from the current policy, whose randomness already smooths the target.

### The actor, trained through its samples {#actor}

The policy is a Gaussian squashed into the allowed range: $a = a_{\max}\tanh(u)$, $u = \mu(s) + \sigma(s)\,\varepsilon$, $\varepsilon \sim \mathcal N(0, 1)$, with the network giving $\mu$ and $\ln\sigma$. Writing the action as a function of the parameters and an independent noise is the **reparameterization**: the actor's loss
$$L(\boldsymbol\theta) = \mathbb E_{s,\,\varepsilon}\Big[\alpha \ln \pol{\pi(a_{\boldsymbol\theta} \mid s)} - \min_i \val{\hat q_i(s, a_{\boldsymbol\theta})}\Big] \label{actor-loss}$$
can be differentiated straight through the sampled action, as DDPG differentiates through its deterministic one ([[dpg]]). The gradient has much less variance than a likelihood-ratio estimate. Up to a constant and the factor $\alpha$, this loss is the divergence of \ref{soft-improve}, averaged over the states of the memory. The squashing changes the density, by the change of variables:
$$\ln \pol{\pi(a \mid s)} = \ln \mathcal N\big(u;\, \mu(s), \sigma(s)^2\big) - \sum_{i} \ln\big(a_{\max}\,(1 - \tanh^2 u_i)\big), \label{squash}$$
summed over the action's dimensions. Forgetting this term makes the policy think a saturated action, deep in the tanh's flat end, is as random as any other.

### Tuning the entropy weight {#alpha}

The right $\alpha$ depends on the scale of the rewards and changes as learning goes on, so the second version of SAC tunes it. It picks a target entropy $\bar{\mathcal H}$, by default minus the number of action dimensions, and adjusts $\ln\alpha$ by gradient steps on
$$J(\alpha) = \mathbb E_{a \sim \pi}\big[-\alpha\,(\ln \pol{\pi(a \mid s)} + \bar{\mathcal H})\big]. \label{alpha-loss}$$
When the policy is less random than the target, $\alpha$ grows and the bonus pushes randomness back up; when it is more random, $\alpha$ shrinks. In the recorded run, $\alpha$ fell from 0.2 to 0.034 within 7,500 steps and to 0.0005 by the end.

### The algorithm {#algorithm}

::: algorithm {#alg-sac} Soft actor–critic (SAC), with a tuned entropy weight
Parameters: memory size $N$, minibatch size $B$, step sizes, target speed $\tau$, target entropy $\bar{\mathcal H}$, random steps at the start
Initialize the actor (giving $\mu$ and $\ln\sigma$), two critics and their target copies, $\ln\alpha$, and an empty memory $\mathcal D$
Repeat for each step:
  In $S$, sample $A = a_{\max} \tanh(u)$, $u \sim \mathcal N(\mu(S), \sigma(S)^2)$; observe $\rew R$, $S'$; store the transition in $\mathcal D$
  Draw $B$ transitions; for each, sample $a' \sim \pol{\pi(\cdot \mid s')}$ and compute the target \ref{soft-target}
  Both critics: a gradient step down the squared error to that target
  Actor: sample $a_{\boldsymbol\theta}$ by reparameterization at each $s$ of the batch, and take a gradient step down \ref{actor-loss}, with \ref{squash} for $\ln\pol\pi$
  Entropy weight: a gradient step on $\ln\alpha$ down \ref{alpha-loss}
  Both target critics move by $\tau$ toward their networks
:::

### On Pendulum {#pendulum}

The recording uses the settings of the DDPG and TD3 recordings: networks of 64 + 64, Adam at $10^{-3}$, a memory of 100,000, batches of 128, $\tau = 0.005$, rewards scaled by 0.1, and $\alpha$ starting at 0.2. Over 20 seeds of each method:

| | DDPG | TD3 | SAC |
| --- | --- | --- | --- |
| seeds that hold the pendulum up at the end | 20 | 20 | 20 |
| first test at −250 or better, median block | 3 | 5 | 3.5 |
| steps until half the seeds train at −250 or better | 6,000 | 9,000 | 7,500 |
| last four tests, range over seeds | −121 to −132 | −122 to −139 | −121 to −128 |
| seeds whose average target ends above 0 | 11 | 0 | 0 |

A sweep of two knobs, 10 seeds per value (the Lab's sweep panel shows every run), finds all 90 runs holding the pendulum up at the end.

- **The entropy weight $\alpha$.** A fixed weight of 0.01, 0.05 or 0.2 learns as fast as the tuned one: half the seeds train at −250 or better within 6,750 to 7,500 steps. At $\alpha = 1$ the bonus outweighs the rewards: the training episodes stay near random, −264 in the median at the end, and half the seeds need 21,000 steps to train at −250. Yet the tests, which play the mean action, hold the pendulum up as well as any other setting. The policy has learned where to aim, and is paid to keep missing on purpose.
- **Target speed $\tau$.** Slow copies slow learning, 15,000 steps at $\tau = 0.001$ against 7,500 at the recording's 0.005, while fast ones do no harm: 6,000 steps at 0.02 and at 0.1. At $\tau = 0.1$ DDPG lost one seed for good; SAC's twin critics and stochastic targets lose none.

SAC combines the speed of DDPG with the honest values of TD3. On a task this small the differences are modest; on the harder locomotion tasks of the original papers, SAC's advantage in stability across seeds was the main result.

### Historical remarks {#history}

Maximum entropy RL grew out of work on inverse reinforcement learning and optimal control (Ziebart et al., 2008) and soft Q-learning. Haarnoja, Zhou, Abbeel and Levine (2018) made it practical with an off-policy actor–critic, SAC, which already took the smaller of two critics. A second version the same year dropped its separate state-value network, tuned $\alpha$ automatically, and showed it learning to walk on a real quadruped robot in about two hours.

## Card

### Idea

Maximize reward plus entropy: a stochastic actor, trained through its own samples, that stays random wherever it can afford to, with TD3's twin critics, soft targets that include future entropy, and an entropy weight α tuned automatically.

::: analogy
A traveler who takes the best road when one is clearly best, but when several are about equally good, keeps varying the route: it costs almost nothing, and the traveler learns about all of them.
:::

### The update {#update}

$$y = \rew{r} + \gam\Big(\min_i \val{\hat q'_i(s', a')} - \alpha \ln \pol{\pi(a' \mid s')}\Big), \quad a' \sim \pol\pi; \qquad \boldsymbol\theta \leftarrow \boldsymbol\theta - \alp\,\nabla_{\boldsymbol\theta}\Big(\alpha \ln \pol{\pi(a_{\boldsymbol\theta} \mid s)} - \min_i \val{\hat q_i(s, a_{\boldsymbol\theta})}\Big)$$

the critics regress on soft targets; the actor's sampled action $a_{\boldsymbol\theta} = a_{\max}\tanh(\mu + \sigma\varepsilon)$ is differentiated through; $\ln\alpha$ moves toward the target entropy.

### One change from TD3 {#change}

A stochastic actor instead of a deterministic one with added noise, and an entropy bonus in every target, with its weight tuned toward a target entropy.

### Backup diagram {#backup}

{{backup sac}}

One sampled transition, then a next action drawn from the current policy, valued by the smaller target critic minus α times its log-probability: the value of the rewards and the randomness still to come.

### Pseudocode

::: pseudocode
Parameters: memory $N$, batch $B$, step sizes, $\tau$, target entropy $\bar{\mathcal H}$ (default: minus the number of action dimensions), initial $\alpha$
Initialize actor $\boldsymbol\theta$ and critics $\mathbf w_1$, $\mathbf w_2$, their target copies, and $\ln\alpha$; memory $\mathcal D$ empty
Repeat for each step:
  Draw $A \sim \pol{\pi(\cdot \mid S, \boldsymbol\theta)}$; take it, observe $\rew{R}$, $S'$; store it in $\mathcal D$
  Sample $B$ transitions; for each, draw $a'_j \sim \pol{\pi(\cdot \mid s'_j)}$
  $y_j \leftarrow \rew{r_j} + \gam\big(\min_i \val{\hat q(s'_j, a'_j, \mathbf w_i^-)} - \alpha \ln \pol{\pi(a'_j \mid s'_j)}\big)$ (just $\rew{r_j}$ at the end)
  Each critic: a gradient step on $\frac1B \sum_j (y_j - \val{\hat q(s_j, a_j, \mathbf w_i)})^2$
  Actor: draw $a_j = a_{\max}\tanh(\mu(s_j) + \sigma(s_j)\varepsilon_j)$; a gradient step on $\frac1B \sum_j \big(\alpha \ln \pol{\pi(a_j \mid s_j)} - \min_i \val{\hat q(s_j, a_j, \mathbf w_i)}\big)$
  $\ln\alpha \leftarrow \ln\alpha - \lambda\big(-\ln \pol{\pi(a_j \mid s_j)} - \bar{\mathcal H}\big)$, averaged over the batch
  Target critics move by $\tau$ toward the critics
  $S \leftarrow S'$
:::

### Perks

- Explores by itself, from its objective; no noise schedule to tune.
- Robust across seeds: on Pendulum, every seed's last tests within 7 of each other.
- Honest values: twin critics, and along the shown run's tests a critic within 1 of the true return, on average.
- The entropy weight tunes itself.

### Flaws

- More computation per step than TD3: two critics, sampling, and the α update.
- The target entropy is still a choice, and the default does not suit every task.
- Its values are soft values: they include future entropy, so they are not plain returns until α is small.

### Knobs

| Knob | Too low | Too high |
| --- | --- | --- |
| entropy weight $\alpha$ (if fixed) | collapses early, like a greedy policy | stays random, ignores the reward |
| target entropy $\bar{\mathcal H}$ (if tuned) | too decisive too soon | too random for too long |
| target speed $\tau$ | slow targets | moving targets |
| reward scale | the entropy dominates | the entropy vanishes |
| memory, batch, step sizes | as in DDPG | |

### Pitfalls

- Forgetting the tanh correction in $\ln\pol\pi$: the entropy is then wrong near the action limits.
- A fixed α with rewards of a new scale: the balance between reward and entropy moves with the scale.
- Testing with sampled actions: deploy the mean, or judge the stochastic policy for what it is.

### Check yourself {#check}

::: question
How is SAC's entropy term different from the entropy bonus of A2C or PPO?
---
In A2C or PPO the bonus is added to the policy's loss at the states visited, to slow its collapse. In SAC the entropy is part of the objective and of the critic's targets, so the agent values future entropy too, and is drawn to states where it can afford to stay random.
:::

::: question
Why can SAC's actor be trained by differentiating through a sampled action?
---
The action is written as a deterministic function of the parameters and an independent noise: a = a_max tanh(μ + σε). For a fixed ε, the critic's value and the log-probability are differentiable functions of θ, so the gradient passes straight through the sample, with much less variance than a likelihood-ratio estimate.
:::

::: question
Near the top, SAC's final policy has a spread of 0.30; during the swing, 0.11. Why the difference?
---
Near the top, any small torque keeps the pendulum up, so the values of nearby actions are about equal and the entropy bonus keeps the policy spread. During the swing, the torque must push with the spin at full strength; spreading would cost reward. Maximum entropy policies stay random exactly where randomness is cheap.
:::
