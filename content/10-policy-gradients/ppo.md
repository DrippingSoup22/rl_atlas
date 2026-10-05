+++
summary = "TRPO's idea with first-order tools: reuse each batch for several passes of plain gradient steps, but stop pushing any action whose probability has already moved more than a factor 1 ± ε from the policy that collected the data. Simple, robust and sample-efficient enough to become the default policy-gradient method, from robots and games to fine-tuning language models."
change = "Replace TRPO's KL constraint and second-order step by a clipped surrogate objective, maximized with several passes of first-order gradient steps over each batch: a sample stops pushing its action once its probability ratio leaves 1 ± ε."
prereqs = ["trpo", "gae", "a2c", "entropy-bonus"]
lab = "ppo-maze"
sources = [
  { text = "Schulman, Wolski, Dhariwal, Radford & Klimov (2017), Proximal policy optimization algorithms", url = "https://arxiv.org/abs/1707.06347" },
  { text = "Schulman, Levine, Moritz, Jordan & Abbeel (2015), Trust region policy optimization, ICML", url = "https://arxiv.org/abs/1502.05477" },
  { text = "Engstrom, Ilyas, Santurkar, Tsipras, Janoos, Rudolph & Madry (2020), Implementation matters in deep policy gradients: a case study on PPO and TRPO, ICLR", url = "https://arxiv.org/abs/2005.12729" },
  { text = "Andrychowicz et al. (2021), What matters in on-policy reinforcement learning? A large-scale empirical study, ICLR", url = "https://arxiv.org/abs/2006.05990" },
  { text = "Ouyang et al. (2022), Training language models to follow instructions with human feedback, NeurIPS", url = "https://arxiv.org/abs/2203.02155" },
  { text = "OpenAI (2018), Spinning Up in Deep RL, Proximal Policy Optimization", url = "https://spinningup.openai.com/en/latest/algorithms/ppo.html" },
]

[story]
scene = "grid"
env = "dyna-maze"
seed = 1008
average = 12
formula = '\step{1}{L^{\text{CLIP}} = \hat{\mathbb E}\Big[\min\big(r\,\err{\hat A},\ \operatorname{clip}(r, 1 - \epsilon, 1 + \epsilon)\,\err{\hat A}\big)\Big] \qquad} \step{2}{r = \frac{\pol{\pi_{\boldsymbol\theta}(A \mid S)}}{\pol{\pi_{\text{old}}(A \mid S)}}}'

[story.numbers]
clip = '\text{pass } 1: 7\%\ \text{of moves clipped},\ \overline{D}_{\mathrm{KL}} = 0.032 \qquad \text{pass } 10: 32\%,\ \overline{D}_{\mathrm{KL}} = 0.051'
free = '\text{pass } 1: \overline{D}_{\mathrm{KL}} = 0.11 \qquad \text{pass } 5: 0.60 \qquad \text{pass } 10: 1.08'

[story.runs]
clip = { algorithm = "ppo", clip = 0.2, alpha = 0.3, alphaW = 0.1, workers = 4, epochs = 10, lambda = 0.9, beta = 0.0, gamma = 0.95, maxSteps = 1000, units = 40, name = "PPO, clip ε = 0.2" }
free = { algorithm = "ppo", clip = 0.0, alpha = 0.3, alphaW = 0.1, workers = 4, epochs = 10, lambda = 0.9, beta = 0.0, gamma = 0.95, maxSteps = 1000, units = 40, name = "the same passes, no clip" }
+++

## Story

::: step {run = "clip", at = 0}
**PPO** in the maze of [[a2c]], with four workers per round. [[trpo|TRPO]] keeps each step safe with a constraint and second-order mathematics. PPO wants the same safety from plain gradient steps, and more: it squeezes each round's experience with **ten passes** over it instead of one.
:::

::: step {run = "clip", at = 1, formula = 2}
Round 1: episodes of 903, 726, 344 and 100 steps, 2073 moves in all, judged by GAE advantages ([[gae]]). In each pass, every move nudges the policy: toward its action if its advantage is positive, away if negative. The **ratio** $r$ tracks how much more, or less, likely the policy now makes each move than when it was collected.
:::

::: step {run = "clip", at = 1, formula = 1, numbers = "clip"}
The clip: once a move's ratio has left the band $1 \pm \epsilon$, $\epsilon = 0.2$, in the direction its advantage favors, that move stops pushing. In the first pass 7% of the moves are already past the band and are left alone; by the tenth, 32%. After ten passes the policy is 0.051 (in average KL) from the one that collected the data: a trust region, without the second-order mathematics.
:::

::: step {run = "free", at = 1, formula = 1, numbers = "free"}
The same round and the same ten passes, **without the clip**. Every pass pushes the same lucky moves further, and the policy drifts away from the one the advantages were measured for: KL 0.11 after one pass, 1.08 after ten, twenty times further than with the clip.
:::

::: step {run = "free", at = 2}
Round 2 is worse, not better: three of the four episodes run 860 steps or more. In its first pass the policy leaps to KL 18 from the one that collected the round. From round 3 on, every episode is cut off at 1000 steps.
:::

::: step {run = "free", at = 40, values = true}
After 40 rounds, the four workers bounce between the start and the tile to its right, 500 times each way: right from the start, left from the next tile, both practically certain. The policy overfit one batch and collapsed into a loop it cannot learn its way out of. Not every run without the clip ends this way: over 40 seeds, 6 collapsed, against 1 with the clip. The drift behind it is the rule, though: on average the first round moves the policy 13 times further without the clip, KL 0.88 against 0.07.
:::

::: step {run = "clip", at = 40, values = true}
With the clip, the same 40 rounds end at 15 steps per episode; the shortest path has 14.
:::

::: step {run = "clip", at = 40, curves = ["clip", "free"], metric = "steps"}
Averaged over 12 runs: with the clip, 21 steps per episode after 10 rounds, and every run ends between 14 and 17. Without it, learning is slower and less reliable: 99 steps per episode after 10 rounds, and the run above never recovers. [Race them in the Lab](lab:ppo-maze).
:::

## Textbook

### From TRPO to PPO {#why}

[[trpo|TRPO]] makes large policy steps safe, but at a price: conjugate gradients, Fisher-vector products and a line search, one step per batch, and trouble with networks that share layers between actor and critic or use noise such as dropout. Proximal policy optimization (PPO; Schulman, Wolski, Dhariwal, Radford and Klimov, 2017) keeps the goal, staying close to the policy that collected the data, and reaches it with an objective that ordinary stochastic gradient ascent can optimize, several times over the same batch.

### The clipped surrogate {#clip}

Write $r_t(\boldsymbol\theta)$ for the **probability ratio** of a sampled move under the new and the old policy,

$$r_t(\boldsymbol\theta) = \frac{\pol{\pi_{\boldsymbol\theta}(A_t \mid S_t)}}{\pol{\pi_{\text{old}}(A_t \mid S_t)}}, \label{eq-ratio}$$

so that TRPO's surrogate is $L(\boldsymbol\theta) = \hat{\mathbb E}[r_t(\boldsymbol\theta)\,\err{\hat A_t}]$ ([[trpo]]). Maximized without a constraint, it rewards pushing $r_t$ as far as possible: up without limit for positive advantages, toward zero for negative ones. PPO clips it:

$$L^{\text{CLIP}}(\boldsymbol\theta) = \hat{\mathbb E}\Big[\min\big(r_t(\boldsymbol\theta)\,\err{\hat A_t},\ \operatorname{clip}\big(r_t(\boldsymbol\theta), 1 - \epsilon, 1 + \epsilon\big)\,\err{\hat A_t}\big)\Big], \label{eq-clip}$$

with $\epsilon$ around 0.1 to 0.3. Per sample the effect is simple. For a positive advantage, the objective grows with $r_t$ up to $1 + \epsilon$ and is flat beyond: once the move has become $1 + \epsilon$ times as likely, it stops pulling. For a negative advantage, the objective is flat below $1 - \epsilon$. The minimum makes the clipped objective a **pessimistic bound** on the unclipped one: a change that went too far in the favored direction earns nothing more, while a change that made things worse is counted in full, so the update is never rewarded for moving far from the old policy.

::: figure {#fig-clip}
{{ppo-clip}}
The clipped objective of one sample as a function of its ratio, for a positive (left) and a negative (right) advantage. Where it is flat, its gradient is zero and the sample no longer pushes.
:::

::: figure {#fig-region}
{{trust-region clip}}
The short corridor of the TRPO entry, from the policy that steps right 20% of the time: the true change of the objective, TRPO's surrogate, and PPO's clipped surrogate, worked out exactly. The clipped surrogate peaks at the edge of the band, $p = 24\%$, an improvement from $\rew{-22.5}$ to $\rew{-19.3}$; beyond it, its gains are capped while its losses are not.
:::

The clip limits the ratio of the moves in the batch, not the policy as a whole; states and actions outside the batch can still move. The average KL divergence after an update is therefore not bounded, only kept small in practice, which is why many implementations also stop the passes early when it exceeds a threshold.

### The algorithm {#algorithm}

PPO collects a batch from $N$ parallel actors, each running $T$ steps, as [[a2c|A2C]] does; computes GAE advantages ([[gae]]), usually **normalized** to mean 0 and standard deviation 1 within the batch; then optimizes $L^{\text{CLIP}}$ for $K$ **epochs**, each a pass over the batch in shuffled minibatches, with Adam. With a shared actor–critic network, the loss combines three terms,

$$L(\boldsymbol\theta) = \hat{\mathbb E}\Big[L^{\text{CLIP}}_t(\boldsymbol\theta) - c_1\,\big(\val{\hat v_{\boldsymbol\theta}(S_t)} - \hat G_t\big)^2 + c_2\,\pol{H\big(\pi_{\boldsymbol\theta}(\cdot \mid S_t)\big)}\Big], \label{eq-loss}$$

the clipped surrogate, a value loss toward the targets $\hat G_t$ (λ-returns), and an [[entropy-bonus|entropy bonus]], maximized together.

::: algorithm {#alg-ppo} PPO, clipped version (actor–critic style)
Input: a policy and value network, $N$ actors, $T$ steps per actor, $K$ epochs, minibatch size $M$, clip range $\epsilon$, GAE's $\gam$ and $\lam$
Repeat for each iteration:
  Each actor runs $\pol{\pi_{\text{old}}}$ for $T$ steps
  Compute GAE advantages $\err{\hat A_1}, \dots, \err{\hat A_{NT}}$ and value targets; normalize the advantages
  For $K$ epochs, for each minibatch of size $M$ of the $NT$ samples: a gradient step on \ref{eq-loss}
  $\pol{\pi_{\text{old}}} \leftarrow \pol{\pi_{\boldsymbol\theta}}$
:::

Common settings: $\epsilon = 0.2$, $K$ from 3 to 10 epochs, $\lam = 0.95$, $\gam = 0.99$, $c_1 = 0.5$ to 1 and $c_2 = 0.01$ for discrete actions. The Lab's version is the same with tabular features: every round, four workers play an episode each; the advantages are normalized; each of the $K$ passes visits the samples in a new random order, and each sample whose ratio is still inside its band takes one small gradient step of size $\alp$; the critic is then fitted to the λ-returns.

### Another way: a KL penalty {#penalty}

The PPO paper also tried a penalty instead of the clip: maximize $\hat{\mathbb E}[r_t\,\err{\hat A_t}] - \beta\,\overline{D}_{\mathrm{KL}}$, and after each iteration double $\beta$ if the KL exceeded 1.5 times a target $d$, halve it if it fell below $d / 1.5$. It worked, but less well than clipping on their benchmarks, and the clipped version became the standard.

### What else matters {#details}

PPO's results depend on more than the clip. Careful studies (Engstrom et al., 2020; Andrychowicz et al., 2021) found that several implementation choices, often unmentioned, matter as much: normalizing advantages and observations, clipping the value loss and the gradient norm, annealing the learning rate, the initialization of the last layers, and the number of epochs and minibatches. Without them PPO's advantage over TRPO largely disappears. The lesson for practice: start from a reference implementation, change one thing at a time, and watch the KL divergence and the share of clipped samples.

### With networks, on Pendulum {#pendulum}

The same knobs matter just as much with networks. On Gymnasium's Pendulum, with networks of 64 + 64 units, four workers and 200,000 steps, each setting was trained from five seeds, everything else as in the recorded run. Without the clip, none of the five learns to swing the pendulum up: its last training episodes still return between $-1220$ and $-1605$, against $-166$ to $-234$ with $\epsilon = 0.2$. The clip has a sweet spot: with $\epsilon = 0.1$ all five learn too, with $0.3$ the returns sink to between $-237$ and $-404$, and with $0.5$ all five end between $-622$ and $-1547$. Passes behave the same way: 4 or 10 per batch work, while 30 overfit each batch and end between $-247$ and $-780$. [The sweeps, in the Lab](lab:ppo-pendulum).

### Example: ten passes in a maze {#example}

In the Dyna maze ([[dyna-q]]) with four workers, $\gam = 0.95$, GAE with $\lam = 0.9$, $K = 10$ passes and a step size of 0.3 per sample, PPO with $\epsilon = 0.2$ takes the average episode from about 630 steps in the first round to 21 after 10 rounds and 15 after 40, with every one of 12 runs ending between 14 and 17. With the same ten passes and no clip, the policy moves an average KL of about 0.9 from the old one in the first round alone, learning is slower, and one run of twelve collapses into a deterministic loop it never leaves.

### Historical remarks {#history}

PPO was introduced by Schulman, Wolski, Dhariwal, Radford and Klimov (2017), who compared the clipped and the penalized objectives with TRPO and A2C on continuous control and Atari. Its simplicity made it the default policy-gradient method: it trained OpenAI Five to play Dota 2 and robots in simulation, and it is the reinforcement-learning step in the fine-tuning of language models from human feedback (Ouyang et al., 2022; [[rlhf]]). Engstrom and colleagues (2020) and Andrychowicz and colleagues (2021) showed how much of its performance rests on implementation details.

## Card

### Idea

Reuse each batch for several passes of plain gradient steps on a clipped surrogate: a sample pushes its action only until its probability has changed by a factor $1 \pm \epsilon$ from the policy that collected it. TRPO's safety, with the tools of supervised learning.

::: analogy
Revising an essay with a rule: you may rework every paragraph as often as you like, but no paragraph may drift more than a little from the draft your reviewers read, since their comments only apply to that draft.
:::

### The update {#update}

$$L^{\text{CLIP}}(\boldsymbol\theta) = \hat{\mathbb E}\Big[\min\big(r_t\,\err{\hat A_t},\ \operatorname{clip}(r_t, 1 - \epsilon, 1 + \epsilon)\,\err{\hat A_t}\big)\Big], \qquad r_t = \frac{\pol{\pi_{\boldsymbol\theta}(A_t \mid S_t)}}{\pol{\pi_{\text{old}}(A_t \mid S_t)}}$$

maximized by $K$ epochs of minibatch gradient ascent on each batch.

### One change from TRPO {#change}

| | keeps the step small with | optimizer | passes per batch |
| --- | --- | --- | --- |
| [[trpo]] | a KL constraint, $\overline{D}_{\mathrm{KL}} \le \delta$ | natural gradient, line search | one |
| PPO | the clipped ratio, $1 \pm \epsilon$ | plain gradient steps (Adam) | several |

### Backup diagram {#backup}

{{backup gae}}

As in TRPO, each move of the batch is judged by its GAE advantage.

### Pseudocode

::: pseudocode
Parameters: clip range $\epsilon$, $K$ passes, step size $\alp$, GAE's $\gam$ and $\lam$, $N$ workers
Set $\boldsymbol\theta$ (here, all moves equally likely) and $\mathbf w$ (here 0)
Repeat for each round:
  Every worker starts an episode {#start}
  Each worker takes a step with $\pol{\pi_{\text{old}}}$, all at once, until all episodes end {#act}
  GAE advantages $\err{\hat A_t}$ for every move, normalized to mean 0 and spread 1 {#advantage}
  $K$ passes in random order: each move with $r_t$ still inside its band takes a step along $\err{\hat A_t}\,\nabla r_t$ {#update}
  Fit the critic to the λ-returns {#critic}
:::

### Perks

- Simple: first-order steps, a few lines on top of A2C.
- Sample-efficient for an on-policy method: several passes per batch. [See the clip at work](lab:ppo-maze)
- Robust across tasks with standard settings; works with shared networks.

### Flaws

- Still on-policy: each batch is discarded after its passes.
- The KL is not bounded, only kept small; too many passes can still overfit a batch.
- Sensitive to implementation details that papers often leave out.

### Knobs

| Knob | Too low | Too high |
| --- | --- | --- |
| $\epsilon$ clip range | timid steps | the clip stops protecting |
| $K$ passes | wastes the batch | overfits the batch, KL grows |
| $\alp$ step size | slow | many samples clipped at once, unstable |
| $\lam$ (GAE) | biased by the critic | noisy |
| $c_2$ entropy weight | commits early | stays random |

### Pitfalls

- Too many epochs or too large a step size: watch the KL and the share of clipped samples, and stop early when the KL grows.
- Forgetting to normalize advantages: the step size then depends on the reward scale.
- Comparing with other methods without matching the implementation details.

### Check yourself {#check}

::: question
A sample has a positive advantage and its ratio has reached 1.3 with $\epsilon = 0.2$. Does it still push its action?
---
No: for a positive advantage the clipped objective is flat above $1 + \epsilon = 1.2$, so its gradient is zero there.
:::

::: question
Why take the minimum of the clipped and unclipped terms?
---
It makes the objective a pessimistic bound: changes that went too far in the favored direction earn nothing more, while changes that made the sample worse still count in full.
:::

::: question
Without the clip, what goes wrong with ten passes over one batch?
---
Each pass pushes the same sampled moves further, far beyond the region where the advantages, measured under the old policy, still apply: the policy overfits the batch and can collapse.
:::
