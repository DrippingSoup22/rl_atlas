+++
summary = "Policy gradients say which way to go, not how far. TRPO measures each step by how much it changes the policy's behavior, the KL divergence between the old and the new policy, and takes the largest step that keeps it below a bound δ. The step follows the natural gradient, and a line search checks that it really improves. Large steps become safe, and learning stops collapsing."
change = "Instead of a step of fixed size in weight space, maximize the surrogate objective (the importance-weighted advantages of the last batch) subject to an average KL divergence of at most δ between the old and the new policy: a natural-gradient step, checked by a line search."
prereqs = ["actor-critic", "gae", "pg-theorem"]
lab = "trpo-cliff"
sources = [
  { text = "Schulman, Levine, Moritz, Jordan & Abbeel (2015), Trust region policy optimization, ICML", url = "https://arxiv.org/abs/1502.05477" },
  { text = "Kakade (2001), A natural policy gradient, Advances in Neural Information Processing Systems 14" },
  { text = "Kakade & Langford (2002), Approximately optimal approximate reinforcement learning, Proceedings of the 19th International Conference on Machine Learning" },
  { text = "Peters & Schaal (2008), Natural actor-critic, Neurocomputing 71" },
  { text = "Amari (1998), Natural gradient works efficiently in learning, Neural Computation 10" },
  { text = "OpenAI (2018), Spinning Up in Deep RL, Trust Region Policy Optimization", url = "https://spinningup.openai.com/en/latest/algorithms/trpo.html" },
]

[story]
scene = "grid"
env = "cliff"
seed = 1000
average = 12
formula = '\step{1}{\max_{\boldsymbol\theta}\ \hat{\mathbb E}\Big[\frac{\pol{\pi_{\boldsymbol\theta}(A \mid S)}}{\pol{\pi_{\text{old}}(A \mid S)}}\,\err{\hat A}\Big]} \step{2}{\quad \text{subject to} \quad \overline{D}_{\mathrm{KL}}\big(\pol{\pi_{\text{old}}} \,\|\, \pol{\pi_{\boldsymbol\theta}}\big) \le \delta}'

[story.runs]
mid = { algorithm = "trpo", delta = 0.05, lambda = 0.9, alphaW = 0.1, workers = 4, gamma = 1.0, maxSteps = 1000, units = 60, name = "δ = 0.05" }
big = { algorithm = "trpo", delta = 0.2, lambda = 0.9, alphaW = 0.1, workers = 4, gamma = 1.0, maxSteps = 1000, units = 60, name = "δ = 0.2" }
d01 = { algorithm = "trpo", delta = 0.01, lambda = 0.9, alphaW = 0.1, workers = 4, gamma = 1.0, maxSteps = 1000, units = 60, name = "δ = 0.01" }
+++

## Story

::: step {run = "mid", at = 0}
**How far to step?** The cliff once more, with the batch learner of [[gae]]: four workers play a round of episodes, and then the policy takes one step, guided by GAE advantages ($\lam = 0.9$). The advantages say which way to move. They do not say how far, and in weight space the same distance can change the policy a little or completely.
:::

::: step {run = "mid", at = 1, advantages = true, arrows = false, trail = false, formula = 1}
Round 1, with a random policy: two workers wander for the full 1000 steps, two reach the gem, and 253 times someone falls. From these four episodes TRPO builds a **surrogate**: how much better the new policy would do, estimated from the old episodes by reweighting each move's advantage by how much more, or less, likely the new policy makes it.
:::

::: step {run = "mid", at = 1, formula = 2}
The surrogate is only trustworthy near the old policy, which collected the data. So TRPO limits the step by its effect on **behavior**: the KL divergence between the old and the new policy, averaged over the states of the round, may not exceed $\delta = 0.05$. The step goes to the edge of this trust region, KL = 0.046, and the moves into the cliff are already less likely: right from the start, for one, from 25% to 14%.
:::

::: step {run = "mid", at = 3}
Round 2 has 96 falls, round 3 only 9. Every round the step goes as far as the trust region allows, unless a line search finds that the full step does not really improve the surrogate, and halves it; that happens for the first time in round 4.
:::

::: step {run = "mid", at = 60, values = true}
After 60 rounds: 15 steps per episode, along the second row above the cliff, and a critic that knows it.
:::

::: step {run = "big", at = 1, formula = 2}
The same first round with a trust region four times larger, $\delta = 0.2$. The same advantages, a longer step: KL = 0.17. At first that pays: round 2 has only 20 falls, against 96.
:::

::: step {run = "big", at = 20}
But each round's four episodes are a noisy sample, and a large step trusts them too far. From round 5 on, the policy circles in the top-left corner, bumping against its walls, until each episode is cut off at 1000 steps. It never leaves: no tile there sends it right any more, the way out, and a policy can only learn from the moves it still makes. Not every run is caught: over 40 seeds, 11 runs with $\delta = 0.2$ end stuck like this, against 1 with $\delta = 0.05$ and none with $\delta = 0.01$. The Lab's sweep of $\delta$ draws the whole curve.
:::

::: step {run = "mid", at = 60, curves = ["d01", "mid", "big"], metric = "steps"}
Averaged over 12 runs. $\delta = 0.01$ learns steadily; $\delta = 0.05$ learns fastest, 31 steps per episode after 10 rounds; both end near 16. With $\delta = 0.2$, two runs of 12 get stuck like this one, and the average stays at 181. [Size the trust region in the Lab](lab:trpo-cliff).
:::

## Textbook

### How far to step {#problem}

A policy-gradient update moves the weights a distance set by the step size along an estimate of the gradient. Two things make the right distance hard to choose. The same distance in weight space can mean a tiny change of behavior or a drastic one, depending on where the weights are: near a deterministic softmax policy, a small step can flip which action is chosen. And a bad step is hard to undo, because an on-policy method learns from the data its policy produces: a policy that collapses, into a corner of the cliff for example, produces data that cannot teach it how to get out. Trust region policy optimization (TRPO; Schulman, Levine, Moritz, Jordan and Abbeel, 2015) measures steps by their effect on the policy, and limits them.

### The surrogate objective {#surrogate}

How much better is a new policy $\pol{\pi'}$ than the current one $\pol{\pi}$? Exactly, the answer averages the current policy's advantages over the new policy's own experience:

::: lemma {#lem-pdl} Performance difference (Kakade and Langford, 2002)
For any two policies, with $\eta_{\pi'}$ the expected discounted number of visits to each state under $\pol{\pi'}$ ([[pg-theorem]]),

$$J(\pol{\pi'}) - J(\pol{\pi}) = \sum_s \eta_{\pi'}(s) \sum_a \pol{\pi'(a \mid s)}\,\err{a_\pi(s, a)}. \label{eq-pdl}$$
:::

::: proof
Write $\err{a_\pi(S_t, A_t)} = \mathbb E[\rew{R_{t+1}} + \gam\,\val{v_\pi(S_{t+1})} - \val{v_\pi(S_t)} \mid S_t, A_t]$ and add these terms up along an episode of $\pol{\pi'}$, discounted: the values telescope, leaving the discounted return of $\pol{\pi'}$ minus $\val{v_\pi(s_0)}$. Taking expectations gives $J(\pol{\pi'}) - J(\pol{\pi})$ on one side and the double sum on the other.
:::

The visits $\eta_{\pi'}$ of the new policy are unknown before it is tried. Replacing them by the current policy's visits gives the **surrogate**, which the data of the current policy can estimate, reweighting each sampled action by the ratio of its new and old probabilities:

$$L_\pi(\pol{\pi'}) = \sum_s \eta_\pi(s) \sum_a \pol{\pi'(a \mid s)}\,\err{a_\pi(s, a)} = \mathbb E_\pi\Big[\frac{\pol{\pi'(A_t \mid S_t)}}{\pol{\pi(A_t \mid S_t)}}\,\err{a_\pi(S_t, A_t)}\Big]. \label{eq-surrogate}$$

At $\pol{\pi'} = \pol{\pi}$ the surrogate is 0, like the true change, and its gradient there is the policy gradient. Away from $\pol{\pi}$ the two part ways, the more so the more the visits change: the surrogate can be trusted only near the policy that collected the data.

::: theorem {#thm-bound} Monotonic improvement (Schulman et al., 2015)
With $\epsilon = \max_{s, a} |\err{a_\pi(s, a)}|$ and a discount $\gam < 1$,

$$J(\pol{\pi'}) - J(\pol{\pi}) \ge L_\pi(\pol{\pi'}) - \frac{4\,\epsilon\,\gam}{(1-\gam)^2}\,\max_s D_{\mathrm{KL}}\big(\pol{\pi(\cdot \mid s)} \,\|\, \pol{\pi'(\cdot \mid s)}\big).$$
:::

Maximizing the right side at each step can only improve $J$, since the bound is tight at $\pol{\pi'} = \pol{\pi}$. Its penalty coefficient is so large that the steps would be tiny; TRPO keeps the idea but replaces the penalty by a constraint on the **average** KL divergence, with a bound $\delta$ chosen by hand:

$$\max_{\boldsymbol\theta}\ L_{\pi_{\text{old}}}(\pol{\pi_{\boldsymbol\theta}}) \quad \text{subject to} \quad \overline{D}_{\mathrm{KL}}(\boldsymbol\theta) = \mathbb E_{\pi_{\text{old}}}\Big[D_{\mathrm{KL}}\big(\pol{\pi_{\text{old}}(\cdot \mid S)} \,\|\, \pol{\pi_{\boldsymbol\theta}(\cdot \mid S)}\big)\Big] \le \delta. \label{eq-trpo}$$

::: figure {#fig-region}
{{trust-region}}
On the short corridor, from the policy that steps right with probability 20%: the true change of the objective for every new policy, and the surrogate, worked out exactly. The surrogate is a straight line here and would jump to an extreme; the trust region, KL ≤ 0.02, keeps the step between 12.8% and 28.7%, where the surrogate is accurate. TRPO steps to its edge, from $\rew{-22.5}$ to $\rew{-16.7}$.
:::

### The natural gradient {#natural}

Near $\boldsymbol\theta_{\text{old}}$ the constrained problem \ref{eq-trpo} simplifies: the surrogate is linear to first order, with gradient $\mathbf g$, the policy gradient, and the average KL divergence is quadratic to second order,

$$\overline{D}_{\mathrm{KL}}(\boldsymbol\theta_{\text{old}} + \boldsymbol\Delta) \approx \tfrac12\,\boldsymbol\Delta^\top \mathbf F\,\boldsymbol\Delta, \qquad \mathbf F = \mathbb E_\pi\big[\nabla \ln \pol{\pi(A \mid S)}\,\nabla \ln \pol{\pi(A \mid S)}^\top\big], \label{eq-fisher}$$

where $\mathbf F$, the **Fisher information matrix**, measures how fast the policy's distributions change in each direction of weight space. Maximizing $\mathbf g^\top \boldsymbol\Delta$ under $\tfrac12\,\boldsymbol\Delta^\top \mathbf F\,\boldsymbol\Delta \le \delta$ gives

$$\boldsymbol\Delta = \sqrt{\frac{2\delta}{\mathbf g^\top \mathbf F^{-1} \mathbf g}}\ \mathbf F^{-1}\mathbf g. \label{eq-step}$$

The direction $\mathbf F^{-1}\mathbf g$ is the **natural gradient** (Amari, 1998; Kakade, 2001): the steepest ascent when distances are measured between policies rather than between weight vectors. It does not depend on how the policy is parameterized, and it takes long steps where the policy is insensitive to its weights and short ones where it is sensitive, the correction a Gaussian policy's $1/\pol{\sigma}^2$ needs ([[policy-parameterization]]).

For a neural network, $\mathbf F$ has as many rows as there are weights and is never formed. TRPO solves $\mathbf F\,\mathbf x = \mathbf g$ by the conjugate gradient method, which needs only products $\mathbf F\,\mathbf v$, computable from the gradient of the KL divergence, in about ten iterations. Because the quadratic model is only approximate, a **backtracking line search** then shrinks the step, halving it, until the true average KL is within $\delta$ and the surrogate has actually improved.

### The algorithm {#algorithm}

::: algorithm {#alg-trpo} TRPO
Input: a differentiable policy $\pol{\pi(a \mid s, \boldsymbol\theta)}$ and value estimate $\val{\hat v(s, \mathbf w)}$, a trust region $\delta$, GAE's $\gam$ and $\lam$
Repeat for each iteration:
  Collect a batch of episodes (or of $N \times T$ steps) with the current policy $\pol{\pi_{\text{old}}}$
  Compute advantages $\err{\hat A_t}$ with GAE, and the policy gradient $\mathbf g$ of the surrogate
  Solve $\mathbf F\,\mathbf x = \mathbf g$ approximately by conjugate gradient; full step $\boldsymbol\Delta = \sqrt{2\delta / \mathbf x^\top \mathbf F\,\mathbf x}\ \mathbf x$
  Line search: for $j = 0, 1, 2, \dots$, try $\boldsymbol\theta_{\text{old}} + 2^{-j}\boldsymbol\Delta$; accept the first with $\overline{D}_{\mathrm{KL}} \le \delta$ and an improved surrogate
  Fit the critic $\val{\hat v}$ to the batch's returns (here: λ-returns)
:::

The Lab's TRPO uses a softmax with one feature per state, where the natural gradient has a closed form: in each state, the step for each action is its estimated advantage, the batch's $\err{\hat A}/\pol{\pi(a \mid s)}$ for that action averaged over the state's visits, and $\mathbf g^\top \mathbf F^{-1} \mathbf g$ is the variance of these steps under the policy, weighted by visits. No conjugate gradient is needed; the line search is as above.

### The size of the trust region {#delta}

$\delta$ replaces the step size, and is easier to set: it is measured in nats of behavior, independent of the parameterization and of the reward scale. Typical values are around 0.01. Too small and learning is slow; too large and the step trusts a noisy batch too far, with the same risk of collapse as a large step size, only rarer. On the cliff, with four episodes per batch, $\delta = 0.05$ learns fastest; with $\delta = 0.2$, two runs of twelve step into the top-left corner within five rounds and never come out, because a policy that no longer tries the way out cannot learn that it was one. With networks, on CartPole, trained from 20 seeds per value: every $\delta$ from 0.003 to 0.03 balances the pole in all 20 runs, and 0.1 in 19; at $\delta = 0.001$ two runs of 20 are still short of it after 100,000 steps; at $\delta = 0.3$ three fall short, and at $\delta = 1$ thirteen do ([the run and its sweep](lab:trpo-cartpole)).

### Historical remarks {#history}

The natural gradient is due to Amari (1998); Kakade (2001) brought it to policy gradients, and Peters and Schaal (2008) built the natural actor–critic on it. Kakade and Langford (2002) proved the performance difference lemma and used it in conservative policy iteration, which mixes old and new policies to guarantee improvement. Schulman, Levine, Moritz, Jordan and Abbeel (2015) turned that guarantee into a practical algorithm for neural networks, with the average-KL constraint, conjugate gradient and line search, and showed it learning locomotion and Atari games; with GAE (Schulman et al., 2016) it learned 3D humanoid running. Its complexity led to [[ppo|PPO]], which keeps the trust region with first-order tools.

## Card

### Idea

Policy gradients give a direction, not a distance. TRPO takes the largest step that keeps the new policy close to the old one in behavior, average KL divergence at most $\delta$, along the natural gradient, and checks with a line search that the step really helps. Monotonic improvement in theory, rare collapses in practice.

::: analogy
A hiker in fog with a map of the slope that is only accurate nearby. Instead of steps of a fixed length, they step as far as the map is reliable, and check the ground before putting their weight down.
:::

### The update {#update}

$$\max_{\boldsymbol\theta}\ \hat{\mathbb E}\Big[\frac{\pol{\pi_{\boldsymbol\theta}(A \mid S)}}{\pol{\pi_{\text{old}}(A \mid S)}}\,\err{\hat A}\Big] \quad \text{subject to} \quad \overline{D}_{\mathrm{KL}}\big(\pol{\pi_{\text{old}}} \,\|\, \pol{\pi_{\boldsymbol\theta}}\big) \le \delta$$

solved approximately by $\boldsymbol\Delta = \sqrt{2\delta / \mathbf g^\top \mathbf F^{-1} \mathbf g}\ \mathbf F^{-1}\mathbf g$ and a line search.

### One change from the actor–critic {#change}

| | step | data |
| --- | --- | --- |
| [[actor-critic]] | fixed size in weight space, each sample | one step at a time |
| TRPO | as large as KL ≤ δ allows, natural gradient, checked | a batch of episodes, GAE advantages |

### Backup diagram {#backup}

{{backup gae}}

Each step of the batch is judged by GAE: its TD errors and those after it, fading by $\gam\lam$.

### Pseudocode

::: pseudocode
Parameters: trust region $\delta$, GAE's $\gam$ and $\lam$, critic step size $\alp^{\mathbf w}$, $N$ workers
Set $\boldsymbol\theta$ (here, all moves equally likely) and $\mathbf w$ (here 0)
Repeat for each round:
  Every worker starts an episode {#start}
  Each worker takes a step with $\pol{\pi_{\text{old}}}$, all at once, until all episodes end {#act}
  GAE advantages $\err{\hat A_t}$ for every step of the round {#advantage}
  Natural-gradient step to the edge of $\overline{D}_{\mathrm{KL}} \le \delta$, halved until the KL fits and the surrogate improves {#update}
  Fit the critic to the λ-returns $\err{\hat A_t} + \val{\hat v(S_t, \mathbf w)}$ {#critic}
:::

### Perks

- Steps sized by behavior: robust to the parameterization and the reward scale.
- Large, safe steps, with a guarantee of improvement in theory. [See it](lab:trpo-cliff)
- Learned complex neural-network policies that earlier methods could not.

### Flaws

- Complex: conjugate gradient, Fisher-vector products, a line search.
- One step per batch: the data is still used once.
- Awkward with shared actor–critic networks and with noise like dropout.

### Knobs

| Knob | Too low | Too high |
| --- | --- | --- |
| $\delta$ trust region | slow | trusts noisy batches too far; can collapse |
| batch size | noisy advantages, wasted steps | slow iterations |
| $\lam$ (GAE) | biased by the critic | noisy |

### Pitfalls

- Reading $\delta$ as a step size: it bounds the change of behavior, averaged over states; single states can change much more.
- Skipping the line search: the quadratic model of the KL is only approximate.
- Measuring the KL in the wrong direction or on the wrong states: it is the old policy's states and actions that the surrogate trusts.

### Check yourself {#check}

::: question
Why can't a policy-gradient method just take larger steps to learn faster?
---
A step of fixed size in weight space can change the policy drastically, and the surrogate estimated from the old policy's data is only accurate near that policy. A collapsed policy then produces data that cannot repair it.
:::

::: question
What does the Fisher information matrix measure here?
---
How fast the policy's action distributions change as the weights move in each direction: $\tfrac12\,\boldsymbol\Delta^\top \mathbf F\,\boldsymbol\Delta$ approximates the average KL divergence of a step $\boldsymbol\Delta$.
:::

::: question
Why is the surrogate only trustworthy near the old policy?
---
It uses the old policy's state visits in place of the new policy's. The further the new policy moves, the more its visits differ, and the more the surrogate departs from the true change of performance.
:::
