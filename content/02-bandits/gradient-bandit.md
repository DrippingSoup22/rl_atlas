+++
summary = "Learn a preference for each arm instead of its value, and pull arms at random with the softmax of the preferences: a reward better than usual makes its arm more likely."
change = "Learn a preference for each arm and choose with the softmax of the preferences, instead of estimating each arm's value and picking the best: a reward above the running average makes its arm more likely, and every other arm less likely."
prereqs = ["epsilon-greedy", "step-size"]
lab = "bandit-gradient"
sources = [
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., §2.8, §2.10 and Figures 2.5 and 2.6", url = "http://incompleteideas.net/book/the-book-2nd.html" },
  { text = "Sutton (1984), Temporal credit assignment in reinforcement learning, PhD thesis, University of Massachusetts Amherst" },
  { text = "Williams (1992), Simple statistical gradient-following algorithms for connectionist reinforcement learning, Machine Learning 8", url = "https://doi.org/10.1007/BF00992696" },
  { text = "Greensmith, Bartlett & Baxter (2004), Variance reduction techniques for gradient estimates in reinforcement learning, Journal of Machine Learning Research 5", url = "https://www.jmlr.org/papers/v5/greensmith04a.html" },
]

[story]
scene = "bandit"
env = "testbed-4"
seed = 93
units = 1000
average = 200
formula = '\step{1}{\pol{\pi(a)} = \frac{e^{\pol{H(a)}}}{\sum_b e^{\pol{H(b)}}} \qquad} \step{2}{\pol{H(A)} \leftarrow \pol{H(A)} + \alp\,\big(\rew{R} - \rew{\bar{R}}\big)\big(1 - \pol{\pi(A)}\big)}'
numbers = '\pol{H(4)} \leftarrow 0 + 0.1\,\big(\rew{5.23} - \rew{3.13}\big)\big(1 - \pol{0.10}\big) = \pol{0.19}'

[story.runs]
baseline = { algorithm = "gradient-bandit", alpha = 0.1, baseline = true, name = "with baseline" }
nobase = { algorithm = "gradient-bandit", alpha = 0.1, baseline = false, name = "without baseline" }
+++

## Story

::: step {run = "baseline", at = 0, truth = false, formula = 1}
**New machines**: these all pay around $\rew{+4}$. And a new kind of player. It keeps no estimates of value at all. Instead it has a **preference** $\pol{H(a)}$ for each arm, and pulls arms *at random*, with probabilities given by the softmax of the preferences. All preferences start at 0, so every arm has a 10% chance: the purple bars.
:::

::: step {run = "baseline", at = 0, truth = false, play = 4, pace = 900, numbers = true, formula = 2}
After each pull it compares the reward with the average reward so far, the **baseline** $\rew{\bar{R}}$ (the dashed line). Better than usual: the pulled arm's preference goes up and every other arm's goes down. Worse than usual: the opposite. On pull 2, arm 4 pays $\rew{5.23}$, well above the baseline of $\rew{3.13}$: its preference rises to 0.19, and the other nine drop by 0.02 each.
:::

::: step {run = "baseline", at = 50, truth = false}
After 50 pulls, arm 5's chance of being pulled has grown to 41%. The policy is still random, but it leans.
:::

::: step {run = "baseline", at = 200, truth = true, focus = 5}
After 200 pulls it has made up its mind: a 98% chance of pulling arm 5. The true payouts confirm that arm 5, at $\rew{5.90}$, is the best.
:::

::: step {run = "nobase", at = 100, truth = true}
The same player **without a baseline** compares every reward with 0. Here every reward is around $\rew{+4}$, so every pull looks like a success and raises the preference of whatever arm was pulled, good or not. After 100 pulls its policy is spread over arms 2, 9 and 6, by luck more than by merit.
:::

::: step {run = "nobase", at = 1000, truth = true, focus = 2}
After 1000 pulls it has locked onto arm 2, worth $\rew{4.53}$, not the best arm. The more an arm is pulled, the more it is reinforced, whatever it pays.
:::

::: step {run = "baseline", at = 1000, truth = true, curves = ["baseline", "nobase"], metric = "optimal"}
Averaged over 200 problems, the baseline makes all the difference here. It does not change the direction the preferences move on average, only how noisy their path is. [Race them in the Lab](lab:bandit-gradient).
:::

## Textbook

### Preferences instead of values {#preferences}

The methods so far estimate action values and choose actions from the estimates ([[epsilon-greedy]], [[ucb]]). Another approach learns a numerical **preference** $\pol{H_t(a)}$ for each action, which has no interpretation as a reward, and turns the preferences into probabilities with the **softmax** distribution (also called Gibbs or Boltzmann distribution):

$$\pol{\pi_t(a)} = \Pr(A_t = a) = \frac{e^{\pol{H_t(a)}}}{\sum_{b=1}^{k} e^{\pol{H_t(b)}}}. \label{softmax}$$

Only differences between preferences matter: adding the same constant to every preference leaves the probabilities unchanged. Initially all preferences are equal, say 0, so every action has probability $1/k$. The policy is always stochastic; it becomes nearly deterministic only when one preference is much larger than the others.

### The update {#update}

After choosing $A_t$ and receiving $\rew{R_t}$, every preference is updated:

$$\pol{H_{t+1}(a)} = \pol{H_t(a)} + \alp\,\big(\rew{R_t} - \rew{\bar{R}_t}\big)\big(\mathbb{1}_{a = A_t} - \pol{\pi_t(a)}\big) \quad \text{for all } a, \label{update-rule}$$

where $\alp > 0$ is a step size and $\rew{\bar{R}_t}$ is the average of the rewards before step $t$ (with $\rew{\bar{R}_1} = \rew{R_1}$), kept incrementally ([[incremental-mean]]). For the chosen action the factor is $1 - \pol{\pi_t(A_t)}$; for every other action it is $-\pol{\pi_t(a)}$. So a reward above the baseline raises the probability of the action just taken and lowers all the others, in proportion to their current probabilities; a reward below the baseline does the reverse. Since $\sum_a (\mathbb{1}_{a = A_t} - \pol{\pi_t(a)}) = 0$, the sum of the preferences never changes.

### A stochastic gradient {#derivation}

The update is not an ad hoc rule. It is a stochastic approximation to gradient ascent on the expected reward,

$$\mathbb{E}\big[\rew{R_t}\big] = \sum_x \pol{\pi_t(x)}\,\val{q_*(x)}, \qquad \pol{H_{t+1}(a)} = \pol{H_t(a)} + \alp\,\frac{\partial\, \mathbb{E}[\rew{R_t}]}{\partial \pol{H_t(a)}}. \label{ascent}$$

Exact gradient ascent is impossible because the values $\val{q_*(x)}$ are unknown. The point is that \ref{update-rule} is equal to \ref{ascent} *in expectation*. The derivation needs one fact about the softmax.

::: lemma {#lem-softmax} Derivative of the softmax
$\dfrac{\partial \pol{\pi_t(x)}}{\partial \pol{H_t(a)}} = \pol{\pi_t(x)}\,\big(\mathbb{1}_{a = x} - \pol{\pi_t(a)}\big)$.
:::

::: proof
Write $Z = \sum_b e^{H_t(b)}$, so $\pi_t(x) = e^{H_t(x)} / Z$ and $\partial Z / \partial H_t(a) = e^{H_t(a)}$. By the quotient rule,
$$\frac{\partial \pi_t(x)}{\partial H_t(a)} = \frac{\mathbb{1}_{a = x}\, e^{H_t(x)}\, Z - e^{H_t(x)}\, e^{H_t(a)}}{Z^2} = \mathbb{1}_{a = x}\, \pi_t(x) - \pi_t(x)\, \pi_t(a). \qquad \square$$
:::

::: theorem {#thm-gradient} The gradient bandit follows the gradient in expectation
For any baseline $B_t$ that does not depend on $A_t$, $\;\mathbb{E}\big[(\rew{R_t} - B_t)(\mathbb{1}_{a = A_t} - \pol{\pi_t(a)})\big] = \dfrac{\partial\, \mathbb{E}[\rew{R_t}]}{\partial \pol{H_t(a)}}$. In particular this holds for $B_t = \rew{\bar{R}_t}$, which depends only on earlier steps.
:::

::: proof
Since $\sum_x \pi_t(x) = 1$ for every value of the preferences, $\sum_x \partial \pi_t(x) / \partial H_t(a) = 0$, so a baseline can be subtracted for free:
$$\frac{\partial\, \mathbb{E}[R_t]}{\partial H_t(a)} = \sum_x q_*(x)\,\frac{\partial \pi_t(x)}{\partial H_t(a)} = \sum_x \big(q_*(x) - B_t\big)\,\frac{\partial \pi_t(x)}{\partial H_t(a)}.$$
Multiplying and dividing each term by $\pi_t(x)$ turns the sum into an expectation over $A_t \sim \pi_t$, and by \ref{lem-softmax},
$$= \mathbb{E}\Big[\big(q_*(A_t) - B_t\big)\,\frac{\partial \pi_t(A_t)/\partial H_t(a)}{\pi_t(A_t)}\Big] = \mathbb{E}\big[\big(q_*(A_t) - B_t\big)\big(\mathbb{1}_{a = A_t} - \pi_t(a)\big)\big].$$
Finally $\mathbb{E}[R_t \mid A_t] = q_*(A_t)$, so $q_*(A_t)$ can be replaced by $R_t$ without changing the expectation.
:::

So the gradient bandit is stochastic gradient ascent on the expected reward: each update moves the preferences uphill on average, and the noise around that average is what the baseline is for.

### The baseline {#baseline}

The theorem holds for *any* baseline that does not depend on the action, including $B_t = 0$, so the baseline does not change the expected update. It changes its variance, and that can matter a great deal. On a testbed whose true values are drawn around $+4$ instead of 0, every reward is positive. Without a baseline, every update raises the preference of the action just taken: the method can only learn which actions are reinforced *more*, from the noisy differences between large positive pushes, and actions that happen to be pulled early gain a head start that feeds itself. With the average reward as the baseline, only rewards better than usual push up.

::: figure {#fig-curves}
{{bandit-curves gradient}}
The fraction of steps on which the best arm was chosen by the gradient bandit, with and without the average reward as a baseline, for $\alp = 0.1$ and $\alp = 0.4$, on a testbed whose values are drawn around $+4$. Each curve averages 500 runs; the Lab computes them when the figure comes into view. After Sutton & Barto, Figure 2.5.
:::

In the runs above, over the last hundred steps the version with a baseline chooses the best arm about 84% of the time with $\alp = 0.1$, and the version without one about 46% of the time. The average reward is not the baseline with the least variance, but it is simple, needs no knowledge of the problem, and works well (Greensmith, Bartlett & Baxter, 2004). The same baseline reappears, with the same justification, in policy-gradient methods with states ([[baseline]]).

### How the bandit methods compare {#comparison}

Each bandit method has one main knob: $\eps$ for ε-greedy, $\alp$ for the gradient bandit, $c$ for UCB, the initial value $\val{Q_1}$ for optimistic greedy. A fair comparison runs each method over a range of its knob and records the average reward over the first 1000 steps.

::: figure {#fig-study}
{{bandit-study}}
A parameter study on the 10-armed testbed: the average reward over the first 1000 steps for each method, as a function of its knob (on a logarithmic scale). Each point averages 200 runs; the Lab computes them when the figure comes into view, so the curves fill in as the runs arrive. After Sutton & Barto, Figure 2.6.
:::

Every curve is an inverted U: each method needs its knob set neither too high nor too low (\ref{fig-study}). Each also does well over a range of about an order of magnitude, so none is fragile. On this testbed UCB does best, followed by optimistic greedy, the gradient bandit and ε-greedy. The ranking is specific to the problem, though: on a nonstationary problem, optimism and UCB lose much of their advantage ([[nonstationary]]).

### Toward policy gradients {#pg}

The factor in the update has a compact meaning. By \ref{lem-softmax}, the gradient of the logarithm of the probability of the chosen action is

$$\frac{\partial \ln \pol{\pi_t(A_t)}}{\partial \pol{H_t(a)}} = \frac{1}{\pol{\pi_t(A_t)}}\,\frac{\partial \pol{\pi_t(A_t)}}{\partial \pol{H_t(a)}} = \mathbb{1}_{a = A_t} - \pol{\pi_t(a)}, \label{log-grad}$$

so \ref{update-rule} reads $\mathbf{H} \leftarrow \mathbf{H} + \alp\,(\rew{R_t} - \rew{\bar{R}_t})\,\nabla_{\mathbf{H}} \ln \pol{\pi_t(A_t)}$: adjust the policy's parameters in the direction that makes the action taken more likely, scaled by how much better than usual the outcome was. With states, a return in place of the reward and any differentiable parameterized policy, this becomes REINFORCE ([[reinforce]], [[policy-parameterization]]), and the theorem above becomes the policy gradient theorem ([[pg-theorem]]). The gradient bandit is the one-state ancestor of the whole family of policy-gradient methods, up to [[ppo]].

### Historical remarks {#history}

Learning preferences and comparing each reward with a running average, *reinforcement comparison*, goes back to Sutton's thesis (1984). Williams (1992) showed that this family of methods follows the gradient of the expected reward, in the general form now called REINFORCE. Sutton and Barto (§2.8) present the gradient bandit with the derivation above and the comparison on the shifted testbed, and summarize the bandit methods with the parameter study of §2.10.

## Card

### Idea

Instead of estimating how much each arm pays, keep a preference for each arm and pull arms at random with the softmax of the preferences. After each pull, compare the reward with the average so far: better than usual makes the arm more likely, worse makes it less likely. It is gradient ascent on the expected reward.

::: analogy
A coach adjusting playing time. A player who does better than the team's usual level gets a bit more time and everyone else a bit less. Doing well only matters relative to the usual level: if everyone always scores, scoring alone says nothing.
:::

### The update {#update}

$$\pol{\pi(a)} = \frac{e^{\pol{H(a)}}}{\sum_b e^{\pol{H(b)}}} \qquad \pol{H(a)} \leftarrow \pol{H(a)} + \alp\,\big(\rew{R} - \rew{\bar{R}}\big)\big(\mathbb{1}_{a = A} - \pol{\pi(a)}\big) \ \text{ for every arm } a$$

$\rew{\bar{R}}$ is the average of the rewards so far: the baseline. The factor $\mathbb{1}_{a = A} - \pol{\pi(a)}$ is $\partial \ln \pol{\pi(A)} / \partial \pol{H(a)}$.

### One change from [[epsilon-greedy]] {#change}

| | learns | chooses |
| --- | --- | --- |
| [[epsilon-greedy]] | values $\val{Q(a)}$ | the best one, or a random arm with probability $\eps$ |
| gradient bandit | preferences $\pol{H(a)}$ | at random, with the softmax of the preferences |

### Backup diagram {#backup}

{{backup bandit}}

The update looks at the arm pulled and its reward, as in every bandit method, but changes the preferences of all arms.

### Pseudocode

::: pseudocode
Parameters: step size $\alp > 0$
Set $\pol{H(a)} = 0$ for every arm $a$ {#init}
For $t = 1, 2, \ldots$:
  Compute $\pol{\pi(a)} = e^{\pol{H(a)}} / \sum_b e^{\pol{H(b)}}$ and draw $A$ from $\pol{\pi}$ {#choose}
  Pull $A$, observe $\rew{R}$ {#act}
  For every arm $a$: $\pol{H(a)} \leftarrow \pol{H(a)} + \alp\,(\rew{R} - \rew{\bar{R}})\,(\mathbb{1}_{a = A} - \pol{\pi(a)})$, with $\rew{\bar{R}}$ the average of the earlier rewards ($\rew{R}$ itself at $t = 1$) {#update}
  $\rew{\bar{R}} \leftarrow \rew{\bar{R}} + \tfrac{1}{t}\,(\rew{R} - \rew{\bar{R}})$
:::

### Perks

- Follows the gradient of the expected reward: a principled method, not a heuristic.
- The policy is stochastic and changes smoothly, which suits problems where the best behavior is itself random.
- The seed of all policy-gradient methods: REINFORCE, actor–critic and PPO extend this update. [See it](lab:bandit-gradient)

### Flaws

- Without a baseline it can be badly misled when all rewards share a sign. [See it](lab:bandit-gradient)
- Never fully stops exploring: probabilities approach 0 but never reach it.
- On the testbed it learns more slowly than UCB or optimistic greedy.

### Knobs

| Knob | Too low | Too high |
| --- | --- | --- |
| $\alp$ step size | preferences change slowly: long random exploration | commits to an arm after a few lucky rewards |
| baseline | without one, every positive reward reinforces whatever was pulled | (no downside: any action-independent baseline keeps the gradient right) |

### Pitfalls

- Updating only the chosen arm's preference: the others must move too, by $-\alp(\rew{R} - \rew{\bar{R}})\,\pol{\pi(a)}$.
- Computing the softmax naively: subtract the largest preference before exponentiating, or large preferences overflow.
- Including the current reward in the baseline before the update: the baseline must not depend on the action just taken.

### Check yourself {#check}

::: question
All preferences are 0 in a 4-armed bandit. Arm 1 is pulled and pays 3 while the baseline is 1. With $\alp = 0.1$, what are the new preferences?
---
$\pol{H(1)} = 0.1 \cdot 2 \cdot (1 - 0.25) = 0.15$ and each other arm gets $-0.1 \cdot 2 \cdot 0.25 = -0.05$. They still add up to 0.
:::

::: question
The baseline does not change the expected update. Why does it matter so much when all rewards are around +4?
---
Without it every update pushes the chosen arm up, by a large amount, so the useful signal (which arm is pushed more) is buried in noise and early luck gets amplified. With it, only better-than-usual rewards push up.
:::

::: question
What does the factor $\mathbb{1}_{a = A} - \pol{\pi(a)}$ have to do with REINFORCE?
---
It is the gradient of $\ln \pol{\pi(A)}$ with respect to $\pol{H(a)}$. The update is $\alp\,(\rew{R} - \rew{\bar{R}})\,\nabla \ln \pol{\pi(A)}$, which is REINFORCE with a baseline, for a single state.
:::
