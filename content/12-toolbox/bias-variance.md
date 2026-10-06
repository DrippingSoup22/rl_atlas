+++
summary = "Every target an RL method learns from mixes two kinds of error. A target made of real rewards is right on average but noisy; a target that leans on the agent's own estimates is steady but inherits their mistakes. n, λ, γ, baselines, target networks and Double Q are all dials on that trade, and the best setting sits between the extremes."
prereqs = ["bootstrapping", "mc-vs-td"]
lab = "gae-cliff"
sources = [
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., chapters 6, 7 and 12", url = "http://incompleteideas.net/book/the-book-2nd.html" },
  { text = "Kearns & Singh (2000), Bias-variance error bounds for temporal difference updates, COLT", url = "https://www.cis.upenn.edu/~mkearns/papers/tdlambda.pdf" },
  { text = "Schulman, Moritz, Levine, Jordan & Abbeel (2016), High-dimensional continuous control using generalized advantage estimation, ICLR", url = "https://arxiv.org/abs/1506.02438" },
  { text = "van Hasselt (2010), Double Q-learning, NeurIPS", url = "https://papers.nips.cc/paper/3964-double-q-learning" },
]

[story]
scene = "chain"
env = "random-walk"
seed = 1
average = 200
formula = '\step{1}{\text{Monte Carlo: } \val{V(S_t)} \leftarrow \val{V(S_t)} + \alp\,\big[\rew{G_t} - \val{V(S_t)}\big] \qquad} \step{2}{\text{TD: } \val{V(S_t)} \leftarrow \val{V(S_t)} + \alp\,\big[\rew{R_{t+1}} + \gam\,\val{V(S_{t+1})} - \val{V(S_t)}\big]}'

[story.runs]
mc = { algorithm = "mc-prediction", alpha = 0.1, gamma = 1.0, v0 = 0.0, units = 100, measures = ["error"], name = "Monte Carlo" }
td = { algorithm = "td0", alpha = 0.1, gamma = 1.0, v0 = 0.0, units = 100, measures = ["error"], name = "TD(0)" }
+++

## Story

::: step {run = "mc", at = 0, seeds = 20, note = "faint: 20 runs · bold: their average · dashed: the true values"}
**Two ways to be wrong.** The random walk of [[td0]]: five states, a walk that steps left or right at random, $\rew{+1}$ for leaving on the right. The true values are the dashed steps, $\tfrac16$ to $\tfrac56$. Twenty learners estimate them, each from its own walks, and each starts every estimate at 0, well below the truth, so that their errors have room to show. Each faint line is one learner; the bold line is their average.
:::

::: step {run = "mc", at = 30, seeds = 20, formula = 1, note = "faint: 20 runs · bold: their average · dashed: the true values"}
**Monte Carlo after 30 walks.** Each estimate moves toward the return that followed the visit, a 0 or a 1 and nothing in between. On average those returns are exactly right, so the average of the 20 learners already sits close to the truth: little **bias**. But each learner moves with its own last few walks, and they scatter: for C, from 0.36 to 0.70. That scatter is **variance**.
:::

::: step {run = "td", at = 30, seeds = 20, formula = 2, note = "faint: 20 runs · bold: their average · dashed: the true values"}
**TD(0) after the same 30 walks.** The learners agree with one another far more: for C, from 0.11 to 0.33. But every one of them is below the truth, in every state. A TD target is the next reward plus the neighbor's estimate, and the neighbors still remember the 0 they started from. That is **bias**: averaging more learners would not remove it, because they all lean on the same kind of wrong guess.
:::

::: step {run = "td", at = 100, seeds = 20, note = "faint: 20 runs · bold: their average · dashed: the true values"}
**After 100 walks**, TD's bias has nearly gone: as the neighbors' estimates improved, so did the targets built on them. Its learners still agree more closely than Monte Carlo's: their spread around their average is about 0.055, against Monte Carlo's 0.094, which a constant step size never averages away.
:::

::: step {run = "td", at = 100, seeds = 20, curves = ["mc", "td"], metric = "error", note = "faint: 20 runs · bold: their average · dashed: the true values"}
**The error, averaged over 200 learners**, counts both kinds. For the first 73 walks Monte Carlo is ahead, because TD's bias, inherited from the starting guesses, costs more than Monte Carlo's scatter. Then TD's bias has faded while Monte Carlo's scatter stays, and TD pulls ahead. Starting from better guesses, as in [[mc-vs-td]], TD leads from the first walks. Every dial in this entry trades the two kinds of error, and the best setting is rarely at either end.
:::

## Textbook

### Two kinds of error {#two}

An RL method learns by pulling an estimate toward a target, and the target is itself an estimate of something nobody knows exactly. It can be wrong in two ways:

- **Bias**: wrong on average. However many targets you average, you converge to the wrong number.
- **Variance**: right on average, but each target scatters around the truth, so many are needed before their average settles.

For an estimator $\hat x$ of a quantity $x$, the two add up:
$$\mathbb E\big[(\hat x - x)^2\big] = \big(\mathbb E[\hat x] - x\big)^2 + \mathbb E\big[(\hat x - \mathbb E[\hat x])^2\big] \label{decomp}$$
the squared bias plus the variance. Reducing one usually increases the other, and the best estimator is rarely at either end.

### The two ends {#ends}

The clearest case is the target for $\val{v_\pi(s)}$:

- **The return** $G_t = \rew{R_{t+1}} + \gam \rew{R_{t+2}} + \gam^2 \rew{R_{t+3}} + \dots$ ([[mc-prediction]]). Its average is exactly $\val{v_\pi(s)}$: no bias. But it adds up the randomness of every later action, transition and reward, so it varies a lot from one episode to the next, more so the longer the episodes.
- **The one-step target** $\rew{R_{t+1}} + \gam\,\val{V(S_{t+1})}$ ([[td0]]). Only one step of randomness, so it varies little. But it uses the current estimate $\val{V(S_{t+1})}$, which is wrong while learning: that error enters the target as bias.

Every method in between mixes the two: $n$ real rewards, then an estimate ([[n-step-td]]), or a weighted blend of all such targets ([[lambda-return]]).

### The dials {#dials}

| dial | toward low variance | toward low bias |
| --- | --- | --- |
| steps before bootstrapping, $n$ | $n = 1$ | $n = \infty$ (the return) |
| trace decay $\lam$, GAE's $\lam$ | $\lam = 0$ | $\lam = 1$ |
| discount $\gam$ | small $\gam$: a short horizon | $\gam$ close to 1 |
| baseline in a policy gradient | with a baseline | (no cost: a baseline adds no bias) |
| target network | copied rarely | copied every step |
| maximum over noisy estimates | Double Q: one picks, the other judges | plain max: biased upward |
| features | coarse: many states share a weight | fine: each state on its own |

A few of these deserve a word.

- **The discount.** With $\gam < 1$ the agent optimizes a different objective, the discounted one, and treats rewards beyond about $1 / (1 - \gam)$ steps as nearly worthless. For problems that really care about the long run, that is a bias, accepted because long horizons make every estimate noisier.
- **Baselines** are the exception to the trade. Subtracting $\val{b(s)}$ from the return in a policy gradient leaves its expectation unchanged and can remove most of its variance ([[baseline]]).
- **The maximum** of noisy estimates is biased upward: the largest of several estimates is more likely to be one that was overestimated. Q-learning's target takes such a maximum. Double Q-learning picks the action with one estimate and evaluates it with the other. That removes the upward bias, though it can err a little the other way ([[double-q]]).

### What it looks like here {#guide}

The guide's experiments show the trade from both sides:

- **n-step TD on the 19-state walk**, $\alp = 0.4$, 100 runs ([[n-step-td]]). Over the first 10 walks the average RMS error is 0.41 with $n = 1$, 0.26 with $n = 4$, and 0.53 with $n = 32$: one step learns slowly, many steps learn noisily, and in between does best. By walk 20, $n = 1$ has caught up (0.14 against 0.27): with this large step size, the noise of the longer targets keeps them from settling.
- **GAE on the cliff**, TRPO, 20 seeds ([[gae]]). With $\lam = 0$ the advantage trusts the critic after one step, and the critic is wrong early on: only 6 of 20 runs end up reaching the goal in under 100 steps. With $\lam = 0.9$, all 20 do; with $\lam = 1$, the whole return, 19 do. Here the bias of trusting a poor critic costs more than the variance of real returns.
- **Maximization bias**, 1,000 runs ([[double-q]]). By episodes 41 to 60, Q-learning still goes left toward the lucky-looking actions more than 10% of the time in 93% of the runs. Double Q-learning has stopped in 79% of them.
- **A baseline on the short corridor** ([[baseline]]). A policy step size of $2^{-9}$ with a baseline is 16 times the largest safe step without one. The same step without a baseline throws 94 runs of 100 off the landscape: less variance, safer steps.

### Bias that grows {#grows}

With a table, a biased target is a temporary problem: as the estimates improve, the bias shrinks, and TD methods converge. With function approximation and off-policy data, the error in the estimates can feed back into the targets faster than it is corrected, and the estimates diverge ([[deadly-triad]]). Target networks ([[target-network]]) and conservative updates are partly ways to keep this feedback in check: they accept a lagging, biased target to stop it from moving with every step.

### Historical remarks {#history}

The bias–variance view of TD methods goes back to the analysis of TD(λ): Kearns and Singh (2000) bounded the error of TD updates in terms of λ and showed the best λ in between. Schulman et al. (2016) framed GAE's λ as a bias–variance dial for policy gradients, and van Hasselt (2010) identified the upward bias of the max and proposed Double Q-learning to remove it.

## Card

### Idea

A target built from real rewards is right on average but noisy (variance); one that leans on the agent's own estimates is steady but inherits their errors (bias). Most RL knobs move a method between the two, and the best setting usually sits in between.

::: analogy
Asking one well-informed but opinionated friend for directions, or polling a crowd. The friend is consistent but may be consistently wrong; the crowd is right on average, but you need many answers before their average means anything.
:::

### The dials {#dials}

- $n$ and $\lam$: small leans on estimates (bias), large on real rewards (variance).
- $\gam$: small shortens the horizon (bias), large lengthens it (variance).
- Baselines: less variance, no bias.
- Double Q: removes the upward bias of a max over noisy estimates.
- Target networks: a lagging target (bias) that does not move with every step (stability).

### Pitfalls

- Treating λ = 1 or n = ∞ as always safest: unbiased can mean too noisy to learn.
- Trusting a bootstrapped target early, when the estimates it uses are still poor.
- Forgetting that γ < 1 changes the objective, not just the speed.
- Calling every improvement a variance reduction: a baseline is unbiased, a shorter horizon is not.

### Check yourself {#check}

::: question
Why is the one-step TD target biased, when the return is not?
---
Its expectation uses the current estimate of the next state's value, which is wrong while learning. The return's expectation is exactly the true value, since it contains only real rewards.
:::

::: question
On the cliff, GAE with λ = 0 succeeded in 6 runs of 20, λ = 0.9 in 20. What went wrong with λ = 0?
---
With λ = 0 each advantage is a single TD error, which trusts the critic completely after one step. Early on the critic is badly wrong, so the advantages are biased and push the policy the wrong way.
:::

::: question
Why does subtracting a baseline not bias a policy gradient?
---
The baseline does not depend on the action, and the expected gradient of log π over actions is zero, so the subtracted term averages to zero. It only changes how much each sample varies.
:::
