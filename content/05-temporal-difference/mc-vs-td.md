+++
summary = "Two ways to learn values from experience: wait for the real outcome, or trust the next prediction. Monte Carlo is unbiased but noisy; TD is biased but steadier, and it exploits the Markov structure of the world."
prereqs = ["mc-prediction", "td0", "bootstrapping"]
lab = "random-walk"
sources = [
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., §6.2–6.3, Examples 6.2–6.4 and Figure 6.2", url = "http://incompleteideas.net/book/the-book-2nd.html" },
  { text = "Sutton (1988), Learning to predict by the methods of temporal differences, Machine Learning 3", url = "https://doi.org/10.1007/BF00115009" },
  { text = "Singh & Sutton (1996), Reinforcement learning with replacing eligibility traces, Machine Learning 22", url = "https://doi.org/10.1007/BF00114726" },
  { text = "Kearns & Singh (2000), Bias-variance error bounds for temporal difference updates, Proceedings of the 13th Annual Conference on Computational Learning Theory" },
]

[story]
scene = "chain"
env = "random-walk"
seed = 1
average = 100
formula = '\step{1}{\text{MC: } \val{V(S_t)} \leftarrow \val{V(S_t)} + \alp\,\big[\rew{G_t} - \val{V(S_t)}\big] \qquad} \step{2}{\text{TD: } \val{V(S_t)} \leftarrow \val{V(S_t)} + \alp\,\big[\rew{R_{t+1}} + \gam\,\val{V(S_{t+1})} - \val{V(S_t)}\big]}'

[story.runs]
mc = { algorithm = "mc-prediction", alpha = 0.1, gamma = 1.0, v0 = 0.5, units = 100, measures = ["error"], name = "Monte Carlo, α = 0.1" }
td = { algorithm = "td0", alpha = 0.1, gamma = 1.0, v0 = 0.5, units = 100, measures = ["error"], name = "TD(0), α = 0.1" }
mcslow = { algorithm = "mc-prediction", alpha = 0.03, gamma = 1.0, v0 = 0.5, units = 100, measures = ["error"], name = "Monte Carlo, α = 0.03" }
+++

## Story

::: step {run = "mc", at = 0, truth = false, formula = 1}
**Two learners watch the same random walks**: Monte Carlo and TD(0), with the same step size, 0.1. Each starts with every estimate at 0.5. They differ only in their targets: the walk's actual outcome, or the reward plus the next state's estimate.
:::

::: step {run = "mc", at = 0, truth = false, play = 1, pace = 420, formula = 1}
Monte Carlo watches the first walk to its end: C, D, C, D, E, and out on the right, $\rew{+1}$. Only then does it learn: every state the walk visited moves toward the outcome. C, D and E all rise to 0.55.
:::

::: step {run = "td", at = 1, truth = false, formula = 2}
TD(0), after the same walk: only E has moved. Its other updates had nothing to learn, because each target, $0 + 0.5$, equaled the estimate. TD will pass E's news back to D, and then to C, on later walks.
:::

::: step {run = "mc", at = 100, truth = true}
After 100 walks, Monte Carlo's estimates still wobble around the true values: every walk drags every state it visits toward a 0 or a 1, so the estimates are only as steady as the last few walks.
:::

::: step {run = "td", at = 100, truth = true}
TD(0) after the same 100 walks is closer. Each of its targets depends on one step only, and on the neighbor's estimate, which already averages many walks. Its targets are biased by those estimates, but far less noisy.
:::

::: step {run = "td", at = 100, truth = true, curves = ["td", "mc", "mcslow"], metric = "error"}
Averaged over 100 runs. A smaller step size steadies Monte Carlo, but slows it down: TD(0) stays ahead. [Race them in the Lab](lab:random-walk).
:::

## Textbook

### Two targets {#targets}

Monte Carlo methods and TD methods both learn value functions from experience without a model, and both use the update *estimate ← estimate + α (target − estimate)*. They differ in the target ([[mc-prediction]], [[td0]]):

$$\text{MC: } \rew{G_t} = \rew{R_{t+1}} + \gam\,\rew{R_{t+2}} + \cdots + \gam^{T-t-1}\,\rew{R_T}, \qquad \text{TD: } \rew{R_{t+1}} + \gam\,\val{V(S_{t+1})}. \label{two-targets}$$

The return is an **unbiased** sample of $\val{v_\pi(S_t)}$, but its **variance** is high: it depends on every action, transition and reward until the end of the episode. The TD target is **biased**, because it uses the estimate $\val{V(S_{t+1})}$ instead of the true value, but its variance is much lower: it depends on one action, one transition and one reward. The bias shrinks as the estimates improve. Which effect dominates depends on the problem and on the stage of learning; error bounds that trade the two off exactly are given by Kearns and Singh (2000).

### Which learns faster? {#speed}

If both TD and Monte Carlo methods converge asymptotically to the correct predictions, which gets there first? At the current time this is an open question, in the sense that no one has been able to prove mathematically that one method converges faster than the other. In practice, TD methods have usually been found to converge faster than constant-α Monte Carlo methods on stochastic tasks. The random walk is an example (\ref{fig-error}).

::: figure {#fig-error}
{{random-walk error}}
The root-mean-square error of TD(0) and constant-α Monte Carlo on the random walk, at several step sizes, averaged over 100 runs. Computed by the Lab when the figure comes into view. After Sutton & Barto, Example 6.2.
:::

### Batch updating {#batch}

Suppose only a finite amount of experience is available, say 10 episodes or 100 time steps. A common approach with incremental learning methods is to present the experience repeatedly until the method converges. Given an approximate value function, the increments are computed for every time step at which a nonterminal state is visited, but the value function is changed only once, by the sum of all the increments; then all the available experience is processed again with the new value function, and so on, until it converges. This is **batch updating**, because updates are made only after processing each complete batch of training data.

Under batch updating, TD(0) converges deterministically to a single answer, independent of the step size, as long as the step size is sufficiently small. Constant-α Monte Carlo also converges deterministically under the same conditions, but to a *different* answer. Understanding the two answers explains the difference between the methods.

::: figure {#fig-batch}
{{random-walk batch}}
Batch training on the random walk: after each new episode, all episodes seen so far are replayed until the estimates converge. The root-mean-square error of the converged estimates, averaged over 100 runs. Computed by the Lab when the figure comes into view. After Sutton & Barto, Figure 6.2.
:::

Batch TD is consistently better than batch Monte Carlo on the random walk (\ref{fig-batch}), even though batch Monte Carlo is optimal in a limited sense: it minimizes the mean squared error with respect to the returns actually observed in the training set. How can batch TD do better against the *true* values? Because it is optimal in a way that is more relevant to predicting returns.

### Example: you are the predictor {#predictor}

::: example {#ex-predictor} You are the predictor (Sutton & Barto, Example 6.4)
You observe the following eight episodes of an unknown Markov reward process, with $\gam = 1$:

| Episode | States and rewards |
| --- | --- |
| 1 | A, 0, B, 0 (then the end) |
| 2 to 7 | B, 1 (six episodes) |
| 8 | B, 0 |

What are the best predictions for $\val{V(A)}$ and $\val{V(B)}$?
:::

Everyone agrees on $\val{V(B)} = \tfrac34$: six of the eight times the process was in B, it immediately terminated with a return of 1. For A there are two reasonable answers. One observes that 100% of the times the process was in A, it went next to B, with a reward of 0; since B is worth $\tfrac34$, A must be worth $\tfrac34$ too. This is the answer batch TD(0) gives. The other observes that A was seen once and the return that followed it was 0; batch Monte Carlo therefore estimates $\val{V(A)} = 0$. This answer gives the minimum squared error on the training data, zero in fact. Yet if the process is Markov, the first answer can be expected to give lower error on *future* data: it uses all eight episodes to estimate B, while the Monte Carlo answer for A rests on one return.

### Certainty equivalence {#certainty}

The example generalizes. Given batch data, the **maximum-likelihood model** of a Markov process estimates each transition probability as the fraction of observed transitions from the state that went to each successor, and each expected reward as the average of the observed rewards. Batch TD(0) converges to the value function that would be exactly correct *if this model were exactly correct*: the **certainty-equivalence estimate**, so called because it is equivalent to assuming that the estimate of the underlying process is known with certainty rather than being approximated. Batch Monte Carlo instead converges to the averages of the observed returns, which ignore the Markov structure.

This also explains why nonbatch TD methods tend to learn faster than Monte Carlo methods: they move, roughly, in the direction of the certainty-equivalence estimate. Computing that estimate directly is expensive: forming the model takes memory proportional to $|\mathcal{S}|^2$, and solving for its values on the order of $|\mathcal{S}|^3$ operations. TD methods approximate the same solution with memory proportional to $|\mathcal{S}|$ and repeated cheap updates.

### When Monte Carlo is the better choice {#mc-wins}

- **When the states are not Markov.** TD's advantage comes from the Markov structure. If the states hide relevant information, as with partial observations, the certainty-equivalence estimate is built on a wrong model, while Monte Carlo averages remain correct.
- **When bootstrapping is unstable.** With function approximation and off-policy training, bootstrapped targets can diverge; returns cannot ([[deadly-triad]]).
- **When only a few states matter.** Monte Carlo can evaluate a single state from episodes started there, at a cost independent of the number of states ([[mc-prediction]]).
- **When rewards are sparse and episodes short.** A single successful episode credits every state along its path at once ([[mc-control]]).

In between the two lie methods that bootstrap after several steps, which often beat both extremes ([[n-step-td]], [[td-lambda]]).

### At a glance {#summary}

| | Monte Carlo | TD(0) |
| --- | --- | --- |
| target | the return $\rew{G_t}$ | $\rew{R_{t+1}} + \gam\,\val{V(S_{t+1})}$ |
| bias | none | from the estimates, vanishing as they improve |
| variance | high: the whole episode | low: one step |
| learns | at the end of each episode | after every step |
| continuing tasks | no | yes |
| uses the Markov property | no | yes |
| batch solution | averages of the observed returns | the certainty-equivalence estimate |
| initial values | matter little | matter: they bias early targets |

### Historical remarks {#history}

The random walk example and the analysis of TD and Monte Carlo methods under batch updating are due to Sutton (1988), who showed that batch TD(0) converges to the certainty-equivalence estimate. Singh and Sutton (1996) compared Monte Carlo and TD methods further, and Kearns and Singh (2000) derived bias–variance error bounds for TD updates. The “you are the predictor” example is from Sutton and Barto (Example 6.4).

## Card

### Idea

Monte Carlo moves each estimate toward the real outcome of the episode: unbiased, but noisy, and only at the end. TD moves it toward the next reward plus the next estimate: biased by the estimates, but steadier, online, and it uses the Markov structure. On most stochastic tasks TD learns faster.

::: analogy
Predicting an election. One forecaster waits for the final result before updating anything. Another updates every night from the latest polls and from the previous forecast. The second is sometimes misled by a bad forecast, but learns far faster from a long campaign.
:::

### In symbols {#formula}

$$\text{MC: } \val{V(S_t)} \leftarrow \val{V(S_t)} + \alp\,\big[\,\rew{G_t} - \val{V(S_t)}\,\big] \qquad \text{TD: } \val{V(S_t)} \leftarrow \val{V(S_t)} + \alp\,\big[\,\rew{R_{t+1}} + \gam\,\val{V(S_{t+1})} - \val{V(S_t)}\,\big]$$

### The trade-off {#tradeoff}

| | Monte Carlo | TD |
| --- | --- | --- |
| bias | none | yes |
| variance | high | low |
| when it learns | end of episode | every step |
| batch answer | sample averages | certainty equivalence |

### Why it matters {#why}

It is the first big design choice of reinforcement learning, and it returns everywhere: $n$-step returns, λ, GAE and the deadly triad are all about how much to rely on real returns and how much on estimates. [See it](lab:random-walk)

### Pitfalls

- Comparing the two at a single step size: each has its own best step size; compare them at their best.
- Assuming TD always wins: with non-Markov states or unstable approximation, Monte Carlo can be better.
- Forgetting the cost of waiting: Monte Carlo learns nothing until an episode ends.

### Check yourself {#check}

::: question
In “you are the predictor”, why does batch Monte Carlo say $\val{V(A)} = 0$ and batch TD say $\tfrac34$?
---
Monte Carlo averages the one return observed after A, which was 0. TD uses the observed transition from A to B, and B's estimate of $\tfrac34$, built from all eight episodes.
:::

::: question
What is the certainty-equivalence estimate?
---
The value function that would be exactly right if the maximum-likelihood model of the process, estimated from the data, were the true process. Batch TD(0) converges to it.
:::

::: question
Why do TD targets have lower variance than returns?
---
A TD target depends on one random transition and reward; a return depends on all of them until the end of the episode.
:::
