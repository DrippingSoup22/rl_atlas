+++
summary = "Instead of picking one n, average all the n-step returns: the one-step return gets weight 1 − λ, and each longer one λ times the weight of the one before. λ = 0 gives the TD(0) target, λ = 1 the Monte Carlo return."
prereqs = ["n-step-td", "td0", "mc-prediction"]
lab = "td-lambda-walk"
sources = [
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., §12.1 and §12.3", url = "http://incompleteideas.net/book/the-book-2nd.html" },
  { text = "Sutton (1988), Learning to predict by the methods of temporal differences, Machine Learning 3", url = "https://doi.org/10.1007/BF00115009" },
  { text = "Watkins (1989), Learning from delayed rewards, PhD thesis, University of Cambridge", url = "https://www.cs.rhul.ac.uk/~chrisw/thesis.html" },
  { text = "Cichosz (1995), Truncating temporal differences: on the efficient implementation of TD(λ) for reinforcement learning, Journal of Artificial Intelligence Research 2", url = "https://doi.org/10.1613/jair.135" },
]

[story]
scene = "chain"
env = "random-walk-19"
seed = 4
average = 100
formula = '''\step{1}{\rew{G_t^{(n)}} = \rew{R_{t+1}} + \dots + \gam^{n-1} \rew{R_{t+n}} + \gam^n \val{V(S_{t+n})} \qquad} \step{2}{\rew{G_t^\lambda} = (1 - \lam) \sum_{n \ge 1} \lam^{n-1} \rew{G_t^{(n)}}}'''

[story.runs]
zero = { algorithm = "offline-lambda", alpha = 0.2, gamma = 1.0, units = 10, measures = ["error"], lambda = 0.0, name = "λ = 0" }
half = { algorithm = "offline-lambda", alpha = 0.2, gamma = 1.0, units = 10, measures = ["error"], lambda = 0.5, name = "λ = 0.5" }
high = { algorithm = "offline-lambda", alpha = 0.2, gamma = 1.0, units = 10, measures = ["error"], lambda = 0.8, name = "λ = 0.8" }
mc = { algorithm = "offline-lambda", alpha = 0.2, gamma = 1.0, units = 10, measures = ["error"], lambda = 1.0, name = "λ = 1" }
+++

## Story

::: step {run = "high", at = 0, play = 1, lead = 10, truth = false}
**The 19-state walk, every estimate at 0.** This first walk wanders back and forth for about a hundred steps before it leaves on the right. The only reward, $\rew{+1}$, comes at the very end. Once the walk is over, every state it visited gets a target. Which target?
:::

::: step {run = "zero", at = 1, truth = false, formula = 1}
**One step ahead, then trust the estimate.** Each state's target is its one-step return: the next reward plus the next state's estimate. Inside the row every reward and every estimate is still 0, so only S, the last state before the exit, has anything to learn: it moves a fifth of the way toward 1, to 0.20. That is TD(0), and it is the λ-return with $\lam = 0$.
:::

::: step {run = "mc", at = 1, truth = false, formula = 1}
**All the way to the end.** With the full return as target, every visit of every state sees the $\rew{+1}$ at the end, and the states the walk passed many times move furthest: L, visited again and again, reaches 0.97. That is Monte Carlo, the λ-return with $\lam = 1$. It learns from one walk what was true of that walk only: the left half of the row is worth much less than it now says.
:::

::: step {run = "half", at = 1, truth = false, formula = 2}
**The λ-return averages every n.** Its target gives the one-step return a weight of $1 - \lam$, the two-step return $(1 - \lam)\lam$, and so on, the weights shrinking by $\lam$ each step further out, all adding up to 1. With $\lam = 0.5$, half the target is the one-step return, a quarter the two-step return. The exit's news reaches back a few states: P, Q and R move a little, S the most.
:::

::: step {run = "high", at = 1, truth = false, formula = 2}
**With λ = 0.8, the news reaches back further,** to N, five states before the exit, but fading as it goes, instead of lifting the whole path to 1 as Monte Carlo did.
:::

::: step {run = "high", at = 10, curves = ["zero", "half", "high", "mc"], metric = "error"}
**Ten walks each, averaged over 100 runs.** With this step size, $\lam = 0.8$ ends closest to the true values, at about 0.18; one step ($\lam = 0$) learns too slowly, at about 0.42, and whole returns ($\lam = 1$) are too noisy, at about 0.62. Somewhere in between beats both ends, and where depends on the step size: the textbook's study maps it. [Traces that compute it online, in the Lab](lab:td-lambda-walk).
:::

## Textbook

### Averaging over n {#idea}

[[n-step-td]] leaves a question open: which $n$? The study on the random walk showed that the best one depends on the step size and on the stage of learning, and that a wrong choice costs a lot. One way out is not to choose. Every $n$-step return is a sound target, in the sense that its average error is smaller than that of the estimate it ends with, and a weighted average of sound targets is sound too, provided the weights are positive and add up to 1. A target made of two thirds of the one-step return and one third of the ten-step return, for example, leans on the short look but still hears what happened ten steps later. Mixtures like this are called **compound returns**; their backup diagram stacks the diagrams of their parts, each labeled with its weight.

The **λ-return** is the compound return that won out, for a practical reason: of all the possible mixtures, its geometric weights are the ones that can be learned step by step, without storing the episode ([[td-lambda]]). It uses every $n$-step return at once, each weighted $\lambda$ times less than the one before, for a single parameter $\lambda \in [0, 1]$.

### The definition {#definition}

::: definition {#def-lambda} The λ-return
For $\lambda \in [0, 1]$, the λ-return from time $t$ is the weighted average of all the $n$-step returns from $t$,
$$\rew{G^\lambda_t} = (1 - \lam) \sum_{n=1}^{\infty} \lam^{n-1}\,\rew{G_{t:t+n}}.$$
:::

The weights $(1-\lambda), (1-\lambda)\lambda, (1-\lambda)\lambda^2, \ldots$ add up to 1. In an episode ending at time $T$, every $n$-step return with $t + n \ge T$ is the full return $\rew{G_t}$, so the infinite sum collapses to a finite one plus a single term for the full return:

$$\rew{G^\lambda_t} = (1 - \lam) \sum_{n=1}^{T-t-1} \lam^{n-1}\,\rew{G_{t:t+n}} \;+\; \lam^{T-t-1}\,\rew{G_t}. \label{episodic}$$

All the weight that would have gone to returns past the end lands on the full return. The two ends of the range are the familiar targets: with $\lambda = 0$ only the one-step return remains, the TD(0) target ([[td0]]); with $\lambda = 1$ only the full return, the Monte Carlo target ([[mc-prediction]]). Like $n$, λ slides between bootstrapping early and not at all; unlike $n$, it does so smoothly. \ref{fig-weights} shows the weights.

::: figure {#fig-weights}
{{lambda-weights}}
The weight the λ-return gives each $n$-step return, in an episode that ends 12 steps after time $t$; the dashed bar is the weight left on the full return. Move the slider to change λ. At λ = 0 all the weight is on the one-step return; at λ = 1, all of it is on the full return.
:::

The weight halves every $\ln 2 / \ln(1/\lambda)$ steps: about 3 steps for λ = 0.8, 7 for λ = 0.9 and 14 for λ = 0.95. The quantity $1/(1-\lambda)$ is a rough time horizon of the return, the average $n$ under these weights.

### A recursive form {#recursion}

Splitting off the first reward gives a recursion that makes the λ-return easy to compute backward from the end of an episode:

$$\rew{G^\lambda_t} = \rew{R_{t+1}} + \gam\,\Big[(1 - \lam)\,\val{V(S_{t+1})} + \lam\,\rew{G^\lambda_{t+1}}\Big], \qquad \rew{G^\lambda_{T-1}} = \rew{R_T}. \label{eq-recursion}$$

At every step, the return bootstraps with weight $1 - \lambda$ (trust the estimate of the next state) and continues with weight $\lambda$ (trust what actually happened next). The same recursion with a different $\lambda$ at each step, or one that depends on the state, defines the more general returns used in later methods.

Written with TD errors $\del_k = \rew{R_{k+1}} + \gam\,\val{V(S_{k+1})} - \val{V(S_k)}$ ([[td-error]]), with the estimates held fixed, the λ-return error is a discounted sum of future TD errors:

$$\rew{G^\lambda_t} - \val{V(S_t)} = \sum_{k=t}^{T-1} (\gam\lam)^{k-t}\,\del_k. \label{eq-errors}$$

This identity is the bridge to [[td-lambda]]: each TD error $\del_k$ is owed, with weight $(\gamma\lambda)^{k-t}$, to every earlier state $S_t$. TD(λ) pays those debts as the errors arrive instead of waiting for the end.

### The offline λ-return algorithm {#offline}

The most direct way to use the λ-return is to wait for the end of each episode, compute $\rew{G^\lambda_t}$ for every step, and then update:

$$\val{V(S_t)} \leftarrow \val{V(S_t)} + \alp\,\big[\,\rew{G^\lambda_t} - \val{V(S_t)}\,\big], \qquad t = 0, 1, \ldots, T - 1. \label{eq-offline}$$

This is the **offline λ-return algorithm**. It is a **forward view**: each update looks forward in time to the rewards and states that followed. Its learning is as good as that of $n$-step methods, and its knob is smoother. \ref{fig-study} compares settings of λ on the 19-state random walk; it closely resembles the study of $n$ ([[n-step-td]]), with intermediate λ doing best.

::: figure {#fig-study}
{{lambda-study offline}}
Average error of the offline λ-return algorithm on the 19-state random walk, against the step size, for several λ. Each point is the RMS error over the 19 states, averaged over the first 10 episodes and over 100 runs. Computed by the Lab. After Sutton & Barto, Figure 12.3.
:::

The best results, an error of about 0.26 near λ = 0.8 and α = 0.4, match the best $n$-step results. λ = 1 is constant-α Monte Carlo and does worst: in this task a single return is a poor target.

### Truncated λ-returns {#truncated}

The offline algorithm has the drawback of Monte Carlo: nothing is learned until the episode ends. Since the weights decay geometrically, little is lost by cutting the λ-return off after a horizon $h$ and putting the remaining weight on the $h$-step return:

$$\rew{G^\lambda_{t:h}} = (1 - \lam) \sum_{n=1}^{h-t-1} \lam^{n-1}\,\rew{G_{t:t+n}} + \lam^{h-t-1}\,\rew{G_{t:h}}. \label{eq-truncated}$$

Updating $\val{V(S_t)}$ toward $\rew{G^\lambda_{t:t+n}}$ at time $t + n$ gives an $n$-step method with λ-style weights, truncated TD(λ) (Cichosz, 1995). The cleaner answer, which needs no horizon and spreads the computation evenly over time, is the backward view of [[td-lambda]].

### Why it matters {#why}

- **One smooth knob.** λ moves continuously from TD(0) to Monte Carlo; a good value is usually easy to find, and the results are not very sensitive to it.
- **A target, not an algorithm.** The λ-return defines what should be learned; [[td-lambda]], [[sarsa-lambda]] and their variants are ways of learning it online. Later methods reuse the same weights on other quantities: generalized advantage estimation averages advantage estimates with them ([[gae]]).
- **A theory of traces.** Eligibility traces are exactly the bookkeeping that turns this forward view into an algorithm that runs step by step.

### Historical remarks {#history}

The λ-return was defined by Watkins (1989), who recognized the forward view behind the TD(λ) algorithm that Sutton (1988) had introduced with eligibility traces. The equivalence of the offline λ-return algorithm and offline TD(λ) was shown by Sutton (1988) for the case of updates made at the end of the episode; Cichosz (1995) studied truncated versions.

## Card

### Idea

Every $n$-step return is a fair target, so average all of them: give the one-step return weight $1 - \lambda$, and each longer one λ times the weight of the one before. λ = 0 is TD(0)'s target, λ = 1 is the Monte Carlo return, and anything in between mixes them smoothly.

::: analogy
Asking several advisers how a decision will turn out, from the one who looks one move ahead to the one who imagines the whole game, and trusting the short-sighted ones a little more, by a fixed factor each.
:::

### In symbols {#formula}

$$\rew{G^\lambda_t} = (1 - \lam) \sum_{n=1}^{T-t-1} \lam^{n-1}\,\rew{G_{t:t+n}} + \lam^{T-t-1}\,\rew{G_t}$$

The weights add up to 1. The same return, one step at a time, computed backward from the end of the episode:

$$\rew{G^\lambda_t} = \rew{R_{t+1}} + \gam\,\big[(1-\lam)\,\val{V(S_{t+1})} + \lam\,\rew{G^\lambda_{t+1}}\big]$$

At each step: trust the estimate with weight $1 - \lambda$, or keep following what happened with weight $\lambda$.

### Weights {#weights}

| λ | weight on 1 step | half-life of the weights | horizon $1/(1-\lambda)$ |
| --- | --- | --- | --- |
| 0 | 1 | — | 1 step |
| 0.5 | 0.5 | 1 step | 2 steps |
| 0.8 | 0.2 | about 3 steps | 5 steps |
| 0.9 | 0.1 | about 7 steps | 10 steps |
| 1 | 0 | — | the whole episode |

### Why it matters {#why}

- It is the target of TD(λ) and SARSA(λ), which learn it step by step with eligibility traces.
- The offline λ-return algorithm learns as well as the best $n$-step method, with a smoother knob. [See TD(λ) learn it](lab:td-lambda-walk)
- The same geometric weights reappear in generalized advantage estimation ([[gae]]).

### Pitfalls

- Forgetting the leftover weight: in an episode, everything past the end lands on the full return.
- Thinking λ = 1 means "no discount": λ weighs returns of different lengths, γ discounts rewards within each.

### Check yourself {#check}

::: question
What does the λ-return become when λ = 0, and when λ = 1?
---
With λ = 0, only the one-step return keeps any weight: the TD(0) target. With λ = 1, all the weight goes to the full return: the Monte Carlo target.
:::

::: question
An episode ends 3 steps after time $t$, and λ = 0.5. What weights do the one-step, two-step and full returns get?
---
$1 - \lambda = 0.5$ for the one-step return, $(1-\lambda)\lambda = 0.25$ for the two-step return, and the rest, $\lambda^2 = 0.25$, for the full return.
:::

::: question
Why can the offline λ-return algorithm not learn during an episode?
---
The λ-return from time $t$ includes the full return, which is known only when the episode ends; every target therefore waits for the end.
:::
