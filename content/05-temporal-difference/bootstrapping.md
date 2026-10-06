+++
summary = "Updating an estimate from other estimates instead of waiting for the real outcome: faster and steadier learning, at the price of trusting guesses that may be wrong."
prereqs = ["td0", "mc-prediction", "policy-evaluation"]
lab = "random-walk"
sources = [
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., §6.1–6.3, §8.13 and §11.3", url = "http://incompleteideas.net/book/the-book-2nd.html" },
  { text = "Samuel (1959), Some studies in machine learning using the game of checkers, IBM Journal of Research and Development 3", url = "https://doi.org/10.1147/rd.33.0210" },
  { text = "Sutton (1988), Learning to predict by the methods of temporal differences, Machine Learning 3", url = "https://doi.org/10.1007/BF00115009" },
  { text = "Baird (1995), Residual algorithms: reinforcement learning with function approximation, Proceedings of the 12th International Conference on Machine Learning" },
]

[story]
scene = "predictor"
episodes = [
  [["A", 0], ["B", 0]],
  [["B", 1]], [["B", 1]], [["B", 1]], [["B", 1]], [["B", 1]], [["B", 1]],
  [["B", 0]],
]
formula = '''\step{1}{\text{Monte Carlo: } \val{V(s)} \leftarrow \text{average of } \rew{G_t} \qquad} \step{2}{\text{TD: } \val{V(s)} \leftarrow \text{average of } \rew{R_{t+1}} + \gam\,\val{V(S_{t+1})}}'''
+++

## Story

::: step {show = 8}
**Eight episodes, and nothing else to go on.** One started in A, paid nothing, moved to B, and ended there with nothing. Six started in B and ended with a reward of 1. One started in B and ended with nothing. No discount. What is each state worth?
:::

::: step {show = 8, mark = "B", mc = true, td = true, formula = 2}
**B is easy.** It was left eight times, six of them with a reward of 1, so both predictors say 0.75: Monte Carlo averages the eight returns that followed B, and TD averages the eight rewards plus the value of what came next, the end, worth 0. They agree because nothing comes after B but rewards.
:::

::: step {show = 8, mark = "A", mc = true, formula = 1}
**Monte Carlo on A: 0.** A appears once, and the return that followed it was 0 + 0 = 0. Monte Carlo only believes returns that actually followed a state, so A is worth exactly what its one episode paid.
:::

::: step {show = 8, mark = "A", mc = true, td = true, link = ["B", "A"], formula = 2}
**TD on A: 0.75.** TD never looks at A's whole return. It sees one step: A paid 0 and led to B. And B, from all eight episodes, is worth 0.75. So A is worth 0 + 0.75: a guess built from another guess. That is **bootstrapping**: A borrows what was learned about B from episodes that never went through A.
:::

::: step {show = 8, mc = true, td = true, link = ["B", "A"], formula = 2}
**Which is right?** If what happens after B does not depend on how B was reached, the world is Markov, and TD's answer is the better bet: the next visit to A will likely lead to B again, and B pays 1 three times in four. Monte Carlo's 0 rests on one unlucky episode. Run on a batch until they settle, TD finds the values of the most likely model of the data, and Monte Carlo the values that best fit the returns seen. The price of bootstrapping is the other side of the same coin: if B's guess is wrong, A inherits the error.
:::

## Textbook

### A guess from a guess {#idea}

An update **bootstraps** if its target contains another estimate. The term comes from the phrase “pulling oneself up by one's bootstraps”: the method improves its estimates using estimates. Dynamic programming bootstraps: each state's new value is computed from the current values of its successors ([[policy-evaluation]]). Temporal-difference methods bootstrap: the TD(0) target $\rew{R_{t+1}} + \gam\,\val{V(S_{t+1})}$ contains the estimate $\val{V(S_{t+1})}$ ([[td0]]). Monte Carlo methods do not: their target, the return $\rew{G_t}$, is made of observed rewards only ([[mc-prediction]]).

At first sight bootstrapping looks circular: how can guesses improve by being compared with other guesses? The answer is that the target is not only a guess. It contains at least one real reward and one real transition, and it is evaluated one step later, closer to the outcome. When the estimates are wrong, the targets are biased; but as each state's estimate improves, the targets of its predecessors improve too, and the errors shrink together. For tabular methods this is made precise by the contraction of the Bellman operator ([[bellman]]).

### Two choices behind every update {#dimensions}

Every update in the atlas can be placed by two choices. How deep does its target look before bootstrapping, and does it average over all possible outcomes or sample one?

| | samples one outcome | averages over all outcomes, with a model |
| --- | --- | --- |
| **bootstraps after one step** | TD(0), [[sarsa]], [[q-learning]] | dynamic programming: [[policy-evaluation]], [[value-iteration]] |
| **waits for the real return** | Monte Carlo: [[mc-prediction]] | exhaustive search of all futures |

The map's *unified view* arranges the tabular methods in this space. Between the two rows lie methods that bootstrap after $n$ steps, or after a weighted mixture of all numbers of steps ([[n-step-td]], [[lambda-return]], [[td-lambda]]); between the columns, methods that sample some outcomes and average others ([[expected-sarsa]]).

### What bootstrapping buys {#benefits}

- **Lower variance.** A bootstrapped target depends on one transition; a return depends on every transition until the end of the episode. Less randomness in the target means faster learning on stochastic tasks ([[mc-vs-td]]).
- **Learning online.** The target is available after one step, so learning happens during the episode, and in tasks with no episodes at all.
- **Sharing information.** A state's estimate improves whenever any path through its successors teaches the successors something. Monte Carlo estimates, by contrast, learn only from returns that actually followed the state. On a Markov process this sharing is the right thing to do: batch TD finds the values of the most likely model of the process, Monte Carlo only the averages of the observed returns ([[mc-vs-td]]).

### What it costs {#costs}

- **Bias.** Until the estimates are right, bootstrapped targets are biased by them. Wrong initial values mislead the early updates of every predecessor.
- **Reliance on the Markov property.** A bootstrapped target assumes that the next state's estimate says everything about the future from there. When states do not capture everything relevant, Monte Carlo targets, which never use another state's estimate, are often more robust.
- **Slow propagation.** A one-step target moves information back one step per update. A reward at the end of a long corridor takes many episodes to reach the start; Monte Carlo would credit the whole corridor in one episode ([[mc-control]]).
- **Instability with approximation.** With function approximation and off-policy training, bootstrapping can make the estimates diverge (Baird, 1995). The combination of the three is the *deadly triad* ([[deadly-triad]]); bootstrapping is the one ingredient that is hardest to give up, because of everything it buys.

### How much to bootstrap {#how-much}

Bootstrapping is not all or nothing. An $n$-step target uses $n$ real rewards and then the estimate of the state reached: $n = 1$ is TD(0), and $n$ equal to the length of the episode is Monte Carlo ([[n-step-td]]). The λ-return mixes all of them, with $\lambda$ setting how far ahead the target looks before relying on estimates ([[lambda-return]]). On most tasks an intermediate amount works best: enough real rewards to dilute the bias of the estimates, few enough to keep the variance down.

### Historical remarks {#history}

Samuel's checkers player (1959) bootstrapped: it adjusted its evaluation of a position toward the evaluation of positions searched from it, an early form of temporal-difference learning. Sutton (1988) formalized learning from successive predictions as temporal-difference learning. Baird (1995) gave the classic example of bootstrapping diverging under off-policy training with linear function approximation.

## Card

### Idea

An update bootstraps when its target contains other estimates. TD and dynamic programming bootstrap; Monte Carlo waits for the real return. Bootstrapping learns online and with less noise, but its targets are only as good as the estimates in them.

::: analogy
Forecasting tomorrow's weather partly from the forecast for the day after. It can be revised every day without waiting, and good forecasts improve each other; but an error in one forecast leaks into the others.
:::

### In symbols {#formula}

$$\text{bootstraps: } \rew{R_{t+1}} + \gam\,\val{V(S_{t+1})} \qquad \text{does not: } \rew{G_t} = \rew{R_{t+1}} + \gam\,\rew{R_{t+2}} + \cdots + \gam^{T-t-1}\,\rew{R_T}$$

In between: $\rew{R_{t+1}} + \cdots + \gam^{n-1}\rew{R_{t+n}} + \gam^n\,\val{V(S_{t+n})}$, the $n$-step target.

### The trade-off {#tradeoff}

| | bootstrapping | full returns |
| --- | --- | --- |
| variance of the target | low | high |
| bias of the target | yes, from the estimates | none |
| learns during the episode | yes | no |
| needs the Markov property | more | less |
| with approximation, off-policy | can diverge | stable |

### Why it matters {#why}

It is the main choice that separates TD learning from Monte Carlo, and one of the three ingredients of the deadly triad. Choosing how much to bootstrap ($n$, $\lambda$) is a central knob of many algorithms.

### Pitfalls

- Bootstrapping from a value that was just updated in the same step: use the estimate from before the update.
- Assuming bootstrapping always helps: with non-Markov states or unstable approximation, full returns may be better.
- Forgetting that a terminal state contributes 0: there is nothing to bootstrap from.

### Check yourself {#check}

::: question
Does value iteration bootstrap? Does Monte Carlo ES?
---
Value iteration does: each value is computed from its successors' values. Monte Carlo ES does not: its targets are returns, made only of observed rewards.
:::

::: question
Why does a bootstrapped target usually have lower variance than a return?
---
It depends on the randomness of one transition only. A return depends on every random action and transition until the end of the episode.
:::

::: question
If bootstrapped targets have lower variance, why not always bootstrap after a single step?
---
One-step targets are biased while the estimates are wrong, and their news travels back only one step per update. With function approximation and off-policy data they can even diverge. An intermediate number of steps, or $\lambda$, is often best.
:::
