+++
summary = "Keep a fading memory of the states you have visited, the eligibility trace. At every step, compute one TD error and move every state by it, in proportion to its trace: recent states learn the most. It learns the λ-return step by step, without waiting."
change = "Instead of waiting n steps for one n-step return, aim at the average of all of them, the λ-return, and learn it online: keep a trace z(s) that fades by γλ each step and grows by 1 on a visit, and move every state by α·δ·z(s) after each step."
prereqs = ["lambda-return", "td0", "td-error"]
lab = "td-lambda-walk"
sources = [
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., §12.2, §12.5 and §12.6", url = "http://incompleteideas.net/book/the-book-2nd.html" },
  { text = "Sutton (1988), Learning to predict by the methods of temporal differences, Machine Learning 3", url = "https://doi.org/10.1007/BF00115009" },
  { text = "Klopf (1972), Brain function and adaptive systems: a heterostatic theory, Air Force Cambridge Research Laboratories" },
  { text = "Singh & Sutton (1996), Reinforcement learning with replacing eligibility traces, Machine Learning 22", url = "https://doi.org/10.1007/BF00114726" },
  { text = "van Seijen & Sutton (2014), True online TD(λ), Proceedings of the 31st International Conference on Machine Learning", url = "http://proceedings.mlr.press/v32/seijen14.html" },
  { text = "Dayan (1992), The convergence of TD(λ) for general λ, Machine Learning 8", url = "https://doi.org/10.1007/BF00992701" },
]

[story]
scene = "chain"
env = "random-walk-19"
seed = 4
average = 100
formula = '\step{1}{\del = \rew{R} + \gam\,\val{V(S^\prime)} - \val{V(S)}} \step{2}{\qquad \htmlClass{q-trc}{z(s)} \leftarrow \gam\lam\,\htmlClass{q-trc}{z(s)}\ (+1 \text{ for } S)} \step{3}{\qquad \val{V(s)} \leftarrow \val{V(s)} + \alp\,\del\,\htmlClass{q-trc}{z(s)}}'

[story.runs]
lam = { algorithm = "td-lambda", lambda = 0.8, alpha = 0.4, gamma = 1.0, units = 20, measures = ["error"], name = "λ = 0.8" }
zero = { algorithm = "td-lambda", lambda = 0.0, alpha = 0.4, gamma = 1.0, units = 20, measures = ["error"], name = "λ = 0 (TD(0))" }
mid = { algorithm = "td-lambda", lambda = 0.4, alpha = 0.4, gamma = 1.0, units = 20, measures = ["error"], name = "λ = 0.4" }
+++

## Story

::: step {run = "lam", at = 0}
**The 19-state walk again**, everything at 0. [[n-step-td|n-step TD]] made each state wait $n$ steps to see what followed it. TD(λ) never waits. Instead, the agent keeps a **memory of where it has been**: every state it passes starts to glow, and the glow fades a little with every step.
:::

::: step {run = "lam", at = 0, play = 1, updates = 12, pace = 260, formula = 2}
Watch the green glow trail the agent over its first twelve steps: the **eligibility trace**. With $\lambda = 0.8$, a state's glow fades by a factor 0.8 per step, so it lasts a few steps. Inside the row nothing is learned yet: every TD error is 0.
:::

::: step {run = "lam", at = 1, formula = 3}
At the exit the TD error is $\rew{+1}$, and it is shared by **every glowing state at once**, each in proportion to its glow. The walk had gone back and forth around P and Q just before leaving, so their glows had built up the most and they learned the most, even more than S; states it left long ago got almost nothing. One walk, one surprise, many states updated.
:::

::: step {run = "zero", at = 1}
With $\lambda = 0$, the glow fades completely in one step: only the state just left is eligible, and TD(λ) is exactly [[td0]]. After the same walk, only S has learned.
:::

::: step {run = "lam", at = 20, ghosts = [1, 5]}
After 20 walks with $\lambda = 0.8$, the estimates are close to the dashed true values. The faint lines are the estimates after 1 and 5 walks.
:::

::: step {run = "lam", at = 20, curves = ["zero", "mid", "lam"], metric = "error"}
Averaged over 100 runs, longer memories learn faster here. Too long, with too large a step size, and the shares overshoot: the textbook has the full picture. [Watch the traces in the Lab](lab:td-lambda-walk).
:::

## Textbook

### From the forward view to the backward view {#idea}

The λ-return is a good target ([[lambda-return]]), but as defined it looks forward in time: the target for a state depends on everything that happens after it, so the update must wait until the end of the episode. **TD(λ)** learns the same thing by looking backward. At each step it computes one TD error, and instead of crediting only the state just left, it credits every state visited recently, in proportion to how recently and how often. The record of “recently and how often” is the **eligibility trace**. The forward view says *what* to learn; the backward view says *how* to learn it online, one step at a time, with the same cost on every step.

### Eligibility traces {#traces}

Next to its table of values $\val{V(s)}$, which changes slowly and is kept for good, the agent keeps a second table of the same size, $\htmlClass{q-trc}{z_t(s)}$, which changes every step and is wiped at the start of each episode. The value table records what the agent believes; the trace table records where it has just been, and so who should share the credit or blame for whatever happens next. Each step, every entry fades by the factor $\gamma\lambda$, and the entry of the state just visited grows by 1:

$$\htmlClass{q-trc}{z_t(s)} = \gam\lam\,\htmlClass{q-trc}{z_{t-1}(s)} + \mathbb{1}[S_t = s], \qquad \htmlClass{q-trc}{z_{-1}(s)} = 0. \label{eq-trace}$$

These are **accumulating traces**: a state visited once, $k$ steps ago, has trace $(\gamma\lambda)^k$, and a state visited several times adds up the faded contributions of all its visits, so its trace can exceed 1. This is why λ is called the trace-decay parameter: it sets how quickly the memory of a visit fades, and with it how far back a surprise reaches.

### The algorithm {#update}

Each step produces one TD error ([[td-error]]),

$$\del_t = \rew{R_{t+1}} + \gam\,\val{V(S_{t+1})} - \val{V(S_t)}, \label{delta}$$

and every state is moved by it, in proportion to its trace:

$$\val{V(s)} \leftarrow \val{V(s)} + \alp\,\del_t\,\htmlClass{q-trc}{z_t(s)} \qquad \text{for every state } s. \label{eq-update}$$

The trace says how much each state is to blame, or to thank, for the surprise just observed. With $\lambda = 0$ the trace is 1 for the current state and 0 for all others, and the update is exactly the TD(0) update ([[td0]]). With $\lambda = 1$ the credit fades only by $\gamma$ per step, as the rewards of a return do: TD(1) behaves like Monte Carlo, but it learns during the episode and also works for tasks that never end.

::: algorithm {#alg-tdl} Tabular TD(λ) for estimating $\val{v_\pi}$, accumulating traces
Input: the policy $\pol{\pi}$ to evaluate, a step size $\alp \in (0, 1]$ and a trace decay $\lam \in [0, 1]$
Set $\val{V(s)}$ to any value for every state, with $\val{V(\textit{terminal})} = 0$
Repeat for each episode:
  Start in $S$; set $\htmlClass{q-trc}{z(s)} \leftarrow 0$ for every state
  Until $S$ is terminal:
    Choose $A$ by $\pol{\pi}$ in $S$; take it, and observe $\rew{R}$ and $S'$
    $\del \leftarrow \rew{R} + \gam\,\val{V(S')} - \val{V(S)}$
    $\htmlClass{q-trc}{z(s)} \leftarrow \gam\lam\,\htmlClass{q-trc}{z(s)}$ for every state; then $\htmlClass{q-trc}{z(S)} \leftarrow \htmlClass{q-trc}{z(S)} + 1$
    $\val{V(s)} \leftarrow \val{V(s)} + \alp\,\del\,\htmlClass{q-trc}{z(s)}$ for every state
    $S \leftarrow S'$
:::

Updating every state on every step sounds expensive, but only states with a noticeable trace need it. With $\gamma\lambda < 1$ the traces of states not visited recently fall below any threshold within a few dozen steps, and implementations keep a short list of the states with nonzero traces.

### Why the two views agree {#equivalence}

The forward view moves $\val{V(S_t)}$ toward $\rew{G^\lambda_t}$. With the estimates held fixed during the episode, that error is a discounted sum of the TD errors that follow ([[lambda-return]]):

$$\rew{G^\lambda_t} - \val{V(S_t)} = \sum_{k=t}^{T-1} (\gam\lam)^{k-t}\,\del_k. \label{eq-errors}$$

So the forward view owes each state $S_t$ a share $(\gamma\lambda)^{k-t}$ of every later TD error $\del_k$. TD(λ) pays exactly those shares as the errors arrive: when $\del_k$ is computed, the trace of $S_t$ holds $(\gamma\lambda)^{k-t}$ (summed over all visits to the same state).

::: theorem {#thm-offline} Equivalence of the offline views
If TD(λ) with accumulating traces adds up its updates during an episode and applies them only at the end (offline TD(λ)), the total update to each state equals that of the offline λ-return algorithm.
:::

::: proof Proof idea
Sum the TD(λ) updates of one state over the episode and exchange the order of the two sums, over the time of the error and over the time of the visit. Each visit at time $t$ collects $\sum_{k \ge t} (\gamma\lambda)^{k-t}\del_k$, which is its λ-return error by \ref{eq-errors}.
:::

Online, when the estimates change during the episode, the two views differ slightly, and the difference grows with the step size. For small α, TD(λ) and the λ-return algorithm behave alike. **True online TD(λ)** (van Seijen & Sutton, 2014) uses a modified trace, the *dutch trace*, and a correction term to match an online version of the forward view exactly, at little extra cost; for tables it is often the better choice.

### Example: the 19-state random walk {#example}

::: figure {#fig-study}
{{lambda-study online}}
Average error of TD(λ) with accumulating traces on the 19-state random walk, against the step size, for several λ. Each point is the RMS error over the 19 states, averaged over the first 10 episodes and over 100 runs. Computed by the Lab. After Sutton & Barto, Figure 12.6.
:::

For small step sizes, \ref{fig-study} is close to the study of the offline λ-return algorithm ([[lambda-return]]): intermediate λ does best, with an error of about 0.27 near λ = 0.8 and α = 0.4. For large λ and α the curves leave the chart and do not come back: the estimates grow without bound. With accumulating traces, a state visited many times in a short stretch builds up a trace above 1, and the update $\alp\,\del\,z(s)$ can then overshoot the target by more than the error it corrects. The offline λ-return algorithm, whose updates each move a state at most α of the way to its target, does not have this problem.

### Replacing traces {#replacing}

A simple remedy is to cap the trace at 1: on a visit, set the trace to 1 instead of adding 1,

$$\htmlClass{q-trc}{z_t(s)} = \begin{cases} 1 & \text{if } S_t = s, \\ \gam\lam\,\htmlClass{q-trc}{z_{t-1}(s)} & \text{otherwise.} \end{cases} \label{eq-replacing}$$

These **replacing traces** (Singh & Sutton, 1996) remember only the most recent visit (\ref{fig-shapes}). They are more robust to large step sizes and often learn faster when states repeat within an episode. Dutch traces, used by true online TD(λ), fall between the two: after fading, a visit adds $1 - \alpha\,z$, where $z$ is the faded trace, instead of 1, so with $\alpha \le 1$ the trace never passes 1.

::: figure {#fig-shapes}
{{trace-shapes}}
The trace of one state visited at steps 2, 4, 5, 6 and 14, with $\gamma\lambda = 0.8$. An accumulating trace adds up the visits and can exceed 1; a replacing trace is reset to 1 at each visit.
:::

### Convergence {#convergence}

For a fixed policy and a table of values, TD(λ) converges to $\val{v_\pi}$ with probability 1 for any $\lambda \in [0, 1]$, under the usual conditions on the step sizes (Dayan, 1992). With linear function approximation, TD(λ) converges to a fixed point whose error is bounded by $\frac{1 - \gamma\lambda}{1 - \gamma}$ times the best possible error: the bound tightens as λ grows, reaching the best possible error at λ = 1. This is one reason to prefer λ near 1 when the features are poor ([[semi-gradient-td]]).

### Historical remarks {#history}

Eligibility traces come from Klopf's (1972) theory of neurons that remain eligible for change for a while after firing. Sutton (1988) introduced TD(λ) with accumulating traces, proved its convergence in the mean and showed the equivalence of its offline version to updating toward λ-returns. Dayan (1992) proved convergence for general λ. Singh and Sutton (1996) introduced replacing traces; van Seijen and Sutton (2014) introduced true online TD(λ) and dutch traces.

## Card

### Idea

Keep a fading memory of where you have been: every state you visit starts to glow, and the glow dims by γλ each step. After every step, compute one TD error and share it among the glowing states in proportion to their glow. It learns the λ-return online, without waiting for anything.

::: analogy
After a pleasant surprise at the end of a walk, giving credit to the last few turns you took, the most recent ones most, instead of only the very last turn.
:::

### The update {#update}

$$\val{V(s)} \leftarrow \val{V(s)} + \alp\,\del\,\htmlClass{q-trc}{z(s)} \quad \text{for every } s, \qquad \htmlClass{q-trc}{z(s)} \leftarrow \gam\lam\,\htmlClass{q-trc}{z(s)} + \mathbb{1}[s = S]$$

One TD error $\del = \rew{R} + \gam\,\val{V(S')} - \val{V(S)}$ per step, shared out by the traces. With λ = 0 it is TD(0).

### One change from n-step TD {#change}

| | target | when |
| --- | --- | --- |
| [[n-step-td]] | one $n$-step return | $n$ steps late |
| TD(λ) | all of them, averaged: the λ-return | at once, one TD error per step shared by the traces |

With λ = 0 it is [[td0]]: only the state just left has a trace.

### Backup diagram {#backup}

{{backup lambda}}

The forward view: every $n$-step return at once, weighted $(1-\lambda)\lambda^{n-1}$, with what is left on the full return.

### Pseudocode

::: pseudocode
Parameters: the policy $\pol{\pi}$, step size $\alp$, discount $\gam$, trace decay $\lam$
Set $\val{V(s)}$ for every state (here 0); a terminal state stays at 0 {#init}
Repeat for each episode:
  Start: $S$; every trace $\htmlClass{q-trc}{z(s)} \leftarrow 0$ {#start}
  Repeat until $S$ is terminal:
    Choose $A$ with $\pol{\pi}$ in $S$ {#choose}
    Take $A$, observe $\rew{R}$ and $S'$ {#act}
    $\del \leftarrow \rew{R} + \gam\,\val{V(S')} - \val{V(S)}$ {#error}
    $\htmlClass{q-trc}{z(s)} \leftarrow \gam\lam\,\htmlClass{q-trc}{z(s)}$ for every state; $\htmlClass{q-trc}{z(S)} \leftarrow \htmlClass{q-trc}{z(S)} + 1$ {#trace}
    $\val{V(s)} \leftarrow \val{V(s)} + \alp\,\del\,\htmlClass{q-trc}{z(s)}$ for every state {#update}
    $S \leftarrow S'$ {#next}
:::

### Perks

- Learns at every step, with no delay: the λ-return without waiting for it. [See it](lab:td-lambda-walk)
- One surprise updates every recently visited state: rewards spread fast.
- Works for tasks that never end, even with λ = 1.

### Flaws

- More work per step: every state with a trace is updated.
- Accumulating traces can overshoot with large α and λ, and diverge.
- Online, it only approximates the λ-return algorithm (true online TD(λ) fixes this).

### Knobs

| Knob | Too low | Too high |
| --- | --- | --- |
| $\lam$ trace decay | credit stays local, like TD(0) | noisy, Monte Carlo-like updates; may overshoot |
| $\alp$ step size | learns slowly | overshoots; with large λ, diverges |

### Pitfalls

- Forgetting to reset the traces at the start of each episode.
- Decaying the traces after adding the current visit instead of before: the current state's trace should be 1 more than its faded value.
- Updating only the current state with $\alp\,\del\,z$: every state with a trace must be updated.

### Check yourself {#check}

::: question
A state was visited 3 steps ago and not since, with γ = 1 and λ = 0.8. What is its trace, and how much does it move when the TD error is +1 and α = 0.4?
---
Its trace is $0.8^3 \approx 0.51$, so it moves by $0.4 \times 1 \times 0.51 \approx 0.2$.
:::

::: question
What is TD(λ) when λ = 0?
---
TD(0): every trace fades to zero in one step, so only the state just left has a trace, equal to 1, and only it is updated, by $\alp\,\del$.
:::

::: question
Why can accumulating traces make TD(λ) diverge with large step sizes, while replacing traces are safer?
---
A state visited many times in a short stretch accumulates a trace above 1, so its update $\alp\,\del\,z$ can exceed the error it corrects and overshoot. A replacing trace never exceeds 1.
:::
