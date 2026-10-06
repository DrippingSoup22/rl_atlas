+++
summary = "SARSA with eligibility traces on state–action pairs: every move you make starts to glow and fades by γλ per step, and each TD error updates every glowing move at once. When a reward finally arrives, the whole trail that led to it learns."
change = "Traces on state–action pairs instead of states, and SARSA's TD error R + γQ(S′, A′) − Q(S, A): every action value moves by α·δ·z(s, a), and the agent acts ε-greedily on them."
prereqs = ["sarsa", "td-lambda", "n-step-sarsa"]
lab = "sarsa-lambda"
sources = [
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., §12.7 and §12.10", url = "http://incompleteideas.net/book/the-book-2nd.html" },
  { text = "Munos, Stepleton, Harutyunyan & Bellemare (2016), Safe and efficient off-policy reinforcement learning, Advances in Neural Information Processing Systems 29", url = "https://arxiv.org/abs/1606.02647" },
  { text = "Rummery & Niranjan (1994), On-line Q-learning using connectionist systems, Technical Report CUED/F-INFENG/TR 166, Cambridge University", url = "https://www.researchgate.net/publication/2500611" },
  { text = "Singh & Sutton (1996), Reinforcement learning with replacing eligibility traces, Machine Learning 22", url = "https://doi.org/10.1007/BF00114726" },
  { text = "Watkins (1989), Learning from delayed rewards, PhD thesis, University of Cambridge", url = "https://www.cs.rhul.ac.uk/~chrisw/thesis.html" },
  { text = "Peng & Williams (1996), Incremental multi-step Q-learning, Machine Learning 22", url = "https://doi.org/10.1007/BF00114731" },
  { text = "van Seijen, Mahmood, Pilarski, Machado & Sutton (2016), True online temporal-difference learning, Journal of Machine Learning Research 17", url = "https://jmlr.org/papers/v17/15-599.html" },
]

[story]
scene = "grid"
env = "dyna-maze"
seed = 41
average = 30
range = 0.5
formula = '\step{1}{\del = \rew{R} + \gam\,\val{Q(S^\prime,A^\prime)} - \val{Q(S,A)}} \step{2}{\qquad \val{Q(s,a)} \leftarrow \val{Q(s,a)} + \alp\,\del\,\htmlClass{q-trc}{z(s,a)}}'

[story.runs]
lam = { algorithm = "sarsa-lambda", lambda = 0.9, alpha = 0.1, epsilon = 0.1, gamma = 0.95, units = 50, name = "SARSA(λ), λ = 0.9" }
zero = { algorithm = "sarsa-lambda", lambda = 0.0, alpha = 0.1, epsilon = 0.1, gamma = 0.95, units = 50, name = "SARSA (λ = 0)" }
rep = { algorithm = "sarsa-lambda", lambda = 0.9, trace = "replacing", alpha = 0.1, epsilon = 0.1, gamma = 0.95, units = 50, name = "λ = 0.9, replacing" }
+++

## Story

::: step {q = "zero", agent = "start"}
**The maze where only the gem pays**, and an agent that knows nothing yet. This time it keeps a trace of **every move it makes**: the move glows, and the glow fades by $\gamma\lambda$ with each step after.
:::

::: step {run = "lam", at = 0, play = 1, pace = 150, arrows = false}
The first episode, with $\lambda = 0.9$: watch the green trail of moves build up behind the agent and fade toward its start. No reward yet, so every TD error is 0 and nothing is learned.
:::

::: step {run = "lam", at = 1, arrows = false, formula = 2}
Then the gem: one TD error of $\rew{+1}$, shared out along the whole glowing trail. Every move of the episode learned something, the recent ones the most. **27 action values** rose from a single reward.
:::

::: step {run = "zero", at = 1, arrows = false}
With $\lambda = 0$ the trace lasts one step, and SARSA(λ) is plain [[sarsa]]. From the same episode, one action value changed: the last one.
:::

::: step {run = "lam", at = 2, arrows = false}
The difference shows in the next episode. Following the trail, SARSA(λ) reaches the gem in **37 moves**; plain SARSA, still searching, needs 354.
:::

::: step {run = "rep", at = 50, curves = ["zero", "lam", "rep"], metric = "steps"}
Averaged over 30 runs. Moves made again and again build up accumulating traces above 1; replacing traces stop at 1 and here learn a little faster still. [Watch the trails in the Lab](lab:sarsa-lambda).
:::

## Textbook

### Traces for control {#idea}

[[td-lambda]] learns state values with eligibility traces; **SARSA(λ)** does the same for action values, and acts $\varepsilon$-greedily with respect to them, as [[sarsa]] does. Its forward view is the λ-return built from $n$-step SARSA returns ([[n-step-sarsa]]),

$$\rew{G^\lambda_t} = (1 - \lam) \sum_{n=1}^{T-t-1} \lam^{n-1}\,\rew{G_{t:t+n}} + \lam^{T-t-1}\,\rew{G_t}, \qquad \rew{G_{t:t+n}} = \rew{R_{t+1}} + \cdots + \gam^{n-1}\rew{R_{t+n}} + \gam^n\,\val{Q(S_{t+n}, A_{t+n})}, \label{forward}$$

and its backward view keeps one trace per state–action pair. It is on-policy: the traces follow the actions actually taken, exploratory ones included.

### The update {#update}

At each step, every trace fades by $\gamma\lambda$ and the trace of the pair just taken grows by 1 (accumulating) or is set to 1 (replacing). The TD error is SARSA's, and every action value moves by its share:

$$\del_t = \rew{R_{t+1}} + \gam\,\val{Q(S_{t+1}, A_{t+1})} - \val{Q(S_t, A_t)}, \qquad \val{Q(s, a)} \leftarrow \val{Q(s, a)} + \alp\,\del_t\,\htmlClass{q-trc}{z_t(s, a)} \ \text{ for all } s, a. \label{eq-update}$$

With $\lambda = 0$ only the pair just taken has a trace, and the update is SARSA's. With $\lambda = 1$ the credit fades only by $\gamma$, like the rewards of a return: on-policy Monte Carlo control done online.

::: algorithm {#alg-sarsal} SARSA(λ)
Input: a step size $\alp \in (0, 1]$, a small $\varepsilon > 0$ and a trace decay $\lam \in [0, 1]$
Set $\val{Q(s, a)}$ to any value for every state and action, with $\val{Q(\textit{terminal}, \cdot)} = 0$
Repeat for each episode:
  Start in $S$; choose $A$ $\varepsilon$-greedily from $\val{Q}$; set every trace $\htmlClass{q-trc}{z(s, a)} \leftarrow 0$
  Until $S$ is terminal:
    Take $A$, observe $\rew{R}$ and $S'$; if $S'$ is not terminal, choose $A'$ $\varepsilon$-greedily from $\val{Q}$
    $\del \leftarrow \rew{R} + \gam\,\val{Q(S', A')} - \val{Q(S, A)}$ (with $\val{Q(S', A')} = 0$ if $S'$ is terminal)
    $\htmlClass{q-trc}{z} \leftarrow \gam\lam\,\htmlClass{q-trc}{z}$ for every pair; then $\htmlClass{q-trc}{z(S, A)} \leftarrow \htmlClass{q-trc}{z(S, A)} + 1$ (accumulating) or $\leftarrow 1$ (replacing)
    $\val{Q(s, a)} \leftarrow \val{Q(s, a)} + \alp\,\del\,\htmlClass{q-trc}{z(s, a)}$ for every pair
    $S \leftarrow S'$; $A \leftarrow A'$
:::

### Example: one episode in a maze {#example}

The effect of traces is easiest to see when rewards are rare. In the Dyna maze ([[dyna-q]]) only the move onto the gem pays, and every action value starts at 0. Over the first episode every TD error is zero, until the last step. That one TD error then updates every move of the episode, in proportion to its trace (\ref{fig-paths}): one-step SARSA learns one action value from the episode, 10-step SARSA ten, SARSA(λ) all of them, by amounts that fade with distance from the gem.

::: figure {#fig-paths}
{{n-step-paths lambda}}
One episode in the Dyna maze, starting with all action values at 0, and the action values it raised: one for one-step SARSA, every move of the episode for SARSA(λ) with λ = 0.9, each arrow as dark as the value it gained. Computed by the Lab. After Sutton & Barto, Figure 7.4 and §12.7.
:::

In the [Lab](lab:sarsa-lambda), averaged over 30 runs, SARSA(λ) needs far fewer steps than SARSA in the early episodes. Replacing traces do a little better than accumulating ones here: on its first episode the agent wanders, repeats the same moves many times, and accumulating traces then give those repeated moves too much of the credit.

### Replacing traces and clearing {#replacing}

With replacing traces, a further option is to clear the traces of the other actions in the state just visited: when $A$ is taken in $S$, set $z(S, a) = 0$ for $a \ne A$. The reasoning is that, having just chosen $A$ in $S$, the agent's earlier choices of other actions in $S$ did not lead to what follows. This variant performed well in early experiments (Singh & Sutton, 1996); with function approximation, where states are not visited exactly, it has no clean counterpart.

### Off-policy traces: Watkins's Q(λ) {#watkins}

Traces combine less easily with off-policy learning. [[q-learning]] learns about the greedy policy while behaving otherwise; a trace that runs back through an exploratory action would credit earlier moves for a future the greedy policy would not have produced. **Watkins's Q(λ)** therefore uses the Q-learning TD error, $\rew{R} + \gam \max_a \val{Q(S', a)} - \val{Q(S, A)}$, and **cuts all traces to zero** whenever the action taken is not greedy. Its traces are short when exploration is frequent, and it loses much of the benefit of traces early in learning. **Peng's Q(λ)** does not cut the traces, at the price of learning something between the values of the behavior and the greedy policies (Peng & Williams, 1996). Later methods cut traces gradually instead of all at once. Tree backup multiplies each step's trace by $\pol{\pi(A \mid S)}$, the target policy's probability of the action taken. Retrace multiplies it by $\lambda \min\big(1, \pol{\pi(A \mid S)} / \pol{b(A \mid S)}\big)$: no cut at all when the behavior took an action the target would take at least as often, a full cut only when the target would never take it. It converges for any behavior policy and keeps traces long when the two policies are close (Munos et al., 2016).

### True online SARSA(λ) {#true-online}

As for prediction, the online backward view only approximates the online forward view, increasingly so as the step size grows. **True online SARSA(λ)** (van Seijen et al., 2016) uses dutch traces and a correction term to match the forward view exactly; with tables or binary features it usually learns faster and is less sensitive to the step size than SARSA(λ) with accumulating or replacing traces.

### Historical remarks {#history}

Rummery and Niranjan (1994) introduced SARSA with eligibility traces along with one-step SARSA. Watkins (1989) proposed Q(λ) with traces cut at exploratory actions; Peng and Williams (1996) proposed the uncut variant. Singh and Sutton (1996) introduced replacing traces and the option of clearing the traces of other actions. Van Seijen and colleagues (2016) developed true online SARSA(λ).

## Card

### Idea

SARSA with a fading memory of its moves: every move it makes starts to glow, and the glow dims by γλ each step. Every TD error updates all glowing moves at once, each by its share. A rare reward teaches the whole trail that led to it, not only the last move.

::: analogy
A trail of breadcrumbs that fades with time: when you find the treasure, every crumb still visible gets credit, the freshest ones most.
:::

### The update {#update}

$$\val{Q(s,a)} \leftarrow \val{Q(s,a)} + \alp\,\del\,\htmlClass{q-trc}{z(s,a)} \ \text{ for every pair}, \qquad \del = \rew{R} + \gam\,\val{Q(S',A')} - \val{Q(S,A)}$$

The trace fades by $\gamma\lambda$ each step and grows by 1 (accumulating) or resets to 1 (replacing) on the pair just taken. With λ = 0 it is SARSA.

### One change from TD(λ) {#change}

| | traces on | TD error |
| --- | --- | --- |
| [[td-lambda]] | states, $\htmlClass{q-trc}{z(s)}$ | $\rew{R} + \gam\,\val{V(S')} - \val{V(S)}$ |
| SARSA(λ) | state–action pairs, $\htmlClass{q-trc}{z(s,a)}$ | $\rew{R} + \gam\,\val{Q(S',A')} - \val{Q(S,A)}$ |

With λ = 0 it is [[sarsa]]: only the pair just taken has a trace.

### Backup diagram {#backup}

{{backup lambda-q}}

Every $n$-step SARSA return at once, weighted $(1-\lambda)\lambda^{n-1}$, with what is left on the full return.

### Pseudocode

::: pseudocode
Parameters: step size $\alp$, exploration $\eps$, discount $\gam$, trace decay $\lam$
Set $\val{Q(s,a)}$ for every state and action (here 0) {#init}
Repeat for each episode:
  Start: $S$; every trace $\htmlClass{q-trc}{z(s,a)} \leftarrow 0$ {#start}
  Choose $A$ $\eps$-greedily from $\val{Q}$ {#choose}
  Repeat until $S$ is terminal:
    Take $A$, observe $\rew{R}$ and $S'$ {#act}
    Choose $A'$ $\eps$-greedily from $\val{Q}$ in $S'$ {#choose-next}
    $\del \leftarrow \rew{R} + \gam\,\val{Q(S',A')} - \val{Q(S,A)}$ {#error}
    $\htmlClass{q-trc}{z} \leftarrow \gam\lam\,\htmlClass{q-trc}{z}$ for every pair; $\htmlClass{q-trc}{z(S,A)} \leftarrow \htmlClass{q-trc}{z(S,A)} + 1$ (or $1$, replacing) {#trace}
    $\val{Q(s,a)} \leftarrow \val{Q(s,a)} + \alp\,\del\,\htmlClass{q-trc}{z(s,a)}$ for every pair {#update}
    $S \leftarrow S'$; $A \leftarrow A'$ {#next}
:::

### Perks

- One reward teaches the whole trail of moves that led to it. [See it](lab:sarsa-lambda)
- Much faster than SARSA when rewards are rare and delayed.
- Online and on-policy: no waiting, no model, works for tasks that never end.

### Flaws

- More work per step: every pair with a trace is updated.
- Off-policy versions must cut or shrink traces at exploratory moves, losing much of the gain.
- Accumulating traces can overshoot on moves repeated many times; replacing or dutch traces are safer.

### Knobs

| Knob | Too low | Too high |
| --- | --- | --- |
| $\lam$ trace decay | credit stays on the last move, like SARSA | credit spread over long detours; noisy |
| $\alp$ step size | learns slowly | overshoots, especially with accumulating traces |
| $\eps$ exploration | may miss better paths | traces full of detours |

### Pitfalls

- Forgetting to reset the traces at the start of each episode.
- Using the trace of the state only: traces must be per state–action pair, or credit goes to actions never taken.
- Mixing in Q-learning's max without cutting the traces: the result is neither SARSA(λ) nor Watkins's Q(λ).

### Check yourself {#check}

::: question
In a maze where only the gem pays and all values start at 0, how many action values does the first episode change with SARSA(λ), λ > 0? With SARSA?
---
With λ > 0, every pair taken during the episode: they all have a trace when the one nonzero TD error arrives at the gem. SARSA changes only the last pair.
:::

::: question
Why does Watkins's Q(λ) cut its traces after an exploratory action?
---
It learns the values of the greedy policy. After a non-greedy action, what follows is not what the greedy policy would have produced, so earlier pairs should not be credited or blamed for it.
:::

::: question
What do replacing traces change, compared with accumulating ones, for a move made five times in a row?
---
An accumulating trace adds up the visits and can grow well above 1, so that move gets most of the credit; a replacing trace is reset to 1 at each visit and never exceeds it.
:::
