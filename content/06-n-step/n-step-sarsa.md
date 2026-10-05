+++
summary = "SARSA with an n-step target: move the value of a state and action toward the next n rewards plus the discounted value of the state and action reached n steps later. When a reward finally comes, the last n moves all learn from it at once."
change = "Learn action values instead of state values, and act ε-greedily on them: the n-step return ends with γ^n Q(S_{t+n}, A_{t+n}), the value of the action the agent actually took n steps later."
prereqs = ["sarsa", "n-step-td"]
lab = "n-step-sarsa"
sources = [
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., §7.2–7.5 and Figure 7.4", url = "http://incompleteideas.net/book/the-book-2nd.html" },
  { text = "Rummery & Niranjan (1994), On-line Q-learning using connectionist systems, Technical Report CUED/F-INFENG/TR 166, Cambridge University", url = "https://www.researchgate.net/publication/2500611" },
  { text = "Precup, Sutton & Singh (2000), Eligibility traces for off-policy policy evaluation, Proceedings of the 17th International Conference on Machine Learning" },
  { text = "De Asis, Hernandez-Garcia, Holland & Sutton (2018), Multi-step reinforcement learning: a unifying algorithm, Proceedings of AAAI", url = "https://arxiv.org/abs/1703.01327" },
]

[story]
scene = "grid"
env = "dyna-maze"
seed = 41
average = 30
range = 0.5
formula = '\step{1}{\val{Q(S_t,A_t)} \leftarrow \val{Q(S_t,A_t)} + \alp\,\big[} \step{2}{\rew{R_{t+1}} + \cdots + \gam^{n-1}\rew{R_{t+n}}} \step{3}{+ \gam^n\,\val{Q(S_{t+n},A_{t+n})}} \step{4}{- \val{Q(S_t,A_t)}\,\big]}'

[story.runs]
one = { algorithm = "n-step-sarsa", n = 1, alpha = 0.1, epsilon = 0.1, gamma = 0.95, units = 50, name = "n = 1 (SARSA)" }
four = { algorithm = "n-step-sarsa", n = 4, alpha = 0.1, epsilon = 0.1, gamma = 0.95, units = 50, name = "n = 4" }
ten = { algorithm = "n-step-sarsa", n = 10, alpha = 0.1, epsilon = 0.1, gamma = 0.95, units = 50, name = "n = 10" }
+++

## Story

::: step {q = "zero", agent = "start"}
**A maze where only the gem pays.** Every move earns nothing, except the one onto the gem, worth $\rew{+1}$; future rewards are discounted by $\gamma = 0.95$. The agent knows nothing yet: every action value is 0.
:::

::: step {run = "one", at = 0, play = 1, pace = 110, arrows = false}
Watch the first episode. Knowing nothing, the agent wanders; this time it finds the gem after 30 moves.
:::

::: step {run = "one", at = 1, arrows = false}
One-step SARSA learned from **one** of those 30 moves: the last, the move onto the gem, now worth a little (the blue triangle by the gem). Every other move was followed by a reward of 0 and a next value of 0, so nothing changed.
:::

::: step {run = "ten", at = 1, arrows = false, formula = 3}
With $n = 10$, each move waits ten steps and adds up the rewards it sees on the way. The last **ten** moves of the episode all saw the gem within their ten steps, so all ten learned at once, each a little less than the one after it, discounted by $\gamma$ per step.
:::

::: step {run = "ten", at = 2, arrows = false, formula = 4}
That head start pays off in the second episode: once the agent comes across the trail of strengthened moves, it follows it to the gem. Its second episode takes 252 moves, against 354 for one-step SARSA.
:::

::: step {run = "four", at = 50, curves = ["one", "four", "ten"], metric = "steps"}
Averaged over 30 runs, the longer looks find the way home in fewer episodes. [Race them in the Lab](lab:n-step-sarsa), or compare with the trails of [[sarsa-lambda]].
:::

## Textbook

### n-step returns for control {#idea}

The $n$-step returns of [[n-step-td]] carry over to control the same way the one-step target of [[td0]] carried over to [[sarsa]]: estimate action values instead of state values, and keep acting $\varepsilon$-greedily with respect to them. The resulting method, **$n$-step SARSA**, is on-policy: the actions inside the $n$-step return are the ones the agent actually took, so it learns the values of the policy it follows, exploration included. One-step SARSA is the case $n = 1$, and the case $n = \infty$ is on-policy Monte Carlo control with a constant step size ([[mc-control]]).

### The update {#update}

The $n$-step return in action values counts the next $n$ rewards and then bootstraps from the value of the state and action reached:

$$\rew{G_{t:t+n}} = \rew{R_{t+1}} + \gam\rew{R_{t+2}} + \cdots + \gam^{n-1}\rew{R_{t+n}} + \gam^n\,\val{Q_{t+n-1}(S_{t+n}, A_{t+n})}, \qquad n \ge 1,\ 0 \le t < T - n, \label{nret}$$

with $\rew{G_{t:t+n}} = \rew{G_t}$ when $t + n \ge T$. The update, made at time $t + n$, is

$$\val{Q_{t+n}(S_t, A_t)} = \val{Q_{t+n-1}(S_t, A_t)} + \alp\,\big[\,\rew{G_{t:t+n}} - \val{Q_{t+n-1}(S_t, A_t)}\,\big], \label{eq-update}$$

and all other action values stay as they are. As for prediction, the first $n - 1$ steps of an episode bring no update and the last $n - 1$ updates come after the episode ends.

### The algorithm {#algorithm}

::: algorithm {#alg-nsarsa} $n$-step SARSA
Input: a step size $\alp \in (0, 1]$, a small $\varepsilon > 0$ and a positive integer $n$
Set $\val{Q(s, a)}$ to any value for every state and action, with $\val{Q(\textit{terminal}, \cdot)} = 0$
Repeat for each episode:
  Start in $S_0$; choose $A_0$ $\varepsilon$-greedily from $\val{Q}$; set $T \leftarrow \infty$
  For $t = 0, 1, 2, \ldots$ until $\tau = T - 1$:
    If $t < T$:
      Take $A_t$, observe $\rew{R_{t+1}}$ and $S_{t+1}$
      If $S_{t+1}$ is terminal: $T \leftarrow t + 1$; else choose $A_{t+1}$ $\varepsilon$-greedily from $\val{Q}$
    $\tau \leftarrow t - n + 1$
    If $\tau \ge 0$:
      $\rew{G} \leftarrow \sum_{i=\tau+1}^{\min(\tau+n,\,T)} \gam^{\,i-\tau-1}\,\rew{R_i}$
      If $\tau + n < T$: $\rew{G} \leftarrow \rew{G} + \gam^n\,\val{Q(S_{\tau+n}, A_{\tau+n})}$
      $\val{Q(S_\tau, A_\tau)} \leftarrow \val{Q(S_\tau, A_\tau)} + \alp\,[\rew{G} - \val{Q(S_\tau, A_\tau)}]$
:::

::: figure {#fig-backup}
{{backup n-step-sarsa}}
The backup diagram of $n$-step SARSA: from a state and action, $n$ sampled steps, then the value of the state and action reached.
:::

### Example: one episode in a maze {#example}

The speed-up is clearest when rewards are rare. In the Dyna maze ([[dyna-q]]), only the move onto the gem pays, and every action value starts at 0. On the first episode the agent wanders until it stumbles on the gem. One-step SARSA then strengthens a single action, the last one; $n$-step SARSA strengthens the last $n$ (\ref{fig-paths}). The next episode starts with much more to go on.

::: figure {#fig-paths}
{{n-step-paths}}
One episode in the Dyna maze, starting with all action values at 0, and the actions whose values it raised: the last one for one-step SARSA, the last ten for 10-step SARSA. Computed by the Lab. After Sutton & Barto, Figure 7.4.
:::

Averaged over runs, the gain carries through the first few dozen episodes (see the [Lab](lab:n-step-sarsa)). Later, when the agent knows the way, longer returns bring more noise than news: with $\varepsilon$-greedy exploration, a long return is more likely to contain an exploratory detour, and on-policy methods learn the cost of those detours.

### n-step Expected SARSA {#expected}

The last step of the return can average over the next action instead of sampling it, as in [[expected-sarsa]]:

$$\rew{G_{t:t+n}} = \rew{R_{t+1}} + \cdots + \gam^{n-1}\rew{R_{t+n}} + \gam^n \sum_a \pol{\pi(a \mid S_{t+n})}\,\val{Q_{t+n-1}(S_{t+n}, a)}. \label{eq-expected}$$

All the earlier actions in the return are still sampled. The variance saved is smaller than for one step, since only one of $n$ actions is averaged, but the target is still steadier than $n$-step SARSA's.

### Off-policy n-step learning {#off-policy}

To learn about a target policy $\pol{\pi}$ while behaving with another policy $\pol{b}$, the $n$-step return must be corrected for the actions inside it, which $\pol{\pi}$ might have chosen differently. The correction is the importance sampling ratio over those actions ([[importance-sampling]]):

$$\rho_{t+1:t+n} = \prod_{k=t+1}^{\min(t+n,\,T-1)} \frac{\pol{\pi(A_k \mid S_k)}}{\pol{b(A_k \mid S_k)}}, \qquad \val{Q(S_t, A_t)} \leftarrow \val{Q(S_t, A_t)} + \alp\,\rho_{t+1:t+n}\,\big[\,\rew{G_{t:t+n}} - \val{Q(S_t, A_t)}\,\big]. \label{offpolicy}$$

The first action $A_t$ needs no correction: its value is being learned whatever it is. For a greedy target policy the ratio is zero as soon as the behavior makes an exploratory move, so only stretches of greedy behavior teach anything, and the ratios can make the updates very variable. The **tree-backup** algorithm avoids ratios altogether: at each step of the return it averages over the actions not taken, weighted by $\pol{\pi}$, and follows the action taken with weight $\pol{\pi(A_k \mid S_k)}$. A unifying algorithm, $Q(\sigma)$, chooses at each step whether to sample or to average (De Asis et al., 2018).

### Historical remarks {#history}

SARSA itself was introduced by Rummery and Niranjan (1994), under the name modified Q-learning, together with versions using eligibility traces ([[sarsa-lambda]]). Off-policy $n$-step methods with importance sampling and the tree-backup algorithm are due to Precup, Sutton and Singh (2000). De Asis, Hernandez-Garcia, Holland and Sutton (2018) unified sampling and expectation in $Q(\sigma)$.

## Card

### Idea

SARSA with a longer look: before updating a move's value, wait $n$ steps, add up the rewards you collect, then add the discounted value of the move you are making by then. When a rare reward arrives, the last $n$ moves all learn from it at once instead of only the last one.

::: analogy
After finally finding the restaurant, remembering the last ten turns that led to it, not only the last one.
:::

### The update {#update}

$$\val{Q(S_t,A_t)} \leftarrow \val{Q(S_t,A_t)} + \alp\,\big[\,\rew{G_{t:t+n}} - \val{Q(S_t,A_t)}\,\big], \qquad \rew{G_{t:t+n}} = \rew{R_{t+1}} + \cdots + \gam^{n-1}\rew{R_{t+n}} + \gam^n\,\val{Q(S_{t+n},A_{t+n})}$$

*New guess = old guess + step size × (target − old guess)*, made $n$ steps late. If the episode ends within the $n$ steps, the target is the plain return.

### One change from n-step TD {#change}

| | learns | target ends with |
| --- | --- | --- |
| [[n-step-td]] | $\val{V(s)}$ of a given policy | $\gam^n\,\val{V(S_{t+n})}$ |
| $n$-step SARSA | $\val{Q(s,a)}$, acting $\eps$-greedily on it | $\gam^n\,\val{Q(S_{t+n},A_{t+n})}$ |

With $n = 1$ it is [[sarsa]].

### Backup diagram {#backup}

{{backup n-step-sarsa}}

From a state and action, $n$ sampled steps, then the value of the state and action reached.

### Pseudocode

::: pseudocode
Parameters: step size $\alp$, exploration $\eps$, discount $\gam$, number of steps $n$
Set $\val{Q(s,a)}$ for every state and action (here 0) {#init}
Repeat for each episode:
  Start: $S_0$; $T \leftarrow \infty$ {#start}
  Choose $A_0$ $\eps$-greedily from $\val{Q}$ {#choose}
  For $t = 0, 1, 2, \ldots$ until $\tau = T - 1$:
    If $t < T$: take $A_t$, observe $\rew{R_{t+1}}$ and $S_{t+1}$; if $S_{t+1}$ is terminal, $T \leftarrow t + 1$ {#act}
    If $t + 1 < T$: choose $A_{t+1}$ $\eps$-greedily from $\val{Q}$ {#choose-next}
    $\tau \leftarrow t - n + 1$; if $\tau \ge 0$: $\rew{G} \leftarrow \sum_{i=\tau+1}^{\min(\tau+n,T)} \gam^{i-\tau-1}\rew{R_i}$, plus $\gam^n\val{Q(S_{\tau+n},A_{\tau+n})}$ if $\tau + n < T$ {#return}
    If $\tau \ge 0$: $\val{Q(S_\tau,A_\tau)} \leftarrow \val{Q(S_\tau,A_\tau)} + \alp\,[\rew{G} - \val{Q(S_\tau,A_\tau)}]$ {#update}
:::

### Perks

- Rare rewards reach $n$ moves back at once: far fewer episodes to find the way. [See it](lab:n-step-sarsa)
- On-policy and simple: one table, no model, no traces.
- A dial between SARSA ($n = 1$) and Monte Carlo control ($n = \infty$).

### Flaws

- Updates wait $n$ steps; the agent acts on values that are $n$ steps out of date.
- Long returns collect exploratory detours: noisier targets once the way is known.
- Off-policy versions need importance sampling ratios, which can make updates very variable.

### Knobs

| Knob | Too low | Too high |
| --- | --- | --- |
| $n$ steps | rare rewards spread slowly | noisy targets full of exploration |
| $\alp$ step size | learns slowly | values chase single returns |
| $\eps$ exploration | may miss better paths | long returns full of detours |

### Pitfalls

- Choosing $A_{t+1}$ after the update instead of before: the target needs the action actually taken $n$ steps later.
- Forgetting the last $n - 1$ updates when the episode ends.
- Expecting $n$-step SARSA to learn the greedy policy's values: like SARSA, it learns those of the exploring policy.

### Check yourself {#check}

::: question
In a maze where only the gem pays and all values start at 0, the first episode takes 30 moves. How many action values does 10-step SARSA change? One-step SARSA?
---
Ten, at most: the moves of the last ten steps, whose ten-step returns contain the gem's reward (fewer if a pair repeats among them). One-step SARSA changes only the value of the last move.
:::

::: question
What is $n$-step SARSA with $n$ longer than every episode?
---
On-policy Monte Carlo control with a constant step size: every target is the full return, with no bootstrapping.
:::

::: question
Off-policy $n$-step learning with a greedy target policy: why does an exploratory move cut learning short?
---
The greedy policy would never have made it, so its probability under the target policy is 0 and the importance sampling ratio of every return containing it is 0.
:::
