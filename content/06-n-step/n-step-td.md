+++
summary = "Look n steps ahead before trusting a guess: move a state's value toward the next n rewards plus the discounted estimate of the state reached n steps later. With n = 1 it is TD(0); with n as long as the episode it is Monte Carlo."
change = "The target uses n real rewards before it bootstraps, R_{t+1} + γR_{t+2} + … + γ^{n−1}R_{t+n} + γ^n V(S_{t+n}), instead of one reward and the next state's estimate; the update of a state waits n steps."
prereqs = ["td0", "mc-prediction", "bootstrapping"]
lab = "n-step-walk"
sources = [
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., §7.1 and Example 7.1", url = "http://incompleteideas.net/book/the-book-2nd.html" },
  { text = "Watkins (1989), Learning from delayed rewards, PhD thesis, University of Cambridge", url = "https://www.cs.rhul.ac.uk/~chrisw/thesis.html" },
  { text = "Sutton (1988), Learning to predict by the methods of temporal differences, Machine Learning 3", url = "https://doi.org/10.1007/BF00115009" },
  { text = "Cichosz (1995), Truncating temporal differences: on the efficient implementation of TD(λ) for reinforcement learning, Journal of Artificial Intelligence Research 2", url = "https://doi.org/10.1613/jair.135" },
]

[story]
scene = "chain"
env = "random-walk-19"
seed = 4
average = 100
formula = '\step{1}{\val{V(S_t)} \leftarrow \val{V(S_t)} + \alp\,\big[} \step{2}{\rew{R_{t+1}} + \gam\rew{R_{t+2}} + \cdots + \gam^{n-1}\rew{R_{t+n}}} \step{3}{+ \gam^n\,\val{V(S_{t+n})}} \step{4}{- \val{V(S_t)}\,\big]}'
numbers = '\val{V(P)} \leftarrow 0 + 0.4\,\big[\rew{0} + \rew{0} + \rew{0} + \rew{1} - \val{0}\big] = \val{0.4}'

[story.runs]
one = { algorithm = "n-step-td", n = 1, alpha = 0.4, gamma = 1.0, units = 20, measures = ["error"], name = "n = 1 (TD(0))" }
four = { algorithm = "n-step-td", n = 4, alpha = 0.4, gamma = 1.0, units = 20, measures = ["error"], name = "n = 4" }
many = { algorithm = "n-step-td", n = 32, alpha = 0.4, gamma = 1.0, units = 20, measures = ["error"], name = "n = 32" }
+++

## Story

::: step {run = "one", at = 0}
**Nineteen states in a row** this time, A to S, between an exit that pays $\rew{-1}$ on the left and one that pays $\rew{+1}$ on the right. The walk starts in the middle, at J, and steps left or right at random. The true values, the dashed steps, climb from −0.9 to +0.9. Every estimate starts at 0.
:::

::: step {run = "one", at = 0, play = 1, pace = 70}
First, one-step TD, $n = 1$. Watch a whole walk: it wanders for 104 steps before leaving on the right.
:::

::: step {run = "one", at = 1}
All that walking, and **one state** learned something: S, the last one before the exit. Inside the row every target was $0 + 0$, equal to the estimate. A reward has to travel back one state per visit, and nineteen states is a long way.
:::

::: step {run = "four", at = 0, play = 1, pace = 70, formula = 3, fine = true}
Now $n = 4$. Each state waits **four steps** before it is updated, and uses the four real rewards it collected on the way, then the estimate of where it got to. The bracket under the row shows the stretch that makes each target.
:::

::: step {run = "four", at = 1, formula = 4, numbers = true}
After the same walk, **four states** have moved: P, Q, R and S each saw the $\rew{+1}$ within their four steps. One walk taught four times as much.
:::

::: step {run = "four", at = 20, ghosts = [1, 5]}
After 20 walks, $n = 4$ has a fair picture of the whole row. The faint lines are its estimates after 1 and 5 walks.
:::

::: step {run = "four", at = 20, curves = ["one", "four", "many"], metric = "error"}
Bigger is not always better. With $n = 32$ almost every target is the walk's actual outcome, a lone $\rew{\pm 1}$, and each one jerks the estimates around. Averaged over 100 runs, the middle setting learns fastest. [Try other values of n in the Lab](lab:n-step-walk).
:::

## Textbook

### Between one step and the whole episode {#idea}

TD(0) looks one step ahead and then bootstraps ([[td0]]); Monte Carlo looks all the way to the end of the episode and never bootstraps ([[mc-prediction]]). Neither extreme is best in general. **$n$-step TD** methods fill the space between them: the target is made of the next $n$ rewards and the estimate of the state reached after them. The two familiar methods are the end points of the family, $n = 1$ and $n = \infty$. This station is about prediction, estimating $\val{v_\pi}$ for a fixed policy $\pol{\pi}$; the control method built on the same target is [[n-step-sarsa]].

There is a second reason to want this dial. In a one-step method the length of a time step plays two roles at once: it is how often the agent acts, and it is how far each update looks. A robot arm controlled fifty times a second, whose actions matter over a few seconds, needs fast decisions but slow credit; with one-step updates, a reward takes hundreds of updates to travel back to the decision that earned it. Looking $n$ steps ahead separates the two: act as often as the task needs, and let each update reach as far as its consequences.

### The n-step return {#return}

Within an episode $S_0, A_0, R_1, S_1, A_1, R_2, \ldots, R_T, S_T$, the Monte Carlo target for $\val{V(S_t)}$ is the full return, and the TD(0) target truncates it after one reward:

$$\rew{G_t} = \rew{R_{t+1}} + \gam\rew{R_{t+2}} + \cdots + \gam^{T-t-1}\rew{R_T}, \qquad \rew{G_{t:t+1}} = \rew{R_{t+1}} + \gam\,\val{V_t(S_{t+1})}. \label{ends}$$

The subscript $t{:}t{+}1$ reads “from time $t$, truncated after time $t+1$, with the rest replaced by an estimate”. The **$n$-step return** truncates after $n$ rewards:

$$\rew{G_{t:t+n}} = \rew{R_{t+1}} + \gam\rew{R_{t+2}} + \cdots + \gam^{n-1}\rew{R_{t+n}} + \gam^n\,\val{V_{t+n-1}(S_{t+n})}, \label{nret}$$

for $n \ge 1$ and $0 \le t < T - n$. Here $\val{V_{t+n-1}}$ is the estimate as it stands when the target is formed. If the episode ends within the $n$ steps ($t + n \ge T$), the missing terms are zero and the $n$-step return is the full return, $\rew{G_{t:t+n}} = \rew{G_t}$.

The $n$-step return needs $\rew{R_{t+n}}$ and $S_{t+n}$, which are known only at time $t + n$. So the update of $\val{V(S_t)}$ is made $n$ steps late:

$$\val{V_{t+n}(S_t)} = \val{V_{t+n-1}(S_t)} + \alp\,\big[\,\rew{G_{t:t+n}} - \val{V_{t+n-1}(S_t)}\,\big], \qquad 0 \le t < T, \label{update}$$

and all other estimates stay as they are. During the first $n - 1$ steps of an episode nothing is updated; to make up for it, the last $n - 1$ updates are made after the episode has ended, with returns that run to the end and contain no estimate.

### Why it works: the error reduction property {#error-reduction}

The $n$-step return is a better target than the estimate it bootstraps from, in a precise sense. Its expectation looks $n$ steps into the true dynamics and only then uses the estimate, discounted by $\gamma^n$:

::: theorem {#thm-error} Error reduction property
For any estimate $V$ and any $n \ge 1$, the worst error of the expected $n$-step return is at most $\gamma^n$ times the worst error of $V$:
$$\max_s \Big|\, \mathbb{E}_\pi\big[\rew{G_{t:t+n}} \mid S_t = s\big] - \val{v_\pi(s)} \Big| \;\le\; \gam^n \max_s \big|\val{V(s)} - \val{v_\pi(s)}\big|.$$
:::

::: proof Proof idea
The first $n$ rewards in $\rew{G_{t:t+n}}$ are real, so their expectation agrees with that of the true return; the two differ only in what follows them. The true return continues with $\gamma^n \val{v_\pi(S_{t+n})}$ in expectation, the $n$-step return with $\gamma^n \val{V(S_{t+n})}$. Their expected difference is therefore $\gamma^n$ times an average of errors $\val{V(s')} - \val{v_\pi(s')}$, which is at most $\gamma^n$ times the largest of them.
:::

In words: however wrong the current estimates are, the $n$-step target is wrong by at most $\gamma^n$ times as much on average, because its first $n$ rewards are real and only what comes after them is guessed. It is the contraction behind iterative policy evaluation ([[policy-evaluation]]), applied $n$ times over, and the basis of the convergence proofs for tabular $n$-step TD. With $\gamma = 1$ the factor is 1 and the bound says nothing by itself; in episodic tasks the target still improves, because every path that ends within $n$ steps contributes a real return with no guess in it.

### The algorithm {#algorithm}

The algorithm keeps the last $n + 1$ states and rewards of the episode. At each time $t$ it acts, then updates the state visited at time $\tau = t - n + 1$, the one whose $n$ rewards have just become complete.

::: algorithm {#alg-nstep} $n$-step TD for estimating $\val{v_\pi}$
Input: the policy $\pol{\pi}$ to evaluate, a step size $\alp \in (0, 1]$ and a positive integer $n$
Set $\val{V(s)}$ to any value for every nonterminal state, and $\val{V(\textit{terminal})} = 0$
Repeat for each episode:
  Start in $S_0$; set $T \leftarrow \infty$
  For $t = 0, 1, 2, \ldots$ until $\tau = T - 1$:
    If $t < T$: take an action chosen by $\pol{\pi}$ in $S_t$, observe $\rew{R_{t+1}}$ and $S_{t+1}$; if $S_{t+1}$ is terminal, $T \leftarrow t + 1$
    $\tau \leftarrow t - n + 1$, the time whose estimate is updated now
    If $\tau \ge 0$:
      $\rew{G} \leftarrow \sum_{i=\tau+1}^{\min(\tau+n,\,T)} \gam^{\,i-\tau-1}\,\rew{R_i}$
      If $\tau + n < T$: $\rew{G} \leftarrow \rew{G} + \gam^n\,\val{V(S_{\tau+n})}$
      $\val{V(S_\tau)} \leftarrow \val{V(S_\tau)} + \alp\,[\rew{G} - \val{V(S_\tau)}]$
:::

Each step still costs one update, but the target now takes $O(n)$ work to compute (or $O(1)$ with a running sum), and the method needs memory for the last $n$ states and rewards. The backup diagram is a chain of $n$ sampled transitions ending in an estimate (\ref{fig-backup}).

::: figure {#fig-backup}
{{backup n-step-td}}
The backup diagram of $n$-step TD: $n$ sampled steps, then the estimate of the state reached. With $n = 1$ it is the diagram of TD(0); when the chain reaches the end of the episode, that of Monte Carlo.
:::

### Example: the 19-state random walk {#example}

::: example {#ex-walk} A longer random walk (Sutton & Barto, Example 7.1)
The random walk of [[td0]] with 19 nonterminal states instead of 5. The walk starts in the center state and steps left or right with equal probability; leaving on the left pays $\rew{-1}$, on the right $\rew{+1}$, and every other reward is zero. Undiscounted, the true values rise in equal steps from $-0.9$ to $+0.9$. All estimates start at 0.
:::

Nineteen states make the effect of $n$ easy to see, because news from the exits has a long way to travel. In one run, the first walk takes 104 steps and leaves on the right. One-step TD changes one estimate, that of the last state; four-step TD changes four, the last four states visited, each of which had the exit within its four-step window. The same walk taught four times as much.

::: figure {#fig-study}
{{n-step-study}}
Average error of $n$-step TD on the 19-state random walk, against the step size, for several values of $n$. Each point is the RMS error over the 19 states, averaged over the first 10 episodes and over 100 runs. Computed by the Lab. After Sutton & Barto, Figure 7.2.
:::

\ref{fig-study} measures how well each setting does early in learning. The best results come from intermediate $n$: around $n = 4$ with $\alp \approx 0.4$, the error is about 0.26, against about 0.34 for the best one-step TD and much more for $n = 256$, which in this task is nearly Monte Carlo. Larger $n$ also prefers smaller step sizes: a long return is noisy, and a large step follows the noise. The curves rise steeply on the right because, with large $n$ and large $\alp$, each update is driven by a single, highly variable return.

### Bias and variance {#bias-variance}

The choice of $n$ trades the two sources of error discussed in [[mc-vs-td]]. A small $n$ bootstraps early: the target has low variance, because it contains few random rewards and transitions, but it inherits the error of the estimate it uses. A large $n$ bootstraps late: by the error reduction property its bias shrinks like $\gamma^n$ times the current error, but every extra step adds randomness. In between, the target is both reasonably accurate and reasonably steady, which is why intermediate $n$ usually learns fastest. The best $n$ depends on the task, on the step size and on how far learning has progressed: early on, when the estimates are poor, looking further helps more.

### Relatives {#relatives}

- **Control.** Replacing state values by action values gives [[n-step-sarsa]], and with it an $n$-step version of [[expected-sarsa]].
- **Averaging over n.** Instead of choosing one $n$, the [[lambda-return]] averages all of them with geometric weights. Its incremental implementation, [[td-lambda]], removes the $n$-step delay.
- **Off-policy.** $n$-step returns can be corrected for a different behavior policy by importance sampling ratios over the $n$ steps ([[importance-sampling]]), or without ratios by the tree-backup algorithm, which mixes sampled and expected actions.

### Historical remarks {#history}

The idea of cutting a return off after $n$ steps and finishing it with an estimate is due to Watkins (1989), whose thesis also gave the error reduction argument above. It came a year after Sutton (1988) had introduced TD(λ), which, as Watkins showed, averages all such returns ([[lambda-return]]). Cichosz (1995) used truncated returns to implement TD(λ) efficiently. The 19-state random walk used here is simply the random walk of [[td0]] made longer, so that rewards have far to travel and the choice of $n$ shows.

## Card

### Idea

Before updating a state, wait $n$ steps. Use the $n$ real rewards collected on the way, then trust the estimate of the state you reached. A reward now reaches $n$ states back in one update instead of one, and $n$ is a dial between TD(0) and Monte Carlo.

::: analogy
Judging a chess move by playing $n$ more moves and then evaluating the position, instead of evaluating right after the move (one step) or playing the game to the end (Monte Carlo).
:::

### The update {#update}

$$\val{V(S_t)} \leftarrow \val{V(S_t)} + \alp\,\big[\,\rew{G_{t:t+n}} - \val{V(S_t)}\,\big], \qquad \rew{G_{t:t+n}} = \rew{R_{t+1}} + \gam\rew{R_{t+2}} + \cdots + \gam^{n-1}\rew{R_{t+n}} + \gam^n\,\val{V(S_{t+n})}$$

*New guess = old guess + step size × (target − old guess)*, with the $n$-step return $\rew{G_{t:t+n}}$ as the target. It is made at time $t + n$. If the episode ends first, the target is the plain return.

### One change from TD(0) {#change}

| | target | made at |
| --- | --- | --- |
| [[td0]] | $\rew{R_{t+1}} + \gam\,\val{V(S_{t+1})}$ | time $t + 1$ |
| $n$-step TD | $\rew{R_{t+1}} + \cdots + \gam^{n-1}\rew{R_{t+n}} + \gam^n\,\val{V(S_{t+n})}$ | time $t + n$ |
| [[mc-prediction]] | $\rew{G_t}$, all rewards to the end | the end of the episode |

### Backup diagram {#backup}

{{backup n-step-td}}

$n$ sampled steps, then the estimate of the state reached.

### Pseudocode

::: pseudocode
Parameters: the policy $\pol{\pi}$, step size $\alp$, discount $\gam$, number of steps $n$
Set $\val{V(s)}$ for every state (here 0); a terminal state stays at 0 {#init}
Repeat for each episode:
  Start: $S_0$; $T \leftarrow \infty$ {#start}
  For $t = 0, 1, 2, \ldots$ until $\tau = T - 1$:
    If $t < T$: choose $A_t$ with $\pol{\pi}$ in $S_t$ {#choose}
    If $t < T$: take it, observe $\rew{R_{t+1}}$ and $S_{t+1}$; if $S_{t+1}$ is terminal, $T \leftarrow t + 1$ {#act}
    $\tau \leftarrow t - n + 1$; if $\tau \ge 0$: $\rew{G} \leftarrow \sum_{i=\tau+1}^{\min(\tau+n, T)} \gam^{i-\tau-1}\rew{R_i}$, plus $\gam^n\val{V(S_{\tau+n})}$ if $\tau + n < T$ {#return}
    If $\tau \ge 0$: $\val{V(S_\tau)} \leftarrow \val{V(S_\tau)} + \alp\,[\rew{G} - \val{V(S_\tau)}]$ {#update}
:::

### Perks

- Rewards spread $n$ states back per update: much faster than TD(0) on long chains. [See it](lab:n-step-walk)
- An intermediate $n$ usually beats both TD(0) and Monte Carlo.
- Targets come with a guarantee: their error shrinks by $\gamma^n$ against the estimate they use.

### Flaws

- Each update waits $n$ steps, and the last $n - 1$ wait for the end of the episode.
- Needs memory for the last $n$ states and rewards.
- The best $n$ depends on the task and the step size, and must be tuned.

### Knobs

| Knob | Too low | Too high |
| --- | --- | --- |
| $n$ steps | slow to spread news, like TD(0) | noisy targets, like Monte Carlo |
| $\alp$ step size | learns slowly | estimates follow the noise; large $n$ needs smaller $\alp$ |

### Pitfalls

- Forgetting the updates after the episode ends: the last $n - 1$ states still need theirs.
- Bootstrapping past the end: once $\tau + n \ge T$, the target is the plain return, with no estimate added.
- Comparing values of $n$ at the same step size: each $n$ has its own best $\alp$.

### Check yourself {#check}

::: question
All estimates start at 0, and the first walk leaves on the right after many steps. How many states does four-step TD change, and which ones?
---
At most four: the states visited in the last four steps, whose four-step windows reached the exit and its reward of +1. If one of them was visited twice in those steps, fewer distinct states change, but that one changes twice.
:::

::: question
Which values of $n$ give TD(0) and Monte Carlo?
---
$n = 1$ gives TD(0). An $n$ at least as long as the episode gives Monte Carlo: every target runs to the end and contains no estimate.
:::

::: question
Why does a larger $n$ usually need a smaller step size?
---
Its target contains more random rewards and transitions, so it varies more from episode to episode. A large step size would follow that noise; a smaller one averages over more episodes.
:::
