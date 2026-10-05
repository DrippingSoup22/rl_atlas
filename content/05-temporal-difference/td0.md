+++
summary = "Learn a guess from a guess: after every single step, move the value of the state you left toward the reward plus the value of the state you reached."
change = "Update after every step toward R + γV(S′), the reward plus the current estimate of the next state, instead of waiting for the end of the episode to use the actual return G."
prereqs = ["mc-prediction", "policy-evaluation", "bellman"]
lab = "random-walk"
sources = [
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., §6.1–6.2 and Example 6.2", url = "http://incompleteideas.net/book/the-book-2nd.html" },
  { text = "Sutton (1988), Learning to predict by the methods of temporal differences, Machine Learning 3", url = "https://doi.org/10.1007/BF00115009" },
  { text = "Samuel (1959), Some studies in machine learning using the game of checkers, IBM Journal of Research and Development 3", url = "https://doi.org/10.1147/rd.33.0210" },
  { text = "Dayan (1992), The convergence of TD(λ) for general λ, Machine Learning 8", url = "https://doi.org/10.1007/BF00992701" },
  { text = "Jaakkola, Jordan & Singh (1994), On the convergence of stochastic iterative dynamic programming algorithms, Neural Computation 6", url = "https://doi.org/10.1162/neco.1994.6.6.1185" },
  { text = "Tesauro (1995), Temporal difference learning and TD-Gammon, Communications of the ACM 38", url = "https://doi.org/10.1145/203330.203343" },
]

[story]
scene = "chain"
env = "random-walk"
seed = 1
average = 100
formula = '\step{1}{\val{V(S)} \leftarrow \val{V(S)} + \alp\,\big[} \step{2}{\rew{R} + \gam\,\val{V(S^\prime)}} \step{3}{- \val{V(S)}\,\big]}'
numbers = '\val{V(E)} \leftarrow 0.5 + 0.1\,\big[\rew{1} + 1 \cdot \val{0} - \val{0.5}\big] = \val{0.55}'

[story.runs]
td = { algorithm = "td0", alpha = 0.1, gamma = 1.0, v0 = 0.5, units = 100, measures = ["error"], name = "TD(0), α = 0.1" }
mc = { algorithm = "mc-prediction", alpha = 0.02, gamma = 1.0, v0 = 0.5, units = 100, measures = ["error"], name = "Monte Carlo, α = 0.02" }
+++

## Story

::: step {run = "td", at = 0, truth = false}
**Five states in a row**, between two exits. The walk starts in the middle, at C, and steps left or right at random. Leaving on the right pays $\rew{+1}$; leaving on the left pays nothing. How good is each state? Its value is the chance of leaving on the right. Every estimate starts at 0.5, the dots.
:::

::: step {run = "td", at = 0, truth = false, play = 1, pace = 450, formula = 3}
TD(0) does not wait for the end of the walk. After **every step**, it moves the value of the state it left toward the reward plus the value of the state it reached. Watch the first walk: C, D, C, D, E, and out on the right.
:::

::: step {run = "td", at = 1, truth = false, numbers = true, formula = 3}
Only E changed. On every step inside the row, the target, a reward of 0 plus the next state's value 0.5, equals the old value: no surprise, no change. On the last step, E paid $\rew{+1}$ and led to the end, worth 0, so $\val{V(E)}$ moved a tenth of the way toward 1.
:::

::: step {run = "td", at = 10, truth = false, ghosts = [0, 1]}
After 10 walks the news has started to spread inward: the right side has risen, the left side has fallen. Each state learns from its neighbors' estimates, which are themselves still learning.
:::

::: step {run = "td", at = 100, truth = true, ghosts = [0, 1, 10]}
After 100 walks the estimates are close to the true values, the dashed steps: $\tfrac16$ for A up to $\tfrac56$ for E. The faint lines are the estimates after 0, 1 and 10 walks.
:::

::: step {run = "td", at = 100, truth = true, curves = ["td", "mc"], metric = "error"}
Averaged over 100 runs, TD(0) gets closer to the true values than Monte Carlo, which waits for each walk to end and moves every state toward the walk's final outcome. [Race them in the Lab](lab:random-walk).
:::

## Textbook

### Learning a guess from a guess {#idea}

Temporal-difference (TD) learning is a combination of Monte Carlo ideas and dynamic programming ideas. Like Monte Carlo methods, TD methods learn directly from raw experience without a model of the environment's dynamics ([[mc-prediction]]). Like dynamic programming, they update estimates based in part on other learned estimates, without waiting for a final outcome: they **bootstrap** ([[policy-evaluation]], [[bootstrapping]]). This station is about the prediction problem: estimating $\val{v_\pi}$ for a given policy. Control methods built on the same idea are [[sarsa]] and [[q-learning]].

### The update {#update}

A Monte Carlo method must wait until the end of the episode to determine the increment to $\val{V(S_t)}$, because only then is the return $\rew{G_t}$ known. A simple every-visit Monte Carlo method suitable for nonstationary environments is

$$\val{V(S_t)} \leftarrow \val{V(S_t)} + \alp\,\big[\,\rew{G_t} - \val{V(S_t)}\,\big]. \label{mc}$$

TD methods need to wait only until the next time step. At time $t + 1$ they form a target from the observed reward and the estimate of the next state, and update immediately. The simplest TD method makes the update

$$\val{V(S_t)} \leftarrow \val{V(S_t)} + \alp\,\big[\,\rew{R_{t+1}} + \gam\,\val{V(S_{t+1})} - \val{V(S_t)}\,\big] \label{td}$$

on the transition to $S_{t+1}$. The target for the Monte Carlo update is $\rew{G_t}$; the target for the TD update is $\rew{R_{t+1}} + \gam\,\val{V(S_{t+1})}$. This method is called **TD(0)**, or one-step TD, because it is the special case $\lambda = 0$ of TD(λ) ([[td-lambda]]) and of $n$-step TD with $n = 1$ ([[n-step-td]]). If $S_{t+1}$ is terminal, $\val{V(S_{t+1})} = 0$.

The relation between the three families of methods shows in three expressions for the same value ([[bellman]]):

$$\val{v_\pi(s)} = \mathbb{E}_\pi\big[\,\rew{G_t} \mid S_t = s\,\big] = \mathbb{E}_\pi\big[\,\rew{R_{t+1}} + \gam\,\rew{G_{t+1}} \mid S_t = s\,\big] = \mathbb{E}_\pi\big[\,\rew{R_{t+1}} + \gam\,\val{v_\pi(S_{t+1})} \mid S_t = s\,\big]. \label{three}$$

Monte Carlo methods use an estimate of the first expression as their target: a sample return, because the expectation is unknown. Dynamic programming uses an estimate of the last: the expectation is computed from the model, but $\val{v_\pi(S_{t+1})}$ is unknown and the current estimate is used instead. The TD target is an estimate for both reasons: it samples the expected value in the last expression, *and* it uses the current estimate $\val{V}$ instead of the true $\val{v_\pi}$. TD combines the sampling of Monte Carlo with the bootstrapping of dynamic programming.

### The TD error {#td-error-sec}

The quantity in brackets in \ref{td} is a sort of error, measuring the difference between the estimated value of $S_t$ and the better estimate $\rew{R_{t+1}} + \gam\,\val{V(S_{t+1})}$. It is called the **TD error** ([[td-error]]):

$$\del_t = \rew{R_{t+1}} + \gam\,\val{V(S_{t+1})} - \val{V(S_t)}. \label{delta}$$

The TD error at each time is the error in the estimate made at that time; it is available only one step later. If the array $\val{V}$ does not change during the episode, as in Monte Carlo methods, the Monte Carlo error can be written as a sum of TD errors:

$$\rew{G_t} - \val{V(S_t)} = \del_t + \gam\,\del_{t+1} + \gam^2\,\del_{t+2} + \cdots + \gam^{T-t-1}\,\del_{T-1}. \label{sum}$$

The identity is not exact if $\val{V}$ is updated during the episode, as it is in TD(0), but if the step size is small it may still hold approximately. It is the key to how TD methods relate to Monte Carlo methods ([[td-lambda]]).

### The algorithm {#algorithm}

::: algorithm {#alg-td} Tabular TD(0) for estimating $\val{v_\pi}$
Input: the policy $\pol{\pi}$ to be evaluated
Algorithm parameter: step size $\alp \in (0, 1]$
Initialize $\val{V(s)}$, for all $s \in \mathcal{S}^+$, arbitrarily except that $\val{V(\textit{terminal})} = 0$
Loop for each episode:
  Initialize $S$
  Loop for each step of the episode, until $S$ is terminal:
    $A \leftarrow$ action given by $\pol{\pi}$ for $S$
    Take action $A$, observe $\rew{R}$, $S'$
    $\val{V(S)} \leftarrow \val{V(S)} + \alp\,[\rew{R} + \gam\,\val{V(S')} - \val{V(S)}]$
    $S \leftarrow S'$
:::

Each step costs one update of one state, in constant time: TD(0) is fully incremental and online. Its backup diagram has a single sampled transition, from the state through the action taken to the next state (\ref{fig-backup}).

::: figure {#fig-backup}
{{backup td0}}
The backup diagram of TD(0): one sampled step, from the state through the action taken and the reward received to the next state, whose estimate completes the target.
:::

### Convergence {#convergence}

Is TD(0) sound? For any fixed policy, it has been proved to converge to $\val{v_\pi}$.

::: theorem {#thm-td} Convergence of tabular TD(0)
For a fixed policy in a finite MDP, TD(0) with a table of values converges in the mean to $\val{v_\pi}$ if the step size is constant and sufficiently small, and with probability 1 if every state is visited infinitely often and the step sizes satisfy the stochastic-approximation conditions $\sum_t \alpha_t = \infty$ and $\sum_t \alpha_t^2 < \infty$.
:::

::: proof Proof idea
In expectation, the TD(0) update of a state is the iterative policy-evaluation update of that state ([[policy-evaluation]]): its expected target is the right-hand side of the Bellman equation evaluated at the current estimates. That operator is a contraction, so TD(0) is a noisy, asynchronous version of a contracting iteration, and results on stochastic approximation apply, as for [[q-learning]] (Sutton, 1988; Dayan, 1992; Jaakkola, Jordan & Singh, 1994).
:::

Most convergence proofs apply only to the table-based case; some also apply to linear function approximation ([[semi-gradient-td]]).

### Example: the random walk {#example}

::: example {#ex-walk} Random walk (Sutton & Barto, Example 6.2)
A Markov reward process, an MDP without actions, has five nonterminal states A to E in a row. Every episode starts in the center state C and proceeds left or right by one state on each step, with equal probability. Episodes terminate on the far left or the far right. Terminating on the right gives a reward of $\rew{+1}$; all other rewards are zero. Undiscounted, the value of each state is the probability of terminating on the right from it: $\tfrac16, \tfrac26, \tfrac36, \tfrac46, \tfrac56$ for A to E.
:::

::: figure {#fig-values}
{{random-walk values}}
The values learned by TD(0) with $\alp = 0.1$ after 0, 1, 10 and 100 episodes of the random walk, starting from 0.5 everywhere, and the true values. Computed by the Lab. After Sutton & Barto, Example 6.2.
:::

After the first episode only one estimate has changed (\ref{fig-values}). The first episode of this run ended on the right, from E: on every earlier step the target was $0 + 0.5$, equal to the estimate, so the TD error was zero, and only the last step, with its reward of 1, changed $\val{V(E)}$, from 0.5 to 0.55. After about 100 episodes the estimates are about as close to the true values as they get with a constant step size: they keep fluctuating with the outcomes of the most recent episodes. \ref{fig-error} compares the learning speed of TD(0) and constant-α Monte Carlo at several step sizes.

::: figure {#fig-error}
{{random-walk error}}
The root-mean-square error between the learned values and the true values, averaged over the five states and over 100 runs, for TD(0) and constant-α Monte Carlo at several step sizes, all starting from 0.5. The Lab computes the runs when the figure comes into view. After Sutton & Barto, Example 6.2.
:::

With its best step size, TD(0) is better than Monte Carlo with its best step size throughout. The best Monte Carlo step sizes, around 0.03 to 0.04, reach an error of about 0.065 after 100 episodes; TD(0) with $\alp = 0.05$ reaches about 0.04. Larger step sizes learn faster at first and then level off higher, for both methods, because each estimate keeps chasing the most recent outcomes ([[step-size]]). Why TD learns faster here, and what each method converges to under repeated training on a fixed set of episodes, is the subject of [[mc-vs-td]].

### Advantages {#advantages}

- **No model.** Unlike dynamic programming, TD methods need neither the rewards nor the transition probabilities.
- **Online and incremental.** Unlike Monte Carlo methods, they update after every step, and need no memory of the episode. This matters when episodes are very long, and for continuing tasks, which have no episodes at all.
- **Learning from incomplete episodes.** Some Monte Carlo methods must ignore or discount episodes on which experimental actions are taken; TD methods learn from each transition regardless of what is done afterward.
- **Often faster.** In practice TD methods have usually been found to converge faster than constant-α Monte Carlo methods on stochastic tasks, as on the random walk; there is no general proof that one is always faster ([[mc-vs-td]]).

### Historical remarks {#history}

The idea of learning a prediction from a later prediction goes back to Samuel's checkers player (Samuel, 1959), which adjusted its evaluation of positions toward the evaluation of positions searched later. Sutton (1988) introduced temporal-difference learning in its modern form, TD(λ), proved convergence in the mean for the tabular case and introduced the random-walk example. Dayan (1992) and Jaakkola, Jordan and Singh (1994) proved convergence with probability 1. With a neural network in place of the table, TD learning produced TD-Gammon, a backgammon program that reached the level of the best human players (Tesauro, 1995).

## Card

### Idea

After every step, move the value of the state you left toward the reward you got plus the value of the state you reached. You learn a guess from a better guess, one step at a time, without waiting for the episode to end and without a model.

::: analogy
Revising your estimate of when you will get home, at every junction: you do not wait until you arrive to learn that the traffic was bad. Each junction's estimate is corrected toward the time to the next junction plus that junction's estimate.
:::

### The update {#update}

$$\val{V(S)} \leftarrow \val{V(S)} + \alp\,\big[\,\rew{R} + \gam\,\val{V(S')} - \val{V(S)}\,\big]$$

*New guess = old guess + step size × (target − old guess)*, with the target $\rew{R} + \gam\,\val{V(S')}$. The bracket is the TD error $\del$. At the end of an episode, $\val{V(S')} = 0$.

### One change from Monte Carlo prediction {#change}

| | target | updates |
| --- | --- | --- |
| [[mc-prediction]] | $\rew{G_t}$, the actual return | at the end of the episode |
| TD(0) | $\rew{R_{t+1}} + \gam\,\val{V(S_{t+1})}$, a reward and a guess | after every step |

### Backup diagram {#backup}

{{backup td0}}

From the state, through the one action taken and the reward received, to the one next state: a single sampled step.

### Pseudocode

::: pseudocode
Parameters: the policy $\pol{\pi}$ to evaluate, step size $\alp \in (0, 1]$, discount $\gam$
Set $\val{V(s)}$ for every state (here 0.5); a terminal state stays at 0 {#init}
Repeat for each episode:
  Start: $S$ {#start}
  Repeat until $S$ is terminal:
    Choose $A$ with $\pol{\pi}$ in $S$ {#choose}
    Take $A$, observe $\rew{R}$ and $S'$ {#act}
    $\val{V(S)} \leftarrow \val{V(S)} + \alp\,[\rew{R} + \gam\,\val{V(S')} - \val{V(S)}]$ {#update}
    $S \leftarrow S'$ {#next}
:::

### Perks

- Learns after every step: online, incremental, and fine for tasks that never end.
- Needs no model, like Monte Carlo, and no complete episodes, unlike it.
- Lower-variance targets than Monte Carlo: usually faster on stochastic tasks. [See it](lab:random-walk)

### Flaws

- Biased targets: they contain the current, possibly wrong, estimate of the next state.
- Information travels back one step per update, so long chains learn slowly ([[n-step-td]]).
- Relies on the Markov property of the states more than Monte Carlo does.

### Knobs

| Knob | Too low | Too high |
| --- | --- | --- |
| $\alp$ step size | learns slowly | estimates keep jumping with the latest transitions |
| $\gam$ discount | short-sighted values | slow to settle in long episodes |

### Pitfalls

- Forgetting the terminal state: its value is 0, not a learned estimate.
- Updating with the next state's value *after* changing it: use the value as it was before this step's update.
- Expecting convergence with a constant step size: the estimates keep fluctuating around the true values.

### Check yourself {#check}

::: question
In the random walk, every estimate starts at 0.5, and the first walk ends on the right. Which estimates change, and why?
---
Only the last state's. On every earlier step the target $0 + 0.5$ equals the estimate, so the TD error is zero. Only the final step, with its reward of 1 and a terminal value of 0, gives a nonzero error.
:::

::: question
What does TD(0) sample, and what does it estimate, compared with dynamic programming?
---
It samples one transition instead of averaging over all of them with the model, and, like dynamic programming, it uses the current estimate of the next state instead of the true value.
:::

::: question
Why can TD(0) learn in a task that never ends, while Monte Carlo cannot?
---
Its target needs only the next reward and the next state's estimate. Monte Carlo needs the return, which in a task that never ends is never complete.
:::
