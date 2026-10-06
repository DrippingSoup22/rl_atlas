+++
summary = "Two learners in one: an actor, the policy, and a critic, a learned value function. After every step the critic's TD error says whether the move went better or worse than expected, and the actor makes it more or less likely on the spot. Bootstrapping from the critic costs a little bias and saves a lot of variance: no waiting for the end of the episode, far less noise."
change = "Replace the complete return by the one-step return, R + γ v̂(S′, w): the weight of ∇ ln π becomes the critic's TD error δ, so the actor learns after every step instead of after every episode."
prereqs = ["baseline", "td0", "semi-gradient-td", "td-error"]
lab = "cliff-actor-critic"
sources = [
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., §13.5 and §13.6", url = "http://incompleteideas.net/book/the-book-2nd.html" },
  { text = "O'Doherty, Dayan, Schultz, Deichmann, Friston & Dolan (2004), Dissociable roles of ventral and dorsal striatum in instrumental conditioning, Science 304", url = "https://doi.org/10.1126/science.1094285" },
  { text = "Barto, Sutton & Anderson (1983), Neuronlike adaptive elements that can solve difficult learning control problems, IEEE Transactions on Systems, Man, and Cybernetics 13" },
  { text = "Sutton (1984), Temporal credit assignment in reinforcement learning, PhD thesis, University of Massachusetts Amherst" },
  { text = "Konda & Tsitsiklis (2000), Actor-critic algorithms, Advances in Neural Information Processing Systems 12" },
  { text = "Bhatnagar, Sutton, Ghavamzadeh & Lee (2009), Natural actor–critic algorithms, Automatica 45" },
]

[story]
scene = "grid"
env = "cliff"
seed = 2
average = 30
formula = '\step{1}{\del = \rew{R} + \gam\,\val{\hat v(S^\prime, \mathbf w)} - \val{\hat v(S, \mathbf w)} \qquad} \step{2}{\boldsymbol\theta \leftarrow \boldsymbol\theta + \alp^{\boldsymbol\theta}\,\del\,\nabla \ln \pol{\pi(A \mid S, \boldsymbol\theta)} \qquad} \step{3}{\mathbf w \leftarrow \mathbf w + \alp^{\mathbf w}\,\del\,\nabla \val{\hat v(S, \mathbf w)}}'

[story.numbers]
bump = '\del = \rew{-1} + \val{0} - \val{0} = \err{-1} \qquad \pol{\pi(\text{down} \mid \text{start})}: 25.0\% \to 23.2\%'
fall = '\del = \rew{-100} + \val{-0.1} - \val{-0.1} = \err{-100} \qquad \pol{\pi(\text{right} \mid \text{start})}: 25.6\% \to 0.0\%'
up = '\del = \rew{-1} + \val{0} - \val{(-10.1)} = \err{+9.1} \qquad \pol{\pi(\text{up} \mid \text{start})}: 36.9\% \to 58.2\%'

[story.runs]
learn = { algorithm = "actor-critic", alpha = 0.1, alphaW = 0.1, lambda = 0.0, gamma = 1.0, maxSteps = 1000, units = 500, measures = ["policy-value"], name = "actor–critic" }
ac = { algorithm = "actor-critic", alpha = 0.1, alphaW = 0.1, lambda = 0.0, gamma = 1.0, maxSteps = 1000, units = 300, name = "actor–critic, α = 0.1" }
mc = { algorithm = "baseline", alpha = 0.0001, alphaW = 0.1, gamma = 1.0, maxSteps = 1000, units = 300, name = "REINFORCE with baseline, α = 0.0001" }
+++

## Story

::: step {run = "learn", at = 0}
**The cliff** ([[q-learning]]): start at the bottom left, reach the gem at the bottom right, $\rew{-1}$ a step and $\rew{-100}$ for falling, which sends the agent back to the start. Two learners share the work. The **actor** is a softmax policy over the four moves of each tile: the arrows, all equally likely at first. The **critic** is an estimate of the value of each tile, $\val{\hat v(s, \mathbf w)}$: the colors, all 0 at first.
:::

::: step {run = "learn", at = 0, play = 1, updates = 1, fine = true, pace = 500, formula = 1, numbers = "bump"}
The first move: down, into the wall. It costs $\rew{-1}$ and leaves the agent at the start, which the critic values at 0. The critic's **TD error** $\del$ says how much better or worse the move went than it expected: $-1$, slightly worse.
:::

::: step {run = "learn", at = 0, play = 1, updates = 1, fine = true, pace = 500, formula = 3, numbers = "bump"}
Both learners use that one number, at once. The actor makes the move a little less likely: down from the start, from 25.0% to 23.2%. The critic lowers its estimate of the start to $\val{-0.1}$. No waiting for the end of the episode.
:::

::: step {run = "learn", at = 0, play = 1, updates = 2, fine = true, pace = 500, formula = 3, numbers = "fall"}
The second move: right, into the cliff. $\rew{-100}$, and back to the start: $\del = -100$. Right from the start drops from 25.6% to practically 0 in one update, and the critic now expects $\val{-10.1}$ from the start.
:::

::: step {run = "learn", at = 0, play = 1, updates = 3, fine = true, pace = 500, formula = 3, numbers = "up"}
The third move: up, for $\rew{-1}$, to a tile the critic still values at 0. From the start the critic now expected $\val{-10.1}$, so this is much better than expected: $\del = +9.1$, and up jumps from 36.9% to 58.2%. Nothing good happened, only something less bad than the critic feared. The critic, not the reward alone, decides which way each move is pushed.
:::

::: step {run = "learn", at = 1}
The rest of the first episode wanders for 976 steps and falls 11 times. Each fall teaches the actor to avoid the move that caused it, and the critic to fear the tiles near the edge. After this one episode, from the start the actor goes up 86% of the time and right practically never.
:::

::: step {run = "learn", at = 50, values = true}
After 50 episodes. Values flow back from the gem through the critic, and the arrows follow them: the most likely path climbs two rows above the cliff, runs along, and comes down to the gem, 15 steps. The policy is still quite random: it follows that exact path only 3% of the time, and the value of the policy, worked out exactly, is $\rew{-24.0}$ per episode.
:::

::: step {run = "learn", at = 500, values = true}
After 500 episodes: worth $\rew{-15.2}$ per episode. The most likely path, now followed exactly in more than half of the episodes, runs two rows above the cliff: 15 steps, two more than the shortest path along the edge. Like [[sarsa|SARSA]], the actor–critic learns about the policy it actually follows, randomness included, so it keeps its distance from the edge. (The episode drawn took another 15-step route.)
:::

::: step {run = "learn", at = 500, curves = ["ac", "mc"], metric = "steps"}
Averaged over 30 runs: the actor–critic needs 16 steps per episode after 200 episodes. [[baseline|REINFORCE with baseline]], which waits for whole returns, drowns in them: the first episodes cost thousands, and with a step size of 0.001, 29 runs of 30 get stuck. Even with 0.0001 it still takes 565 steps per episode after 300 episodes, and half its runs still hit the 1000-step limit. [Watch the critic and the actor in the Lab](lab:cliff-actor-critic).
:::

## Textbook

### From baseline to critic {#bootstrap}

[[baseline|REINFORCE with baseline]] judges each action by its complete return minus a learned state value, $\rew{G_t} - \val{\hat v(S_t, \mathbf w)}$. Its value function only compares; it never stands in for anything. **Actor–critic** methods use it to bootstrap: the return is replaced by the one-step return, the first reward plus the discounted estimate of the next state, exactly as TD(0) replaces the Monte Carlo target ([[td0]]):

$$\boldsymbol\theta \leftarrow \boldsymbol\theta + \alp^{\boldsymbol\theta}\,\gam^t\,\big(\rew{R_{t+1}} + \gam\,\val{\hat v(S_{t+1}, \mathbf w)} - \val{\hat v(S_t, \mathbf w)}\big)\,\nabla \ln \pol{\pi(A_t \mid S_t, \boldsymbol\theta)} = \boldsymbol\theta + \alp^{\boldsymbol\theta}\,\gam^t\,\del_t\,\nabla \ln \pol{\pi(A_t \mid S_t, \boldsymbol\theta)}. \label{eq-actor}$$

The weight of the score is the [[td-error|TD error]] $\del_t$, and the same number updates the value function by [[semi-gradient-td|semi-gradient TD(0)]]:

$$\mathbf w \leftarrow \mathbf w + \alp^{\mathbf w}\,\del_t\,\nabla \val{\hat v(S_t, \mathbf w)}. \label{eq-critic}$$

The policy is the **actor**, which chooses; the value function is the **critic**, which judges each choice by the surprise it produced. The method is fully online and incremental: states, actions and rewards are used as they arrive and then forgotten, and continuing tasks pose no problem.

### What the TD error measures {#advantage}

Given the state and the action, the expected TD error under the true values is the advantage of the action ([[pg-theorem]]):

$$\mathbb E\big[\rew{R_{t+1}} + \gam\,\val{v_\pi(S_{t+1})} - \val{v_\pi(S_t)} \,\big|\, S_t = s, A_t = a\big] = \val{q_\pi(s, a)} - \val{v_\pi(s)} = \err{a_\pi(s, a)}. \label{eq-adv}$$

So with an accurate critic, $\del_t$ is an unbiased, one-sample estimate of how much better $A_t$ was than the policy's average action in $S_t$: positive for better than usual, negative for worse. It judges the action, not the outcome: on the cliff, a step that only costs $\rew{-1}$ is a good step where the critic feared a fall (the Story's third move).

### Bias and variance {#bias}

A return $\rew{G_t}$ sums the randomness of every later action and transition; the one-step return contains one reward and one transition, plus an estimate. The variance of the update drops sharply, and with it the noise that forced REINFORCE to take tiny steps. The price is **bias**: the critic is only an estimate, and where it is wrong, actions are misjudged; the expected update is no longer exactly the gradient of $J$. The same trade-off runs through all of temporal-difference learning ([[n-step-td]]): $n$-step returns, or $\lambda$-returns with eligibility traces ([[td-lambda]]), interpolate between the one-step actor–critic ($n = 1$, $\lambda = 0$) and REINFORCE with baseline ($n = \infty$, $\lambda = 1$). The same dial reappears, in batch form, as [[gae|generalized advantage estimation]].

### The algorithms {#algorithm}

::: algorithm {#alg-ac} One-step actor–critic (episodic)
Input: a differentiable policy $\pol{\pi(a \mid s, \boldsymbol\theta)}$, a differentiable state-value estimate $\val{\hat v(s, \mathbf w)}$, step sizes $\alp^{\boldsymbol\theta} > 0$ and $\alp^{\mathbf w} > 0$, a discount $\gam$
Set the weights $\boldsymbol\theta$ and $\mathbf w$ (for example, to 0)
Repeat for each episode:
  Start: $S$; $I \leftarrow 1$
  Repeat until $S$ is terminal:
    Choose $A \sim \pol{\pi(\cdot \mid S, \boldsymbol\theta)}$; take it, observe $S'$ and $\rew{R}$
    $\del \leftarrow \rew{R} + \gam\,\val{\hat v(S', \mathbf w)} - \val{\hat v(S, \mathbf w)}$ (with $\val{\hat v(S', \mathbf w)} = 0$ if $S'$ is terminal)
    $\mathbf w \leftarrow \mathbf w + \alp^{\mathbf w}\,\del\,\nabla \val{\hat v(S, \mathbf w)}$
    $\boldsymbol\theta \leftarrow \boldsymbol\theta + \alp^{\boldsymbol\theta}\,I\,\del\,\nabla \ln \pol{\pi(A \mid S, \boldsymbol\theta)}$
    $I \leftarrow \gam I$, $S \leftarrow S'$
:::

The factor $I = \gam^t$ is the $\gam^t$ of the policy gradient theorem; most implementations leave it out ([[pg-theorem]]). With eligibility traces, each learner keeps a trace of its gradients, and every TD error updates all recently visited states and actions at once:

::: algorithm {#alg-ac-traces} Actor–critic with eligibility traces (episodic)
Input: as above, plus trace-decay rates $\lam^{\boldsymbol\theta}, \lam^{\mathbf w} \in [0, 1]$
Repeat for each episode:
  Start: $S$; $\mathbf z^{\boldsymbol\theta} \leftarrow \mathbf 0$, $\mathbf z^{\mathbf w} \leftarrow \mathbf 0$, $I \leftarrow 1$
  Repeat until $S$ is terminal:
    Choose $A \sim \pol{\pi(\cdot \mid S, \boldsymbol\theta)}$; take it, observe $S'$ and $\rew{R}$
    $\del \leftarrow \rew{R} + \gam\,\val{\hat v(S', \mathbf w)} - \val{\hat v(S, \mathbf w)}$
    $\mathbf z^{\mathbf w} \leftarrow \gam\lam^{\mathbf w}\,\mathbf z^{\mathbf w} + \nabla \val{\hat v(S, \mathbf w)}$, $\quad \mathbf z^{\boldsymbol\theta} \leftarrow \gam\lam^{\boldsymbol\theta}\,\mathbf z^{\boldsymbol\theta} + I\,\nabla \ln \pol{\pi(A \mid S, \boldsymbol\theta)}$
    $\mathbf w \leftarrow \mathbf w + \alp^{\mathbf w}\,\del\,\mathbf z^{\mathbf w}$, $\quad \boldsymbol\theta \leftarrow \boldsymbol\theta + \alp^{\boldsymbol\theta}\,\del\,\mathbf z^{\boldsymbol\theta}$
    $I \leftarrow \gam I$, $S \leftarrow S'$
:::

The Lab uses one $\lam$ for both learners.

### Two learners, two speeds {#speeds}

The actor learns from the critic's judgments, so the critic must keep up with the actor: each change of the policy changes the values the critic is chasing. The convergence proofs for actor–critic methods (Konda and Tsitsiklis, 2000; Bhatnagar, Sutton, Ghavamzadeh and Lee, 2009) use two time scales: the critic's step sizes shrink more slowly than the actor's, so that from the actor's point of view the critic has always converged. In practice the critic's step size is tuned like that of any TD method, and the actor's is chosen smaller or comparable.

::: figure {#fig-speed}
{{critic-speed}}
The critic's step size on the cliff, with the actor's fixed at $\alp^{\boldsymbol\theta} = 0.1$. A slow critic ($\alp^{\mathbf w} = 0.02$) gives stale judgments: after 300 episodes, more than half of the runs still take over 20 steps per episode. A fast one ($\alp^{\mathbf w} = 0.5$) learns fastest at first, but its noisy judgments settle the actor on worse paths: $\rew{-17.5}$ per episode at the end, against $\rew{-15.7}$ for $\alp^{\mathbf w} = 0.1$. Computed live by the Lab.
:::

### Continuing tasks {#continuing}

Without episodes, performance is the average reward per step ([[pg-theorem]]), and the TD error uses differential values: the reward minus a running estimate $\bar R$ of the average reward,

$$\del = \rew{R} - \bar R + \val{\hat v(S', \mathbf w)} - \val{\hat v(S, \mathbf w)}, \qquad \bar R \leftarrow \bar R + \alp^{\bar R}\,\del, \label{eq-average}$$

with the critic and actor updated by \ref{eq-critic} and \ref{eq-actor} without discounting and without $I$. The actor–critic is the natural policy-gradient method for continuing tasks, where REINFORCE, which needs the end of an episode, does not apply.

### Example: the cliff {#example}

On the cliff, with tabular features, $\alp^{\boldsymbol\theta} = \alp^{\mathbf w} = 0.1$ and $\gam = 1$, the actor–critic needs 16 steps per episode on average after 200 episodes. Its first episodes are long, but each fall is punished at once, at the move that caused it, and the critic soon makes the edge of the cliff look dangerous. It ends on a path two rows above the cliff, 15 steps long, for the same reason SARSA does: it learns the value of the random policy it follows. REINFORCE with baseline on the same task learns from episodes whose returns run to thousands at first; its updates are so large that with a step size of 0.001 the policy saturates in 29 runs of 30, and with 0.0001 half the runs still hit the 1000-step limit after 300 episodes.

### Historical remarks {#history}

The actor–critic architecture predates the policy gradient theorem by almost two decades. Barto, Sutton and Anderson (1983) balanced a pole with an associative search element (the actor) trained by an adaptive critic element, which learned to predict reinforcement and turned it into an internal signal, in effect the TD error. Sutton (1984) studied such architectures and their credit assignment, and TD learning itself grew out of the critic. Konda and Tsitsiklis (2000) proved convergence of actor–critic methods with linear critics on two time scales, and Bhatnagar, Sutton, Ghavamzadeh and Lee (2009) did so for natural-gradient versions. Asynchronous and batched actor–critics with neural networks ([[a2c]]) made the architecture central to deep reinforcement learning. The split has a counterpart in the brain: in imaging studies, the ventral striatum tracks reward-prediction errors whether or not a choice is involved, like a critic, while the dorsal striatum is engaged only when actions must be chosen, like an actor (O'Doherty et al., 2004).

## Card

### Idea

An actor (the policy) and a critic (a state-value estimate) learn together. After each step the critic's TD error says whether the move went better or worse than expected; the actor makes it more or less likely at once, and the critic updates its estimate with the same number.

::: analogy
A trainee and a coach at the trainee's side. After each move the coach says “better than I expected” or “worse”, and the trainee adjusts on the spot. The coach's expectations improve as they watch.
:::

### The update {#update}

$$\del = \rew{R} + \gam\,\val{\hat v(S', \mathbf w)} - \val{\hat v(S, \mathbf w)}$$

$$\boldsymbol\theta \leftarrow \boldsymbol\theta + \alp^{\boldsymbol\theta}\,I\,\del\,\nabla \ln \pol{\pi(A \mid S, \boldsymbol\theta)}, \qquad \mathbf w \leftarrow \mathbf w + \alp^{\mathbf w}\,\del\,\nabla \val{\hat v(S, \mathbf w)}$$

### One change from REINFORCE with baseline {#change}

| | weight of $\nabla \ln \pol{\pi(A_t \mid S_t)}$ | learns |
| --- | --- | --- |
| [[baseline]] | $\rew{G_t} - \val{\hat v(S_t, \mathbf w)}$: complete return | after each episode |
| Actor–critic | $\rew{R_{t+1}} + \gam\,\val{\hat v(S_{t+1}, \mathbf w)} - \val{\hat v(S_t, \mathbf w)}$: bootstrapped | after each step |

### Backup diagram {#backup}

{{backup actor-critic}}

One step and the critic's estimate of the next state: the TD error judges the action just taken.

### Pseudocode

::: pseudocode
Parameters: step sizes $\alp^{\boldsymbol\theta}$ (actor) and $\alp^{\mathbf w}$ (critic), discount $\gam$; a policy $\pol{\pi(a \mid s, \boldsymbol\theta)}$ and a value estimate $\val{\hat v(s, \mathbf w)}$
Set $\boldsymbol\theta$ (here, all moves equally likely) and $\mathbf w$ (here 0)
Repeat for each episode:
  Start: $S$; $I \leftarrow 1$ {#start}
  Repeat until $S$ is terminal:
    Choose $A \sim \pol{\pi(\cdot \mid S, \boldsymbol\theta)}$ {#choose}
    Take $A$, observe $\rew{R}$ and $S'$ {#act}
    $\del \leftarrow \rew{R} + \gam\,\val{\hat v(S', \mathbf w)} - \val{\hat v(S, \mathbf w)}$ (0 for a terminal $S'$) {#error}
    $\mathbf w \leftarrow \mathbf w + \alp^{\mathbf w}\,\del\,\nabla \val{\hat v(S, \mathbf w)}$, $\ \boldsymbol\theta \leftarrow \boldsymbol\theta + \alp^{\boldsymbol\theta}\,I\,\del\,\nabla \ln \pol{\pi(A \mid S, \boldsymbol\theta)}$ {#update}
    $I \leftarrow \gam I$, $S \leftarrow S'$ {#next}
:::

### Perks

- Learns after every step: online, and fine for continuing tasks.
- Far less variance than REINFORCE: larger steps, faster learning. [See it on the cliff](lab:cliff-actor-critic)
- A stochastic policy, continuous actions, and a value function, all at once.

### Flaws

- Biased: a wrong critic misjudges the actor's moves.
- Two learners and two step sizes that must be balanced.
- On-policy: experience is used once.

### Knobs

| Knob | Too low | Too high |
| --- | --- | --- |
| $\alp^{\boldsymbol\theta}$ actor step size | slow policy | the policy collapses on the critic's early mistakes |
| $\alp^{\mathbf w}$ critic step size | stale judgments, slow learning | noisy judgments mislead the actor |
| $\lam$ trace decay | one-step bias | Monte Carlo variance |
| $\gam$ discount | short-sighted | slow, noisy credit |

### Pitfalls

- A critic slower than the actor: the actor optimizes against out-of-date values.
- Reading $\del$ as a reward: it is a surprise, positive whenever things went better than expected, even when they went badly.
- Forgetting the 0 for terminal states in the target: the critic then bootstraps past the end.

### Check yourself {#check}

::: question
On the cliff, a move costs $-1$ and the actor–critic makes it more likely. How?
---
The critic expected worse from the state it left, for example $-10$ after a fall there, so $\del = -1 + \val{\hat v(S')} - \val{\hat v(S)}$ was positive: better than expected.
:::

::: question
Why is the actor–critic biased while REINFORCE with baseline is not?
---
Its target bootstraps from the critic's estimate of the next state. Where that estimate is wrong, the expected update is no longer the true gradient. REINFORCE with baseline only uses the estimate to compare, which adds no bias.
:::

::: question
What does the TD error estimate, when the critic is exact?
---
The advantage of the action taken: its expectation given $S_t = s$ and $A_t = a$ is $\val{q_\pi(s, a)} - \val{v_\pi(s)}$.
:::
