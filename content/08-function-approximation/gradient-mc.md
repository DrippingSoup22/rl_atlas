+++
summary = "Monte Carlo prediction with weights instead of a table: play the episode, then nudge the weights so that each visited state's estimate moves toward the return that followed it. The return is an unbiased target, so this is true stochastic gradient descent on the value error, with its guarantee: with small enough steps, it settles at the best fit the features allow."
change = "Replace the table by an estimate computed from weights, v̂(s, w), and move the weights along the gradient: w ← w + α [G − v̂(S, w)] ∇v̂(S, w). With one feature per state, it is Monte Carlo prediction again."
prereqs = ["mc-prediction", "value-error", "features"]
lab = "walk-aggregation"
sources = [
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., §9.3 and Example 9.1", url = "http://incompleteideas.net/book/the-book-2nd.html" },
  { text = "Widrow, Gupta & Maitra (1973), Punish/reward: learning with a critic in adaptive threshold systems, IEEE Transactions on Systems, Man, and Cybernetics 3", url = "https://doi.org/10.1109/TSMC.1973.4309272" },
  { text = "Bradtke & Barto (1996), Linear least-squares algorithms for temporal difference learning, Machine Learning 22", url = "https://doi.org/10.1007/BF00114723" },
  { text = "Robbins & Monro (1951), A stochastic approximation method, Annals of Mathematical Statistics 22", url = "https://doi.org/10.1214/aoms/1177729586" },
]

[story]
scene = "line"
env = "walk-1000"
seed = 3
average = 20
formula = '\step{1}{\rew{G_t} = \rew{R_{t+1}} + \gam\,\rew{G_{t+1}}} \step{2}{\qquad \mathbf w \leftarrow \mathbf w + \alp\,\big[\rew{G_t} - \val{\hat v(S_t, \mathbf w)}\big]\,\nabla \val{\hat v(S_t, \mathbf w)}}'

[story.runs]
big = { algorithm = "gradient-mc", features = "groups", cells = 10, alpha = 0.01, gamma = 1.0, units = 2000, measures = ["ve"], name = "α = 0.01" }
mid = { algorithm = "gradient-mc", features = "groups", cells = 10, alpha = 0.0005, gamma = 1.0, units = 2000, measures = ["ve"], name = "α = 0.0005" }
small = { algorithm = "gradient-mc", features = "groups", cells = 10, alpha = 0.0001, gamma = 1.0, units = 2000, measures = ["ve"], name = "α = 0.0001" }
+++

## Story

::: step {run = "big", at = 0}
**The 1000-state walk with ten weights**, one per group of 100 states ([[features]]). Every weight starts at 0, so every estimate is 0. The learner is [[mc-prediction|Monte Carlo]], almost unchanged: wait for the walk to end, then learn from the return.
:::

::: step {run = "big", at = 1, formula = 2}
**The first walk** takes 40 jumps and leaves on the right: every visited state's return is $\rew{+1}$. Then one update per visited state, each nudging its estimate toward that return. With weights instead of a table, nudging an estimate means moving the weight that computes it, the weight of its group (the strip under the plot). The groups the walk passed through rise; the four on the left, never visited, stay at 0. With $\alpha = 0.01$ each nudge is small: one walk is one noisy example, not the answer.
:::

::: step {run = "big", at = 2}
The second walk leaves on the left, and drags everything it visited down again. Every walk pulls the staircase toward its own return, $\rew{+1}$ or $\rew{-1}$: the estimates are averages of these pulls, and with $\alpha = 0.01$ the average is short.
:::

::: step {run = "big", at = 2000, ghosts = [100]}
Two thousand walks later it is still jumping. The faint line is the staircase after 100 walks, the solid one after 2000, just after a few walks that left on the left: the middle steps, whose true values are near 0, sit between $-0.5$ and $-0.9$. A large step size remembers only the last few walks.
:::

::: step {run = "mid", at = 2000, ghosts = [100, 500]}
With $\alpha = 0.0005$ every walk counts for less, and the staircase settles: after 100 walks (faint) it has barely started, after 500 it has the shape, after 2000 each step sits near the average true value of its group, weighted by how often the walk visits each state. That is the best ten numbers can do ([[value-error]]).
:::

::: step {run = "mid", at = 2000, curves = ["small", "mid", "big"], metric = "ve"}
Averaged over 20 runs: the large step learns fastest at first and then stays noisy; the small one is smooth but slow; the middle one wins by 2000 walks. Shrinking the step over time gets both. [Try other step sizes in the Lab](lab:walk-aggregation).
:::

## Textbook

### From Monte Carlo to gradient Monte Carlo {#idea}

[[mc-prediction|Monte Carlo prediction]] moves each visited state's estimate toward the return that followed it. With a table, “moving the estimate” means changing one entry. With an estimate computed from weights, it means changing the weights in the direction that brings that estimate closer to the return, which is the gradient of its squared error ([[value-error]]):

$$\mathbf w \leftarrow \mathbf w + \alp\,\big[\,\rew{G_t} - \val{\hat v(S_t, \mathbf w)}\,\big]\,\nabla \val{\hat v(S_t, \mathbf w)}. \label{eq-update}$$

For linear features the gradient is the feature vector $\mathbf x(S_t)$, so each weight moves in proportion to its feature, and every state that shares features with $S_t$ moves along. With one feature per state the update reduces to tabular constant-$\alpha$ Monte Carlo.

### The algorithm {#algorithm}

::: algorithm {#alg-gmc} Gradient Monte Carlo for estimating $\val{v_\pi}$
Input: the policy $\pol{\pi}$ to evaluate, a differentiable estimate $\val{\hat v(s, \mathbf w)}$, a step size $\alp > 0$
Set the weights $\mathbf w$ (for example, all 0)
Repeat for each episode:
  Generate an episode $S_0, A_0, \rew{R_1}, \dots, S_{T-1}, A_{T-1}, \rew{R_T}$ following $\pol{\pi}$
  For $t = 0, 1, \dots, T - 1$: $\mathbf w \leftarrow \mathbf w + \alp\,[\rew{G_t} - \val{\hat v(S_t, \mathbf w)}]\,\nabla \val{\hat v(S_t, \mathbf w)}$
:::

The returns are computed backward from the end of the episode, $\rew{G_t} = \rew{R_{t+1}} + \gam\,\rew{G_{t+1}}$, before the weights are touched; the updates then go forward through the episode. Unlike the tabular version, the order now matters a little, since each update changes the estimates the later ones start from. A state visited several times is updated once per visit, the every-visit form: with function approximation it is the natural one, and a first-visit version gains nothing, since the visits of other states move the estimate anyway.

### Why it converges {#convergence}

The return is an **unbiased** target: its expected value, given the state, is the true value $\val{v_\pi(S_t)}$. Each update is therefore a noisy but unbiased version of a gradient step on the value error, and gradient Monte Carlo is a true stochastic gradient method.

::: theorem {#thm-gmc} Gradient Monte Carlo finds a local minimum
If the step sizes shrink so that $\sum_t \alpha_t = \infty$ and $\sum_t \alpha_t^2 < \infty$, gradient Monte Carlo converges with probability 1 to a local minimum of $\overline{\text{VE}}$. With linear features the minimum is global: the weights settle at the best fit the features allow, in the sense of the value error.
:::

The states are drawn by the agent's own experience, so they come up as often as $\mu$ says, and the updates descend exactly the error weighted by $\mu$. On the 1000-state walk with ten groups, this means each group's weight converges to the $\mu$-weighted average of its states' true values (Figure \ref{fig-alpha} shows how fast).

### Choosing the step size {#stepsize}

With a constant step size the weights never settle completely: each episode pulls them toward its own returns, and they jitter around the optimum by an amount that grows with $\alpha$. A large $\alpha$ learns fast and jitters a lot, a small one is accurate and slow, as the story shows. For linear features a useful rule of thumb relates $\alpha$ to how many experiences the learner should average over. If the feature vectors have roughly the same squared length, an update moves the visited state's estimate by $\alpha\,\lVert \mathbf x \rVert^2$ of its error, and

$$\alpha = \frac{1}{\tau\, \mathbb E\big[\mathbf x^\top \mathbf x\big]} \label{eq-tau}$$

makes the estimate of a state seen over and over approach its target in about $\tau$ updates. With one active binary feature ($\lVert\mathbf x\rVert^2 = 1$) this is the tabular $\alpha = 1/\tau$; with 50 tilings ($\lVert\mathbf x\rVert^2 = 50$) the same averaging needs an $\alpha$ fifty times smaller. On the walk, each group collects hundreds of updates per hundred walks, which is why good step sizes there look so small.

::: figure {#fig-alpha}
{{walk-alpha}}
Gradient Monte Carlo on the 1000-state walk with 10 groups, for four step sizes. Large steps learn fast and stay noisy; small ones are accurate eventually. Computed live by the Lab.
:::

### Relatives {#relatives}

- **Least-squares Monte Carlo.** For linear features, the weights that best fit a batch of returns can be computed directly, by linear regression, instead of by many small steps. This uses the data more efficiently and has no step size, at a cost per step that grows with the square of the number of features. Least-squares TD (Bradtke and Barto, 1996) is the bootstrapping counterpart.
- **Gradient Monte Carlo control.** Learn action values $\hat q(s, a, \mathbf w)$ the same way and act ε-greedily on them; with function approximation Monte Carlo control is rarely used, since the episodes it waits for can be very long under a poor early policy. [[semi-gradient-sarsa]] is the usual choice.
- **Semi-gradient TD.** Replace the return by a bootstrapped target and learn during the episode: [[semi-gradient-td]].

### Historical remarks {#history}

Using the least-mean-square rule to learn from the outcome of a whole episode is an old idea: Widrow, Gupta and Maitra (1973) trained an adaptive linear element to play blackjack this way, with an end-of-game reward as the only teacher. The convergence of stochastic gradient methods with unbiased targets follows from stochastic approximation theory (Robbins and Monro, 1951). Sutton and Barto's 1000-state walk (Example 9.1), which the Lab and the story use, was designed to show state aggregation at work.

## Card

### Idea

Monte Carlo with weights: play the episode, then move the weights so that each visited state's estimate gets closer to the return that followed it. The return is unbiased, so this is true gradient descent on the value error.

::: analogy
Adjusting a recipe after each dinner by how the guests liked it: a tweak to the salt changes every dish that uses salt, a little each time.
:::

### The update {#update}

$$\mathbf w \leftarrow \mathbf w + \alp\,\big[\,\rew{G_t} - \val{\hat v(S_t, \mathbf w)}\,\big]\,\nabla \val{\hat v(S_t, \mathbf w)}$$

With linear features, $\nabla \val{\hat v(S_t, \mathbf w)} = \mathbf x(S_t)$: each weight moves in proportion to its feature.

### One change from Monte Carlo prediction {#change}

| | estimate | an update moves |
| --- | --- | --- |
| [[mc-prediction]] | a table entry $\val{V(s)}$ | one state |
| Gradient MC | $\val{\hat v(s, \mathbf w)}$ | the weights, and every state that shares them |

### Backup diagram {#backup}

{{backup mc}}

The same backup as Monte Carlo: the whole episode to its end. What changes is what the update writes to.

### Pseudocode

::: pseudocode
Parameters: step size $\alp$; features $\mathbf x(s)$, so $\val{\hat v(s, \mathbf w)} = \mathbf w^\top \mathbf x(s)$
Set $\mathbf w$ (here 0)
Repeat for each episode:
  Start: $S_0$ {#start}
  Repeat until the episode ends: choose $A_t$ from $\pol{\pi}$ {#choose}
    take it, observe $\rew{R_{t+1}}$ and $S_{t+1}$ {#act}
  For $t = 0, 1, \dots, T-1$: $\mathbf w \leftarrow \mathbf w + \alp\,[\rew{G_t} - \val{\hat v(S_t, \mathbf w)}]\,\mathbf x(S_t)$ {#update}
:::

### Perks

- True gradient descent: converges to the best fit the features allow (linear case).
- Simple, and unbiased: no estimate is trusted before it has been learned.
- Works with any differentiable estimate, linear or not. [See it](lab:walk-aggregation)

### Flaws

- Waits for the end of each episode; useless for tasks that never end.
- High variance: every return carries the noise of a whole episode, so small steps and many episodes are needed.
- Learns nothing during a long first episode.

### Knobs

| Knob | Too low | Too high |
| --- | --- | --- |
| $\alp$ step size | learns very slowly | estimates jump with each episode's return |
| features | too coarse: a poor best fit | too fine: little generalization, slow learning |

### Pitfalls

- Keeping a tabular step size with many active features: the effective step is $\alp\,\lVert\mathbf x\rVert^2$.
- Expecting exact values: it converges to the best fit, which can still be far off where features are coarse.
- Judging the fit everywhere: rarely visited states count little in the value error, and may be poor.

### Check yourself {#check}

::: question
What makes gradient Monte Carlo a true gradient method, unlike semi-gradient TD?
---
Its target, the return, has the true value as its expectation and does not depend on the weights, so each update is an unbiased sample of a gradient step on the value error.
:::

::: question
With state aggregation, what value does each group's weight converge to?
---
The average of the true values of its states, each weighted by how often the agent visits it ($\mu$): the best single number for the group in the sense of the value error.
:::

::: question
Why do good step sizes on the 1000-state walk look so small, like 0.0005?
---
Each walk updates its groups many times (once per visit, dozens of visits per walk), so a group collects hundreds of updates every hundred walks. To average over many walks, each update must move the weight very little.
:::
