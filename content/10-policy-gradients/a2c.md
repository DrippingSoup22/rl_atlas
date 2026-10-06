+++
summary = "Actor–critic with many workers. Several copies of the agent explore at once with the same policy; every few steps their experience is pooled into one update of a shared actor and critic, with n-step returns. The parallel workers decorrelate the data, which made deep policy-gradient learning stable without a replay memory. A3C lets the workers update asynchronously; A2C waits for all of them."
change = "Run N workers in parallel on copies of the environment and, every n steps, update a shared actor and critic once from all of their n-step advantages, instead of updating after every step of a single agent."
prereqs = ["actor-critic", "n-step-td"]
lab = "a2c-maze"
sources = [
  { text = "Mnih, Badia, Mirza, Graves, Lillicrap, Harley, Silver & Kavukcuoglu (2016), Asynchronous methods for deep reinforcement learning, ICML", url = "https://arxiv.org/abs/1602.01783" },
  { text = "Wu, Mansimov, Liao, Grosse & Ba (2017), Scalable trust-region method for deep reinforcement learning using Kronecker-factored approximation (ACKTR, with the synchronous A2C baseline), NeurIPS", url = "https://arxiv.org/abs/1708.05144" },
  { text = "OpenAI (2017), OpenAI Baselines: ACKTR & A2C, blog post" },
  { text = "Recht, Ré, Wright & Niu (2011), Hogwild!: a lock-free approach to parallelizing stochastic gradient descent, NeurIPS", url = "https://arxiv.org/abs/1106.5730" },
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., §13.5", url = "http://incompleteideas.net/book/the-book-2nd.html" },
]

[story]
scene = "grid"
env = "dyna-maze"
seed = 2
average = 12
formula = '\step{1}{\err{\hat A_t} = \rew{R_{t+1}} + \gam\,\rew{R_{t+2}} + \dots + \gam^{k-1}\rew{R_{t+k}} + \gam^k\,\val{\hat v(S_{t+k})} - \val{\hat v(S_t)} \qquad} \step{2}{\boldsymbol\theta \leftarrow \boldsymbol\theta + \frac{\alp}{N} \sum_{\text{workers, steps}} \err{\hat A_t}\,\nabla \ln \pol{\pi(A_t \mid S_t, \boldsymbol\theta)}}'

[story.runs]
learn = { algorithm = "a2c", alpha = 2.0, alphaW = 0.3, workers = 4, n = 5, beta = 0.0, gamma = 0.95, maxSteps = 1000, units = 60, name = "A2C, n = 5" }
n1 = { algorithm = "a2c", alpha = 2.0, alphaW = 0.3, workers = 4, n = 1, beta = 0.0, gamma = 0.95, maxSteps = 1000, units = 60, name = "every step (n = 1)" }
n5 = { algorithm = "a2c", alpha = 2.0, alphaW = 0.3, workers = 4, n = 5, beta = 0.0, gamma = 0.95, maxSteps = 1000, units = 60, name = "every 5 steps (n = 5)" }
n20 = { algorithm = "a2c", alpha = 2.0, alphaW = 0.3, workers = 4, n = 20, beta = 0.0, gamma = 0.95, maxSteps = 1000, units = 60, name = "every 20 steps (n = 20)" }
+++

## Story

::: step {run = "learn", at = 0}
**Four workers in a maze.** The goal G pays $\rew{+1}$ and ends the episode; every other step pays nothing, but with $\gam = 0.95$ a later reward is worth less, so shorter is better. Four copies of the agent start together. They share one actor, a softmax policy per tile (the arrows), and one critic, a value per tile (the colors). Each explores on its own.
:::

::: step {run = "learn", at = 0, play = 1, updates = 2, pace = 90, formula = 1}
All four step at the same time, each its own way. Every $n = 5$ steps the workers pool their last five steps each into **one** update of the shared weights, each step judged by its $n$-step advantage. Watch the first two updates: they change nothing. No worker has found the goal yet, and the critic expects nothing anywhere, so every advantage is 0.
:::

::: step {run = "learn", at = 0, play = 1, updates = 10, instant = true, formula = 2}
The next seven updates go the same way. Then, at step 47, the first worker steps into the goal. In the tenth update its last two steps get positive advantages: the step up into the goal $\err{+1}$, and the one before it, a bump into the right wall, $\err{+0.95}$. The $n$-step return credits every step of the stretch, useful or not; later updates will sort them out. Up, from the tile below the goal, goes from 25% to 36% by the end of the round.
:::

::: step {run = "learn", at = 1, trail = false}
The round ends when every worker has finished its episode: after 47, 97 and 993 steps, and one cut off at the 1000-step limit. Wherever any of them reached the goal, the critic now expects something, and from there the good news spreads back, $n$ steps at a time.
:::

::: step {run = "learn", at = 10, values = true}
After 10 rounds, 40 episodes: values have spread from the goal over most of the maze, though they have barely reached the start. The workers still wander, 108 steps per episode on average, but each round teaches the shared policy four episodes' worth.
:::

::: step {run = "learn", at = 60, values = true}
After 60 rounds the workers take 17 steps on average, against 14 for the shortest path: one policy, learned from 240 episodes played four at a time.
:::

::: step {run = "learn", at = 60, curves = ["n1", "n5", "n20"], metric = "steps"}
Averaged over 12 runs, three update intervals. Updating every step ($n = 1$) passes the reward back one step per update and is slowest: about 30 steps per episode at the end. Every 20 steps learns fastest at first; every 5 steps ends best, at 18. [Race them in the Lab](lab:a2c-maze).
:::

## Textbook

### Many workers, one learner {#workers}

The one-step [[actor-critic|actor–critic]] learns from a single stream of experience, in which consecutive samples are strongly correlated: the same few states, visited one after the other, under a policy that changes after each step. With neural networks this correlation makes learning unstable, the problem that DQN solved with a replay memory ([[experience-replay]]). A replay memory, however, holds data from old policies, which on-policy methods cannot use.

Mnih and colleagues (2016) found another way to decorrelate the data: run **many workers in parallel**, each with its own copy of the environment, all acting with the same policy. At any moment the workers are in different states, at different stages of different episodes, so a batch of their recent experience is far more varied than a stretch of one agent's experience. The data stays on-policy, no memory is needed, and the computation spreads over many processors.

### $n$-step advantages {#advantages}

Each worker plays $n$ steps (or until its episode ends), then judges each of those steps by an $n$-step advantage ([[n-step-td]]): the rewards up to the end of the stretch, plus the critic's estimate of the state where it stops, minus the critic's estimate of the state where the step started,

$$\err{\hat A_t} = \sum_{i=0}^{k-1} \gam^i\,\rew{R_{t+i+1}} + \gam^k\,\val{\hat v(S_{t+k}, \mathbf w)} - \val{\hat v(S_t, \mathbf w)}, \label{eq-adv}$$

where $t + k$ is the end of the stretch, so the first step of a stretch looks $n$ steps ahead and the last looks one step ahead; at the end of an episode the estimate is 0. The update pools all workers' steps:

$$\boldsymbol\theta \leftarrow \boldsymbol\theta + \frac{\alp^{\boldsymbol\theta}}{N} \sum_{\text{workers}} \sum_t \Big[\err{\hat A_t}\,\nabla \ln \pol{\pi(A_t \mid S_t, \boldsymbol\theta)} + \beta\,\nabla \pol{H\big(\pi(\cdot \mid S_t, \boldsymbol\theta)\big)}\Big], \label{eq-actor}$$

$$\mathbf w \leftarrow \mathbf w + \frac{\alp^{\mathbf w}}{N} \sum_{\text{workers}} \sum_t \err{\hat A_t}\,\nabla \val{\hat v(S_t, \mathbf w)}, \label{eq-critic}$$

where $N$ is the number of workers and the last term of \ref{eq-actor} is an **entropy bonus** of weight $\beta$, which keeps the policy from becoming deterministic too soon ([[entropy-bonus]]). The critic's update moves each estimate toward its $n$-step return, $\err{\hat A_t} + \val{\hat v(S_t, \mathbf w)}$. Like most deep-RL code, A2C leaves out the $\gam^t$ factor of the [[pg-theorem|policy gradient theorem]].

The interval $n$ is a bias–variance knob, as for any $n$-step method: short stretches bootstrap often from a critic that may be wrong, long ones carry the noise of many rewards. Mnih and colleagues used $n = 5$.

### A3C and A2C {#a3c}

In **A3C**, *asynchronous advantage actor–critic*, each worker runs in its own thread with a local copy of the weights. After its $n$ steps it computes its gradients, applies them to the shared weights without waiting for, or locking out, the other workers, and copies the shared weights back, in the lock-free style of Hogwild! (Recht et al., 2011). Workers therefore act with slightly different, slightly stale policies. In **A2C**, the synchronous version, a coordinator waits for all workers to finish their $n$ steps, makes one update from the whole batch, and then all workers continue with the same new policy. A2C is simpler, deterministic given the seeds, and makes better use of a GPU through larger batches; OpenAI (2017) found that it performed at least as well as their asynchronous implementation. Both are on-policy: each batch is used once.

### The algorithm {#algorithm}

::: algorithm {#alg-a2c} A2C, synchronous advantage actor–critic
Input: a differentiable policy $\pol{\pi(a \mid s, \boldsymbol\theta)}$ and value estimate $\val{\hat v(s, \mathbf w)}$, step sizes $\alp^{\boldsymbol\theta}, \alp^{\mathbf w}$, a discount $\gam$, an interval $n$, $N$ workers, an entropy weight $\beta \ge 0$
Set the weights $\boldsymbol\theta$ and $\mathbf w$; start every worker in an episode
Repeat:
  Each worker takes up to $n$ steps with $\pol{\pi(\cdot \mid \cdot, \boldsymbol\theta)}$, starting a new episode when one ends
  For each worker and each of its steps $t$: $\err{\hat A_t} \leftarrow$ the $n$-step advantage \ref{eq-adv}
  $\boldsymbol\theta \leftarrow \boldsymbol\theta + \frac{\alp^{\boldsymbol\theta}}{N} \sum \big[\err{\hat A_t}\,\nabla \ln \pol{\pi(A_t \mid S_t, \boldsymbol\theta)} + \beta\,\nabla \pol{H}\big]$
  $\mathbf w \leftarrow \mathbf w + \frac{\alp^{\mathbf w}}{N} \sum \err{\hat A_t}\,\nabla \val{\hat v(S_t, \mathbf w)}$
:::

In the Lab, a **round** lets each worker play one whole episode, with an update every $n$ ticks of the clock; a worker that finishes early waits for the others, and the round's numbers are averages over the workers. In practice, each worker starts a new episode as soon as one ends, so that every update has $N \times n$ samples.

### Deep A2C {#deep}

With neural networks, the actor and the critic usually share a body, with two heads: the softmax (or Gaussian) policy and a single value output. One loss trains both,

$$L(\boldsymbol\theta) = -\sum_t \err{\hat A_t}\,\ln \pol{\pi(A_t \mid S_t, \boldsymbol\theta)} + c_v \sum_t \big(\hat G_t - \val{\hat v(S_t, \boldsymbol\theta)}\big)^2 - \beta \sum_t \pol{H\big(\pi(\cdot \mid S_t, \boldsymbol\theta)\big)}, \label{eq-loss}$$

where $\hat G_t$ is the $n$-step return, $\err{\hat A_t}$ is treated as a constant (no gradient flows through it), and $c_v$ balances the two heads. Mnih and colleagues trained Atari agents this way on 16 CPU threads, faster than DQN on GPUs, and also solved continuous-control tasks with Gaussian policies. A2C is also the starting point of the methods that limit each policy step ([[trpo]], [[ppo]]) and of the advantage estimates that tune the bias–variance trade more finely ([[gae]]).

The guide's recorded A2C keeps two separate networks of 64 + 64 units, with eight workers and a step every 5 of their steps. On CartPole, trained from five seeds per value: learning rates of $3 \cdot 10^{-4}$ and $10^{-3}$ balance the pole in all five runs; at $3 \cdot 10^{-3}$ one run of five falls short, and at $10^{-2}$ one collapses, its last training episodes lasting 9 steps on average. [The run and its sweeps, in the Lab](lab:a2c-cartpole).

### Example: four workers in a maze {#example}

In the Dyna maze ([[dyna-q]]), with a reward of 1 at the goal, $\gam = 0.95$, tabular features, $N = 4$ workers, $\alp^{\boldsymbol\theta} = 2$, $\alp^{\mathbf w} = 0.3$ and no entropy bonus, A2C brings the average episode from about 600 steps to 18 within 60 rounds of four episodes, with $n = 5$; the shortest path has 14 steps. Updating every step ($n = 1$) learns slowest, because each update passes the news of the goal back by a single step; $n = 20$ learns fastest at first and ends a little worse than $n = 5$, at 20 steps.

### Historical remarks {#history}

Asynchronous actor–critic learning with parallel workers, and its one-step and $n$-step value-based siblings, were introduced by Mnih, Badia, Mirza, Graves, Lillicrap, Harley, Silver and Kavukcuoglu (2016), who showed that A3C trained faster on Atari than DQN while using only CPUs. The synchronous variant became the default after OpenAI's baselines (2017), whose authors saw no benefit from asynchrony, and it served as the baseline of Wu, Mansimov, Liao, Grosse and Ba (2017). The asynchronous updates follow the lock-free parallel gradient descent of Recht, Ré, Wright and Niu (2011).

## Card

### Idea

Many workers explore copies of the environment at once with the same policy. Every $n$ steps, their experience is pooled into one update of a shared actor and critic, each step judged by its $n$-step advantage. Parallel workers make the data varied enough for stable learning without a replay memory.

::: analogy
A team of scouts mapping a forest. Each walks a different trail; every hour they meet, compare notes and agree on one shared map, then set off again with it.
:::

### The update {#update}

$$\err{\hat A_t} = \sum_{i=0}^{k-1} \gam^i\,\rew{R_{t+i+1}} + \gam^k\,\val{\hat v(S_{t+k})} - \val{\hat v(S_t)}$$

$$\boldsymbol\theta \leftarrow \boldsymbol\theta + \frac{\alp}{N} \sum \big[\err{\hat A_t}\,\nabla \ln \pol{\pi(A_t \mid S_t, \boldsymbol\theta)} + \beta\,\nabla \pol{H}\big]$$

### One change from the actor–critic {#change}

| | experience | judged by | updates |
| --- | --- | --- | --- |
| [[actor-critic]] | one agent | one-step TD error | every step |
| A2C | $N$ workers in parallel | $n$-step advantage | every $n$ steps, all workers at once |

### Backup diagram {#backup}

{{backup a2c}}

Each worker's stretch of up to $n$ steps, ending in the critic's estimate; the workers' gradients are summed into one update.

### Pseudocode

::: pseudocode
Parameters: step sizes $\alp^{\boldsymbol\theta}$ and $\alp^{\mathbf w}$, discount $\gam$, interval $n$, $N$ workers, entropy weight $\beta$
Set $\boldsymbol\theta$ (here, all moves equally likely) and $\mathbf w$ (here 0)
Repeat for each round:
  Every worker starts an episode {#start}
  Each worker takes a step with $\pol{\pi(\cdot \mid S, \boldsymbol\theta)}$ (all at once, until all episodes end) {#act}
  Every $n$ steps, and at the end: $n$-step advantages $\err{\hat A_t}$ of every worker's recent steps; $\boldsymbol\theta \leftarrow \boldsymbol\theta + \frac{\alp^{\boldsymbol\theta}}{N} \sum [\err{\hat A_t}\,\nabla \ln \pol{\pi} + \beta\,\nabla \pol{H}]$, $\mathbf w \leftarrow \mathbf w + \frac{\alp^{\mathbf w}}{N} \sum \err{\hat A_t}\,\nabla \val{\hat v}$ {#update}
:::

### Perks

- Stable learning with neural networks, on-policy, without a replay memory.
- Scales with processors: more workers, more data per second.
- Simple: an actor–critic with a batch dimension. [See the workers](lab:a2c-maze)

### Flaws

- On-policy: each batch is used once, so it needs many samples.
- Many interacting knobs: $n$, $N$, two step sizes, $\beta$.
- A3C's stale, asynchronous updates are hard to reproduce exactly; A2C fixes that by waiting.

### Knobs

| Knob | Too low | Too high |
| --- | --- | --- |
| $n$ steps per update | slow credit, bias from the critic | noisy returns, fewer updates |
| $N$ workers | correlated data | more compute per update |
| $\alp^{\boldsymbol\theta}$, $\alp^{\mathbf w}$ | slow | the policy collapses, the critic diverges |
| $\beta$ entropy weight | early commitment to a poor policy | the policy stays too random |

### Pitfalls

- Letting gradients flow through the advantage in a shared network: the actor's loss then also moves the critic, the wrong way.
- Workers that start in lockstep from the same state with the same seed: their data is not decorrelated at all.
- Reusing a batch for several updates: that is the job of [[ppo]], which adds the safeguards it needs.

### Check yourself {#check}

::: question
Why do parallel workers help, when they all follow the same policy?
---
They are in different states at different moments, so a batch of their recent steps is varied, like a sample from a replay memory, but every step comes from the current policy, as an on-policy method requires.
:::

::: question
In an update with $n = 5$, which step of a worker's stretch looks furthest ahead?
---
The first: its advantage sums five rewards before bootstrapping. The last step of the stretch looks one step ahead.
:::

::: question
What is the difference between A3C and A2C?
---
A3C's workers update the shared weights asynchronously, each with its own slightly stale copy. A2C waits for all workers and makes one synchronous update from the whole batch, so all workers always act with the same policy.
:::
