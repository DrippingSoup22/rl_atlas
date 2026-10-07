+++
summary = "DQN for continuous actions: an actor network names the action, a critic network learns its value from replayed experience, and the actor follows the critic's slope. Both learn against slowly following copies of themselves. Fast and sample-efficient on Pendulum, where all 20 seeds swing the pendulum up within a few thousand steps; but its critic overestimates, and its knobs need care."
change = "Replace DQN's max over actions by an actor network a = μ(s) trained along the critic's slope (the deterministic policy gradient), and let the target copies follow the networks a little at every step instead of being copied every C steps."
prereqs = ["dpg", "dqn", "target-network"]
lab = "ddpg-pendulum"
sources = [
  { text = "Lillicrap, Hunt, Pritzel, Heess, Erez, Tassa, Silver & Wierstra (2016), Continuous control with deep reinforcement learning, ICLR", url = "https://arxiv.org/abs/1509.02971" },
  { text = "Silver, Lever, Heess, Degris, Wierstra & Riedmiller (2014), Deterministic policy gradient algorithms, ICML", url = "https://proceedings.mlr.press/v32/silver14.html" },
  { text = "Fujimoto, van Hoof & Meger (2018), Addressing function approximation error in actor-critic methods, ICML", url = "https://arxiv.org/abs/1802.09477" },
  { text = "Towers et al. (2024), Gymnasium: a standard interface for reinforcement learning environments (Pendulum-v1)", url = "https://gymnasium.farama.org/environments/classic_control/pendulum/" },
]

[story]
scene = "pendulum"
env = "pendulum"
formula = '''\step{1}{y = \rew{r} + \gam\,\val{\hat q'(s', \pol{\mu'(s')})} \qquad} \step{2}{\nabla_{\boldsymbol\theta} J \approx \nabla_a \val{\hat q(s, a)}\big|_{a = \pol{\mu(s)}}\,\nabla_{\boldsymbol\theta}\pol{\mu(s)}}'''

[story.runs]
learn = { recording = "ddpg-pendulum", name = "DDPG" }
+++

## Story

::: step {run = "learn", at = 0, map = "action"}
**A pendulum, and a motor too weak to lift it.** The torque goes from −2 to 2; far from the top, gravity wins, so the pendulum has to swing up. Every step costs its distance from upright, its spin and a little torque: 0 is the best a step can do. The map shows the **actor**: for each angle and spin, the torque it would apply. Before learning, those torques are small and arbitrary.
:::

::: step {run = "learn", at = 3, formula = 1}
**A critic learns what each torque is worth**, from batches replayed from memory, with targets from slow copies of both networks (formula 1). After 4,500 steps the test episode comes near the top again and again but cannot stay, and ends at −539.
:::

::: step {run = "learn", at = 4, map = "action", formula = 2}
**The actor follows the critic's slope** (formula 2): in each state it nudges its torque the way the critic says the value rises. After 6,000 steps the test changes completely: one swing back, upright by step 30, held to the end. The map shows the plan: below the horizontal, push with the spin to pump the swing; near the top, push against the lean.
:::

::: step {run = "learn", at = 40, map = "value"}
**The critic after 60,000 steps** rates some states at **+32**, though every step costs something and no state can be worth more than 0. DDPG's critic overestimates, and the actor, following its slope, goes where it is most wrong. Here the pendulum still holds; [[td3]] attacks the overestimation.
:::

::: step {run = "learn", at = 40, curves = ["learn"], metric = "test"}
**Twenty seeds**, the test return after each block. Every seed swings the pendulum up within 4,500 to 7,500 steps and holds it to the end. [Play it in the Lab](lab:ddpg-pendulum).
:::

## Textbook

### From DQN to continuous actions {#from-dqn}

DQN needs $\max_{a'} \val{\hat q(s', a')}$ twice: to act and to build targets ([[dqn]]). With a continuous action that max is an optimization problem at every step. DDPG (Lillicrap et al., 2016) replaces it with an **actor** network $\pol{\mu(s, \boldsymbol\theta)}$ that outputs the action, trained with the deterministic policy gradient ([[dpg]]), and keeps the rest of DQN: a replay memory, minibatches, and target networks.

### The two updates {#updates}

The **critic** $\val{\hat q(s, a, \mathbf w)}$ regresses toward targets built from target copies of both networks:
$$y = \rew{r} + \gam\,\val{\hat q\big(s', \pol{\mu(s', \boldsymbol\theta^-)}, \mathbf w^-\big)}, \qquad L(\mathbf w) = \frac1B \sum_j \big(y_j - \val{\hat q(s_j, a_j, \mathbf w)}\big)^2, \label{critic}$$
with just $\rew r$ when $s'$ ends the episode. The **actor** climbs the critic's estimate of its own actions,
$$\nabla_{\boldsymbol\theta} J \approx \frac1B \sum_j \nabla_a \val{\hat q(s_j, a, \mathbf w)}\big|_{a = \pol{\mu(s_j)}}\;\nabla_{\boldsymbol\theta}\pol{\mu(s_j, \boldsymbol\theta)}. \label{actor}$$
In practice: compute $\val{\hat q(s, \pol{\mu(s)})}$ for the batch and take a gradient step that raises it, with the critic's weights held fixed.

### Soft target updates {#soft}

DQN copies its network into the target every $C$ steps ([[target-network]]). DDPG lets both target networks follow a little at every step:
$$\mathbf w^- \leftarrow \tau\,\mathbf w + (1 - \tau)\,\mathbf w^-, \qquad \boldsymbol\theta^- \leftarrow \tau\,\boldsymbol\theta + (1 - \tau)\,\boldsymbol\theta^-, \label{eq-soft}$$
with a small $\tau$: 0.001 in the paper, 0.005 here. The targets then trail the networks by about $1/\tau$ updates, smoothly, without the jumps of a periodic copy.

### The algorithm {#algorithm}

::: algorithm {#alg-ddpg} Deep deterministic policy gradient (DDPG)
Parameters: memory size $N$, minibatch size $B$, step sizes for the critic and the actor, target speed $\tau$, noise spread $\sigma$, random steps at the start
Initialize the actor $\pol{\mu(\cdot, \boldsymbol\theta)}$ and the critic $\val{\hat q(\cdot, \cdot, \mathbf w)}$ at random; their targets $\boldsymbol\theta^- \leftarrow \boldsymbol\theta$, $\mathbf w^- \leftarrow \mathbf w$; an empty memory $\mathcal D$
Repeat for each step:
  In $S$, take $A = \pol{\mu(S, \boldsymbol\theta)} + \epsilon$, $\epsilon \sim \mathcal N(0, \sigma^2)$, clipped to the allowed range (a random action during the first steps); observe $\rew R$, $S'$; store $(S, A, \rew R, S')$ in $\mathcal D$
  Draw $B$ transitions from $\mathcal D$ and compute their targets \ref{critic}
  Critic: a gradient step down the squared error of \ref{critic}
  Actor: a gradient step up $\frac1B \sum_j \val{\hat q(s_j, \pol{\mu(s_j, \boldsymbol\theta)}, \mathbf w)}$, along \ref{actor}
  Both targets move toward their networks, \ref{eq-soft}
:::

The critic is updated first, as in the paper, so the actor climbs slopes that already include the latest batch.

### Exploration {#exploration}

The actor is deterministic, so the agent explores by adding noise to its action: $a = \pol{\mu(s)} + \epsilon$, clipped to the allowed range. The paper used noise correlated in time (an Ornstein–Uhlenbeck process); Fujimoto et al. (2018) found it brought no benefit over independent Gaussian noise, which is what the recording uses: a spread of 0.2, a tenth of the largest torque. Learning is off-policy, so the noise affects what the memory contains, not what the targets mean.

### The recorded run {#recorded}

Networks of 64 + 64 units, Adam with a step size of $10^{-3}$ for both, a memory of 100,000 steps, batches of 128, one update per step after the first 1,000 random ones, $\gam = 0.99$, $\tau = 0.005$, and rewards scaled by 0.1 for learning, so that the critic's targets stay in a comfortable range ([[normalization]]). Twenty seeds, 60,000 steps each: all twenty swing the pendulum up and hold it in their final tests. On this problem DDPG is fast and reliable.

How much do the knobs matter? A sweep trained each of three knobs at several values, 10 seeds per value (the Lab's sweep panel shows every run). On Pendulum DDPG is forgiving: 109 of the 110 runs end holding the pendulum up, and the knobs change the speed more than the odds.

- **Target speed $\tau$** matters most. Half the seeds train at −250 or better within 15,000 steps with $\tau = 0.001$, within 6,750 with the recording's 0.005, and within 4,500 with 0.02. At $\tau = 0.1$ the copies follow the networks too closely: one seed of the ten swung the pendulum up by block 4, then at block 7 its test fell to −1,672 and stayed exactly there to the end, the same score every test, with no recovery.
- **Exploration noise** barely changes the end. With no noise at all, every seed still learns: the first 1,000 random steps explore enough here. With a spread of 0.6 the training episodes stay poor (−452 in the median at the end), while the tests, played without noise, still hold the pendulum up.
- **Reward scale**: rewards multiplied by 1 do as well as by 0.1; by 0.01, half the seeds need 15,750 steps instead of 6,750 ([[normalization]]).

### Overestimation {#overestimation}

The critic is trained on values, and the actor chases its slopes: wherever the critic errs upward, the actor goes, and the target then reads the same inflated value through the target actor. The errors feed on themselves. On Pendulum this shows plainly, because no value can be above 0. In the shown run, along the greedy test episodes of blocks 11 to 40, the critic rated the states 31 above the discounted return that actually followed. Over the 20 seeds, the average target in the last 10 blocks ended above 0, an impossible value, in 10 of them. Here the bias does no visible harm, and every seed still learns. On harder tasks Fujimoto et al. (2018) found it degrades the policy, and their fixes became [[td3]].

### Why it works, and when it does not {#theory}

The deterministic policy gradient theorem ([[dpg]]) says that the actor's update \ref{actor} is the gradient of the return, if $\nabla_a \val{\hat q}$ is the true action-value gradient. DDPG's critic offers no such guarantee. It is a network trained by semi-gradient steps on bootstrapped, off-policy targets, all three parts of the [[deadly-triad]], and the actor uses only its slopes, the part of a value function that regression fits least directly: two critics with the same error in their values can have very different slopes. So, as for [[dqn]], no theorem covers the method. What makes it work in practice is the machinery borrowed from DQN, replay and slowly moving targets, which keep the critic's targets steady enough for its slopes to be roughly right where the data is. Where they are wrong, nothing stops the actor from following them, which is why DDPG is sensitive to its knobs and its seeds, and why its successors spend their effort on the critic ([[td3]], [[sac]]).

### Historical remarks {#history}

Lillicrap et al. (2016) showed one algorithm, with one set of knobs, learning over 20 simulated physics tasks, from joint angles and from pixels, including legged locomotion and a car-driving game. It made deep RL for continuous control practical, and its brittleness to knobs and seeds, documented by Henderson et al. (2018), motivated TD3 and SAC.

## Card

### Idea

An actor $\pol{\mu(s)}$ names a continuous action; a critic $\val{\hat q(s, a)}$ learns its value from a replay memory, against slowly following target copies; the actor follows the critic's slope.

::: analogy
A pupil and a coach. The pupil (the actor) plays; the coach (the critic) watches recordings and says which small change would score better; the pupil adjusts in that direction.
:::

### The update {#update}

$$y = \rew{r} + \gam\,\val{\hat q'(s', \pol{\mu'(s')})}, \qquad \boldsymbol\theta \leftarrow \boldsymbol\theta + \alp\,\nabla_a \val{\hat q(s, a)}\big|_{a = \pol{\mu(s)}}\,\nabla_{\boldsymbol\theta}\pol{\mu(s)}$$

the critic regresses on $y$, the actor follows its slope, and both target copies move by $\tau$ toward the networks every step.

### One change from DQN {#change}

An actor network replaces the max over actions, trained with the deterministic policy gradient; and the target networks follow softly at every step instead of being copied every $C$ steps.

### Backup diagram {#backup}

{{backup ddpg}}

One sampled transition, then a single next action: the one the target actor names, valued by the target critic. No max, no average over actions.

### Pseudocode

::: pseudocode
Parameters: memory $N$, batch $B$, step sizes for actor and critic, $\tau$, noise spread $\sigma$, random steps at the start
Initialize actor $\boldsymbol\theta$ and critic $\mathbf w$ at random; $\boldsymbol\theta^- \leftarrow \boldsymbol\theta$, $\mathbf w^- \leftarrow \mathbf w$; memory $\mathcal D$ empty
Repeat for each step:
  $A \leftarrow \pol{\mu(S, \boldsymbol\theta)} + $ noise of spread $\sigma$, clipped to the allowed range; take it, observe $\rew{R}$, $S'$
  Store $(S, A, \rew{R}, S')$ in $\mathcal D$; sample $B$ transitions from $\mathcal D$
  $y_j \leftarrow \rew{r_j} + \gam\,\val{\hat q(s'_j, \pol{\mu(s'_j, \boldsymbol\theta^-)}, \mathbf w^-)}$ (just $\rew{r_j}$ at the end)
  Critic: a gradient step on $\frac1B \sum_j (y_j - \val{\hat q(s_j, a_j, \mathbf w)})^2$
  Actor: a gradient step that raises $\frac1B \sum_j \val{\hat q(s_j, \pol{\mu(s_j, \boldsymbol\theta)}, \mathbf w)}$
  $\mathbf w^- \leftarrow \tau\mathbf w + (1 - \tau)\mathbf w^-$; $\boldsymbol\theta^- \leftarrow \tau\boldsymbol\theta + (1 - \tau)\boldsymbol\theta^-$
  $S \leftarrow S'$
:::

### Perks

- Continuous actions of any dimension, with no max or integral over them.
- Off-policy with replay: very sample-efficient. On Pendulum, half of the 20 seeds have a block of training episodes averaging −250 or better within 6,000 steps; the recorded PPO needed 35,000.
- Simple: DQN plus an actor.

### Flaws

- The critic overestimates, and the deterministic actor exploits exactly those errors.
- Brittle: sensitive to its knobs, its seed and the scale of the rewards.
- Exploration noise is a hand-set knob, with no sense of what is unknown.
- A deterministic policy cannot represent a random strategy, when one is needed.

### Knobs

| Knob | Too low | Too high |
| --- | --- | --- |
| target speed $\tau$ | targets lag far behind, slow learning | targets chase the networks, as without copies |
| noise spread $\sigma$ | little exploration beyond the random start | training episodes dominated by noise |
| memory $N$ | replay of the latest policy only | fitting long-gone behavior |
| step sizes | slow | the actor overshoots the critic's slopes |
| reward scale | tiny targets, vanishing gradients | huge targets and steps |

### Pitfalls

- Reading the critic's values as truth: check them against returns, or against a known bound.
- Judging the policy by training returns, which include the exploration noise; test without it.
- Forgetting to bound the actor's output to the allowed actions (a tanh and a scale).
- Treating a time limit as a terminal state.

### Check yourself {#check}

::: question
Why can the actor's update not use the target critic?
---
The actor should improve against the current estimate of its actions' values, and the critic's slope at μ(s) is what it follows. The target critic is a slow copy meant to give the critic stable targets; following it would make the actor chase an older estimate. (The target actor does appear, inside the critic's target.)
:::

::: question
On Pendulum every step's reward is 0 or less. What does a critic value of +32 say?
---
That the critic overestimates: no sum of non-positive rewards can be positive. The errors of the critic are amplified by the actor, which moves toward actions the critic rates too high, and by the target, which reads those same actions through the target actor.
:::

::: question
What does a soft update with τ = 0.005 do that a hard copy every 200 steps does not?
---
Both keep the target about 200 updates behind. The soft one does it smoothly, a little each step, so the targets never jump; with a hard copy, the targets stay frozen and then change all at once.
:::
