+++
summary = "A Gaussian policy that draws a fresh sample at every step shakes: on a robot, that is wear, heat and pushes that cancel out. State-dependent exploration makes the noise a function of the state instead, ε(s) = Σᵢ zᵢ(s) θᵢ with θ drawn every few steps, so the same state gets the same nudge and nearby states nearby nudges. The policy stays a Gaussian whose spread is learned and depends on the state, so PPO, A2C and SAC can use it as they are. gSDE trained SAC directly on real robots, where per-step noise had to be stopped for damaging the hardware."
prereqs = ["action-noise", "parameter-noise", "policy-parameterization"]
lab = "gsde-car"
sources = [
  { text = "Raffin, Kober & Stulp (2021), Smooth exploration for robotic reinforcement learning, CoRL", url = "https://arxiv.org/abs/2005.05719" },
  { text = "Rückstieß, Felder & Schmidhuber (2008), State-dependent exploration for policy gradient methods, ECML PKDD", url = "https://link.springer.com/chapter/10.1007/978-3-540-87481-2_16" },
  { text = "Raffin et al. (2021), Stable-Baselines3: reliable reinforcement learning implementations, JMLR (and the RL Zoo's tuned settings)", url = "https://jmlr.org/papers/v22/20-1364.html" },
  { text = "Plappert et al. (2018), Parameter space noise for exploration, ICLR", url = "https://arxiv.org/abs/1706.01905" },
]

[story]
scene = "hillcar"
env = "mountain-car-continuous"
formula = '''\step{1}{a = \pol{\mu(s)} + \sum_i z_i(s)\,\theta_{ij} \qquad \theta_{ij} \sim \mathcal N(0, \knob{\sigma_{ij}}^2) \qquad} \step{2}{\pol{\hat\sigma_j(s)} = \sqrt{\textstyle\sum_i \big(z_i(s)\,\knob{\sigma_{ij}}\big)^2}}'''

[story.runs]
white = { recording = "sac-car", name = "SAC" }
sde = { recording = "sac-car-gsde", name = "SAC with gSDE" }
pink = { recording = "sac-car-pink-noise", name = "SAC, pink noise" }
+++

## Story

::: step {run = "white", at = 40, map = "action"}
**SAC on the hill.** SAC draws its force from a bell curve whose mean and spread the actor computes, with a fresh sample at every step: white noise, as TD3's ([[action-noise]]). On none of 20 seeds does a training episode reach the flag in 30,000 steps. The actor settles on standing still, and its spread narrows from 0.8 to 0.09.
:::

::: step {run = "sde", at = 0, formula = 1}
**gSDE: noise that is a function of the state** (formula 1). The actor's last hidden layer, 64 numbers $\mathbf z(s)$ worked out from the car's position and speed, times a matrix of normal numbers drawn anew every 8 steps. While a matrix is held, the same state gets the same nudge, and nearby states nearby ones; each draw brings a new nudging function, and the actor learns how wide the draws should be.
:::

::: step {run = "sde", at = 10, map = "value"}
**Smooth nudges reach the flag.** Within 5,250 steps a training episode of this seed gets there, and after 7,500 the critic rates the approach to the flag at up to 81. The actor's own push, though, depends on where the car is and not on how it moves, left on the left of the valley and right on the right, and a push like that cannot pump a swing ([[parameter-noise]]): its test episode stays at the bottom.
:::

::: step {run = "sde", at = 19, map = "action", formula = 2}
**Still a bell curve.** For a given state, the force is the mean plus a weighted sum of normal numbers: a Gaussian whose spread depends on the state (formula 2), so SAC learns with it as with its own samples. After 14,250 steps the spread runs from 0.02 to 1.0 along the test episode, the bar under the car; the actor now pushes with the motion in most of the map, and the test reaches the flag in 187 steps.
:::

::: step {run = "pink", at = 6, map = "action"}
**Pink noise in SAC's samples**: each episode's normal numbers replaced by pink ones, so that every force is still drawn from the actor's bell curve, but consecutive forces are related. This seed's training reaches the flag within 1,500 steps, its test after 4,500.
:::

::: step {run = "pink", at = 40, curves = ["white", "sde", "pink"], metric = "test"}
**Twenty seeds of each.** SAC with its own samples: no seed ends up at the flag. With gSDE: 12, half of them within 15,750 steps. With pink noise: all 20, half within 6,375. A gSDE matrix lasts 8 steps, a ninth of the car's swing; pink noise keeps power at every time scale, the swing's included. Held for 32 steps instead, close to half a swing, the matrices took 9 seeds of 10 to the flag in the Lab's sweep, against 5 at 8 steps.
:::

::: step {run = "sde", at = 40, curves = ["white", "sde", "pink"], metric = "jitter-train"}
**The smoothest of the three.** How much the training forces jump from step to step, once the first 1,000 random steps are over: 0.4 on average with gSDE, 1.0 with pink noise, and with SAC's own samples 15 at first, falling to 0.4 as its spread narrows on a policy that stands still. On a robot, smooth training is what gSDE is for. [Race the three in the Lab](lab:gsde-car).
:::

## Textbook

### Noise that depends on the state {#idea}

A Gaussian policy samples every action afresh: $a_t = \pol{\mu(s_t)} + \pol{\sigma}\varepsilon_t$ with a new $\varepsilon_t$ at every step. On a simulator that is harmless; on a robot the motors receive a jittering command, which wears them, heats them and, through the robot's inertia, explores little: pushes a few steps apart cancel ([[action-noise]]). Correlated noise fixes the timing, but cannot be added to an on-policy method's samples, because PPO and A2C must know the probability of each action they took.

**State-dependent exploration** (SDE, Rückstieß et al., 2008) keeps a Gaussian policy and makes its noise a function of the state:
$$a = \pol{\mu(s)} + \varepsilon(s), \qquad \varepsilon(s) = \boldsymbol\theta_\varepsilon^\top \mathbf z(s), \qquad \theta_{ij} \sim \mathcal N(0, \knob{\sigma_{ij}}^2), \label{eq-sde}$$
where $\mathbf z(s)$ are features of the state and the matrix $\boldsymbol\theta_\varepsilon$ is drawn once and then held. While it is held, the noise is a fixed, smooth function of the state: a state that comes back gets the same nudge, a nearby state a nearby one, as with parameter noise ([[parameter-noise]]). And yet the policy remains a Gaussian whose density is known. For one state, the action is $\pol{\mu(s)}$ plus a sum of normals, so
$$\pol{\pi(a_j \mid s)} = \mathcal N\big(\pol{\mu_j(s)},\ \pol{\hat\sigma_j(s)}^2\big), \qquad \pol{\hat\sigma_j(s)} = \sqrt{\textstyle\sum_i \big(z_i(s)\,\knob{\sigma_{ij}}\big)^2}, \label{eq-spread}$$
a spread that depends on the state, through the features, with one learnable $\knob{\sigma_{ij}}$ per pair of feature and action dimension. Its log-probability and gradient are those of any Gaussian ([[policy-parameterization]]): PPO, A2C and SAC can use it unchanged.

### What gSDE changed {#generalized}

Rückstieß et al. used the state itself as the features and drew the matrix once per episode. Raffin, Kober and Stulp (2021) found two problems with that for deep reinforcement learning, and fixed them in **generalized SDE (gSDE)**:

- **Features.** With the state as features, the spread grows with the number of state variables and depends on their units, so the initial $\knob\sigma$ must be tuned per task and the state normalized. gSDE uses the policy network's own last hidden layer, $\mathbf z(s)$ just before the output $\pol{\mu(s)} = \mathbf W\mathbf z(s)$: the noise then depends on the state non-linearly, and its size depends only on the network. The features are held fixed when the noise's gradient is taken: the noise does not reshape them.
- **How often to draw.** One matrix per episode explores too little over a long episode. gSDE draws a new one every $n$ steps: $n = 1$ is ordinary per-step noise, $n$ = the episode is the original SDE, and everything between trades smoothness for variety.

With several workers, each draws its own matrix, which explores more broadly than one shared matrix.

### Smoothness, measured {#continuity}

Raffin et al. measure how much an agent shakes with a **continuity cost**,
$$C = 100 \times \mathbb E_t\Big[\Big(\frac{a_{t+1} - a_t}{\Delta a_{\max}}\Big)^2\Big], \label{eq-continuity}$$
with $\Delta a_{\max}$ the width of the action range: 0 for a constant action, 100 for an action that jumps from one end of the range to the other at every step. Measured during training, it stands for the wear on a robot. The Lab charts it as **jitter**, for the test episodes and for the training episodes.

On four PyBullet locomotion tasks with SAC, 10 runs of a million steps each, per-step noise cost 3.6 to 5.1 during training; gSDE with a new matrix every 8 steps cost 2.4 to 4.1, and scored higher on three of the tasks and a little lower on the fourth, HalfCheetah. Ornstein–Uhlenbeck and parameter noise were smoother still, but scored lower. For PPO the interval mattered most: drawn too rarely, the noise did not explore enough in long episodes; drawn every step, the old problems came back. PPO with gSDE beat PPO with per-step Gaussian noise on all four tasks.

On continuous Mountain Car, over 20 seeds of 30,000 steps here, SAC with its own per-step samples never reached the flag, even in training. With gSDE, a new matrix every 8 steps, 12 seeds ended up reaching it, and its training forces were the smoothest of all: a jitter of 0.4 on average once its first random steps were over, against 1.0 with pink noise in SAC's samples, which reached the flag on all 20 seeds ([[action-noise]]). On the car the interval decides much. In the Lab's sweep, 10 seeds per interval, a new matrix every step took no seed to the flag, every 4 steps 2, every 8 steps 5, every 32 steps 9 and every 128 steps 6, and the longer each matrix lasted, the smoother the training forces: a jitter of 3.1, 0.8, 0.4, 0.2 and 0.07. Held for 32 steps, about half the car's swing of 73, gSDE nearly matched pink noise, with a fifth of its jitter: the interval should match how long a push must last.

### On real robots {#robots}

The experiment that motivated it was on a robot: a tendon-driven elastic neck. With per-step noise, SAC had to be stopped early, because the high-frequency commands were wearing the tendons through friction on their bearings. With gSDE, SAC trained for two hours directly on the robot and reached the precision of the existing model-based controller (under 3 mm and 3°). A policy trained with per-step noise behind a 2 Hz low-pass filter was as precise, but its actions jumped twice as much. The same recipe taught an elastic quadruped to walk and an RC car to drive a track, all on the real hardware, without a simulator or a filter.

### Settings {#settings}

Stable-Baselines3 implements gSDE for A2C, PPO and SAC (`use_sde`, `sde_sample_freq`), and its RL Zoo uses it in many tuned settings: PPO on Pendulum draws a new matrix every 4 steps, SAC on continuous Mountain Car starts from a small spread ($\ln\knob\sigma = -3.67$). The library's default draws one matrix per batch of experience, and for PPO that can be far too rarely: in a trial of three seeds here, PPO on Pendulum with one matrix per batch of 512 steps ended with test returns of $-593$, $-1{,}051$ and $-1{,}215$, nowhere near upright, and with a new matrix every 4 steps two of the three ended near $-125$, holding it up. The trainer of this guide draws every 4 steps for PPO and A2C, and every 8 for SAC, as the paper did. Even so, gSDE did not make PPO better on Pendulum here: in the Lab's sweep, 10 seeds of 200,000 steps, it learned to swing the pendulum up as fast as PPO's own per-step noise, with half the jitter, but held it less reliably, ending up holding it on 4 seeds against 8. With SAC the spread starts small ($\ln\knob\sigma = -3$ in the paper, and here) and is learned along with the entropy bonus; with PPO it starts larger, here $\ln\knob\sigma = 0$, as PPO's own spread does.

::: example {#ex-gsde-numbers} One action from gSDE
Take a policy whose last hidden layer has three units, at a state where they read $\mathbf z = (0.5, -1, 0.2)$, and one action dimension with spreads $\knob{\sigma_{i}} = (0.4, 0.3, 1)$. The action's spread there is $\sqrt{(0.5 \cdot 0.4)^2 + (1 \cdot 0.3)^2 + (0.2 \cdot 1)^2} = \sqrt{0.04 + 0.09 + 0.04} = 0.41$. Now draw the matrix for the next $n$ steps: three unit normals, say $(0.5, 1, -0.2)$, times the spreads, $\boldsymbol\theta = (0.2, 0.3, -0.2)$. The noise in this state is $0.5 \cdot 0.2 - 1 \cdot 0.3 + 0.2 \cdot (-0.2) = -0.24$, and it is $-0.24$ every time the agent comes back here until the next draw. A state whose features read $(0.6, -1, 0.2)$ gets $-0.22$: close states, close nudges.
:::

### Historical remarks {#history}

Rückstieß, Felder and Schmidhuber (2008) introduced state-dependent exploration for policy gradients with linear policies, arguing that per-step noise makes the gradient's variance grow with the episode's length. Raffin, Kober and Stulp (2021) generalized it for deep networks, with the policy's features and periodic resampling, built it into Stable-Baselines3, and trained on three real robots with it.

## Card

### Idea

Make the exploration noise a function of the state, $\varepsilon(s) = \boldsymbol\theta^\top\mathbf z(s)$ with $\mathbf z$ the policy's last hidden layer and $\boldsymbol\theta$ drawn every $n$ steps. The noise is smooth and consistent, and the policy is still a Gaussian with a known density, so on-policy methods can use it.

::: analogy
A new driver wobbles the wheel at random every second. Another has a slightly different style every few minutes, steering a bit wide in left curves, say, and keeps it until the next change: consistent, smooth, and still varied over the drive.
:::

### The update {#update}

The noise is a function of the state: the policy's features $z(s)$ times a random matrix, drawn again every $n$ steps.

$$a = \pol{\mu(s)} + \sum_i z_i(s)\,\theta_{ij}, \quad \theta_{ij} \sim \mathcal N(0, \knob{\sigma_{ij}}^2)$$

So each action dimension is normal around the mean, with a spread that depends on the state.

$$\pol{\pi(a_j \mid s)} = \mathcal N\Big(\pol{\mu_j(s)}, \sum_i \big(z_i(s)\knob{\sigma_{ij}}\big)^2\Big)$$

### Pseudocode

::: pseudocode
Parameters: the interval $n$ between draws ($-1$: once per episode), the starting log spreads $\ln\knob{\sigma_{ij}}$; the rest is SAC's
Initialize SAC: the actor (its last hidden layer $\mathbf z(s)$, its mean $\pol{\mu(s)}$), the matrix of log spreads, two critics and their target copies, $\ln\alpha$; an empty memory $\mathcal D$
Repeat for each episode:
  Repeat for each step $t$ of the episode:
    If $t = 0$ or $t$ is a multiple of $n$: draw the matrix, $\theta_{ij} \sim \mathcal N(0, \knob{\sigma_{ij}}^2)$
    $A \leftarrow a_{\max}\tanh\big(\pol{\mu(S)} + \boldsymbol\theta^\top\mathbf z(S)\big)$ (a random action during the first steps); take it, observe $\rew R$, $S'$; store $(S, A, \rew R, S')$ in $\mathcal D$
    SAC's update from a batch of $\mathcal D$, its samples drawn with a fresh matrix: the policy is the Gaussian of mean $\pol{\mu(s)}$ and spreads $\pol{\hat\sigma_j(s)} = \sqrt{\sum_i (z_i(s)\knob{\sigma_{ij}})^2}$, squashed by $\tanh$, the features $\mathbf z$ held fixed in the spreads
    $S \leftarrow S'$
:::

### Pitfalls

- Drawing the matrix too rarely: over a long episode the agent explores one behavior only.
- Drawing it every step: per-step noise again, with its jitter.
- Letting the noise's gradient reshape the features: gSDE holds them fixed there.
- Using the raw state as features without normalizing it: the spread then depends on the state's units.

### Check yourself {#check}

::: question
PPO cannot use colored noise in its samples. Why can it use gSDE?
---
With the matrix drawn, gSDE's action in a state is the mean plus a sum of normals, a Gaussian whose spread is known: σ̂ⱼ(s) = √(Σᵢ (zᵢ(s)σᵢⱼ)²). PPO stores that probability with each action, as it would for any Gaussian, so its ratios remain correct.
:::

::: question
What do the two ends of the sampling interval give?
---
A new matrix every step is ordinary per-step Gaussian noise: varied but jittery. One matrix per episode is the original SDE: smooth and consistent, but one behavior per episode. gSDE picks an interval in between, such as 4 or 8 steps.
:::
