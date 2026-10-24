+++
summary = "DDPG and TD3 explore by adding noise to the actor's action, SAC by drawing its action from a bell curve: either way a fresh, independent number at every step, white noise. A body with inertia averages such pushes away before it moves. Noise correlated in time pushes the same way for a while and gets further: Ornstein–Uhlenbeck noise, red noise and, halfway between white and red, pink noise. On continuous Mountain Car, TD3 with white noise learns to stand still, and with pink noise of the same spread it learns to swing up to the flag."
prereqs = ["td3", "exploration-strategies"]
lab = "action-noise-car"
sources = [
  { text = "Eberhard, Hollenstein, Pinneri & Martius (2023), Pink noise is all you need: colored noise exploration in deep reinforcement learning, ICLR", url = "https://iclr.cc/virtual/2023/oral/12540" },
  { text = "Lillicrap et al. (2016), Continuous control with deep reinforcement learning, ICLR", url = "https://arxiv.org/abs/1509.02971" },
  { text = "Fujimoto, van Hoof & Meger (2018), Addressing function approximation error in actor-critic methods, ICML", url = "https://arxiv.org/abs/1802.09477" },
  { text = "Uhlenbeck & Ornstein (1930), On the theory of the Brownian motion, Physical Review 36", url = "https://doi.org/10.1103/PhysRev.36.823" },
  { text = "Timmer & König (1995), On generating power law noise, Astronomy and Astrophysics 300", url = "https://ui.adsabs.harvard.edu/abs/1995A%26A...300..707T/abstract" },
  { text = "Dabney, Ostrovski & Barreto (2021), Temporally-extended ε-greedy exploration, ICLR", url = "https://arxiv.org/abs/2006.01782" },
  { text = "Korenkevych, Mahmood, Vasan & Bergstra (2019), Autoregressive policies for continuous control deep reinforcement learning, IJCAI", url = "https://arxiv.org/abs/1903.11524" },
  { text = "Ciosek, Vuong, Loftin & Hofmann (2019), Better exploration with optimistic actor-critic, NeurIPS", url = "https://arxiv.org/abs/1910.12807" },
  { text = "Raffin et al. (2021), Stable-Baselines3: reliable reinforcement learning implementations, JMLR (and the RL Zoo's tuned settings)", url = "https://jmlr.org/papers/v22/20-1364.html" },
  { text = "Towers et al. (2024), Gymnasium (MountainCarContinuous-v0)", url = "https://gymnasium.farama.org/environments/classic_control/mountain_car_continuous/" },
]

[story]
scene = "hillcar"
env = "mountain-car-continuous"
formula = '''\step{1}{a_t = \operatorname{clip}\big(\pol{\mu(s_t)} + \knob{\sigma}\,\varepsilon_t\big) \qquad} \step{2}{\varepsilon_1, \dots, \varepsilon_T:\ \text{power} \propto 1/f^{\knob{\beta}}}'''

[story.runs]
white = { recording = "td3-car-white-noise", name = "white" }
ou = { recording = "td3-car-ou-noise", name = "Ornstein–Uhlenbeck" }
pink = { recording = "td3-car-pink-noise", name = "pink" }
red = { recording = "td3-car-red-noise", name = "red" }
+++

## Story

::: step {run = "white", at = 0, map = "action"}
**A car in a valley, and an engine weaker than the slope.** The force runs from −1 to 1; every step costs 0.1 × force², the flag on the right pays 100, and the episode ends there or after 999 steps. Pushing toward the flag is not enough: the car must swing back and forth, a little higher each time. The map shows the actor's push for each position and speed. To swing, the push must follow the speed: right while the car moves right, left while it moves left.
:::

::: step {run = "white", at = 20, map = "value", formula = 1}
**White noise: a fresh number at every step** (formula 1), of spread 0.5, half the largest force. In 20,000 steps of training, no episode of this seed reaches the flag: each costs about 24 in pushes and pays nothing. The critic learns exactly that, every state worth a little less than 0, and the actor learns to stand still: its test episode, played without noise, wobbles at the bottom of the valley.
:::

::: step {run = "pink", at = 5, map = "value", formula = 2}
**Pink noise of the same spread, drawn one episode at a time** (formula 2, with β = 1): consecutive pushes agree for a while, so they add up. Within 5,000 steps a training episode reaches the flag, and the critic begins to rate the states that lead there above 0.
:::

::: step {run = "pink", at = 15, map = "action"}
**The critic knows; the actor does not yet.** After 15,000 steps the critic rates the approach to the flag at up to 177. But the actor pushes right in every state, whatever the speed, and a push that ignores the speed cannot pump a swing: its test episode still wobbles at the bottom.
:::

::: step {run = "pink", at = 29, map = "action"}
**After 29,000 steps** the actor pushes left in some of the states where the car rolls left. One swing back, then up: its test episode reaches the flag in 108 steps. By the end it follows the speed in 91% of the map, full force either way, and reaches the flag in 78.
:::

::: step {run = "pink", at = 40, curves = ["white", "ou", "pink", "red"], metric = "test"}
**Twenty seeds of each.** With white noise, 1 seed ends up reaching the flag; with Ornstein–Uhlenbeck noise, 1; with red noise, 7; with pink noise, 18. In training, red and pink noise reach the flag on every seed, white noise on 2, and Ornstein–Uhlenbeck noise, which restarts from 0 at each episode and grows slowly, on 4. Red noise finds the flag as surely as pink, but it spends its spread on slow drifts and tries the pushes near the actor's own far less, and fewer of its learners end up at the flag.
:::

::: step {run = "pink", at = 40, curves = ["white", "ou", "pink", "red"], metric = "jitter-train"}
**What a motor would feel.** How much the training pushes jump from one step to the next, on average after the first 1,000 steps of random pushes: 11.4 with white noise, 2.3 with pink, 0.1 with red, 0.03 with Ornstein–Uhlenbeck. The colors that explore better are also the gentler ones ([[smooth-policies]]). SAC's samples are white noise too: [[gsde]] races it against pink noise and noise that depends on the state. [Race the four colors in the Lab](lab:action-noise-car).
:::

## Textbook

### Why the timing of the noise matters {#why}

A deterministic actor explores by adding noise to its action. DDPG and TD3 add a fresh normal number at every step, $a_t = \operatorname{clip}(\pol{\mu(s_t)} + \knob{\sigma}\varepsilon_t)$ with $\varepsilon_t \sim \mathcal N(0, 1)$, independent from one step to the next; SAC draws its action from a bell curve around its mean, which comes to the same, a new independent draw every step ([[sac]]). Independent numbers have the same power at every frequency, as white light has every color: they are **white noise**.

A body with mass does not follow a force, it integrates it. In continuous Mountain Car a force of 1 changes the car's speed by 0.0015 per step, so where the car goes depends on the *sum* of many pushes. Independent pushes of spread $\knob\sigma$ add up like a random walk: after $n$ steps their sum is typically $\knob{\sigma}\sqrt n$, against $\knob\sigma n$ for pushes that keep their sign. The car swings in its valley with a period of about 73 steps, and to pump the swing a push must keep its direction for half of that, about 36 steps: white noise delivers $\sqrt{36} = 6$ pushes' worth where 36 were needed. The pushes cancel before the car moves. A larger $\knob\sigma$ takes it further, but the range caps every push, and clipping turns more and more of them into pushes at its ends: in the Lab's sweep, 10 seeds per spread, TD3 with white noise ended up reaching the flag on no seed at $\knob\sigma = 0.2$, on 1 at 0.5 and on 4 at 1, the full force, where pink noise of spread 0.5 took all 10 there.

Noise that is **correlated in time** keeps its sign for a while and lets the pushes add up. How long it keeps it is the noise's *color*.

::: figure {#fig-noise-colors}
{{noise-colors}}
Four kinds of noise of the same spread, one episode each, and where each takes the car with no policy at all (the actor's action is 0, only the noise pushes). The counts below come from many such episodes, computed in your browser.
:::

### Ornstein–Uhlenbeck noise {#ou}

DDPG's paper used an **Ornstein–Uhlenbeck process**, the velocity of a particle in Brownian motion, "to explore well in physical environments that have momentum" (Lillicrap et al., 2016):
$$\varepsilon_{t+1} = \varepsilon_t + \theta\,(0 - \varepsilon_t)\,\Delta t + \knob{\sigma}\sqrt{\Delta t}\;\xi_t, \qquad \xi_t \sim \mathcal N(0, 1), \label{eq-ou}$$
a random walk pulled gently back toward 0. The paper gives $\theta = 0.15$ and $\knob\sigma = 0.2$. OpenAI Baselines and Stable-Baselines3 take $\Delta t = 0.01$, so that each step keeps $1 - \theta\Delta t = 99.85\%$ of the noise: it takes hundreds of steps to forget a push. The process restarts from 0 at each episode, and its spread grows slowly from there: after 35 steps it reaches a third of its settled value, after 100 steps half, after 300 steps three quarters. Here the noise is scaled so that its settled spread is $\knob\sigma$ itself, as for the other colors. In Baselines and Stable-Baselines3, $\knob\sigma$ is instead the size of each step's kick, and the settled spread is $\knob\sigma/\sqrt{2\theta} \approx 1.8\,\knob\sigma$: the RL Zoo's $\knob\sigma = 0.5$ for Mountain Car settles near 0.9, almost the full force.

TD3's authors found that Ornstein–Uhlenbeck noise brought no benefit over independent Gaussian noise on their MuJoCo tasks, and used the latter (Fujimoto et al., 2018), which is how most code since has explored.

### Colored noise {#colored}

White and Ornstein–Uhlenbeck noise are two members of a family. Write the noise's power at frequency $f$, how much of its variance comes from oscillations of that frequency, as
$$S(f) \propto \frac{1}{f^{\knob\beta}}. \label{eq-color}$$
- $\knob\beta = 0$: **white** noise, the same power at every frequency; consecutive numbers are independent.
- $\knob\beta = 2$: **red** noise, also called Brownian, the sum of white noise; almost all its power is in slow drifts. An Ornstein–Uhlenbeck process with $\Delta t = 0.01$ is red over the length of an episode.
- $\knob\beta = 1$: **pink** noise, halfway: the same power in every octave, as much between periods of 10 and 20 steps as between 100 and 200.

Measured on the trainer's own noise over 999-step episodes, consecutive numbers correlate at 0.77 for pink noise and 0.995 for red, against 0 for white. In the band of periods that pumps the car's swing (36 to 144 steps), pink noise puts 19% of its variance, white 4%, red 5% and Ornstein–Uhlenbeck 3%: red noise and Ornstein–Uhlenbeck spend 94% to 96% of theirs on drifts slower than that, which push one way for most of an episode.

Colored noise is drawn **one episode at a time** (Timmer & König, 1995): draw a Gaussian number for every frequency of the episode, scale it by $f^{-\knob\beta/2}$, transform back to time with an inverse Fourier transform, and scale the result to a spread of 1. The agent then adds $\knob\sigma\varepsilon_t$ to its action at step $t$, exactly as with white noise. For a stochastic policy such as SAC's, the colored numbers replace the normals in its sample, $a_t = \tanh(\pol{\mu(s_t)} + \pol{\sigma(s_t)}\,\varepsilon_t)$: each action still follows the policy's own distribution, but consecutive actions are now related.

### Which color {#which}

There is a trade-off. White noise explores *locally*: it tries actions near the policy's own at every step, which is what fine control needs. Red noise travels *far*: it pushes one way long enough to reach distant states, but spends its variance on slow drifts and tries the actions around the policy's own much less. Pink noise does both at once, some of its power at every time scale.

Eberhard et al. (2023) compared the colors as action noise for SAC and MPO across a wide range of benchmark tasks and found pink noise significantly better than white noise, Ornstein–Uhlenbeck noise and the other alternatives they tried; they recommend it as the default action noise for continuous control. The same choice fits DDPG and TD3, which their package also serves. The best color for one task can be redder or whiter; pink is the one that is rarely far from the best. On continuous Mountain Car, over 20 seeds of 40,000 steps, TD3 ended up reaching the flag on 18 seeds with pink noise, on 7 with red noise, and on 1 each with white and with Ornstein–Uhlenbeck noise. SAC, whose samples are white noise, reached it on none of 20 seeds in 30,000 steps, and on all 20 with pink numbers in its samples.

::: example {#ex-car-noise} Noise alone on the hill
Drive the car with noise alone, the actor's action held at 0, at a spread of half the largest force. Over 400 episodes of 999 steps, white noise never reaches the flag: the furthest the car gets toward it averages $x = -0.22$, a third of the way from the valley's bottom ($-0.52$) to the flag ($0.45$). Ornstein–Uhlenbeck noise never reaches it either. Red noise reaches it in 11% of the episodes, pink noise in 39%. With the full force as the spread, white noise reaches it in 2% of the episodes, Ornstein–Uhlenbeck in 12%, red in 55% and pink in 72%. A learner can only learn the flag's worth from episodes that reach it.
:::

### The noise's size {#scale}

- **Relative to the action's range.** The spread is a fraction of the largest action: TD3 uses 0.1, the RL Zoo's settings for Mountain Car 0.5. Scaling every action dimension to $[-1, 1]$ first, as Stable-Baselines3 advises, lets one spread mean the same for every joint ([[policy-spread]]).
- **Clipped after adding.** The noisy action is clipped to the allowed range. With a spread near the range itself, clipping turns much of the noise into pushes at the extremes.
- **Large enough to reach.** On the car the spread mattered for every color. Pink noise ended up at the flag on 1 seed of 10 at $\knob\sigma = 0.2$ and on all 10 at 0.5 and at 1, faster at 1 (half the seeds within 12,500 steps, against 18,000), with training pushes that jumped two and a half times as much ([[smooth-policies]]).
- **A schedule.** Some implementations lower the spread over training, or give each parallel worker its own. Lower it only once the task is solved: the noise is what finds the reward.
- **A random start.** Before learning, TD3 and SAC act uniformly at random for a while (TD3's paper: the first 1,000 or 10,000 steps, depending on the task; 1,000 here). Uniform random actions are white noise too: in Mountain Car they never reach the flag.

### Why any noise is allowed {#off-policy}

DDPG, TD3 and SAC learn off-policy ([[on-off-policy]]): their critics learn the value of the actor's own actions from whatever transitions are in the memory. The noise changes what the memory contains, not what the targets mean, so any noise, of any color, is allowed. On-policy methods such as PPO are different: their updates weigh each action by its probability under the policy that took it, so the noise must be part of the policy. A Gaussian policy whose noise is correlated in time needs a different construction, [[gsde]].

### Other ways to correlate exploration {#others}

- **Action repeat.** Hold each action for $k$ steps: DQN on Atari repeated each one for 4 frames, and agents learning from pixels on the DeepMind Control Suite repeat theirs 2 to 8 times. Fewer decisions, longer pushes, and smoother control ([[smooth-policies]]).
- **Noise in the weights.** Perturb the actor's weights once per episode: the same state then gets the same exploratory action, consistently, for the whole episode ([[parameter-noise]]).
- **Temporally extended ε-greedy.** For a list of actions: when exploring, repeat the random action for a random number of steps, drawn from a heavy-tailed distribution (Dabney et al., 2021).
- **Autoregressive noise.** Replace each step's normal number with a smooth process whose every value is still a standard normal: the action is then a Gaussian given the last few states and actions, so PPO can learn with it too, and an Ornstein–Uhlenbeck process is its simplest case. The faster the control, the more per-step noise cancels out: on a real arm commanded 125 times a second, PPO with per-step noise learned nothing in five hours, and with this noise it learned in half of its runs (Korenkevych et al., 2019).
- **Optimism.** Shift the exploring action toward where an optimistic estimate rises, the two critics' mean plus their disagreement, as the optimistic actor–critic does on top of SAC (Ciosek et al., 2019); or reward novelty directly ([[exploration-strategies]]).

### Historical remarks {#history}

Uhlenbeck and Ornstein (1930) described their process as the velocity of a particle in a fluid, kicked by molecules and slowed by friction. DDPG (Lillicrap et al., 2016) used it as exploration noise; TD3 (Fujimoto et al., 2018) dropped it for independent Gaussian noise, and for five years most code explored with white noise. Eberhard et al. (2023) put white and red noise into one family and made pink noise the recommended default, with a drop-in package for Stable-Baselines3.

## Card

### Idea

Exploration noise has a timing as well as a size. White noise, a fresh number every step, cancels out on anything with inertia; noise correlated in time keeps pushing the same way. Pink noise, power $\propto 1/f$, mixes every time scale and is a strong default for DDPG, TD3 and SAC.

::: analogy
Pushing a child on a swing. Random taps, each in any direction, leave the swing where it is; pushes in time with the swing build it up. Red noise is one long push that walks the swing off to one side.
:::

### The colors {#colors}

The color of a noise: how its power falls with the frequency $f$. White ($\knob\beta = 0$) has no memory, pink ($\knob\beta = 1$) some, red ($\knob\beta = 2$) a lot.

$$\text{power} \propto 1/f^{\knob\beta}$$

One sequence per episode, scaled to spread 1, added to the deterministic action.

$$a_t = \operatorname{clip}\big(\pol{\mu(s_t)} + \knob\sigma\varepsilon_t\big)$$

Ornstein–Uhlenbeck noise: each value a fraction of the last plus a small normal, starting from 0 at each episode.

$$\varepsilon_{t+1} = 0.9985\,\varepsilon_t + \text{a small normal}$$

### Pseudocode

::: pseudocode
Parameters: the noise's color (white, Ornstein–Uhlenbeck, pink or red), its spread $\knob\sigma$, the longest episode $T$; the rest is TD3's
Initialize TD3: the actor, twin critics, their target copies; an empty memory $\mathcal D$
Repeat for each episode:
  Draw its noise $\varepsilon_1, \dots, \varepsilon_T$: white, independent normals; pink or red, a normal per frequency $f$ times $f^{-\knob\beta/2}$, transformed back to time and scaled to spread 1; Ornstein–Uhlenbeck, from $\varepsilon_0 = 0$, $\varepsilon_{t+1} = 0.9985\,\varepsilon_t + 0.055\,\xi_t$ with $\xi_t \sim \mathcal N(0, 1)$
  Repeat for each step $t$ of the episode:
    $A \leftarrow \operatorname{clip}(\pol{\mu(S, \boldsymbol\theta)} + \knob\sigma\,\varepsilon_t)$ (a random action during the first steps); take it, observe $\rew R$, $S'$
    Store $(S, A, \rew R, S')$ in $\mathcal D$
    TD3's update from a batch of $\mathcal D$: the critics toward the smaller of the two targets; every $d$ steps, the actor and the copies
    $S \leftarrow S'$
:::

### Pitfalls

- Turning up white noise when exploration fails: on a body with inertia, more white noise is mostly more clipping.
- Comparing colors at different spreads: compare them at the same variance.
- Forgetting that Ornstein–Uhlenbeck noise restarts from 0 each episode and takes hundreds of steps to grow, with the usual $\Delta t = 0.01$.
- Leaving the noise on in the test episodes, or judging the policy by the noisy training episodes.
- Correlated noise in an on-policy method's samples: its stored probabilities no longer describe the actions taken.

### Check yourself {#check}

::: question
Why does white noise of spread 1 rarely swing the car far, when a constant push of 1 for 36 steps would?
---
The car integrates the force. Independent pushes add up like a random walk: 36 of them sum to about 6 in typical size, with random signs, against 36 for pushes that keep their direction. The car's swing needs pushes that last half a swing, about 36 steps.
:::

::: question
Pink and red noise both keep their sign longer than white noise. Why can red noise be worse?
---
Red noise puts almost all its variance into slow drifts that push one way for most of an episode. It travels far in one direction but rarely alternates at the pace the task needs, and it tries actions near the policy's own much less. Pink noise keeps some power at every time scale.
:::

::: question
Why can DDPG use any color of noise, while PPO cannot simply add colored noise to its samples?
---
DDPG learns off-policy: the noise only decides what the memory contains. PPO's update weighs each action by its probability under the policy that took it; with noise correlated in time those probabilities are wrong, unless the correlation is built into the policy, as gSDE does.
:::
