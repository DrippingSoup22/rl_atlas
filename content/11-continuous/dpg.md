+++
summary = "With continuous actions, a policy can name one action per state, a = μ(s), instead of a distribution over all of them. Its gradient is then the critic's slope with respect to the action, passed back through the actor: move the action the way the critic says the value rises. No sum over actions, no likelihood ratios, and it can learn off-policy; exploration has to be added from outside."
prereqs = ["pg-theorem", "actor-critic", "policy-parameterization"]
lab = "ddpg-pendulum"
sources = [
  { text = "Silver, Lever, Heess, Degris, Wierstra & Riedmiller (2014), Deterministic policy gradient algorithms, ICML", url = "https://proceedings.mlr.press/v32/silver14.html" },
  { text = "Heess, Wayne, Silver, Lillicrap, Erez & Tassa (2015), Learning continuous control policies by stochastic value gradients, Advances in Neural Information Processing Systems 28", url = "https://arxiv.org/abs/1510.09142" },
  { text = "Degris, White & Sutton (2012), Off-policy actor-critic, ICML", url = "https://arxiv.org/abs/1205.4839" },
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., section 13.7", url = "http://incompleteideas.net/book/the-book-2nd.html" },
]

[story]
scene = "throw"
env = "throw"
seed = 1
average = 100
formula = '\step{1}{\mathbf w \leftarrow \mathbf w + \alp_{\mathbf w}\big[\rew{R} - \val{\hat q(A, \mathbf w)}\big]\,\nabla_{\mathbf w}\val{\hat q(A, \mathbf w)} \qquad} \step{2}{\pol{\mu} \leftarrow \pol{\mu} + \alp\,\frac{\partial \val{\hat q(a, \mathbf w)}}{\partial a}\Big|_{a = \pol{\mu}}}'

[story.runs]
learn = { algorithm = "dpg", alpha = 1.0, alphaW = 0.2, noise = 10.0, gamma = 1.0, units = 300, measures = ["aim"], name = "noise σ = 10°" }
narrow = { algorithm = "dpg", alpha = 1.0, alphaW = 0.2, noise = 2.0, gamma = 1.0, units = 300, measures = ["aim"], name = "noise σ = 2°" }
still = { algorithm = "dpg", alpha = 1.0, alphaW = 0.2, noise = 0.0, gamma = 1.0, units = 300, measures = ["aim"], name = "no noise" }
+++

## Story

::: step {run = "learn", at = 0}
**An aim, not a bell.** The throw of [[policy-parameterization]]: one decision per episode, the angle, and the ball flies $40 \sin 2a$ meters, give or take a couple of meters of wind; 40 m at best, at 45°. There, the policy was a bell over angles, and it learned from how each random draw turned out. A **deterministic** policy is just its aim, $\pol{\mu} = 20°$ here: one angle, nothing drawn. On its own it would throw at 20° forever. To learn which way is better, it needs a critic.
:::

::: step {run = "learn", at = 0, play = 6, pace = 300, formula = 1}
**Noise to explore, a critic to learn.** Each throw is the aim plus noise, a random 10° or so either way (the dashed bell). The throws come from a behavior that is not the policy being learned, so DPG learns off-policy. After each throw, the critic $\val{\hat q(a)}$, a curve over all angles, moves toward how far that throw flew, formula (1). The first throw, at 34.0°, flies 39.5 m.
:::

::: step {run = "learn", at = 10, recent = 10}
**After 10 throws**, all between 10° and 34°, the critic has a shape where they landed: it rises to the right, from 7 m at 0° to 31 m at the aim. Beyond them it knows little: it says 20 m at 45°, where a throw flies 40, and nearly 0 above 70°. It does not need to be right there. The actor will only ask it about the angle it aims at.
:::

::: step {run = "learn", at = 10, recent = 10, formula = 2}
**The actor climbs the critic's slope**, formula (2): it moves the aim by the slope at the aim. At 26.2°, the critic says each degree higher adds 0.43 m, so the aim moves up 0.43°. That is the deterministic policy gradient: no probabilities and no log-likelihoods, only which way is up at the aim, and how steeply.
:::

::: step {run = "learn", checkpoints = [25, 50, 100], recent = 15}
As the aim climbs, the throws follow it, and the critic learns the curve where they land: the aim is at 34.9° after 25 throws, 40.9° after 50, and 44.8° after 100.
:::

::: step {run = "learn", at = 300, recent = 20}
**At the top the slope is flat**, and the aim stays near 45°: 45.3° after 300 throws, nudged a degree or two either way as the wind jostles the critic. The critic is right where it was asked, 40.3 m at 45°, and rough where it was not: 5.8 m at 0°, where a throw goes nowhere. Of 100 runs, every one ends between 43° and 49°.
:::

::: step {run = "still", at = 50, recent = 20}
**Without noise**, every throw lands at 20°. The critic learns a bump around 20°, as high as the throws flew, with its top at the aim: the slope there is flat, and the aim never moves. A deterministic actor depends on exploration it does not do itself, and [[ddpg]] adds noise to its torques for the same reason.
:::

::: step {run = "learn", at = 300, recent = 20, curves = ["learn", "narrow", "still"], metric = "aim"}
**The aim, averaged over 100 runs**, for three amounts of noise. With 10°, the critic sees a wide stretch of the curve, and the aim passes 44° after 100 throws. With 2°, the throws land so close together that the wind hides the slope between them: 31° after 300 throws. Without noise, nothing moves. [Throw in the Lab](lab:dpg-throw).
:::

## Textbook

### Continuous actions, again {#again}

Value-based methods choose and evaluate actions through a max over them ([[q-learning]]). With a torque, a steering angle or the joints of a robot arm, the actions form a continuum, and that max becomes an optimization problem at every step. Policy gradients avoid it with a parameterized distribution, often a Gaussian whose mean and spread the network outputs ([[policy-parameterization]]). They then estimate the gradient from sampled actions, and that estimate is noisy: the narrower the Gaussian becomes, the larger the variance of $\nabla \ln \pol\pi$.

There is a more direct route. Let the policy output the action itself, $a = \pol{\mu(s, \boldsymbol\theta)}$, and ask a critic $\val{\hat q(s, a)}$ how the value changes when that action is nudged.

### The deterministic policy gradient {#theorem}

For a deterministic policy, the performance gradient is (Silver et al., 2014):
$$\nabla_{\boldsymbol\theta} J(\boldsymbol\theta) = \mathbb E_{s \sim \rho^{\mu}}\Big[\nabla_{\boldsymbol\theta}\,\pol{\mu(s, \boldsymbol\theta)}\;\nabla_a \val{q_\mu(s, a)}\big|_{a = \pol{\mu(s)}}\Big], \label{dpg}$$
where $\rho^\mu$ is the discounted distribution of states the policy visits. Read from right to left, this is the chain rule. $\nabla_a \val{q_\mu}$ says in which direction a change of action raises the value, and how steeply; $\nabla_{\boldsymbol\theta}\pol\mu$ says how the parameters move the action. The update moves the parameters so that the action slides uphill on the critic.

Compare the stochastic policy gradient theorem ([[pg-theorem]]): there the expectation runs over states *and actions*, and each action's log-probability is pushed up or down by its value. Here the action integral is gone. Silver et al. also showed that the stochastic gradient of a Gaussian policy tends to this one as its spread shrinks to zero, so the deterministic gradient is the limit case, not a different idea.

### Off-policy for free {#off}

Because no expectation over the policy's actions remains, the gradient can be estimated from states visited by *another* policy, without importance sampling over actions. A behavior policy can explore with noise while the deterministic actor learns from what it saw, using the gradient \ref{dpg} with $s$ drawn from the behavior's state distribution instead of $\rho^\mu$. This is an approximation, made in the same spirit as the off-policy actor–critic of Degris, White and Sutton (2012), and it works well in practice. The critic learns $\val{q_\mu}$ off-policy as well, with Q-learning-style targets whose next action comes from the actor: $\rew{r} + \gam\,\val{\hat q(s', \pol{\mu(s')})}$.

That combination, a deterministic actor, a critic learned off-policy and a replay memory, is [[ddpg]].

### What the critic must get right {#critic}

The actor follows the critic's *slope*, $\nabla_a \val{\hat q}$, but the critic is trained only to match *values*. Nothing makes its slopes right, and where it errs, the actor moves the wrong way with confidence. If the critic overestimates some actions, the actor is drawn to exactly those: it climbs the errors. The fixes of [[td3]] (twin critics, delayed actor steps, smoothed targets) all target this weakness.

Silver et al. also gave a form of critic whose slopes are guaranteed to give the right gradient, a *compatible* critic linear in $\nabla_{\boldsymbol\theta}\pol\mu$, mirroring the stochastic case ([[actor-critic]]). Deep RL methods use general networks instead and accept the bias.

### Exploration from outside {#exploration}

A deterministic policy does the same thing in the same state every time, so it never explores by itself. The behavior policy adds noise to its action: independent Gaussian noise, or noise correlated over time (an Ornstein–Uhlenbeck process) so that a push lasts long enough to matter in a physical system. The amount of noise is a knob, set by hand ([[exploration-strategies]]). [[sac]] goes the other way: it keeps a stochastic policy and makes its randomness part of the objective.

The two routes meet in one trick. Write a stochastic action as a deterministic function of the state and an independent noise, $a = \pol{\mu(s)} + \pol{\sigma(s)}\,\varepsilon$ with $\varepsilon \sim \mathcal N(0, 1)$, and the chain rule of \ref{dpg} applies to each sampled action, with the noise held fixed. This **reparameterization** gives stochastic policies a gradient through the critic's slope instead of through $\nabla \ln \pol\pi$, usually with much lower variance (Heess et al., 2015). SAC's actor learns this way.

### Historical remarks {#history}

Silver et al. (2014) proved the deterministic policy gradient theorem, its limit relation to the stochastic one, and compatible critics for it, and showed that off-policy deterministic actor–critics outperform their stochastic counterparts on problems with many action dimensions. Lillicrap et al. (2016) combined it with DQN's replay memory and target networks to learn from pixels and joint angles: DDPG.

## Card

### Idea

A deterministic policy outputs one action per state, $a = \pol{\mu(s)}$. Its gradient follows the critic's slope in action space: nudge the action the way $\val{\hat q}$ says the value rises, and carry that through the actor's parameters.

::: analogy
Adjusting a shower knob by feel. You do not try random positions and keep the good ones; you feel which way warmer lies and turn a little that way.
:::

### The gradient {#gradient}

$$\nabla_{\boldsymbol\theta} J = \mathbb E_s\Big[\nabla_{\boldsymbol\theta}\pol{\mu(s)}\;\nabla_a \val{\hat q(s, a)}\big|_{a = \pol{\mu(s)}}\Big]$$

the chain rule through the critic: how the action moves the value, times how the parameters move the action.

### Pseudocode

::: pseudocode
Parameters: step sizes $\alp$ (actor) and $\alp_{\mathbf w}$ (critic), discount $\gam$, exploration noise; an actor $\pol{\mu(s, \boldsymbol\theta)}$ and a critic $\val{\hat q(s, a, \mathbf w)}$
Repeat for each episode:
  Start: $S$ {#start}
  Repeat until $S$ is terminal:
    Choose $A = \pol{\mu(S, \boldsymbol\theta)}$ plus noise {#choose}
    Take $A$, observe $\rew{R}$ and $S'$ {#act}
    $\del \leftarrow \rew{R} + \gam\,\val{\hat q(S', \pol{\mu(S')}, \mathbf w)} - \val{\hat q(S, A, \mathbf w)}$ (no $\val{\hat q(S', \cdot)}$ term at the end); $\ \mathbf w \leftarrow \mathbf w + \alp_{\mathbf w}\,\del\,\nabla_{\mathbf w} \val{\hat q(S, A, \mathbf w)}$ {#critic}
    $\boldsymbol\theta \leftarrow \boldsymbol\theta + \alp\,\nabla_{\boldsymbol\theta}\pol{\mu(S, \boldsymbol\theta)}\,\nabla_a \val{\hat q(S, a, \mathbf w)}\big|_{a = \pol{\mu(S)}}$ {#update}
    $S \leftarrow S'$ {#next}
:::

### Why it matters {#why}

- No expectation over actions: lower variance than a sampled policy gradient, and cheap with many action dimensions.
- Off-policy: states from any exploring behavior will do, so a replay memory works.
- The limit of a Gaussian policy gradient as its spread goes to zero.

### Pitfalls

- Trusting the critic's slope: it was trained on values, and the actor climbs its errors.
- Forgetting exploration: the actor alone never tries anything new.
- Expecting a deterministic policy to bluff or randomize when the best play is random, as in games.

### Check yourself {#check}

::: question
Why does the deterministic policy gradient not need importance sampling to learn off-policy?
---
It has no expectation over the policy's actions, only over states. The critic is queried at the actor's own action, μ(s), whatever action the behavior took there, so there is no action probability to correct for. (Using the behavior's states instead of the policy's is an approximation, but a mild one.)
:::

::: question
The critic overestimates the value of strong torques in some states. What does the actor do there?
---
It moves toward them: the actor follows the critic's slope, so wherever the critic wrongly rises, the actor follows. That is why critic errors matter more for deterministic actor–critics, and why TD3 makes the critic pessimistic.
:::

::: question
How is the deterministic policy gradient related to the Gaussian policy gradient of REINFORCE?
---
It is its limit: as the Gaussian's spread shrinks to zero, the stochastic policy gradient tends to the deterministic one (Silver et al., 2014). The sampled estimate becomes noisier as the spread shrinks, while the deterministic form computes the limit directly from the critic's slope.
:::
