+++
summary = "Nothing in the return asks for smooth actions, and trained policies often jitter or switch between the ends of the action range: free in a simulator, costly on a motor. CAPS adds two terms to the actor's loss, one pulling the next state's action toward this one, one pulling nearby states' actions together; on a real drone it cut the power drawn by almost 80%. Robot learners also penalize the change of action in the reward, filter the actions, bound how fast the policy may change with the state, or act less often."
prereqs = ["td3", "action-noise"]
lab = "smooth-pendulum"
sources = [
  { text = "Mysore, Mabsout, Mancuso & Saenko (2021), Regularizing action policies for smooth control with reinforcement learning, ICRA", url = "https://arxiv.org/abs/2012.06644" },
  { text = "Raffin, Kober & Stulp (2021), Smooth exploration for robotic reinforcement learning, CoRL", url = "https://arxiv.org/abs/2005.05719" },
  { text = "Hwangbo, Lee, Dosovitskiy, Bellicoso, Tsounis, Koltun & Hutter (2019), Learning agile and dynamic motor skills for legged robots, Science Robotics 4", url = "https://arxiv.org/abs/1901.08652" },
  { text = "Rudin, Hoeller, Reist & Hutter (2022), Learning to walk in minutes using massively parallel deep reinforcement learning, CoRL", url = "https://arxiv.org/abs/2109.11978" },
  { text = "Peng, Coumans, Zhang, Lee, Tan & Levine (2020), Learning agile robotic locomotion skills by imitating animals, RSS", url = "https://arxiv.org/abs/2004.00784" },
  { text = "Seyde et al. (2021), Is bang-bang control all you need? Solving continuous control with Bernoulli policies, NeurIPS", url = "https://arxiv.org/abs/2111.02552" },
  { text = "Miyato, Kataoka, Koyama & Yoshida (2018), Spectral normalization for generative adversarial networks, ICLR", url = "https://arxiv.org/abs/1802.05957" },
  { text = "Chen, He, Wang, Liao, Ze, Li, Sastry, Wu, Sreenath, Gupta & Peng (2024), Learning smooth humanoid locomotion through Lipschitz-constrained policies", url = "https://arxiv.org/abs/2410.11825" },
  { text = "Smith, Kostrikov & Levine (2022), A walk in the park: learning to walk in 20 minutes with model-free reinforcement learning", url = "https://arxiv.org/abs/2208.07860" },
]

[story]
scene = "pendulum"
env = "pendulum"
formula = '''\step{1}{L = -\hat{\mathbb E}\big[\val{\hat q(s, \pol{\mu(s)})}\big] + \knob{\lambda_T}\,\hat{\mathbb E}\,\big\|\pol{\mu(s)} - \pol{\mu(s')}\big\| \qquad} \step{2}{+\ \knob{\lambda_S}\,\hat{\mathbb E}\,\big\|\pol{\mu(s)} - \pol{\mu(\bar s)}\big\|, \quad \bar s \sim \mathcal N(s, \knob{\sigma}^2)}'''

[story.runs]
plain = { recording = "td3-pendulum-jitter", name = "TD3" }
caps = { recording = "td3-pendulum-caps", name = "TD3 with CAPS" }
+++

## Story

::: step {run = "plain", at = 9, map = "action"}
**A pendulum held upright.** After 13,500 steps of training, TD3's test episode swings the pendulum up by step 23 and holds it there with a steady torque. Its **jitter**, how much the torque jumps from one step to the next, is 0.1: 0 would be a constant torque, 100 a jump across the whole range at every step.
:::

::: step {run = "plain", at = 11, map = "action"}
**3,000 steps later: the same return, another controller.** At the top the torque now flips between −1.9 and +1.7 at every step, 99 flips in the last 100 steps: jitter 71. The return cannot tell, −115.7 both times. Pendulum charges only 0.001 × torque² per step, and flips that fast cancel before the pendulum can move. Nothing in the reward asks for calm, and on a motor this is heat and wear.
:::

::: step {run = "caps", at = 13, map = "action", formula = 1}
**CAPS adds a term to the actor's loss** (formula 1): the distance between the torque in a state and the torque in the next state of the same transition, with weight $\knob{\lambda_T} = 0.1$. Same seed, after 19,500 steps: the pendulum held with a steady torque of −0.6, jitter 0.2.
:::

::: step {run = "caps", at = 40, map = "action", formula = 2}
**And a second** (formula 2): the distance between the torque in a state and in a copy of it nudged by noise of spread 0.05, with weight $\knob{\lambda_S} = 0.3$: nearby states, nearby torques. After 60,000 steps the jitter is 0.6, and the return −119, against −118 for plain TD3, still chattering at 55.
:::

::: step {run = "caps", at = 40, curves = ["plain", "caps"], metric = "jitter"}
**Twenty seeds of each**, the test episodes' jitter averaged over the seeds. Plain TD3's climbs once the pendulum is held, from below 1 to 16 on average over the second half, as 8 of its 20 seeds come to chatter; with CAPS it stays near 1.
:::

::: step {run = "caps", at = 40, curves = ["plain", "caps"], metric = "test"}
**At almost no cost in return.** Every seed of both ends up swinging the pendulum up and holding it; with CAPS it takes a little longer, 9,000 steps for half the seeds against 7,500. Stronger weights can freeze the actor at the start: in the Lab's sweep, a temporal weight of 0.3 froze 2 seeds of 10, their torque stuck at one end of the range. [Race them in the Lab](lab:smooth-pendulum).
:::

## Textbook

### Why trained policies jitter {#why}

The return counts what the actions achieve, not how they vary, and a learner takes whatever action the critic rates highest in each state. Where the critic's slope changes sign between nearby states, the action jumps from one end of the range to the other. Several things push that way:

- **Optimal control often is bang-bang.** For many tasks with bounded actions, the fastest way to a goal is full force one way, then full force the other. Trained Gaussian policies drift there: a Bernoulli policy that only ever picks the two ends of each action matched state-of-the-art results on several benchmarks (Seyde et al., 2021; [[policy-spread]]).
- **Acting faster than the body moves.** Mysore et al. (2021) observed that networks allowed to act much faster than the system's dynamics are more prone to learn jittery control: a jump that the body smooths out costs the return nothing.
- **Nothing penalizes it.** The exploration noise was white, so the critic has seen jumpy actions everywhere ([[action-noise]]), and the actor has no reason to prefer a calm one.

In a simulator this is free. On a motor, high-frequency switching means heat, wear, a power bill and, through a gearbox or a tendon, damage ([[gsde]] tells of a robot whose tendons per-step noise wore down). Even a simple target-tracking task makes it plain: TD3 learned an aggressive, nearly on–off controller that oscillated around the target, where the ideal one is linear (Mysore et al., 2021).

### CAPS: smoothness in the actor's loss {#caps}

CAPS (Conditioning for Action Policy Smoothness, Mysore et al., 2021) adds two terms to the actor's loss and leaves the critic and the reward alone:
$$L(\boldsymbol\theta) = L_\pi(\boldsymbol\theta) + \knob{\lambda_T}\,\big\|\pol{\mu(s_t)} - \pol{\mu(s_{t+1})}\big\|_2 + \knob{\lambda_S}\,\big\|\pol{\mu(s_t)} - \pol{\mu(\bar s_t)}\big\|_2, \qquad \bar s_t \sim \mathcal N(s_t, \knob{\sigma}^2). \label{eq-caps}$$
- The **temporal** term pulls the action in the next state, taken from the same transition, toward this state's action: consecutive actions alike.
- The **spatial** term pulls the action in a slightly perturbed state toward this one: nearby states, nearby actions, which also makes the policy less sensitive to noisy sensors. $\knob\sigma$ is set to the noise expected in the measurements.

Both are Euclidean distances, averaged over the batch. For DDPG and TD3, $L_\pi = -\hat{\mathbb E}[\val{\hat q(s, \pol{\mu(s)})}]$; for PPO and SAC, their own actor losses, with the terms applied to the policy's mean. In effect the terms bound how fast the policy may change, in time and in state, around the states the agent visits.

On four Gym tasks with DDPG, TD3, SAC and PPO, every CAPS agent was smoother than its plain counterpart, at a small price in return where the smoother agent reached its goal more slowly. TD3 on Pendulum, for instance, went from a smoothness score of 43.9 to 5.92 (thousandths; lower is smoother), with an average return of −173 against −153. On a quadrotor drone, PPO with CAPS drew 4.9 A in flight against 22.9 A for the best previous neural controller, almost 80% less, and every agent it trained was safe to fly, where the earlier pipeline needed its controllers cherry-picked. Each term alone fell short: the temporal one left the motors noisy after the transfer to the real drone, the spatial one forced the commands into bands.

The trainer of this guide has CAPS for DDPG and TD3. On Pendulum, over 20 seeds, plain TD3's test episodes jitter by 16 on average in the second half of training, and on 8 of the seeds by more than 10: the torque comes to jump by a third of its range or more at every step. With CAPS ($\knob{\lambda_T} = 0.1$, $\knob{\lambda_S} = 0.3$, $\knob\sigma = 0.05$) the jitter averages 1, and the pendulum is held up nearly as well, near $-120$ per test episode either way, a little later: after 9,000 steps for half the seeds, against 7,500. Past the first 15,000 steps, 4 of CAPS's 600 test episodes fell below $-250$, against none of plain TD3's. The Lab's sweep, 10 seeds per weight, shows both sides of the weights. Each term alone already does most of the work: the test episodes' jitter averaged 1.8 with the spatial term alone and 2.6 with the temporal one alone, against 0.5 with both and 18 for plain TD3 on the same seeds. Stronger weights froze some seeds at the start, their torque stuck at one end of the range: a temporal weight of 0.3 froze 2 seeds of 10, a spatial one of 1 froze 1 and left another short of holding the pendulum.

::: example {#ex-caps} The two terms on one transition
A batch holds the transition $s = (\cos\theta, \sin\theta, \dot\theta) = (0.98, 0.17, -1.0)$ to $s' = (0.99, 0.12, -1.1)$. The actor gives torque 1.6 in $s$ and $-0.4$ in $s'$: it switches hard between consecutive steps. With $\knob{\lambda_T} = 1$ the temporal term adds $|1.6 - (-0.4)| = 2$ to the loss, and its gradient pushes the two torques toward each other, with a pull of $\knob{\lambda_T}$ on each, whatever the size of the gap. If the critic's slope favors the switch more strongly than that, the switch survives; if not, it fades.
:::

### Measuring smoothness {#measure}

Two measures are common:
- **The mean weighted frequency** of the action's spectrum, CAPS's $Sm = \frac{2}{n f_s}\sum_i M_i f_i$ over the spectrum's frequencies $f_i$ and amplitudes $M_i$: high when high frequencies carry much amplitude.
- **The mean squared change** of the action per step, gSDE's continuity cost $C = 100\,\mathbb E_t[((a_{t+1} - a_t)/\Delta a_{\max})^2]$: 0 for a constant action, 100 for a jump across the whole range every step ([[gsde]]). The Lab charts it as **jitter**.

Both depend on the task: compare them between agents on the same task, not across tasks.

### Other tools {#others}

- **Penalties in the reward.** Legged robots trained in simulation pay for the change of action between steps, for the torques and for the joint accelerations (Hwangbo et al., 2019; Rudin et al., 2022). It works, at the cost of many weights to tune, and it goes through the critic: the policy learns smoothness only as well as the critic learns its value ([[reward-design]]).
- **Low-pass filters** on the actions, before the motors: Peng et al. (2020) filtered the target joint angles of a quadruped imitating a dog. A filter must be in place during training too: a policy trained without it can overshoot, oscillate or lose control altogether when one is added afterward, because the filter changes the dynamics the policy learned (Mysore et al., 2021). And a filter adds memory that the observation must then carry, or the state is no longer Markov: training a quadruped in simulation, Smith et al. (2022) found that a low-pass filter on the joint targets slowed learning, for that reason.
- **Bounding the policy's slope.** Spectral normalization divides each layer's weights by their largest singular value, bounding how fast the network's output can change with its input (Miyato et al., 2018); a penalty on the policy's gradient with respect to its input does the same softly, and made humanoid walking policies smooth enough for real robots (Chen et al., 2024).
- **Acting less often.** Repeat each action for a few steps, or lower the control frequency: fewer decisions leave less room to jitter ([[action-noise]]).
- **Smooth exploration.** Correlated noise or gSDE makes the training actions smooth, which protects the robot while it learns; it does not make the learned policy smooth by itself ([[gsde]]).

### Historical remarks {#history}

Smoothness was a concern of control engineers long before reinforcement learning: filters, rate limits and penalties on control effort are standard in classical and optimal control. Deep reinforcement learning met the problem when its controllers left simulation: Koch et al.'s drone controllers (Neuroflight) oscillated enough to overheat motors, which led Mysore et al. (2021) to CAPS; legged-robot learners from ETH Zurich and Google built smoothness terms into their rewards and filters into their pipelines.

## Card

### Idea

Ask for smoothness directly: CAPS adds to the actor's loss the distance between the actions of consecutive states and between the actions of a state and a slightly perturbed copy. The policy stays nearly as good and stops jittering.

::: analogy
A driver who swerves between lanes reaches the exit as fast as a calm one, and wears out the tires, the passengers and the road. Telling the driver "keep the wheel steady" works better than fining every scream from the back seat.
:::

### The update {#update}

The policy's loss plus two penalties: one for changing the action between consecutive states, one for changing it between nearby states.

$$L = L_\pi + \knob{\lambda_T}\,\|\pol{\mu(s)} - \pol{\mu(s')}\| + \knob{\lambda_S}\,\|\pol{\mu(s)} - \pol{\mu(\bar s)}\|$$

The nearby state $\bar s$ is drawn around $s$.

$$\bar s \sim \mathcal N(s, \knob\sigma^2)$$

### Pseudocode

::: pseudocode
Parameters: the weights $\knob{\lambda_T}$ and $\knob{\lambda_S}$, the perturbation $\knob\sigma$; the rest is TD3's
Initialize TD3: the actor, twin critics, their target copies; an empty memory $\mathcal D$
Repeat for each step:
  Act with $\pol{\mu(S)}$ plus exploration noise; observe $\rew R$, $S'$; store $(S, A, \rew R, S')$ in $\mathcal D$
  Draw $B$ transitions; the critics' update, as in TD3
  Every $d$ steps, the actor: for each transition, $\bar s \leftarrow s + \knob\sigma\,\boldsymbol\xi$ with $\boldsymbol\xi \sim \mathcal N(\mathbf 0, \mathbf I)$
    A gradient step down $-\frac1B\sum \val{\hat q(s, \pol{\mu(s)})} + \frac{\knob{\lambda_T}}{B}\sum \|\pol{\mu(s)} - \pol{\mu(s')}\| + \frac{\knob{\lambda_S}}{B}\sum \|\pol{\mu(s)} - \pol{\mu(\bar s)}\|$
    The target copies follow
:::

### Pitfalls

- Weights so large that the actor ignores the critic: the policy becomes smooth and useless. With a $\tanh$ output it can even freeze at the start: the untrained critic pushes every action the same way, the terms hold the actions together, and once all of them sit at the flat ends of $\tanh$, no gradient brings them back.
- A spatial perturbation larger than the states' meaningful differences: nearby states that need different actions get forced together.
- Adding a filter after training: the policy was not trained with its delay.
- Judging smoothness by the training episodes, whose noise dominates: measure the test episodes.

### Check yourself {#check}

::: question
A penalty on the change of action, added to the reward, also asks for smoothness. Why does CAPS get it more directly?
---
The penalty reaches the policy only through the critic, which must learn it along with everything else, and its weight competes with every other term of the reward. CAPS acts on the actor's outputs directly, where the smoothness is.
:::

::: question
What does each of CAPS's terms protect against?
---
The temporal term against jumps between consecutive actions, the jitter itself. The spatial term against a policy that reacts strongly to tiny differences in state, which makes it fragile to sensor noise and to the gap between simulation and the real system.
:::
