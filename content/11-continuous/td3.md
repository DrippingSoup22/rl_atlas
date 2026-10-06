+++
summary = "DDPG with three fixes against its critic's overestimation: twin critics, with targets from the smaller of the two; an actor that moves only every second critic step; and target actions blurred with clipped noise. On Pendulum its average target over 20 seeds never rises above what the world can pay, where DDPG's does in 29 of 40 blocks; the caution costs a little speed."
change = "Use the smaller of two critics' estimates in the target (clipped double Q), update the actor and the target copies only every d critic steps, and add clipped noise to the target action."
prereqs = ["ddpg", "double-q"]
lab = "td3-pendulum"
sources = [
  { text = "Fujimoto, van Hoof & Meger (2018), Addressing function approximation error in actor-critic methods, ICML", url = "https://arxiv.org/abs/1802.09477" },
  { text = "van Hasselt, Guez & Silver (2016), Deep reinforcement learning with double Q-learning, AAAI", url = "https://arxiv.org/abs/1509.06461" },
  { text = "Lillicrap et al. (2016), Continuous control with deep reinforcement learning, ICLR", url = "https://arxiv.org/abs/1509.02971" },
]

[story]
scene = "pendulum"
env = "pendulum"
formula = '''\step{1}{y = \rew{r} + \gam \min_{i = 1, 2} \val{\hat q'_i\big(s', \pol{\mu'(s')} + \epsilon\big)} \qquad} \step{2}{\text{actor and copies: once every } d = 2 \text{ critic steps}}'''

[story.runs]
ddpg = { recording = "ddpg-pendulum", name = "DDPG" }
td3 = { recording = "td3-pendulum", name = "TD3" }
+++

## Story

::: step {run = "ddpg", at = 40, map = "value"}
**Overestimation, measured.** On Pendulum every reward is 0 or less, so every true value is too. DDPG's critic after 60,000 steps rates some states at **+30**. Along the greedy test episodes of blocks 11 to 40, it rates the states it passes, on average, 52 above the discounted return that actually followed. The actor follows the critic's slope, so it goes where these errors are largest ([[ddpg]]).
:::

::: step {run = "td3", at = 0, formula = 1}
**TD3 changes the target.** It trains two critics on the same batches. The target, formula (1), takes the *smaller* of their two estimates, so a value one critic got too high is checked by the other. And it evaluates the target actor's torque with a little clipped noise $\epsilon$ added, so the target cannot rest on a narrow, lucky peak of the critic. Everything else, the memory, the networks and the step sizes, is DDPG's.
:::

::: step {run = "td3", at = 40, map = "value"}
**TD3's critic after 60,000 steps:** from −316 to −16. Nothing above 0. Along its test episodes it errs the other way, rating the states 20 *below* the return that followed. The smaller of two estimates is a cautious estimate.
:::

::: step {run = "td3", at = 40, curves = ["ddpg", "td3"], metric = "q"}
**The critics' targets, averaged over 20 seeds each.** DDPG's average rises above 0 at block 12 and stays there to the end, 29 of the 40 blocks; in its last 10 blocks, 11 of the 20 seeds average above 0. TD3's average never comes near: −52 at the end, and no seed of 20 averages above 0 over its last 10 blocks.
:::

::: step {run = "td3", at = 5, map = "action", formula = 2}
**The actor waits for the critic.** TD3 moves the actor and the target copies only once every two critic steps, formula (2), so the critic's errors shrink before the actor follows them. Caution costs some speed: this run first swings up and holds in the test after 7,500 steps, at −128; DDPG's did it after 4,500. Averaged over 20 seeds, after 5 blocks DDPG's tests are at −145 and TD3's at −274.
:::

::: step {run = "td3", at = 40, curves = ["ddpg", "td3"], metric = "test"}
**Both learn, on every seed.** By block 8 TD3's average test catches up, and at the end both hold the pendulum: the last four tests of every seed average between −121 and −132 for DDPG and between −122 and −139 for TD3. On Pendulum, DDPG's overestimation does no visible harm. On harder tasks, such as the walking robots of the original paper, Fujimoto et al. found it does, and TD3 learned better there. [Race them in the Lab](lab:td3-pendulum).
:::

## Textbook

### Overestimation in actor–critics {#over}

Q-learning overestimates because its target maximizes over noisy estimates ([[double-q]]). DDPG has no explicit max, but its actor does the maximizing: it is trained to climb $\val{\hat q}$, so it settles where the critic is highest, which is often where the critic is most wrong. The target then evaluates the target actor's choice with the target critic, carrying the error into the next targets, and the bias accumulates over updates. Fujimoto, van Hoof and Meger (2018) measured it on DDPG and on the robot locomotion tasks of the time, and proposed three changes. The algorithm that combines them is TD3, "twin delayed DDPG".

### Clipped double Q-learning {#clipped}

Double DQN removes the bias by letting the online network choose the next action and the target network value it ([[dqn-extensions]]). In an actor–critic, the policy changes slowly, so the online and target critics stay too similar for that split to help. TD3 instead trains two critics $\val{\hat q_1}$, $\val{\hat q_2}$, each with its own target copy, on the same batches, and both regress to one shared target:
$$y = \rew{r} + \gam \min_{i = 1, 2} \val{\hat q_i\big(s', \tilde a, \mathbf w_i^-\big)}, \qquad \tilde a = \pol{\mu(s', \boldsymbol\theta^-)} + \epsilon. \label{target}$$
The min can underestimate, and that is deliberate: an underestimated action is simply not chosen, while an overestimated one attracts the actor and spreads.

### Delayed policy updates {#delayed}

A critic that has just changed has fresh errors. TD3 updates the actor, and moves the target copies, only once every $d$ critic updates ($d = 2$), so the actor follows a critic that has had time to settle. The actor still climbs the first critic, $\nabla_a \val{\hat q_1(s, a)}$ at $a = \pol{\mu(s)}$, as in DDPG.

### Target policy smoothing {#smoothing}

Similar actions should have similar values. TD3 evaluates the target at a blurred action:
$$\epsilon = \operatorname{clip}\big(\mathcal N(0, \tilde\sigma^2), -c, c\big), \label{noise}$$
with $\tilde\sigma = 0.2$ and $c = 0.5$ in units of the largest action. The target becomes an average over a small neighborhood of actions, so a narrow, spurious peak of the critic cannot pull the targets up.

### On Pendulum {#pendulum}

The recording uses DDPG's settings exactly, plus the three changes. Over 20 seeds:

- **Values.** No seed's average target over its last 10 blocks is above 0; DDPG's is above 0 in 11 seeds of 20.
- **Speed.** Half of TD3's seeds have a block of training episodes averaging −250 or better within 9,000 steps; half of DDPG's, within 6,000. The first test at −250 or better comes at block 5 in the median, against DDPG's block 3.
- **The end.** Every seed of both holds the pendulum up in its last tests.

Pendulum is small and forgiving, so DDPG's bias costs it nothing visible here. The point of TD3 is robustness on problems where it does: on the locomotion tasks of the original paper, TD3 outperformed DDPG and the other methods of its time.

### Historical remarks {#history}

Fujimoto, van Hoof and Meger (2018) introduced TD3 with a careful study of function approximation error in actor–critics, showing overestimation in DDPG and the variance that a fast-moving target adds. TD3 and [[sac]], published the same year, became the standard off-policy methods for continuous control.

## Card

### Idea

DDPG, made cautious: two critics and the smaller of their estimates in the target, an actor that updates half as often as the critics, and target actions blurred with clipped noise.

::: analogy
Two estimators bid on a job, and you take the lower bid; and before acting on an estimate, you wait for it to be checked once more.
:::

### The update {#update}

$$y = \rew{r} + \gam \min_{i = 1, 2} \val{\hat q'_i\big(s', \pol{\mu'(s')} + \epsilon\big)}, \qquad \epsilon = \operatorname{clip}(\mathcal N(0, \tilde\sigma^2), -c, c)$$

both critics regress on $y$; every $d$ steps the actor climbs $\val{\hat q_1(s, \pol{\mu(s)})}$ and the target copies move by $\tau$.

### One change from DDPG {#change}

Three changes against overestimation: the min of twin critics in the target, delayed actor and target updates, and noise on the target action.

### Backup diagram {#backup}

{{backup td3}}

One sampled transition, then the target actor's action with a little noise, valued by the smaller of the two target critics.

### Pseudocode

::: pseudocode
Parameters: memory $N$, batch $B$, step sizes, $\tau$, exploration noise $\sigma$, target noise $\tilde\sigma$ clipped at $c$, delay $d$
Initialize actor $\boldsymbol\theta$ and critics $\mathbf w_1$, $\mathbf w_2$ at random, and their target copies; memory $\mathcal D$ empty
Repeat for each step $t$:
  $A \leftarrow \pol{\mu(S, \boldsymbol\theta)} + $ noise of spread $\sigma$, clipped; take it, observe $\rew{R}$, $S'$; store it in $\mathcal D$
  Sample $B$ transitions; $\tilde a_j \leftarrow \pol{\mu(s'_j, \boldsymbol\theta^-)} + \operatorname{clip}(\mathcal N(0, \tilde\sigma^2), -c, c)$, clipped to the allowed range
  $y_j \leftarrow \rew{r_j} + \gam \min_i \val{\hat q(s'_j, \tilde a_j, \mathbf w_i^-)}$ (just $\rew{r_j}$ at the end)
  Each critic $i$: a gradient step on $\frac1B \sum_j (y_j - \val{\hat q(s_j, a_j, \mathbf w_i)})^2$
  Every $d$ steps:
    Actor: a gradient step that raises $\frac1B \sum_j \val{\hat q(s_j, \pol{\mu(s_j, \boldsymbol\theta)}, \mathbf w_1)}$
    All target copies move by $\tau$ toward their networks
  $S \leftarrow S'$
:::

### Perks

- Removes the overestimation that DDPG's actor feeds on: on Pendulum, no seed's average target ends above 0, against 11 of 20 for DDPG.
- More robust than DDPG across tasks and seeds, at the same cost per step.
- Simple: three small changes, each with its own reason.

### Flaws

- Can underestimate: on Pendulum, 20 below the return that followed, along the shown run's tests.
- Slower when overestimation was harmless: on Pendulum, a median of 9,000 steps to good training episodes, against DDPG's 6,000.
- Still a deterministic actor with hand-set exploration noise.

### Knobs

| Knob | Too low | Too high |
| --- | --- | --- |
| delay $d$ | the actor chases fresh critic errors, as in DDPG | the actor learns too rarely, slowly |
| target noise $\tilde\sigma$ | sharp critic peaks pull the targets up | targets average over very different actions |
| noise clip $c$ | little smoothing | rare large noise blurs the targets |
| and DDPG's | $\tau$, exploration noise, memory, step sizes | |

### Pitfalls

- Using the min of the two critics for the actor's update too: the actor climbs one critic; the min belongs in the target.
- Forgetting to clip the target action back into the allowed range after adding noise.
- Reading TD3's lower values as worse performance: they are more honest estimates, judge by returns.

### Check yourself {#check}

::: question
Why does DDPG overestimate, when it has no max in its target?
---
Its actor does the maximizing: trained to climb the critic, it settles where the critic is highest, often where it is most wrong. The target then values the target actor's choice, carrying the error forward, and the errors accumulate.
:::

::: question
Why does TD3 take the min of two critics rather than Double DQN's split between online and target networks?
---
In an actor–critic the policy changes slowly, so the online and target critics are too alike for the split to decorrelate their errors. Two independently trained critics disagree more, and the min of them leans the target toward underestimation, which the actor does not exploit.
:::

::: question
On Pendulum DDPG and TD3 both succeed on all 20 seeds. Was TD3's caution wasted?
---
On this task, mostly: it cost speed (9,000 steps for half the seeds instead of 6,000) for no gain in the end. But its values are honest, and on harder tasks where overestimation misleads the actor, the same changes make the difference. A method is chosen for the problems it will meet, not one example.
:::
