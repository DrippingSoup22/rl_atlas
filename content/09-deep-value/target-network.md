+++
summary = "Compute the targets with a copy of the network that changes only every few hundred steps, so that between copies learning is a regression toward fixed targets instead of a chase after targets that move with every update. Copy too often and the targets move again; too rarely and the values crawl."
prereqs = ["neural-networks", "semi-gradient-td", "value-iteration"]
lab = "dqn-target"
sources = [
  { text = "Mnih et al. (2015), Human-level control through deep reinforcement learning, Nature 518", url = "https://doi.org/10.1038/nature14236" },
  { text = "Riedmiller (2005), Neural fitted Q iteration: first experiences with a data efficient neural reinforcement learning method, ECML", url = "https://doi.org/10.1007/11564096_32" },
  { text = "Ernst, Geurts & Wehenkel (2005), Tree-based batch mode reinforcement learning, Journal of Machine Learning Research 6", url = "https://www.jmlr.org/papers/v6/ernst05a.html" },
  { text = "Lillicrap et al. (2016), Continuous control with deep reinforcement learning, ICLR", url = "https://arxiv.org/abs/1509.02971" },
  { text = "van Hasselt, Doron, Strub, Hessel, Sonnerat & Modayil (2018), Deep reinforcement learning and the deadly triad", url = "https://arxiv.org/abs/1812.02648" },
]

[story]
scene = "cartpole"
env = "cartpole"
formula = '''\step{1}{y = \rew{r} + \gam \max_{a'} \val{\hat q(s', a', \mathbf w)} \qquad} \step{2}{y = \rew{r} + \gam \max_{a'} \val{\hat q(s', a', \mathbf w^-)}}'''

[story.runs]
copy = { recording = "dqn-cartpole", name = "copied every 500 steps" }
none = { recording = "dqn-cartpole-no-target", name = "no target network" }
+++

## Story

::: step {run = "none", at = 0}
**The same DQN, one thing taken away.** Same pole, same seed, same network, memory and batches. The only change: the targets are computed with the weights being trained, not with a frozen copy. Before any learning, the values on the map run from −0.1 to 1.2.
:::

::: step {run = "none", at = 1, formula = 1}
**5,000 steps later, the values run from 161 to 257**, though no state here is worth more than 100. Each update raises $\val{\hat q(s, a)}$ toward its target; but $s'$, a fiftieth of a second later, looks almost the same to the network, so its values rise too, and with them the target. The network chases its own tail. The pole falls after 10 steps; with the copy, the same seed kept it up for 162.
:::

::: step {run = "none", at = 3}
**And yet, after 15,000 steps, it balances** for 414 steps, until the cart runs off the track, with values up to 234. A greedy policy needs only the right *order* of the two pushes in each state: two estimates that are both far too high can still say which push is better.
:::

::: step {run = "none", checkpoints = [4, 5], hold = 2200}
**It does not last.** The loop keeps feeding itself: the values reach 495, then 1,060. On numbers that large, the gap between the two pushes drowns in the errors, and which push wins changes from one update to the next. The test episodes last 125 and 20 steps.
:::

::: step {run = "none", at = 40, map = "action"}
**At the end**, the values run from $-75$ to $163$, though every step pays $\rew{+1}$ and no state is worth more than 100. The order of the pushes still looks sensible on the map, yet the cart runs off the track after 122 steps. Along the way the run kept the pole up for the full 500 steps six times, and lost it every time.
:::

::: step {run = "copy", at = 40, curves = ["copy", "none"], metric = "q", log = true, ref = [100, "the most a state is worth"]}
**Twenty seeds each**, on a log scale. Without the copy, every seed's values pass 100 within 15,000 steps, and peak between 3,329 and 107,559. With the copy, the values creep up from below and pass 100, if at all, only in the second half of training: each period moves them one backup further, never more.
:::

::: step {run = "copy", at = 40, formula = 2, curves = ["copy", "none"], metric = "return"}
**The copy holds the target still.** Between copies, the targets are fixed numbers and each period is a plain regression: the loop is cut. Without it, no run of 20 ends well; with a copy every 500 steps, 13 do. [The Lab sweeps the copy period](lab:dqn-target).
:::

## Textbook

### A target that moves {#why}

The semi-gradient update moves $\val{\hat q(S, A, \mathbf w)}$ toward $\rew{R} + \gam \max_{a'} \val{\hat q(S', a', \mathbf w)}$, a target computed with the very weights it changes ([[semi-gradient-td]]). With a table that does no harm: raising the entry for $(S, A)$ leaves the entries for $S'$ alone. With a network it does: $S'$ is usually the state right after $S$, the two look alike, and the update that raises $\val{\hat q(S, A)}$ raises the estimates at $S'$ too, and with them the target. The next update chases a target that has moved, often away from it. The loop can settle, oscillate, or push the values up without end; it is one face of the [[deadly-triad]], and the reason the values of an unlucky run can climb past anything the world can pay.

### Holding the target still {#copy}

::: definition {#def-target} Target network
A second set of weights $\mathbf w^-$, used only to compute targets. Every $C$ steps it is overwritten with the current weights, $\mathbf w^- \leftarrow \mathbf w$. The loss is
$$L(\mathbf w) = \mathbb E\Big[\big(\rew{r} + \gam \max_{a'} \val{\hat q(s', a', \mathbf w^-)} - \val{\hat q(s, a, \mathbf w)}\big)^2\Big]. \label{loss}$$
A soft version moves the copy a little every step instead, $\mathbf w^- \leftarrow \tau \mathbf w + (1 - \tau)\,\mathbf w^-$ with a small $\tau$. The copy is then an average of past weights about $1/\tau$ steps old; a hard copy every $C$ steps is on average $C/2$ steps old, so $\tau = 0.005$ behaves roughly like $C = 400$.
:::

Between two copies the targets are fixed functions of the data, and learning is ordinary regression: fit $\val{\hat q(s, a, \mathbf w)}$ to $\rew{r} + \gam \max_{a'} \val{\hat q(s', a', \mathbf w^-)}$ on the transitions at hand. Each period then performs, approximately, one step of value iteration ([[value-iteration]]): the network fits the Bellman optimality backup of the frozen copy, and the next copy starts from the result. Done in large batches, this is fitted Q iteration (Ernst et al., 2005; Riedmiller, 2005); DQN does it incrementally, a minibatch at a time, with replayed transitions ([[experience-replay]]).

### How often to copy {#period}

$C$ trades stability for speed. With a small $C$ the targets move almost as fast as without a copy. With a large $C$ each copy is one backup, so information travels one step backward per period: the news that the pole fell, after 400 steps of balancing, reaches the early states only after many copies.

DQN on CartPole, with the guide's other settings and 20 seeds per value, shows both ends. A run counts as ending well when its last training episodes keep the pole up for 450 steps or more on average:

| target copied every | runs that end well | their last training episodes, steps |
| --- | --- | --- |
| never (targets from the current weights) | 0 of 20 | 46 to 446 |
| 100 steps | 11 of 20 | 161 to 500 |
| 500 steps | 16 of 20 | 402 to 500 |
| 2,000 steps | 4 of 20 | 118 to 500 |

Without a copy, no run ends well, and 17 of the 20 end below 200 steps. Copying every 2,000 steps, only 4 do. The best period is in the middle, and it is a knob like the step size: its sweet spot depends on the world and on the other knobs.

### Overestimation stays {#overestimation}

A target network makes the targets stable, not unbiased. The max over noisy estimates still picks up their noise, so the targets lean high, and the copy passes the lean on from one period to the next: maximization bias, as with [[double-q|tabular Q-learning]]. Double DQN lets the current network choose the next action and the target network evaluate it ([[dqn-extensions]]).

### Historical remarks {#history}

Fitting a network or an ensemble of trees to fixed targets in batches came first (Ernst et al., 2005; Riedmiller, 2005). The first DQN (Mnih et al., 2013) computed its targets with the current weights; the 2015 version added the periodic copy, and in its ablations removing the copy lowered the score on all five games tested, though removing replay usually cost more. Lillicrap et al. (2016) used the soft copy, with $\tau = 0.001$, for continuous control. Van Hasselt et al. (2018) measured how often DQN's values diverge in practice, and found the target network among the ingredients that keep them bounded.

## Card

### Idea

Compute the targets with a frozen copy of the network, refreshed every $C$ steps. Between copies, learning is plain regression toward fixed targets; each copy is one step of approximate value iteration.

::: analogy
Aiming at a target that someone moves after every shot, against one they move only once you have had a few hundred tries.
:::

### The loss {#formula}

$$L(\mathbf w) = \mathbb E\Big[\big(\rew{r} + \gam \max_{a'} \val{\hat q(s', a', \mathbf w^-)} - \val{\hat q(s, a, \mathbf w)}\big)^2\Big], \qquad \mathbf w^- \leftarrow \mathbf w \ \text{every } C \text{ steps}$$

### Why it matters {#why}

- Breaks the loop in which an update raises its own target.
- Turns each period into a supervised regression problem, which networks handle well.
- On CartPole, without it none of 20 runs ends well; copying every 500 steps, 16 of 20 do.

### Pitfalls

- Copying too often brings the moving target back; too rarely, values creep backward one step per copy.
- It does not remove overestimation: the max still picks up noise.
- The period is in steps, not episodes: it should be retuned when the episode length or the update frequency changes.

### Check yourself {#check}

::: question
Why does a moving target hurt a network more than a table?
---
A table's update changes only the entry for $(S, A)$; the target uses the entries for $S'$, which stay put. A network's update changes its estimates everywhere, especially at states that look like $S$, and $S'$ usually does: raising the estimate raises the target, which the next update chases.
:::

::: question
With the target network frozen, what problem does each period solve?
---
A regression: fit $\val{\hat q(s, a, \mathbf w)}$ to the fixed targets $\rew{r} + \gam \max_{a'} \val{\hat q(s', a', \mathbf w^-)}$. That is one approximate step of value iteration, applying the Bellman optimality backup to the frozen copy.
:::

::: question
Why can a very long period slow learning down?
---
Each copy moves value information one step backward, as one sweep of value iteration does. With a long period, news from late in an episode needs many periods, and so many steps, to reach the states early in it.
:::
