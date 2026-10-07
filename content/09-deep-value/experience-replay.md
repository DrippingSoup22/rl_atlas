+++
summary = "Keep the last transitions in a memory and learn from random minibatches of them, instead of from each step as it comes: the updates no longer share the correlations of consecutive steps, and every transition is used many times. It is allowed because Q-learning is off-policy: an old transition is still good evidence about the optimal values."
prereqs = ["neural-networks", "q-learning", "dyna-q"]
lab = "dqn-replay"
sources = [
  { text = "Lin (1992), Self-improving reactive agents based on reinforcement learning, planning and teaching, Machine Learning 8", url = "https://doi.org/10.1007/BF00992699" },
  { text = "Mnih et al. (2015), Human-level control through deep reinforcement learning, Nature 518", url = "https://doi.org/10.1038/nature14236" },
  { text = "Zhang & Sutton (2017), A deeper look at experience replay, NIPS Deep Reinforcement Learning Symposium", url = "https://arxiv.org/abs/1712.01275" },
  { text = "Fedus, Ramachandran, Agarwal, Bengio, Larochelle, Rowland & Dabney (2020), Revisiting fundamentals of experience replay, ICML", url = "https://arxiv.org/abs/2007.06700" },
  { text = "Schaul, Quan, Antonoglou & Silver (2016), Prioritized experience replay, ICLR", url = "https://arxiv.org/abs/1511.05952" },
]

[story]
scene = "cartpole"
env = "cartpole"
formula = '''\step{1}{(s, a, \rew{r}, s') = \text{the last 128 steps} \qquad} \step{2}{(s, a, \rew{r}, s') \sim \operatorname{Uniform}(\mathcal D),\ |\mathcal D| = 10{,}000}'''

[story.runs]
replay = { recording = "dqn-cartpole", name = "memory of 10,000 steps" }
none = { recording = "dqn-cartpole-no-replay", name = "no replay" }
+++

## Story

::: step {run = "none", at = 0, formula = 1}
**The same DQN, with a memory of 128 steps**: exactly one batch. Every 4 steps the network learns from the last 128 transitions, nothing older. Each transition is still used as many times as before; what changes is what it is mixed with: 128 consecutive moments of a single stretch of play.
:::

::: step {run = "none", checkpoints = [10, 11, 12, 13], hold = 2200}
**It learns, and unlearns.** The training episodes swing from block to block: the full 500 steps on average, then 261, then 89, then 43. Each stretch of batches pulls the whole network toward the states it has just been in, and away from everything else. With the memory of 10,000, the same seed stayed between 204 and 238.
:::

::: step {run = "none", at = 29}
**Block 29: it has it.** The test episode lasts the full 500 steps, the cart never more than a meter from the middle. Nothing on the map looks wrong.
:::

::: step {run = "none", at = 30, map = "action"}
**5,000 steps later, it has lost it.** The network now pushes left in every state on the map, and the pole falls after 9 steps. While the pole stays up, the last 128 steps are 128 moments of the same balanced episode, nearly identical states. A network trained only on those has no reason to keep what it knew about the rest.
:::

::: step {run = "replay", at = 40, formula = 2, curves = ["replay", "none"], metric = "return"}
**Twenty seeds each.** Without replay, most runs balance the pole at some point, but only 9 of 20 still do at the end. With a memory of 10,000 steps, each batch mixes moments from about 20 episodes, and 13 of 20 hold on. Bigger is not always better: [the Lab](lab:dqn-replay) shows how the odds move with the size of the memory.
:::

## Textbook

### Learning from each step as it comes {#why}

Online Q-learning updates on each transition once, in the order it was lived ([[q-learning]]). With a table that is fine: an update changes one entry, and the next transition, however similar, changes its own. With a network every update moves every estimate ([[neural-networks]]), and consecutive transitions are anything but independent: a pole leaning right stays a pole leaning right for dozens of steps, and so does the push that answers it. A run of such updates drags the whole function toward that one region's targets, and the network forgets what it had learned elsewhere. And each transition, which may have cost a slow simulation or a real robot's time, is used once and thrown away.

### The memory {#memory}

::: definition {#def-replay} Replay memory
A replay memory $\mathcal D$ holds the last $N$ transitions $(S_t, A_t, R_{t+1}, S_{t+1})$, with a mark when $S_{t+1}$ ended the episode. After every step the new transition goes in, pushing out the oldest once the memory is full. Every few steps a minibatch of $B$ transitions is drawn from it uniformly at random, and the network takes a gradient step on their average squared TD error,
$$L(\mathbf w) = \mathbb E_{(s, a, r, s') \sim \mathcal D}\Big[\big(\rew{r} + \gam \max_{a'} \val{\hat q(s', a', \mathbf w^-)} - \val{\hat q(s, a, \mathbf w)}\big)^2\Big], \label{loss}$$
where $\mathbf w^-$ are the weights of the [[target-network]], a copy that changes only now and then.
:::

A random minibatch mixes transitions from many moments and many episodes, so its gradient is closer to the gradient over everything the agent has seen, and much less tied to where the agent happens to be. Each transition is also reused: with a memory of $N = 10{,}000$ steps, a batch of $B = 128$ every 4 steps, each step adds one transition and draws 32 samples, so a transition is drawn about 32 times before it leaves the memory.

Replaying old transitions is only correct for an off-policy method. A transition records what the world did, $s, a \to r, s'$, and Q-learning's target asks only for the best value at $s'$, whatever the policy that chose $a$; so a transition gathered by a clumsier, more exploratory policy is still a fair sample of the dynamics behind the Bellman optimality equation. SARSA's target uses the next action of the current policy, and policy-gradient methods average over the current policy's own choices; an old transition says nothing direct about either without importance-sampling corrections ([[importance-sampling]]).

### How big a memory {#size}

The size $N$ decides how old the data may be, and both ends can fail. With $N$ equal to the batch size, each batch is exactly the last $B$ steps: replay in name only, as correlated as online learning. With a very large $N$, the memory keeps transitions from a policy that was much worse, in states the current policy no longer visits, and the network keeps spending its updates on them.

DQN on CartPole, with the guide's other settings and 20 seeds per size, shows how often. A run counts as ending well when its last training episodes keep the pole up for 450 steps or more on average:

| memory $N$ | runs that end well | their last training episodes, steps |
| --- | --- | --- |
| 128 (no replay) | 9 of 20 | 200 to 500 |
| 1,000 | 12 of 20 | 240 to 500 |
| 10,000 | 13 of 20 | 203 to 500 |
| 100,000 | 11 of 20 | 119 to 500 |

Without replay the runs still learn, but many never settle: 17 of the 20 average 450 steps per episode for a while, and only 9 still do at the end. With replay, every run gets there, and 11 to 13 stay. On a world this small the gaps are modest: Fisher's exact test gives 34% for no replay against 10,000 steps, so even that one could be luck (see [[seeds]]). The largest memory has the lowest end, one run at 119 steps. Larger studies find that the age of the oldest policy in the memory matters, and the number of updates per new transition with it (Fedus et al., 2020).

### Replay as planning {#planning}

A replay memory is a model of the world that only remembers what happened: a sample from it is a transition that really occurred. Updating on remembered transitions between real steps is then a form of planning, as in [[dyna-q]], whose model also only knew transitions it had seen. Two differences: Dyna's model generalized nothing and could be queried for any remembered pair, while the memory hands back whole stored transitions; and Dyna picked them at random, which [[prioritized-sweeping]] improved by updating where the surprise was largest. Prioritized replay does the same for a replay memory ([[dqn-extensions]]).

### Historical remarks {#history}

Lin (1992) introduced experience replay, for agents that learned with networks in simulated worlds, next to learning from a teacher's lessons. Two decades later it became one of the two ingredients that let DQN learn Atari games from pixels (Mnih et al., 2013, 2015). Schaul et al. (2016) made it prioritized; Zhang and Sutton (2017) showed that the size of the memory is a sensitive knob in its own right, too small and too large both hurting.

## Card

### Idea

Store the last $N$ transitions and learn from random minibatches of them. The updates stop following the correlations of consecutive steps, and each transition is used many times. Only off-policy methods may do this, because an old transition is still evidence about the optimal values.

::: analogy
Studying from a shuffled deck of old flashcards instead of re-reading only today's page.
:::

### How it works {#how}

$$\mathcal D \leftarrow \mathcal D \cup \{(S_t, A_t, R_{t+1}, S_{t+1})\} \ \text{(the oldest out when full)}, \qquad (s, a, r, s')_{1..B} \sim \operatorname{Uniform}(\mathcal D),$$

then one gradient step on the batch's average of $\big(\rew{r} + \gam \max_{a'} \val{\hat q(s', a', \mathbf w^-)} - \val{\hat q(s, a, \mathbf w)}\big)^2$.

### Why it matters {#why}

- Breaks the correlation of consecutive steps, so a network does not drift toward the region it is in.
- Reuses each transition many times: about 32 times in the guide's DQN.
- Averages over many past behaviors, which smooths learning.

### Pitfalls

- Only for off-policy learning; on-policy methods need corrections to learn from old data.
- Too small a memory is online learning again; too large a memory keeps fitting a policy long gone. On CartPole without replay, 17 of 20 seeds reach 450 steps per episode for a while, and only 9 stay there.
- Transitions that end an episode must be marked, or their targets bootstrap from a state that never comes.

### Check yourself {#check}

::: question
Why may DQN learn from a transition that a much older policy produced, while SARSA may not?
---
DQN's target, $\rew{r} + \gam \max_{a'} \val{\hat q(s', a')}$, depends on the transition only through what the world did; it does not care which policy chose $a$. SARSA's target uses the next action of the current policy, which an old transition does not contain, and its estimate is of that policy's values, not of the optimal ones.
:::

::: question
With a memory as large as the batch, what is each batch made of?
---
The last $B$ transitions, in full: every update learns from the most recent, most correlated steps, exactly what replay was meant to avoid.
:::

::: question
Why can a very large memory hurt?
---
It holds transitions from policies that were much worse, in states the current policy no longer visits. The network keeps spending updates fitting those, and the recent experience that matters most is a smaller share of every batch.
:::
