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
+++

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

The size $N$ decides how old the data may be, and both ends fail. With $N$ equal to the batch size, each batch is exactly the last $B$ steps: replay in name only, as correlated as online learning. With a very large $N$, the memory keeps transitions from a policy that was much worse, in states the current policy no longer visits, and the network keeps spending its updates on them.

DQN on CartPole, with the guide's other settings and five seeds per size, shows both ends. A run counts as ending well when its last training episodes keep the pole up for 450 steps or more on average:

| memory $N$ | runs that end well | their last training episodes, steps |
| --- | --- | --- |
| 128 (no replay) | 2 of 5 | 415 to 500 |
| 1,000 | 3 of 5 | 305 to 500 |
| 10,000 | 5 of 5 | 493 to 500 |
| 100,000 | 2 of 5 | 153 to 499 |

Without replay the runs still learn, but they wobble: no run collapses, and most never quite settle. With 100,000 steps of memory, one run ends at 153 steps. Larger studies agree that the age of the oldest policy in the memory matters, and the number of updates per new transition with it (Fedus et al., 2020).

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
- Too small a memory is online learning again; too large a memory keeps fitting a policy long gone. On CartPole, 10,000 steps let all five seeds end well; 128 and 100,000 only two.
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
