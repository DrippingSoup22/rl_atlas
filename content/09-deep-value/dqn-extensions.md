+++
summary = "Three changes to DQN, each aimed at one weakness: Double DQN lets one network choose the next action and the other judge it, against overestimation; a dueling network learns how good a state is apart from how much each action adds; prioritized replay draws surprising transitions more often. Each pays off where its weakness is large: on CartPole only Double DQN is a clear gain, and the dueling split even hurts."
change = "Change one part of DQN: the target (Double DQN), the network's head (dueling), or how the replay memory is sampled (prioritized)."
prereqs = ["dqn", "double-q", "prioritized-sweeping"]
lab = "dqn-extensions"
sources = [
  { text = "van Hasselt (2010), Double Q-learning, NIPS", url = "https://papers.nips.cc/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html" },
  { text = "van Hasselt, Guez & Silver (2016), Deep reinforcement learning with double Q-learning, AAAI", url = "https://arxiv.org/abs/1509.06461" },
  { text = "Wang, Schaul, Hessel, van Hasselt, Lanctot & de Freitas (2016), Dueling network architectures for deep reinforcement learning, ICML", url = "https://arxiv.org/abs/1511.06581" },
  { text = "Schaul, Quan, Antonoglou & Silver (2016), Prioritized experience replay, ICLR", url = "https://arxiv.org/abs/1511.05952" },
  { text = "Hessel et al. (2018), Rainbow: combining improvements in deep reinforcement learning, AAAI", url = "https://arxiv.org/abs/1710.02298" },
]

[story]
scene = "cartpole"
env = "cartpole"
formula = '''\step{1}{y = \rew{r} + \gam\,\val{\hat q\big(s', \operatorname*{argmax}_{a'} \hat q(s', a', \mathbf w), \mathbf w^-\big)}}'''

[story.runs]
dqn = { recording = "dqn-cartpole", name = "DQN" }
double = { recording = "double-dqn-cartpole", name = "Double DQN" }
dueling = { recording = "dueling-dqn-cartpole", name = "dueling DQN" }
prio = { recording = "prioritized-dqn-cartpole", name = "prioritized replay" }
+++

## Story

::: step {run = "double", at = 20, formula = 1}
**One change in the target.** DQN's target takes the largest of the target network's values at the next state. If those values are noisy, the largest one is usually one that was overestimated, so the max picks up the noise ([[double-q]]). Double DQN splits the job between its two networks. The online network, the one being trained, chooses the next action; the target network says what that action is worth. The two err in different ways, so a choice that the online network overrated is valued more soberly by the copy. Here is Double DQN at block 20: its test episode lasts 343 steps, and its values reach 95.3.
:::

::: step {run = "double", at = 40, curves = ["dqn", "double"], metric = "q"}
**The largest value in each batch**, averaged over 20 seeds. A state here is worth at most $1 / (1 - \gam) = 100$ to a learner that bootstraps through the time limit. DQN's average passes it at block 39 and ends at 101; Double DQN's peaks at 97. Seed by seed, DQN's estimates went past 100 on 15 of its 20 seeds, in 86 blocks in all and up to 129; Double DQN's on 7 seeds, in 28 blocks, up to 104. With 20 seeds, a difference of 15 against 7 is unlikely to be luck (Fisher's exact test: 2.5%; [[seeds]]). The two learn about as fast: half the seeds first averaged 450 steps per training episode by block 21 for Double DQN, by block 23 for DQN, and each ends well on 16 seeds of 20.
:::

::: step {run = "prio", at = 20, curves = ["dqn", "prio"], metric = "return"}
**Prioritized replay** draws a transition more often the larger its last TD error: the memory's surprises, as prioritized sweeping did for a model. It does speed learning up. Every seed's training episodes reach 450 steps by block 28, where DQN's take until block 37, and the run shown here already balances its test episode for 500 steps at block 20. But it does not hold: only 8 of 20 seeds end well, against DQN's 16, and averaged over the 20 seeds the last block is at 405 steps against DQN's 480. That gap is unlikely to be luck (2.3%). The memory is sampled away from the uniform mix that replay was there to provide, and the importance weights only partly correct for it.
:::

::: step {run = "dueling", at = 40, curves = ["dqn", "double", "dueling", "prio"], metric = "return"}
**All four**, 20 seeds each. A dueling network learns a state's value and each action's advantage in two separate streams. That pays when most actions are worth about the same, as in the many Atari states where most of the 18 joystick moves change nothing. CartPole has two actions, and the wrong one always matters. Here the split brings no benefit: 11 of 20 runs end well, against DQN's 16, though with 20 seeds that gap could still be luck (18%). Its estimates went past 100 on 12 seeds, up to 124. An extension answers a particular weakness. Where that weakness is absent, it costs without paying.
:::

## Textbook

### One weakness each {#why}

DQN works, but it has known weaknesses ([[dqn]]). Its targets lean high, its network must learn every action's value even where the choice hardly matters, and its memory is sampled without regard to what each transition can still teach. Each extension below changes one thing to answer one weakness, and each was shown on Atari to improve on DQN across most games. They target different parts of the algorithm, so they combine: Rainbow (Hessel et al., 2018) put six of them together.

### Double DQN {#double}

The max of noisy estimates is biased upward. If $\val{\hat q(s', a')}$ equals the true value plus a zero-mean error for each $a'$, then $\mathbb E[\max_{a'} \hat q] \ge \max_{a'} \mathbb E[\hat q]$, with equality only when there is no noise. The more actions there are, and the noisier the estimates, the larger the gap. Q-learning's target takes exactly such a max, and bootstrapping passes the excess on from state to state. Double Q-learning removes the bias by keeping two independent estimates, choosing with one and evaluating with the other ([[double-q]]). DQN already has two networks, so Double DQN (van Hasselt, Guez & Silver, 2016) uses them:
$$y = \rew{r} + \gam\,\val{\hat q\big(s', \operatorname*{argmax}_{a'} \hat q(s', a', \mathbf w), \mathbf w^-\big)}. \label{eq-double}$$
The online network $\mathbf w$ chooses; the target network $\mathbf w^-$ evaluates. The two are not independent, since one is a recent copy of the other, but their errors differ enough to remove most of the bias. On Atari, DQN's value estimates ran far above the returns its policies actually earned in some games. Double DQN's stayed close to them, and its scores rose. The change costs nothing: one more forward pass of the online network on the next states.

On CartPole the bias is modest, since every action is learned often, but it is there. The learner bootstraps through the time limit at 500 steps, so no state is worth more than $1 / (1 - \gam) = 100$ to it. Over 20 seeds, DQN's estimates went past 100 on 15, by up to 29%; Double DQN's on 7, by at most 4%. Both learned about as fast and ended well on 16 seeds of 20.

### Dueling networks {#dueling}

A Q-network learns, for each state, one number per action. Often the state's value is what matters, and the actions differ little: a car far from any other needs no particular steering. A dueling network (Wang et al., 2016) splits the head of the network into two streams, a state value $\val{v(s)}$ and an advantage $\err{a(s, a)}$ for each action, and recombines them as
$$\val{\hat q(s, a)} = \val{\hat v(s)} + \Big(\err{\hat a(s, a)} - \frac{1}{|\mathcal A|}\sum_{a'} \err{\hat a(s, a')}\Big). \label{eq-dueling}$$
Subtracting the mean advantage makes the split unique. Without it, any constant could move from $\val{\hat v}$ to $\err{\hat a}$ without changing $\val{\hat q}$. Every update now trains $\val{\hat v(s)}$, whatever action was taken, so the value of a state is learned from all of its transitions rather than from those of one action. On Atari, with up to 18 actions and many states where most of them do the same thing, this improved the scores of most games.

On CartPole there are two actions, and the wrong one always speeds up the fall. The state's value is not easier to learn than the two action values, so the split brings no gain. It does add a stream whose output can drift. Over 20 seeds, the dueling runs ended well 11 times, against DQN's 16, and their estimates went past 100 on 12 seeds, up to 124. This is the extension's weakness, not a failure of the method: it is designed for many actions of similar value.

### Prioritized replay {#prioritized}

Uniform sampling spends as many updates on transitions the network already predicts well as on those it gets wrong. Prioritized replay (Schaul et al., 2016) gives each transition a priority $p_i = |\err{\delta_i}| + \epsilon$, its latest TD error plus a small constant, and samples it with probability
$$P(i) = \frac{p_i^{\,\alpha}}{\sum_k p_k^{\,\alpha}}, \label{eq-per}$$
with $\alpha = 0.6$ here ($\alpha = 0$ is uniform sampling). New transitions enter with the highest priority, so each is replayed at least once soon. Skewing the sampling biases the updates toward the transitions drawn most often, which importance weights correct: each sample's loss is scaled by $\big(N \cdot P(i)\big)^{-\beta}$, divided by the largest weight in the batch, with $\beta$ raised from 0.4 to 1 over the run, so that the correction is complete by the end. A sum tree finds and updates priorities in logarithmic time.

This is prioritized sweeping ([[prioritized-sweeping]]) for a memory instead of a model: spend updates where the news is. On Atari it was one of the two most valuable of Rainbow's components. On CartPole it was the quickest to first average 450 steps (every seed by block 28, against block 37 for DQN), but it did not hold: 12 of its 20 runs fell back below 450 by the end, some as low as 240.

### Historical remarks {#history}

Van Hasselt proposed Double Q-learning for tables in 2010 and brought it to DQN with Guez and Silver in 2016. The dueling architecture and prioritized replay appeared the same year, also from DeepMind. Rainbow (Hessel et al., 2018) combined them with three more: multi-step returns ([[n-step-td]]), distributional value estimates and noisy networks for exploration. The combination far outscored each one alone, and its ablations ranked prioritized replay and multi-step returns as the most important parts.

## Card

### Idea

Three independent repairs to DQN. Double DQN chooses the next action with the online network and values it with the target network, so the max stops picking up noise. A dueling head learns $\val{v(s)}$ apart from the advantages $\err{a(s, a)}$. Prioritized replay samples transitions in proportion to their TD error, with importance weights to undo the bias.

::: analogy
A second opinion before trusting the best-looking offer; a map of the terrain before the turn-by-turn directions; and studying hardest the exercises you got wrong.
:::

### The update {#update}

$$y = \rew{r} + \gam\,\val{\hat q\big(s', \operatorname*{argmax}_{a'} \hat q(s', a', \mathbf w), \mathbf w^-\big)}, \qquad \val{\hat q(s, a)} = \val{\hat v(s)} + \err{\hat a(s, a)} - \overline{\err{\hat a(s, \cdot)}}, \qquad P(i) \propto |\err{\delta_i}|^{\alpha}$$

### One change from DQN {#change}

Each extension changes one part: the target (Double), the network's head (dueling), or the sampling of the memory (prioritized).

### Backup diagram {#backup}

{{backup double-q}}

Double DQN's backup is Double Q-learning's: one estimate chooses the next action, the other values it. Here the two estimates are the online network and its target copy.

### Pseudocode

::: pseudocode
As DQN, with each extension changing one line:
Double: $y_j \leftarrow \rew{r_j} + \gam\,\val{\hat q(s'_j, a^*, \mathbf w^-)}$, where $a^* = \operatorname{argmax}_{a'} \val{\hat q(s'_j, a', \mathbf w)}$
Dueling: the network outputs $\val{\hat v(s)}$ and $\err{\hat a(s, \cdot)}$; $\val{\hat q(s, a)} = \val{\hat v(s)} + \err{\hat a(s, a)} - \operatorname{mean}_{a'} \err{\hat a(s, a')}$
Prioritized: draw transition $i$ with probability $\propto p_i^{\alpha}$; weight its loss by $(N P(i))^{-\beta} / \max_j w_j$
  After the step, set $p_i \leftarrow |\err{\delta_i}| + \epsilon$; a new transition gets the highest priority so far
:::

### Perks

- Double: removes most of the overestimation at no cost; on CartPole, estimates above 100 on 7 seeds of 20 instead of 15, with the same 16 of 20 ending well.
- Dueling: learns state values from every transition; a large gain where many actions are equivalent.
- Prioritized: spends updates on what the network still gets wrong; on CartPole, the quickest of the four to first balance the pole.
- They combine (Rainbow).

### Flaws

- Double: the two networks are not independent, so some bias remains.
- Dueling: an extra stream to fit; on CartPole, with two actions that always matter, it did not help (11 of 20 runs ended well, against DQN's 16).
- Prioritized: two more knobs ($\alpha$, $\beta$), a sum tree, and less stable late learning here (8 of 20 runs ended well, against DQN's 16).

### Knobs

| Knob | Too low | Too high |
| --- | --- | --- |
| $\alpha$ (prioritized) | uniform replay, no speed-up | replays a few transitions over and over |
| $\beta$ start (prioritized) | biased updates early | slower early learning |
| $\epsilon$ in $p_i$ | transitions with small errors are never drawn again | priorities flatten toward uniform |

### Pitfalls

- Double DQN: choose with $\mathbf w$ and evaluate with $\mathbf w^-$, not the other way around.
- Dueling: without subtracting the mean (or the max) advantage, $\val{\hat v}$ and $\err{\hat a}$ are not identifiable.
- Prioritized: forgetting the importance weights, or updating the priorities with the wrong indices after the step.
- Expecting an extension to help where its weakness is absent.

### Check yourself {#check}

::: question
Why is the max of noisy, unbiased estimates biased upward?
---
For each action, $\max_{a'} \hat q(s', a') \ge \hat q(s', b)$ for any fixed $b$; taking expectations, $\mathbb E[\max_{a'} \hat q] \ge \max_b \mathbb E[\hat q(s', b)]$. The max tends to pick whichever estimate happens to be too high. Equality needs no noise at all.
:::

::: question
In Double DQN, what would happen if the target network both chose and evaluated the next action?
---
That is plain DQN: the same noisy estimate both selects the action that looks best and reports its value, so the overestimate stays. Splitting the two roles between networks with different errors is what removes most of the bias.
:::

::: question
Why can a dueling network hurt on CartPole, when it helps on Atari?
---
Its gain comes from learning a state's value from every transition, which matters when many actions are worth about the same. CartPole has two actions that always differ, so there is nothing to gain, while the second stream adds outputs that can drift. Over 20 seeds, its runs ended well 11 times, against DQN's 16.
:::
