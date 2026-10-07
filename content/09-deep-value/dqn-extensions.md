+++
summary = "Three changes to DQN, each aimed at one weakness: Double DQN lets one network choose the next action and the other judge it, against overestimation; a dueling network learns how good a state is apart from how much each action adds; prioritized replay draws surprising transitions more often. Each pays off where its weakness is large: on CartPole only Double DQN is a clear gain, and the dueling split even hurts."
change = "Change one part of DQN: the target (Double DQN), the network's head (dueling), or how the replay memory is sampled (prioritized)."
prereqs = ["dqn", "double-q", "prioritized-sweeping"]
lab = "dqn-extensions"
sources = [
  { text = "van Hasselt (2010), Double Q-learning, NIPS", url = "https://papers.nips.cc/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html" },
  { text = "Bellemare, Dabney & Munos (2017), A distributional perspective on reinforcement learning, Proceedings of the 34th International Conference on Machine Learning", url = "https://arxiv.org/abs/1707.06887" },
  { text = "Fortunato et al. (2018), Noisy networks for exploration, International Conference on Learning Representations", url = "https://arxiv.org/abs/1706.10295" },
  { text = "Badia, Piot, Kapturowski, Sprechmann, Vitvitskyi, Guo & Blundell (2020), Agent57: outperforming the Atari human benchmark, Proceedings of the 37th International Conference on Machine Learning", url = "https://arxiv.org/abs/2003.13350" },
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
**One change in the target.** A max over noisy values usually picks one that was overestimated ([[double-q]]). Double DQN splits the job: the network being trained chooses the next action, and the target network says what it is worth. The two err differently, so an overrated choice gets a soberer price. Here at block 20, its pole stays up for 172 steps.
:::

::: step {run = "double", at = 40, curves = ["dqn", "double"], metric = "q"}
**The largest value in each batch**, 20 seeds each. No state here is worth more than 100. DQN's estimates went past it on 12 seeds, by up to 30%; Double DQN's on 9, by at most 11%. Otherwise the two are alike: they learn about as fast, and 13 and 11 runs of 20 end well, a gap that luck easily makes.
:::

::: step {run = "prio", at = 20, curves = ["dqn", "prio"], metric = "return"}
**Prioritized replay** replays surprises more often: steps with a large TD error, as [[prioritized-sweeping]] did with a model. Here it learns about as fast as DQN and ends well a little more often, 15 runs of 20 against 13: a gap that luck makes as often as not ([[seeds]]). Its gains show on harder games, where surprises are rare and costly to miss.
:::

::: step {run = "dueling", at = 40, curves = ["dqn", "double", "dueling", "prio"], metric = "return"}
**Dueling heads** learn a state's value and each action's advantage separately, a gain where most actions do the same, as in many Atari states. CartPole has two actions and the wrong one always matters, so here the split brings nothing: 12 runs of 20 end well, against DQN's 13. An extension answers one weakness; where that weakness is absent, it costs without paying.
:::

## Textbook

### One weakness each {#why}

DQN works, but it has known weaknesses ([[dqn]]). Its targets lean high, its network must learn every action's value even where the choice hardly matters, and its memory is sampled without regard to what each transition can still teach. Each extension below changes one thing to answer one weakness, and each was shown on Atari to improve on DQN across most games. They target different parts of the algorithm, so they combine: Rainbow (Hessel et al., 2018) put six of them together.

### Double DQN {#double}

The max of noisy estimates is biased upward. If $\val{\hat q(s', a')}$ equals the true value plus a zero-mean error for each $a'$, then $\mathbb E[\max_{a'} \hat q] \ge \max_{a'} \mathbb E[\hat q]$, with equality only when there is no noise. The more actions there are, and the noisier the estimates, the larger the gap. Q-learning's target takes exactly such a max, and bootstrapping passes the excess on from state to state. Double Q-learning removes the bias by keeping two independent estimates, choosing with one and evaluating with the other ([[double-q]]). DQN already has two networks, so Double DQN (van Hasselt, Guez & Silver, 2016) uses them:
$$y = \rew{r} + \gam\,\val{\hat q\big(s', \operatorname*{argmax}_{a'} \hat q(s', a', \mathbf w), \mathbf w^-\big)}. \label{eq-double}$$
The online network $\mathbf w$ chooses; the target network $\mathbf w^-$ evaluates. The two are not independent, since one is a recent copy of the other, but their errors differ enough to remove most of the bias. On Atari, DQN's value estimates ran far above the returns its policies actually earned in some games. Double DQN's stayed close to them, and its scores rose. The change costs nothing: one more forward pass of the online network on the next states.

On CartPole the bias is modest, since every action is learned often, but it is there. The learner bootstraps through the time limit at 500 steps, so no state is worth more than $1 / (1 - \gam) = 100$ to it. Over 20 seeds, DQN's estimates went past 100 on 12, by up to 30%; Double DQN's on 9, by at most 11%. Both learned about as fast, and they ended well on 13 and 11 seeds of 20, a gap within luck.

### Dueling networks {#dueling}

A Q-network learns, for each state, one number per action. Often the state's value is what matters, and the actions differ little: a car far from any other needs no particular steering. A dueling network (Wang et al., 2016) splits the head of the network into two streams, a state value $\val{v(s)}$ and an advantage $\err{a(s, a)}$ for each action, and recombines them as
$$\val{\hat q(s, a)} = \val{\hat v(s)} + \Big(\err{\hat a(s, a)} - \frac{1}{|\mathcal A|}\sum_{a'} \err{\hat a(s, a')}\Big). \label{eq-dueling}$$
Subtracting the mean advantage makes the split unique. Without it, any constant could move from $\val{\hat v}$ to $\err{\hat a}$ without changing $\val{\hat q}$. Every update now trains $\val{\hat v(s)}$, whatever action was taken, so the value of a state is learned from all of its transitions rather than from those of one action. On Atari, with up to 18 actions and many states where most of them do the same thing, this improved the scores of most games.

On CartPole there are two actions, and the wrong one always speeds up the fall. The state's value is not easier to learn than the two action values, so the split brings no gain. It does add a stream whose output can drift. Over 20 seeds, the dueling runs ended well 12 times, against DQN's 13, and their estimates went past 100 on 14 seeds, up to 124. This is the extension's weakness, not a failure of the method: it is designed for many actions of similar value.

### Prioritized replay {#prioritized}

Uniform sampling spends as many updates on transitions the network already predicts well as on those it gets wrong. Prioritized replay (Schaul et al., 2016) gives each transition a priority $p_i = |\err{\delta_i}| + \epsilon$, its latest TD error plus a small constant, and samples it with probability
$$P(i) = \frac{p_i^{\,\alpha}}{\sum_k p_k^{\,\alpha}}, \label{eq-per}$$
with $\alpha = 0.6$ here ($\alpha = 0$ is uniform sampling). New transitions enter with the highest priority, so each is replayed at least once soon. Skewing the sampling biases the updates toward the transitions drawn most often, which importance weights correct: each sample's loss is scaled by $\big(N \cdot P(i)\big)^{-\beta}$, divided by the largest weight in the batch, with $\beta$ raised from 0.4 to 1 over the run, so that the correction is complete by the end. A sum tree finds and updates priorities in logarithmic time.

This is prioritized sweeping ([[prioritized-sweeping]]) for a memory instead of a model: spend updates where the news is. On Atari it was one of the two most valuable of Rainbow's components. On CartPole it changes little: its runs first averaged 450 steps about when DQN's did (half of them by block 20), and 15 of 20 ended well, against DQN's 13, a gap within luck.

### Rainbow's other three {#rainbow}

- **Multi-step returns.** The target sums $n$ rewards before bootstrapping, $n = 3$ in Rainbow, so news travels $n$ steps per update ([[n-step-td]]). The replayed steps came from an older policy and Rainbow does not correct for it; with small $n$ the error is tolerable.
- **Distributional values.** Instead of the mean return, the network predicts its whole distribution, as probabilities over 51 fixed values, and learns it with a distributional Bellman update (Bellemare, Dabney & Munos, 2017). The greedy action still maximizes the mean, but learning the full shape gives the network richer targets.
- **Noisy networks.** ε-greedy is replaced by learned noise in the weights of the last layers (Fortunato et al., 2018): the network decides how much to randomize, state by state, and the noise shrinks where it stops paying off.

Agent57 (Badia et al., 2020), which adds learned exploration bonuses, recurrent memory and a choice among many discount and exploration settings to this family, was the first to beat the human benchmark on all 57 Atari games, Montezuma's Revenge included.

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

- Double: removes most of the overestimation at little cost; on CartPole, estimates above 100 on 9 seeds of 20 instead of 12, by at most 11% instead of 30%, with about as many ending well (11 of 20, against 13).
- Dueling: learns state values from every transition; a large gain where many actions are equivalent.
- Prioritized: spends updates on what the network still gets wrong; on CartPole, the quickest of the four to first balance the pole.
- They combine (Rainbow).

### Flaws

- Double: the two networks are not independent, so some bias remains.
- Dueling: an extra stream to fit; on CartPole, with two actions that always matter, it did not help (12 of 20 runs ended well, against DQN's 13).
- Prioritized: two more knobs ($\alpha$, $\beta$) and a sum tree; on CartPole it changed little (15 of 20 runs ended well, against DQN's 13).

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
Its gain comes from learning a state's value from every transition, which matters when many actions are worth about the same. CartPole has two actions that always differ, so there is nothing to gain, while the second stream adds outputs that can drift. Over 20 seeds, its runs ended well 12 times, against DQN's 13.
:::
