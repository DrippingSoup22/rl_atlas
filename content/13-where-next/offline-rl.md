+++
summary = "Learn a policy from a fixed dataset of past experience, with no further interaction: logs of a system, recordings of operators, the replay memory of an earlier agent. The catch is that nothing can check the values of actions the data never shows, and plain off-policy methods overestimate exactly those. Offline methods stay close to the data: they constrain the policy, make the values pessimistic, or learn only from actions that were actually taken."
prereqs = ["on-off-policy", "q-learning", "importance-sampling"]
lab = "offline-cliff"
sources = [
  { text = "Levine, Kumar, Tucker & Fu (2020), Offline reinforcement learning: tutorial, review, and perspectives on open problems", url = "https://arxiv.org/abs/2005.01643" },
  { text = "Fujimoto, Meger & Precup (2019), Off-policy deep reinforcement learning without exploration (BCQ), ICML", url = "https://arxiv.org/abs/1812.02900" },
  { text = "Kumar, Zhou, Tucker & Levine (2020), Conservative Q-learning for offline reinforcement learning, NeurIPS", url = "https://arxiv.org/abs/2006.04779" },
  { text = "Fujimoto & Gu (2021), A minimalist approach to offline reinforcement learning (TD3+BC), NeurIPS", url = "https://arxiv.org/abs/2106.06860" },
  { text = "Kostrikov, Nair & Levine (2022), Offline reinforcement learning with implicit Q-learning, ICLR", url = "https://arxiv.org/abs/2110.06169" },
  { text = "Chen et al. (2021), Decision transformer: reinforcement learning via sequence modeling, NeurIPS", url = "https://arxiv.org/abs/2106.01345" },
  { text = "Fu, Kumar, Nachum, Tucker & Levine (2020), D4RL: datasets for deep data-driven reinforcement learning", url = "https://arxiv.org/abs/2004.07219" },
]

[story]
scene = "grid"
env = "cliff"
digits = 0
seed = 2
range = 20
average = 100
formula = '''\step{1}{\val{Q(s,a)} \leftarrow \val{Q(s,a)} + \alp\,\big[\rew{r} + \gam \max_{a'} \val{Q(s',a')} - \val{Q(s,a)}\big]} \step{2}{\qquad \max_{a' :\, (s', a') \in \mathcal D}}'''

[story.runs]
plain = { algorithm = "offline-q", alpha = 0.5, epsilon = 0.1, gamma = 1.0, q0 = 0.0, logEpisodes = 10, units = 41, measures = ["deployed"], name = "Q-learning on the log" }
constrained = { algorithm = "offline-bcq", alpha = 0.5, epsilon = 0.1, gamma = 1.0, q0 = 0.0, logEpisodes = 10, units = 41, measures = ["deployed"], name = "batch-constrained" }
+++

## Story

::: step {run = "plain", at = 0, play = 1, pace = 45, arrows = false}
**The cliff, with no one to explore it.** All we have is a log: a careful walker crossed ten times, up the first column, along the top row and down to the goal, with a random move one time in ten. Watch it being written. From now on, nobody acts: whatever we learn must come from these moves alone.
:::

::: step {run = "plain", at = 1, arrows = false, fog = true}
**The log: 207 moves.** The walker never fell, and its crossings cost between 17 and 27 steps. The fog covers the 13 open tiles it never stood on. On the 24 it did, it mostly tried one move, sometimes two: 41 pairs of a tile and a move, out of the 148 on the open tiles. Every value still starts at 0.
:::

::: step {run = "plain", at = 41, values = true, range = 2, path = true, fog = true, formula = 1}
**Forty passes of Q-learning over the log.** Every tried move's value now sits between $-1$ and $-2$, though the start is 17 steps from the goal. Each target takes the max at the next tile, and the max picks a move the log never tried, still at its starting 0. So every tile says 0: the goal looks one step away from everywhere. From the start, three moves were never tried. The first of them, right, wins, and the policy walks into the cliff.
:::

::: step {run = "constrained", at = 41, values = true, path = true, fog = true, formula = 2}
**One change: the max only looks at moves the log tried** in the next tile, and so does the policy. Now each value is what the log can vouch for: $-17$ at the start, counting down to $-1$ beside the goal. The policy walks the careful route in 17 steps, better than the walker itself, which averaged 20.7 because of its random moves.
:::

::: step {run = "constrained", at = 41, curves = ["plain", "constrained"], metric = "deployed", domain = [-100, 0]}
**A hundred logs each.** Each line plays the learned policy once from the start, until the goal, a fall, or 100 steps. Plain Q-learning never finds the goal from any of the 100 logs. Two hundred crossings instead of ten help it only on 9 logs of 100. Starting every value at $-17$ or below works; at $-15$, it works on 4 logs of 100. The constrained learner walks the 17 steps from every log, even from a walker so random that it averaged $-88$. It takes its time: each pass carries the goal's news only a little way back, and the policy settles on the route after 30 passes on a typical log, 34 at most. It cannot find the 13-step path along the edge, because nobody ever walked it. [Try it in the Lab](lab:offline-cliff).
:::

## Textbook

### Learning without trying {#setting}

Everything in this guide so far learns by interacting: try, observe, adjust. In many places, interaction is the scarce thing. A hospital cannot experiment with treatments to learn a policy, a recommender cannot show random content to millions of people, a robot fleet has years of logs but each new trial costs time and wear. **Offline RL**, also called batch RL, learns a policy from a fixed dataset $\mathcal D$ of transitions $(s, a, \rew r, s')$ collected by some behavior policy $b$, with no further interaction. If it works, data becomes reusable, as it is in supervised learning.

### Why off-policy methods fail here {#fail}

Q-learning and DQN are off-policy, so it is tempting to run them on the dataset ([[on-off-policy]]). Their targets take $\max_{a'} \val{\hat q(s', a')}$, and the max ranges over *every* action, including ones the data never shows in $s'$. Their estimates there are whatever the network happens to output, with nothing to correct them. The max picks the most overestimated of them, the targets carry the error to other states, and the learned policy heads for exactly those untested actions. Online, trying such an action would reveal the mistake. Offline, nothing ever does. This *extrapolation error* makes naive off-policy learning on fixed data fail, often far below the policy that collected the data (Fujimoto, Meger and Precup, 2019).

::: example {#ex-cliff} A careful walker's log
On the cliff ([[sarsa]]), a walker that goes up the first column, along the top row and down to the goal, with a random move one time in ten, logs ten crossings: about 200 moves. Q-learning, swept over the log 40 times with every value starting at 0, never deploys a policy that reaches the goal, on any of 100 such logs. Every target's max reads a move the log never tried, still at 0, so every tile's value is 0, and the policy takes untried moves: into a wall, or into the cliff. With 200 crossings it reaches the goal from only 9 logs of 100: a single untried move on the way is enough to break it. Restricting the max and the policy to the moves the log tried deploys the walker's 17-step route from every log.
:::

### Staying close to the data {#close}

The families of offline methods differ in how they keep the policy where the data can vouch for it:

- **Constrain the policy.** Choose only actions that the behavior policy would plausibly take. BCQ (Fujimoto et al., 2019) generates candidate actions from a model of the behavior and picks the best of them. TD3+BC (Fujimoto and Gu, 2021) simply adds a behavior-cloning term to TD3's actor loss, pulling the actor toward the dataset's actions.
- **Pessimistic values.** Conservative Q-learning (Kumar et al., 2020) adds a term that pushes down the values of actions the policy would choose and pushes up those of the dataset's actions, so values off the data are underestimated rather than overestimated.
- **Learn only from what happened.** Implicit Q-learning (Kostrikov et al., 2022) never evaluates actions outside the dataset: it fits values with an asymmetric loss that approximates the best *seen* action, then extracts a policy by weighted imitation of the dataset's own actions.
- **Sequence modeling.** The Decision Transformer (Chen et al., 2021) treats trajectories as sequences and learns to output actions conditioned on the return it is asked to achieve, without value functions at all.

### What data can and cannot do {#data}

An offline method can at best stitch together the good parts of what the data contains; it cannot discover what nobody tried. With expert data, imitation may be enough ([[imitation]]). With mixed data, offline RL can beat every trajectory in it by combining them: the good start of one, the good end of another. Benchmarks such as D4RL (Fu et al., 2020) measure this on datasets from random, mediocre and expert behavior. Evaluating a policy without running it is its own problem, *off-policy evaluation*, tackled with importance sampling ([[importance-sampling]]) or with value functions fit to the data.

Offline training is also a common first stage: learn from logs, then improve with a little careful online interaction (offline-to-online fine-tuning).

## Card

### Idea

Learn a policy from a fixed dataset, with no interaction. The danger is overestimating actions the data never shows, since nothing can correct those estimates; offline methods constrain the policy to the data, make values pessimistic off the data, or learn only from actions actually taken.

::: analogy
Learning to cook only from a stack of other cooks' notes: you can combine their best moves, but an ingredient nobody ever tried is a guess, and a confident guess there is how dinners get ruined.
:::

### The families {#families}

- Policy constraints: BCQ, TD3+BC.
- Conservative values: CQL.
- In-sample learning: IQL.
- Sequence models: Decision Transformer.

### Pseudocode

::: pseudocode
Input: a log $\mathcal D$ of transitions $(s, a, \rew r, s')$, collected by someone else {#log}
Initialize $\val{Q(s, a)}$ for all $s$, $a$
Repeat, a pass over the log at a time:
  For each $(s, a, \rew r, s')$ in $\mathcal D$, in a random order:
    $\val{Q(s,a)} \leftarrow \val{Q(s,a)} + \alp\,\big[\rew r + \gam \max_{a' :\, (s', a') \in \mathcal D} \val{Q(s', a')} - \val{Q(s,a)}\big]$ {#update}
Deploy $\pol{\pi(s)} = \operatorname{arg\,max}_{a :\, (s, a) \in \mathcal D} \val{Q(s, a)}$
:::

Without the restriction to pairs in $\mathcal D$, this is plain Q-learning on the log, and fails.

### Pitfalls

- Running a plain off-policy method on fixed data: the max finds the unsupported actions.
- Expecting to beat the data where it covers nothing.
- Judging a policy by its own value estimates instead of a careful off-policy evaluation.

### Check yourself {#check}

::: question
DQN learns off-policy from a replay memory. Why does it fail on a fixed dataset?
---
Online, the actions its max overrates get tried, and their real returns correct the estimates. Offline, nothing corrects them: the max keeps picking the unsupported actions, the error spreads through the targets, and the policy heads for actions the data never tested.
:::

::: question
How does conservative Q-learning avoid that trap?
---
It adds a penalty that lowers the values of the actions the current policy prefers and raises those of the dataset's own actions. Off the data, the values then err low rather than high, so the policy is not drawn there.
:::

::: question
Can an offline method do better than every trajectory in its dataset?
---
Yes, by stitching: a good beginning from one trajectory and a good ending from another, joined where they pass through the same states. It cannot do better where the data contains nothing useful.
:::
