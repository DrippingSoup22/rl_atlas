+++
summary = "Q-learning with a neural network for the action values, made to work by two additions: a replay memory that feeds the updates random minibatches of old transitions, and a target network that holds the targets still between copies. With one network and one set of knobs it learned 49 Atari games from their pixels."
change = "Replace Q-learning's table by a neural network, trained by semi-gradient steps on random minibatches from a replay memory, with targets computed by a copy of the network refreshed every few hundred steps."
prereqs = ["q-learning", "neural-networks", "experience-replay", "target-network"]
lab = "dqn-cartpole"
sources = [
  { text = "Mnih, Kavukcuoglu, Silver, Graves, Antonoglou, Wierstra & Riedmiller (2013), Playing Atari with deep reinforcement learning, NIPS Deep Learning Workshop", url = "https://arxiv.org/abs/1312.5602" },
  { text = "Mnih et al. (2015), Human-level control through deep reinforcement learning, Nature 518", url = "https://doi.org/10.1038/nature14236" },
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., §16.5", url = "http://incompleteideas.net/book/the-book-2nd.html" },
  { text = "Riedmiller (2005), Neural fitted Q iteration, ECML", url = "https://doi.org/10.1007/11564096_32" },
  { text = "Henderson, Islam, Bachman, Pineau, Precup & Meger (2018), Deep reinforcement learning that matters, AAAI", url = "https://arxiv.org/abs/1709.06560" },
  { text = "Barto, Sutton & Anderson (1983), Neuronlike adaptive elements that can solve difficult learning control problems, IEEE Transactions on Systems, Man, and Cybernetics 13", url = "https://doi.org/10.1109/TSMC.1983.6313077" },
]

[story]
scene = "cartpole"
env = "cartpole"
formula = '''\step{1}{\val{\hat q(s, a, \mathbf w)} \leftarrow \rew{r} + \gam \max_{a'} \val{\hat q(s', a', \mathbf w^-)} \qquad} \step{2}{(s, a, \rew{r}, s') \sim \mathcal D}'''

[story.runs]
learn = { recording = "dqn-cartpole", name = "DQN" }
noreplay = { recording = "dqn-cartpole-no-replay", name = "without replay" }
notarget = { recording = "dqn-cartpole-no-target", name = "without a target network" }
+++

## Story

::: step {run = "learn", at = 0, map = "action"}
**A pole on a cart**, and two moves: push the cart left or right. Every step the pole stays within 12° of upright and the cart on its track pays $\rew{+1}$, for at most 500 steps. The state is four numbers: where the cart is, how fast it moves, the pole's angle and how fast it turns. On the right is a slice of that state space, the cart at rest in the middle: the pole's angle across, its spin up and down. Before any learning, a network of 4,610 random weights gives each push a value. The values are small and mean nothing, but the network does prefer one push in every state, and here it is the same one everywhere: push left.
:::

::: step {run = "learn", at = 1, formula = 2}
**5,000 steps later.** ε started at 1, every move random, and has fallen to 0.76. Every step went into the replay memory. After 1,000 steps of play, the network started learning: every 4 steps, one gradient step on a batch of 128 transitions drawn from the memory at random, so 1,000 steps by now. Played greedily, from the same start as every test, the pole now stays up for 87 steps.
:::

::: step {run = "learn", at = 2, formula = 1}
**After 10,000 steps: 311.** Each target comes from the target network, a copy refreshed every 500 steps, so the values climb in stages: they now run from $-5.3$ to $21.1$. The map has split in two. Where the pole leans or spins to the right, the network pushes right, under it; where it leans left, it pushes left. That is the whole trick of balancing, found from the reward alone.
:::

::: step {run = "learn", at = 10}
**Then a long plateau.** ε reaches its floor, 0.05, after 20,000 steps, and for the next 26 blocks the test episodes last between 172 and 353 steps. The pole is no longer the problem. In 23 of those 26 episodes, the cart drifts until it runs off the end of the track. The map shows why: it is a slice with the cart at rest in the middle. The network has learned to keep the pole up, but where the cart is barely matters to it yet. Running off the track costs only the steps that are lost, 200 steps away, and $\gam^{200} \approx 0.13$ makes that hard to see.
:::

::: step {run = "learn", at = 30}
**Block 30, after 150,000 steps: 500 steps**, the most an episode lasts. The values now reach $99.0$, close to the most a state can be worth to this learner. Episodes are cut at 500 steps, but the learner bootstraps through the cut, as it should, so it values a pole that stays up forever: $1 + 0.99 + 0.99^2 + \dots = 100$. The cart still wanders, as far as 2.17 m from the middle, but it now stops short of the end.
:::

::: step {run = "learn", at = 40, map = "value"}
**At the end, block 40:** still 500, with the cart now within 16 cm of the middle the whole time. The highest value on the map is $101.6$, above the 100 that any state can be worth. The max in the target picks up the estimates' noise, and the copy passes it on: on 15 of the 20 seeds, the average of the largest values goes past 100 for some blocks, by up to 29%. A modest overestimation here, a large one on harder games: [[dqn-extensions|Double DQN]] is the cure.
:::

::: step {run = "learn", at = 40, curves = ["learn", "noreplay", "notarget"], metric = "return"}
**Take away either remedy.** Each line averages 20 seeds, trained with the same settings except one. Without replay, each batch is the last 128 steps. The runs still learn, but they wobble: the last block averages 374 steps, and only 7 of the 20 runs end well, averaging 450 or more over their last four blocks. Without a target network, the targets come from the weights being changed: no run of 20 ends well, and the last block averages 140. With both, 16 of the 20 end well, and the last block averages 480.
:::

## Textbook

### From Q-learning to DQN {#why}

Q-learning learns $\val{q_*}$ from its own experience, off-policy and without a model ([[q-learning]]). With a table it needs one entry per state and action, which rules out any problem whose states are images, or even just four real numbers. The obvious fix is to replace the table by a neural network $\val{\hat q(s, a, \mathbf w)}$ with one output per action, and to update it with the semi-gradient step on the TD error ([[semi-gradient-sarsa]]). On its own, this fails more often than not. Consecutive transitions are correlated, and a network forgets ([[experience-replay]]); the target moves with every update, and a network chases it ([[target-network]]). With function approximation, bootstrapping and off-policy learning all present, nothing guarantees convergence ([[deadly-triad]]).

DQN keeps Q-learning's target and adds two remedies, one for each problem. Neither restores a guarantee, but together they made the method work across a whole family of hard problems, with one network and one set of knobs.

### The algorithm {#algorithm}

::: algorithm {#alg-dqn} Deep Q-network (DQN)
Parameters: memory size $N$, minibatch size $B$, steps between updates $k$, steps between copies $C$, step size $\alp$, an exploration schedule for $\eps$
Initialize the network's weights $\mathbf w$ at random, the target network's $\mathbf w^- \leftarrow \mathbf w$, an empty memory $\mathcal D$
Repeat for each step $t$:
  In $S$, choose $A$ ε-greedy on $\val{\hat q(S, \cdot, \mathbf w)}$; take it, observe $\rew{R}$, $S'$
  Store $(S, A, \rew{R}, S')$ in $\mathcal D$, dropping the oldest transition once there are $N$
  Every $k$ steps: draw $B$ transitions $(s_j, a_j, \rew{r_j}, s'_j)$ from $\mathcal D$ at random
    $y_j = \rew{r_j}$ if $s'_j$ is terminal, else $\rew{r_j} + \gam \max_{a'} \val{\hat q(s'_j, a', \mathbf w^-)}$
    Take a gradient step on $\frac1B \sum_j \big(y_j - \val{\hat q(s_j, a_j, \mathbf w)}\big)^2$
  Every $C$ steps: $\mathbf w^- \leftarrow \mathbf w$
  $S \leftarrow S'$, or a new episode's start if $S'$ is terminal
:::

The gradient step treats $y_j$ as a constant: no gradient flows through the target, as in every semi-gradient method. The loss is the squared TD error averaged over the batch, so one step moves $\val{\hat q(s_j, a_j)}$ toward its target for every transition in it, and drags along the estimates of all the states that look like them.

### On Atari {#atari}

The 2015 network read the screen itself. Each frame was shrunk to $84 \times 84$ gray pixels, and the last four frames were stacked, so the input showed motion. Three convolutional layers and a hidden layer of 512 units fed one output per joystick action, up to 18. The agent chose an action every fourth frame and repeated it in between. Rewards were clipped to $-1$, $0$ or $+1$, so that one step size fit every game's score scale. The TD error was clipped to $[-1, 1]$ in the gradient, which amounts to the Huber loss. The replay memory held the last million transitions, the target network was copied every 10,000 updates, and ε fell from 1 to 0.1 over the first million frames.

With these settings unchanged, the same algorithm learned 49 games, from Pong to Breakout and Space Invaders. It beat the best earlier learning methods on 43 of them and reached at least three quarters of a professional tester's score on 29. It failed where rewards are rare and need a long chain of exact moves before the first one: on Montezuma's Revenge it scored zero, because random exploration almost never finds the first key.

### On CartPole {#cartpole}

CartPole is far smaller: four numbers instead of a screen, two actions, and episodes of at most 500 steps. It is old, too: Barto, Sutton and Anderson (1983) balanced a pole with an actor and a critic made of two neuron-like elements. The guide's recorded DQN uses two hidden layers of 64 rectifier units and Adam. Its memory holds 10,000 steps, and it takes one step on 128 transitions every 4 steps. The target is copied every 500 steps, and ε falls from 1 to 0.05 over the first 20,000 steps. It uses the plain squared error, not the Huber loss: the Huber loss caps the pull of any error larger than 1, and CartPole's values have to climb to about 100. Over 20 seeds each, 16 runs end well with the squared error and none with the Huber loss, whose best run ends at 445 steps.

Those numbers were not the first ones tried. With a memory of 100,000 steps, three of 20 seeds ended below 160 steps, and the [[experience-replay|replay entry]] shows how the size of the memory changes the odds. Every knob behaves this way. Over 20 seeds per value, the step size gives:

| step size $\alp$ | runs that end well |
| --- | --- |
| $10^{-4}$ | 16 of 20 |
| $2.5 \cdot 10^{-4}$ | 13 of 20 |
| $5 \cdot 10^{-4}$ | 16 of 20 |
| $10^{-3}$ | 11 of 20 |
| $2.5 \cdot 10^{-3}$ | 3 of 20 |

A run ends well here when its last training episodes keep the pole up for 450 steps or more on average. A smaller step does no worse within 200,000 steps. A larger one keeps knocking over what it learned: at $2.5 \cdot 10^{-3}$, every run averages 450 steps per episode for a while, and 17 of them end between 205 and 412. With a table, a step size that is a little off costs some speed; with a network it changes the odds that the run works at all. Deep RL results are reported over many seeds for this reason (Henderson et al., 2018), and the Lab's sweeps show these odds for every knob.

### Why it works, and when it does not {#theory}

No theorem covers DQN. With the target network frozen, each period of $C$ steps is a regression toward the Bellman optimality backup of the frozen copy, one step of approximate value iteration. Fitted Q iteration has guarantees when the regression error stays small at every step. DQN fits only a few gradient steps per period, on whatever the memory holds, so those conditions are never checked. In practice its values can overshoot the truth, as the story's last steps show, and a run can lose for a while what it learned and then find it again. The recordings show such dips even in runs that end well.

### Historical remarks {#history}

Networks had been trained with Q-learning before, for instance in neural fitted Q iteration (Riedmiller, 2005). That method fit the network on a fixed batch of transitions, with targets frozen during each fit. DQN made the fitting incremental. Mnih and colleagues presented it on seven Atari games in 2013, without a target network, and on 49 games, with one, in 2015. Most of deep reinforcement learning since then builds on it, or reacts to it: its value side in [[dqn-extensions]], and its rivals in the policy-gradient methods ([[a2c]], [[ppo]]).

## Card

### Idea

Q-learning with a network for $\val{\hat q}$, trained on random minibatches from a replay memory, with targets from a frozen copy of the network that is refreshed every $C$ steps.

::: analogy
A student who reviews a shuffled box of past exercises every evening, and checks the answers against last week's notes rather than against today's half-learned ones.
:::

### The update {#update}

$$\mathbf w \leftarrow \mathbf w + \alp\,\frac1B \sum_{j=1}^{B} \Big(\rew{r_j} + \gam \max_{a'} \val{\hat q(s'_j, a', \mathbf w^-)} - \val{\hat q(s_j, a_j, \mathbf w)}\Big)\,\nabla_{\mathbf w}\val{\hat q(s_j, a_j, \mathbf w)}$$

over a minibatch drawn from the replay memory; $\mathbf w^- \leftarrow \mathbf w$ every $C$ steps.

### One change from Q-learning {#change}

A network instead of the table, trained by semi-gradient steps on minibatches drawn at random from a replay memory, with targets from a copy of the network that changes only every $C$ steps.

### Backup diagram {#backup}

{{backup q-learning}}

The backup is Q-learning's: a sampled transition, and the best next action's value. What changes is where the transition comes from, the replay memory rather than the step just taken, and whose values the max reads: the target network's.

### Pseudocode

::: pseudocode
Parameters: memory $N$, batch $B$, update every $k$ steps, copy every $C$ steps, step size $\alp$, schedule for $\eps$
Initialize $\mathbf w$ at random; $\mathbf w^- \leftarrow \mathbf w$; memory $\mathcal D$ empty
Repeat for each step:
  Choose $A$ in $S$ ε-greedy on $\val{\hat q(S, \cdot, \mathbf w)}$; take it, observe $\rew{R}$, $S'$
  Store $(S, A, \rew{R}, S')$ in $\mathcal D$
  Every $k$ steps: sample $B$ transitions from $\mathcal D$
    $y_j \leftarrow \rew{r_j} + \gam \max_{a'} \val{\hat q(s'_j, a', \mathbf w^-)}$ (just $\rew{r_j}$ at the end)
    Gradient step on $\frac1B \sum_j (y_j - \val{\hat q(s_j, a_j, \mathbf w)})^2$
  Every $C$ steps: $\mathbf w^- \leftarrow \mathbf w$
  $S \leftarrow S'$
:::

### Perks

- Learns action values from raw, high-dimensional input: pixels, joint angles, anything a network can read.
- Off-policy: every transition is reused many times from the memory.
- One network, one set of knobs, across 49 Atari games.

### Flaws

- No guarantee: all three parts of the deadly triad are present.
- Overestimates values: the max picks up noise.
- Sensitive to its knobs and its seed. On CartPole a step size five times too large drops the runs that end well from 16 in 20 to 3.
- Only discrete actions: the max over $a'$ needs a short list of actions.
- Needs many samples: the recorded run first kept the pole up for 500 steps after 150,000 steps of training; the recorded PPO, after 15,000.

### Knobs

| Knob | Too low | Too high |
| --- | --- | --- |
| memory $N$ | replay in name only, correlated batches | keeps fitting a long-gone policy |
| copy period $C$ | targets move again | values crawl, one backup per copy |
| step size $\alp$ | slow | knocks over what it learned |
| $\eps$ schedule | too little exploration early | too many random moves late |
| batch $B$ | noisy steps | slow, and fewer updates per sample |

### Pitfalls

- A transition that ends the episode must have target $\rew{r}$ alone; one cut by a time limit should still bootstrap.
- Rewards of very different sizes call for one step size per game: Atari clipped them; elsewhere, scale them.
- Judging a setting by one seed: with a 100,000-step memory, the same settings ended anywhere from 116 to 500 steps, depending on the seed.
- Reading the training curve as the policy's quality: training episodes include the ε moves, and test episodes do not.

### Check yourself {#check}

::: question
Which of the deadly triad's three parts does DQN have, and what does it do about them?
---
All three: a network approximates the values, its targets bootstrap, and it learns off-policy from a memory filled by older ε-greedy policies. It keeps them and adds two stabilizers: replay breaks the correlations in the data, and the target network holds the targets still between copies. That makes divergence rare in practice, though nothing rules it out.
:::

::: question
Why does the story's DQN keep the pole up for hundreds of steps yet run off the track?
---
Falling off the track happens hundreds of steps after the drift that causes it begins. With $\gam = 0.99$, a loss 200 steps away is discounted by $0.99^{200} \approx 0.13$, so the values barely tell drifting from not drifting until the network has learned the rest. The pole, whose fall is a few steps away, is learned first.
:::

::: question
Why are DQN's values on CartPole bounded by 100, and what does a value of 101.6 mean?
---
Each step pays at most 1, and the learner bootstraps through the 500-step time limit, which the state does not show, so to it a state is worth at most $1 + \gam + \gam^2 + \dots = 1/(1 - 0.99) = 100$. An estimate above that is an overestimate, the bias the max in the target introduces.
:::
