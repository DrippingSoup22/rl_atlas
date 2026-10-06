+++
summary = "A neural network is a linear method on top of features it learns itself: layers of weighted sums, each bent by a simple nonlinearity, trained by gradient steps that the chain rule computes. It can fit far more than hand-made features, at the price of the guarantees linear methods had."
prereqs = ["features", "semi-gradient-td", "deadly-triad"]
lab = "ppo-cartpole"
sources = [
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., §9.7", url = "http://incompleteideas.net/book/the-book-2nd.html" },
  { text = "Dohare, Hernandez-Garcia, Lan, Rahman, Mahmood & Sutton (2024), Loss of plasticity in deep continual learning, Nature 632", url = "https://doi.org/10.1038/s41586-024-07711-7" },
  { text = "Nikishin, Schwarzer, D'Oro, Bacon & Courville (2022), The primacy bias in deep reinforcement learning, Proceedings of the 39th International Conference on Machine Learning", url = "https://arxiv.org/abs/2205.07802" },
  { text = "Goodfellow, Bengio & Courville (2016), Deep Learning, MIT Press, chapters 6 and 8", url = "https://www.deeplearningbook.org/" },
  { text = "Rumelhart, Hinton & Williams (1986), Learning representations by back-propagating errors, Nature 323", url = "https://doi.org/10.1038/323533a0" },
  { text = "Cybenko (1989), Approximation by superpositions of a sigmoidal function, Mathematics of Control, Signals and Systems 2", url = "https://doi.org/10.1007/BF02551274" },
  { text = "Kingma & Ba (2015), Adam: a method for stochastic optimization, ICLR", url = "https://arxiv.org/abs/1412.6980" },
  { text = "He, Zhang, Ren & Sun (2015), Delving deep into rectifiers, ICCV", url = "https://arxiv.org/abs/1502.01852" },
  { text = "Tesauro (1995), Temporal difference learning and TD-Gammon, Communications of the ACM 38", url = "https://doi.org/10.1145/203330.203343" },
]

[story]
scene = "cartpole"
env = "cartpole"
formula = '''\step{1}{\mathbf h_1 = \varphi(W_1 s + \mathbf b_1) \qquad \mathbf h_2 = \varphi(W_2 \mathbf h_1 + \mathbf b_2) \qquad} \step{2}{\val{\hat q(s, \cdot\,, \mathbf w)} = W_3 \mathbf h_2 + \mathbf b_3}'''

[story.runs]
learn = { recording = "dqn-cartpole", name = "DQN's network" }
+++

## Story

::: step {run = "learn", at = 0, formula = 1}
**A network reads a state.** CartPole's state is four numbers: the cart's position and speed, the pole's angle and spin. Two hidden layers of 64 units turn them into features: each unit adds up its inputs with its own weights and bends the sum with $\max(0, z)$. Nobody chooses these features. Like all 4,610 weights, they start random.
:::

::: step {run = "learn", at = 0, formula = 2, map = "action"}
**Two outputs, one per push.** The last layer is a linear method on the features: one weighted sum for “push left”, one for “push right”. Before training the values are small, but not noisy: a network is a smooth function, so nearby states get nearby values. This one happens to prefer pushing left everywhere on the map.
:::

::: step {run = "learn", at = 1, formula = 2}
**One update moves everything.** Each gradient step changes weights that every state uses. After 1,000 of them, the whole map has moved, its values now between 1.9 and 9.9, including states the cart has never visited. A table would only have changed the states it saw.
:::

::: step {run = "learn", at = 2}
**Features appear.** After 2,250 steps of learning the map has a shape: low values in the two corners where the pole leans one way and spins further that way. No one told the network that this is dangerous: the hidden units found it because it predicts the targets.
:::

::: step {run = "learn", at = 40}
**After 200,000 steps** those corners are still the worst, about 44 against 99 in the middle, and the pole stays up for the full 500 steps. Tile coding would have needed someone to choose the tiles. The network grew its own features, with the same gradient steps that fit the values.
:::

## Textbook

### Features that learn {#why}

A linear method estimates a value as a weighted sum of features, $\val{\hat v(s, \mathbf w)} = \mathbf w^\top \mathbf x(s)$ ([[features]]). Learning only moves the weights; the features are fixed, chosen by hand, and they decide what can be learned at all. Tiles of the wrong width blur what matters, polynomials of too low an order cannot bend where the truth bends, and in a world of many dimensions (an image, a robot's joints) no hand-made set of features is both small enough and good enough.

A neural network learns its features too. Each hidden unit computes a weighted sum of its inputs and bends it with a fixed nonlinear function; that bent sum is a feature, and its weights are learned like all the others. The output is a linear method on top of the last layer of such features. Everything that follows comes from this one change: the estimate is still a smooth function of the weights, so a gradient step still improves it, but the gradient is no longer just the feature vector.

### Layers {#mlp}

::: definition {#def-mlp} Multilayer perceptron
A layer maps a vector $\mathbf h$ to $\varphi(W\mathbf h + \mathbf b)$: a matrix of weights $W$, a vector of biases $\mathbf b$, and a nonlinearity $\varphi$ applied to each component, usually the rectifier $\varphi(z) = \max(0, z)$ (ReLU) or $\tanh z$. A multilayer perceptron stacks layers and ends with a linear one. With two hidden layers and one output per action,
$$\mathbf h_1 = \varphi(W_1 s + \mathbf b_1), \qquad \mathbf h_2 = \varphi(W_2 \mathbf h_1 + \mathbf b_2), \qquad \val{\hat q(s, \cdot\,, \mathbf w)} = W_3 \mathbf h_2 + \mathbf b_3, \label{net}$$
where $\mathbf w$ collects every weight and bias.
:::

One output per action means one forward pass gives every action value of a state, which is what a greedy choice and a Q-learning target need. The same body serves other ends: one output gives a critic $\val{\hat v(s, \mathbf w)}$, and one output per action passed through a softmax gives a policy ([[policy-parameterization]]).

The recorded runs in this guide use two hidden layers of 64 units. On CartPole, whose state has 4 numbers and which has 2 actions, DQN's network has $4 \cdot 64 + 64 + 64 \cdot 64 + 64 + 64 \cdot 2 + 2 = 4{,}610$ weights: more than tile coding would need for 4 dimensions, but none of them chosen by hand.

::: remark
Without the nonlinearity, the layers would collapse: a product of matrices is one matrix, and the network would be a linear method again. With it, a network with one hidden layer of enough units can approximate any continuous function on a bounded set as closely as asked (Cybenko, 1989). That says nothing about how many units, or whether gradient steps find the weights; in practice depth helps, because a deep network can reuse the features of one layer in many features of the next.
:::

### Gradients by the chain rule {#backprop}

The learning rule does not change. Semi-gradient TD moves the weights along the gradient of the estimate it updated ([[semi-gradient-td]]):
$$\mathbf w \leftarrow \mathbf w + \alp\,\err{\delta}\,\nabla_{\mathbf w} \val{\hat q(S, A, \mathbf w)}. \label{update}$$
For a linear method that gradient is the feature vector. For a network it is computed by **backpropagation**, the chain rule applied layer by layer from the output back. Write $\mathbf z_2 = W_2 \mathbf h_1 + \mathbf b_2$ for the second layer before its nonlinearity, and $\mathbf e_A$ for the output that holds action $A$. Then
$$\frac{\partial \hat q}{\partial W_3} = \mathbf e_A \mathbf h_2^\top, \qquad \mathbf g_2 = \frac{\partial \hat q}{\partial \mathbf z_2} = \big(W_3^\top \mathbf e_A\big) \odot \varphi'(\mathbf z_2), \qquad \frac{\partial \hat q}{\partial W_2} = \mathbf g_2\, \mathbf h_1^\top,$$
and the same step once more gives $W_1$: each layer's gradient is the signal arriving from above, times the derivative of its nonlinearity, times what entered the layer. One backward pass costs about as much as two forward passes, whatever the number of weights. (The recorder computes these by hand and checks them against numerical differences.)

### Batches and step sizes {#adam}

Networks are trained on **minibatches**: the gradient of the average loss over a few dozen to a few hundred examples, which is less noisy than one and cheaper than all. With many weights of different scales one step size rarely fits all of them, and most deep RL uses **Adam** (Kingma & Ba, 2015), which keeps for each weight a running average of its gradient $\mathbf m$ and of its square $\mathbf u$,
$$\mathbf m \leftarrow \beta_1 \mathbf m + (1 - \beta_1)\,\mathbf g, \qquad \mathbf u \leftarrow \beta_2 \mathbf u + (1 - \beta_2)\,\mathbf g^2, \qquad \mathbf w \leftarrow \mathbf w - \alp\,\frac{\hat{\mathbf m}}{\sqrt{\hat{\mathbf u}} + \epsilon},$$
where the hats undo the averages' pull toward their starting value 0. Each weight then moves about $\alp$ per step when its gradient is consistent, and less when it flips sign. The recorded runs use Adam with $\beta_1 = 0.9$, $\beta_2 = 0.999$, and a cap on the size of the whole gradient.

The weights start small and random: identical weights would compute identical features forever. Their spread is scaled to the number of inputs of each layer (He et al., 2015, for rectifiers), so that signals neither vanish nor explode on their way through the layers.

### What a network changes {#changes}

With tiles, an update touched the few weights of the tiles a state lit, and the estimates of states far away did not move. In a network every weight takes part in every estimate, so one update moves the estimates of all states, in ways that depend on everything learned so far. That is what lets a network generalize from few examples, and it is also what makes it fragile in reinforcement learning:

- **The data are not independent.** Consecutive steps of an episode look alike, so a run of updates all pull in the same direction, and the network can drift away from what it learned elsewhere and forget it.
- **The targets move.** A bootstrapped target uses the same network that is being updated, so every step changes the target it is chasing.
- **The guarantees are gone.** Even on-policy, semi-gradient TD with a nonlinear approximator can diverge; off-policy, the deadly triad is fully armed ([[deadly-triad]]).
- **Plasticity fades.** Trained for long on a stream whose targets keep changing, a network slowly loses the ability to learn new things: units stop responding, weights grow, and later updates achieve less (Dohare et al., 2024). Resetting part of the network now and then, while keeping the replay memory, is a simple remedy (Nikishin et al., 2022).

[[dqn|DQN]] is Q-learning with a network plus two remedies aimed at the first two points: [[experience-replay]] breaks up the correlations, and a [[target-network]] holds the targets still for a while. Neither restores a guarantee; together they made learning from raw pixels work.

The other cost is sensitivity: a network adds knobs (layers, widths, learning rate, batch size) and its results depend on them, and on the seed, more than a table's did. The Lab's sweeps of the recorded runs show how often a setting ends well, and how quickly that drops when one knob moves.

### Historical remarks {#history}

Rosenblatt's perceptron (1958) learned one layer of weights. Backpropagation, found several times, became widely known with Rumelhart, Hinton and Williams (1986). Networks met reinforcement learning early: Tesauro's TD-Gammon (1992 to 1995) trained a network with one hidden layer by TD($\lambda$) through self-play and reached the level of the best human backgammon players. Attempts to repeat that success elsewhere were mostly disappointing, until DQN (Mnih et al., 2013, 2015) learned 49 Atari games from pixels with one architecture and one set of knobs.

## Card

### Idea

A neural network is a feature maker and a linear method in one: layers of weighted sums, each bent by a simple nonlinearity. Gradient steps tune all the weights at once, so the features are learned instead of designed, and the chain rule computes the gradient one layer at a time.

::: analogy
Hand-made features are a fixed set of lenses chosen before looking; a network grinds its own lenses while it learns what to look at.
:::

### The network {#formula}

$$\mathbf h_1 = \varphi(W_1 s + \mathbf b_1), \qquad \mathbf h_2 = \varphi(W_2 \mathbf h_1 + \mathbf b_2), \qquad \val{\hat q(s, \cdot\,, \mathbf w)} = W_3 \mathbf h_2 + \mathbf b_3$$

trained with the same semi-gradient step as a linear method, $\mathbf w \leftarrow \mathbf w + \alp\,\err{\delta}\,\nabla_{\mathbf w}\val{\hat q(S, A, \mathbf w)}$, the gradient computed by backpropagation, usually over a minibatch and with Adam's step sizes.

### Why it matters {#why}

- No features to design: the same network learns CartPole's angles and an Atari screen's pixels.
- One forward pass gives every action's value, or a policy, or a critic.
- Generalization is global: one update moves every estimate, which helps when the data are scarce and hurts when they are correlated.

### Pitfalls

- Without the nonlinearity the layers collapse into one linear map.
- Identical starting weights stay identical: every hidden unit learns the same feature.
- Inputs of very different scales (meters and radians per second) slow learning; scale them first.
- Correlated data and moving targets can make a network forget what it knew, or diverge: hence replay and target networks.
- The learning rate and the seed matter more than with tables: judge a setting over several seeds.

### Check yourself {#check}

::: question
Why does a network with two layers and no nonlinearity add nothing to a linear method?
---
$W_2(W_1 s + \mathbf b_1) + \mathbf b_2 = (W_2 W_1) s + (W_2 \mathbf b_1 + \mathbf b_2)$: one matrix and one bias, a linear function of the state, so it can represent nothing that one linear layer could not.
:::

::: question
In semi-gradient TD with a network, what replaces the feature vector $\mathbf x(S)$ of the linear update?
---
The gradient $\nabla_{\mathbf w}\val{\hat q(S, A, \mathbf w)}$, computed by backpropagation. For a linear method it is exactly the feature vector; for a network it depends on the current weights, so the “features” the update uses change as the network learns.
:::

::: question
Why is learning from consecutive steps of one episode harder for a network than for a table?
---
Consecutive states look alike and their targets pull the same way. A table only changes the entries it visits; a network changes all its estimates with every step, so a long run of similar updates drags the whole function toward one region's targets and can undo what it learned elsewhere.
:::
