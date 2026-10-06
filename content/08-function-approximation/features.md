+++
summary = "A feature is a number that describes some aspect of a state; a feature vector describes it with a handful of them. Linear methods estimate a value as a weighted sum of the features, and an update moves each weight in proportion to its feature. Which states share features decides what the learner generalizes to: groups, tiles, polynomials, cosines and bumps are the classic choices."
prereqs = ["why-approximate"]
lab = "walk-features"
sources = [
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., §9.4–9.5", url = "http://incompleteideas.net/book/the-book-2nd.html" },
  { text = "Rahimi & Recht (2007), Random features for large-scale kernel machines, Advances in Neural Information Processing Systems 20", url = "https://proceedings.neurips.cc/paper/2007/file/013a006f03dbc5392effeb8f18fda755-Paper.pdf" },
  { text = "Albus (1975), A new approach to manipulator control: the cerebellar model articulation controller (CMAC), Journal of Dynamic Systems, Measurement, and Control 97", url = "https://doi.org/10.1115/1.3426922" },
  { text = "Hinton (1984), Distributed representations, Technical Report CMU-CS-84-157, Carnegie Mellon University" },
  { text = "Sutton (1996), Generalization in reinforcement learning: successful examples using sparse coarse coding, Advances in Neural Information Processing Systems 8", url = "https://papers.nips.cc/paper/1995/hash/8f1d43620bc6bb580df6e80b0dc05c48-Abstract.html" },
  { text = "Konidaris, Osentoski & Thomas (2011), Value function approximation in reinforcement learning using the Fourier basis, Proceedings of the 25th AAAI Conference on Artificial Intelligence", url = "https://doi.org/10.1609/aaai.v25i1.7903" },
  { text = "Broomhead & Lowe (1988), Multivariable functional interpolation and adaptive networks, Complex Systems 2" },
]

[story]
scene = "line"
env = "walk-1000"
seed = 2
average = 20
formula = '\step{1}{\val{\hat v(s, \mathbf w)} = w_1 x_1(s) + w_2 x_2(s) + \dots + w_d x_d(s)} \step{2}{\qquad \mathbf w \leftarrow \mathbf w + \alp\,\big[\rew{G} - \val{\hat v(S, \mathbf w)}\big]\,\mathbf x(S)}'

[story.runs]
groups = { algorithm = "gradient-mc", features = "groups", cells = 5, alpha = 0.0005, gamma = 1.0, units = 2000, measures = ["ve"], name = "5 groups" }
tiles = { algorithm = "gradient-mc", features = "tiles", tilings = 50, cells = 5, alpha = 0.00001, gamma = 1.0, units = 2000, measures = ["ve"], name = "50 tilings" }
fourier = { algorithm = "gradient-mc", features = "fourier", order = 5, alpha = 0.0001, gamma = 1.0, units = 2000, measures = ["ve"], name = "cosines up to 5" }
+++

## Story

::: step {run = "groups", at = 0, formula = 1}
**A thousand states, five numbers.** Instead of a table of a thousand values, describe each state by a few numbers, its **features**, and estimate its value as a weighted sum of them, formula (1). The simplest features are yes-or-no questions: is the state in the first fifth of the line? In the second? Every state answers yes to exactly one of the five. Five weights to learn, all at 0 for now.
:::

::: step {run = "groups", at = 1, formula = 2}
**One walk, one lesson for many states.** After a walk, Monte Carlo moves each weight by the error times its feature, formula (2): only the weight whose feature is on moves. The strip under the chart shows how far each state's value moved with the last update. The whole fifth moved by the same amount, and nothing else. A lesson learned at one state is applied to every state that shares its features: that is **generalization**.
:::

::: step {run = "groups", checkpoints = [10, 100, 500, 2000]}
**Lessons add up.** After 10, 100, 500 and 2,000 walks, the five weights settle, and the estimate becomes five flat steps along the true values, the dashed line. Within a fifth every state gets the same value, right or wrong: the features decide what the learner can tell apart.
:::

::: step {run = "tiles", at = 1}
**Overlapping features.** Tile coding cuts the line into fifths fifty times, each cutting shifted a little from the last. A state answers yes to fifty features, one per cutting, and a nearby state shares most of them. So an update's footprint is a tent: the closer a state is to the one visited, the more it learns, and states 200 or more away learn nothing.
:::

::: step {run = "fourier", at = 1}
**Global features.** Cosines of rising frequency, $\cos(\pi c s)$ for $c = 0, \dots, 5$, with $s$ scaled to run from 0 to 1. Every feature is nonzero almost everywhere, so every lesson moves every state, some up and some down: the whole strip moves at once. Generalization is no longer local.
:::

::: step {run = "fourier", at = 2000, ghosts = [100]}
After 2,000 walks, six weights trace the thousand values closely; the faint line is where they were after 100. Six numbers, a thousand states.
:::

::: step {run = "fourier", at = 2000, curves = ["groups", "tiles", "fourier"], metric = "ve"}
**The features made the difference.** Averaged over 20 runs, with the same learner and the same walks, the cosines and the tiles end close to the truth, and the five groups stay stuck on their coarse staircase. The Textbook adds polynomials and radial bumps. [Compare them in the Lab](lab:walk-features).
:::

## Textbook

### Linear methods {#linear}

A **feature** is a function $x_i(s)$ that turns a state into a number: “the car's speed”, “is the king in check?”, “how far is the nearest wall?”. A **feature vector** $\mathbf x(s) = (x_1(s), \dots, x_d(s))$ describes the state with $d$ such numbers, and a **linear method** estimates its value as their weighted sum:

$$\val{\hat v(s, \mathbf w)} = \mathbf w^\top \mathbf x(s) = \sum_{i=1}^{d} w_i\, x_i(s). \label{eq-linear}$$

“Linear” refers to the weights, not to the state: the features themselves can be as nonlinear in the state as we like, and usually are. What linearity buys is a gradient that is the feature vector itself, $\nabla \val{\hat v(s, \mathbf w)} = \mathbf x(s)$, whatever the weights. The general update of this part ([[value-error]]) then reads

$$\mathbf w \leftarrow \mathbf w + \alp\,\big[\,u - \val{\hat v(S, \mathbf w)}\,\big]\,\mathbf x(S): \label{eq-linear-update}$$

each weight moves in proportion to how active its feature was. A feature that is 0 in the updated state leaves its weight alone; a large feature moves its weight a lot. And since the visited state's estimate is $\mathbf w^\top \mathbf x(S)$, the update moves it by $\alp\,[u - \hat v]\,\lVert \mathbf x(S) \rVert^2$: the more features are on, the bigger the step a given $\alp$ makes, which is why step sizes for features are set per feature count (\ref{tiles}).

Every other state moves too, by $\alp\,[u - \hat v]\,\mathbf x(s)^\top \mathbf x(S)$: in proportion to the overlap of its features with the visited state's. That overlap, drawn as the footprint strip in the Lab's walk, is what the features make the learner generalize to. Choosing features is choosing it.

::: figure {#fig-shapes}
{{feature-shapes}}
Four ways to describe the 1000 states of the walk. Groups and tiles are 1 on a stretch of states and 0 elsewhere; polynomials and cosines are smooth curves over all of them.
:::

### State aggregation {#aggregation}

The simplest features cut the state space into groups and give each group one feature, equal to 1 for its states and 0 elsewhere. Exactly one feature is on at a time, so the estimate of every state in a group is that group's weight, and an update moves one group as a block. The estimates form a staircase whose steps can be as fine as the groups. Aggregation generalizes perfectly within a group and not at all across groups: neighbors on either side of a boundary learn nothing from each other. A table is the extreme case with one state per group.

### Polynomials {#polynomials}

When a state is a few numbers $s_1, \dots, s_k$, scaled to run from 0 to 1, the products $s_1^{c_1} s_2^{c_2} \cdots s_k^{c_k}$ with each power up to $n$ make $(n+1)^k$ features. They include the constant 1 (a weight that shifts every value), each number alone, and the **interactions** between numbers, which matter whenever the effect of one depends on another (a pole's angle matters differently depending on its angular speed). Polynomials can represent any smooth function in principle, but they learn poorly online: high powers are tiny almost everywhere and huge near the edge, so their weights move at wildly different rates, and the number of features explodes with $k$. In practice they are rarely the first choice.

### Fourier cosines {#fourier}

The **Fourier basis** uses waves instead of powers: for each vector $\mathbf c$ of whole numbers from 0 to $n$,

$$x_{\mathbf c}(s) = \cos\big(\pi\, \mathbf c^\top \mathbf s\big), \qquad \mathbf s \in [0, 1]^k, \label{eq-fourier}$$

again $(n+1)^k$ features. Each one oscillates along the direction $\mathbf c$, the more often the larger the entries of $\mathbf c$; together they can represent any reasonable function on the unit cube, and they are better balanced than powers of $s$, all of the same size. They do well on smooth value functions and struggle with jumps, where the waves ring. Konidaris, Osentoski and Thomas (2011), who brought the basis to reinforcement learning, suggest giving each feature its own step size, smaller for higher frequencies: $\alpha_{\mathbf c} = \alpha / \lVert \mathbf c \rVert$ (and $\alpha$ for $\mathbf c = \mathbf 0$).

::: figure {#fig-basis}
{{basis-study}}
Gradient Monte Carlo on the 1000-state walk with polynomials and with cosines of orders 5, 10 and 20. The cosines learn faster and settle lower; the order matters much less than the kind of feature. Computed live by the Lab; after Sutton & Barto, Figure 9.5.
:::

### Coarse coding {#coarse}

Lay many overlapping regions over the state space, circles for a two-number state, intervals for one, and give each region a binary feature: 1 if the state is inside, 0 if not. This is **coarse coding**. A state turns on the few features whose regions contain it, and two states share as many features as they share regions, so their estimates are tied together more the closer they are. The size and shape of the regions set how far and in which directions generalization reaches: wide circles spread each update far, narrow ones keep it local, long thin regions generalize along one direction more than another. The number of regions sets how finely the estimates can eventually vary. The two are separate controls, as [[generalization]] shows.

### Tile coding {#tiles}

**Tile coding** is coarse coding organized for speed. The state space is partitioned into tiles, a grid, say, and the partition is repeated several times, each copy (a **tiling**) shifted by a fraction of a tile. A state lies in exactly one tile of each tiling, so with $m$ tilings exactly $m$ features are on, always. That has three practical virtues:

- **cheap**: computing the features means finding $m$ tile indices, and the estimate is the sum of $m$ weights, no multiplications;
- **a predictable step**: since $\lVert \mathbf x \rVert^2 = m$ for every state, a step size of $\alp = 1/(m\,k)$ moves a visited state's estimate exactly $1/k$ of the way to its target, the way a table with step $1/k$ would;
- **resolution beyond the tiles**: the tilings' boundaries interleave, so estimates can vary on a scale $m$ times finer than one tile, while each update still generalizes over a whole tile's width.

How the tilings are shifted matters. Shifting every tiling along the diagonal, by the same fraction in each coordinate, lines up their corners and makes generalization stronger along the diagonal than across it; shifting by different multiples per coordinate (1, 3, 5, … times the base fraction, as the Lab does) spreads the corners evenly. Tilings need not be grids either: stripes that ignore one coordinate generalize fully along it, and mixing tilings of different shapes mixes their generalizations. When a grid would need too many tiles, **hashing** folds the huge set of tiles into a fixed-size table of weights, letting distant tiles collide at random: the collisions cost a little accuracy, and memory stops depending on the size of the space. Tile coding goes back to Albus's CMAC (1975) and was popular in reinforcement learning long before neural networks.

::: figure {#fig-touch}
{{touch-tiles}}
Touch a tile. The square is a state made of two numbers, covered by tilings; hover to see the tiles that hold a point, click to teach it. With one tiling, a click lifts one square block; with several, it lifts a blurred patch, sharpest where the tiles overlap.
:::

::: figure {#fig-tiles}
{{tiling-study}}
One tiling of 200-state tiles against fifty such tilings shifted 4 states apart. The same width of generalization, fifty times the resolution: the fifty tilings learn about as fast and settle much lower. Computed live by the Lab; after Sutton & Barto, Figure 9.10.
:::

### Radial basis functions {#rbf}

Coarse coding with soft edges: each feature is a bump centered at a point $\mathbf c_i$ of the state space,

$$x_i(s) = \exp\!\Big(-\frac{\lVert \mathbf s - \mathbf c_i \rVert^2}{2\sigma_i^2}\Big), \label{eq-rbf}$$

1 at the center and fading smoothly with distance, at a rate set by the width $\sigma_i$. The estimate then varies smoothly instead of in steps. The price is that every feature is nonzero everywhere and must be computed and multiplied, and in more than a few dimensions the number of bumps needed to cover the space grows quickly. Letting the learner move the centers and widths too gives an RBF network, a nonlinear method (Broomhead and Lowe, 1988).

### Choosing features {#choosing}

Features are where knowledge of the problem enters. Good features make states that should have similar values look similar, and states that should differ look different; they include interactions when the numbers act together; they are scaled so that no feature dominates the step. A few rules of thumb:

- smooth values over a few continuous numbers: cosines or tile coding;
- a problem with many numbers, only some of which interact: several tilings, each over a small subset of the numbers;
- discontinuities (a cliff, a wall): local features (tiles, groups) placed so a boundary falls where the jump is.
- many numbers and no idea which matter: a large set of **random features**, such as cosines of random combinations of the numbers (Rahimi & Recht, 2007), with a linear learner on top; cheap, and surprisingly hard to beat.

When nobody knows the right features, the learner can learn them: a neural network is a linear method on top of features that are themselves adjusted by gradient descent ([[neural-networks]], [[dqn]]).

### Historical remarks {#history}

Coarse coding takes its name from Hinton (1984), who described distributed representations of this kind as a model of neural codes. Tile coding is Albus's CMAC (1975), designed for robot arm control, and Sutton (1996) showed it working well with SARSA on several control problems where it had been thought to fail. Polynomial approximation of value functions goes back to Bellman and his colleagues in the early 1960s; the Fourier basis was introduced to reinforcement learning by Konidaris, Osentoski and Thomas (2011); radial basis function networks are due to Broomhead and Lowe (1988).

## Card

### Idea

Describe a state by a few numbers, its features, and estimate its value as their weighted sum. An update moves each weight in proportion to its feature; every state that shares active features moves along. Which states share features decides what the learner generalizes to.

::: analogy
Pricing a house from its size, number of rooms and distance to the station instead of remembering the price of every house: a new house gets a price at once, from what it has in common with the others.
:::

### Kinds of features {#kinds}

| kind | features on at once | generalizes | good for |
| --- | --- | --- | --- |
| groups (aggregation) | 1 | within a group only | quick, coarse first tries |
| tile coding | one per tiling | over a tile's width, finely | a few continuous numbers |
| polynomials | all | everywhere | rarely the best online |
| Fourier cosines | all | everywhere, balanced | smooth values |
| radial basis | all, fading | locally, smoothly | smooth values, few dimensions |

### The estimate {#formula}

$$\val{\hat v(s, \mathbf w)} = \mathbf w^\top \mathbf x(s), \qquad \nabla \val{\hat v(s, \mathbf w)} = \mathbf x(s)$$

An update moves the visited state by $\alp\,\delta\,\lVert\mathbf x\rVert^2$ and any other state by $\alp\,\delta\,\mathbf x(s)^\top \mathbf x(S)$.

### Why it matters {#why}

- The features decide what is learned from each update, and so how fast. [See it](lab:walk-features)
- With tile coding, $m$ tilings give $m$ times the resolution at the same generalization width.
- Linear methods are simple, fast and well understood; their theory carries over to features learned by networks.

### Pitfalls

- Features of very different sizes: their weights learn at very different speeds.
- Forgetting the constant feature, or the interactions between numbers.
- Keeping $\alp$ when the number of active features changes: with tile coding, divide by the number of tilings.

### Check yourself {#check}

::: question
With 8 tilings and step size $\alp = 0.1/8$, how far does one update move the visited state's estimate toward its target?
---
A tenth of the way: 8 features are on, each weight moves by $\alp\,\delta = 0.1\,\delta/8$, and the estimate, their sum, moves by $8 \times 0.1\,\delta/8 = 0.1\,\delta$.
:::

::: question
Why does an update with polynomial features change the estimate of every state?
---
Every power of $s$ is nonzero almost everywhere, so every weight moves, and every state's estimate depends on every weight.
:::

::: question
What is the advantage of several shifted tilings over one tiling with smaller tiles?
---
Each update still generalizes over a whole tile's width, so learning stays fast, while the interleaved boundaries let the estimates vary on a much finer scale.
:::
