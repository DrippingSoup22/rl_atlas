+++
summary = "A table keeps one number per state, and learns about a state only by visiting it. Most problems have far too many states for that, or infinitely many. The way out: compute values from a few adjustable weights shared by many states, so that what is learned in one state carries over to similar ones."
prereqs = ["value-functions", "mc-prediction", "td0", "dp-limits"]
lab = "walk-aggregation"
sources = [
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., introduction to Part II and §9.1–9.2", url = "http://incompleteideas.net/book/the-book-2nd.html" },
  { text = "Bellman, Kalaba & Kotkin (1963), Polynomial approximation: a new computational technique in dynamic programming, Mathematics of Computation 17", url = "https://doi.org/10.2307/2003997" },
  { text = "Samuel (1959), Some studies in machine learning using the game of checkers, IBM Journal of Research and Development 3", url = "https://doi.org/10.1147/rd.33.0210" },
  { text = "Tesauro (1995), Temporal difference learning and TD-Gammon, Communications of the ACM 38", url = "https://doi.org/10.1145/203330.203343" },
  { text = "Bertsekas & Tsitsiklis (1996), Neuro-Dynamic Programming, Athena Scientific" },
]

[story]
scene = "line"
env = "walk-1000"
seed = 5
average = 100
formula = '\step{1}{\text{table: } \val{V(s)} \text{, one number per state}} \step{2}{\qquad \text{features: } \val{\hat v(s, \mathbf w)} = \mathbf w^\top \mathbf x(s) \text{, ten numbers for all}}'

[story.runs]
table = { algorithm = "gradient-mc", features = "table", alpha = 0.1, gamma = 1.0, units = 100, measures = ["ve"], name = "a table (1000 numbers)" }
groups = { algorithm = "gradient-mc", features = "groups", cells = 10, alpha = 0.005, gamma = 1.0, units = 100, measures = ["ve"], name = "10 groups (10 numbers)" }
+++

## Story

::: step {run = "table", at = 0, footprint = false}
**A walk with a thousand states.** The agent starts in the middle, at state 500, and every step jumps up to 100 states left or right, at random. Leaving on the right pays $\rew{+1}$, leaving on the left $\rew{-1}$. The dashed line is the true value of each state; the gray hill underneath is how much time the walk spends in each, highest in the middle where it starts.
:::

::: step {run = "table", at = 1, formula = 1}
First, the method of Part 4: [[mc-prediction|Monte Carlo]] with **a table**, one number per state. At the end of a walk, each state it visited moves toward the return, on its own: the thin needle in the strip under the plot is how far the last update moved each state. After this first walk, which left on the right, 114 states moved up. The other 886 still say 0, however close they are to states that have. No state learns anything from its neighbors.
:::

::: step {run = "table", checkpoints = [10, 100]}
After 10 walks, 557 states have been visited; after 100, 972. But most of them only a handful of times (a median of 7), and the states near the ends barely twice: the table is a ragged cloud around the truth. With a million states it would still be waiting for most of its first visits.
:::

::: step {run = "groups", at = 1, formula = 2}
Now the same walks with **ten numbers**: the states are cut into ten groups of 100, and all the states of a group share one weight. Every update now moves a whole group: after the same first walk, the strip shows a box. One visit teaches a hundred states at once.
:::

::: step {run = "groups", at = 100}
After the same 100 walks, a staircase that already follows the truth. It can never be exact (a group has one value for 100 different states), but it is right *roughly everywhere*, long before the table is right anywhere.
:::

::: step {run = "groups", at = 100, curves = ["table", "groups"], metric = "ve"}
Averaged over 100 runs, the shared weights pull ahead from the first walks. Grouping is the crudest way to share; [[features]] are the general idea, and the rest of this part is about learning with them. [Try it in the Lab](lab:walk-aggregation).
:::

## Textbook

### Where tables run out {#idea}

Every method so far stored its estimates in a table: one entry for each state, or for each state and action, each learned separately. That has three costs, and in most problems worth solving all three are fatal.

- **Memory.** Backgammon has about $10^{20}$ positions, Go about $10^{170}$; a robot that sees through a camera has more possible images than there are atoms in the universe. No table holds them.
- **Data.** A table learns about a state only by visiting it, and needs many visits per state to average out the noise. In a large problem most states are never visited even once, and the ones that are come back rarely.
- **Continuity.** A car's position and speed, a joint's angle, a temperature: when the state is made of real numbers there are infinitely many states, and the exact same one essentially never recurs.

Dynamic programming already ran into the first cost ([[dp-limits]]); learning from experience runs into all three. What the table lacks is any notion of **similarity**: state 501 is a stranger to state 500, whatever their values. In real problems, similar situations usually have similar values, and an agent that exploits this can learn about the states it has seen and *guess well* about the ones it has not. This ability is called **generalization**, and it is the reason to give up tables.

### Values computed from weights {#approx}

Instead of a table, the agent keeps a function with a fixed set of adjustable numbers, the **weights** $\mathbf w = (w_1, \dots, w_d)$, and computes the value of any state from them:

$$\val{\hat v(s, \mathbf w)} \approx \val{v_\pi(s)}, \qquad d \ll \text{number of states}. \label{eq-approx}$$

The function can be a weighted sum of [[features]] of the state (linear methods, the subject of this part), a neural network whose weights are its connections ([[neural-networks]]), a decision tree, or anything else that maps states to numbers through adjustable parameters. Learning now means changing the weights. Because there are far fewer weights than states, changing one weight changes the estimates of many states at once: generalization is built in, whether we want it or not.

### Every update is a training example {#examples}

All the methods of Parts 3 to 7 can be read as making individual updates of the form $s \mapsto u$: move the estimate of state $s$ toward a target $u$. They differed only in the target:

| method | target $u$ for state $S_t$ |
| --- | --- |
| Monte Carlo | the return $\rew{G_t}$ |
| TD(0) | $\rew{R_{t+1}} + \gam\,\val{\hat v(S_{t+1}, \mathbf w)}$ |
| $n$-step TD | the $n$-step return $\rew{G_{t:t+n}}$ |
| dynamic programming | $\mathbb E_\pi[\rew{R_{t+1}} + \gam\,\val{\hat v(S_{t+1}, \mathbf w)} \mid S_t = s]$ |

With a table, an update simply overwrites part of one entry. With function approximation, each update becomes a **training example** for a supervised learner: “at input $s$, the output should be closer to $u$”. That opens the whole toolbox of supervised learning, with two twists that rule out much of it:

- the examples arrive **one at a time**, while the agent acts, and the learner must keep up incrementally rather than fit a fixed dataset;
- the targets **move**: the policy changes during control, and bootstrapped targets depend on the very weights being learned.

Methods that learn from a stream with moving targets are what reinforcement learning needs; the rest of this part builds them on the simplest learner that qualifies, gradient descent on a weighted sum of features.

### A table is a special case {#tabular}

Give each state a feature of its own, equal to 1 in that state and 0 everywhere else. A weighted sum of these features just picks out one weight per state: the weights *are* the table, and an update moves exactly one of them. Everything in this part therefore contains the tabular methods as a special case, and the Lab runs a table this way (the first runs of the story above). Generalization starts as soon as features are shared between states, as with the ten groups of the story: the feature of a group is 1 in all its 100 states.

### What generalization costs {#cost}

Sharing weights has a price. With fewer weights than states, the estimates can no longer all be right: making one state's value more accurate generally makes another's less so, and the learner has to decide which states matter most. That choice is made precise by the [[value-error|value error]], which weighs each state's error by how often it is visited. Generalization can also mislead: an update meant for one state moves states that should have stayed put, and in the worst combinations (off-policy learning with bootstrapping) the errors feed on themselves until the weights diverge ([[deadly-triad]]). Choosing what is shared, through the features, is where most of the craft lies ([[generalization]]).

::: figure {#fig-groups}
{{walk-fit mc}}
The 1000-state walk learned by Monte Carlo with 10 groups of 100 states: each group settles on one value for all its states, a staircase around the truth. Where the walk spends more time (the gray bars), the steps fit more closely. Computed live by the Lab; after Sutton & Barto, Figure 9.1.
:::

### Historical remarks {#history}

Approximating value functions is as old as dynamic programming: Bellman and Dreyfus (1959) and Bellman, Kalaba and Kotkin (1963) fitted polynomials to values that could not be tabulated. Samuel's checkers player (1959) learned a weighted sum of hand-made board features from its own play, the first program to learn values from experience with function approximation. Tesauro's TD-Gammon (1992–1995) replaced the weighted sum with a small neural network trained by TD learning and reached the level of the best human players, a result that convinced many that the combination could work at scale. Bertsekas and Tsitsiklis (1996) collected the theory under the name neuro-dynamic programming.

## Card

### Idea

A table learns each state separately and only by visiting it. Too many states, or continuous ones, and it never finishes. Compute values from a few weights shared by many states instead: every update then teaches similar states too.

::: analogy
A table is a pupil who memorizes each exercise's answer; function approximation is a pupil who learns the method, and can answer exercises never seen before, sometimes wrongly.
:::

### Table or function {#compare}

| | table | function of weights |
| --- | --- | --- |
| memory | one number per state | $d$ weights, $d \ll$ number of states |
| an update changes | one state | every state that shares the weights |
| unseen states | stay at their first guess | get a guess from similar states |
| can be exact | yes, eventually | usually not: errors are traded between states |

### Why it matters {#why}

- Real problems have too many states for a table, or infinitely many. [See it](lab:walk-aggregation)
- Generalization lets an agent act sensibly in situations it has never met.
- Every method of Parts 3–7 carries over: each update becomes a training example $s \mapsto u$.

### Pitfalls

- Expecting exact values: with fewer weights than states, some states must be wrong.
- Sharing weights between states whose values differ sharply: the shared value fits none of them.
- Assuming tabular guarantees still hold: off-policy bootstrapping can diverge ([[deadly-triad]]).

### Check yourself {#check}

::: question
Why does a table learn slowly on the 1000-state walk, even though each walk visits about a hundred states?
---
Each update moves one state only, and each state needs many visits to average out the noise of the returns. Most states are visited rarely, so most entries stay rough for a long time.
:::

::: question
How can a table be written as a weighted sum of features?
---
With one feature per state, equal to 1 in that state and 0 elsewhere. The weighted sum then picks out one weight per state: the weights are the table.
:::

::: question
What does the agent give up when it shares weights between states?
---
Exactness: with fewer weights than states, it can no longer make every state's estimate right, and improving one state can worsen others.
:::
