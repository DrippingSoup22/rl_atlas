+++
summary = "Generalization is learning about states you have not visited from states you have; discrimination is telling apart states whose values differ. Features that are shared widely generalize fast and blur; features that are narrow discriminate finely and learn slowly. Many overlapping wide features get much of both: how far an update reaches and how finely values can vary are separate choices."
prereqs = ["features", "why-approximate"]
lab = "walk-features"
sources = [
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., §9.5, §9.9–9.11 and Example 9.3", url = "http://incompleteideas.net/book/the-book-2nd.html" },
  { text = "Hinton (1984), Distributed representations, Technical Report CMU-CS-84-157, Carnegie Mellon University" },
  { text = "McCloskey & Cohen (1989), Catastrophic interference in connectionist networks: the sequential learning problem, Psychology of Learning and Motivation 24", url = "https://doi.org/10.1016/S0079-7421(08)60536-8" },
  { text = "Atkeson, Moore & Schaal (1997), Locally weighted learning, Artificial Intelligence Review 11", url = "https://doi.org/10.1023/A:1006559212014" },
  { text = "Sutton, Mahmood & White (2016), An emphatic approach to the problem of off-policy temporal-difference learning, Journal of Machine Learning Research 17", url = "https://jmlr.org/papers/v17/14-488.html" },
]

[story]
scene = "line"
env = "walk-1000"
seed = 2
average = 20
formula = '\step{1}{\Delta\val{\hat v(s)} = \alp\,\del\;\mathbf x(s)^\top \mathbf x(S)}'

[story.runs]
wide = { algorithm = "gradient-mc", features = "groups", cells = 4, alpha = 0.0005, gamma = 1.0, units = 2000, measures = ["ve"], name = "4 wide groups" }
narrow = { algorithm = "gradient-mc", features = "groups", cells = 50, alpha = 0.0005, gamma = 1.0, units = 2000, measures = ["ve"], name = "50 narrow groups" }
many = { algorithm = "gradient-mc", features = "tiles", tilings = 20, cells = 4, alpha = 0.000025, gamma = 1.0, units = 2000, measures = ["ve"], name = "20 shifted wide tilings" }
+++

## Story

::: step {run = "wide", at = 200, formula = 1}
**Four wide groups** of 250 states each. An update moves every state that shares a feature with the visited one, by the share they have in common: the strip under the plot. Here that means a quarter of the walk at once. After 200 walks the four steps already have roughly the right shape.
:::

::: step {run = "wide", at = 2000}
And after 2000 walks they are still four steps. A group has one value for 250 states whose true values run from one end of a slope to the other: no amount of data can fix that. Wide features **generalize** well and **discriminate** poorly.
:::

::: step {run = "narrow", at = 200}
**Fifty narrow groups** of 20 states. Now each update touches a sliver, and each sliver needs its own visits. After 200 walks most slivers have barely moved: the line is ragged and close to 0 near the ends, which the walk rarely reaches.
:::

::: step {run = "narrow", at = 2000}
After 2000 walks the slivers have caught up and the line follows the curve, raggedly: each sliver has only its own visits to average. Narrow features discriminate well and generalize poorly: they need many more visits.
:::

::: step {run = "many", at = 200}
**Twenty tilings of the same wide tiles**, each shifted 12.5 states from the last. Each update still reaches 250 states either side, a tent in the strip, so learning starts as fast as with the four groups...
:::

::: step {run = "many", at = 2000}
...but the boundaries of twenty tilings interleave, and the estimates can vary every 12.5 states. After 2000 walks: a smooth, close fit. How far an update reaches is set by the width of the features; how finely values can vary, by how many there are.
:::

::: step {run = "many", at = 2000, curves = ["wide", "narrow", "many"], metric = "ve"}
Averaged over 20 runs: the wide groups start fast and stall, the narrow ones start slow and keep improving, the shifted tilings do both. [Compare features in the Lab](lab:walk-features).
:::

## Textbook

### Two demands that pull apart {#idea}

A learner with fewer weights than states must do two things at once. It must **generalize**: let what it learns in one state inform similar states it has visited rarely or never, or it learns as slowly as a table. And it must **discriminate**: keep apart states whose values differ, or its estimates blur everything into a few averages. The two pull in opposite directions. A table discriminates perfectly and generalizes not at all; a single feature shared by every state generalizes perfectly and discriminates nothing. Every set of features sits somewhere between, and where it sits decides both how fast learning goes and how good it can eventually get.

### What an update touches {#kernel}

For a linear method, an update at state $S$ with error $\del$ changes the estimate of any state $s$ by

$$\Delta \val{\hat v(s)} = \alp\,\del\;\mathbf x(s)^\top \mathbf x(S), \label{eq-kernel}$$

proportional to the overlap $k(s, S) = \mathbf x(s)^\top \mathbf x(S)$ of their features. This overlap, seen as a function of $s$, is the **generalization pattern** of the features at $S$: a box for state aggregation, a tent for tile coding, a smooth bump for radial basis functions, waves spread over the whole space for polynomials and cosines. The Lab's walk draws it under every update. Two states with zero overlap are fully discriminated, never moved by each other's updates; two states with identical features are not discriminated at all.

The same overlap defines **kernel methods**, which skip features altogether: they store past examples and estimate a new state's value as a weighted average of their targets, weighted by a kernel $k(s, s')$ chosen directly. Any linear method is a kernel method in disguise, with $k$ the overlap of its features; memory-based methods such as nearest neighbors and locally weighted regression (Atkeson, Moore and Schaal, 1997) work this way, trading weights for a memory of examples that grows with experience.

### Reach and resolution are separate {#width}

Coarse coding makes the trade-off concrete. With binary features over overlapping intervals or circles, the **width** of the features sets how far each update reaches, and the **number** of features (how densely they cover the space) sets how finely the estimates can eventually vary. It is tempting to think that wide features must give blurry estimates; they do not, as long as there are many of them, offset from one another, because the estimate at a point is the sum of many overlapping features whose edges fall in different places.

::: figure {#fig-widths}
{{coarse-widths}}
A square pulse learned from random examples by linear coarse coding with narrow, medium and broad intervals, 50 features in each case, after more and more examples. Broad features learn the rough shape fastest; by the end all three are about equally sharp, because the final resolution is set by the number of features, not their width. Computed live; after Sutton & Barto, Figure 9.8.
:::

The width matters most early on, when it decides how much each example teaches; the number matters most late, when it decides what can be represented. Tile coding builds on exactly this: a few tilings of large tiles learn fast, and their interleaved boundaries keep fine resolution, as in the story. The **shape** of the features matters too: long thin regions generalize along their length and discriminate across it, which is the right thing when the value depends strongly on one direction and weakly on another. Choosing shapes is one of the ways knowledge of a problem enters ([[features]]).

::: figure {#fig-touch}
{{touch-tiles}}
The same in two dimensions. With one tiling, a click lifts a square block: wide generalization, coarse discrimination. With more tilings of the same tiles, a click lifts a blurred patch, and clicks close together can now be told apart.
:::

### Interference {#interference}

Generalization works in both directions: what an update teaches a neighbor may be wrong for it. Learning about one region then **interferes** with what was learned in another, and if the agent spends a long time in one part of the space, the estimates elsewhere can drift far from where they were, even be overwritten entirely. With local features (tiles, groups) the damage stays within the reach of the features. With global ones (polynomials, cosines) and especially with neural networks, every update touches everything, and the effect can be dramatic, known as **catastrophic interference** (McCloskey and Cohen, 1989). It is one reason deep reinforcement learning keeps a replay memory and trains on old experience mixed with new ([[experience-replay]]).

A related lever is to tell the learner which states matter. Weighting the updates by an **interest** in each state, and passing that interest along to the states that bootstrap from it (the **emphasis**), lets accuracy be spent where it is needed, and is the basis of emphatic TD methods, which also tame off-policy learning (Sutton, Mahmood and White, 2016).

### Historical remarks {#history}

The tension between generalization and discrimination is old in psychology and in pattern recognition. Hinton (1984) showed how distributed codes, many coarse features each active for many inputs, can represent fine distinctions, the principle behind coarse coding; Sutton and Barto's Example 9.3, recomputed above, makes it visible. McCloskey and Cohen (1989) documented catastrophic interference in neural networks. Atkeson, Moore and Schaal (1997) surveyed memory-based, locally weighted learning, and Sutton, Mahmood and White (2016) introduced interest and emphasis in emphatic TD.

## Card

### Idea

Generalize: learn about unvisited states from visited ones. Discriminate: keep apart states whose values differ. Wide features generalize and blur, narrow ones discriminate and learn slowly. Many overlapping wide features do both: reach is set by width, resolution by number.

::: analogy
Judging a restaurant by its street: a quick guess for every restaurant you have never tried, unfair to the good one on a bad street. Judging by the street and the chef and the price together tells them apart.
:::

### Features and what they give {#kinds}

| features | generalization | discrimination | learns |
| --- | --- | --- | --- |
| a few wide groups | wide | coarse | fast, then stalls |
| many narrow groups | narrow | fine | slowly |
| many shifted wide tilings | wide | fine | fast, and keeps improving |
| one feature per state | none | perfect | like a table |

### What an update touches {#formula}

$$\Delta \val{\hat v(s)} = \alp\,\del\;\mathbf x(s)^\top \mathbf x(S)$$

The overlap of features with the updated state: the generalization pattern.

### Why it matters {#why}

- It decides how many visits learning needs, and how good the estimates can get. [See it](lab:walk-features)
- Width and number of features are separate controls: tile coding uses both.
- Interference, its dark side, is why deep RL needs replay memories.

### Pitfalls

- Making features narrower to gain accuracy, and losing all learning speed.
- Global features in a task that stays long in one region: the rest of the estimates drift.
- Judging features by their final fit only: early learning speed often matters more.

### Check yourself {#check}

::: question
In the coarse-coding figure, broad features learn fastest but end as sharp as narrow ones. Why?
---
Each example moves many broad features, so early learning spreads widely. In the end the resolution is set by the number of features and how their edges interleave, which was the same, 50, in all three cases.
:::

::: question
With linear features, by how much does an update at $S$ change the estimate of another state $s$?
---
By $\alpha\,\delta\,\mathbf x(s)^\top \mathbf x(S)$: in proportion to the overlap of their feature vectors. No overlap, no change.
:::

::: question
Why do 20 shifted tilings of wide tiles learn better than 4 wide groups, given the same tile width?
---
Each update reaches just as far, so early learning is as fast, but the interleaved boundaries of 20 tilings let the estimates vary every few states instead of every 250, so they can fit the curve much more closely.
:::
