+++
summary = "Three ingredients that are each harmless, and each needed at scale, can together make learning diverge: function approximation, bootstrapping and off-policy training. Then an update meant to pull one estimate toward another pushes the other away too, and the weights can grow without bound even when every true value is 0. Remove any one ingredient and the danger goes."
prereqs = ["semi-gradient-td", "off-policy-mc", "importance-sampling", "value-error"]
lab = "deadly-triad"
sources = [
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., §11.1–11.3, §11.7–11.8 and Example 11.1", url = "http://incompleteideas.net/book/the-book-2nd.html" },
  { text = "Baird (1995), Residual algorithms: reinforcement learning with function approximation, Proceedings of the 12th International Conference on Machine Learning", url = "https://doi.org/10.1016/B978-1-55860-377-6.50013-X" },
  { text = "Tsitsiklis & Van Roy (1997), An analysis of temporal-difference learning with function approximation, IEEE Transactions on Automatic Control 42", url = "https://doi.org/10.1109/9.580874" },
  { text = "Sutton, Maei, Precup, Bhatnagar, Silver, Szepesvári & Wiewiora (2009), Fast gradient-descent methods for temporal-difference learning with linear function approximation, Proceedings of the 26th International Conference on Machine Learning", url = "https://doi.org/10.1145/1553374.1553501" },
  { text = "Sutton, Mahmood & White (2016), An emphatic approach to the problem of off-policy temporal-difference learning, Journal of Machine Learning Research 17", url = "https://jmlr.org/papers/v17/14-488.html" },
  { text = "van Hasselt, Doron, Strub, Hessel, Sonnerat & Modayil (2018), Deep reinforcement learning and the deadly triad, arXiv:1812.02648", url = "https://arxiv.org/abs/1812.02648" },
]

[story]
scene = "star"
env = "baird"
seed = 1
average = 20
formula = '\step{1}{\rho = \frac{\pol{\pi(A \mid S)}}{b(A \mid S)}} \step{2}{\qquad \mathbf w \leftarrow \mathbf w + \alp\,\rho\,\big[\rew{R} + \gam\,\val{\hat v(S^\prime, \mathbf w)} - \val{\hat v(S, \mathbf w)}\big]\,\mathbf x(S)}'

[story.runs]
off = { algorithm = "semi-gradient-td", features = "own", behavior = "behavior", policy = "target", alpha = 0.01, gamma = 0.99, units = 1000, name = "all three" }
table = { algorithm = "semi-gradient-td", features = "table", behavior = "behavior", policy = "target", alpha = 0.01, gamma = 0.99, units = 1000, name = "a table instead" }
on = { algorithm = "semi-gradient-td", features = "own", policy = "target", alpha = 0.01, gamma = 0.99, units = 1000, name = "on-policy instead" }
+++

## Story

::: step {run = "off", at = 0}
**Seven states and nothing to win.** Every reward is 0, so every true value is 0: there is nothing to learn but zeros. The dashed action jumps to one of the six upper states, the solid one drops to state 7. Values are computed from eight weights: upper state $i$ is worth $2w_i + w_8$, state 7 is worth $w_7 + 2w_8$. The weights start so that the upper states say 3 and state 7 says 12. Wrong, but harmless, one would think.
:::

::: step {run = "off", at = 0, formula = 1}
The agent **behaves** by taking the dashed action 6 times out of 7. It wants to learn the values of a **target** policy that always takes the solid action. Off-policy, each update is weighted by how much more likely the target policy was to take that action: a solid action counts $1 / (1/7) = 7$ times, a dashed one, which the target never takes, counts 0.
:::

::: step {run = "off", at = 1, play = 1, pace = 400, formula = 2}
Step 2: from state 4, the solid action, to state 7. The target is $0 + 0.99 \times 12 = 11.88$, so state 4 rises from 3 to 6.1, as it should. But the update also moved $w_8$, which state 7 uses twice: **state 7 rose too**, from 12 to 13.2, and every other upper state a little. The estimate was pulled toward its target, and the target ran ahead.
:::

::: step {run = "off", at = 2, play = 1, pace = 400}
Step 3: in state 7 itself, the solid action again. Its target is 0.99 times its own estimate, only 1% below it, so it drops by a mere 0.04. Upper states pulled up toward state 7 move it up a lot; state 7's own updates pull it down a little. And the dashed jumps, which would visit the upper states with targets of their own, count 0.
:::

::: step {run = "off", at = 300}
After 300 steps, the estimates are between 58 and 84, every one of them wrong by that much.
:::

::: step {run = "off", at = 1000}
After 1000 steps, around 450, and the weights' size has grown from 10 to 336. Run it longer and it reaches thousands after 2000 steps, millions after 5000. Nothing random misled it: it is the updates themselves that feed each other.
:::

::: step {run = "table", at = 1000}
The same off-policy updates with **a table** instead of shared weights: each state has its own number, an update of state 4 cannot touch state 7, and nothing runs away. The upper states drift toward state 7's value, and all of them creep down toward 0, slowly: with $\gamma = 0.99$, state 7 loses only 1% of its error per own update.
:::

::: step {run = "on", at = 1000, curves = ["off", "table", "on"], metric = "weights"}
**On-policy**, following the target policy itself, also stays put. Averaged over 20 runs, on a log scale: the size of the weights explodes only when all three ingredients are there. [Run the three in the Lab](lab:deadly-triad).
:::

## Textbook

### Three ingredients {#idea}

Every method so far either came with a guarantee or behaved well in practice. One combination breaks that. The **deadly triad** is

- **function approximation**: estimates computed from weights shared by many states ([[why-approximate]]);
- **bootstrapping**: targets that include current estimates, as in TD and dynamic programming ([[bootstrapping]]);
- **off-policy training**: learning about one policy from data generated by another ([[importance-sampling]]).

With all three, the estimates can diverge, growing without bound, even in the simplest setting: linear features, a fixed target policy, expected updates with no sampling noise at all. With any two, they do not: on-policy linear TD converges ([[value-error]]); off-policy learning with a table is stable; off-policy Monte Carlo, which does not bootstrap, is still a gradient method with function approximation and does not diverge, however slowly the variance of its ratios lets it learn. The danger is not in any one ingredient but in their meeting, and each is hard to give up. Function approximation is unavoidable in large problems; bootstrapping is what makes learning fast and possible in tasks that never end; off-policy learning lets an agent learn about the greedy policy while exploring, learn from old or others' data, and learn many things at once from one stream of experience.

### A two-state example {#tiny}

The mechanism fits in two states. Let state A have the single feature 1 and state B the feature 2, so their estimates are $w$ and $2w$; A leads to B with reward 0, and nothing more is known. The TD update from A, with target $\gam \cdot 2w$, is

$$w \leftarrow w + \alp\,\big[\,0 + \gam\,2w - w\,\big] \cdot 1 = \big(1 + \alp\,(2\gamma - 1)\big)\,w. \label{eq-tiny}$$

For $\gamma > 0.5$ the factor exceeds 1: each update makes $w$ larger, and with it B's estimate, and with it the next target. The estimate of A chases a target that it drags along, faster than it can catch it. On-policy, this would not last: the agent would also visit B and update it, toward whatever follows B, which pulls $w$ back. Off-policy, the transitions out of B may never be weighted in, and nothing stops the chase.

### Baird's counterexample {#baird}

::: example {#ex-baird} Baird's counterexample
Seven states: six upper states and one lower. Two actions: dashed, which goes to one of the six upper states with equal probability, and solid, which goes to the lower state. All rewards are 0 and $\gamma = 0.99$, so the true values are all 0. The behavior policy $b$ takes dashed with probability $6/7$, solid with $1/7$, which makes the next state uniform over all seven; the target policy $\pi$ always takes solid. Values are linear in eight weights: upper state $i$ has features giving $\hat v = 2w_i + w_8$, the lower state $\hat v = w_7 + 2w_8$. The weights start at $(1, 1, 1, 1, 1, 1, 10, 1)$.
:::

The features are not at fault: they are linearly independent, and with eight weights for seven states any values can be represented, the right ones included ($\mathbf w = \mathbf 0$). Yet semi-gradient off-policy TD with importance ratios ([[semi-gradient-td]]) diverges for any positive step size, and so does semi-gradient dynamic programming, which updates every state at once with the exact expected target and no randomness at all. The cause is the distribution of the updates: the target policy would spend nearly all its time in the lower state, where the update pulls estimates down; the behavior policy spends six sevenths of its time elsewhere, in upper states whose solid-action updates (weighted by 7) pull toward the lower state's value, raising $w_8$, which raises the lower state's value, and so on.

::: figure {#fig-baird}
{{baird-weights}}
The weights of Baird's counterexample under semi-gradient off-policy TD (sampled transitions, each weighted by its importance ratio) and under semi-gradient dynamic programming (every state at once, expected targets). Both diverge; the expected version shows the trend without the noise. Computed live by the Lab; after Sutton & Barto, Figure 11.2.
:::

### Why on-policy is safe {#why-on}

Linear semi-gradient TD moves the weights, on average, by $\alp\,(\mathbf b - \mathbf A \mathbf w)$ ([[value-error]]), where $\mathbf A = \sum_s d(s)\, \mathbf x(s)\,\big(\mathbf x(s) - \gam\,\mathbb E[\mathbf x(S') \mid s]\big)^\top$ and $d$ is the distribution of the updated states. When $d$ is the on-policy distribution, the states the target policy itself would visit, $\mathbf A$ is positive definite: the expected update always has a component pointing toward the fixed point, whatever the weights. The intuition is the two-state example: on-policy, every state whose estimate serves as a target is itself updated as often as it is used, so no estimate can be dragged along without being corrected. Under another distribution, some estimates serve as targets far more often than they are corrected, $\mathbf A$ can have directions of negative curvature, and along those the weights run away. The importance ratios fix the targets, not $d$.

### Escaping the triad {#escape}

Each way out drops or softens one ingredient, or changes what is being descended.

- **Stay on-policy**, or close to it: SARSA instead of Q-learning when stability matters; [[semi-gradient-sarsa]] does not diverge (though it may chatter).
- **Bootstrap less**: $n$-step returns and $\lambda$ close to 1 move toward Monte Carlo, which is stable; the danger grows with how much each target leans on estimates.
- **Correct the distribution**: emphatic TD reweights each update by an emphasis that reproduces the on-policy weighting of the target policy (Sutton, Mahmood and White, 2016). It is stable, with high variance.
- **Descend a real objective**: gradient-TD methods (GTD2, TDC; Sutton et al., 2009) follow the true gradient of the projected Bellman error, at the cost of a second set of weights and a second step size; they converge off-policy with linear features.
- **Slow the chase**: with neural networks, a separate target network, updated only occasionally, keeps targets from following each update, and a replay memory mixes the data. They do not remove the triad but make divergence rare in practice ([[target-network]], [[dqn]]); van Hasselt et al. (2018) measured how rarely, and when it still happens.

### Historical remarks {#history}

Baird (1995) built the counterexample that bears his name, showing that even linear off-policy TD and dynamic programming can diverge, and proposed residual-gradient methods as a cure. Tsitsiklis and Van Roy (1997) gave a simpler two-state example and the analysis of when linear TD converges. The name deadly triad is Sutton's, popularized in the second edition of Sutton and Barto's book. Gradient-TD methods are due to Sutton, Maei, Szepesvári and colleagues (2008–2009), emphatic TD to Sutton, Mahmood and White (2016); van Hasselt et al. (2018) studied the triad in deep Q-learning.

## Card

### Idea

Function approximation, bootstrapping and off-policy training: each is safe, each is needed at scale, and together they can make the estimates diverge. An update meant to pull one estimate toward another drags the other along, and off-policy nothing pulls it back. Drop any one and the danger goes.

::: analogy
Two hikers roped together, each trying to stay a step behind the other: if only one of them ever adjusts, they march off together, faster and faster, in the same wrong direction.
:::

### The three, and what each pair gives {#three}

| ingredients | example | stable? |
| --- | --- | --- |
| approximation + bootstrapping | on-policy linear TD | yes |
| approximation + off-policy | off-policy gradient Monte Carlo | yes |
| bootstrapping + off-policy | tabular Q-learning | yes |
| all three | off-policy semi-gradient TD, DQN | can diverge |

### The chase in one line {#formula}

$$w \leftarrow \big(1 + \alp\,(2\gamma - 1)\big)\,w$$

Two states worth $w$ and $2w$, updating only the first toward the second: for $\gamma > 0.5$, $w$ grows every step.

### Why it matters {#why}

- It explains why deep Q-learning needed target networks and replay to work at all. [See it](lab:deadly-triad)
- It marks where theory stops guaranteeing anything, and where to be careful.
- Gradient-TD, emphatic TD and on-policy methods are the principled ways out.

### Pitfalls

- Blaming the features: Baird's features can represent the true values exactly, and it still diverges.
- Blaming noise: semi-gradient dynamic programming, with no randomness at all, diverges too.
- Assuming importance ratios make off-policy updates safe: they correct targets, not the distribution of updates.

### Check yourself {#check}

::: question
In Baird's counterexample every reward is 0. Why do the weights grow anyway?
---
Updates of the upper states pull them toward the lower state's estimate, and through the shared weight $w_8$ raise that estimate too. Off-policy, the lower state is updated too rarely to pull back, so each update raises the targets of the next.
:::

::: question
Which ingredient of the triad does each remove: a table, Monte Carlo returns, following the target policy?
---
A table removes function approximation (no shared weights); Monte Carlo returns remove bootstrapping; following the target policy removes off-policy training. Each alone is enough for stability.
:::

::: question
In the two-state example, why does the problem disappear on-policy?
---
On-policy, the agent also visits state B and updates it toward what follows B. That update pulls $w$ back down, and every estimate used as a target is corrected as often as it is used.
:::
