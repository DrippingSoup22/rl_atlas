+++
summary = "Two jobs, interleaved at any grain: bring the values up to date with the policy, and make the policy greedy for the values. Most reinforcement learning methods are built on this loop."
prereqs = ["policy-iteration", "value-iteration"]
lab = "dp-iteration"
sources = [
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., §4.6", url = "http://incompleteideas.net/book/the-book-2nd.html" },
  { text = "Bertsekas & Tsitsiklis (1996), Neuro-Dynamic Programming, Athena Scientific, §6.2" },
  { text = "Konda & Tsitsiklis (2000), Actor-critic algorithms, Advances in Neural Information Processing Systems 12" },
]

[story]
scene = "grid"
env = "frozen-lake"
digits = 2
formula = '''\step{1}{\text{evaluate: } \val{V} \to \val{v_{\pi}} \qquad} \step{2}{\text{improve: } \pol{\pi} \to \operatorname{greedy}(\val{V})}'''

[story.runs]
pi = { algorithm = "policy-iteration", gamma = 0.99, theta = 0.001, judge = 1.0, units = 120 }
vi = { algorithm = "value-iteration", gamma = 0.99, theta = 0.001, judge = 1.0, units = 120 }
ql = { algorithm = "q-learning", gamma = 0.99, alpha = 0.1, epsilon = 0.3, judge = 1.0, units = 5000, seed = 6 }
+++

## Story

::: step {run = "pi", at = 13, values = true, formula = 1}
**Two jobs, taken in turn.** The first: work out what the current policy is worth. Policy iteration has done it for the random policy on Frozen Lake, sweep after sweep, until the values settled: they are tiny, because walking at random the agent reaches the gem about once in a hundred tries.
:::

::: step {run = "pi", at = 14, values = true, formula = 2}
**The second job: make the policy greedy for those values.** Every arrow changes, and at once the values on the tiles are out of date: they are what the random policy was worth, and that policy no longer exists. Each job undoes part of the other's work. The new policy, already, reaches the gem 78% of the time.
:::

::: step {run = "pi", at = 120, values = true}
**Turn after turn, the undoing shrinks,** until neither job changes anything: the values are the policy's own, and the policy is greedy for them. A pair that satisfies both is optimal, and this one reaches the gem 82.4% of the time, starting with the famous move left, into the wall. (The start shows 0.53 because the values are discounted by $\gam = 0.99$ per step, and on this ice the gem is many steps away.)
:::

::: step {run = "vi", at = 14, values = true}
**The same two jobs, finely interleaved.** Value iteration evaluates for one sweep only, and folds the improvement into every update, through the max. It never finishes evaluating anything, yet after 14 sweeps its greedy policy reaches the gem 78% of the time, and after 99 sweeps the same 82.4%.
:::

::: step {run = "ql", at = 5000, values = true}
**Finer still: one sample at a time, with no model.** Q-learning evaluates with a single TD update per step and improves with the ε-greedy choice at every step. Its values are noisier, but after 1,509 episodes its greedy policy was already the optimal one, left into the wall included, and it still is after 5,000. Three grains of the same pattern, and the same answer: that pattern, **generalized policy iteration**, is behind nearly every method in this atlas.
:::

## Textbook

### Two jobs {#processes}

Every method of dynamic programming does two jobs. One, **policy evaluation**, adjusts the values until they describe the current policy. The other, **policy improvement**, adjusts the policy until it is greedy with respect to the current values. The methods differ only in how they schedule the jobs. [[policy-iteration]] finishes each one before starting the other; [[value-iteration]] gives evaluation a single sweep before each improvement; asynchronous methods switch between them state by state ([[dp-limits]]).

The schedule turns out to matter little. However the two jobs take turns, as long as neither stops visiting any state, they end in the same place: optimal values and an optimal policy. **Generalized policy iteration** (GPI) names this shared pattern, evaluation and improvement working on each other, without fixing how finely they alternate or how either one is done. The pattern reaches far beyond dynamic programming. Open up a learning method and you will usually find a policy and an estimate of its values, with the estimate chasing the values of the policy and the policy leaning toward whatever the estimate favors.

### Undoing each other, then agreeing {#dynamics}

Each job undoes part of the other's work. Once the policy changes, the values that justified the change describe a policy that no longer exists. Once the values are brought up to date, the policy may no longer be greedy with respect to them. Yet the undoing shrinks each time, and the two settle on a pair that both accept. \ref{fig-gpi} draws this as two lines: values that match their policy, and policies that are greedy for their values. Each job steps onto its own line and off the other, and the zigzag closes in on the point where the lines cross.

::: figure {#fig-gpi}
{{gpi}}
Generalized policy iteration. Evaluation drives the values toward those of the policy ($v = v_\pi$); improvement drives the policy toward the greedy one ($\pi = \text{greedy}(v)$). Each step undoes some of the other's work, yet together they converge to the only point that satisfies both: the optimal value function and an optimal policy. After Sutton & Barto, §4.6.
:::

The meeting point is optimal, and it is the only stable one.

::: theorem {#thm-stable} The fixed point of GPI is optimal
If a value function $v$ and a policy $\pol{\pi}$ are stable under both processes, that is, $v = \val{v_\pi}$ and $\pol{\pi}$ is greedy with respect to $v$, then $v = \val{v_*}$ and $\pol{\pi}$ is optimal.
:::

::: proof
Because $\pol{\pi}$ is greedy with respect to $v = \val{v_\pi}$, the Bellman equation of $\pol{\pi}$ reads $v(s) = \max_a \sum_{s', r} p(s', r \mid s, a)\,[r + \gam\,v(s')]$ for every state. That is the Bellman optimality equation, whose only solution is $\val{v_*}$ ([[optimality]]). A policy greedy with respect to $\val{v_*}$ is optimal.
:::

### One pattern, many grains {#granularity}

The methods of the atlas differ in how they carry out each process and how finely they interleave them, but nearly all of them have this structure:

| Method | Evaluation | Improvement |
| --- | --- | --- |
| [[policy-iteration]] | exact, to convergence, with the model | greedy, in every state, after each evaluation |
| [[value-iteration]] | one sweep, with the model | greedy, folded into every update (the max) |
| [[mc-control]] | averages of returns, one episode at a time | ε-greedy, after each episode |
| [[sarsa]], [[q-learning]] | one TD update, one step at a time | ε-greedy, at every step |
| [[actor-critic]] | a critic learns values by TD | an actor follows the policy gradient a little |
| [[ppo]] | an advantage estimate from a batch of experience | several clipped gradient steps on the policy |

From top to bottom, evaluation becomes cheaper and less accurate: exact computation, then averages of samples, then single bootstrapped samples, then approximations. Improvement becomes more gradual: a full greedy switch, then ε-greedy, then a small gradient step. What stays is the interplay.

### When it converges {#convergence}

With exact evaluation, exact greedy improvement and a table of values, GPI converges, whatever the grain, as long as every state keeps being updated. With sampling and approximation the guarantees weaken. Evaluation may be noisy or biased, and improvement may be only approximately greedy. A classical result bounds the damage for *approximate policy iteration*: if every evaluation is accurate to within $\delta$ and every improvement is greedy to within $\epsilon$, the policies eventually come within $(\epsilon + 2\gam\delta)/(1 - \gam)^2$ of optimal (Bertsekas & Tsitsiklis, 1996). The policies need not converge at all; they may keep cycling within that distance of the optimum. When evaluation itself can diverge, as with off-policy bootstrapping and function approximation, even that bound is lost ([[deadly-triad]]).

Making improvement gradual is one way to keep the two processes in step. Actor–critic methods change the policy slowly enough for the critic's evaluation to keep up; under such two-timescale step sizes they converge (Konda & Tsitsiklis, 2000). Trust-region methods limit how far one improvement may move the policy ([[trpo]], [[ppo]]).

### Historical remarks {#history}

The term *generalized policy iteration* was introduced by Sutton and Barto in the first edition of their book (1998), to name the structure shared by dynamic programming and the learning methods that followed it. Bertsekas and Tsitsiklis (1996) analyzed approximate policy iteration and gave the error bound above.

## Card

### Idea

Every control method does two things. It **evaluates**: it moves the values toward those of the current policy. And it **improves**: it moves the policy toward the greedy one for the current values. Each step partly undoes the other, but together they settle at the optimal policy, the only point where both are satisfied.

::: analogy
A company and its accountant. Management changes the strategy based on the latest figures, and the accountant updates the figures for the new strategy. Each change makes the other out of date, yet back and forth they settle on the best strategy with accurate figures.
:::

### In symbols {#formula}

$$\text{evaluation: } \val{V} \to \val{v_\pi} \qquad \text{improvement: } \pol{\pi} \to \operatorname{greedy}(\val{V}) \qquad \text{fixed point: } \val{V} = \val{v_*},\ \pol{\pi} = \pol{\pi_*}$$

### The family {#family}

| | evaluation | improvement |
| --- | --- | --- |
| policy iteration | to convergence | greedy |
| value iteration | one sweep | greedy (the max) |
| SARSA, Q-learning | one step | ε-greedy |
| actor–critic | a critic | a small policy-gradient step |

### Why it matters {#why}

It is the skeleton of reinforcement learning. Faced with a new method, ask how it evaluates, how it improves, and how finely it interleaves the two.

### Pitfalls

- Improving faster than the evaluation can follow: the policy chases values that describe an older policy.
- Assuming GPI always converges: with sampling and function approximation it may oscillate or diverge.
- Looking for separate phases: in most methods both processes happen in every update.

### Check yourself {#check}

::: question
In SARSA, which part of the algorithm is evaluation and which is improvement?
---
The TD update of $\val{Q}$ is evaluation of the current ε-greedy policy. Acting ε-greedily with respect to the updated $\val{Q}$ is improvement, done at every step.
:::

::: question
Why is a value function that is consistent with a policy that is greedy with respect to it optimal?
---
Then the policy's Bellman equation takes the max over actions, so it is the Bellman optimality equation, whose only solution is $\val{v_*}$.
:::

::: question
Policy iteration and value iteration are both GPI. What is different about their grain?
---
Policy iteration evaluates the policy until its values settle before improving it. Value iteration cuts evaluation to a single backup per state and folds the improvement into it, through the max over actions.
:::
