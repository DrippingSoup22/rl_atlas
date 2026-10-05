+++
summary = "Two processes, interleaved at any grain: make the values agree with the policy, and make the policy greedy with respect to the values. Almost every reinforcement learning method is built this way."
prereqs = ["policy-iteration", "value-iteration"]
lab = "dp-iteration"
sources = [
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., §4.6", url = "http://incompleteideas.net/book/the-book-2nd.html" },
  { text = "Bertsekas & Tsitsiklis (1996), Neuro-Dynamic Programming, Athena Scientific, §6.2" },
  { text = "Konda & Tsitsiklis (2000), Actor-critic algorithms, Advances in Neural Information Processing Systems 12" },
]
+++

## Textbook

### Two processes {#processes}

Policy iteration consists of two simultaneous, interacting processes ([[policy-iteration]]). **Policy evaluation** makes the value function consistent with the current policy. **Policy improvement** makes the policy greedy with respect to the current value function. In policy iteration they alternate, each completing before the other begins. In value iteration only a single sweep of evaluation happens between improvements ([[value-iteration]]). In asynchronous dynamic programming the two are interleaved at an even finer grain, sometimes a single state at a time ([[dp-limits]]).

As long as both processes keep updating all states, the end result is the same: convergence to the optimal value function and an optimal policy. **Generalized policy iteration** (GPI) is the name for this general idea of letting evaluation and improvement interact, independent of the granularity and other details of the two. Almost every reinforcement learning method is well described as GPI: it has an identifiable policy and value function, the policy is always being improved with respect to the value function, and the value function is always being driven toward the value function of the policy.

### Competing and cooperating {#dynamics}

The two processes compete in one sense and cooperate in another. They pull in opposite directions: making the policy greedy with respect to the values typically makes the values incorrect for the changed policy, and making the values consistent with the policy typically makes the policy no longer greedy. In the long run, however, they find a single joint solution. \ref{fig-gpi} pictures each process as driving toward one of two goals, represented as lines in the space of value functions and policies. Driving toward one goal moves away from the other, yet the zigzag of steps closes in on the point where the lines meet.

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
