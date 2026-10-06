+++
summary = "An on-policy method learns about the policy it follows, exploration and all. An off-policy method learns about one policy, usually the greedy one, from experience gathered by another. Off-policy learning is what lets an agent reuse old data, learn from others, or learn while exploring freely; it pays in variance and stability."
prereqs = ["policy", "q-learning", "sarsa"]
lab = "cliff-race"
sources = [
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., sections 5.5–5.7, 6.4–6.6, 7.3 and chapter 11", url = "http://incompleteideas.net/book/the-book-2nd.html" },
  { text = "Munos, Stepleton, Harutyunyan & Bellemare (2016), Safe and efficient off-policy reinforcement learning (Retrace), NeurIPS", url = "https://arxiv.org/abs/1606.02647" },
  { text = "Espeholt et al. (2018), IMPALA: scalable distributed deep-RL with importance weighted actor-learner architectures (V-trace), ICML", url = "https://arxiv.org/abs/1802.01561" },
]

[story]
scene = "grid"
env = "cliff"
seed = 4
average = 20
formula = '\step{1}{\text{Q-learning: } \rew{R} + \gam \max_a \val{Q(S^\prime\!, a)} \qquad} \step{2}{\text{SARSA: } \rew{R} + \gam\,\val{Q(S^\prime\!, A^\prime)}}'

[story.runs]
walker = { algorithm = "q-learning", alpha = 0.5, epsilon = 1.0, gamma = 1.0, maxSteps = 100000, units = 50, name = "Q-learning, walking at random" }
sarsa = { algorithm = "sarsa", alpha = 0.5, epsilon = 0.1, gamma = 1.0, units = 500, seed = 7, name = "SARSA (on-policy)" }
q = { algorithm = "q-learning", alpha = 0.5, epsilon = 0.1, gamma = 1.0, units = 500, seed = 7, name = "Q-learning (off-policy)" }
+++

## Story

::: step {run = "walker", at = 0, trail = false}
**Two policies.** Every learner has a policy that acts, the **behavior** policy, and a policy it learns about, the **target** policy. Here they could not be more different. On the cliff (start at the bottom left, the gem at the bottom right, $\rew{-1}$ a step, $\rew{-100}$ and back to the start for a fall), this agent picks every move at random, and it never stops doing so. What it learns about is the greedy policy: the best move from each tile.
:::

::: step {run = "walker", at = 1}
Its first episode lasts 5,010 steps. 409 of them are falls into the cliff, until a random move happens to land on the gem. Nothing in this walk looks like a way to the gem.
:::

::: step {run = "walker", checkpoints = [2, 5, 10], path = true, arrows = false, trail = false, formula = 1}
**Q-learning learns from these walks.** Its target is the reward plus the value of the **best** move from the next tile, whatever the walker does next: it asks what the greedy policy would get from there. After 5 episodes, the greedy path runs along the edge of the cliff to the gem in 13 steps, the shortest way. The values count the steps down: about $\val{-13}$ at the start, $\val{-1}$ next to the gem.
:::

::: step {run = "walker", at = 50, path = true, arrows = false, trail = false}
The walker never once walked that path. This is **off-policy** learning: the data came from one policy, and the answer is about another. Of 20 runs, 18 have the shortest path after 10 random episodes, and all 20 after 50. Learning from someone else's behavior is what lets an agent reuse old experience, as DQN's replay memory does ([[experience-replay]]), or learn from other agents and from people.
:::

::: step {run = "sarsa", at = 500, path = true, formula = 2}
**On-policy** learning could not do this. SARSA's target uses the move the agent will actually make next: fed these random walks, it would learn what walking at random is worth, about $\val{-65{,}000}$ from the start, and nothing about the way to the gem. On-policy pays off when the exploring never stops. With $\eps = 0.1$, as in [[sarsa]], SARSA's values count its own random steps, so they warn it off the edge: its greedy path takes the top row, 17 steps, two rows clear of the cliff.
:::

::: step {run = "q", at = 500, path = true, formula = 1}
Q-learning with the same $\eps = 0.1$ and the same seed learns the edge path, 13 steps. It is the better policy, but the agent does not follow it: it still takes a random step one time in ten, and next to the edge a random step down is a fall.
:::

::: step {run = "q", at = 500, path = true, curves = ["sarsa", "q"], metric = "return", domain = [-100, 0]}
**What each one earns while it learns**, averaged over 20 runs: over the last 100 episodes, about $\rew{-27}$ per episode for SARSA and $\rew{-52}$ for Q-learning. Off-policy learns the best policy; on-policy learns the best way to behave while still exploring. Which one you want depends on whether the exploring ever stops. [Race them in the Lab](lab:cliff-race).
:::

## Textbook

### Two policies {#two}

Any learning agent has a **behavior policy** $b$, the one that chooses its actions and so produces its data, and a **target policy** $\pol\pi$, the one it is learning about. When they are the same, learning is **on-policy**. When they differ, it is **off-policy**: the agent learns how good, or how to improve, a policy it is not following.

Off-policy learning needs **coverage**. If $\pol\pi$ might take an action in some state, $b$ must take it there too, sometimes; experience can say nothing about actions never tried.

### The cliff {#cliff}

The cliff walk ([[q-learning]]) shows the difference at its sharpest. Both learners follow ε-greedy behavior with $\eps = 0.1$:

- **SARSA** is on-policy: its target uses the action it will actually take next, exploration included. It learns what its exploring self is worth, and that self sometimes slips off the edge. So it learns to keep a safe distance from the cliff.
- **Q-learning** is off-policy: its target uses the best next action, $\max_{a'} \val{Q(s', a')}$, whatever it will actually do. It learns the values of the greedy policy, and the greedy path runs right along the edge.

Q-learning's greedy policy is the better one: run without exploration, it reaches the goal fastest. But the agent keeps exploring while it learns, and its random steps along the edge send it off the cliff. Over 20 runs each, SARSA ends up losing less than 40 per episode, exploration included, in all 20; Q-learning in 1. Which answer is right depends on the question: what is the best policy, or what is the best way to behave while still exploring?

### How off-policy methods learn {#how}

Data from $b$ has the wrong distribution for $\pol\pi$, and the fix depends on what is estimated:

- **Importance sampling** reweights each return by how much more or less likely $\pol\pi$ was to produce it: $\rho = \prod_k \pol{\pi(A_k \mid S_k)} / b(A_k \mid S_k)$ ([[importance-sampling]]). Unbiased, but the product of many ratios can have enormous variance.
- **One-step action values need no ratio.** $\val{Q(s, a)}$ already conditions on the action taken, and the next action's choice is handled inside the target: Q-learning maximizes over it, Expected SARSA averages it under $\pol\pi$ ([[expected-sarsa]]).
- **Multi-step returns** need ratios again for the actions after the first, or tree backups that use $\pol\pi$'s probabilities instead ([[n-step-sarsa]]). In deep RL, Retrace and V-trace clip the ratios to keep the variance in check, accepting a little bias.

### Why off-policy is worth it {#why}

- **Reusing experience.** A replay memory holds transitions from many past policies ([[experience-replay]]). Learning from them is off-policy by nature, which is why DQN, DDPG, TD3 and SAC are built on off-policy targets.
- **Exploring freely.** The agent can explore as much as it likes while still learning about the greedy policy.
- **Learning from others.** Demonstrations, logs from an older system, another agent's play: all are data from some other behavior ([[offline-rl]]).
- **Learning many things at once.** One stream of experience can teach the values of many target policies in parallel.

### What it costs {#cost}

Variance, from importance ratios. And stability: off-policy data, function approximation and bootstrapping together can make estimates diverge, the **deadly triad** ([[deadly-triad]]). In Baird's counterexample, semi-gradient TD with all three lets its weights grow past 100 in all 20 of the Lab's runs. Replace the shared features with a table, or the off-policy updates with on-policy ones, and the weights stay small in all 20.

### In between {#between}

The line is not always sharp. **PPO** and **TRPO** collect a batch with the current policy, then take several steps on it, so after the first step the data comes from a slightly older policy. They correct with the probability ratio and keep it near 1, by a clip or a constraint ([[ppo]]). **A2C** is strictly on-policy: each batch is used once and thrown away ([[a2c]]). That is why on-policy methods need many fresh samples, and why they suit fast simulators where samples are cheap.

| method | on or off | reuses old data |
| --- | --- | --- |
| SARSA, Expected SARSA (as on-policy), MC control | on | no |
| REINFORCE, actor–critic, A2C | on | no |
| TRPO, PPO | nearly on: a few steps per batch | a little |
| Q-learning, Double Q, DQN and its extensions | off | yes: replay |
| DDPG, TD3, SAC | off | yes: replay |
| off-policy MC with importance sampling | off | yes |

## Card

### Idea

The behavior policy makes the data; the target policy is the one being learned about. On-policy: the same policy. Off-policy: different ones, which lets the agent reuse old experience, explore freely or learn from others, at a price in variance and stability.

::: analogy
Learning to drive by reviewing your own drives (on-policy), or by studying recordings of other drivers (off-policy). The recordings are plentiful, but they show what others did, not what you would have done.
:::

### Side by side {#side}

- **On-policy**: learns the value of how it actually behaves, exploration included. Simple and stable; needs fresh data.
- **Off-policy**: learns the value of another policy, usually the greedy one. Data-efficient through replay; needs coverage and care.

### Pitfalls

- Judging Q-learning by the greedy policy and SARSA by its behavior: they answer different questions.
- Reusing old batches in an on-policy method without a correction.
- Off-policy data plus function approximation plus bootstrapping: watch for diverging values.
- Behavior that never tries an action the target policy needs.

### Check yourself {#check}

::: question
On the cliff with ε = 0.1, why does Q-learning lose more per episode than SARSA, when its greedy policy is better?
---
Q-learning learns the greedy path along the edge, but it keeps exploring while it learns, and a random step there falls off the cliff. SARSA learns the value of its exploring behavior, which includes those falls, so it keeps away from the edge.
:::

::: question
Why can Q-learning learn from a replay memory without importance sampling, while off-policy Monte Carlo cannot?
---
A one-step action value conditions on the action taken, and the next action is handled by the max in the target, so no ratio is needed. A Monte Carlo return depends on every later action the behavior policy took, so it must be reweighted.
:::

::: question
Is PPO on-policy or off-policy?
---
Essentially on-policy: it collects data with the current policy and discards it after a few passes. Within those passes the policy has moved a little, so it corrects with probability ratios, and clips them to stay close to the policy that collected the data.
:::
