+++
summary = "The agent maximizes exactly what the reward says, not what you meant. Choosing the reward is choosing the problem."
prereqs = ["state-action-reward", "return", "optimality"]
sources = [
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., §3.2 and §17.4", url = "http://incompleteideas.net/book/the-book-2nd.html" },
  { text = "Ng, Harada & Russell (1999), Policy invariance under reward transformations: theory and application to reward shaping, Proceedings of the 16th International Conference on Machine Learning" },
  { text = "Randløv & Alstrøm (1998), Learning to drive a bicycle using reinforcement learning and shaping, Proceedings of the 15th International Conference on Machine Learning" },
  { text = "Clark & Amodei (2016), Faulty reward functions in the wild, OpenAI blog" },
  { text = "Mnih et al. (2015), Human-level control through deep reinforcement learning, Nature 518 (reward clipping)", url = "https://doi.org/10.1038/nature14236" },
  { text = "Christiano, Leike, Brown, Martic, Legg & Amodei (2017), Deep reinforcement learning from human preferences, Advances in Neural Information Processing Systems 30" },
]

[story]
scene = "grid"
env = "checkpoint"
seed = 1
average = 20
range = 2
formula = '''\step{1}{0.2\,(1 + \gam^2 + \gam^4 + \dots) = \frac{0.2}{1 - \gam^2} \approx 2.05 \qquad} \step{2}{F(s, s') = \gam\,\Phi(s') - \Phi(s)}'''

[story.runs]
naive = { algorithm = "q-learning", gamma = 0.95, alpha = 0.5, epsilon = 0.1, maxSteps = 200, units = 300, measures = [], name = "+0.2 on the checkpoint" }
shaped = { algorithm = "q-learning", gamma = 0.95, alpha = 0.5, epsilon = 0.1, maxSteps = 200, units = 300, measures = [], world = "checkpoint-shaped", name = "the same bonus, as a potential" }
+++

## Story

::: step {q = "zero", agent = "start"}
**A goal, and a helping hand.** Reaching the gem pays $\rew{+1}$, with rewards discounted by $\gam = 0.95$. The wall makes the way long, so the designer adds a hint: $\rew{+0.2}$ for stepping on the checkpoint ⚑, which lies on the way down past the wall. A little reward for progress should speed learning up.
:::

::: step {run = "naive", at = 300, values = true, path = true}
**After 300 episodes, the agent never reaches the gem.** It walks to the checkpoint and steps off and on it, again and again, until its 200 steps run out. It does that in all 20 runs. The values around the checkpoint are about 2, twice what the gem pays.
:::

::: step {run = "naive", at = 300, values = true, path = true, formula = 1}
**It did exactly what it was paid to do.** Stepping off and back on pays $\rew{0.2}$ every second step, forever: discounted, about 2.05, while the gem, ten moves away, is worth $\gam^{9} \approx 0.63$ from the checkpoint. This is not a learning failure: for this reward, circling is the optimal policy, and a perfect planner would circle too. The reward said "be on the checkpoint", not "get to the gem".
:::

::: step {run = "shaped", at = 300, values = true, path = true, formula = 2}
**The same hint, given as a potential.** Give the checkpoint a potential $\Phi = 0.2$, and pay $\gam\,\Phi(s') - \Phi(s)$ on every move: stepping on pays $0.95 \times 0.2 = 0.19$, stepping off costs $0.2$. A loop now loses a little each time around, and along any path from start to end the hints add up to nothing at all. In all 20 runs the agent walks to the gem by a shortest path, 14 steps.
:::

::: step {run = "shaped", at = 300, curves = ["naive", "shaped"], metric = "steps"}
**Twenty runs of each.** With the bonus, every episode runs to the 200-step limit once the loop is found. With the potential, the episodes shrink to the shortest path. Shaping of this form never changes which policies are optimal, a guarantee proved by Ng, Harada and Russell: it can only change how fast they are found.
:::

## Textbook

### The reward defines the problem {#hypothesis}

In reinforcement learning the goal is communicated to the agent through the reward signal alone ([[agent-environment]]). Whatever the designer intends, the agent will pursue what the reward actually measures, and a capable agent will pursue it more thoroughly than the designer anticipated. Choosing the reward is therefore not a detail of implementation; it is the specification of the task.

The first rule follows from this: **reward what you want achieved, not how you think it should be achieved** ([[state-action-reward]]). A chess agent rewarded for capturing pieces learns to capture pieces, even at the cost of the game. Knowledge about how to reach the goal is better given to the agent through its initial policy or value estimates, which experience can correct, than through the reward, which defines what counts as correct.

### Sparse and dense rewards {#sparse}

A **sparse** reward comes rarely, typically only when the goal is reached: $\rew{+1}$ for winning, 0 otherwise. It says exactly what is wanted, but it makes learning hard, because an agent that has never reached the goal has nothing to learn from, and random exploration may take very long to get there ([[explore-exploit]]).

A **dense** reward gives feedback at every step, for instance for progress toward the goal. It speeds up learning, but each extra term is an opportunity to reward the wrong thing. Randløv and Alstrøm (1998) rewarded a simulated bicycle for moving toward its goal and found that it learned to ride in circles near the start: approaching the goal was rewarded, and riding away again cost nothing, so loops were profitable.

### Potential-based shaping {#shaping}

There is a safe way to add dense guidance. **Shaping** adds a term $F$ to the reward of each transition, $\tilde r = \rew{r} + F(s, a, s')$. Ng, Harada and Russell (1999) found the form of $F$ that never changes which policies are optimal.

::: theorem {#thm-shaping} Ng, Harada & Russell, 1999
Let $\Phi$ be any bounded function of the state, a **potential**, and add to every transition the shaping reward
$$F(s, a, s') = \gam\,\Phi(s') - \Phi(s). \label{shaping-term}$$
Then in a discounted MDP with $\gam < 1$, every policy's action values change by the same amount for all actions in a state, $\tilde q_\pi(s,a) = \val{q_\pi(s,a)} - \Phi(s)$, so the optimal policies of the shaped and the original problem are the same.
:::

::: proof
Along any trajectory the shaping rewards telescope:
$$\sum_{k=0}^{n-1} \gam^k \big(\gam\,\Phi(S_{t+k+1}) - \Phi(S_{t+k})\big) = \gam^n\,\Phi(S_{t+n}) - \Phi(S_t) \;\longrightarrow\; -\Phi(S_t) \quad \text{as } n \to \infty,$$
because $\Phi$ is bounded and $\gam^n \to 0$. So the shaped return from $(S_t, A_t)$ is the original return minus $\Phi(S_t)$, whatever the actions. Taking expectations, $\tilde q_\pi(s,a) = \val{q_\pi(s,a)} - \Phi(s)$ for every policy. The shift depends on the state only, so in every state the ranking of the actions is unchanged, and so are the optimal policies.
:::

Ng, Harada and Russell also showed a converse: without further knowledge of the MDP, potential-based terms are the only shaping terms guaranteed to preserve optimal policies. A natural potential is an estimate of the state's value: with $\Phi(s)$ equal to minus the distance to the goal, moving one step closer earns a small bonus and moving away a small penalty, and a loop earns exactly nothing, which rules out the bicycle's circles.

### Specification gaming {#gaming}

When the reward and the intention differ, agents find the difference. Clark and Amodei (2016) describe an agent trained on a boat-racing game, rewarded by the game's score: it found a lagoon where it could circle and hit the same targets again and again, scoring more than by finishing the race. Such behavior is called **specification gaming** or reward hacking. It is not a malfunction: the agent is doing what it was asked. The remedies are a better reward, constraints on behavior, or learning the reward from people instead of writing it by hand.

### Scale {#scale}

The scale of the rewards does not change which policy is optimal (multiplying all rewards by a positive constant multiplies all values by it), but it changes how learning behaves: large rewards produce large errors and large updates. DQN clipped every reward to $[-1, 1]$ so that one set of hyperparameters worked across all Atari games (Mnih et al., 2015). Clipping does change the problem, since the agent can no longer tell a large reward from a small one, and later methods often normalize returns instead ([[normalization]]).

### Learning the reward {#learning}

When a good reward is hard to write down, it can be learned. **Inverse reinforcement learning** infers a reward from demonstrations of the desired behavior ([[imitation]]). **Learning from human preferences** shows a person pairs of behaviors, asks which is better, fits a reward model to the answers and trains the agent on that model (Christiano et al., 2017). The same idea, applied to language models, is RLHF ([[rlhf]]).

## Card

### Idea

The agent maximizes exactly what the reward measures, so the reward must describe the **goal**, not a method. Rewards that come rarely are honest but hard to learn from; frequent ones speed learning but can reward the wrong thing. Shaping with a potential, $\gam\Phi(s') - \Phi(s)$, adds guidance without changing what is optimal.

::: analogy
Paying a programmer per line of code. You get plenty of lines.
:::

### In symbols {#formula}

$$\tilde r = \rew{r} + \gam\,\Phi(s') - \Phi(s) \quad\Longrightarrow\quad \tilde q_\pi(s,a) = \val{q_\pi(s,a)} - \Phi(s)$$

### Rules of thumb {#rules}

| Do | Avoid |
| --- | --- |
| reward the outcome you want | rewarding subgoals or methods |
| shape with a potential $\gam\Phi(s') - \Phi(s)$ | bonuses that can be collected in a loop |
| watch what the agent actually does | trusting the reward curve alone |

### Why it matters {#why}

Every algorithm in the atlas assumes the reward is right. If it is not, a better algorithm only finds the loophole faster.

### Pitfalls

- A bonus for being near the goal: the agent may stay near it rather than reach it, if reaching it ends the bonus.
- Per-step bonuses in episodic tasks: the agent learns to never finish.
- Clipping rewards without noticing that big and small rewards now look the same.

### Check yourself {#check}

::: question
Why does a shaping term $\gam\Phi(s') - \Phi(s)$ not change the optimal policy?
---
Along any path the terms telescope, so every return from a state shifts by the same $-\Phi(s)$, whatever the actions. The ranking of actions stays the same.
:::

::: question
A cleaning robot is rewarded for every piece of dirt it picks up. What could go wrong?
---
It may learn to spill dirt and pick it up again. The reward pays for an activity, not for a clean room.
:::

::: question
In the checkpoint world the gem pays 1, and entering the checkpoint pays 0.2 each time. With $\gam = 0.95$, why is stepping off and on forever worth more than the gem?
---
Starting next to the checkpoint, the loop pays 0.2 every second step: $0.2\,(1 + \gamma^2 + \gamma^4 + \dots) = 0.2/(1 - \gamma^2) \approx 2.05$, more than twice the gem's single payment. For that reward, circling forever is the optimal policy.
:::
