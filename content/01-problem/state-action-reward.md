+++
summary = "The three signals of the loop: what the agent is told, what it does, and the single number that grades the result."
story_in = ["agent-environment", "mdp"]
prereqs = ["agent-environment"]
sources = [
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., §3.1 and §3.2", url = "http://incompleteideas.net/book/the-book-2nd.html" },
  { text = "Mnih et al. (2015), Human-level control through deep reinforcement learning, Nature 518 (frame stacking)", url = "https://doi.org/10.1038/nature14236" },
]
+++

## Textbook

### States {#states}

The state $S_t \in \mathcal{S}$ is the information on which the agent bases its decision at time $t$. It can be almost anything: the position of the pieces on a board, a vector of sensor readings, a summary of what has happened so far, or a symbolic description such as “battery low”. The set $\mathcal{S}$ may be finite (the 25 cells of a small gridworld) or continuous (positions and velocities, a vector in $\mathbb{R}^d$).

A good state contains everything that matters for the decision and for predicting what happens next. When it does, the past adds nothing once the present state is known; this is the Markov property, made precise in [[mdp]]. When it does not, the agent must make up for the missing information, typically by remembering recent history. The classic example is a single video frame, which shows where a ball is but not where it is going; DQN therefore used the last four frames together as its state (Mnih et al., 2015).

It helps to distinguish what the agent senses, its **observation**, from the state it uses. In most of this atlas the two coincide: the environment hands the agent a state that is already sufficient. In episodic tasks the set $\mathcal{S}^+$ adds the terminal state, in which an episode ends ([[episodes]]).

### Actions {#actions}

The action $A_t \in \mathcal{A}(S_t)$ is the agent's choice at time $t$. The set of available actions may depend on the state, as legal moves depend on the position in chess. It may be finite (left or right, one of four directions) or continuous (a torque, a steering angle); continuous actions need policies that output distributions over real numbers ([[policy-parameterization]]).

Actions can be chosen at very different levels of abstraction: the voltages applied to a motor, the direction of the next step, or whether to accept a job offer. The level is a design decision. Low-level actions give fine control but make useful behavior a long sequence of decisions; high-level actions shorten the sequence but presuppose controllers that can carry them out.

Every action has two effects that the agent must weigh: on the reward that follows, and on the state it leads to, and through that state on all later rewards.

### Rewards {#rewards}

The reward $\rew{R_{t+1}} \in \mathbb{R}$ is a single number that the environment emits after each action. Its sole role is to define the goal: the agent's objective is to maximize the rewards it receives in total ([[return]]). It is part of the environment, outside the agent's control.

A reward should say **what** to achieve, not **how** to achieve it. Sutton and Barto give the example of chess: an agent should be rewarded for winning, not for subgoals such as capturing pieces, or it may learn to capture pieces even at the cost of losing the game. Knowledge about how to reach the goal belongs in the agent's initial policy or value estimates, not in the reward ([[reward-design]]).

The sign and the scale of rewards matter in specific ways.

- A reward of $\rew{-1}$ on every step, as in the cliff world, makes every extra step costly, so the agent learns to finish quickly.
- Multiplying every reward by the same positive constant does not change which policy is best; it only rescales the values, which matters for step sizes and for learning with neural networks.
- Adding a constant $c$ to every reward does not change which policy is best in a continuing task: with discount $\gam < 1$ it adds the same amount, $c/(1-\gam)$, to the value of every state. In an episodic task it can change it: a positive bonus on every step makes long episodes attractive, and a maze agent rewarded $+1$ per step would learn to avoid the exit.

The expected reward of an action, $r(s,a) = \mathbb{E}[\rew{R_{t+1}} \mid S_t = s, A_t = a]$, is often all that matters, so rewards can be random without changing the problem much ([[mdp]]). Texts write the reward function in three forms, $r(s)$, $r(s, a)$ or $r(s, a, s')$, depending on what it is allowed to depend on. They describe the same problems: the richer forms can always be averaged down to $r(s, a)$, which is all the Bellman equations use, and a reward that depends on more can be turned into $r(s)$ by putting the extra information into the state.

### Timing {#timing}

The order of events within a step is fixed: the agent sees $S_t$, chooses $A_t$, and the environment answers with $\rew{R_{t+1}}$ and $S_{t+1}$ together. A short excerpt of a cliff-world trajectory, for an agent walking along the edge:

| $t$ | $S_t$ | $A_t$ | $\rew{R_{t+1}}$ | $S_{t+1}$ |
| --- | --- | --- | --- | --- |
| 0 | start | up | $\rew{-1}$ | one cell above the start |
| 1 | one cell above the start | right | $\rew{-1}$ | the next cell along the edge |
| 2 | the next cell along the edge | down | $\rew{-100}$ | start (it fell) |

The reward of the fall is $\rew{R_3}$, the consequence of $A_2$. Every update rule in the atlas is written with this convention: for example, [[sarsa]] updates the value of $(S_t, A_t)$ with $\rew{R_{t+1}}$.

### Designing the three {#design}

Formulating a problem for reinforcement learning means choosing the three signals, and each choice is a trade-off.

| Choice | More of it gives | At the cost of |
| --- | --- | --- |
| Richer states | more information for each decision | more to learn, and less generalization between situations |
| Finer actions | more precise control | longer chains of decisions before anything pays off |
| More frequent rewards | faster feedback | a greater risk of rewarding the wrong thing |

## Card

### Idea

Three signals pass between agent and environment. The **state** is what the agent is told about the situation, the **action** is what it does, and the **reward** is a single number that grades the result. The reward defines the goal: what to achieve, never how.

### In symbols {#formula}

$$S_t \in \mathcal{S}, \qquad A_t \in \mathcal{A}(S_t), \qquad \rew{R_{t+1}} \in \mathbb{R}$$

The reward and the next state that follow $A_t$ carry the index $t+1$.

### Examples {#examples}

| | State | Action | Reward |
| --- | --- | --- | --- |
| Cliff world | the cell | up, down, left, right | $\rew{-1}$ a step, $\rew{-100}$ for a fall |
| Chess | the position | a legal move | win $\rew{+1}$, loss $\rew{-1}$ |
| Atari game | the last 4 frames | a joystick input | the change in score |

### Why it matters {#why}

The same learning algorithm can succeed or fail depending only on these choices. A state that hides what matters, or a reward that pays for the wrong thing, defeats any algorithm.

### Pitfalls

- Rewarding subgoals instead of the goal: the agent learns the subgoal, even at the expense of the goal ([[reward-design]]).
- A state that leaves out what matters, such as a single frame that shows a ball but not its direction.
- A constant bonus per step in an episodic task: it changes what is best, and the agent may learn to never finish.

### Check yourself {#check}

::: question
Why might a chess agent rewarded for capturing pieces play worse than one rewarded only for winning?
---
It learns to capture pieces, even when giving up the game to do so. The reward should describe the goal, not a way to reach it.
:::

::: question
Adding 5 to every reward of a continuing task with $\gam = 0.9$: what happens to the values and to the best policy?
---
Every value grows by $5/(1-0.9) = 50$. The comparison between actions is unchanged, so the best policy stays the same.
:::

::: question
Why did DQN use the last four frames of an Atari game as its state?
---
A single frame does not show motion. Four frames together reveal where objects are heading, which the decision needs.
:::
