+++
summary = "An agent acts, an environment answers with a reward and a new situation, and the loop repeats: every method in the atlas lives in this loop."
sources = [
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., §1.1, §3.1 and Figure 3.1", url = "http://incompleteideas.net/book/the-book-2nd.html" },
  { text = "OpenAI (2018), Spinning Up in Deep RL, Part 1: Key Concepts in RL", url = "https://spinningup.openai.com/en/latest/spinningup/rl_intro.html" },
  { text = "Abel, Dabney, Harutyunyan, Ho, Littman, Precup & Singh (2021), On the expressivity of Markov reward, Advances in Neural Information Processing Systems 34", url = "https://arxiv.org/abs/2111.00876" },
  { text = "Bowling, Martin, Abel & Dabney (2023), Settling the reward hypothesis, Proceedings of the 40th International Conference on Machine Learning", url = "https://arxiv.org/abs/2212.10420" },
]

[story]
scene = "loop"
+++

## Story

::: step {env = false, wires = false}
**Reinforcement learning is learning by doing.** The learner, and the one who decides, is called the **agent**. Nobody tells it what to do. It tries things and sees what happens.
:::

::: step {wires = false}
Everything outside the agent is the **environment**. Here it is a short corridor with a gem at the far end. The agent does not know the rules of the corridor; it only sees what the environment shows it.
:::

::: step {phase = "observe"}
Time moves in steps, $t = 0, 1, 2, \ldots$ At each step the environment tells the agent its **state** $S_t$: here, which cell it is in.
:::

::: step {phase = "act"}
The agent answers with an **action** $A_t$: left or right. The rule it uses to choose is its **policy** $\pol{\pi}$. This one goes right three times out of four.
:::

::: step {phase = "respond"}
One step later the environment answers with two things: a **reward** $\rew{R_{t+1}}$, a single number ($\rew{-1}$ for every step, $\rew{+10}$ at the gem), and the next state $S_{t+1}$.
:::

::: step {phase = "loop", trajectory = true}
Then the loop goes round again. Writing down what happens gives a **trajectory**: $S_0, A_0, \rew{R_1}, S_1, A_1, \rew{R_2}, \ldots$ Everything an RL agent learns, it learns from streams like this one.
:::

::: step {phase = "loop", trajectory = true, boundary = true}
Where does the agent end? Anything it cannot change at will belongs to the environment: a robot's motors and sensors, and even the calculation of its own reward. The agent is only the part that decides.
:::

::: step {phase = "loop", trajectory = true, tally = true}
The goal is not the next reward but the **total** over time. Every step right costs $\rew{-1}$ now and brings the $\rew{+10}$ closer. Accepting small costs now for a bigger gain later is the heart of reinforcement learning; [[return]] makes “the total” precise.
:::

## Textbook

### The interface {#interface}

Reinforcement learning studies a learner that interacts with its surroundings over time and tries to influence them to its advantage. The interaction is cut into discrete time steps $t = 0, 1, 2, \ldots$ At each step the **agent** receives a description of the situation, the **state** $S_t \in \mathcal{S}$, and selects an **action** $A_t \in \mathcal{A}(S_t)$ from the actions available in that state. One step later, partly as a consequence of that action, the **environment** produces a numerical **reward** $\rew{R_{t+1}} \in \mathbb{R}$ and a new state $S_{t+1}$, and the cycle repeats (\ref{fig-loop}). The steps need not be equal slices of clock time: they are successive decisions, a motor command every millisecond or a choice of route every day, and the framework is the same.

::: figure {#fig-loop}
{{loop}}
The agent–environment interaction. The agent sends an action; the environment returns a reward and the next state. After Sutton & Barto, Figure 3.1.
:::

The record of an interaction is a **trajectory**,

$$S_0,\ A_0,\ \rew{R_1},\ S_1,\ A_1,\ \rew{R_2},\ S_2,\ A_2,\ \rew{R_3},\ \ldots \label{trajectory}$$

The indices follow one convention throughout the atlas: $\rew{R_{t+1}}$ and $S_{t+1}$ are produced together, by the environment, in response to $A_t$. The subscript $t+1$ says that they belong to the next step, not that the reward comes from the next action. Some texts write $R_t$ for the same reward; nothing else changes.

How the agent maps states to actions is its **policy** ([[policy]]), $\pol{\pi(a \mid s)}$, the probability of choosing $a$ in $s$. How the environment produces rewards and next states is described by its dynamics, which the agent usually does not know ([[mdp]]). Learning means changing the policy on the basis of experience so that the rewards collected improve.

### Where the agent ends {#boundary}

The line between agent and environment is not the skin of a robot or an animal. The convention is: **anything the agent cannot change arbitrarily belongs to the environment**. The motors and joints of a robot, its sensors, its battery, and the mechanism that computes its reward are all part of the environment, even when they are physically inside the robot. The agent is the decision-making part alone.

The boundary marks the limit of the agent's control, not of its knowledge. An agent may know exactly how its environment works, as when the environment is a Rubik's cube or a board game, and still face a hard problem: knowing the rules of chess does not make one a strong player. Conversely, an agent may know almost nothing about its environment and still learn to act well in it; that is the usual case in this atlas.

The boundary is also a choice of the designer. One physical system can contain several agents: a high-level agent may choose goals that become part of the environment of a low-level agent controlling the motors.

### The goal {#goal}

The agent's purpose is expressed through the reward alone. At each step the reward is a single number, and the agent's aim is to maximize the reward it receives **in total over time**, not at the next step. Sutton and Barto call the claim that this is enough the *reward hypothesis*: whatever we mean by a goal can be expressed as the maximization of the expected cumulative sum of a scalar signal. The hypothesis is a modeling choice, a strong one, and much of the art of applying reinforcement learning lies in choosing rewards that express the intended goal ([[reward-design]]). The hypothesis also has limits that can be stated exactly. Some goals, written as a set of acceptable behaviors or as a ranking of outcomes, cannot be captured by any reward that depends only on the current state and action (Abel et al., 2021); Bowling et al. (2023) give the precise conditions on an agent's preferences under which a cumulative scalar reward is enough.

“Total over time” is made precise by the return ([[return]], [[discount]]), and “expected” by value functions ([[value-functions]]).

### What makes the problem hard {#hard}

Three features set reinforcement learning apart from learning from examples.

- **Evaluative feedback.** The reward says how good the action taken was, not which action would have been best. A supervised learner is shown the correct answer; a reinforcement learner must try alternatives to find out whether something better exists. This is the exploration–exploitation dilemma ([[explore-exploit]]).
- **Delayed consequences.** An action influences the next state, and through it all later rewards. A reward may be the consequence of a decision taken many steps earlier, so the agent must assign credit across time.
- **Experience depends on behavior.** What the agent observes depends on how it acts. The data are not a fixed sample: changing the policy changes the data, which is why learning and acting cannot be separated.

### Examples {#examples}

| Problem | State | Action | Reward |
| --- | --- | --- | --- |
| Playing chess | the position of the pieces | a legal move | +1 for a win, −1 for a loss, 0 otherwise |
| A robot learning to walk | joint angles and velocities, body orientation | torques at the joints | distance moved forward, minus a little for energy used |
| Cooling a data center | temperatures, loads, the time of day | fan and chiller settings | minus the energy cost, minus a penalty for overheating |
| A recycling robot ([[mdp]]) | the battery level | search, wait, or go recharge | cans collected; a penalty if the battery runs flat |

In each case the same three signals pass between agent and environment; only their meaning changes. That generality is the strength of the framework, and also its limitation: the framework says nothing about how states should be represented, which is often the hardest part in practice.

### Remarks {#remarks}

The interface is the same one control engineers use, with other names: their *controller* is the agent, their *plant* is the environment, and their *control signal* is the action. Where control theory usually minimizes a cost, reinforcement learning maximizes a reward; a reward is a negative cost.

In many problems the agent does not receive the full state of the environment but only a partial **observation**, as a poker player sees only their own cards. The atlas mostly assumes that the state given to the agent contains everything it needs to decide; [[mdp]] states this assumption precisely, and explains what changes when it fails.

## Card

### Idea

The agent acts; the environment answers with a reward and the next state; the loop repeats. Everything an agent learns, it learns from this stream of states, actions and rewards, and its goal is the **total** reward over time, not the next one.

::: analogy
Learning a video game without the manual. You press buttons, watch the screen and the score, and slowly work out which moves pay off, including moves whose reward only comes much later.
:::

### In symbols {#formula}

One step: the agent sees $S_t$, picks $A_t$, and the environment answers with $\rew{R_{t+1}}$ and $S_{t+1}$. A whole interaction is a trajectory:

$$S_0,\ A_0,\ \rew{R_1},\ S_1,\ A_1,\ \rew{R_2},\ S_2,\ \ldots$$

### Who is who {#who}

| | |
| --- | --- |
| Agent | the part that decides: its policy $\pol{\pi}$ |
| Environment | everything else, including the robot's body and the reward calculation |
| State $S_t$ | what the agent is told about the situation |
| Action $A_t$ | what the agent does |
| Reward $\rew{R_{t+1}}$ | one number grading the last step |

### Why it matters {#why}

Every algorithm in the atlas, from [[epsilon-greedy]] to [[ppo]], is a way of choosing actions in this loop and improving the choice from what comes back. Only the representation of states and the way of learning change.

### Pitfalls

- The reward grades the action; it does not say which action would have been better. The agent has to find that out by trying ([[explore-exploit]]).
- The boundary is about control, not about the body: a robot's motors and its reward computation are part of the environment.
- Maximizing the next reward is not the goal. An agent that only looks one step ahead will not walk toward a distant gem.

### Check yourself {#check}

::: question
In the convention of the atlas, which action produced the reward $\rew{R_5}$?
---
$A_4$. The reward and the next state that follow an action carry the next time index.
:::

::: question
A robot's battery is inside its body. Is it part of the agent or of the environment?
---
The environment. The agent cannot change the battery level at will; it can only take actions that affect it.
:::

::: question
Why can't an agent simply be told the correct action, as in supervised learning?
---
Often nobody knows it. The reward only says how good the outcome was, so the agent must try alternatives and compare.
:::
