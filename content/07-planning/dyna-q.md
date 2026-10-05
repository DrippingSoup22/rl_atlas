+++
summary = "Q-learning with a notebook: after every real step, write down what happened, then replay n remembered moves to yourself and learn from them too. Experience is used many times over, so far fewer real steps are needed."
change = "After each real Q-learning update, record the transition in a model (state, action → reward, next state) and make n more Q-learning updates on transitions drawn at random from the model."
prereqs = ["q-learning", "models"]
lab = "dyna-maze"
sources = [
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., §8.2 and Example 8.1", url = "http://incompleteideas.net/book/the-book-2nd.html" },
  { text = "Sutton (1990), Integrated architectures for learning, planning, and reacting based on approximating dynamic programming, Proceedings of the 7th International Conference on Machine Learning", url = "https://doi.org/10.1016/B978-1-55860-141-3.50030-4" },
  { text = "Sutton (1991), Dyna, an integrated architecture for learning, planning, and reacting, SIGART Bulletin 2", url = "https://doi.org/10.1145/122344.122377" },
  { text = "Lin (1992), Self-improving reactive agents based on reinforcement learning, planning and teaching, Machine Learning 8", url = "https://doi.org/10.1007/BF00992699" },
  { text = "van Hasselt, Hessel & Aslanides (2019), When to use parametric models in reinforcement learning?, Advances in Neural Information Processing Systems 32", url = "https://arxiv.org/abs/1906.05243" },
]

[story]
scene = "grid"
env = "dyna-maze"
seed = 28
average = 30
range = 0.6
formula = '\step{1}{\textit{Model}(S, A) \leftarrow \rew{R}, S^\prime} \step{2}{\qquad \text{then } n \text{ times: } \val{Q(s,a)} \leftarrow \val{Q(s,a)} + \alp\,\big[\rew{r} + \gam \max_{a^\prime} \val{Q(s^\prime,a^\prime)} - \val{Q(s,a)}\big]}'

[story.runs]
fifty = { algorithm = "dyna-q", planning = 50, alpha = 0.1, epsilon = 0.1, gamma = 0.95, units = 50, name = "50 planning steps" }
five = { algorithm = "dyna-q", planning = 5, alpha = 0.1, epsilon = 0.1, gamma = 0.95, units = 50, name = "5 planning steps" }
none = { algorithm = "dyna-q", planning = 0, alpha = 0.1, epsilon = 0.1, gamma = 0.95, units = 50, name = "no planning (Q-learning)" }
+++

## Story

::: step {q = "zero", agent = "start"}
**The maze where only the gem pays**, again. Every value starts at 0, and this time the agent also keeps a **notebook**, its model: for every move it has tried, where it led and what it paid.
:::

::: step {run = "fifty", at = 0, play = 1, pace = 45, arrows = false, fog = true, formula = 1}
The first episode is a blind search. The fog marks the tiles the notebook knows nothing about; watch it lift as the agent explores. Between real steps it already replays remembered moves, the dashed sparks, but with no reward seen yet they teach nothing.
:::

::: step {run = "fifty", at = 1, arrows = false, fog = true}
It found the gem after 120 steps. That last move earned the first real reward, and the 50 imagined moves right after it already passed the news back a few steps: 5 values are now above zero.
:::

::: step {run = "fifty", at = 1, play = 1, pace = 140, arrows = false, fog = true, formula = 2}
Now the second episode. After **every** real step come 50 imagined ones, drawn from the notebook, each one a Q-learning update. The news spreads from the gem through the whole known part of the maze while the agent is still walking.
:::

::: step {run = "fifty", at = 2, fog = true}
The second episode took 21 steps, and the arrows now lead home from almost anywhere the agent has been. Real experience: 141 steps. Updates made from it: about 7000.
:::

::: step {run = "none", at = 2, fog = true}
Without planning, the same agent is plain [[q-learning]]: each real step teaches one update. After two episodes and over 2000 real steps, it has learned the values of just two moves, the last two before the gem.
:::

::: step {run = "five", at = 50, curves = ["none", "five", "fifty"], metric = "steps"}
Averaged over 30 runs: after the first episode, which is a random search for all three, even 5 planning steps per real step find the way in a handful of episodes. [Race them in the Lab](lab:dyna-maze).
:::

## Textbook

### Learning, planning and acting together {#idea}

Every real step tells the agent two things: how good the move turned out to be, and what the move does. Model-free methods keep only the first, folded into a value, and forget the transition itself. An agent that also keeps the second can come back to it later. Each real transition then feeds two paths ([[models]]): **direct reinforcement learning** updates the values from it right away, and **model learning** writes it down so that **planning** can replay it, and every other remembered transition, as many times as there is time for. **Dyna** is the name for an agent built this way, with acting, learning, model learning and planning all running together (\ref{fig-dyna}). **Dyna-Q** is the plainest version: Q-learning is the update for both real and replayed transitions, and the model is a table holding, for each state and action tried, the reward and next state last seen.

::: figure {#fig-dyna}
{{dyna-architecture}}
The Dyna architecture. Real experience improves the values directly (direct RL) and the model (model learning); the model produces simulated experience, which improves the values again (planning). Acting uses the values.
:::

Why replay a transition already learned from? Because what it teaches changes. When the agent first steps next to the goal, the value of the square it lands on is still 0, so the update teaches nothing; replayed after the goal has been found, the same transition passes the goal's value one step back, and replayed again later, further back still. Learning only from fresh experience, an agent must physically walk a path many times to push values along it; replaying lets it do the walking in its head. The price is trust: the replayed transitions are only as good as the model, while direct learning can never be misled that way. Dyna keeps both paths, so its values stay tied to real experience while planning multiplies what each experience is worth.

### The algorithm {#algorithm}

The model is a table indexed by state and action. In a deterministic environment, recording the last observed reward and next state is enough. Planning picks pairs at random among those already tried, so it never asks the model about a move the agent has never made.

::: algorithm {#alg-dyna} Tabular Dyna-Q
Input: a step size $\alp \in (0, 1]$, a small $\varepsilon > 0$ and a number of planning steps $n$
Set $\val{Q(s, a)}$ and $\textit{Model}(s, a)$ for every state and action ($\val{Q}$ to 0, the model to “unknown”)
Repeat forever:
  $S \leftarrow$ the current (nonterminal) state
  $A \leftarrow \varepsilon$-greedy in $S$ from $\val{Q}$
  Take $A$; observe $\rew{R}$ and $S'$
  $\val{Q(S, A)} \leftarrow \val{Q(S, A)} + \alp\,[\rew{R} + \gam \max_a \val{Q(S', a)} - \val{Q(S, A)}]$
  $\textit{Model}(S, A) \leftarrow \rew{R}, S'$ (assuming a deterministic environment)
  Repeat $n$ times:
    $S \leftarrow$ a random state already visited; $A \leftarrow$ a random action already taken in $S$
    $\rew{R}, S' \leftarrow \textit{Model}(S, A)$
    $\val{Q(S, A)} \leftarrow \val{Q(S, A)} + \alp\,[\rew{R} + \gam \max_a \val{Q(S', a)} - \val{Q(S, A)}]$
:::

With $n = 0$ this is Q-learning. Each real step costs $n + 1$ updates; in exchange, values spread through the known part of the state space $n$ times faster.

### Example: the Dyna maze {#example}

::: example {#ex-maze} Dyna maze (Sutton & Barto, Example 8.1)
A 6 × 9 grid with seven wall tiles. The agent starts each episode at S, on the left, and the episode ends when it reaches the goal G in the top right corner. Moves are deterministic, and a move into a wall or the edge does nothing. Only reaching the goal pays, $\rew{+1}$; future reward is discounted by $\gamma = 0.95$. All values start at 0, and the agent uses $\alpha = 0.1$ and $\varepsilon = 0.1$.
:::

::: figure {#fig-curves}
{{dyna-curves}}
Steps per episode in the Dyna maze for Dyna-Q with 0, 5 and 50 planning steps, averaged over 30 runs. The first episode, a random search of about 1000 steps for all three, is left out. Computed by the Lab. After Sutton & Barto, Figure 8.2.
:::

All three agents need about 1000 steps to stumble on the goal the first time; with every value at 0, they are walking at random. After that the curves separate (\ref{fig-curves}): without planning the agent needs some 25 episodes to settle on a short path, with 5 planning steps about 5, with 50 about 3. A snapshot in the middle of the second episode shows why (\ref{fig-midway}). The agent without planning learned exactly one thing from its first episode, the value of the move onto the goal, and in the second episode the good news creeps back at most one square each time the agent passes by. The planning agent has been replaying its whole first episode in the meantime, so the value of the goal has already flowed back along many of the routes it walked, and its greedy moves point the way from much of the maze.

::: figure {#fig-midway}
{{dyna-midway}}
The greedy action in every tile with a positive value, halfway through the second episode, without and with 50 planning steps; the dot is the agent. Computed by the Lab. After Sutton & Barto, Figure 8.3.
:::

### Experience, computation and wrong models {#tradeoffs}

Planning trades computation for experience: in the maze, with 50 planning steps per real step, the agent settles on a short path after about 1500 real steps on average, against about 5000 without planning. This is the right trade when real experience is slow, costly or dangerous, and computation is cheap. It also means the agent depends on its model being right:

- **Stochastic environments.** Remembering only the last outcome is wrong when outcomes are random. The model can count outcomes and sample from their frequencies, or planning can use expected updates over those frequencies.
- **Changing environments.** If the world changes, the model is wrong until the agent tries the changed moves again, and planning confidently spreads the old answer. When the change makes things worse, the agent discovers it as soon as it acts on the old belief, though finding a new way can still take long; when it makes things better, an agent that never revisits the old choices may never find out. [[dyna-q-plus]] adds a bonus for trying moves that have not been tried in a long time.
- **The order of planning.** Picking remembered pairs uniformly wastes most updates on pairs whose values do not change. [[prioritized-sweeping]] chooses the pairs that matter.

### Relatives {#relatives}

- **Experience replay.** Storing transitions and replaying them, as DQN does ([[dqn]]), is Dyna with a model that remembers transitions instead of summarizing them (Lin, 1992). A replay buffer can only answer about what has happened; a learned model can generalize, but can also be wrong (van Hasselt, Hessel & Aslanides, 2019).
- **Model-based deep RL.** Learning a neural network model and planning with it, in the background or at decision time, follows the same architecture at scale ([[model-based-deep]]).

### Historical remarks {#history}

Sutton (1990, 1991) introduced the Dyna architecture as an integration of learning, planning and reacting, building on the view of planning as learning from simulated experience. Lin (1992) proposed experience replay, closely related to planning with a nonparametric model.

## Card

### Idea

Q-learning with a notebook. After every real step, update as Q-learning does, write the transition in the notebook (this move, from here, led there and paid this), then replay $n$ remembered moves to yourself and update on those too. Each real step is reused many times as the values around it change.

::: analogy
After a day in a new city, lying in bed and going over the streets you walked: every replay links what you learned at the end of the day back to where you started.
:::

### The update {#update}

$$\val{Q(S,A)} \leftarrow \val{Q(S,A)} + \alp\,\big[\,\rew{R} + \gam \max_a \val{Q(S',a)} - \val{Q(S,A)}\,\big]$$

The Q-learning update, made once for the real transition and $n$ more times for transitions $(S, A, \rew{R}, S')$ replayed from the model.

### One change from Q-learning {#change}

| | updates per real step | from |
| --- | --- | --- |
| [[q-learning]] | 1 | the real transition |
| Dyna-Q | $1 + n$ | the real transition, then $n$ remembered ones |

### Backup diagram {#backup}

{{backup q-learning}}

The same one-step backup as Q-learning, for real and for simulated transitions.

### Pseudocode

::: pseudocode
Parameters: step size $\alp$, exploration $\eps$, discount $\gam$, planning steps $n$
Set $\val{Q(s,a)}$ (here 0) and an empty model {#init}
Repeat for each episode:
  Start: $S$ {#start}
  Repeat until $S$ is terminal:
    Choose $A$ $\eps$-greedily from $\val{Q}$ {#choose}
    Take $A$, observe $\rew{R}$ and $S'$ {#act}
    $\val{Q(S,A)} \leftarrow \val{Q(S,A)} + \alp\,[\rew{R} + \gam \max_a \val{Q(S',a)} - \val{Q(S,A)}]$ {#update}
    $\textit{Model}(S,A) \leftarrow \rew{R}, S'$ {#model}
    $n$ times: a remembered $(s, a)$ at random, $\rew{r}, s' \leftarrow \textit{Model}(s,a)$, and the same update on it {#plan}
    $S \leftarrow S'$ {#next}
:::

### Perks

- Far fewer real steps: each one is reused $n$ times. [See it](lab:dyna-maze)
- Simple: Q-learning plus a table, and the planning can run whenever there is time.
- Values spread back from rewards while the agent is still acting.

### Flaws

- $n + 1$ updates per real step: more computation.
- Plans with a model that may be wrong; a better path that appears may never be found ([[dyna-q-plus]]).
- Random planning order wastes updates on pairs that do not change ([[prioritized-sweeping]]).

### Knobs

| Knob | Too low | Too high |
| --- | --- | --- |
| $n$ planning steps | little better than Q-learning | more computation per step for little extra gain |
| $\alp$ step size | values spread slowly | noisy values (in deterministic worlds, $\alp = 1$ is fine) |
| $\eps$ exploration | the model stays incomplete | many wasted real steps |

### Pitfalls

- Planning from pairs never tried: the model has nothing to say about them.
- Storing only the last outcome in a stochastic world: the model then misrepresents the odds.
- Counting only real steps when comparing with Q-learning: Dyna-Q spends $n$ times more computation.

### Check yourself {#check}

::: question
With $n = 50$ planning steps, how many updates does Dyna-Q make in an episode of 20 real steps?
---
$20 \times (1 + 50) = 1020$: one direct update and 50 planning updates per real step.
:::

::: question
What is Dyna-Q with $n = 0$?
---
Q-learning: no planning, only the direct update after each real step. The model is kept but never used.
:::

::: question
Why does Dyna-Q pick remembered pairs only among those already tried?
---
Its model knows only the outcomes it has observed. For a pair never tried there is nothing to replay, and making something up would teach wrong values.
:::
