+++
summary = "Plan where it matters: keep a queue of the moves whose values would change the most, update the most urgent first, and when a value changes, queue the moves that lead into it. The news of a reward sweeps backward along the ways it can be reached."
change = "Instead of replaying remembered moves at random, replay them in order of priority, the size of the change their update would make, and after each update queue the moves predicted to lead into the state that changed."
prereqs = ["dyna-q", "models"]
lab = "sweeping"
sources = [
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., §8.4 and Example 8.4", url = "http://incompleteideas.net/book/the-book-2nd.html" },
  { text = "Moore & Atkeson (1993), Prioritized sweeping: reinforcement learning with less data and less real time, Machine Learning 13", url = "https://doi.org/10.1007/BF00993104" },
  { text = "Peng & Williams (1993), Efficient learning and planning within the Dyna framework, Adaptive Behavior 1", url = "https://doi.org/10.1177/105971239300100403" },
  { text = "van Seijen & Sutton (2013), Planning by prioritized sweeping with small backups, Proceedings of the 30th International Conference on Machine Learning", url = "https://arxiv.org/abs/1301.2343" },
  { text = "Schaul, Quan, Antonoglou & Silver (2016), Prioritized experience replay, International Conference on Learning Representations", url = "https://arxiv.org/abs/1511.05952" },
]

[story]
scene = "grid"
env = "dyna-maze"
seed = 46
average = 30
range = 0.6
formula = '\step{1}{P = \big|\,\rew{R} + \gam \max_a \val{Q(S^\prime,a)} - \val{Q(S,A)}\,\big|} \step{2}{\qquad \text{update the largest } P \text{ first; then queue the moves into its state}}'

[story.runs]
ps = { algorithm = "prioritized-sweeping", planning = 5, alpha = 0.5, epsilon = 0.1, gamma = 0.95, units = 30, name = "prioritized sweeping" }
dq = { algorithm = "dyna-q", planning = 5, alpha = 0.5, epsilon = 0.1, gamma = 0.95, units = 30, name = "Dyna-Q, random order" }
+++

## Story

::: step {q = "zero", agent = "start"}
**The maze where only the gem pays**, and an agent with a notebook, as in [[dyna-q]], allowed 5 planning updates per real step. Dyna-Q spends them on remembered moves picked at random. But early on, almost every remembered move has a value of 0 leading to a state worth 0: replaying it changes nothing.
:::

::: step {run = "ps", at = 1, arrows = false, formula = 1}
Prioritized sweeping keeps a **queue**. A move gets in only if its update would change its value, and the bigger the change, the higher its priority. For the whole first episode the queue is empty: no surprises. Then the gem, and the first entry.
:::

::: step {run = "ps", at = 1, play = 1, pace = 160, fine = true, arrows = false, formula = 2}
Now watch the dashed sparks in the second episode: they no longer pop up at random. Each update of a value queues the remembered moves that **lead into** that state, so the sparks march backward from the gem, along the ways the agent has come.
:::

::: step {run = "ps", at = 2}
After two episodes, 52 action values have learned something, and the arrows lead to the gem from most of the maze the agent has seen. The second episode took 37 steps.
:::

::: step {run = "dq", at = 2}
Dyna-Q with the same budget of 5 updates per step, picked at random: after two episodes, 16 values have learned something. Most of its planning updates were spent where nothing could change.
:::

::: step {run = "ps", at = 30, curves = ["dq", "ps"], metric = "steps"}
Steps per episode, averaged over 30 runs, with the same number of planning updates for both. [Race them in the Lab](lab:sweeping).
:::

## Textbook

### Focusing the updates {#idea}

Watch [[dyna-q]] plan early in learning and most of its effort is wasted. It replays remembered moves at random, and almost all of them lead from a square worth 0 to another square worth 0: the update computes 0 and writes 0. The only replays that teach anything are those next to the goal, and later those next to squares that have just gained a value. In a 54-square maze that is a small fraction of the replays; in a maze of a million squares it would be almost none.

Yet the useful replays are easy to find, because news travels backward. A pair's target is built from the value of the state it leads to, so its update can change anything only after that state's value has changed. Whenever a value changes, then, the pairs worth updating next are the ones that lead *into* that state, and once they change, the ones that lead into theirs. **Prioritized sweeping** follows this chain with a priority queue. A pair enters the queue when its update would move its value by more than a small threshold $\theta$, ranked by how much, and planning always takes the most urgent pair first. The result looks like a wave: values spread outward from wherever something surprising happened, along every route that leads there, and stop spreading where the changes become negligible.

### The algorithm {#algorithm}

The priority of a pair is the size of its TD error under the model,

$$P(s, a) = \Big|\,\rew{r} + \gam \max_{a'} \val{Q(s', a')} - \val{Q(s, a)}\,\Big|, \qquad \rew{r}, s' = \textit{Model}(s, a). \label{eq-priority}$$

After each real step the pair just taken is queued if its priority exceeds $\theta$. Then up to $n$ times: take the pair with the highest priority off the queue, update it, and recompute the priorities of all pairs predicted to lead into its state, queueing those above $\theta$.

::: algorithm {#alg-ps} Prioritized sweeping for a deterministic environment
Input: a step size $\alp$, a small $\varepsilon > 0$, a threshold $\theta \ge 0$ and a number of planning steps $n$
Set $\val{Q(s, a)}$ to 0, $\textit{Model}(s, a)$ to “unknown” for every pair, and start with an empty queue
Repeat forever:
  $S \leftarrow$ the current (nonterminal) state; $A \leftarrow \varepsilon$-greedy in $S$ from $\val{Q}$
  Take $A$; observe $\rew{R}$ and $S'$; $\textit{Model}(S, A) \leftarrow \rew{R}, S'$
  $P \leftarrow |\rew{R} + \gam \max_a \val{Q(S', a)} - \val{Q(S, A)}|$; if $P > \theta$, queue $(S, A)$ with priority $P$
  Repeat $n$ times, while the queue is not empty:
    $(S, A) \leftarrow$ the first pair of the queue, removed from it; $\rew{R}, S' \leftarrow \textit{Model}(S, A)$
    $\val{Q(S, A)} \leftarrow \val{Q(S, A)} + \alp\,[\rew{R} + \gam \max_a \val{Q(S', a)} - \val{Q(S, A)}]$
    For every $(\bar S, \bar A)$ predicted to lead to $S$, with predicted reward $\bar R$:
      $P \leftarrow |\bar R + \gam \max_a \val{Q(S, a)} - \val{Q(\bar S, \bar A)}|$; if $P > \theta$, queue $(\bar S, \bar A)$ with priority $P$
:::

A pair already in the queue keeps the higher of its two priorities. The predecessors of a state are read off the model; keeping a list of them for each state makes this step cheap.

### Example: the Dyna maze {#example}

::: figure {#fig-curves}
{{sweeping-curves}}
Steps per episode in the Dyna maze for Dyna-Q and prioritized sweeping, both with 5 planning updates per real step, $\alpha = 0.5$, $\varepsilon = 0.1$ and $\gamma = 0.95$, averaged over 30 runs. The first episode is left out. Computed by the Lab.
:::

With the same number of planning updates per step, prioritized sweeping finds a short path in fewer episodes (\ref{fig-curves}). Counting updates instead of episodes makes the gap plainer. In 20 runs in the Lab, Dyna-Q needed a median of about 10 500 updates, real and planned, before its greedy path was the shortest one, 14 steps; prioritized sweeping needed about 1100 with $\alpha = 0.5$, and about 175 with $\alpha = 1$. The step size matters here because the world is deterministic: with $\alpha = 1$ each pair taken from the queue gets its final value in a single update, while with smaller steps the same pair has to come back to the queue several times. The bigger the maze, the larger the share of Dyna-Q's random replays that change nothing, so the gap grows with the size of the problem.

### Stochastic environments {#stochastic}

In a stochastic environment the model keeps counts of the outcomes of each pair, and planning can use **expected updates** over the estimated distribution:

$$\val{Q(s, a)} \leftarrow \sum_{s', r} \hat p(s', r \mid s, a)\,\Big[\rew{r} + \gam \max_{a'} \val{Q(s', a')}\Big]. \label{eq-expected}$$

Each such update costs as many computations as there are outcomes, many of them unlikely; this is the trade-off of expected and sample updates discussed in [[models]]. **Small backups** (van Seijen & Sutton, 2013) update a pair from a single successor at a time, weighted by its probability, getting the focus of prioritized sweeping at the cost of sample updates without their noise.

### Other ways to focus {#focus}

Prioritized sweeping focuses backward from changes. **Forward focusing** instead concentrates on states reachable soon under the current policy, as trajectory sampling and real-time dynamic programming do ([[models]]). Prioritizing replays by surprise carried over to deep RL as **prioritized experience replay** (Schaul et al., 2016), where transitions in a DQN replay buffer are sampled with probability growing with their last TD error ([[dqn-extensions]]).

### Historical remarks {#history}

Prioritized sweeping was developed independently and at the same time by Moore and Atkeson (1993), who named it, and by Peng and Williams (1993), whose version is called Queue-Dyna. Van Seijen and Sutton (2013) introduced small backups. Schaul, Quan, Antonoglou and Silver (2016) brought prioritization to experience replay.

## Card

### Idea

Plan where it matters. Keep a queue of remembered moves whose values would change if updated, the biggest change first. Each time a value changes, queue the moves that lead into that state. The news of a reward sweeps backward along every way of reaching it, and no update is wasted where nothing would change.

::: analogy
When a road closes, the traffic service updates the junctions next to it first, then the junctions leading to those, instead of recomputing the whole city map at random.
:::

### The update {#update}

$$\val{Q(S,A)} \leftarrow \val{Q(S,A)} + \alp\,\big[\,\rew{R} + \gam \max_a \val{Q(S',a)} - \val{Q(S,A)}\,\big]$$

The Q-learning update on remembered moves, in order of priority $P = |\rew{R} + \gam \max_a \val{Q(S',a)} - \val{Q(S,A)}|$.

### One change from Dyna-Q {#change}

| | which remembered moves are replayed |
| --- | --- |
| [[dyna-q]] | $n$ at random among those tried |
| prioritized sweeping | up to $n$ from a queue, the largest expected change first; predecessors are queued after each update |

### Backup diagram {#backup}

{{backup q-learning}}

The same one-step backup; only the order of the planning updates changes.

### Pseudocode

::: pseudocode
Parameters: step size $\alp$, exploration $\eps$, discount $\gam$, planning steps $n$, threshold $\theta$
Set $\val{Q(s,a)}$ (here 0), an empty model and an empty queue {#init}
Repeat for each episode:
  Start: $S$ {#start}
  Repeat until $S$ is terminal:
    Choose $A$ $\eps$-greedily from $\val{Q}$ {#choose}
    Take $A$, observe $\rew{R}$ and $S'$ {#act}
    $\textit{Model}(S,A) \leftarrow \rew{R}, S'$ {#model}
    $P \leftarrow |\rew{R} + \gam \max_a \val{Q(S',a)} - \val{Q(S,A)}|$; if $P > \theta$, queue $(S,A)$ with priority $P$ {#priority}
    Up to $n$ times: pop the first pair, update it, and queue each pair leading into its state whose priority exceeds $\theta$ {#plan}
    $S \leftarrow S'$ {#next}
:::

### Perks

- No planning update is spent where nothing would change. [See it](lab:sweeping)
- Far fewer updates than Dyna-Q to reach the optimal policy, more so in bigger worlds.
- The queue empties when the values are consistent with the model: planning stops by itself.

### Flaws

- More bookkeeping: a priority queue and the predecessors of every state.
- With stochastic dynamics, expected updates can be costly.
- Focuses on what the model says; like Dyna-Q, it plans confidently with a wrong model.

### Knobs

| Knob | Too low | Too high |
| --- | --- | --- |
| $n$ planning steps | the sweep lags behind the agent | more computation per step |
| $\theta$ threshold | tiny, useless updates fill the queue | real changes never get queued |
| $\alp$ step size | each pair needs many visits to the queue | (in deterministic worlds, $\alp = 1$ is fine) |

### Pitfalls

- Forgetting to queue the predecessors: the sweep stops after one step.
- Keeping a pair in the queue twice: keep it once, with its highest priority.
- Expecting the first episode to be faster: with no reward seen, the queue is empty and nothing is planned.

### Check yourself {#check}

::: question
At the start, all values are 0 and the only reward is at the goal. What does prioritized sweeping plan during the first episode?
---
Nothing: every remembered move has a TD error of 0, so its priority is 0 and nothing enters the queue. The first entry is the move onto the goal.
:::

::: question
After the value of a state changes, which pairs need new priorities, and why only those?
---
The pairs predicted to lead into that state: only their targets contain its value. Every other pair's update would change exactly as much as before.
:::

::: question
Why does prioritized sweeping need far fewer updates than Dyna-Q, especially in large worlds?
---
Dyna-Q spends most of its updates on pairs whose successors' values have not changed, which teaches nothing; the larger the world, the larger that share. Prioritized sweeping only updates pairs whose values would change.
:::
