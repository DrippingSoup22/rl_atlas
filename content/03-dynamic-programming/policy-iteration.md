+++
summary = "Evaluate a policy, make it greedy with respect to its values, and repeat. Each round gives a strictly better policy, and after a handful of rounds an optimal one."
change = "Alternate evaluation with greedy improvement: after the values of the policy settle, make the policy greedy with respect to them, and evaluate again, until the policy stops changing."
prereqs = ["policy-evaluation", "policy-improvement"]
lab = "dp-iteration"
sources = [
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., §4.3 and Exercise 4.4", url = "http://incompleteideas.net/book/the-book-2nd.html" },
  { text = "Howard (1960), Dynamic Programming and Markov Processes, MIT Press" },
  { text = "Puterman & Shin (1978), Modified policy iteration algorithms for discounted Markov decision problems, Management Science 24" },
  { text = "Puterman & Brumelle (1979), On the convergence of policy iteration in stationary dynamic programming, Mathematics of Operations Research 4", url = "https://doi.org/10.1287/moor.4.1.60" },
  { text = "Ye (2011), The simplex and policy-iteration methods are strongly polynomial for the Markov decision problem with a fixed discount rate, Mathematics of Operations Research 36", url = "https://doi.org/10.1287/moor.1110.0516" },
]

[story]
scene = "grid"
env = "frozen-lake"
digits = 2
formula = '\step{1}{\text{evaluate: } \val{V} \leftarrow \mathcal{T}_{\pol{\pi}}\,\val{V} \text{ until it settles}} \step{2}{\qquad \text{improve: } \pol{\pi(s)} \leftarrow \operatorname*{arg\,max}_a \textstyle\sum_{s^\prime\!,\,r} p(s^\prime\!, \rew{r} \mid s, a)\,\big[\rew{r} + \gam\,\val{V(s^\prime)}\big]}'

[story.runs]
pi = { algorithm = "policy-iteration", gamma = 0.99, theta = 0.001, units = 120 }
+++

## Story

::: step {run = "pi", at = 0, values = true}
**Frozen Lake.** Reach the gem without falling through a hole. The ice is slippery: a move goes the intended way only one time in three, and otherwise slides to one side of it. Only the gem pays, $\rew{+1}$, and rewards are discounted by $\gam = 0.99$. Policy iteration knows these rules. It starts from the random policy: four arrows in every cell.
:::

::: step {run = "pi", at = 0, values = true, play = 3, pace = 120, formula = 1}
**Evaluate.** Sweep the states with the Bellman equation of the random policy, as in [[policy-evaluation]]. Value seeps out from the gem, sweep after sweep.
:::

::: step {run = "pi", at = 13, values = true, formula = 1}
After 13 sweeps the values barely move any more. They are tiny: walking at random, the agent almost never reaches the gem. From the start, about once in a hundred tries.
:::

::: step {run = "pi", at = 13, values = true, play = 1, pace = 450, formula = 2}
**Improve.** In every cell, switch to the move whose one-step lookahead, reward plus discounted value of where it may slide, is best ([[policy-improvement]]). Every cell where a choice is possible changes.
:::

::: step {run = "pi", at = 14, values = true, focus = [0, 0], formula = 2}
Look at the new policy at the start: it points **left, into the wall**. On this ice, that is clever. The move either bumps into the wall, or slides up into the wall, or slides down the safe left edge. It can never slide toward a hole. This policy reaches the gem 78% of the time.
:::

::: step {run = "pi", at = 82, values = true, formula = 1}
**Evaluate again**, now the new policy. Its values are far higher: $\val{0.52}$ at the start. Then **improve** again...
:::

::: step {run = "pi", at = 83, values = true, focus = [0, 2], formula = 2}
...and this time only one cell changes, the outlined one, from left to up. Evaluate once more, improve once more: nothing changes. The policy is stable, so it is optimal.
:::

::: step {run = "pi", at = 112, values = true}
Done: three policies, the random one and two improvements, and 112 sweeps in all. The optimal policy reaches the gem 82% of the time, the best any policy can do on this ice. [Race it against value iteration in the Lab](lab:dp-iteration).
:::

## Textbook

### Evaluate, improve, repeat {#loop}

One round of greedy improvement turns a policy into a better one ([[policy-improvement]]), and nothing prevents a second round: evaluate the new policy, improve it, evaluate again. Alternating the two steps produces a chain of policies, each at least as good as the one before, along with their value functions:

$$\pol{\pi_0} \xrightarrow{\;E\;} \val{v_{\pi_0}} \xrightarrow{\;I\;} \pol{\pi_1} \xrightarrow{\;E\;} \val{v_{\pi_1}} \xrightarrow{\;I\;} \pol{\pi_2} \xrightarrow{\;E\;} \cdots \xrightarrow{\;I\;} \pol{\pi_*} \xrightarrow{\;E\;} \val{v_*}, \label{sequence}$$

where $\xrightarrow{E}$ denotes a policy evaluation ([[policy-evaluation]]) and $\xrightarrow{I}$ a greedy policy improvement. This is **policy iteration**.

::: theorem {#thm-finite} Policy iteration terminates with an optimal policy
In a finite MDP, policy iteration started from any deterministic policy reaches an optimal policy and its value function after a finite number of iterations.
:::

::: proof
By the policy improvement theorem, each greedy improvement gives a policy that is strictly better in at least one state, unless the policy is already optimal ([[policy-improvement]]). So no policy can occur twice in the sequence. A finite MDP has only finitely many deterministic policies, $|\mathcal{A}|^{|\mathcal{S}|}$ of them, so the sequence must end, and it can only end at a policy that improvement leaves unchanged, which is optimal.
:::

The bound $|\mathcal{A}|^{|\mathcal{S}|}$ is astronomically pessimistic. Real runs usually need only a handful of improvement steps: Frozen Lake below needs three, and the last of them only confirms that nothing changes. Ye (2011) proved that, for a fixed discount rate $\gam < 1$, the number of iterations is bounded by a polynomial in the numbers of states and actions, roughly $\tfrac{|\mathcal{S}|\,|\mathcal{A}|}{1 - \gam} \log \tfrac{|\mathcal{S}|}{1 - \gam}$.

### The algorithm {#algorithm}

::: algorithm {#alg-pi} Policy iteration (using iterative policy evaluation) for estimating $\pol{\pi} \approx \pol{\pi_*}$
1. Start with any values $\val{V(s)}$ and any actions $\pol{\pi(s)} \in \mathcal{A}(s)$ for every state; $\val{V(\textit{terminal})} = 0$
2. Policy evaluation
Loop:
  $\Delta \leftarrow 0$
  Loop for each $s \in \mathcal{S}$:
    $v \leftarrow \val{V(s)}$
    $\val{V(s)} \leftarrow \sum_{s',r} p(s', r \mid s, \pol{\pi(s)})\,[\rew{r} + \gam\,\val{V(s')}]$
    $\Delta \leftarrow \max(\Delta, |v - \val{V(s)}|)$
until $\Delta < \theta$, the tolerance
3. Policy improvement
$\textit{policy-stable} \leftarrow \textit{true}$
For each $s \in \mathcal{S}$:
  $\textit{old-action} \leftarrow \pol{\pi(s)}$
  $\pol{\pi(s)} \leftarrow \operatorname*{arg\,max}_a \sum_{s',r} p(s', r \mid s, a)\,[\rew{r} + \gam\,\val{V(s')}]$
  If $\textit{old-action} \ne \pol{\pi(s)}$, then $\textit{policy-stable} \leftarrow \textit{false}$
If $\textit{policy-stable}$: return $\pol{\pi}$, an optimal policy, and $\val{V} \approx \val{v_*}$. Otherwise back to step 2
:::

The evaluation of each new policy starts not from scratch but from the values of the policy before it: a good first guess, since consecutive policies usually differ in only a few states.

The stopping test has a subtle bug (Sutton & Barto, Exercise 4.4). If several actions are equally good in some state, the arg max may switch between them from one improvement to the next, and the policy never becomes “stable”, although every policy in the cycle is optimal. Two fixes: change the action only when the new one is *strictly* better, or test whether the values, rather than the actions, stopped changing. The Lab's version keeps all tied actions, with equal probability, and stops when that set no longer changes.

### Example: Frozen Lake {#example}

::: example {#ex-lake} Frozen Lake
The world is a $4 \times 4$ grid of ice with four holes. The agent starts in the top-left corner; the episode ends when it reaches the gem in the bottom-right corner, with reward $\rew{+1}$, or falls into a hole, with reward 0. Every other transition has reward 0. The ice is slippery: each action moves the agent in the intended direction with probability $\tfrac13$, and to each of the two perpendicular directions with probability $\tfrac13$; a move off the grid leaves the agent in place. Here $\gam = 0.99$.
:::

Policy iteration starts from the random policy, whose values are tiny: walking at random, the agent reaches the gem from the start with probability about 0.014. The first improvement changes the action in every state where there is a choice, and the resulting policy $\pol{\pi_1}$ already reaches the gem 78% of the time. The second improvement changes a single state; the third changes nothing, so $\pol{\pi_2}$ is optimal (\ref{fig-lake}). With $\theta = 0.001$, the whole run takes 112 sweeps: 13 to evaluate the random policy, about 70 for $\pol{\pi_1}$ and about 30 for $\pol{\pi_2}$, plus the three improvement steps. Almost all the work is evaluation.

::: figure {#fig-lake}
{{frozen pi}}
The policies of policy iteration on Frozen Lake, each with its value function: the random policy $\pol{\pi_0}$, the first improvement $\pol{\pi_1}$, and the optimal policy $\pol{\pi_2}$, which the next improvement leaves unchanged. Computed by the Lab.
:::

The optimal policy is not the one a person would guess. In the start state it moves *left*, into the wall: the move either bumps the wall or slides up into it, leaving the agent in place, or slides down along the left edge, which is safe. And in the cell between two holes it moves toward one of them: that move falls in with probability $\tfrac13$, while a move up or down slides into a hole with probability $\tfrac23$. Dynamic programming finds such policies without being told anything but the dynamics.

### Why so few iterations {#newton}

Policy iteration is Newton's method in disguise. Each iteration solves the Bellman equation of the current policy exactly, which amounts to linearizing the Bellman *optimality* equation around the current values and solving the linearized system, as Newton's method does for an ordinary equation (Puterman & Brumelle, 1979). Like Newton's method, it converges very fast once it is close to the solution. Its cost is in each step: an exact evaluation is a linear solve, or many sweeps.

That cost suggests a compromise: stop each evaluation early, after a fixed number of sweeps $m$, before improving. This is **modified policy iteration**. With $m = 1$ it becomes [[value-iteration]]; as $m \to \infty$ it becomes policy iteration. All of these converge to an optimal policy, and in practice an intermediate $m$, or a loose tolerance $\theta$, is often fastest.

### Properties {#properties}

- **Few, expensive iterations.** The number of improvements is usually small; each one is preceded by a full evaluation.
- **Monotone.** Every policy in the sequence is at least as good as the one before, in every state. Stopping early always returns a policy no worse than the last one.
- **Exact termination.** Unlike value iteration, it can stop with a policy that is provably optimal, not just a value function that is close to optimal.
- **Needs the model.** Both halves use $p(s', r \mid s, a)$ ([[dp-limits]]).
- **The template of control.** Interleaving evaluation and improvement, however loosely, is the structure of almost every control method that follows ([[gpi]]).

### Historical remarks {#history}

Policy iteration was introduced by Howard (1960), building on Bellman's work. Puterman and Brumelle (1979) showed its equivalence to Newton's method, and Ye (2011) proved that it runs in strongly polynomial time for a fixed discount rate. Modified policy iteration was analyzed by Puterman and Shin (1978).

## Card

### Idea

Start with any policy. Compute its values exactly, then make it greedy with respect to them: in every state, the move with the best one-step lookahead. Repeat. Every round gives a better policy, and when a round changes nothing, the policy is optimal.

::: analogy
Revising a travel plan: work out what your current plan really costs from every stop, then change each leg to whatever looks cheapest given those costs. Recompute, revise again. When no revision helps, the plan is the best one.
:::

### The update {#update}

$$\text{evaluate: } \val{V} \leftarrow \text{the values of } \pol{\pi} \qquad \text{improve: } \pol{\pi(s)} \leftarrow \operatorname*{arg\,max}_a \sum_{s', r} p(s', \rew{r} \mid s, a)\,\big[\,\rew{r} + \gam\,\val{V(s')}\,\big]$$

Stop when improvement leaves the policy unchanged.

### One change from policy evaluation {#change}

| | does |
| --- | --- |
| [[policy-evaluation]] | computes the values of one fixed policy |
| policy iteration | evaluates, makes the policy greedy, and evaluates the new policy, until it stops changing |

### Backup diagram {#backup}

{{backup v-pi}}

The evaluation backs up over the actions the current policy takes; the improvement then picks, in each state, the action whose backup is best.

### Pseudocode

::: pseudocode
Parameters: the model $p$, a tolerance $\theta > 0$, discount $\gam$
Start from any policy $\pol{\pi}$ (here, random) and $\val{V(s)} = 0$ {#init}
Repeat:
  Repeat (evaluation):
    For each state $s$: {#sweep}
      $\val{V(s)} \leftarrow \sum_a \pol{\pi(a \mid s)} \sum_{s',r} p(s',r \mid s,a)\,[\rew{r} + \gam\,\val{V(s')}]$ {#update}
  until the largest change $\Delta < \theta$ {#delta}
  For each state $s$: $\pol{\pi(s)} \leftarrow \operatorname*{arg\,max}_a \sum_{s',r} p(s',r \mid s,a)\,[\rew{r} + \gam\,\val{V(s')}]$ {#improve}
  If the policy changed, evaluate the new one {#unstable}
until the policy is stable: it is optimal {#stable}
:::

### Perks

- Converges in few improvement steps, often a handful even for large problems. [See it](lab:dp-iteration)
- Every intermediate policy is at least as good as the previous one.
- Stops with a policy that is exactly optimal (up to the evaluation tolerance).

### Flaws

- Each improvement waits for a full evaluation: many sweeps when $\gam$ is close to 1.
- Needs a complete model of the world.
- Every sweep touches every state: impossible for huge state spaces ([[dp-limits]]).

### Knobs

| Knob | Too low | Too high |
| --- | --- | --- |
| $\theta$ evaluation tolerance | long evaluations of policies about to be replaced | improves from inaccurate values (still converges, more rounds) |
| $\gam$ discount | short-sighted policies | slow evaluations |

### Pitfalls

- A stopping test that compares actions: ties between equally good actions can make the policy flip forever.
- Starting each evaluation from zero: start from the previous policy's values instead.
- Evaluating to full precision every time: a loose tolerance usually reaches the optimal policy just as surely, and faster.

### Check yourself {#check}

::: question
Why must policy iteration stop after finitely many rounds in a finite MDP?
---
Every round gives a strictly better policy unless the current one is optimal, so no policy can repeat, and there are only finitely many deterministic policies.
:::

::: question
On Frozen Lake, why does the optimal policy move left, into the wall, at the start?
---
On slippery ice the move either leaves the agent in place or slides it down the safe left edge. No outcome of that move brings it closer to a hole.
:::

::: question
What do you get if you stop every evaluation after a single sweep?
---
Value iteration: one sweep of evaluation, then improvement, which together amount to one sweep of the Bellman optimality update.
:::
