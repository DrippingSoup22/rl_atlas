+++
summary = "Once you know how good a policy is, make it better: in every state, switch to the move that looks best one step ahead. The new policy is never worse, and if nothing changes, it is optimal."
prereqs = ["policy-evaluation", "value-functions", "optimality"]
lab = "dp-iteration"
sources = [
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., §4.2 and Figure 4.1", url = "http://incompleteideas.net/book/the-book-2nd.html" },
  { text = "Bellman (1957), Dynamic Programming, Princeton University Press" },
  { text = "Howard (1960), Dynamic Programming and Markov Processes, MIT Press" },
  { text = "Puterman (1994), Markov Decision Processes: Discrete Stochastic Dynamic Programming, Wiley, §6.4" },
]

[story]
scene = "grid"
env = "small-gridworld"
gamma = 1.0
digits = 0
range = 22
formula = '\step{1}{\val{q_\pi(s,a)} = \sum_{s^\prime\!,\,r} p(s^\prime\!, \rew{r} \mid s, a)\,\big[\,\rew{r} + \gam\,\val{v_\pi(s^\prime)}\,\big]} \step{2}{\qquad \pol{\pi^\prime(s)} = \operatorname*{arg\,max}_a \val{q_\pi(s,a)}}'
numbers = '\val{q_\pi(s, \text{up})} = \rew{-1} + \val{(-14)} = \val{-15} \qquad \val{q_\pi(s, \text{down})} = \rew{-1} + \val{(-20)} = \val{-21}'
+++

## Story

::: step {v = "random", values = true, policy = "random", agent = "none"}
**The 4 × 4 gridworld again**, with the values of the random policy that [[policy-evaluation]] computed: minus the average number of steps to a corner, walking at random. They say how good each state is *if you keep walking at random*. Can a policy do better?
:::

::: step {v = "random", values = true, focus = [1, 1], next = true, agent = "none", formula = 1, numbers = true}
Look one step ahead. From the outlined cell, try each move once, and walk at random afterwards. Up or left lands on a cell worth $\val{-14}$, so after paying $\rew{-1}$ those moves are worth $\val{-15}$. Down or right lands on $\val{-20}$: worth $\val{-21}$. The random policy averages all four, which gives the cell's value, $\val{-18}$.
:::

::: step {q = "random", agent = "none", formula = 1, range = 22}
Do the same for every cell and every move: these are the action values $\val{q_\pi(s,a)}$ of the random policy, four triangles per cell. In every cell some moves are better than the random policy's average, and the model tells us exactly which.
:::

::: step {v = "random", values = true, policy = "improved", agent = "none", formula = 2}
The **improved policy** takes, in every cell, the move with the highest $\val{q_\pi(s,a)}$: it is greedy with respect to $\val{v_\pi}$. Where two moves tie, it splits between them. Every arrow now leads toward a nearest corner.
:::

::: step {v = "optimal", values = true, policy = "improved", agent = "none", formula = 2}
How good is the new policy? Evaluating it gives $\val{-1}$, $\val{-2}$, $\val{-3}$: the number of steps to the nearest corner, negated. No policy can do better. Here a single improvement step was enough to go from the random walk to an optimal policy.
:::

::: step {v = "optimal", values = true, policy = "optimal", agent = "none", formula = 2}
Improve once more, now from these values: nothing changes. When greedy improvement leaves a policy as it is, its values satisfy the Bellman optimality equation, and the policy is optimal. That is how [[policy-iteration]] knows when to stop.
:::

## Textbook

### Is there a better policy? {#question}

Policy evaluation says how good a policy is ([[policy-evaluation]]); the next question is how to do better. Take a deterministic policy $\pol{\pi}$ whose values $\val{v_\pi}$ have been computed, and pick a state $s$. Does some action beat $\pol{\pi(s)}$ there? A cheap test is a one-time detour: take some action $a$ in $s$, and return to $\pol{\pi}$ from the next step on. The detour is worth the action value

$$\val{q_\pi(s,a)} = \sum_{s',\,r} p(s', r \mid s, a)\,\big[\,\rew{r} + \gam\,\val{v_\pi(s')}\,\big], \label{q-from-v}$$

computed from $\val{v_\pi}$ and the model. If the detour wins, $\val{q_\pi(s,a)} > \val{v_\pi(s)}$, a single pass through $a$ beats sticking with $\pol{\pi}$. Would making the detour a habit, taking $a$ *every* time $s$ comes up, help as well? Each later visit is another chance at the same gain, so one would hope so, and the policy improvement theorem confirms it, in a more general form.

### The policy improvement theorem {#theorem}

::: theorem {#thm-improve} Policy improvement theorem
Let $\pol{\pi}$ and $\pol{\pi'}$ be deterministic policies such that, for all $s \in \mathcal{S}$,
$$\val{q_\pi(s, \pol{\pi'(s)})} \ge \val{v_\pi(s)}. \label{condition}$$
Then $\pol{\pi'}$ is at least as good as $\pol{\pi}$: $\val{v_{\pi'}(s)} \ge \val{v_\pi(s)}$ for all $s$. If the inequality \ref{condition} is strict in some state, then $\val{v_{\pi'}}$ is strictly larger in that state.
:::

::: proof
Starting from \ref{condition}, expand $\val{q_\pi}$ with \ref{q-from-v} and apply \ref{condition} again at the next state, repeatedly. Writing $\mathbb{E}_{\pi'}$ for expectations when actions are chosen by $\pol{\pi'}$,
$$\begin{aligned} v_\pi(s) &\le q_\pi(s, \pi'(s)) \\ &= \mathbb{E}\big[R_{t+1} + \gamma\, v_\pi(S_{t+1}) \mid S_t = s, A_t = \pi'(s)\big] \\ &= \mathbb{E}_{\pi'}\big[R_{t+1} + \gamma\, v_\pi(S_{t+1}) \mid S_t = s\big] \\ &\le \mathbb{E}_{\pi'}\big[R_{t+1} + \gamma\, q_\pi(S_{t+1}, \pi'(S_{t+1})) \mid S_t = s\big] \\ &= \mathbb{E}_{\pi'}\big[R_{t+1} + \gamma R_{t+2} + \gamma^2 v_\pi(S_{t+2}) \mid S_t = s\big] \\ &\le \mathbb{E}_{\pi'}\big[R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \gamma^3 v_\pi(S_{t+3}) \mid S_t = s\big] \\ &\;\;\vdots \\ &\le \mathbb{E}_{\pi'}\big[R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots \mid S_t = s\big] = v_{\pi'}(s). \end{aligned}$$
The remainder term $\gamma^k v_\pi(S_{t+k})$ vanishes as $k \to \infty$, for $\gamma < 1$ or when episodes end. If \ref{condition} is strict at $s$, the first inequality is strict.
:::

The theorem compares whole policies using only one-step lookahead from the values of the old one. No new evaluation is needed to know that the new policy is no worse.

### Greedy improvement {#greedy}

The theorem suggests changing the policy in *every* state at once, to the action that looks best by \ref{q-from-v}: the **greedy** policy with respect to $\val{v_\pi}$,

$$\pol{\pi'(s)} = \operatorname*{arg\,max}_a \val{q_\pi(s,a)} = \operatorname*{arg\,max}_a \sum_{s',\,r} p(s', r \mid s, a)\,\big[\,\rew{r} + \gam\,\val{v_\pi(s')}\,\big]. \label{greedy-policy}$$

It satisfies the condition of the theorem, since $\max_a \val{q_\pi(s,a)} \ge \val{q_\pi(s, \pol{\pi(s)})} = \val{v_\pi(s)}$, so it is at least as good as $\pol{\pi}$. This step, making a policy greedy with respect to its own value function, is called **policy improvement**.

### When improvement stops {#optimal}

Suppose the greedy policy $\pol{\pi'}$ is no better than $\pol{\pi}$: $\val{v_{\pi'}} = \val{v_\pi}$. Then, by \ref{greedy-policy}, for every state

$$\val{v_{\pi'}(s)} = \max_a \sum_{s',\,r} p(s', r \mid s, a)\,\big[\,\rew{r} + \gam\,\val{v_{\pi'}(s')}\,\big], \label{optimality}$$

which is the Bellman optimality equation ([[optimality]]). Its only solution is $\val{v_*}$, so $\pol{\pi}$ and $\pol{\pi'}$ are both optimal. Greedy improvement therefore never stalls by accident: either the new policy is strictly better in some state, or the old one was optimal already.

### Stochastic policies {#stochastic}

Nothing here needs the policies to be deterministic. For stochastic ones the condition of the theorem becomes $\sum_a \pol{\pi'(a \mid s)}\,\val{q_\pi(s,a)} \ge \val{v_\pi(s)}$, and the proof is the same. If several actions tie for the maximum in \ref{greedy-policy}, any policy that puts all of its probability on them, split in any way, is a valid improvement. The atlas splits ties evenly, which is why some cells show two arrows. The theorem also holds within restricted classes of policies: an ε-greedy policy with respect to $\val{q_\pi}$ is at least as good as any ε-soft policy $\pol{\pi}$, which is what on-policy control methods rely on ([[mc-control]], [[sarsa]]).

### Example: the 4 × 4 gridworld {#example}

In the gridworld of [[policy-evaluation]], the random policy has values from $-14$ to $-22$. The greedy policy with respect to them heads straight for a nearest corner, and its values are $-1$, $-2$ and $-3$: it is optimal, after a single improvement. The arrows in \ref{fig-sweeps} show something more: the greedy policy is already optimal with respect to the values after only three sweeps of evaluation, long before they have converged. Improvement does not need exact values, only values that rank the actions correctly. This is the idea behind [[value-iteration]], which improves after every single sweep.

::: figure {#fig-sweeps}
{{dp-sweeps}}
The values of the random policy after $k$ sweeps of policy evaluation (the numbers), and the greedy policy with respect to each of them (the arrows). From $k = 3$ on, the greedy policy is optimal, although the values are still far from converged. Computed by the Lab. After Sutton & Barto, Figure 4.1.
:::

### Improvement without a model {#model-free}

Greedy improvement with \ref{greedy-policy} needs the model $p$, to look one step ahead from state values. With action values it needs nothing else: $\pol{\pi'(s)} = \operatorname*{arg\,max}_a \val{q_\pi(s,a)}$ can be read off a table. This is why methods that learn from experience without a model estimate $\val{q}$ rather than $\val{v}$ when they have to improve a policy ([[model-based-free]], [[exploring-starts]], [[sarsa]]).

### Historical remarks {#history}

The policy improvement theorem and the method built on it, policy iteration, are due to Bellman (1957) and Howard (1960); Puterman (1994) gives a modern account. The proof by repeated expansion follows Sutton and Barto (§4.2).

## Card

### Idea

Given the values of a policy, look one step ahead in every state: which move leads to the best reward plus value of where it lands? Switch to that move everywhere. The new policy is never worse than the old one, and if it is the same, the policy was already optimal.

::: analogy
Planning a commute from a map of how long each junction takes to get home by your usual route. At every junction, take the turn that leads to the quickest junction. You can only gain, and if no turn changes, your route was already the fastest.
:::

### In symbols {#formula}

$$\pol{\pi'(s)} = \operatorname*{arg\,max}_a \sum_{s', r} p(s', \rew{r} \mid s, a)\,\big[\,\rew{r} + \gam\,\val{v_\pi(s')}\,\big] = \operatorname*{arg\,max}_a \val{q_\pi(s,a)}$$

Theorem: if $\val{q_\pi(s, \pol{\pi'(s)})} \ge \val{v_\pi(s)}$ in every state, then $\val{v_{\pi'}} \ge \val{v_\pi}$ everywhere.

### Why it matters {#why}

It is the improvement half of every control method. Dynamic programming alternates it with evaluation ([[policy-iteration]]); learning methods interleave it with estimation, one step at a time ([[gpi]]).

### Pitfalls

- Improving with state values without a model: the lookahead needs $p$. Use action values instead.
- Breaking ties arbitrarily and then testing “did the policy change?”: a policy that flips between equally good actions never looks stable. Compare values, or break ties consistently.
- Expecting the greedy policy to be optimal after one step: it is only no worse; usually several rounds are needed.

### Check yourself {#check}

::: question
Why can the greedy policy never be worse than the policy whose values it was built from?
---
In every state it picks an action with $\val{q_\pi(s,a)} \ge \val{v_\pi(s)}$, and the policy improvement theorem turns that one-step advantage into an advantage over the whole future.
:::

::: question
Greedy improvement leaves a policy unchanged. What follows?
---
Its values satisfy the Bellman optimality equation, so the policy is optimal.
:::

::: question
Why do model-free methods learn action values when they want to improve a policy?
---
Improving from state values needs the model to look one step ahead. With action values the best action can be read off directly.
:::
