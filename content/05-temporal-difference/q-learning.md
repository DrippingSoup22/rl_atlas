+++
summary = "Learn the value of the best move, even while you are still exploring."
change = "Build the target from the best next move, the max over a of Q(S′, a), instead of the move the agent will actually make next."
prereqs = ["sarsa", "value-functions", "epsilon-greedy", "td-error"]
lab = "cliff-race"
sources = [
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., §6.5, §6.7, Examples 6.6 and 6.7", url = "http://incompleteideas.net/book/the-book-2nd.html" },
  { text = "Watkins (1989), Learning from Delayed Rewards, PhD thesis, University of Cambridge" },
  { text = "Watkins & Dayan (1992), Q-learning, Machine Learning 8", url = "https://doi.org/10.1007/BF00992698" },
  { text = "Jaakkola, Jordan & Singh (1994), On the convergence of stochastic iterative dynamic programming algorithms, Neural Computation 6", url = "https://doi.org/10.1162/neco.1994.6.6.1185" },
  { text = "Tsitsiklis (1994), Asynchronous stochastic approximation and Q-learning, Machine Learning 16", url = "https://doi.org/10.1007/BF00993306" },
  { text = "Baird (1995), Residual algorithms: reinforcement learning with function approximation, Proceedings of the 12th International Conference on Machine Learning" },
  { text = "van Hasselt (2010), Double Q-learning, Advances in Neural Information Processing Systems 23" },
  { text = "Mnih et al. (2015), Human-level control through deep reinforcement learning, Nature 518", url = "https://doi.org/10.1038/nature14236" },
]

[story]
scene = "grid"
env = "cliff"
algorithm = "q-learning"
alpha = 0.5
epsilon = 0.1
gamma = 1.0
episodes = 500
seed = 7
formula = '\step{1}{\val{Q(S,A)}} \step{2}{\leftarrow \val{Q(S,A)} + \alp\,\big[} \step{3}{\rew{R}} \step{4}{+ \gam \max_a \val{Q(S^\prime\!,a)}} \step{5}{- \val{Q(S,A)}\,\big]}'
numbers = '\val{Q(S,A)} \leftarrow 0 + 0.5\,\big[\rew{-1} + 1 \cdot \val{0} - \val{0}\,\big] = \val{-0.5}'
+++

## Story

::: step {q = "zero", agent = "start"}
**The cliff world again.** Start at the bottom left, reach the gem at the bottom right. Each step costs $\rew{-1}$; stepping into the cliff costs $\rew{-100}$ and sends the agent back to the start.
:::

::: step {q = "zero", agent = "start", arrows = true}
Like [[sarsa]], Q-learning keeps one number per tile and direction, $\val{Q(s,a)}$, drawn as four triangles per tile, all 0 at the start. And like SARSA it moves with **ε-greedy**: usually the best-looking direction, with probability $\eps = 0.1$ a random one.
:::

::: step {q = "zero", from = "start", agent = [2, 0], reward = -1}
It goes up and pays $\rew{-1}$.
:::

::: step {q = "zero", agent = [2, 0], glow = [2, 0, "best"]}
Here is the difference. SARSA would now commit to its next move and use that one. Q-learning instead **looks at the best move available from the new tile**, the largest of its four numbers, whether or not it will actually take it. All four are 0 here, so they are all “best”.
:::

::: step {q = "zero", agent = [2, 0], formula = 5}
So its target is the reward plus the value of the *best* next move, $\rew{R} + \gam \max_a \val{Q(S',a)}$. The rest is the same nudge as always: move the old guess a fraction $\alp$ of the way toward the target.
:::

::: step {q = "zero", set = [[3, 0, "up", -0.5]], agent = [2, 0], glow = [3, 0, "up"], formula = 5, numbers = true}
With numbers: the target is $-1 + 1 \cdot 0 = -1$, the surprise is $\del = -1 - 0 = -1$, and with $\alp = 0.5$ the guess moves halfway: *up from the start* is now worth $\val{-0.5}$.
:::

::: step {q = "trained", ripple = true, path = true, agent = "start"}
Repeat for 500 episodes. Value flows back from the goal, and the best directions trace **the shortest route, straight along the edge of the cliff**: 13 steps. Q-learning has found the optimal path.
:::

::: step {q = "trained", path = true, explore = true}
The catch: the agent still explores. One random step down from the edge and it falls. Q-learning's numbers describe the best path for an agent that never slips, so it keeps walking the edge and keeps falling now and then. Learning about one policy (the greedy one) while following another (ε-greedy) is what **off-policy** means. [Watch the race against SARSA](lab:cliff-race).
:::

## Textbook

### The goal: optimal action values {#goal}

As for [[sarsa]], the setting is a finite Markov decision process whose dynamics $p(s', r \mid s, a)$ are unknown to the agent, and the task is control ([[prediction-control]]). Q-learning aims directly at the optimal action-value function ([[optimality]]),

$$\val{q_*(s,a)} = \max_{\pi}\, \val{q_\pi(s,a)}, \label{qstar}$$

the largest expected return that can be obtained after taking action $a$ in state $s$. Knowing $\val{q_*}$ solves the control problem without a model: every policy that is greedy with respect to it,

$$\pol{\pi_*(s)} \in \operatorname*{arg\,max}_a \val{q_*(s,a)}, \label{greedy}$$

is optimal. No knowledge of the dynamics is needed to act on it.

### The Bellman optimality equation {#bellman}

The optimal action values satisfy the Bellman optimality equation ([[bellman]]):

$$\val{q_*(s,a)} = \mathbb{E}\big[\,\rew{R_{t+1}} + \gam \max_{a'} \val{q_*(S_{t+1}, a')} \;\big|\; S_t = s,\ A_t = a\,\big] = \sum_{s',\,r} p(s', r \mid s, a)\,\big[\,\rew{r} + \gam \max_{a'} \val{q_*(s', a')}\,\big]. \label{bellman-opt}$$

Unlike the Bellman equation of a fixed policy, it contains no policy at all: the next action is not drawn from anything, it is the best one. Write the right-hand side as an operator that maps an action-value function $Q$ to a new one,

$$(\mathcal{T} Q)(s,a) = \sum_{s',\,r} p(s', r \mid s, a)\,\big[\,\rew{r} + \gam \max_{a'} Q(s', a')\,\big], \label{operator}$$

so that \ref{bellman-opt} says $\mathcal{T}\val{q_*} = \val{q_*}$: the optimal action values are a fixed point of $\mathcal{T}$.

::: lemma {#lem-contraction} Contraction
If $\gam < 1$, $\mathcal{T}$ is a $\gam$-contraction in the max norm: for any two action-value functions $Q$ and $Q'$,
$$\lVert \mathcal{T}Q - \mathcal{T}Q' \rVert_\infty \le \gam\,\lVert Q - Q' \rVert_\infty .$$
:::

::: proof
For any pair $(s, a)$, the rewards cancel and
$$\begin{aligned} \big|(\mathcal{T}Q)(s,a) - (\mathcal{T}Q')(s,a)\big| &= \gam\,\Big|\sum_{s'} p(s' \mid s,a)\,\big[\max_{a'} Q(s',a') - \max_{a'} Q'(s',a')\big]\Big| \\ &\le \gam \sum_{s'} p(s' \mid s,a)\,\max_{a'} \big|Q(s',a') - Q'(s',a')\big| \\ &\le \gam\,\lVert Q - Q' \rVert_\infty, \end{aligned}$$
using $\lvert \max_x f(x) - \max_x g(x) \rvert \le \max_x \lvert f(x) - g(x) \rvert$ and $\sum_{s'} p(s' \mid s, a) = 1$. Taking the maximum over $(s, a)$ gives the claim.
:::

By the Banach fixed-point theorem, $\mathcal{T}$ has exactly one fixed point, $\val{q_*}$, and the iteration $Q_{k+1} = \mathcal{T} Q_k$ converges to it from any starting point, geometrically: $\lVert Q_k - \val{q_*} \rVert_\infty \le \gam^k\, \lVert Q_0 - \val{q_*} \rVert_\infty$. This iteration is value iteration written for action values ([[value-iteration]]). It needs $p$. Q-learning is what remains of it when $p$ is unknown.

### The update {#update}

Q-learning replaces the expectation in \ref{operator} by one experienced transition $(S_t, A_t, \rew{R_{t+1}}, S_{t+1})$, and moves the estimate for the visited pair a fraction $\alp$ of the way toward it:

$$\val{Q(S_t,A_t)} \leftarrow \val{Q(S_t,A_t)} + \alp\,\big[\,\rew{R_{t+1}} + \gam \max_a \val{Q(S_{t+1},a)} - \val{Q(S_t,A_t)}\,\big]. \label{q-update}$$

Given $S_t = s$ and $A_t = a$, the target $\rew{R_{t+1}} + \gam \max_a \val{Q(S_{t+1},a)}$ is an unbiased sample of $(\mathcal{T}Q)(s,a)$ for the current estimates. So \ref{q-update} can be read as

$$\val{Q(s,a)} \leftarrow (1 - \alp)\,\val{Q(s,a)} + \alp\,\big[\,(\mathcal{T}\val{Q})(s,a) + w_t\,\big], \label{noisy}$$

where $w_t$ is noise with zero mean. Q-learning is a stochastic, asynchronous form of value iteration: it updates one pair at a time, in the order in which the agent happens to visit them, using samples instead of expectations. If $S_{t+1}$ is terminal, the max term is 0. The bracket in \ref{q-update} is a TD error ([[td-error]]) measured against the greedy target.

### Why it is off-policy {#off-policy}

The behavior policy $\pol{b}$ that chooses $A_t$, ε-greedy in practice, decides only which pairs are updated and how often. It plays no part in the target: given $(S_t, A_t)$, the next state and reward come from the environment, and the next action is not sampled but maximized over. The expected update of a pair is therefore the same whatever $\pol{b}$ is, and its fixed point is $\val{q_*}$, the value function of the greedy policy, although the agent never follows that policy while it learns. This is the sense in which Q-learning is **off-policy** ([[on-off-policy]]): it learns about one policy, the *target policy* (greedy with respect to $\val{Q}$), from experience generated by another, the *behavior policy*.

For the same reason, one-step Q-learning needs no importance-sampling correction, unlike off-policy Monte Carlo ([[off-policy-mc]], [[importance-sampling]]). Importance sampling corrects for the actions the behavior policy chose after the pair being updated, and the one-step target contains no such action. The flip side is a valuable freedom: any transition $(s, a, r, s')$ is valid training data, whenever and however it was collected, from old episodes, demonstrations or other agents. Replaying stored transitions is the basis of [[experience-replay]] and of [[dqn]].

### The algorithm {#algorithm}

::: algorithm {#alg-q} Q-learning (off-policy TD control) for estimating $\pol{\pi} \approx \pol{\pi_*}$
Parameters: step size $\alp \in (0, 1]$, small $\eps > 0$
Initialize $\val{Q(s,a)}$ for all $s \in \mathcal{S}^+$, $a \in \mathcal{A}(s)$, arbitrarily, except that $\val{Q(\textit{terminal}, \cdot)} = 0$
Loop for each episode:
  Initialize $S$
  Loop for each step of the episode, until $S$ is terminal:
    Choose $A$ from $S$ using the policy derived from $\val{Q}$ (e.g. ε-greedy)
    Take action $A$, observe $\rew{R}$ and $S'$
    $\val{Q(S,A)} \leftarrow \val{Q(S,A)} + \alp\,[\rew{R} + \gam \max_a \val{Q(S',a)} - \val{Q(S,A)}]$
    $S \leftarrow S'$
:::

Unlike SARSA, the next action is chosen at the start of the next step, after the update, because the target does not depend on it. The costs are the same: $O(|\mathcal{A}|)$ per step for the max and the action choice, one number per state–action pair, learning during the episode.

### Convergence {#convergence}

::: theorem {#thm-q} Watkins & Dayan, 1992
Consider a finite MDP with $\gam < 1$ and bounded rewards, and let $\val{Q}$ be stored in a table. If every state–action pair is updated infinitely often and the step sizes satisfy, for every pair,
$$\sum_{t} \alp_t(s,a) = \infty \qquad\text{and}\qquad \sum_{t} \alp_t^2(s,a) < \infty, \label{robbins-monro}$$
then $\val{Q_t} \to \val{q_*}$ with probability 1.
:::

::: proof Proof idea
By \ref{noisy}, Q-learning is a noisy version of the iteration $Q \leftarrow \mathcal{T}Q$ applied to one component at a time. Because $\mathcal{T}$ is a contraction (\ref{lem-contraction}), the expected update brings the estimates closer to $\val{q_*}$ in max norm. The step-size conditions make the steps large enough in total to reach the fixed point, $\sum_t \alp_t = \infty$, and small enough for the zero-mean noise to average out, $\sum_t \alp_t^2 < \infty$. General theorems on asynchronous stochastic approximation make this argument rigorous (Jaakkola, Jordan & Singh, 1994; Tsitsiklis, 1994). Watkins and Dayan's original proof used a different construction, the *action-replay process*.
:::

Compare with SARSA's convergence theorem ([[sarsa]]): Q-learning places no condition on how the behavior policy evolves, only that it keeps trying every action in every state. In a task where every state can be reached, ε-greedy with a constant $\eps > 0$ is enough. The estimates converge to $\val{q_*}$, while the behavior, which never stops exploring, never becomes optimal itself.

### Example: the cliff again {#cliff}

::: example {#ex-cliff} Cliff walking (Sutton & Barto, Example 6.6)
The world is a $4 \times 12$ grid. The agent starts in the bottom-left cell $S$, and the episode ends in the bottom-right cell $G$. The actions move one cell up, down, left or right; a move off the grid leaves the agent where it is. Every transition gives $\rew{-1}$, except that the cells between $S$ and $G$ on the bottom row form a cliff: stepping into one gives $\rew{-100}$ and sends the agent back to $S$. The task is undiscounted, $\gam = 1$.
:::

::: figure {#fig-paths}
{{cliff-paths}}
The greedy path of each method after 500 episodes ($\eps = 0.1$, $\alp = 0.5$, the same random seed for both). The Lab computes both runs when the figure first comes into view.
:::

After 500 episodes with $\eps = 0.1$ and $\alp = 0.5$, Q-learning's greedy policy follows the optimal path, 13 steps along the edge of the cliff (\ref{fig-paths}), and its estimates approximate $\val{q_*}$. But the agent behaves ε-greedily, and along the edge each step carries a probability $\eps / 4 = 0.025$ of stepping down into the cliff.

::: figure {#fig-curves}
{{cliff-curves}}
Sum of rewards per episode while learning: SARSA (dashed) and Q-learning (solid), with $\eps = 0.1$ and $\alp = 0.5$. Each curve averages 100 independent runs and is smoothed over 10 episodes; early episodes below $\rew{-100}$ are cut off.
:::

As a result, Q-learning collects less reward per episode while learning than SARSA, which learns the values of the exploring policy and moves away from the edge (\ref{fig-curves}). Learning the optimal policy and behaving well while learning are different goals, and an off-policy method serves the first. If $\eps$ is reduced over time, both methods converge to the optimal path and the gap disappears.

### Maximization bias {#max-bias}

The target uses the same noisy estimates both to choose the best next action and to evaluate it. Even when every estimate is unbiased on its own, their maximum is not:

$$\mathbb{E}\Big[\max_a \val{Q(s',a)}\Big] \;\ge\; \max_a\, \mathbb{E}\big[\val{Q(s',a)}\big], \label{jensen}$$

by Jensen's inequality, because the maximum is a convex function. For example, if the true values of $k$ actions are all 0 and each estimate carries independent standard normal noise, the expected maximum of the estimates is well above 0:

| Actions $k$ | 2 | 3 | 4 | 10 |
| --- | --- | --- | --- | --- |
| $\mathbb{E}\big[\max_a \val{Q(s',a)}\big]$ | 0.56 | 0.85 | 1.03 | 1.54 |

Through bootstrapping, this positive bias propagates back to the states that lead to $s'$. Q-learning can therefore prefer an action that leads to many uncertain options over an action whose outcome is certain and slightly better (Sutton & Barto, Example 6.7). [[double-q]] removes the bias by keeping two independent estimates, using one to choose the maximizing action and the other to evaluate it (van Hasselt, 2010).

### Properties {#properties}

- **It learns the optimum, not its own behavior.** Its estimates describe the greedy policy whatever exploration is used, so exploration can be chosen freely; the cost is paid in online reward ([[explore-exploit]]).
- **It reuses any experience.** Every transition is valid data for the update, which makes experience replay and learning from demonstrations possible.
- **It is simple and cheap.** One max over the actions and one update per step.
- **Information travels back slowly.** As in SARSA, one step per update. Planning with a learned model speeds it up ([[dyna-q]]). Multi-step versions are harder than for SARSA, because the rewards after an exploratory action no longer describe the greedy policy; Watkins's Q(λ) cuts its eligibility traces at such actions.
- **It is fragile with function approximation.** With a parameterized function in place of the table, Q-learning combines all three ingredients of the deadly triad, function approximation, bootstrapping and off-policy learning, and can diverge ([[deadly-triad]]; Baird, 1995). [[dqn]] stabilizes it with experience replay and a [[target-network]].

### Relatives {#relatives}

| Method | Target | Relation to Q-learning |
| --- | --- | --- |
| [[sarsa]] | $\rew{R_{t+1}} + \gam\,\val{Q(S_{t+1}, A_{t+1})}$ | the on-policy sibling: it samples the next action |
| [[expected-sarsa]] | $\rew{R_{t+1}} + \gam \sum_a \pol{\pi(a \mid S_{t+1})}\,\val{Q(S_{t+1}, a)}$ | with a greedy target policy it is Q-learning |
| [[double-q]] | $\rew{R_{t+1}} + \gam\,\val{Q_2(S_{t+1}, \operatorname*{arg\,max}_a Q_1(S_{t+1}, a))}$ | removes the maximization bias |
| [[value-iteration]] | $\sum_{s',r} p(s', r \mid s, a)\,[\rew{r} + \gam \max_{a'} \val{Q(s', a')}]$ | the model-based ancestor: an expected update instead of a sample |
| [[dqn]] | $\rew{R_{t+1}} + \gam \max_a \val{Q(S_{t+1}, a; \mathbf{w}^-)}$ | Q-learning with a neural network, replay and a target network |

### Historical remarks {#history}

Q-learning was introduced by Watkins (1989) in his PhD thesis; Watkins and Dayan (1992) published the convergence proof. Jaakkola, Jordan and Singh (1994) and Tsitsiklis (1994) later derived its convergence from general results on stochastic approximation. Sutton and Barto count it among the early breakthroughs of the field, because it made off-policy control simple (§6.5). Combined with deep neural networks, experience replay and a target network, it became DQN (Mnih et al., 2015), which learned to play dozens of Atari games from raw pixels, many of them at a human level.

## Card

### Idea

Q-learning learns $\val{Q(s,a)}$, how good each move is, by nudging it toward the reward it got plus the value of the **best** move from where it landed. Because the target always assumes the best next move, it learns the values of the greedy policy, the one that always picks the best move, even while the agent explores.

::: analogy
A route planner that assumes you will take the fastest road at every junction, even on the days you wander. Its estimates describe the ideal trip, not the trip you actually drive.
:::

### The update {#update}

$$\val{Q(S,A)} \leftarrow \val{Q(S,A)} + \alp\,\big[\,\rew{R} + \gam \max_a \val{Q(S',a)} - \val{Q(S,A)}\,\big]$$

Read it as *new guess = old guess + step size × (target − old guess)*. The target is $\rew{R} + \gam \max_a \val{Q(S',a)}$ and the bracket is the surprise $\del$. At a terminal tile there is nothing left to earn, so the max term is 0.

### One change from SARSA {#change}

Same update, one different term in the target:

| | target |
| --- | --- |
| [[sarsa]] | $\rew{R} + \gam\,\val{Q(S',A')}$: the move it will make next |
| Q-learning | $\rew{R} + \gam \max_a \val{Q(S',a)}$: the best move |

### Backup diagram {#backup}

{{backup q-learning}}

From the move just taken (dot), through the reward and the next state (circle), to **all** the possible next moves, keeping the best one (the arc means max).

### Pseudocode

::: pseudocode
Parameters: step size $\alp \in (0, 1]$, exploration rate $\eps > 0$, discount $\gam$
Set $\val{Q(s,a)} = 0$ for every tile and move {#init}
Repeat for each episode:
  Put the agent at the start: $S$ {#start}
  Repeat until $S$ is terminal:
    Choose $A$ in $S$ with ε-greedy on $\val{Q}$ {#choose}
    Take $A$, observe $\rew{R}$ and $S'$ {#act}
    $\val{Q(S,A)} \leftarrow \val{Q(S,A)} + \alp\,[\rew{R} + \gam \max_a \val{Q(S',a)} - \val{Q(S,A)}]$ {#update}
    $S \leftarrow S'$ {#next}
:::

### Perks

- Learns the values of the **best** policy, whatever exploration it uses meanwhile. [See it](lab:cliff-race)
- One line of arithmetic per step, and it learns during the episode, not only at the end.
- It can learn from any experience: old episodes, other agents, demonstrations. That is what lets [[dqn]] reuse past experience.
- With tables, it is guaranteed to reach the optimal values, as long as every move keeps being tried and the step size shrinks appropriately.

### Flaws

- Ignores the risk of its own exploration: on the cliff it walks the edge and, while learning, falls more often than SARSA. [See it](lab:cliff-race)
- The max over noisy estimates is biased upward, so moves can look better than they are. [[double-q]] fixes this.
- One number per tile and move: it does not generalize and does not scale to big worlds ([[why-approximate]]).
- Combined with function approximation, it can diverge ([[deadly-triad]]).

### Knobs

| Knob | Too low | Too high |
| --- | --- | --- |
| $\alp$ step size | learns very slowly | values jump around and follow the latest luck |
| $\eps$ exploration | may never discover a better route | wastes steps and, on the cliff, falls a lot |
| $\gam$ discount | short-sighted: ignores distant rewards | slow to settle when episodes are long or never end |

### Pitfalls

- Treat the value after a terminal state as 0: take the max over $\val{Q(S',a)}$ only when $S'$ is not terminal.
- To see what Q-learning has learned, test the greedy policy. Its ε-greedy behavior will always look worse.
- Break ties between equal values at random. Always picking the first action makes exploration lopsided.

### Check yourself {#check}

::: question
Q-learning's target uses the best next move. Is that the move it actually takes next?
---
Not necessarily. It acts with ε-greedy, so sometimes it takes a random move. The target ignores that and always assumes the best move.
:::

::: question
Why is Q-learning called off-policy?
---
It learns about one policy (the greedy one) while following another (ε-greedy). The policy it learns about is “off” the policy it uses.
:::

::: question
On the cliff, Q-learning finds the shortest path but earns less reward per episode than SARSA while learning. Why?
---
Its path runs along the edge. With 10% random moves it sometimes steps into the cliff and pays −100. SARSA's values include those slips, so it learns a safer route.
:::
