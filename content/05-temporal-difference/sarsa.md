+++
summary = "Learn how good each move is by following the moves you really make, exploration included."
change = "Learn Q(S, A), the value of a move, instead of V(S), the value of a state, so the agent can choose moves without knowing the rules of the world."
prereqs = ["td0", "value-functions", "epsilon-greedy"]
lab = "cliff-race"
sources = [
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., §5.4, §6.4 and Example 6.6", url = "http://incompleteideas.net/book/the-book-2nd.html" },
  { text = "Rummery & Niranjan (1994), On-line Q-learning using connectionist systems, Cambridge University tech. report CUED/F-INFENG/TR 166" },
  { text = "Sutton (1996), Generalization in reinforcement learning: successful examples using sparse coarse coding, Advances in Neural Information Processing Systems 8" },
  { text = "Singh, Jaakkola, Littman & Szepesvári (2000), Convergence results for single-step on-policy reinforcement-learning algorithms, Machine Learning 38", url = "https://doi.org/10.1023/A:1007678930559" },
]

[story]
scene = "grid"
env = "cliff"
algorithm = "sarsa"
alpha = 0.5
epsilon = 0.1
gamma = 1.0
episodes = 500
seed = 7
formula = '\step{1}{\val{Q(S,A)}} \step{2}{\leftarrow \val{Q(S,A)} + \alp\,\big[} \step{3}{\rew{R}} \step{4}{+ \gam\,\val{Q(S^\prime\!,A^\prime)}} \step{5}{- \val{Q(S,A)}\,\big]}'
numbers = '\val{Q(S,A)} \leftarrow 0 + 0.5\,\big[\rew{-1} + 1 \cdot \val{0} - \val{0}\,\big] = \val{-0.5}'
+++

## Story

::: step {q = "zero", agent = "start"}
**This is the cliff world.** The agent starts at the bottom left and wants to reach the gem at the bottom right. Every step costs $\rew{-1}$. Between them lies a cliff: stepping into it costs $\rew{-100}$ and sends the agent back to the start.
:::

::: step {q = "zero", agent = "start"}
The agent keeps a table with one number for every tile and every direction, $\val{Q(s,a)}$: *how good is it to go this way from here?* Each tile is drawn as four triangles, one per direction. At first every number is 0, so every triangle is gray. The agent knows nothing yet.
:::

::: step {q = "zero", agent = "start", glow = [3, 0, "up"], arrows = true}
To move, it uses **ε-greedy**: most of the time it takes the direction with the highest number, but with probability $\eps = 0.1$ it picks one at random. Here all four are tied, so the choice is random: *up*.
:::

::: step {q = "zero", from = "start", agent = [2, 0], reward = -1, glow = [2, 0, "right"]}
It steps up and pays $\rew{-1}$. Now comes the part that gives SARSA its name: **before learning anything, it already picks its next move** $A'$ from the new tile, with the same ε-greedy rule. Here: *right*. One update uses five things, $S, A, \rew{R}, S', A'$: S-A-R-S-A.
:::

::: step {q = "zero", agent = [2, 0], glow = [2, 0, "right"], formula = 5}
The update nudges the old guess toward a **target**: the reward it just got plus the value of the move it is about to make. $\gam$ is the discount (here 1: later steps count fully) and $\alp$ is the step size, how far to move toward the target.
:::

::: step {q = "zero", set = [[3, 0, "up", -0.5]], agent = [2, 0], glow = [3, 0, "up"], formula = 5, numbers = true}
With numbers: the target is $-1 + 1 \cdot 0 = -1$, so the guess for *up from the start* moves halfway there, to $\val{-0.5}$. One triangle changes color. That is the whole algorithm, repeated at every step.
:::

::: step {q = "trained", ripple = true, path = true, agent = "start"}
After 500 episodes the numbers tell a story. Values have spread back from the goal, and following the best direction on each tile gives a path that **keeps a safe distance from the edge**.
:::

::: step {q = "trained", path = true, explore = true}
Why the detour? SARSA learns the value of the moves it actually makes, and 1 in 10 of them is random. Next to the edge, a random step down means a fall, and those falls are part of its numbers. So it learns the best route *for an agent that sometimes stumbles*. That is **on-policy** learning. [[q-learning]] changes one term of the target and gets a very different path.
:::

## Textbook

### The control problem {#problem}

The setting is a finite Markov decision process ([[mdp]]): a set of states $\mathcal{S}$, a set of actions $\mathcal{A}(s)$ in each state, and dynamics $p(s', r \mid s, a)$, the probability of arriving in state $s'$ with reward $r$ after taking action $a$ in state $s$. The agent does not know $p$. It only observes the transitions it experiences. We assume an episodic task ([[episodes]]): every episode ends in a terminal state, and a new episode then begins.

The task is **control** ([[prediction-control]]): find a policy $\pol{\pi}$ that maximizes, from every state, the expected return

$$\rew{G_t} = \rew{R_{t+1}} + \gam\,\rew{R_{t+2}} + \gam^2\,\rew{R_{t+3}} + \cdots = \sum_{k=0}^{T-t-1} \gam^k\,\rew{R_{t+k+1}}, \label{return}$$

where $T$ is the time step at which the episode ends and $\gam \in [0, 1]$ is the discount rate ([[return]], [[discount]]). The object SARSA estimates is the action-value function of a policy ([[value-functions]]),

$$\val{q_\pi(s,a)} = \mathbb{E}_\pi\big[\,\rew{G_t} \mid S_t = s,\ A_t = a\,\big], \label{q}$$

the expected return when the agent starts in $s$, takes action $a$, and follows $\pol{\pi}$ afterwards. SARSA learns $\val{q_\pi}$ from experience with the bootstrapping idea of [[td0]], and improves $\pol{\pi}$ while it learns.

### Why action values {#why-q}

Classical solution methods alternate two steps: evaluate the current policy, then improve it by acting greedily with respect to its values ([[policy-improvement]], [[gpi]]). With state values, the improvement step needs the dynamics:

$$\pol{\pi'(s)} = \operatorname*{arg\,max}_a \sum_{s',\,r} p(s', r \mid s, a)\,\big[\,\rew{r} + \gam\,\val{v_\pi(s')}\,\big].$$

An agent that does not know $p$ cannot carry out this step. With action values, the same step needs nothing but the values themselves:

$$\pol{\pi'(s)} = \operatorname*{arg\,max}_a \val{q_\pi(s,a)}. \label{greedy}$$

This is why model-free control methods ([[model-based-free]]) estimate $\val{q}$ rather than $\val{v}$. The price is size: one estimate per state–action pair, $|\mathcal{S}|\,|\mathcal{A}|$ numbers instead of $|\mathcal{S}|$, and each must be learned from visits to that particular pair.

### The update {#update}

Splitting off the first reward of the return, $\rew{G_t} = \rew{R_{t+1}} + \gam\,\rew{G_{t+1}}$, and conditioning on the next state and action gives the Bellman equation for $\val{q_\pi}$ ([[bellman]]):

$$\val{q_\pi(s,a)} = \mathbb{E}_\pi\big[\,\rew{R_{t+1}} + \gam\,\val{q_\pi(S_{t+1}, A_{t+1})} \mid S_t = s,\ A_t = a\,\big], \label{bellman-q}$$

where $S_{t+1}$ and $\rew{R_{t+1}}$ are drawn from $p(\cdot, \cdot \mid s, a)$ and $A_{t+1}$ from $\pol{\pi(\cdot \mid S_{t+1})}$. One experienced transition $(S_t, A_t, \rew{R_{t+1}}, S_{t+1}, A_{t+1})$ is one sample of the quantity inside the expectation. Replacing the unknown $\val{q_\pi}$ by the current estimate $\val{Q}$ gives the **SARSA target** $\rew{R_{t+1}} + \gam\,\val{Q(S_{t+1}, A_{t+1})}$, and SARSA moves the estimate for the pair just visited a fraction $\alp$ of the way toward it:

$$\val{Q(S_t,A_t)} \leftarrow \val{Q(S_t,A_t)} + \alp\,\big[\,\rew{R_{t+1}} + \gam\,\val{Q(S_{t+1},A_{t+1})} - \val{Q(S_t,A_t)}\,\big]. \label{sarsa}$$

The bracket is the TD error ([[td-error]]),

$$\del_t = \rew{R_{t+1}} + \gam\,\val{Q(S_{t+1},A_{t+1})} - \val{Q(S_t,A_t)}, \label{delta}$$

and $\alp \in (0, 1]$ is the step size ([[step-size]]). If $S_{t+1}$ is terminal, $\val{Q(S_{t+1}, \cdot)}$ is defined to be 0 and the target is just $\rew{R_{t+1}}$. The update uses the five quantities $S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1}$, which give the method its name.

Two properties of the target shape everything that follows. It is **biased**: it uses $\val{Q}$ instead of $\val{q_\pi}$, so its expectation matches the right-hand side of \ref{bellman-q} only once the estimates are right. And it has **low variance**: its randomness comes from a single transition and a single action choice, whereas the full return $\rew{G_t}$ used by Monte Carlo methods ([[mc-prediction]]) depends on every transition until the end of the episode ([[mc-vs-td]]).

Under a fixed policy, the sequence of state–action pairs $(S_0, A_0), (S_1, A_1), \ldots$ is itself a Markov chain with rewards, and \ref{sarsa} is exactly TD(0) applied to that chain. The convergence results for TD(0) prediction therefore carry over to SARSA's estimates of $\val{q_\pi}$ for a fixed policy.

### On-policy control {#on-policy}

To turn prediction into control, SARSA acts with a policy derived from its current estimates, usually ε-greedy ([[epsilon-greedy]]):

$$\pol{\pi(a \mid s)} = \begin{cases} 1 - \eps + \eps / |\mathcal{A}(s)| & \text{if } a = \operatorname*{arg\,max}_{a'} \val{Q(s,a')}, \\ \eps / |\mathcal{A}(s)| & \text{otherwise,} \end{cases} \label{eps-greedy}$$

and the action $A_{t+1}$ in its target is drawn from this same policy. Because $\val{Q}$ changes after every step, so does the policy: evaluation and improvement are interleaved at the finest possible grain, one transition at a time. This is generalized policy iteration in its most incremental form ([[gpi]]).

SARSA is **on-policy** ([[on-off-policy]]): the policy that generates the experience (the *behavior policy*) and the policy whose values are learned (the *target policy*) are the same. Its estimates are therefore the values of an exploring agent, and they include the cost of the agent's own random actions.

A policy is *ε-soft* if $\pol{\pi(a \mid s)} \ge \eps / |\mathcal{A}(s)|$ for every state and action; ε-greedy policies are the greediest of them. The policy improvement theorem still holds within this class: an ε-greedy policy with respect to $\val{q_\pi}$ is at least as good as any ε-soft policy $\pol{\pi}$, with equality only if $\pol{\pi}$ is already the best ε-soft policy (Sutton & Barto, §5.4). Consequently, a fixed point of SARSA with a constant $\eps$ is the action-value function of the **best ε-soft policy**: the best an agent can do if it must keep exploring at rate $\eps$, not an optimal policy. To reach an optimal policy, exploration has to fade out (\ref{convergence}).

### The algorithm {#algorithm}

::: algorithm {#alg-sarsa} SARSA (on-policy TD control)
Parameters: step size $\alp \in (0, 1]$, small $\eps > 0$
Initialize $\val{Q(s,a)}$ for all $s \in \mathcal{S}^+$, $a \in \mathcal{A}(s)$, arbitrarily, except that $\val{Q(\textit{terminal}, \cdot)} = 0$
Loop for each episode:
  Initialize $S$
  Choose $A$ from $S$ using the policy derived from $\val{Q}$ (e.g. ε-greedy)
  Loop for each step of the episode, until $S$ is terminal:
    Take action $A$, observe $\rew{R}$ and $S'$
    Choose $A'$ from $S'$ using the policy derived from $\val{Q}$ (e.g. ε-greedy)
    $\val{Q(S,A)} \leftarrow \val{Q(S,A)} + \alp\,[\rew{R} + \gam\,\val{Q(S',A')} - \val{Q(S,A)}]$
    $S \leftarrow S'$; $A \leftarrow A'$
:::

Here $\mathcal{S}^+$ is the set of all states including the terminal one. The action $A'$ used in the target is the action actually taken at the next step; choosing a fresh action instead would change the algorithm. Each step costs one action selection, $O(|\mathcal{A}|)$ for the greedy part, and one update of constant cost; memory is one number per state–action pair. The method is fully incremental and online: it learns during an episode, and it applies unchanged to continuing tasks, where episodes never end and a Monte Carlo return can never be formed.

### Convergence {#convergence}

Convergence to an optimal policy requires two things: every state–action pair must keep being tried, and the policy must become greedy in the limit.

::: definition {#def-glie} GLIE
A learning policy is **greedy in the limit with infinite exploration** (GLIE) if every state–action pair is visited infinitely often and the policy converges, with probability 1, to the greedy policy with respect to the current action-value estimates.
:::

ε-greedy exploration is GLIE when $\eps$ decays toward zero slowly enough, for example $\eps_t(s) = c / n_t(s)$ with $0 < c < 1$, where $n_t(s)$ is the number of visits to $s$ so far (Singh et al., 2000).

::: theorem {#thm-sarsa} Singh, Jaakkola, Littman & Szepesvári, 2000
Consider a finite MDP with $\gam < 1$ and rewards of bounded variance, and let $\val{Q}$ be stored in a table. If the learning policy is GLIE and the step sizes satisfy, for every pair $(s, a)$,
$$\sum_{t} \alp_t(s,a) = \infty \qquad\text{and}\qquad \sum_{t} \alp_t^2(s,a) < \infty, \label{robbins-monro}$$
then SARSA's estimates converge with probability 1 to the optimal action values $\val{q_*}$, and its policy converges to an optimal policy.
:::

Here $\alp_t(s,a) = 0$ whenever $(s,a)$ is not the pair updated at step $t$. The conditions \ref{robbins-monro} are the Robbins–Monro conditions of stochastic approximation: the steps must add up to infinity, so the estimates can travel any distance, while their squares must add up to a finite number, so the noise eventually averages out ([[step-size]]). The sample-average choice $\alp_t(s,a) = 1/n_t(s,a)$, one over the number of updates the pair has received, satisfies both; a constant step size violates the second.

::: proof Proof idea
Add and subtract the greedy value in the target:
$$\rew{R_{t+1}} + \gam\,\val{Q(S_{t+1},A_{t+1})} = \underbrace{\rew{R_{t+1}} + \gam \max_{a} \val{Q(S_{t+1},a)}}_{\text{Q-learning target}} + \underbrace{\gam\,\big[\val{Q(S_{t+1},A_{t+1})} - \max_a \val{Q(S_{t+1},a)}\big]}_{c_t}.$$
The first part is the target of [[q-learning]], whose expected update is a contraction toward $\val{q_*}$. The second part, $c_t$, vanishes as the policy becomes greedy, which GLIE guarantees. A stochastic-approximation lemma that tolerates a perturbation tending to zero then gives the result.
:::

In practice SARSA is usually run with a constant $\alp$ and a constant $\eps$, as in the example below. Its estimates then never settle completely: they keep moving with the most recent transitions. That is a drawback in a fixed problem and an advantage when the problem itself changes over time ([[nonstationary]]).

### Example: walking along a cliff {#cliff}

::: example {#ex-cliff} Cliff walking (Sutton & Barto, Example 6.6)
The world is a $4 \times 12$ grid. The agent starts in the bottom-left cell $S$, and the episode ends in the bottom-right cell $G$. The actions move one cell up, down, left or right; a move off the grid leaves the agent where it is. Every transition gives $\rew{-1}$, except that the cells between $S$ and $G$ on the bottom row form a cliff: stepping into one gives $\rew{-100}$ and sends the agent back to $S$. The task is undiscounted, $\gam = 1$, so the best possible return is $\rew{-13}$: one $\rew{-1}$ for each step of the shortest path.
:::

\ref{fig-paths} shows the greedy policy of SARSA and of Q-learning after 500 episodes, both with $\eps = 0.1$ and $\alp = 0.5$. Q-learning's path runs along the edge of the cliff and is optimal. SARSA's keeps its distance.

::: figure {#fig-paths}
{{cliff-paths}}
The cliff world and the greedy path of each method after 500 episodes ($\eps = 0.1$, $\alp = 0.5$, the same random seed for both). The Lab computes both runs when the figure first comes into view.
:::

The reason is that SARSA evaluates the policy it follows, and that policy explores. Next to the edge, every step carries a probability $\eps / 4 = 0.025$ of the exploratory action *down*, which costs $\rew{-100}$ and the walk back from $S$. A rough estimate: along the 11 edge cells, this amounts to about $11 \times 0.025 \times (100 + 6) \approx 29$ reward lost per episode, where 6 is the typical number of steps lost by restarting. The detour away from the edge costs only a few extra steps. SARSA's values include this risk, because they are the values of the exploring policy; the values of the greedy policy do not.

::: figure {#fig-curves}
{{cliff-curves}}
Sum of rewards per episode while learning: SARSA (dashed) and Q-learning (solid), with $\eps = 0.1$ and $\alp = 0.5$. Each curve averages 100 independent runs and is smoothed over 10 episodes; early episodes below $\rew{-100}$ are cut off.
:::

\ref{fig-curves} shows the consequence for the reward collected while learning. Q-learning learns the values of the optimal policy, but it keeps walking the edge with an ε-greedy policy and keeps falling, so its online performance is worse: in the last episodes it collects about $\rew{-50}$ per episode, against about $\rew{-25}$ for SARSA. If $\eps$ were gradually reduced, both methods would converge to the optimal path.

### Properties {#properties}

- **Bias and variance.** The target is biased by the current estimates but has far lower variance than a Monte Carlo return. On stochastic tasks, one-step TD methods have usually been found to learn faster than constant-α Monte Carlo methods (Sutton & Barto, §6.2), and SARSA inherits this ([[mc-vs-td]], [[mc-control]]).
- **Online learning.** SARSA learns from each transition, during the episode. Monte Carlo control cannot easily be used when some policies never reach a terminal state: an agent stuck in a loop never finishes its episode and never learns. SARSA learns during the episode that such a policy is poor and switches to another (Sutton & Barto, Example 6.5).
- **Slow propagation.** One update per step moves information back by one step. If the only reward comes at the end of a long episode, after the first episode only the pair just before it has changed, and many episodes pass before the reward is felt at the start. Multi-step methods address this ([[n-step-sarsa]], [[sarsa-lambda]]).
- **Dependence on exploration.** The learned values, and through them the policy, depend on $\eps$. This is what you want when the agent must keep exploring while it acts, such as a robot that learns on the job, and not what you want when only the final greedy policy matters.
- **Function approximation.** With a parameterized function in place of the table, the same update becomes semi-gradient SARSA ([[semi-gradient-sarsa]]). Being on-policy, it avoids one ingredient of the deadly triad and does not diverge the way off-policy methods can ([[deadly-triad]]), although its estimates may keep oscillating instead of converging.

### Relatives {#relatives}

The one-step TD control methods differ only in what follows $\gam$ in the target:

| Method | Target | Learns the values of |
| --- | --- | --- |
| SARSA | $\rew{R_{t+1}} + \gam\,\val{Q(S_{t+1}, A_{t+1})}$ | the policy it follows (on-policy) |
| [[expected-sarsa]] | $\rew{R_{t+1}} + \gam \sum_a \pol{\pi(a \mid S_{t+1})}\,\val{Q(S_{t+1}, a)}$ | any target policy $\pol{\pi}$ |
| [[q-learning]] | $\rew{R_{t+1}} + \gam \max_a \val{Q(S_{t+1}, a)}$ | the greedy policy (off-policy) |

Expected SARSA averages over the next action instead of sampling it, which removes the variance caused by the random choice of $A_{t+1}$. When its target policy is greedy, it is exactly Q-learning. Waiting for more rewards before bootstrapping gives n-step SARSA and SARSA(λ) ([[n-step-sarsa]], [[sarsa-lambda]]); replacing the table by a parameterized function gives semi-gradient SARSA ([[semi-gradient-sarsa]]).

### Historical remarks {#history}

SARSA was introduced by Rummery and Niranjan (1994), who called it *modified connectionist Q-learning*; the name Sarsa was proposed by Sutton (1996). Its convergence in the tabular case, under GLIE exploration, was proved by Singh, Jaakkola, Littman and Szepesvári (2000). The presentation here follows Sutton and Barto (2018, §6.4).

## Card

### Idea

SARSA learns $\val{Q(s,a)}$, how good each move is, by nudging it toward the reward it got plus the value of **the move it will actually make next**. Its numbers describe the agent as it really behaves, random exploratory moves included.

::: analogy
Learning a mountain trail while you sometimes trip. The route you end up preferring keeps away from steep drops, because your own clumsiness is part of what you experienced.
:::

### The update {#update}

$$\val{Q(S,A)} \leftarrow \val{Q(S,A)} + \alp\,\big[\,\rew{R} + \gam\,\val{Q(S',A')} - \val{Q(S,A)}\,\big]$$

Read it as *new guess = old guess + step size × (target − old guess)*. The target $\rew{R} + \gam\,\val{Q(S',A')}$ uses the next move $A'$, chosen *before* the update with the same ε-greedy rule. The bracket is the surprise $\del$. At a terminal tile there is no next move, so that term is 0.

### One change from TD(0) {#change}

[[td0]] learns the value of tiles; SARSA learns the value of moves:

| | update |
| --- | --- |
| [[td0]] | $\val{V(S)} \leftarrow \val{V(S)} + \alp\,[\rew{R} + \gam\,\val{V(S')} - \val{V(S)}]$ |
| SARSA | $\val{Q(S,A)} \leftarrow \val{Q(S,A)} + \alp\,[\rew{R} + \gam\,\val{Q(S',A')} - \val{Q(S,A)}]$ |

With values of moves, choosing what to do is easy: take the move with the highest number. No model of the world is needed.

### Backup diagram {#backup}

{{backup sarsa}}

From the move just taken (dot), through the reward and the next state (circle), to the one next move the agent chose (dot).

### Pseudocode

::: pseudocode
Parameters: step size $\alp \in (0, 1]$, exploration rate $\eps > 0$, discount $\gam$
Set $\val{Q(s,a)} = 0$ for every tile and move {#init}
Repeat for each episode:
  Put the agent at the start: $S$ {#start}
  Choose $A$ in $S$ with ε-greedy on $\val{Q}$ {#choose}
  Repeat until $S$ is terminal:
    Take $A$, observe $\rew{R}$ and $S'$ {#act}
    Choose $A'$ in $S'$ with ε-greedy on $\val{Q}$ {#choose-next}
    $\val{Q(S,A)} \leftarrow \val{Q(S,A)} + \alp\,[\rew{R} + \gam\,\val{Q(S',A')} - \val{Q(S,A)}]$ {#update}
    $S \leftarrow S'$ and $A \leftarrow A'$ {#next}
:::

### Perks

- Learns the value of what it really does, so while exploring it prefers safer routes. [See it](lab:cliff-race)
- One line of arithmetic per step, and it learns during the episode, not only at the end.
- If exploration fades out over time (ε shrinking toward 0), it still reaches the optimal policy.
- Learning about the policy it follows keeps it steadier than Q-learning once tables are replaced by approximations.

### Flaws

- With constant exploration it settles on the best *careful* route, not the best route. [See it](lab:cliff-race)
- Its values depend on ε: change the exploration and you change what it learns.
- One number per tile and move: it does not generalize and does not scale to big worlds ([[why-approximate]]).
- Information travels back one step per update, so long chains of decisions learn slowly ([[n-step-sarsa]] helps).

### Knobs

| Knob | Too low | Too high |
| --- | --- | --- |
| $\alp$ step size | learns very slowly | values jump around; on the cliff the learned route gets worse |
| $\eps$ exploration | may never discover a better route | the safe detour gets wider and slower |
| $\gam$ discount | short-sighted: ignores distant rewards | slow to settle when episodes are long or never end |

### Pitfalls

- Pick $A'$ before the update and then **really take it** on the next step. Picking a fresh action instead silently turns SARSA into a different algorithm.
- At a terminal state there is no $A'$: the target is just $\rew{R}$.
- Break ties between equal values at random, or exploration becomes lopsided.

### Check yourself {#check}

::: question
What do the five letters S-A-R-S-A stand for?
---
State, Action, Reward, next State, next Action: the five things one update uses.
:::

::: question
On the cliff, why does SARSA's path keep away from the edge?
---
Its values include its own random moves. Next to the edge a random step down costs −100, so the edge tiles look worse than the tiles one row up.
:::

::: question
If ε dropped to 0 over time, which path would SARSA end up with?
---
The shortest one, along the edge, like Q-learning. Without random moves the edge is no longer dangerous.
:::
