+++
summary = "Like SARSA, but instead of the one next move the agent happens to pick, the target averages over all next moves, each weighted by how likely the policy is to take it."
change = "Use the expected value of the next move under the policy, a sum over all actions weighted by their probabilities, instead of the value of the one next move that was sampled."
prereqs = ["sarsa", "q-learning", "epsilon-greedy"]
lab = "cliff-expected"
sources = [
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., §6.6 and Figure 6.3", url = "http://incompleteideas.net/book/the-book-2nd.html" },
  { text = "John (1994), When the best move isn't optimal: Q-learning with exploration, Proceedings of the 12th National Conference on Artificial Intelligence" },
  { text = "van Seijen, van Hasselt, Whiteson & Wiering (2009), A theoretical and empirical analysis of Expected Sarsa, IEEE Symposium on Adaptive Dynamic Programming and Reinforcement Learning", url = "https://doi.org/10.1109/ADPRL.2009.4927542" },
  { text = "van Hasselt (2011), Insights in Reinforcement Learning: formal analysis and empirical evaluation of temporal-difference learning algorithms, PhD thesis, Utrecht University" },
]

[story]
scene = "grid"
env = "cliff"
algorithm = "expected-sarsa"
alpha = 0.5
epsilon = 0.1
gamma = 1.0
episodes = 500
seed = 7
formula = '\step{1}{\val{Q(S,A)}} \step{2}{\leftarrow \val{Q(S,A)} + \alp\,\big[} \step{3}{\rew{R}} \step{4}{+ \gam \sum_a \pol{\pi(a \mid S^\prime)}\,\val{Q(S^\prime\!,a)}} \step{5}{- \val{Q(S,A)}\,\big]}'
numbers = '\val{Q(S,A)} \leftarrow 0 + 0.5\,\big[\rew{-1} + 1 \cdot \big(\tfrac14 \cdot 0 + \tfrac14 \cdot 0 + \tfrac14 \cdot 0 + \tfrac14 \cdot 0\big) - \val{0}\,\big] = \val{-0.5}'
+++

## Story

::: step {q = "zero", agent = "start"}
**The cliff world again**: start at the bottom left, reach the gem at the bottom right, $\rew{-1}$ per step and $\rew{-100}$ for stepping into the cliff. The agent explores with ε-greedy, $\eps = 0.1$, like [[sarsa]] and [[q-learning]].
:::

::: step {q = "zero", from = "start", agent = [2, 0], reward = -1, arrows = true}
It steps up and pays $\rew{-1}$. Now it needs a target: the reward plus the value of what comes next. SARSA would use the one move it is about to make; Q-learning the best move. **Expected SARSA uses all four.**
:::

::: step {q = "zero", agent = [2, 0], arrows = true, formula = 4}
Its target is the reward plus the **average** of the four next values, each weighted by how likely ε-greedy is to take that move: $1 - \eps + \eps/4 = 92.5\%$ for the best move and $\eps/4 = 2.5\%$ for each of the others. The arrows show those probabilities. Here all four are tied at 0, so each gets 25%.
:::

::: step {q = "zero", set = [[3, 0, "up", -0.5]], agent = [2, 0], glow = [3, 0, "up"], formula = 5, numbers = true}
With numbers: the target is $-1 + 0 = -1$, so *up from the start* moves halfway there, to $\val{-0.5}$, as it would for SARSA. The difference is not this update but every later one: the target no longer depends on which move the dice happen to pick next.
:::

::: step {q = "trained", ripple = true, path = true, agent = "start"}
After 500 episodes the greedy path keeps exactly one row of margin from the cliff: 15 steps. SARSA's path keeps two rows (17 steps); Q-learning's runs along the edge (13 steps). Expected SARSA, like SARSA, includes its own exploration in its values, but it does so precisely, without the noise of sampling it.
:::

::: step {q = "trained", path = true, explore = true}
Walking with $\eps = 0.1$: a random step down from this row lands on the edge row, not in the cliff. While learning, Expected SARSA collects about $\rew{-21}$ per episode in the last hundred episodes, against $\rew{-27}$ for SARSA and $\rew{-53}$ for Q-learning. [Race all three in the Lab](lab:cliff-expected).
:::

## Textbook

### The expected target {#target}

SARSA's target uses the action $A_{t+1}$ the agent actually takes next, a sample from its policy ([[sarsa]]). **Expected SARSA** replaces the sample by its expectation under the policy:

$$\val{Q(S_t,A_t)} \leftarrow \val{Q(S_t,A_t)} + \alp\,\Big[\,\rew{R_{t+1}} + \gam \sum_a \pol{\pi(a \mid S_{t+1})}\,\val{Q(S_{t+1},a)} - \val{Q(S_t,A_t)}\,\Big]. \label{update-rule}$$

Given the next state $S_{t+1}$, this algorithm moves deterministically in the same direction as SARSA moves *in expectation*, which is where its name comes from. If $S_{t+1}$ is terminal, the sum is 0. Its backup diagram has the next state branching into every action, each weighted by the policy, without the arc of a max (\ref{fig-backup}).

::: figure {#fig-backup}
{{backup expected-sarsa}}
The backup diagram of Expected SARSA: from the pair, through the reward and the next state, to every next action, each weighted by its probability under the policy.
:::

### Less variance {#variance}

Expected SARSA is more complex computationally than SARSA: each update sums over the actions, $O(|\mathcal{A}|)$, the same order as Q-learning's max. In return, it eliminates the variance due to the random selection of $A_{t+1}$. Conditioned on the transition, the SARSA target is a random variable whose mean is the Expected SARSA target:

$$\mathbb{E}\big[\,\rew{R_{t+1}} + \gam\,\val{Q(S_{t+1}, A_{t+1})} \;\big|\; S_t, A_t, \rew{R_{t+1}}, S_{t+1}\big] = \rew{R_{t+1}} + \gam \sum_a \pol{\pi(a \mid S_{t+1})}\,\val{Q(S_{t+1},a)}, \label{mean}$$

so by the law of total variance, the Expected SARSA target has less variance, by exactly the variance of $\gam\,\val{Q(S_{t+1}, A_{t+1})}$ over the choice of $A_{t+1}$. Given the same amount of experience, Expected SARSA generally performs slightly better than SARSA, and it tolerates larger step sizes.

### On-policy, off-policy, and Q-learning {#policies}

Nothing forces the policy in the expectation to be the policy that generates the behavior. With $\pol{\pi}$ equal to the behavior policy, for instance ε-greedy, Expected SARSA is an on-policy method like SARSA. With a different target policy it is an off-policy method, and it needs no importance sampling, because the next action is averaged over, not sampled ([[importance-sampling]]). In particular, if the target policy is greedy with respect to $\val{Q}$, the expectation is the max, and Expected SARSA *is* Q-learning ([[q-learning]]). In this sense Expected SARSA subsumes and generalizes Q-learning while reliably improving over SARSA. Except for the small additional computational cost, it may completely dominate both of the other well-known TD control algorithms.

### The algorithm {#algorithm}

::: algorithm {#alg-es} Expected SARSA (on-policy, with an ε-greedy target policy)
Parameters: step size $\alp \in (0, 1]$, small $\eps > 0$
Initialize $\val{Q(s,a)}$ for all $s \in \mathcal{S}^+$, $a \in \mathcal{A}(s)$, arbitrarily, except that $\val{Q(\textit{terminal}, \cdot)} = 0$
Loop for each episode:
  Initialize $S$
  Loop for each step of the episode, until $S$ is terminal:
    Choose $A$ from $S$ using the policy derived from $\val{Q}$ (e.g. ε-greedy)
    Take action $A$, observe $\rew{R}$, $S'$
    $\val{Q(S,A)} \leftarrow \val{Q(S,A)} + \alp\,[\rew{R} + \gam \sum_a \pol{\pi(a \mid S')}\,\val{Q(S',a)} - \val{Q(S,A)}]$
    $S \leftarrow S'$
:::

As with Q-learning, the next action need not be chosen before the update, since the target does not depend on it. Its convergence in the tabular case, under the same kinds of conditions as SARSA's, was proved by van Seijen, van Hasselt, Whiteson and Wiering (2009).

### Example: the cliff again {#cliff}

On the cliff ([[sarsa]]) with $\eps = 0.1$ and $\alp = 0.5$, Expected SARSA's greedy path after 500 episodes keeps one row of margin from the edge, 15 steps long, between SARSA's 17-step path and Q-learning's 13-step path along the edge. Over the last hundred of 500 episodes, averaged over 30 runs, it collects about $\rew{-21}$ per episode, against $\rew{-27}$ for SARSA and $\rew{-53}$ for Q-learning.

::: figure {#fig-alpha}
{{cliff-alpha}}
Average reward per episode on the cliff as a function of the step size, over the first 100 episodes (light) and over episodes 401 to 500 (dark), for SARSA, Expected SARSA and Q-learning with $\eps = 0.1$. The Lab computes the runs when the figure comes into view. After Sutton & Barto, Figure 6.3.
:::

\ref{fig-alpha} shows how the three methods depend on the step size. Expected SARSA improves steadily as $\alp$ grows, all the way to $\alp = 1$, while SARSA does best at an intermediate step size and deteriorates beyond it. The reason is that in cliff walking all the state transitions are deterministic and all the randomness comes from the policy. Expected SARSA removes the randomness of the policy from its target, so a large step size costs it nothing, while SARSA's sampled targets make large steps follow the noise.

### Relatives {#relatives}

| Method | Value of the next state in the target |
| --- | --- |
| [[sarsa]] | $\val{Q(S_{t+1}, A_{t+1})}$: the next move actually chosen |
| Expected SARSA | $\sum_a \pol{\pi(a \mid S_{t+1})}\,\val{Q(S_{t+1}, a)}$: all next moves, weighted by the policy |
| [[q-learning]] | $\max_a \val{Q(S_{t+1}, a)}$: the best next move (Expected SARSA with a greedy target policy) |
| [[double-q]] | $\val{Q_2(S_{t+1}, \operatorname*{arg\,max}_a Q_1(S_{t+1}, a))}$: the best move by one table, valued by the other |

Multi-step versions of the same idea, $n$-step Expected SARSA and Tree Backup, average over the actions not taken at every step, which makes them off-policy without importance sampling ([[n-step-sarsa]]).

### Historical remarks {#history}

An early form of the method, a variant of Q-learning that accounts for its own exploration, was suggested by John (1994). Van Seijen, van Hasselt, Whiteson and Wiering (2009) introduced Expected SARSA under that name, established its convergence and showed its advantages over SARSA and Q-learning; van Hasselt (2011) generalized it to arbitrary target policies, which includes Q-learning. The step-size comparison on the cliff follows Sutton and Barto (Figure 6.3).

## Card

### Idea

Expected SARSA's target averages the values of all next moves, weighted by how likely the policy is to take each one, instead of using the one move that was sampled. Same direction as SARSA on average, without the noise of the dice. With a greedy target policy, it is Q-learning.

::: analogy
Planning the next leg of a trip by averaging over every road you might take, weighted by how often you take each, instead of judging by the one road you happened to take this time.
:::

### The update {#update}

$$\val{Q(S,A)} \leftarrow \val{Q(S,A)} + \alp\,\Big[\,\rew{R} + \gam \sum_a \pol{\pi(a \mid S')}\,\val{Q(S',a)} - \val{Q(S,A)}\,\Big]$$

With ε-greedy, the weights are $1 - \eps + \eps/|\mathcal{A}|$ for the best move and $\eps/|\mathcal{A}|$ for each other. At a terminal state the sum is 0.

### One change from SARSA {#change}

| | value of the next state |
| --- | --- |
| [[sarsa]] | $\val{Q(S',A')}$: the one move sampled next |
| Expected SARSA | $\sum_a \pol{\pi(a \mid S')}\,\val{Q(S',a)}$: all moves, weighted by the policy |

### Backup diagram {#backup}

{{backup expected-sarsa}}

From the move just taken, through the reward and the next state, to every next move, each weighted by its probability (no arc: an average, not a max).

### Pseudocode

::: pseudocode
Parameters: step size $\alp \in (0, 1]$, exploration rate $\eps > 0$, discount $\gam$
Set $\val{Q(s,a)} = 0$ for every tile and move {#init}
Repeat for each episode:
  Put the agent at the start: $S$ {#start}
  Repeat until $S$ is terminal:
    Choose $A$ in $S$ with ε-greedy on $\val{Q}$ {#choose}
    Take $A$, observe $\rew{R}$ and $S'$ {#act}
    $\val{Q(S,A)} \leftarrow \val{Q(S,A)} + \alp\,[\rew{R} + \gam \sum_a \pol{\pi(a \mid S')}\,\val{Q(S',a)} - \val{Q(S,A)}]$ {#update}
    $S \leftarrow S'$ {#next}
:::

### Perks

- Less variance than SARSA: the target no longer depends on which next move is sampled. [See it](lab:cliff-expected)
- Tolerates large step sizes, especially when the world itself is deterministic.
- Works on-policy or off-policy without importance sampling; with a greedy target it is Q-learning.

### Flaws

- Each update sums over all actions: more work per step than SARSA.
- Needs the policy's probabilities for the next state, not just a sample.
- With an exploring target policy, like SARSA, it learns the values of an agent that keeps exploring.

### Knobs

| Knob | Too low | Too high |
| --- | --- | --- |
| $\alp$ step size | learns slowly | usually harmless where transitions are deterministic; noisy where they are not |
| $\eps$ exploration | rarely discovers better routes | the learned route keeps a wider margin |
| $\gam$ discount | short-sighted | slow to settle in long episodes |

### Pitfalls

- Using the greedy policy's weights in the sum while acting ε-greedily: that is Q-learning, not on-policy Expected SARSA.
- Splitting ties wrongly: tied best moves share the greedy probability equally.
- At a terminal next state, forgetting that the expectation is 0.

### Check yourself {#check}

::: question
With four moves and $\eps = 0.1$, what weight does the target give the best move?
---
$1 - 0.1 + 0.1/4 = 0.925$, and $0.025$ to each of the other three.
:::

::: question
Under which target policy is Expected SARSA exactly Q-learning?
---
The greedy policy: the expectation over a policy that always takes the best move is the max.
:::

::: question
Why can Expected SARSA use $\alp = 1$ on the cliff without trouble?
---
The cliff's transitions are deterministic and Expected SARSA averages out the randomness of the policy, so its targets have no noise left for a large step size to chase.
:::
