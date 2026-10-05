+++
summary = "Learn to act from whole episodes without choosing where they start: keep exploring with an ε-soft policy, and improve it toward ε-greedy after every episode."
change = "Drop exploring starts and explore with an ε-greedy policy instead: every action keeps a probability of at least ε/|A|, and the method learns the best policy among those that keep exploring."
prereqs = ["exploring-starts", "epsilon-greedy", "on-off-policy"]
lab = "frozen-mc"
sources = [
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., §5.4", url = "http://incompleteideas.net/book/the-book-2nd.html" },
  { text = "Singh, Jaakkola, Littman & Szepesvári (2000), Convergence results for single-step on-policy reinforcement-learning algorithms, Machine Learning 38", url = "https://doi.org/10.1023/A:1007678930559" },
  { text = "Barto & Duff (1994), Monte Carlo matrix inversion and reinforcement learning, Advances in Neural Information Processing Systems 6" },
]

[story]
scene = "grid"
env = "frozen-lake"
digits = 2
seed = 10
average = 20
formula = '\step{1}{\pol{\pi(a \mid s)} = \begin{cases} 1 - \eps + \eps / |\mathcal{A}| & a = \operatorname*{arg\,max}_{a^\prime} \val{Q(s,a^\prime)} \\ \eps / |\mathcal{A}| & \text{otherwise} \end{cases} \qquad} \step{2}{\val{Q(S_t,A_t)} \leftarrow \text{average of the returns}}'

[story.runs]
mc = { algorithm = "mc-control", epsilon = 0.1, alpha = 0.0, gamma = 0.99, judge = 1.0, units = 3000, measures = ["greedy"], name = "MC control (averages)" }
q = { algorithm = "q-learning", epsilon = 0.1, alpha = 0.1, gamma = 0.99, judge = 1.0, units = 3000, measures = ["greedy"], name = "Q-learning (α = 0.1)" }
+++

## Story

::: step {run = "mc", at = 0, values = true, formula = 1}
**Frozen Lake once more**, but now the agent does not know the rules, and it cannot choose where an episode starts: it always starts in the corner. It learns action values from whole episodes, as Monte Carlo ES does. To keep trying every move, it acts **ε-greedily**: one move in ten is random.
:::

::: step {run = "mc", at = 0, play = 1, pace = 150, formula = 2}
Episode 1: the agent wanders for 8 steps and falls through the ice. Walking back, every move it tried is credited with the return that followed it, which is 0. The estimates stay at 0, but each now counts one visit.
:::

::: step {run = "mc", at = 52, play = 1, pace = 90, formula = 2}
Episode 53 finally reaches the gem, after 27 slippery steps. Walking back, each move's first visit is credited with the return that followed it: $\rew{+1}$, discounted by 0.99 per step. The news travels all the way back to the start in this one episode.
:::

::: step {run = "mc", at = 53, values = true}
After that single episode, a trail of value leads from the start to the gem. A one-step temporal-difference method would have moved the news back by just one step.
:::

::: step {run = "mc", at = 300, values = true}
After 300 episodes, the greedy policy in these values reaches the gem half the time. The agent itself does worse, because one move in ten is still random.
:::

::: step {run = "mc", at = 3000, values = true, focus = [0, 0]}
After 3000 episodes the greedy policy reaches the gem 78% of the time, against 82% for the optimal policy. At the start it moves left, into the wall, like the optimal policy found by dynamic programming, here learned without the rules.
:::

::: step {run = "mc", at = 3000, values = true, curves = ["mc", "q"], metric = "greedy"}
Averaged over 20 runs, Monte Carlo gets going sooner: one successful episode credits a whole path. But its targets are whole returns, either 0 or a discounted 1, so its estimates are noisy, and they describe the exploring policy. Q-learning catches up after about a thousand episodes and ends higher. [Race them in the Lab](lab:frozen-mc).
:::

## Textbook

### Exploring without exploring starts {#soft}

Exploring starts are an unlikely assumption ([[exploring-starts]]). The only general way to ensure that all actions are selected infinitely often is for the agent to continue to select them. There are two approaches. **On-policy** methods evaluate or improve the policy that is used to make decisions; **off-policy** methods evaluate or improve a policy different from the one used to generate the data ([[on-off-policy]], [[off-policy-mc]]). Monte Carlo ES is an on-policy method. This station shows how to design an on-policy Monte Carlo control method that does not need exploring starts.

In on-policy control the policy is generally **soft**: $\pol{\pi(a \mid s)} > 0$ for all states and actions, but gradually shifted closer to a deterministic optimal policy. The methods here use ε-greedy policies ([[epsilon-greedy]]): most of the time they choose an action with maximal estimated value, but with probability $\eps$ they choose an action at random. All non-greedy actions get the minimal probability $\eps / |\mathcal{A}(s)|$, and the greedy action gets the rest, $1 - \eps + \eps / |\mathcal{A}(s)|$. A policy is **ε-soft** if $\pol{\pi(a \mid s)} \ge \eps / |\mathcal{A}(s)|$ for all states and actions; among ε-soft policies, ε-greedy policies are in a sense the closest to greedy.

### Improvement among ε-soft policies {#improvement}

The overall idea of on-policy Monte Carlo control is still generalized policy iteration ([[gpi]]): estimate the action values of the current policy from its episodes, and improve the policy. Without exploring starts, the policy cannot simply be made greedy, because that would stop the exploration of non-greedy actions. Fortunately, GPI does not require the policy to be taken all the way to greedy, only *toward* it.

::: theorem {#thm-soft} Policy improvement for ε-soft policies
Let $\pol{\pi}$ be any ε-soft policy, and let $\pol{\pi'}$ be the ε-greedy policy with respect to $\val{q_\pi}$. Then $\val{v_{\pi'}(s)} \ge \val{v_\pi(s)}$ for all $s$. Equality holds for all states only when $\pol{\pi}$ is optimal among the ε-soft policies.
:::

::: proof
For any state $s$, writing $m = |\mathcal{A}(s)|$,
$$\begin{aligned} q_\pi(s, \pi'(s)) &= \sum_a \pi'(a \mid s)\, q_\pi(s,a) = \frac{\varepsilon}{m} \sum_a q_\pi(s,a) + (1 - \varepsilon) \max_a q_\pi(s,a) \\ &\ge \frac{\varepsilon}{m} \sum_a q_\pi(s,a) + (1 - \varepsilon) \sum_a \frac{\pi(a \mid s) - \varepsilon/m}{1 - \varepsilon}\, q_\pi(s,a) = \sum_a \pi(a \mid s)\, q_\pi(s,a) = v_\pi(s). \end{aligned}$$
The inequality holds because the weights $(\pi(a \mid s) - \varepsilon/m)/(1 - \varepsilon)$ are nonnegative, since $\pi$ is ε-soft, and sum to 1, so their weighted average of $q_\pi(s, \cdot)$ is at most the maximum. The policy improvement theorem then gives $v_{\pi'} \ge v_\pi$. For the equality case, consider a new environment that behaves like the original, except that with probability $\varepsilon$ it replaces the agent's action by a random one; the best ε-soft policies in the original environment are exactly the best policies in the new one, and equality means that $v_\pi$ satisfies the new environment's Bellman optimality equation.
:::

So policy iteration works for ε-soft policies: each improvement step is guaranteed to give a better ε-soft policy, until the best ε-soft policy is reached. The only cost is that the method achieves the best policy among the ε-soft ones, not the best policy overall; but exploring starts are no longer needed.

### The algorithm {#algorithm}

::: algorithm {#alg-mcc} On-policy first-visit Monte Carlo control (for ε-soft policies), for estimating $\pol{\pi} \approx \pol{\pi_*}$
Algorithm parameter: small $\eps > 0$
Initialize: $\pol{\pi} \leftarrow$ an arbitrary ε-soft policy; $\val{Q(s,a)} \in \mathbb{R}$ arbitrarily, for all $s, a$; $\textit{Returns}(s,a) \leftarrow$ an empty list, for all $s, a$
Repeat forever (for each episode):
  Generate an episode following $\pol{\pi}$: $S_0, A_0, \rew{R_1}, \ldots, S_{T-1}, A_{T-1}, \rew{R_T}$
  $\rew{G} \leftarrow 0$
  Loop for each step of the episode, $t = T-1, T-2, \ldots, 0$:
    $\rew{G} \leftarrow \gam\,\rew{G} + \rew{R_{t+1}}$
    Unless the pair $S_t, A_t$ appears in $S_0, A_0, S_1, A_1, \ldots, S_{t-1}, A_{t-1}$:
      Append $\rew{G}$ to $\textit{Returns}(S_t, A_t)$
      $\val{Q(S_t, A_t)} \leftarrow \text{average}(\textit{Returns}(S_t, A_t))$
      $A^* \leftarrow \operatorname*{arg\,max}_a \val{Q(S_t, a)}$, with ties broken arbitrarily
      For all $a \in \mathcal{A}(S_t)$: $\pol{\pi(a \mid S_t)} \leftarrow 1 - \eps + \eps/|\mathcal{A}(S_t)|$ if $a = A^*$, else $\eps/|\mathcal{A}(S_t)|$
:::

In practice the policy need not be stored: acting ε-greedily with respect to the current $\val{Q}$ is the same thing.

### What it converges to {#limit}

With a constant $\eps$, the method approaches the best ε-soft policy and its action values, which still include the cost of exploring: like [[sarsa]], it learns the values of an agent that keeps making random moves, and prefers routes on which random moves are less dangerous. To reach an optimal policy, exploration must fade out: if $\eps$ decays toward zero in a way that keeps every pair visited infinitely often, a schedule that is greedy in the limit with infinite exploration (GLIE), the policy converges to an optimal one (Singh et al., 2000).

### Example: Frozen Lake {#example}

On Frozen Lake ([[policy-iteration]]), where only the gem pays and the ice is slippery, Monte Carlo control with $\eps = 0.1$ and averages learns from the very first successful episode: the return is credited to every pair that episode visited, so a whole path from the start to the gem gains value at once. \ref{fig-race} compares it with [[q-learning]] on the same task, by the chance that the greedy policy with respect to each one's estimates reaches the gem.

::: figure {#fig-race}
{{frozen-mc}}
The chance that the greedy policy reaches the gem, for on-policy Monte Carlo control (averages, $\eps = 0.1$) and Q-learning ($\alp = 0.1$, $\eps = 0.1$), with $\gam = 0.99$. Each curve averages 20 runs; the line marks the optimal policy. The Lab computes them when the figure comes into view.
:::

Monte Carlo is ahead for the first few hundred episodes. Then Q-learning overtakes it, and after 3000 episodes its greedy policy reaches the gem about 79% of the time against about 50% for Monte Carlo. Two differences explain it. Monte Carlo's targets are whole returns, either 0 or a discounted 1, which vary a great deal from episode to episode; Q-learning's one-step targets vary much less ([[mc-vs-td]]). And on-policy Monte Carlo estimates the values of the exploring policy, while Q-learning estimates those of the greedy policy directly ([[on-off-policy]]). A constant step size does not rescue Monte Carlo here: with $\alp = 0.05$ its estimates are noisier still, and it does worse.

### Properties {#properties}

- **No exploring starts.** Exploration comes from the policy itself, so the method can learn from real interaction.
- **Learns the best ε-soft policy,** not the best policy, unless $\eps$ decays.
- **Credits whole paths at once.** A single rewarding episode updates every pair along the way, which helps when rewards are rare at first.
- **High-variance targets.** Every return depends on all the random events of its episode.
- **Episodic only,** and learning happens only at the ends of episodes.

### Historical remarks {#history}

On-policy Monte Carlo control with ε-soft policies, and the argument that improvement works within the ε-soft class, follow Sutton and Barto (§5.4). The convergence of on-policy methods under GLIE exploration was established by Singh, Jaakkola, Littman and Szepesvári (2000). Barto and Duff (1994) discussed Monte Carlo methods in the context of dynamic programming and reinforcement learning.

## Card

### Idea

Learn action values from whole episodes, and act ε-greedily on them: mostly the best-looking action, sometimes a random one. The random actions keep every action tried, so no exploring starts are needed. The price: it learns the best policy for an agent that keeps exploring.

::: analogy
Learning the best way through a new city with a rule: one turn in ten is chosen at random. The rule keeps showing you streets you would never have tried, and your favorite route ends up one that still works when you take a wrong turn now and then.
:::

### The update {#update}

$$\val{Q(S_t,A_t)} \leftarrow \val{Q(S_t,A_t)} + \frac{1}{N(S_t,A_t)}\,\big[\,\rew{G_t} - \val{Q(S_t,A_t)}\,\big] \qquad \text{then act ε-greedily with respect to } \val{Q}$$

After every episode, at the first visit to each pair.

### One change from Monte Carlo ES {#change}

| | explores by | learns the best |
| --- | --- | --- |
| [[exploring-starts]] | a random start state and first action | policy |
| on-policy MC control | ε-greedy actions throughout | ε-soft policy |

### Backup diagram {#backup}

{{backup mc-q}}

From the state–action pair being updated along the episode that was played, to its end.

### Pseudocode

::: pseudocode
Parameters: exploration rate $\eps > 0$, discount $\gam$
Set $\val{Q(s,a)} = 0$ and $N(s,a) = 0$ for every state and action {#init}
Repeat for each episode:
  Start at the start state $S_0$ {#start}
  Play the episode to the end, choosing actions ε-greedily with respect to $\val{Q}$ {#generate}
  $\rew{G} \leftarrow 0$ {#g0}
  For $t = T-1$ down to $0$: $\rew{G} \leftarrow \gam\,\rew{G} + \rew{R_{t+1}}$ {#return}
    If the pair $S_t, A_t$ was already visited earlier in the episode, skip it {#first}
    $N \leftarrow N + 1$ and $\val{Q(S_t,A_t)} \leftarrow \val{Q(S_t,A_t)} + \tfrac{1}{N}\,[\rew{G} - \val{Q(S_t,A_t)}]$ {#update}
:::

### Perks

- Needs neither a model nor control over where episodes start. [See it](lab:frozen-mc)
- One rewarding episode credits every move along its path.
- Simple: averages, and an ε-greedy choice.

### Flaws

- Learns the best ε-soft policy, which pays for its own exploration, unless ε decays.
- Noisy targets: whole returns vary a lot. On Frozen Lake, Q-learning overtakes it. [See it](lab:frozen-mc)
- Learns only at the end of episodes.

### Knobs

| Knob | Too low | Too high |
| --- | --- | --- |
| $\eps$ exploration | some moves are almost never tried; their values stay unknown | the learned policy is cautious, and acting is often random |
| $\alp$ (if constant instead of averages) | slow to forget old returns | even noisier estimates |

### Pitfalls

- Judging what was learned by the ε-greedy behavior: test the greedy policy.
- Expecting the optimal policy with a constant ε: the method converges to the best ε-soft policy.
- Forgetting the first-visit check: counting repeated visits in one episode turns it into every-visit MC, a different estimator.

### Check yourself {#check}

::: question
What does "ε-soft" mean, and why does the policy have to be ε-soft?
---
Every action has probability at least $\eps / |\mathcal{A}|$. Without that, actions that look bad would never be tried, and without exploring starts their values could never be corrected.
:::

::: question
Why is ε-greedy improvement guaranteed not to make an ε-soft policy worse?
---
Its expected one-step value in each state, $\tfrac{\eps}{m}\sum_a q + (1-\eps)\max_a q$, is at least the policy's own average $\sum_a \pol{\pi(a \mid s)}\,q$, because the ε-soft policy's extra weight above $\eps/m$ can at best pick the max. The policy improvement theorem does the rest.
:::

::: question
On Frozen Lake, why does Monte Carlo start faster than Q-learning?
---
When an episode finally reaches the gem, Monte Carlo credits every move along the whole path at once. Q-learning passes the news back only one step per update.
:::
