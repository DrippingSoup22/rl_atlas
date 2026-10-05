+++
summary = "Pull the arm whose value could plausibly be highest: its estimate plus a bonus for uncertainty that shrinks every time the arm is tried."
change = "Explore where the estimates are uncertain instead of at random: pull the arm with the highest estimate plus a bonus that shrinks the more often the arm has been pulled."
prereqs = ["epsilon-greedy", "optimistic-init"]
lab = "bandit-ucb"
sources = [
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., §2.7, Figure 2.4 and Exercise 2.8", url = "http://incompleteideas.net/book/the-book-2nd.html" },
  { text = "Lai & Robbins (1985), Asymptotically efficient adaptive allocation rules, Advances in Applied Mathematics 6", url = "https://doi.org/10.1016/0196-8858(85)90002-8" },
  { text = "Agrawal (1995), Sample mean based index policies with O(log n) regret for the multi-armed bandit problem, Advances in Applied Probability 27" },
  { text = "Auer, Cesa-Bianchi & Fischer (2002), Finite-time analysis of the multiarmed bandit problem, Machine Learning 47", url = "https://doi.org/10.1023/A:1013689704352" },
  { text = "Kocsis & Szepesvári (2006), Bandit based Monte-Carlo planning, European Conference on Machine Learning" },
  { text = "Garivier & Moulines (2011), On upper-confidence bound policies for switching bandit problems, Algorithmic Learning Theory" },
  { text = "Bellemare et al. (2016), Unifying count-based exploration and intrinsic motivation, Advances in Neural Information Processing Systems 29" },
]

[story]
scene = "bandit"
env = "testbed"
seed = 58
units = 1000
average = 200
formula = '\step{1}{A \leftarrow \operatorname*{arg\,max}_a \Big[\,\val{Q(a)}} \step{2}{+\; c\,\sqrt{\frac{\ln t}{N(a)}}\;\Big]}'
numbers = '\val{Q(2)} + 2\sqrt{\tfrac{\ln 17}{5}} = \val{1.55} + 1.51 = 3.06'

[story.runs]
ucb = { algorithm = "ucb", c = 2.0, alpha = 0.0, name = "UCB (c = 2)" }
eps = { algorithm = "epsilon-greedy", epsilon = 0.1, alpha = 0.0, name = "ε-greedy (ε = 0.1)" }
+++

## Story

::: step {run = "ucb", at = 0, truth = false, formula = 1}
**The ten machines once more.** UCB keeps the same estimates as before, the average reward of each arm, but chooses differently. It does not trust an estimate by itself; it asks how high the arm's value *could plausibly be*.
:::

::: step {run = "ucb", at = 0, truth = false, play = 10, pace = 600, formula = 2}
To the estimate it adds a **bonus** that is large when the arm has been pulled rarely. The whisker above each bar shows estimate plus bonus. An arm never pulled has an infinite bonus, so the first ten pulls try every arm once, without any randomness.
:::

::: step {run = "ucb", at = 16, truth = false, numbers = true, formula = 2}
After 16 pulls, arm 2 has been pulled 5 times and averages $\val{1.55}$; its bonus is $2\sqrt{\ln 17 / 5} = 1.51$, so its top is at 3.06, the highest. Arm 1 has paid $\rew{-0.43}$ on its only pull, yet its bonus of 3.37 lifts it almost as high. The fewer pulls, the more an arm might be better than it looks.
:::

::: step {run = "ucb", at = 100, truth = false}
Every pull shrinks the pulled arm's bonus, and the bonus of each waiting arm creeps up as the step count $t$ grows. So after 100 pulls the tops are nearly level: UCB keeps pulling whichever arm's top is highest. Arm 2 has had 66 of those pulls.
:::

::: step {run = "ucb", at = 1000, truth = true}
After 1000 pulls: 920 on arm 2. The hopeless arms were barely touched. Arm 3, worth $\rew{-1.99}$, was pulled 3 times; ε-greedy pulled it 14 times, because its random pulls cannot tell a hopeless arm from a promising one.
:::

::: step {run = "ucb", at = 1000, truth = true, curves = ["ucb", "eps"], metric = "return"}
Averaged over 200 problems, UCB earns more per step than ε-greedy once its first ten pulls are done. Note the spike at step 11, when every arm has been tried once. [Race them in the Lab](lab:bandit-ucb).
:::

## Textbook

### Exploring where it is needed {#idea}

Exploration is needed because the estimates are uncertain ([[explore-exploit]]). ε-greedy explores without regard to that uncertainty: its random choices fall on an arm that has been tried a thousand times and is clearly bad as often as on an arm tried twice whose estimate could be far off ([[epsilon-greedy]]). It would be better to choose among the non-greedy actions according to their potential for actually being optimal, taking into account both how close their estimates are to the maximum and how uncertain those estimates are.

### The rule {#rule}

**Upper-confidence-bound** action selection chooses, at step $t$,

$$A_t = \operatorname*{arg\,max}_a \Big[\,\val{Q_t(a)} + c\,\sqrt{\frac{\ln t}{N_t(a)}}\;\Big], \label{ucb-rule}$$

where $N_t(a)$ is the number of times $a$ was chosen before step $t$, $c > 0$ sets how much to explore, and an action with $N_t(a) = 0$ counts as maximizing, so every action is tried once first. The square-root term measures the uncertainty in the estimate of $a$. The bracket is thus a kind of upper bound on the possible true value of $a$, with $c$ setting the confidence level.

The two parts of the bonus move in opposite directions. Each time $a$ is chosen, $N_t(a)$ grows and its bonus shrinks. Each time another action is chosen, $t$ grows while $N_t(a)$ does not, so the bonus of $a$ rises, slowly, since $\ln t$ is unbounded but grows ever more slowly. Every action is therefore chosen again eventually, but an action with a low estimate, or one that has already been tried often, is chosen less and less frequently as time goes on. This is the principle of optimism in the face of uncertainty ([[optimistic-init]]) made systematic: the optimism of each arm is tailored to how much is known about it, and it never runs out.

### Where the bonus comes from {#hoeffding}

The form of the bonus comes from a concentration inequality. If rewards lie in $[0, 1]$ and $\val{Q}$ is the average of $n$ of them, Hoeffding's inequality bounds the chance that the true value exceeds the estimate by more than $u$:

$$\Pr\big(\val{q_*(a)} > \val{Q} + u\big) \le e^{-2 n u^2}. \label{hoeffding-ineq}$$

Asking for a probability that shrinks with time, $t^{-4}$, and solving $e^{-2nu^2} = t^{-4}$ gives $u = \sqrt{2 \ln t / n}$: the bonus of \ref{ucb-rule} with $c = \sqrt{2}$. This is the algorithm UCB1 of Auer, Cesa-Bianchi and Fischer (2002). For rewards on a different scale, $c$ scales with the width of their range. On the testbed, whose rewards have standard deviation 1, $c = 2$ is a common choice.

::: theorem {#thm-ucb1} Auer, Cesa-Bianchi & Fischer, 2002
For a $k$-armed bandit with rewards in $[0, 1]$, UCB1 chooses each suboptimal action $a$ in expectation at most $\dfrac{8 \ln T}{\Delta_a^2} + 1 + \dfrac{\pi^2}{3}$ times in the first $T$ steps, where $\Delta_a$ is the gap between the best value and $\val{q_*(a)}$. Its regret is therefore at most
$$\sum_{a:\,\Delta_a > 0} \Big(\frac{8 \ln T}{\Delta_a} + \big(1 + \tfrac{\pi^2}{3}\big)\,\Delta_a\Big). \label{regret-bound}$$
:::

::: proof Proof idea
A suboptimal action $a$ is chosen only if its upper bound exceeds that of a best action. That requires one of three things: the best action's estimate is far below its value, the estimate of $a$ is far above its value, or $a$ has been tried too few times for its bonus to have shrunk below about $\Delta_a / 2$. By \ref{hoeffding-ineq}, the first two have probability at most $t^{-4}$ each at step $t$, which summed over $t$ gives the constant term. The third can happen only while $N_t(a) < 8 \ln T / \Delta_a^2$.
:::

Regret that grows like $\ln T$ is the best possible rate (Lai & Robbins, 1985). With a constant $\eps$, ε-greedy's regret grows linearly ([[k-armed-bandit]]).

### Results on the testbed {#testbed}

::: figure {#fig-curves}
{{bandit-curves ucb}}
Average reward per step of UCB with $c = 2$ and of ε-greedy with $\eps = 0.1$, both with sample averages, on the 10-armed testbed. Each curve averages 1000 runs; the Lab computes them when the figure comes into view. After Sutton & Barto, Figure 2.4.
:::

UCB does better than ε-greedy except during the first $k$ steps, when it is still trying each action once (\ref{fig-curves}). Over the last hundred steps of the runs above it earns about 1.52 per step, against 1.39 for ε-greedy, close to the 1.54 a player knowing the values would get. The spike at step 11 has a simple cause (Sutton & Barto, Exercise 2.8). After ten steps every arm has been tried once, so every bonus is the same, $c\sqrt{\ln 11}$, and the choice at step 11 goes to the arm with the best single reward, which is the best arm more often than not. Then its count doubles, its bonus drops below the bonuses of the arms tried once, and the next steps return to them.

The value of $c$ matters. Too small, and UCB is nearly greedy; too large, and it keeps testing arms long after the evidence is clear. The parameter study in [[gradient-bandit]] compares UCB across values of $c$ with the other bandit methods.

### Limits {#limits}

UCB is more difficult than ε-greedy to extend beyond bandits.

- **Changing values.** The counts $N_t(a)$ measure how much is known about an arm only if what was learned stays true. If values drift, old pulls overstate the knowledge. Variants discount old pulls or keep only a recent window (Garivier & Moulines, 2011), but the plain rule is built for stationary problems ([[nonstationary]]).
- **Large state spaces.** With states, the bonus needs a count for every state–action pair. When states are rarely or never revisited, as with images, counts are useless, and they must be replaced by *pseudo-counts* or other measures of novelty (Bellemare et al., 2016; [[exploration-strategies]]).
- **Function approximation.** The confidence interval assumes independent samples of a fixed quantity; estimates that share parameters, bootstrap, and chase a moving policy break that assumption.

Yet UCB has a prominent second life in planning. UCT (Kocsis & Szepesvári, 2006) applies the rule at every node of a search tree, treating the choice among moves as a bandit, and its descendants guide the tree search of AlphaGo and AlphaZero ([[mcts]]).

### Historical remarks {#history}

Lai and Robbins (1985) introduced index policies based on upper confidence bounds and proved them asymptotically optimal. Agrawal (1995) gave simpler indices computed from sample means, and Auer, Cesa-Bianchi and Fischer (2002) introduced UCB1 with its finite-time guarantee, the form used today. Sutton and Barto (§2.7) compare it with ε-greedy on the testbed. Kocsis and Szepesvári (2006) carried it into tree search as UCT.

## Card

### Idea

Pull the arm whose value could plausibly be the highest: its estimate plus a bonus for how little it has been tried. Arms tried rarely get a big bonus; arms tried often must earn their place with their estimate. Exploration goes where the uncertainty is.

::: analogy
Choosing between restaurants by giving the ones you have barely tried the benefit of the doubt. A place you have visited thirty times is judged by its food; a place you visited once still gets a generous rating, because one meal says little.
:::

### The rule {#update}

$$A_t = \operatorname*{arg\,max}_a \Big[\,\val{Q_t(a)} + c\sqrt{\frac{\ln t}{N_t(a)}}\;\Big] \qquad \val{Q(A)} \leftarrow \val{Q(A)} + \tfrac{1}{N(A)}\big[\rew{R} - \val{Q(A)}\big]$$

An arm never pulled ($N = 0$) counts as best. The bonus shrinks when the arm is pulled and grows slowly, with $\ln t$, while it waits.

### One change from [[epsilon-greedy]] {#change}

| | explores |
| --- | --- |
| [[epsilon-greedy]] | a random arm with probability $\eps$, whatever is known about it |
| UCB | the arm with the highest estimate + bonus: uncertain arms first |

### Backup diagram {#backup}

{{backup bandit}}

From the arm pulled to its reward, as in every bandit method. UCB changes how the arm is chosen, not what its estimate is updated from.

### Pseudocode

::: pseudocode
Parameters: confidence level $c > 0$
Set $\val{Q(a)} = 0$ and $N(a) = 0$ for every arm $a$ {#init}
For $t = 1, 2, \ldots$:
  Choose $A$ maximizing $\val{Q(a)} + c\sqrt{\ln t / N(a)}$; an arm with $N(a) = 0$ counts as the maximum {#choose}
  Pull $A$, observe $\rew{R}$ {#act}
  $N(A) \leftarrow N(A) + 1$ and $\val{Q(A)} \leftarrow \val{Q(A)} + \tfrac{1}{N(A)}\,[\rew{R} - \val{Q(A)}]$ {#update}
:::

### Perks

- Explores where it matters: hopeless arms are soon left alone. [See it](lab:bandit-ucb)
- Its regret grows only like $\ln t$, the best possible rate, with a guarantee for any number of steps.
- No randomness: the same history always leads to the same choice.

### Flaws

- Built for stationary problems: old pulls keep counting as knowledge after the values change.
- Hard to extend to large state spaces, where states are rarely counted twice.
- $c$ must suit the scale of the rewards.

### Knobs

| Knob | Too low | Too high |
| --- | --- | --- |
| $c$ confidence | nearly greedy: can lock onto a worse arm | keeps testing arms long after the evidence is clear |

### Pitfalls

- Dividing by $N(a) = 0$: untried arms must be handled first, as infinitely good.
- Starting the step count at 0: $\ln 0$ is undefined and $\ln 1 = 0$ gives no bonus at all; $t$ counts from 1.
- Using a constant step size: the bonus assumes that $\val{Q(a)}$ is an average of $N(a)$ rewards.

### Check yourself {#check}

::: question
Two arms have the same estimate. One was pulled 100 times, the other 4 times. Which does UCB prefer, and why?
---
The one pulled 4 times: its bonus, proportional to $\sqrt{1/N}$, is five times larger. Its estimate is less certain, so its value could more plausibly be high.
:::

::: question
Why does UCB eventually try every arm again, however bad it looks?
---
While an arm waits, $t$ grows and so does $\ln t$ in its bonus, without limit. Eventually its upper bound exceeds the others, though for a bad arm that takes very long.
:::

::: question
Why does UCB's average reward spike at step 11 on the testbed?
---
After ten steps each arm has been tried once, so all bonuses are equal and UCB picks the arm with the best single reward, often the best arm. Then that arm's bonus shrinks and UCB goes back to the others for a while.
:::
