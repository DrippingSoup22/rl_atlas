+++
summary = "A policy-gradient learner tends to commit to the first thing that works: once an action is nearly certain, the others are never tried and never improve. The entropy bonus adds a small reward for staying uncertain, a push toward uniform choices that weakens as the policy finds something clearly better. It is the usual way policy-gradient methods keep exploring."
prereqs = ["a2c", "policy-parameterization", "explore-exploit"]
lab = "entropy-gems"
sources = [
  { text = "Williams & Peng (1991), Function optimization using connectionist reinforcement learning algorithms, Connection Science 3" },
  { text = "Mnih, Badia, Mirza, Graves, Lillicrap, Harley, Silver & Kavukcuoglu (2016), Asynchronous methods for deep reinforcement learning, ICML", url = "https://arxiv.org/abs/1602.01783" },
  { text = "Schulman, Wolski, Dhariwal, Radford & Klimov (2017), Proximal policy optimization algorithms", url = "https://arxiv.org/abs/1707.06347" },
  { text = "Haarnoja, Zhou, Abbeel & Levine (2018), Soft actor-critic: off-policy maximum entropy deep reinforcement learning with a stochastic actor, ICML", url = "https://arxiv.org/abs/1801.01290" },
  { text = "Ahmed, Le Roux, Norouzi & Schuurmans (2019), Understanding the impact of entropy on policy optimization, ICML" },
]

[story]
scene = "grid"
env = "two-gems"
seed = 1
average = 20
formula = '\step{1}{\boldsymbol\theta \leftarrow \boldsymbol\theta + \frac{\alp}{N} \sum \Big[\err{\hat A_t}\,\nabla \ln \pol{\pi(A_t \mid S_t, \boldsymbol\theta)}} \step{2}{+ \beta\,\nabla \pol{H\big(\pi(\cdot \mid S_t, \boldsymbol\theta)\big)}} \step{1}{\Big]} \step{2}{\qquad \pol{H} = -\sum_a \pol{\pi(a \mid s)} \ln \pol{\pi(a \mid s)}}'

[story.runs]
greedy = { algorithm = "a2c", alpha = 2.0, alphaW = 0.3, workers = 4, n = 5, beta = 0.0, gamma = 0.95, maxSteps = 500, units = 200, name = "no entropy bonus" }
bonus = { algorithm = "a2c", alpha = 2.0, alphaW = 0.3, workers = 4, n = 5, beta = 0.1, gamma = 0.95, maxSteps = 500, units = 200, name = "entropy bonus, β = 0.1" }
+++

## Story

::: step {run = "greedy", at = 0}
**Two gems.** A small one, two steps from the start, pays $\rew{0.3}$; a big one, seven steps away, pays $\rew{1}$. Either ends the episode, and with $\gam = 0.95$ the big gem is still worth much more from the start: $0.95^6 \approx 0.74$, against $0.95 \times 0.3 \approx 0.29$. Four [[a2c|A2C]] workers share one policy, all directions equally likely.
:::

::: step {run = "greedy", at = 1, formula = 1}
The first round, and the four paths it took: three workers stumbled on the small gem, one on the big gem. The small gem is close, so it is found often and early, and every time, the moves that led to it become more likely.
:::

::: step {run = "greedy", at = 20}
After 20 rounds the policy has committed: from the start, up, toward the small gem, 82% of the time. All four workers now go to it, in 3 steps on average.
:::

::: step {run = "greedy", at = 200, values = true}
After 200 rounds: up 97% of the time, and the critic values the start at 0.28, exactly what the small gem is worth. The big gem is never seen again, so nothing can teach the policy that it was better. A policy that stops exploring stops learning.
:::

::: step {run = "bonus", at = 0, formula = 2}
The same workers, the same first round, with one change: an **entropy bonus**. The entropy $\pol{H}$ of the policy in a state measures how undecided it is: largest when all moves are equally likely, zero when one is certain. Each update now also pushes the policy up the gradient of its entropy, weighted by $\beta = 0.1$: lopsided choices are pulled back toward even ones.
:::

::: step {run = "bonus", at = 40}
After 40 rounds the policy at the start is still undecided: up 41%, left 28%, right 15%, down 16%. The small gem keeps paying, but the bonus keeps the policy from committing to it, so the workers keep wandering, and now and then one finds the big gem.
:::

::: step {run = "bonus", at = 150, values = true}
Around round 100 the big gem starts to win: its advantage, larger each time it is found, outweighs the pull of the small one. After 150 rounds, right from the start is the favorite, 58%, and all four workers reach the big gem. The policy is still random enough to wander a little: 8.5 steps on average, against 7 for the shortest path.
:::

::: step {run = "bonus", at = 200, curves = ["greedy", "bonus"], metric = "return"}
Averaged over 20 runs: without the bonus, every run settles on the small gem, $\rew{0.30}$ per episode. With $\beta = 0.1$, 18 runs go for the big one after 200 rounds and the last two get there by round 240: $\rew{0.91}$ per episode after 200 rounds. [Change β in the Lab](lab:entropy-gems).
:::

## Textbook

### Premature commitment {#premature}

A policy-gradient method moves its policy toward actions that turned out better than expected. Early on, *better than expected* can simply mean *the first thing that worked*. If one action is found to pay before the alternatives have been tried enough, its probability rises; it is then chosen more, its advantage is confirmed more, and its probability rises again. As $\pol{\pi(a \mid s)} \to 1$, the other actions are almost never sampled, their advantages are never estimated, and the gradient of $J$ with respect to them vanishes ([[reinforce]]). The policy has converged, to a local optimum it cannot leave: the [[explore-exploit|exploration–exploitation dilemma]] in policy form.

Value-based methods explore with a fixed rule, such as $\varepsilon$-greedy. A stochastic policy explores by its own randomness, which is learned and therefore free to disappear too soon.

### Entropy {#entropy}

The **entropy** of the policy in a state measures how spread out its choices are:

$$\pol{H\big(\pi(\cdot \mid s)\big)} = -\sum_a \pol{\pi(a \mid s)} \ln \pol{\pi(a \mid s)}. \label{eq-entropy}$$

It is largest, $\ln |\mathcal A|$, for the uniform policy and zero for a deterministic one. For a Gaussian policy it depends only on the spread: $\pol{H} = \ln \pol{\sigma} + \frac12 \ln(2\pi e)$ ([[policy-parameterization]]).

### The bonus {#bonus}

The **entropy bonus** adds the entropy of the policy, in the states the agent visits, to the objective:

$$J_\beta(\boldsymbol\theta) = J(\boldsymbol\theta) + \beta\,\mathbb E_\pi\big[\pol{H\big(\pi(\cdot \mid S_t, \boldsymbol\theta)\big)}\big], \label{eq-objective}$$

and each sampled update gains a term $\beta\,\nabla \pol{H(\pi(\cdot \mid S_t, \boldsymbol\theta))}$, as in [[a2c|A2C]] and [[ppo|PPO]]. The weight $\beta \ge 0$ sets how much uncertainty is worth compared with return.

::: lemma {#lem-grad} The gradient of a softmax policy's entropy
For a softmax over preferences $\pol{h(s, b)}$,

$$\frac{\partial \pol{H}}{\partial \pol{h(s, b)}} = -\pol{\pi(b \mid s)}\,\big(\ln \pol{\pi(b \mid s)} + \pol{H}\big). \label{eq-grad}$$
:::

::: proof
With $\ln \pol{\pi(a)} = \pol{h(a)} - \ln \sum_c e^{\pol{h(c)}}$, the softmax gives $\partial \pol{\pi(a)} / \partial \pol{h(b)} = \pol{\pi(a)}\,(\mathbb 1[a = b] - \pol{\pi(b)})$. Then $\partial \pol{H} / \partial \pol{h(b)} = -\sum_a (\ln \pol{\pi(a)} + 1)\,\pol{\pi(a)}\,(\mathbb 1[a = b] - \pol{\pi(b)}) = -\pol{\pi(b)}\,(\ln \pol{\pi(b)} + 1) + \pol{\pi(b)} \sum_a \pol{\pi(a)}\,(\ln \pol{\pi(a)} + 1)$, and the last sum is $1 - \pol{H}$.
:::

The bracket $\ln \pol{\pi(b)} + \pol{H}$ is negative for actions less likely than $e^{-\pol{H}}$ and positive for the more likely ones, so the bonus raises the preferences of rare actions and lowers those of common ones: a push toward uniform, zero at the uniform policy itself. It is strongest for actions that are neither rare nor dominant and fades as probabilities approach 0, so it cannot keep alive an action the return has firmly ruled out. For a Gaussian policy, $\partial \pol{H} / \partial \ln \pol{\sigma} = 1$: a constant push to widen the bell, balanced against the return's push to narrow it.

### Choosing $\beta$ {#beta}

The bonus trades return for exploration, and the right amount depends on the scale of the rewards. Too little and the policy commits early; too much and it stays so random that it collects less than it could, or never commits at all. With rewards normalized to about unit size, values around $\beta = 0.01$ are common (Mnih et al., 2016, for Atari; Schulman et al., 2017, add an entropy term of the same size to PPO's objective). Some methods let $\beta$ decay over training, or adjust it automatically to keep the entropy near a target ([[sac]]).

::: figure {#fig-study}
{{entropy-study}}
A2C on the two gems for seven entropy weights, 16 runs each. Without the bonus no run reaches the big gem; $\beta = 0.05$ reaches it in 44% of runs, $\beta = 0.1$ in 94%, with $\rew{0.88}$ per episode at the end. Larger weights always find it, but keep the policy too random to collect it reliably: $\rew{0.58}$ per episode with $\beta = 0.5$. Computed live by the Lab.
:::

### Entropy in the objective {#max-ent}

The bonus \ref{eq-objective} rewards uncertainty only in the states the agent visits now, and only for the step at hand. *Maximum-entropy* reinforcement learning goes further and adds the entropy to every reward, $\rew{R_{t+1}} + \alpha\,\pol{H(\pi(\cdot \mid S_t))}$, so that the agent also values *reaching* states where it can keep its options open. Values and policies then become “soft”, with a softmax in place of the max of the Bellman optimality equation. [[sac|Soft actor–critic]] is built this way, and tunes the temperature $\alpha$ automatically.

### Example: two gems {#example}

In a 5 × 8 grid, the start is two steps from a small gem worth 0.3 and seven steps from a big gem worth 1; either ends the episode, other steps pay nothing, and $\gam = 0.95$. With four workers, $n = 5$, $\alp^{\boldsymbol\theta} = 2$ and $\alp^{\mathbf w} = 0.3$, A2C without a bonus commits to the small gem in every one of 20 runs, within about 20 rounds; with $\beta = 0.1$, 18 of the 20 runs go for the big gem after 200 rounds, and the other two by round 240, after a long undecided phase in which the small gem's early lead is held back by the bonus.

### Historical remarks {#history}

Williams and Peng (1991) added an entropy term to REINFORCE to prevent its premature convergence, and showed it helped on function-optimization problems. Mnih and colleagues (2016) made the bonus a standard part of deep actor–critic methods, and Schulman and colleagues (2017) kept it in PPO. The maximum-entropy view, with entropy inside the return, underlies soft Q-learning and soft actor–critic (Haarnoja et al., 2018). Ahmed, Le Roux, Norouzi and Schuurmans (2019) studied why the bonus helps: it smooths the optimization landscape, connecting local optima that a plain policy gradient cannot cross.

## Card

### Idea

Add the entropy of the policy, its uncertainty, to what the learner maximizes. The bonus pushes each state's choices toward uniform, more for lopsided ones, so the policy keeps trying alternatives until one is clearly better.

::: analogy
A new city's restaurants. Eating every night at the first good one means never finding the great one around the corner; reserving a few nights for trying new places, fewer as you know the city better, is the bonus.
:::

### The bonus {#bonus}

$$\boldsymbol\theta \leftarrow \boldsymbol\theta + \alp \sum \Big[\err{\hat A_t}\,\nabla \ln \pol{\pi(A_t \mid S_t, \boldsymbol\theta)} + \beta\,\nabla \pol{H\big(\pi(\cdot \mid S_t, \boldsymbol\theta)\big)}\Big], \qquad \pol{H} = -\sum_a \pol{\pi(a \mid s)} \ln \pol{\pi(a \mid s)}$$

### Why it matters {#why}

- Without it, policy-gradient methods often commit to the first thing that works. On the two gems: 0 of 20 runs find the big gem without the bonus, 20 of 20 with $\beta = 0.1$. [See it](lab:entropy-gems)
- It is in almost every deep policy-gradient method: A2C, A3C, PPO, and, inside the reward, SAC.

### Knob {#knob}

| Knob | Too low | Too high |
| --- | --- | --- |
| $\beta$ entropy weight | commits to the first thing that works | stays too random to collect what it found |

### Pitfalls

- A $\beta$ that ignores the reward scale: the bonus is compared with the advantages, so rescaling rewards rescales the right $\beta$.
- Reading high entropy as progress: a policy can be uncertain because it has learned nothing.
- Expecting the bonus to find rewards far off the beaten path: it keeps choices spread, it does not direct [[exploration-strategies|exploration]].

### Check yourself {#check}

::: question
Why does a policy-gradient learner stop exploring on its own?
---
Its randomness is learned. Once one action is nearly certain, the others are rarely sampled, their advantages are never estimated, and the gradient that could revive them is nearly zero.
:::

::: question
Which way does the entropy bonus push the preference of an action with probability 0.7 among four actions?
---
Down: its probability is above $e^{-\pol{H}}$ (the policy is lopsided), so $-\pol{\pi}\,(\ln \pol{\pi} + \pol{H}) < 0$. The rarer actions' preferences go up.
:::

::: question
How does maximum-entropy RL differ from the entropy bonus?
---
It adds the entropy to every reward, so the agent values reaching states where it can stay uncertain, not just being uncertain where it is now.
:::
