+++
summary = "How much better than expected was each action? Generalized advantage estimation answers with a weighted sum of the TD errors that follow it, with weights that fade by γλ per step. λ = 0 trusts the critic after one step: steady, but wrong where the critic is wrong. λ = 1 trusts only the rewards: honest, but noisy. In between does best, and λ becomes the knob that trades bias for variance."
prereqs = ["actor-critic", "td-lambda", "pg-theorem"]
lab = "gae-cliff"
sources = [
  { text = "Schulman, Moritz, Levine, Jordan & Abbeel (2016), High-dimensional continuous control using generalized advantage estimation, ICLR", url = "https://arxiv.org/abs/1506.02438" },
  { text = "Kimura & Kobayashi (1998), An analysis of actor/critic algorithms using eligibility traces: reinforcement learning with imperfect value function, ICML" },
  { text = "Sutton (1988), Learning to predict by the methods of temporal differences, Machine Learning 3", url = "https://doi.org/10.1007/BF00115009" },
  { text = "Schulman, Wolski, Dhariwal, Radford & Klimov (2017), Proximal policy optimization algorithms", url = "https://arxiv.org/abs/1707.06347" },
]

[story]
scene = "grid"
env = "cliff"
seed = 1
average = 12
formula = '\step{1}{\err{\hat A_t} = \sum_{k \ge 0} (\gam\lam)^k\,\del_{t+k} \qquad} \step{2}{\del_t = \rew{R_{t+1}} + \gam\,\val{\hat v(S_{t+1})} - \val{\hat v(S_t)}}'

[story.runs]
lam9 = { algorithm = "trpo", lambda = 0.9, delta = 0.02, alphaW = 0.1, workers = 4, gamma = 1.0, maxSteps = 1000, units = 60, name = "λ = 0.9" }
lam0 = { algorithm = "trpo", lambda = 0.0, delta = 0.02, alphaW = 0.1, workers = 4, gamma = 1.0, maxSteps = 1000, units = 60, name = "λ = 0: one step, then the critic" }
lam1 = { algorithm = "trpo", lambda = 1.0, delta = 0.02, alphaW = 0.1, workers = 4, gamma = 1.0, maxSteps = 1000, units = 60, name = "λ = 1: the whole return" }
+++

## Story

::: step {run = "lam9", at = 0, formula = 1}
**The cliff**, and a batch learner: four workers play a round of episodes with the same policy, then the policy takes one carefully sized step (the method is TRPO, [[trpo]]). The step follows the **advantages**: for every move of the round, how much better or worse it did than the critic expected. This story is about how to estimate them.
:::

::: step {run = "lam9", at = 1, trail = false}
Round 1, with a policy that picks directions at random and a critic that knows nothing, 0 everywhere. One worker reaches the gem in 44 steps, one in 416; two wander for the full 1000 steps. 212 falls in all. Every move is now judged three ways, and the colors show the verdicts: each triangle is one move, orange for worse than expected, on a scale set by the worst.
:::

::: step {run = "lam0", at = 1, advantages = true, trail = false, arrows = false, formula = 2}
**λ = 0**: each move is judged by its own TD error, one reward and the critic's word about where it led. With a critic at 0, that is just the reward: $\rew{-1}$ for an ordinary move, $\rew{-100}$ for a fall. Only the moves into the cliff stand out. A move that wandered away from the gem and one that went toward it look exactly alike.
:::

::: step {run = "lam1", at = 1, advantages = true, trail = false, arrows = false}
**λ = 1**: each move is judged by everything that followed it, the whole rest of its episode. A move early in a 1000-step episode carries the blame for the hundreds of steps after it, whatever it did; moves near the end look good by comparison. Honest, but it says more about *when* a move happened than about what it did.
:::

::: step {run = "lam9", at = 1, advantages = true, trail = false, arrows = false}
**λ = 0.9**: each TD error is shared with the moves before it, fading by 0.9 a step. The $\rew{-100}$ of a fall now also lands on the ten or so moves that led up to it, but not on moves far away in time. The blame takes a shape: the edge of the cliff and the ways into it.
:::

::: step {run = "lam9", at = 60, values = true}
After 60 rounds with $\lam = 0.9$: 17 steps per episode, along the top row, as far from the edge as the grid allows, and a critic that expects exactly that: $\val{-17}$ from the start.
:::

::: step {run = "lam0", at = 60, values = true}
With $\lam = 0$, the same 60 rounds go nowhere. The policy climbs to the top row and then, from its second tile, bumps against the top wall for the rest of each episode, cut off at 1000 steps. Its critic still believes that tile is about 10 steps from the gem, $\val{-10.3}$: learning from one-step targets, it discovers the cost of the loop one step per round. And one-step advantages trust the critic completely.
:::

::: step {run = "lam9", at = 60, curves = ["lam0", "lam9", "lam1"], metric = "steps"}
Averaged over 12 runs: $\lam = 0.9$ ends at 16 steps per episode. $\lam = 1$ learns, but slowly and unevenly, still 99 on average after 60 rounds; $\lam = 0$ stays near 750, most of its runs stuck like the one above. [Try other λ in the Lab](lab:gae-cliff).
:::

## Textbook

### Which advantage? {#problem}

Every policy-gradient method weights the score $\nabla \ln \pol{\pi(A_t \mid S_t, \boldsymbol\theta)}$ by an estimate $\Psi_t$ of how good $A_t$ was ([[pg-theorem]]). The ideal weight is the advantage $\err{a_\pi(S_t, A_t)} = \val{q_\pi(S_t, A_t)} - \val{v_\pi(S_t)}$: it gives the right gradient with low variance. It is not known, and the estimates on offer sit at two extremes. The **return minus a baseline**, $\rew{G_t} - \val{\hat v(S_t)}$ ([[baseline]]), is unbiased whatever the critic, but carries the noise of every later reward. The **TD error** $\del_t = \rew{R_{t+1}} + \gam\,\val{\hat v(S_{t+1})} - \val{\hat v(S_t)}$ ([[actor-critic]]) carries the noise of one reward only, but is biased wherever $\val{\hat v}$ is wrong. Generalized advantage estimation (GAE; Schulman, Moritz, Levine, Jordan and Abbeel, 2016) fills in the range between them with one parameter.

### $k$-step advantages {#k-step}

Summing $k$ TD errors, discounted, telescopes into a $k$-step return minus the critic:

::: lemma {#lem-telescope} $k$-step advantages
$$\err{\hat A^{(k)}_t} = \sum_{l=0}^{k-1} \gam^l\,\del_{t+l} = \rew{R_{t+1}} + \gam\,\rew{R_{t+2}} + \dots + \gam^{k-1}\,\rew{R_{t+k}} + \gam^k\,\val{\hat v(S_{t+k})} - \val{\hat v(S_t)}. \label{eq-k}$$
:::

::: proof
In the sum, $\gam^l\,\del_{t+l} = \gam^l\,\rew{R_{t+l+1}} + \gam^{l+1}\,\val{\hat v(S_{t+l+1})} - \gam^l\,\val{\hat v(S_{t+l})}$; every value but the first and the last appears once with each sign.
:::

$\err{\hat A^{(1)}_t} = \del_t$ is the actor–critic's estimate, and as $k \to \infty$ (or the episode ends) the estimate becomes $\rew{G_t} - \val{\hat v(S_t)}$. The larger $k$, the less the estimate leans on the critic, and the more rewards it adds up: less bias, more variance. A2C's $n$-step advantages are of this kind ([[a2c]]).

### The generalized advantage estimator {#gae}

Rather than choosing one $k$, GAE averages all of them, with the weights of the λ-return ([[td-lambda]]), $(1 - \lam)\,\lam^{k-1}$:

$$\err{\hat A^{\text{GAE}(\gamma, \lambda)}_t} = (1 - \lam) \sum_{k=1}^{\infty} \lam^{k-1}\,\err{\hat A^{(k)}_t} = \sum_{l=0}^{\infty} (\gam\lam)^l\,\del_{t+l}. \label{eq-gae}$$

The second form, which follows by substituting \ref{eq-k} and summing the geometric series, is how it is computed: a sum of the TD errors after step $t$, each weighted by $(\gam\lam)^l$. The two ends are the familiar estimators:

$$\lam = 0:\ \err{\hat A_t} = \del_t, \qquad \lam = 1:\ \err{\hat A_t} = \sum_{l \ge 0} \gam^l\,\del_{t+l} = \rew{G_t} - \val{\hat v(S_t)}. \label{eq-ends}$$

::: figure {#fig-weights}
{{gae-weights}}
The weight of each TD error after step $t$ in its advantage, $(\gam\lam)^l$, with $\gam = 0.99$. Drag λ: at 0 only the step's own TD error counts; at 1 every later one does, fading only by $\gam$.
:::

### Two discounts, two meanings {#two}

$\gam$ and $\lam$ both shorten the horizon of the estimate, and both cut variance, but they do different things. The discount $\gam$ changes the problem: it defines the discounted return, and a $\gam < 1$ biases the estimate of the undiscounted objective even with a perfect critic. $\lam$ changes only the estimator: with the true values, $\mathbb E[\del_{t+l}] = 0$ for every later step given what came before, so $\err{\hat A^{\text{GAE}}_t}$ has the right expectation for every $\lam$. Bias from $\lam < 1$ appears only where the critic is wrong. That is why $\lam$ can usually be set lower than $\gam$: in Schulman and colleagues' experiments the best $\lam$ was a little below 1, with $\gam$ close to 1, and PPO's default is $\lam = 0.95$ ([[ppo]]).

### Computing it {#compute}

The sum \ref{eq-gae} is computed backward over a stretch of experience in one pass, like a return:

$$\err{\hat A_t} = \del_t + \gam\lam\,\err{\hat A_{t+1}}, \label{eq-backward}$$

starting from $\err{\hat A_T} = 0$ at the end of an episode. When a stretch is cut before the episode ends, as in batched methods that collect a fixed number of steps, the last TD error bootstraps from the critic's estimate of the state where the stretch stops, and the recursion starts there. The critic is usually trained toward the matching λ-returns, $\err{\hat A_t} + \val{\hat v(S_t)}$, as in the Lab, or toward plain returns. In a batch, the advantages are often **normalized**, shifted and scaled to mean 0 and standard deviation 1, so that the step size does not depend on the scale of the rewards ([[ppo]]).

::: algorithm {#alg-gae} GAE advantages for one stretch of experience
Input: states $S_t, \dots, S_{t+T}$, rewards $\rew{R_{t+1}}, \dots, \rew{R_{t+T}}$, a critic $\val{\hat v}$, $\gam$ and $\lam$
$\err{\hat A} \leftarrow 0$; $\val{\hat v_{\text{end}}} \leftarrow 0$ if $S_{t+T}$ is terminal, else $\val{\hat v(S_{t+T})}$
For $k = T - 1, T - 2, \dots, 0$:
  $\del \leftarrow \rew{R_{t+k+1}} + \gam\,\val{\hat v(S_{t+k+1})} - \val{\hat v(S_{t+k})}$ (with $\val{\hat v_{\text{end}}}$ for the last state)
  $\err{\hat A} \leftarrow \del + \gam\lam\,\err{\hat A}$; $\err{\hat A_{t+k}} \leftarrow \err{\hat A}$
:::

### How much λ? {#lambda}

The best $\lam$ depends on how noisy the rewards are and how good the critic is. On the cliff, a deterministic world where the critic starts out ignorant and episodes are long, TRPO with $\lam = 0.9$ ends at 16 steps per episode after 60 rounds, while $\lam = 0$ leaves most runs stuck in loops the wrong critic cannot see, and $\lam = 1$ is slowed by the noise of returns hundreds of steps long. On Frozen Lake, where the ice makes every return noisy, the best value is lower.

::: figure {#fig-study}
{{gae-study}}
PPO on Frozen Lake for seven values of λ: the exact value of the start under the learned policy, averaged over the first 100 rounds and 16 runs. The middle of the range, $\lam = 0.5$, does best (0.151); $\lam = 0$ (0.078) and especially $\lam = 1$ (0.025) learn less. Computed live by the Lab.
:::

### Historical remarks {#history}

Weighting TD errors by powers of $\lam$ is the idea of TD($\lam$) (Sutton, 1988), whose λ-return the estimator inherits. Kimura and Kobayashi (1998) used an advantage estimate of the same form in an actor–critic with eligibility traces. Schulman, Moritz, Levine, Jordan and Abbeel (2016) named it, analyzed $\lam$ as a bias–variance knob separate from $\gam$, and combined it with TRPO to learn locomotion with neural networks; with PPO (Schulman et al., 2017) it became the standard advantage estimate of deep policy-gradient methods.

## Card

### Idea

Estimate each action's advantage as a sum of the TD errors that follow it, weighted by $(\gam\lam)^k$. $\lam = 0$ is the one-step TD error (low variance, biased by the critic), $\lam = 1$ is the return minus the critic (unbiased, noisy). An intermediate $\lam$, often around 0.95, gets most of both.

::: analogy
Judging a chess move. You can trust your evaluation of the position right after it (quick, but only as good as your evaluation), or wait for the result of the game (honest, but every later blunder counts too). Strong players look a few moves ahead and trust their evaluation from there.
:::

### The estimator {#estimator}

$$\err{\hat A_t} = \sum_{k \ge 0} (\gam\lam)^k\,\del_{t+k}, \qquad \del_t = \rew{R_{t+1}} + \gam\,\val{\hat v(S_{t+1})} - \val{\hat v(S_t)}$$

computed backward: $\err{\hat A_t} = \del_t + \gam\lam\,\err{\hat A_{t+1}}$.

### The two ends {#ends}

| $\lam$ | estimate | bias | variance |
| --- | --- | --- | --- |
| 0 | $\del_t$, one step then the critic | where the critic is wrong | low |
| between | a fading sum of TD errors | some | some |
| 1 | $\rew{G_t} - \val{\hat v(S_t)}$, the whole return | none | high |

### Why it matters {#why}

- It is the advantage estimate of TRPO and PPO, and of most deep policy-gradient code.
- On the cliff, $\lam = 0.9$ learns in 60 rounds what $\lam = 0$ never learns and $\lam = 1$ learns slowly. [See it](lab:gae-cliff)

### Pitfalls

- Treating $\lam$ like $\gam$: lowering $\gam$ changes the goal; lowering $\lam$ only trusts the critic more.
- Forgetting to bootstrap at the end of a cut stretch, or bootstrapping past the end of an episode.
- A critic trained toward different targets than the advantages assume: harmless in theory, confusing when debugging.

### Check yourself {#check}

::: question
What is $\err{\hat A_t}$ with $\lam = 1$ and $\gam = 1$, written without TD errors?
---
$\rew{G_t} - \val{\hat v(S_t)}$: the TD errors telescope into the return minus the critic's estimate of the first state.
:::

::: question
With a perfect critic, is GAE biased for $\lam < 1$?
---
No: each later TD error then has expectation zero, so the advantage estimate is unbiased for every $\lam$. Bias appears only where the critic is wrong.
:::

::: question
On the cliff, why can $\lam = 0$ get stuck in a loop forever?
---
With $\lam = 0$ the advantage of each move is the TD error alone, which trusts the critic about the next state. Where the critic is wrong about the states of the loop, no reward ever enters the estimate to correct it.
:::
