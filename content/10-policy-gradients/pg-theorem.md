+++
summary = "The gradient of performance seems to need a model: changing the policy changes which states the agent visits, in ways only the world knows. The policy gradient theorem shows it does not. The gradient is an average, over the states and actions the policy itself produces, of how good each action is times how fast its log-probability grows. Experience from the policy is enough to estimate it."
story_in = ["reinforce", "baseline"]
prereqs = ["why-policy", "policy-parameterization", "value-functions", "bellman"]
lab = "corridor-reinforce"
sources = [
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., §13.2, §13.3 and §13.6", url = "http://incompleteideas.net/book/the-book-2nd.html" },
  { text = "Agarwal, Kakade, Lee & Mahajan (2021), On the theory of policy gradient methods: optimality, approximation, and distribution shift, Journal of Machine Learning Research 22", url = "https://jmlr.org/papers/v22/19-736.html" },
  { text = "Sutton, McAllester, Singh & Mansour (2000), Policy gradient methods for reinforcement learning with function approximation, Advances in Neural Information Processing Systems 12" },
  { text = "Williams (1992), Simple statistical gradient-following algorithms for connectionist reinforcement learning, Machine Learning 8", url = "https://doi.org/10.1007/BF00992696" },
  { text = "Glynn (1990), Likelihood ratio gradient estimation for stochastic systems, Communications of the ACM 33" },
  { text = "Marbach & Tsitsiklis (2001), Simulation-based optimization of Markov reward processes, IEEE Transactions on Automatic Control 46" },
  { text = "Konda & Tsitsiklis (2000), Actor-critic algorithms, Advances in Neural Information Processing Systems 12" },
  { text = "Baxter & Bartlett (2001), Infinite-horizon policy-gradient estimation, Journal of Artificial Intelligence Research 15" },
  { text = "Schulman, Moritz, Levine, Jordan & Abbeel (2016), High-dimensional continuous control using generalized advantage estimation, ICLR", url = "https://arxiv.org/abs/1506.02438" },
  { text = "Nota & Thomas (2020), Is the policy gradient a gradient?, AAMAS", url = "https://arxiv.org/abs/1906.07073" },
  { text = "OpenAI (2018), Spinning Up in Deep RL, Part 3: Intro to Policy Optimization", url = "https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html" },
]
+++

## Textbook

### A gradient through the world {#problem}

A policy-gradient method climbs a performance measure $J(\boldsymbol\theta)$ ([[why-policy]]). In an episodic task the natural measure is the value of the start state under the policy,

$$J(\boldsymbol\theta) = \val{v_{\pi_{\boldsymbol\theta}}(s_0)}, \label{eq-J}$$

and the method needs its gradient, or at least an unbiased estimate of it. Here is the difficulty. A change of $\boldsymbol\theta$ changes the action probabilities in every state, which is easy to differentiate, since the policy is ours ([[policy-parameterization]]). But it also changes *which states the agent visits*, and how often: that effect runs through the world's dynamics, which a model-free agent does not know. A naive gradient would need the derivative of the state distribution with respect to $\boldsymbol\theta$.

The policy gradient theorem removes the difficulty: the derivative of the state distribution does not appear.

### The theorem {#theorem}

Write $\eta(s)$ for the expected discounted number of visits to $s$ in an episode that starts in $s_0$ and follows $\pol{\pi}$,

$$\eta(s) = \sum_{t=0}^\infty \gam^t \Pr\{S_t = s \mid S_0 = s_0, \pol{\pi}\}. \label{eq-eta}$$

::: theorem {#thm-pg} Policy gradient theorem (episodic case)
For any policy $\pol{\pi(a \mid s, \boldsymbol\theta)}$ differentiable in $\boldsymbol\theta$, and a finite episodic MDP,

$$\nabla J(\boldsymbol\theta) = \sum_s \eta(s) \sum_a \val{q_\pi(s, a)}\,\nabla \pol{\pi(a \mid s, \boldsymbol\theta)}. \label{eq-pg}$$
:::

::: proof
The value of any state is an average of action values, $\val{v_\pi(s)} = \sum_a \pol{\pi(a \mid s)}\,\val{q_\pi(s, a)}$, so by the product rule

$$\nabla \val{v_\pi(s)} = \sum_a \Big[\nabla \pol{\pi(a \mid s)}\,\val{q_\pi(s, a)} + \pol{\pi(a \mid s)}\,\nabla \val{q_\pi(s, a)}\Big].$$

The action value is the expected reward plus the discounted value of the next state ([[bellman]]), and neither the rewards nor the transition probabilities depend on $\boldsymbol\theta$: $\nabla \val{q_\pi(s, a)} = \gam \sum_{s'} p(s' \mid s, a)\,\nabla \val{v_\pi(s')}$. Substituting gives a recursion for $\nabla \val{v_\pi}$,

$$\nabla \val{v_\pi(s)} = \sum_a \nabla \pol{\pi(a \mid s)}\,\val{q_\pi(s, a)} + \gam \sum_{s'} \Pr\{s \to s' \text{ in one step}\}\,\nabla \val{v_\pi(s')},$$

which, unrolled step after step from $s_0$, adds up the first term over every state $x$ the agent can reach, weighted by $\gam^t$ times the probability of being in $x$ at step $t$:

$$\nabla \val{v_\pi(s_0)} = \sum_x \sum_{t=0}^\infty \gam^t \Pr\{S_t = x \mid s_0, \pol{\pi}\} \sum_a \nabla \pol{\pi(a \mid x)}\,\val{q_\pi(x, a)}.$$

The inner sum over $t$ is $\eta(x)$. The unrolling ends because the task is episodic: the probability of still being in the episode after $t$ steps goes to 0.
:::

Normalizing $\eta$ into a distribution, $\mu(s) = \eta(s) / \sum_{s'} \eta(s')$, the share of the (discounted) time spent in $s$, gives the form usually quoted,

$$\nabla J(\boldsymbol\theta) \propto \sum_s \mu(s) \sum_a \val{q_\pi(s, a)}\,\nabla \pol{\pi(a \mid s, \boldsymbol\theta)}, \label{eq-prop}$$

where the constant is the expected (discounted) length of an episode. A constant factor does not change the direction, and any step size absorbs it. ($\mu$ here is the distribution of states, as in [[value-error]], not the mean of a Gaussian policy.)

In words: the gradient is a weighted sum, over states, of the directions that make each action more likely, each weighted by the action's value. States count as often as the policy visits them. The way the visits themselves would change has disappeared.

### Expectations that experience can sample {#sample}

The sum \ref{eq-pg} still runs over all states and all actions. To estimate it from experience, turn it into an expectation over what the policy does. Two identities do the work.

::: lemma {#lem-log} The log-derivative trick
Wherever $\pol{\pi(a \mid s, \boldsymbol\theta)} > 0$, $\nabla \pol{\pi(a \mid s, \boldsymbol\theta)} = \pol{\pi(a \mid s, \boldsymbol\theta)}\,\nabla \ln \pol{\pi(a \mid s, \boldsymbol\theta)}$. Consequently the score has mean zero under the policy:

$$\sum_a \pol{\pi(a \mid s, \boldsymbol\theta)}\,\nabla \ln \pol{\pi(a \mid s, \boldsymbol\theta)} = \nabla \sum_a \pol{\pi(a \mid s, \boldsymbol\theta)} = \nabla 1 = \mathbf 0. \label{eq-zero}$$
:::

With the first identity, the inner sum of \ref{eq-pg} becomes an average over the action the policy takes, and the outer sum, weighted by discounted visits, an average over the steps of an episode:

$$\nabla J(\boldsymbol\theta) = \mathbb E_\pi\Big[\sum_{t=0}^{T-1} \gam^t\,\val{q_\pi(S_t, A_t)}\,\nabla \ln \pol{\pi(A_t \mid S_t, \boldsymbol\theta)}\Big] = \mathbb E_\pi\Big[\sum_{t=0}^{T-1} \gam^t\,\rew{G_t}\,\nabla \ln \pol{\pi(A_t \mid S_t, \boldsymbol\theta)}\Big], \label{eq-reinforce}$$

the second form because the return $\rew{G_t}$ has expectation $\val{q_\pi(S_t, A_t)}$ given $S_t$ and $A_t$. Everything inside the last expectation is observed in one episode, so a single episode gives an unbiased estimate of the gradient: that is [[reinforce|REINFORCE]].

The same result can be reached from whole trajectories. The probability of an episode $\tau = (S_0, A_0, \rew{R_1}, S_1, \dots)$ is a product of the policy's choices and the world's transitions; in its logarithm the two separate, and the world's terms do not depend on $\boldsymbol\theta$:

$$\nabla \ln \Pr\{\tau \mid \boldsymbol\theta\} = \sum_t \nabla \ln \pol{\pi(A_t \mid S_t, \boldsymbol\theta)}. \label{eq-trajectory}$$

So $\nabla J = \mathbb E[\rew{G_0} \sum_t \nabla \ln \pol{\pi(A_t \mid S_t)}]$. This is where the model drops out: the gradient of the log-probability of a whole episode needs only the policy. Then \ref{eq-zero} trims it: a reward received *before* step $t$ is already fixed when $A_t$ is drawn, and the score of $A_t$ averages to zero against it, so only the rewards from $t$ on, $\gam^t \rew{G_t}$, stay with each step's score. Dropping the past keeps the estimate unbiased and usually reduces its variance.

### Baselines, advantages and the family of estimators {#forms}

Identity \ref{eq-zero} allows more. Any function of the state, $b(s)$, can be subtracted from the action values without changing the gradient, because $\sum_a b(s)\,\nabla \pol{\pi(a \mid s)} = b(s)\,\nabla 1 = \mathbf 0$:

$$\nabla J(\boldsymbol\theta) \propto \sum_s \mu(s) \sum_a \big(\val{q_\pi(s, a)} - b(s)\big)\,\nabla \pol{\pi(a \mid s, \boldsymbol\theta)}. \label{eq-baseline}$$

The baseline changes nothing on average but can change the variance of the estimate a lot ([[baseline]]). With $b = \val{v_\pi}$, the weight becomes the **advantage**, how much better the action is than the policy's average in that state:

$$\err{a_\pi(s, a)} = \val{q_\pi(s, a)} - \val{v_\pi(s)}. \label{eq-advantage}$$

Most policy-gradient methods are instances of one template (Schulman et al., 2016): an estimate

$$\widehat{\nabla J} = \sum_t \Psi_t\,\nabla \ln \pol{\pi(A_t \mid S_t, \boldsymbol\theta)}, \label{eq-psi}$$

where the weight $\Psi_t$ is one of the return of the whole episode, the return from step $t$ ($\rew{G_t}$), the return minus a baseline ($\rew{G_t} - b(S_t)$), the action value $\val{q_\pi(S_t, A_t)}$, the advantage $\err{a_\pi(S_t, A_t)}$, or the TD error $\del_t = \rew{R_{t+1}} + \gam\,\val{v_\pi(S_{t+1})} - \val{v_\pi(S_t)}$, whose expectation given $S_t$ and $A_t$ is the advantage. With the true values all of these give the gradient on average; they differ in variance. With *learned* values, the last ones trade some bias for much less variance: the methods from [[baseline]] to [[ppo]] are, in large part, different choices of $\Psi_t$ ([[actor-critic]], [[gae]]).

### Checking it on the short corridor {#corridor}

The short corridor ([[why-policy]], Example 13.1 of Sutton and Barto) is small enough to check the theorem by hand. Take the policy that steps right with probability $p = 0.3$ in every cell, from a softmax of two preferences, so that raising the preference for right by $\mathrm d h$ raises $p$ by $p(1 - p)\,\mathrm d h$. Differentiating $J(p) = -2(2 - p)/(p(1 - p))$ gives the exact slope along that preference,

$$\frac{\partial J}{\partial h_{\text{right}}} = \frac{\mathrm d J}{\mathrm d p}\,p(1 - p) = \frac{2\,(p^2 - 4p + 2)}{p\,(1 - p)} = 8.48 \text{ at } p = 0.3. \label{eq-slope}$$

The theorem computes the same number from the visits and the action values, without differentiating $J$. An episode visits the three cells 8.10, 4.76 and 3.33 times on average (16.19 steps, which is $-J$). Since $\partial \pol{\pi(\text{right})}/\partial h_{\text{right}} = p(1-p)$ and $\partial \pol{\pi(\text{left})}/\partial h_{\text{right}} = -p(1-p)$, each cell contributes its visits times $p(1 - p)\,\big(\val{q(s, \text{right})} - \val{q(s, \text{left})}\big)$:

| cell | visits $\eta(s)$ | $\val{q(s, \text{right})} - \val{q(s, \text{left})}$ | visits × difference |
| --- | --- | --- | --- |
| 1 | 8.10 | $-13.86 - (-17.19) = 3.33$ | 26.98 |
| 2 (swapped) | 4.76 | $-17.19 - (-11.00) = -6.19$ | $-29.48$ |
| 3 | 3.33 | $-1.00 - (-13.86) = 12.86$ | 42.86 |
| total | 16.19 | | 40.36 |

Their total times $p(1 - p) = 0.21$ is $0.21 \times 40.36 = 8.48$, the slope of \ref{eq-slope}. In the swapped cell, stepping right is worse than stepping left, and that cell alone pulls the other way; cells 1 and 3 win, so the gradient says: step right more often.

::: figure {#fig-estimates}
{{pg-estimates}}
Single-episode estimates of the same slope, $\sum_t \rew{G_t}\,\partial \ln \pol{\pi(A_t)}/\partial h_{\text{right}}$, from 50,000 episodes at $p = 0.3$. Without a baseline (top) they average 8.4, close to the true 8.48, but spread 38.8 on either side, and 42% of them point downhill. Subtracting a baseline (bottom) keeps the average and cuts the spread to 24.4 ([[baseline]]).
:::

The estimates are right on average and wrong one at a time. That is the price of not knowing the model, and the reason why every method after [[reinforce]] works to reduce their variance.

### Local or global? {#landscape}

$J$ is not concave in $\boldsymbol\theta$, so a gradient method could in principle stop on a hill that is not the highest. For a softmax policy with one preference per state and action, though, there are no such false summits: with exact gradients and a start distribution that gives every state some weight, gradient ascent converges to an optimal policy (Agarwal, Kakade, Lee & Mahajan, 2021). The reason is the performance difference lemma ([[policy-improvement]]): any policy that is not optimal has some state where some action has a positive advantage, and with full coverage the gradient feels it. The catch is speed. Where the policy has become nearly deterministic the gradient is tiny, and plateaus can be long. With function approximation, or with states the start distribution never reaches, true local optima return.

### The continuing case {#continuing}

For tasks without episodes, performance is the **average reward** per step, $r(\pi) = \lim_{h \to \infty} \frac1h \sum_{t=1}^h \mathbb E[\rew{R_t}]$, and values are differential: how much more reward than average follows a state or an action ([[semi-gradient-sarsa]]). The theorem then holds with equality, with the stationary distribution of the policy as weights:

$$\nabla r(\pi) = \sum_s \mu(s) \sum_a \val{q_\pi(s, a)}\,\nabla \pol{\pi(a \mid s, \boldsymbol\theta)}, \label{eq-average}$$

where $\val{q_\pi}$ is now the differential action value. The proof follows the same steps, using the stationarity of $\mu$ in place of the end of episodes.

### Discounting, in theory and in practice {#discount}

The factor $\gam^t$ in \ref{eq-reinforce} is part of the gradient of $J$: later steps matter less to the value of the start. Most implementations drop it and weight every step of an episode alike, keeping $\gam$ only inside the returns. The resulting update is not the gradient of $J$, nor, in general, of any function (Nota and Thomas, 2020); it is closer to the gradient of the undiscounted performance, with $\gam$ acting as a knob that cuts variance by discounting distant rewards ([[gae]]). It works well in practice. The Lab follows each method's usual form: REINFORCE and the actor–critic keep $\gam^t$, as Sutton and Barto write them; A2C, TRPO and PPO drop it, as their authors' code does.

### Compatible function approximation {#compatible}

::: remark
Sutton, McAllester, Singh and Mansour (2000) also asked when a *learned* $\val{\hat q_{\mathbf w}}$ can replace $\val{q_\pi}$ in \ref{eq-pg} without biasing the gradient. It can if the approximation is linear in the score, $\val{\hat q_{\mathbf w}(s, a)} = \mathbf w^\top \nabla \ln \pol{\pi(a \mid s, \boldsymbol\theta)}$, and $\mathbf w$ minimizes the squared error to $\val{q_\pi}$ under the policy's distribution of states and actions. Such features are called compatible; the best $\mathbf w$ then turns out to be the natural gradient ([[trpo]]). The conditions are rarely met in practice, but the result explains why a critic of advantages, which average to zero in each state like the score itself, fits naturally into a policy gradient.
:::

### Historical remarks {#history}

Estimating the gradient of an expectation through the derivative of a log-probability is the likelihood-ratio method of simulation (Glynn, 1990); Williams (1992) brought it to reinforcement learning as REINFORCE. The theorem with function approximation is due to Sutton, McAllester, Singh and Mansour (2000) and, independently and for the average reward, to Marbach and Tsitsiklis (2001); Konda and Tsitsiklis (2000) built their analysis of actor–critic methods on it, and Baxter and Bartlett (2001) gave a related estimator for infinite horizons. The general template \ref{eq-psi} is from Schulman, Moritz, Levine, Jordan and Abbeel (2016). The corridor check is ours, on Sutton and Barto's Example 13.1.

## Card

### Idea

The gradient of a policy's performance needs no model. It is an average, over the states and actions the policy itself produces, of each action's value times the gradient of its log-probability. So an episode played with the policy gives an unbiased, if noisy, estimate of the uphill direction.

::: analogy
To find out whether to serve more often to the left, you do not need a theory of your opponent: play, note how each point went, and push the choice you made in each point up or down by how the point turned out. Over many points, the pushes add up to the right direction.
:::

### The theorem {#statement}

$$\nabla J(\boldsymbol\theta) \propto \sum_s \mu(s) \sum_a \val{q_\pi(s, a)}\,\nabla \pol{\pi(a \mid s, \boldsymbol\theta)}$$

or, as an average over the policy's own experience,

$$\nabla J(\boldsymbol\theta) \propto \mathbb E_\pi\big[\val{q_\pi(S, A)}\,\nabla \ln \pol{\pi(A \mid S, \boldsymbol\theta)}\big].$$

The state distribution $\mu$ is sampled by following the policy; its gradient never appears.

### One template {#template}

$\widehat{\nabla J} = \sum_t \Psi_t\,\nabla \ln \pol{\pi(A_t \mid S_t, \boldsymbol\theta)}$, with $\Psi_t$:

| $\Psi_t$ | used by | variance |
| --- | --- | --- |
| $\rew{G_t}$, the return from $t$ | [[reinforce]] | high |
| $\rew{G_t} - \val{\hat v(S_t)}$ | [[baseline]] | lower |
| $\del_t = \rew{R_{t+1}} + \gam\,\val{\hat v(S_{t+1})} - \val{\hat v(S_t)}$ | [[actor-critic]] | low, some bias |
| $n$-step or GAE advantage $\err{\hat A_t}$ | [[a2c]], [[trpo]], [[ppo]] | tunable ([[gae]]) |

### Why it matters {#why}

- It is the foundation of every policy-gradient method: they differ in how they estimate $\Psi_t$ and how far they step.
- It shows why a baseline is free: the score averages to zero, so subtracting any $b(s)$ changes only the variance.
- On the short corridor the single-episode estimates average to the true slope, but 42% of them point downhill. [See them add up](lab:corridor-reinforce)

### Pitfalls

- Expecting one episode to point uphill: the estimate is right only on average.
- Thinking the gradient needs the derivative of the state distribution: the theorem's whole point is that it does not.
- Weighting a step's score by rewards received before it: unbiased, but needless noise.
- Calling the usual implementation, which drops $\gam^t$, the gradient of the discounted value: it is not.

### Check yourself {#check}

::: question
Why does the transition model disappear from $\nabla \ln \Pr\{\tau \mid \boldsymbol\theta\}$?
---
The probability of an episode is a product of policy terms and transition terms. Its logarithm is a sum, and the transition terms do not depend on $\boldsymbol\theta$, so their gradient is zero.
:::

::: question
Why can any baseline $b(s)$ be subtracted from the returns without bias?
---
Because $\sum_a \pol{\pi(a \mid s)}\,\nabla \ln \pol{\pi(a \mid s)} = \nabla \sum_a \pol{\pi(a \mid s)} = \nabla 1 = \mathbf 0$: the score averages to zero in every state, so $b(s)$ times it does too.
:::

::: question
On the short corridor at $p = 0.3$, stepping right is the worse action in the swapped cell. Why does the gradient still say “step right more often”?
---
The gradient adds up all cells, each weighted by its visits. The swapped cell's pull ($-29.5$) is outweighed by cells 1 and 3 ($27.0 + 42.9$), so the total is positive.
:::
