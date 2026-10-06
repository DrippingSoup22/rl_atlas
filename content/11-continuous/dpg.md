+++
summary = "With continuous actions, a policy can name one action per state, a = μ(s), instead of a distribution over all of them. Its gradient is then the critic's slope with respect to the action, passed back through the actor: move the action the way the critic says the value rises. No sum over actions, no likelihood ratios, and it can learn off-policy; exploration has to be added from outside."
prereqs = ["pg-theorem", "actor-critic", "policy-parameterization"]
lab = "ddpg-pendulum"
sources = [
  { text = "Silver, Lever, Heess, Degris, Wierstra & Riedmiller (2014), Deterministic policy gradient algorithms, ICML", url = "https://proceedings.mlr.press/v32/silver14.html" },
  { text = "Degris, White & Sutton (2012), Off-policy actor-critic, ICML", url = "https://arxiv.org/abs/1205.4839" },
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., section 13.7", url = "http://incompleteideas.net/book/the-book-2nd.html" },
]
+++

## Textbook

### Continuous actions, again {#again}

Value-based methods choose and evaluate actions through a max over them ([[q-learning]]). With a torque, a steering angle or the joints of a robot arm, the actions form a continuum, and that max becomes an optimization problem at every step. Policy gradients avoid it with a parameterized distribution, often a Gaussian whose mean and spread the network outputs ([[policy-parameterization]]). They then estimate the gradient from sampled actions, and that estimate is noisy: the narrower the Gaussian becomes, the larger the variance of $\nabla \ln \pol\pi$.

There is a more direct route. Let the policy output the action itself, $a = \pol{\mu(s, \boldsymbol\theta)}$, and ask a critic $\val{\hat q(s, a)}$ how the value changes when that action is nudged.

### The deterministic policy gradient {#theorem}

For a deterministic policy, the performance gradient is (Silver et al., 2014):
$$\nabla_{\boldsymbol\theta} J(\boldsymbol\theta) = \mathbb E_{s \sim \rho^{\mu}}\Big[\nabla_{\boldsymbol\theta}\,\pol{\mu(s, \boldsymbol\theta)}\;\nabla_a \val{q_\mu(s, a)}\big|_{a = \pol{\mu(s)}}\Big], \label{dpg}$$
where $\rho^\mu$ is the discounted distribution of states the policy visits. Read from right to left, this is the chain rule. $\nabla_a \val{q_\mu}$ says in which direction a change of action raises the value, and how steeply; $\nabla_{\boldsymbol\theta}\pol\mu$ says how the parameters move the action. The update moves the parameters so that the action slides uphill on the critic.

Compare the stochastic policy gradient theorem ([[pg-theorem]]): there the expectation runs over states *and actions*, and each action's log-probability is pushed up or down by its value. Here the action integral is gone. Silver et al. also showed that the stochastic gradient of a Gaussian policy tends to this one as its spread shrinks to zero, so the deterministic gradient is the limit case, not a different idea.

### Off-policy for free {#off}

Because no expectation over the policy's actions remains, the gradient can be estimated from states visited by *another* policy, without importance sampling over actions. A behavior policy can explore with noise while the deterministic actor learns from what it saw, using the gradient \ref{dpg} with $s$ drawn from the behavior's state distribution instead of $\rho^\mu$. This is an approximation, made in the same spirit as the off-policy actor–critic of Degris, White and Sutton (2012), and it works well in practice. The critic learns $\val{q_\mu}$ off-policy as well, with Q-learning-style targets whose next action comes from the actor: $\rew{r} + \gam\,\val{\hat q(s', \pol{\mu(s')})}$.

That combination, a deterministic actor, a critic learned off-policy and a replay memory, is [[ddpg]].

### What the critic must get right {#critic}

The actor follows the critic's *slope*, $\nabla_a \val{\hat q}$, but the critic is trained only to match *values*. Nothing makes its slopes right, and where it errs, the actor moves the wrong way with confidence. If the critic overestimates some actions, the actor is drawn to exactly those: it climbs the errors. The fixes of [[td3]] (twin critics, delayed actor steps, smoothed targets) all target this weakness.

Silver et al. also gave a form of critic whose slopes are guaranteed to give the right gradient, a *compatible* critic linear in $\nabla_{\boldsymbol\theta}\pol\mu$, mirroring the stochastic case ([[actor-critic]]). Deep RL methods use general networks instead and accept the bias.

### Exploration from outside {#exploration}

A deterministic policy does the same thing in the same state every time, so it never explores by itself. The behavior policy adds noise to its action: independent Gaussian noise, or noise correlated over time (an Ornstein–Uhlenbeck process) so that a push lasts long enough to matter in a physical system. The amount of noise is a knob, set by hand ([[exploration-strategies]]). [[sac]] goes the other way: it keeps a stochastic policy and makes its randomness part of the objective.

### Historical remarks {#history}

Silver et al. (2014) proved the deterministic policy gradient theorem, its limit relation to the stochastic one, and compatible critics for it, and showed that off-policy deterministic actor–critics outperform their stochastic counterparts on problems with many action dimensions. Lillicrap et al. (2016) combined it with DQN's replay memory and target networks to learn from pixels and joint angles: DDPG.

## Card

### Idea

A deterministic policy outputs one action per state, $a = \pol{\mu(s)}$. Its gradient follows the critic's slope in action space: nudge the action the way $\val{\hat q}$ says the value rises, and carry that through the actor's parameters.

::: analogy
Adjusting a shower knob by feel. You do not try random positions and keep the good ones; you feel which way warmer lies and turn a little that way.
:::

### The gradient {#gradient}

$$\nabla_{\boldsymbol\theta} J = \mathbb E_s\Big[\nabla_{\boldsymbol\theta}\pol{\mu(s)}\;\nabla_a \val{\hat q(s, a)}\big|_{a = \pol{\mu(s)}}\Big]$$

the chain rule through the critic: how the action moves the value, times how the parameters move the action.

### Why it matters {#why}

- No expectation over actions: lower variance than a sampled policy gradient, and cheap with many action dimensions.
- Off-policy: states from any exploring behavior will do, so a replay memory works.
- The limit of a Gaussian policy gradient as its spread goes to zero.

### Pitfalls

- Trusting the critic's slope: it was trained on values, and the actor climbs its errors.
- Forgetting exploration: the actor alone never tries anything new.
- Expecting a deterministic policy to bluff or randomize when the best play is random, as in games.

### Check yourself {#check}

::: question
Why does the deterministic policy gradient not need importance sampling to learn off-policy?
---
It has no expectation over the policy's actions, only over states. The critic is queried at the actor's own action, μ(s), whatever action the behavior took there, so there is no action probability to correct for. (Using the behavior's states instead of the policy's is an approximation, but a mild one.)
:::

::: question
The critic overestimates the value of strong torques in some states. What does the actor do there?
---
It moves toward them: the actor follows the critic's slope, so wherever the critic wrongly rises, the actor follows. That is why critic errors matter more for deterministic actor–critics, and why TD3 makes the critic pessimistic.
:::

::: question
How is the deterministic policy gradient related to the Gaussian policy gradient of REINFORCE?
---
It is its limit: as the Gaussian's spread shrinks to zero, the stochastic policy gradient tends to the deterministic one (Silver et al., 2014). The sampled estimate becomes noisier as the spread shrinks, while the deterministic form computes the limit directly from the critic's slope.
:::
