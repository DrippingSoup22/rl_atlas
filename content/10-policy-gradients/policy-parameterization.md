+++
summary = "A policy with weights must turn numbers into a way of choosing, and be differentiable in them. For a few discrete actions: a softmax over learned preferences. For a real-valued action, like an angle: a bell curve whose center and width are learned. Both come with a simple formula for the gradient of the log-probability, which is all a policy-gradient method needs."
prereqs = ["why-policy", "gradient-bandit", "features"]
lab = "throw"
sources = [
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., §13.1, §13.7 and Exercises 13.3 and 13.4", url = "http://incompleteideas.net/book/the-book-2nd.html" },
  { text = "Chou, Maturana & Scherer (2017), Improving stochastic policy gradients in continuous control with deep reinforcement learning using the Beta distribution, Proceedings of the 34th International Conference on Machine Learning", url = "https://proceedings.mlr.press/v70/chou17a.html" },
  { text = "Williams (1992), Simple statistical gradient-following algorithms for connectionist reinforcement learning, Machine Learning 8", url = "https://doi.org/10.1007/BF00992696" },
  { text = "Bridle (1990), Probabilistic interpretation of feedforward classification network outputs, with relationships to statistical pattern recognition, in Neurocomputing: Algorithms, Architectures and Applications, Springer" },
  { text = "Gullapalli (1990), A stochastic reinforcement learning algorithm for learning real-valued functions, Neural Networks 3" },
  { text = "OpenAI (2018), Spinning Up in Deep RL, Part 1: Key Concepts in RL (stochastic policies)", url = "https://spinningup.openai.com/en/latest/spinningup/rl_intro.html" },
  { text = "Haarnoja, Zhou, Abbeel & Levine (2018), Soft actor-critic: off-policy maximum entropy deep reinforcement learning with a stochastic actor, ICML", url = "https://arxiv.org/abs/1801.01290" },
]

[story]
scene = "throw"
env = "throw"
seed = 1
average = 100
formula = '\step{1}{\pol{\pi(a)} = \frac{1}{\pol{\sigma}\sqrt{2\pi}}\,\exp\Big(-\frac{(a - \pol{\mu})^2}{2\pol{\sigma}^2}\Big) \qquad} \step{2}{\frac{\partial \ln \pol{\pi(a)}}{\partial \pol{\mu}} = \frac{a - \pol{\mu}}{\pol{\sigma}^2} \qquad} \step{3}{\frac{\partial \ln \pol{\pi(a)}}{\partial \ln \pol{\sigma}} = \frac{(a - \pol{\mu})^2}{\pol{\sigma}^2} - 1}'

[story.runs]
learn = { algorithm = "baseline", alpha = 0.003, alphaW = 0.1, gamma = 1.0, features = "own", units = 1000, measures = ["aim"], name = "a Gaussian policy" }
+++

## Story

::: step {run = "learn", at = 0, formula = 1}
**A throw.** One decision per episode: the angle at which to throw a ball. The ball flies $40 \sin 2a$ meters, 40 m at best, at 45° (the gold curve, which the learner never sees), and the wind adds or takes away a couple of meters. The reward is the distance. An angle is a real number, so there is no list of actions to score one by one. Instead the policy is a **bell curve** over angles, a normal distribution with an aim $\pol{\mu}$ and a spread $\pol{\sigma}$, here 20° and 10°. Each throw is drawn from it, and learning means moving the bell.
:::

::: step {run = "learn", at = 0, play = 1, pace = 400}
The first throw is drawn at 34.0°, more than one spread above the aim. It flies 39.5 m.
:::

::: step {run = "learn", at = 1, formula = 2}
Was that better than expected? The learner keeps a running average of how far its throws fly, the **baseline** (the dashed line). Before this throw it expected nothing, so 39.5 m is a pleasant surprise. A good surprise pulls the aim **toward** the angle thrown, in proportion to $(a - \pol{\mu})/\pol{\sigma}^2$, the slope of $\ln \pol{\pi}$ at that angle: the aim moves from 20.0° to 21.7°.
:::

::: step {run = "learn", at = 1, formula = 3}
The spread learns the same way. For a throw more than one spread from the aim, $(a - \pol{\mu})^2/\pol{\sigma}^2 - 1$ is positive: after a good surprise the bell **widens**, to make such throws more likely. A good throw within one spread would narrow it. This one lay 1.4 spreads away, and the spread grows from 10.0° to 11.2°.
:::

::: step {run = "learn", at = 1, play = 7, pace = 160}
The next throws. The baseline is still far behind, so almost every throw counts as good, even the second, thrown low at 15.9°: the aim follows it down a little, to 21.4°, and the bell narrows, because that throw fell within one spread. But throws above the aim fly farther and pull harder, so on balance the aim climbs. The eighth, 9.7° for 14.5 m, is the first worse than expected: the aim moves away from it.
:::

::: step {run = "learn", at = 50}
After 50 throws the aim is 32.4°, the spread 11.8°, and the baseline, at 34.4 m, has caught up with what the throws achieve. From now on only throws that beat the average pull the aim toward them, and those are mostly the high ones.
:::

::: step {run = "learn", at = 200, play = 3, pace = 160}
After 200 throws the aim is 44.1° and the spread has shrunk to 5.1°. Near the top of the curve, throws far from the aim on either side do worse than average and push the bell narrower; throws close to it do better and narrow it too. The policy is growing sure.
:::

::: step {run = "learn", at = 1000}
After 1000 throws: aim 46.3°, spread 2.1°, and 97% of throws fly 39.3 m or more before the wind. The aim is not exactly 45°, and it never quite settles: this close to the top, a degree or two either way costs centimeters, and the wind's couple of meters drown that out.
:::

::: step {run = "learn", at = 1000, curves = ["learn"], metric = "aim"}
Averaged over 100 runs, the aim passes 38° after 100 throws and settles around 45° after 500. The same recipe works for several real-valued actions at once, one bell for each, and for a handful of discrete actions with a softmax in place of the bell. [Throw in the Lab](lab:throw), with and without the baseline.
:::

## Textbook

### What a parameterized policy needs {#needs}

A policy-gradient method ([[why-policy]]) adjusts the weights $\boldsymbol\theta$ of a policy $\pol{\pi(a \mid s, \boldsymbol\theta)}$ along an estimate of the gradient of performance. Every estimate in this part has the same ingredient: the gradient of the **log-probability** of the action taken, $\nabla \ln \pol{\pi(A_t \mid S_t, \boldsymbol\theta)}$, called the *score function* or, in Sutton and Barto's terms, the *eligibility vector* ([[pg-theorem]]). So the parameterization must

1. give a proper distribution over actions in every state, for every $\boldsymbol\theta$;
2. be differentiable in $\boldsymbol\theta$, with a score function that is cheap to compute;
3. be easy to sample from;
4. keep every action possible, $\pol{\pi(a \mid s, \boldsymbol\theta)} > 0$, so that the policy explores, while being able to approach a deterministic policy when one is best.

Two families cover most uses: the **softmax** for a finite set of actions, and the **Gaussian** for real-valued ones. Others appear in specific settings (\ref{other}).

### Softmax over preferences {#softmax}

For a finite set of actions, give each state–action pair a numerical **preference** $\pol{h(s, a, \boldsymbol\theta)}$ and turn preferences into probabilities with the softmax, the exponential of each divided by their sum:

$$\pol{\pi(a \mid s, \boldsymbol\theta)} = \frac{e^{\pol{h(s, a, \boldsymbol\theta)}}}{\sum_b e^{\pol{h(s, b, \boldsymbol\theta)}}}. \label{eq-softmax}$$

The probabilities are positive and sum to 1 whatever the preferences; adding the same constant to every preference of a state changes nothing; and a preference far above the others makes its action nearly certain, without ever reaching probability 1. The preferences can be any differentiable function of $\boldsymbol\theta$: a neural network's outputs, or, as in this part, linear in features,

$$\pol{h(s, a, \boldsymbol\theta)} = \boldsymbol\theta^\top \mathbf x(s, a). \label{eq-linear}$$

The Lab gives each action its own copy of the state's features, $\mathbf x(s, a)$ equal to $\mathbf x(s)$ in $a$'s block of weights and 0 elsewhere, the construction that semi-gradient SARSA uses for action values ([[semi-gradient-sarsa]]). With one-hot features of the state this is a table of preferences, one per state and action: the [[gradient-bandit]] algorithm's preferences, made state-dependent.

::: lemma {#lem-softmax} The softmax score function
For the softmax of linear preferences \ref{eq-softmax}–\ref{eq-linear},

$$\nabla \ln \pol{\pi(a \mid s, \boldsymbol\theta)} = \mathbf x(s, a) - \sum_b \pol{\pi(b \mid s, \boldsymbol\theta)}\,\mathbf x(s, b). \label{eq-score-softmax}$$

With one copy of $\mathbf x(s)$ per action, the block of action $a$ gets $\big(1 - \pol{\pi(a \mid s)}\big)\,\mathbf x(s)$ and the block of every other action $b$ gets $-\pol{\pi(b \mid s)}\,\mathbf x(s)$.
:::

::: proof
$\ln \pol{\pi(a \mid s, \boldsymbol\theta)} = \boldsymbol\theta^\top \mathbf x(s, a) - \ln \sum_b e^{\boldsymbol\theta^\top \mathbf x(s, b)}$. The gradient of the first term is $\mathbf x(s, a)$; that of the second is $\sum_b e^{\boldsymbol\theta^\top \mathbf x(s, b)}\,\mathbf x(s, b) \big/ \sum_c e^{\boldsymbol\theta^\top \mathbf x(s, c)} = \sum_b \pol{\pi(b \mid s, \boldsymbol\theta)}\,\mathbf x(s, b)$. With per-action copies, $\mathbf x(s, b)$ is $\mathbf x(s)$ placed in block $b$, which gives the second statement.
:::

The score is the features of the action taken minus their average under the policy: a step along it makes the action taken more likely and every other action less likely, each in proportion to its current probability. Its size is at most twice that of the largest feature vector, whatever the probabilities, so softmax updates are well behaved.

::: figure {#fig-softmax}
{{softmax-play}}
Three preferences and the probabilities their softmax gives. Adding the same amount to every preference changes nothing; doubling them all sharpens the policy toward its favorite without making it certain.
:::

Preferences are not action values. A softmax over *estimated values*, $e^{\val{\hat q(s, a)}/\tau}$, is a way to explore (Boltzmann exploration): as the estimates converge, it converges to a fixed stochastic policy set by the temperature $\tau$, never to the best deterministic one unless $\tau$ is lowered on a schedule, and never to the best stochastic one except by accident. Learned preferences have no target values. They move wherever the gradient of performance sends them: toward a probability of 59% on the short corridor ([[why-policy]]), or apart without bound when one action is best, so that the policy becomes as close to deterministic as the problem rewards.

### Gaussian policies {#gaussian}

When the action is a real number (an angle, a torque, a price), a policy can be a **normal distribution** whose mean and standard deviation depend on the state:

$$\pol{\pi(a \mid s, \boldsymbol\theta)} = \frac{1}{\pol{\sigma(s, \boldsymbol\theta)}\sqrt{2\pi}}\,\exp\!\Big(-\frac{\big(a - \pol{\mu(s, \boldsymbol\theta)}\big)^2}{2\,\pol{\sigma(s, \boldsymbol\theta)}^2}\Big), \label{eq-gauss}$$

a density rather than a probability, which changes nothing in what follows. The weights split in two, $\boldsymbol\theta = (\boldsymbol\theta_\mu, \boldsymbol\theta_\sigma)$: the mean is linear in features, and the standard deviation is the exponential of a linear function, which keeps it positive whatever the weights:

$$\pol{\mu(s, \boldsymbol\theta)} = \boldsymbol\theta_\mu^\top \mathbf x_\mu(s), \qquad \pol{\sigma(s, \boldsymbol\theta)} = \exp\big(\boldsymbol\theta_\sigma^\top \mathbf x_\sigma(s)\big). \label{eq-musigma}$$

The mean is where the policy aims; the standard deviation is how much it explores, and it is learned too. It can shrink toward 0 as the policy grows sure, and the policy then approaches the deterministic one that always takes $\pol{\mu(s)}$.

::: lemma {#lem-gauss} The Gaussian score function
For the Gaussian policy \ref{eq-gauss}–\ref{eq-musigma},

$$\nabla_{\boldsymbol\theta_\mu} \ln \pol{\pi(a \mid s, \boldsymbol\theta)} = \frac{a - \pol{\mu(s, \boldsymbol\theta)}}{\pol{\sigma(s, \boldsymbol\theta)}^2}\,\mathbf x_\mu(s), \qquad \nabla_{\boldsymbol\theta_\sigma} \ln \pol{\pi(a \mid s, \boldsymbol\theta)} = \bigg(\frac{\big(a - \pol{\mu(s, \boldsymbol\theta)}\big)^2}{\pol{\sigma(s, \boldsymbol\theta)}^2} - 1\bigg)\,\mathbf x_\sigma(s). \label{eq-score-gauss}$$
:::

::: proof
$\ln \pol{\pi} = -\ln \pol{\sigma} - \tfrac12 \ln 2\pi - (a - \pol{\mu})^2 / (2\pol{\sigma}^2)$. Its derivative with respect to $\pol{\mu}$ is $(a - \pol{\mu})/\pol{\sigma}^2$, and $\nabla_{\boldsymbol\theta_\mu}\pol{\mu} = \mathbf x_\mu(s)$. Its derivative with respect to $\ln \pol{\sigma}$, using $\pol{\sigma}^{-2} = e^{-2 \ln \pol{\sigma}}$, is $-1 + (a - \pol{\mu})^2/\pol{\sigma}^2$, and $\nabla_{\boldsymbol\theta_\sigma} \ln \pol{\sigma} = \mathbf x_\sigma(s)$.
:::

Read as instructions, the two parts say: an action that turned out better than expected pulls the mean toward itself, harder the farther it lies in units of $\pol{\sigma}^2$; and it widens the spread if it lay more than one standard deviation from the mean, narrows it otherwise. An action that turned out worse does the opposite.

::: figure {#fig-gauss}
{{gaussian-play}}
A Gaussian policy over angles and the two parts of $\nabla \ln \pol{\pi(a)}$ for every action $a$, each drawn to its own scale. The push on the mean changes sign at $\pol{\mu}$; the push on the spread changes sign at $\pol{\mu} \pm \pol{\sigma}$ (the shaded band).
:::

### Scale and stability {#scale}

The Gaussian score has a property the softmax lacks: its size depends on $\pol{\sigma}$. A typical action lies about $\pol{\sigma}$ from the mean, so the push on the mean is about $1/\pol{\sigma}$, and it grows as the policy grows sure. A narrow policy that meets an unusual action with a large surprise can then take a very long step. Three things keep this in hand in practice:

- **Units.** The mean and the spread are measured in a unit chosen for the action, so that one step size suits both; the Lab's throw measures angles in tens of degrees. With deep networks, actions are usually rescaled to $[-1, 1]$.
- **The spread's parameterization.** Learning $\ln \pol{\sigma}$ rather than $\pol{\sigma}$ keeps it positive and makes its updates multiplicative. Deep policy-gradient code often makes $\ln \pol{\sigma}$ a free parameter per action dimension, independent of the state, and sometimes bounds it from below.
- **Scale-aware steps.** The natural gradient ([[trpo]]) measures steps by how much they change the distribution rather than the weights. For the mean of a Gaussian it multiplies the plain gradient by $\pol{\sigma}^2$, which removes exactly the $1/\pol{\sigma}^2$ above.

The softmax has its own failure of scale, a slower one: as one preference runs ahead, the policy's probabilities stop changing visibly while its weights keep growing, and exploration dies out ([[entropy-bonus]]).

### Several actions, bounded actions {#bounds}

With several real-valued actions, the usual policy is a **diagonal Gaussian**: one mean and one standard deviation per action dimension, the dimensions drawn independently, so the log-probability is the sum of the dimensions' log-probabilities and so is its gradient. Correlated actions need a full covariance matrix, which is rarely worth its cost.

Real actions are usually bounded (a steering wheel turns so far), and a Gaussian is not. Three common answers:

1. **Clip** the sampled action at the bounds, and keep the score of the unclipped sample. The throw does this: its angles are clipped to 0°–90°. It is simple, but piles probability on the bounds and makes the gradient blind to them.
2. **Squash** the sample through a bounded function, such as $\tanh$, and correct the log-probability for the change of variable, as [[sac|soft actor–critic]] does.
3. **Use a bounded distribution**, such as a Beta distribution rescaled to the action range. Chou, Maturana and Scherer (2017) found it learned faster than a clipped Gaussian, whose gradient is biased near the bounds.

### Other parameterizations {#other}

A continuous action range can be cut into bins and treated with a softmax, at the price of resolution and of ignoring that neighboring bins are alike. A **deterministic** policy $a = \pol{\mu(s, \boldsymbol\theta)}$ has no log-probability at all and needs a gradient of its own, through a learned action-value function ([[dpg]], [[ddpg]]). Mixtures, flows and other richer distributions appear in later work; the principle is always the same: a distribution that can be sampled and whose log-probability can be differentiated.

### Example: the throw {#example}

::: example {#ex-throw} The throw
One state and one real-valued action, the angle $a$ of a throw, clipped to $[0°, 90°]$. The ball flies $40 \sin 2a$ meters, plus wind, a normal amount with standard deviation 2 m, and the distance is the reward; the episode ends there. The policy is a Gaussian with constant features, $\pol{\mu} = \theta_\mu$ and $\pol{\sigma} = e^{\theta_\sigma}$, measured in tens of degrees; it starts at $\pol{\mu} = 20°$, $\pol{\sigma} = 10°$. The best policy throws at exactly 45° with no spread, for 40 m on average.
:::

The Story's learner updates both weights after every throw with the score \ref{eq-score-gauss} of the angle thrown, times the difference between the distance and a running average of the distances ([[baseline]]). Its aim climbs from 20° to about 45° in a few hundred throws. Its spread first widens in the Story's run, while throws far above the aim keep paying off, and then narrows to about 2°; averaged over 100 runs, it shrinks steadily from 10° to about 5° after 100 throws and under 2° after 1000. The world is a bandit with a continuous arm: the [[gradient-bandit|gradient-bandit]] idea with a bell in place of a softmax.

### Historical remarks {#history}

The softmax is the Gibbs, or Boltzmann, distribution of statistical physics; Bridle (1990) gave it its name in neural networks. Learning automata had adjusted action probabilities directly decades earlier, and Williams (1992) set out gradient learning for stochastic units in general, Gaussian units with a learned mean and standard deviation among them. Gullapalli (1990) trained such units to produce real-valued actions. The parameterization \ref{eq-musigma} and its score functions follow Sutton and Barto (2018, §13.7). Diagonal Gaussians with state-independent log standard deviations became the default of deep policy-gradient code, and squashed Gaussians that of soft actor–critic (Haarnoja et al., 2018).

## Card

### Idea

A learned policy needs a way to turn weights into probabilities that can be sampled and differentiated. For a few discrete actions: a softmax over learned preferences. For real-valued actions: a Gaussian with a learned mean and a learned spread. Each has a simple score function, the gradient of the log-probability, which every policy-gradient method multiplies by how well the action turned out.

::: analogy
A darts player who aims at a spot and knows how much their hand wobbles. Practice moves the aim toward the throws that scored and, as the player improves, shrinks the wobble.
:::

### Two families {#families}

| | softmax | Gaussian |
| --- | --- | --- |
| actions | a few, discrete | real-valued |
| weights give | a preference $\pol{h(s, a)}$ per action | a mean $\pol{\mu(s)}$ and a spread $\pol{\sigma(s)}$ |
| sample | pick with chances $e^{\pol{h}} / \sum e^{\pol{h}}$ | $\pol{\mu} + \pol{\sigma} z$, $z \sim \mathcal N(0, 1)$ |
| score $\nabla \ln \pol{\pi}$ | $\mathbf x(s, a) - \sum_b \pol{\pi(b \mid s)}\,\mathbf x(s, b)$ | mean: $(a - \pol{\mu})\,\mathbf x / \pol{\sigma}^2$; spread: $\big((a - \pol{\mu})^2 / \pol{\sigma}^2 - 1\big)\,\mathbf x$ |
| near-deterministic when | one preference runs ahead | $\pol{\sigma}$ shrinks |

### Why it matters {#why}

- Every policy-gradient method, from [[reinforce]] to [[ppo]], uses one of these two, or a cousin, and its score function.
- A Gaussian policy chooses among infinitely many actions without ever maximizing over them. [See it learn to throw](lab:throw)

### Pitfalls

- Learning $\pol{\sigma}$ directly instead of $\ln \pol{\sigma}$: one step can make it negative.
- Forgetting that the Gaussian score grows like $1/\pol{\sigma}$: a policy that has grown sure takes long, noisy steps.
- Confusing preferences with action values: a softmax over values explores at a fixed temperature, learned preferences move wherever performance improves.
- Clipping actions to their bounds and expecting the policy to learn about the bounds: the score of a clipped action does not see the clipping.

### Check yourself {#check}

::: question
A softmax policy has preferences $(2, 0)$. What are the probabilities, and what are they after adding 3 to both?
---
$e^2/(e^2 + 1) \approx 0.88$ and $0.12$, both times. The softmax depends only on differences between preferences.
:::

::: question
A Gaussian policy aims at $\pol{\mu} = 30°$ with $\pol{\sigma} = 10°$. A throw at 35° turns out better than expected. Which way do the aim and the spread move?
---
The aim moves up, toward 35°, since $a - \pol{\mu} > 0$. The spread shrinks: the throw lay half a standard deviation from the aim, so $(a - \pol{\mu})^2/\pol{\sigma}^2 - 1 = -0.75 < 0$.
:::

::: question
Why does the Gaussian score need more care as the policy becomes sure of itself?
---
The push on the mean is $(a - \pol{\mu})/\pol{\sigma}^2$, about $1/\pol{\sigma}$ for a typical action. As $\pol{\sigma}$ shrinks, the same surprise produces ever larger steps.
:::
