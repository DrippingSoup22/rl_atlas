+++
summary = "Instead of adding noise to every action, perturb the actor's weights once per episode and act with the perturbed copy for the whole episode. The same state then gets the same exploratory action, and the exploration is a consistent, smooth function of the state. Since nobody can say how much noise a weight needs, the perturbation adapts until the actions move by a set amount, and the inputs must be of comparable size: on continuous Mountain Car, whose speed never exceeds 0.07 while its position ranges over 1.8, noise in the weights finds the flag only once the observations are normalized. NoisyNets learn their noise instead; evolution strategies explore with weight noise alone."
prereqs = ["action-noise", "ddpg"]
lab = "parameter-noise-car"
sources = [
  { text = "Plappert, Houthooft, Dhariwal, Sidor, Chen, Chen, Asfour, Abbeel & Andrychowicz (2018), Parameter space noise for exploration, ICLR", url = "https://arxiv.org/abs/1706.01905" },
  { text = "Fortunato et al. (2018), Noisy networks for exploration, ICLR", url = "https://arxiv.org/abs/1706.10295" },
  { text = "Rückstieß, Felder & Schmidhuber (2008), State-dependent exploration for policy gradient methods, ECML PKDD", url = "https://link.springer.com/chapter/10.1007/978-3-540-87481-2_16" },
  { text = "Salimans, Ho, Chen, Sidor & Sutskever (2017), Evolution strategies as a scalable alternative to reinforcement learning", url = "https://arxiv.org/abs/1703.03864" },
  { text = "Mania, Guy & Recht (2018), Simple random search provides a competitive approach to reinforcement learning, NeurIPS", url = "https://arxiv.org/abs/1803.07055" },
  { text = "Hessel et al. (2018), Rainbow: combining improvements in deep reinforcement learning, AAAI", url = "https://arxiv.org/abs/1710.02298" },
]

[story]
scene = "hillcar"
env = "mountain-car-continuous"
formula = '''\step{1}{\tilde{\boldsymbol\theta} = \boldsymbol\theta + \knob{\sigma_k}\,\boldsymbol\xi \quad \text{once per episode} \qquad} \step{2}{\knob{\sigma_{k+1}} = \begin{cases} 1.01\,\knob{\sigma_k} & d(\pol{\mu}, \pol{\tilde\mu}) < \knob{\delta} \\ \knob{\sigma_k} / 1.01 & \text{otherwise} \end{cases}}'''

[story.runs]
action = { recording = "td3-car-white-noise", name = "noise on the actions" }
weights = { recording = "td3-car-param-noise", name = "noise in the weights" }
norm = { recording = "td3-car-param-noise-norm", name = "normalized inputs" }
+++

## Story

::: step {run = "weights", at = 0, formula = 1}
**Noise in the weights.** At the start of each episode TD3 copies its actor, adds a little noise to every weight, and drives with the copy, with no other noise (formula 1). The same position and speed then get the same push all episode long: each episode tries one slightly different policy.
:::

::: step {run = "weights", at = 20, map = "action", formula = 2}
**How much noise.** Every 50 steps a second perturbed copy is compared with the actor on remembered states, and the noise grows or shrinks by 1% until their pushes differ by 0.5, the white noise's spread (formula 2). With the raw observations, the copies push steadily, often hard, but almost never swing: in 40,000 steps of training, this seed's copies reach the flag once.
:::

::: step {run = "weights", at = 40, map = "action"}
**The speed is too small to matter.** The car's speed never exceeds 0.07, while its position ranges over 1.8. A perturbation changes the actor's response to each input in proportion to its size, so each copy's extra push depends on where the car is, not on how it moves, and a push like that cannot pump a swing. After 40,000 steps the critic has seen the flag, but the actor's own push still ignores the speed: on the map it changes across the positions, in stripes, and hardly with the speed. Its test episode stands still.
:::

::: step {run = "norm", at = 2, map = "value"}
**The same, with normalized observations**: each input has its running mean subtracted and is divided by its running spread, as Plappert et al. did. Now the speed weighs as much as the position, and some copies push with the motion: within 2,000 steps one drives to the flag, and the critic rates the states near it up to 44.
:::

::: step {run = "norm", at = 27, map = "action"}
**After 27,000 steps the actor swings by itself**: its test episode reaches the flag in 152 steps. By the end it pushes with the speed in 92% of the map.
:::

::: step {run = "norm", at = 40, curves = ["action", "weights", "norm"], metric = "test"}
**Twenty seeds of each.** Noise on the actions: 1 seed ends up reaching the flag. Noise in the weights with raw observations: 1 too, although its copies do reach the flag in training on 15 seeds, late, half of them after 27,000 steps, and rarely. With normalized observations: 15, and their copies reach the flag in training on every seed, half of them within 3,500 steps.
:::

::: step {run = "norm", at = 40, curves = ["action", "weights", "norm"], metric = "jitter-train"}
**And gently.** A perturbed actor is a smooth function of the state: with normalized observations its training pushes jump by 0.4 from step to step on average, against 11.4 with white noise on the actions. [Race the three in the Lab](lab:parameter-noise-car).
:::

## Textbook

### The same state, the same exploration {#consistent}

Noise added to the actions, of any color, is blind to the state: a state visited twice in an episode gets two different exploratory actions ([[action-noise]]). **Parameter noise** perturbs the policy instead. At the start of each episode, every weight of the actor gets a little Gaussian noise,
$$\tilde{\boldsymbol\theta} = \boldsymbol\theta + \knob{\sigma}\,\boldsymbol\xi, \qquad \boldsymbol\xi \sim \mathcal N(\mathbf 0, \mathbf I), \label{eq-perturb}$$
and the agent acts with the perturbed copy $\pol{\tilde\mu} = \pol{\mu(\cdot, \tilde{\boldsymbol\theta})}$ for the whole episode, with no other noise (Plappert et al., 2018). The exploratory action is now a function of the state: the same state gets the same action every time it comes back, and nearby states get nearby actions. Each episode tries one coherent, slightly different policy, rather than the same policy shaken at every step.

In continuous Mountain Car this is what the car needs. A perturbed actor that happens to push a little harder in the direction the car is already moving pumps the swing at every pass, for the whole episode, where independent noise would cancel ([[action-noise]]).

### How much noise: adapt it in action space {#adaptive}

Nobody can say how much noise a weight needs: it depends on the layer, on the network's size, and it changes as training sharpens the weights. Plappert et al. measure the noise where it matters, in the actions. Every so often (every 50 steps here) they perturb a second copy with the current $\knob\sigma$ and compare its actions with the actor's on a batch of states from the memory:
$$d(\pol\mu, \pol{\tilde\mu}) = \sqrt{\frac1N \sum_{i=1}^{N} \mathbb E_s\big[(\pol{\mu(s)}_i - \pol{\tilde\mu(s)}_i)^2\big]}, \label{eq-distance}$$
over the $N$ action dimensions. If the distance is below a threshold $\knob\delta$, the noise grows by a factor of 1.01; if above, it shrinks by the same factor. The threshold is chosen in action units: since Gaussian action noise of spread $\knob\sigma$ is at exactly this distance $\knob\sigma$ from the actor, setting $\knob\delta$ to an action-noise spread gives parameter noise of the same size. The paper used 0.2 on tasks with dense rewards and 0.6 on sparse ones.

The paper also put **layer normalization** between the actor's layers ([[normalization]]): with every layer's units rescaled to the same spread, the same $\knob\sigma$ moves every layer by a comparable amount. The trainer of this guide has no layer normalization; the adaptive scale does the work alone.

### What the actor sees {#inputs}

Plappert et al. normalized the observations too, and on the car that decides everything. The car's speed never exceeds 0.07 either way, while its position ranges over 1.8, from $-1.2$ to 0.6. A perturbation of the first layer's weights changes the actor's response to each input in proportion to that input's size, so it changes the response to the speed far less than the response to the position: the perturbed actor's extra push depends, in effect, on where the car is and not on how it moves.

No push that depends on the position alone can take the car to the flag. Whatever such a push adds while the car moves one way over a stretch of the track, it takes back when the car comes back the other way over the same stretch, as if only the valley's shape had changed; in 240,000 trials of random functions of the position, from starts across the valley's bottom, none reached the flag. Pumping the swing needs a push that follows the speed: to the right while the car moves right, to the left while it moves left. Normalized by their running mean and spread ([[normalization]]), both inputs reach the network at the same scale, and some perturbed actors push with the motion.

::: example {#ex-car-weights} Perturbed actors alone on the hill
Take an untrained actor, as TD3 starts with, remember 1,000 steps of random actions, adapt the perturbation on them until the actions move by 0.5, the size of the action noise in the example of [[action-noise]], and let one perturbed copy drive the car for an episode, without learning. With raw inputs, 1 of 300 such copies reached the flag. With both inputs normalized, 82 of 300 did, in 277 steps on average. The copies that got there pushed in the direction the car was moving 71% of the time, the others 44%: a perturbed actor whose push happens to follow the speed pumps the swing at every pass. White noise on the actions, of the same size, never reached the flag; pink noise reached it in 39% of the episodes.
:::

Training shows the same. In the Lab's race, 20 seeds of 40,000 steps each, TD3 with noise in its weights and normalized inputs ends up reaching the flag on 15 seeds, half of them within 14,000 steps; with raw inputs, on 1, as with white noise on its actions. Normalizing does little for noise on the actions, which does not pass through the inputs: in the Lab's sweep of TD3 with white noise, 10 seeds each, normalized observations brought 2 seeds to the flag, against 1.

### Off-policy and on-policy {#on-off}

For DDPG and TD3 nothing else changes: the perturbed actor only collects experience; the critics and the unperturbed actor learn from the memory as usual ([[on-off-policy]]). In the paper's experiments, DDPG with parameter noise broke out of a poor habit on HalfCheetah (flipping onto its back and wiggling forward) that every other kind of noise kept. On a sparse cart-pole swing-up and a sparse Mountain Car, only parameter noise learned: with Gaussian or Ornstein–Uhlenbeck noise on its actions, DDPG never reached a state with a reward. On tasks with well-shaped rewards it did as well as action noise, and so did no noise at all: those tasks do not test exploration.

For an on-policy method the perturbation becomes part of the policy, a distribution over weights, and its gradient follows from it (Rückstieß et al., 2008; the paper did this for TRPO). For DQN the paper perturbed a policy head trained to imitate the greedy action, adapted against the KL divergence that ε-greedy would cause, and kept a little ε-greedy exploration on top.

The authors are clear about the limit: parameter noise shines when a consistent behavior is what exploration needs, such as "keep going right" in a long chain. When the right action must change from state to state in a way a random perturbation is unlikely to produce, it has no special advantage.

### Noisy networks {#noisy}

Noisy networks (Fortunato et al., 2018) go one step further and **learn** the noise. Each weight of a noisy layer is $w = \mu_w + \sigma_w \varepsilon_w$, with $\mu_w$ and $\sigma_w$ learned by gradient descent along with everything else, and $\varepsilon_w$ drawn anew. The network can then shrink the noise where exploration no longer pays and keep it where it does. Noisy networks replaced ε-greedy in Rainbow (Hessel et al., 2018), the combination of DQN's extensions ([[dqn-extensions]]).

### Evolution strategies {#es}

Push the idea to its end and drop the gradient of the return: perturb the weights, play an episode with each perturbation, and move the weights toward the perturbations that scored best. Evolution strategies (Salimans et al., 2017) and augmented random search (Mania et al., 2018) learn MuJoCo locomotion this way, with no critic and no backpropagation through time, and parallelize across thousands of cores. They need more samples than an actor–critic, and they ignore which step of an episode earned what: on 21 Atari games, DQN with parameter noise beat the evolution strategies' scores on 15, with 25 times less experience (Plappert et al., 2018). Parameter noise keeps the gradient-based learner and borrows only their exploration.

### Historical remarks {#history}

Rückstieß, Felder and Schmidhuber (2008) argued for exploration that depends on the state, through noise in the parameters of a linear policy, for policy gradients. Plappert et al. (2018) made it work for deep networks, with layer normalization and the adaptive scale, for DQN, DDPG and TRPO; Fortunato et al. (2018) published noisy networks the same year. [[gsde]] returned to Rückstieß's idea with deep networks, for robots.

## Card

### Idea

Perturb the actor's weights once per episode and explore with the perturbed copy: the same state gets the same exploratory action, consistently and smoothly. Adapt the perturbation's size so that the actions move by a set amount.

::: analogy
To learn a dance, one dancer jitters a little at every beat; another dances a whole song in a slightly different style each time. The second tries more coherent variations.
:::

### The update {#update}

Perturb the actor's weights, once per episode.

$$\tilde{\boldsymbol\theta} = \boldsymbol\theta + \knob\sigma\boldsymbol\xi$$

Adapt the scale so that the perturbed actions differ from the plain ones by about $\knob\delta$. $d$ is the root mean squared difference of the two actors' actions over a batch of remembered states.

$$\knob\sigma \leftarrow \begin{cases} 1.01\,\knob\sigma & d(\pol\mu, \pol{\tilde\mu}) < \knob\delta \\ \knob\sigma / 1.01 & \text{otherwise} \end{cases}$$

### Pseudocode

::: pseudocode
Parameters: the target distance $\knob\delta$ (in action units, like an action noise's spread), a first perturbation scale $\knob\sigma = 0.1$; the rest is TD3's
Initialize TD3: the actor $\boldsymbol\theta$, twin critics, their target copies; an empty memory $\mathcal D$
Repeat for each episode:
  Perturb: $\tilde{\boldsymbol\theta} \leftarrow \boldsymbol\theta + \knob\sigma\boldsymbol\xi$, $\boldsymbol\xi \sim \mathcal N(\mathbf 0, \mathbf I)$
  Repeat for each step:
    $A \leftarrow \pol{\mu(S, \tilde{\boldsymbol\theta})}$, no other noise (a random action during the first steps); take it, observe $\rew R$, $S'$; store $(S, A, \rew R, S')$ in $\mathcal D$
    TD3's update of $\boldsymbol\theta$, the critics and the copies, from a batch of $\mathcal D$
    Every 50 steps: perturb a second copy by $\knob\sigma$; $d \leftarrow$ the root mean squared difference of its actions and $\pol{\mu}$'s on a batch of $\mathcal D$; $\knob\sigma \leftarrow 1.01\,\knob\sigma$ if $d < \knob\delta$, else $\knob\sigma / 1.01$
    $S \leftarrow S'$
:::

### Pitfalls

- Choosing the weights' noise by hand: its effect on the actions depends on the network and drifts during training; adapt it in action space.
- Perturbing layers of very different scales by the same amount, without layer normalization.
- Inputs of very different sizes: the perturbation then barely touches the small ones (the car's speed). Normalize the observations.
- Expecting it to help where the actions must vary from state to state in a way no small perturbation produces.
- Testing with the perturbed actor: the test plays the unperturbed one.

### Check yourself {#check}

::: question
A state comes back three times in an episode. What does each kind of noise do there?
---
Action noise, white or colored, gives it three different exploratory actions. Parameter noise gives it the same action three times: the perturbed actor is a fixed function of the state for the whole episode.
:::

::: question
Why adapt the perturbation's size against a distance between actions, rather than fix it?
---
How much a weight's noise moves the actions depends on the layer, the network's size and the current weights, which change as training goes on. Measuring the effect in the actions, and steering it to a threshold in action units, keeps the exploration's size meaningful and comparable to action noise.
:::
