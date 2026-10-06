+++
summary = "When someone can show good behavior, an agent can learn from the showing instead of from rewards. Behavior cloning copies the expert's actions like a supervised problem, but small mistakes lead it to states the expert never visited, where it errs more. DAgger asks the expert about the states the learner actually reaches; inverse RL recovers the reward the expert seems to pursue; adversarial methods learn to behave indistinguishably from the expert."
prereqs = ["reward-design", "offline-rl"]
lab = "imitation-bridge"
sources = [
  { text = "Pomerleau (1989), ALVINN: an autonomous land vehicle in a neural network, Advances in Neural Information Processing Systems 1" },
  { text = "Chi, Feng, Du, Xu, Cousineau, Burchfiel & Song (2023), Diffusion policy: visuomotor policy learning via action diffusion, Robotics: Science and Systems", url = "https://arxiv.org/abs/2303.04137" },
  { text = "Ross, Gordon & Bagnell (2011), A reduction of imitation learning and structured prediction to no-regret online learning (DAgger), AISTATS", url = "https://arxiv.org/abs/1011.0686" },
  { text = "Ng & Russell (2000), Algorithms for inverse reinforcement learning, ICML", url = "https://ai.stanford.edu/~ang/papers/icml00-irl.pdf" },
  { text = "Abbeel & Ng (2004), Apprenticeship learning via inverse reinforcement learning, ICML", url = "https://ai.stanford.edu/~ang/papers/icml04-apprentice.pdf" },
  { text = "Ziebart, Maas, Bagnell & Dey (2008), Maximum entropy inverse reinforcement learning, AAAI", url = "https://cdn.aaai.org/AAAI/2008/AAAI08-227.pdf" },
  { text = "Ho & Ermon (2016), Generative adversarial imitation learning, NeurIPS", url = "https://arxiv.org/abs/1606.03476" },
]

[story]
scene = "grid"
env = "ice-bridge"
seed = 7
average = 100
formula = '''\step{1}{\pol{\pi(s)} \leftarrow \pol{\pi_E(s)} \text{ for the states the expert reaches}} \step{2}{\qquad \pol{\pi(s)} \leftarrow \pol{\pi_E(s)} \text{ for the states the learner reaches}}'''

[story.runs]
clone = { algorithm = "bc", gamma = 0.95, judge = 1.0, maxSteps = 100, calmExpert = true, generalize = "nearest", units = 20, measures = ["policy-value"], name = "behavior cloning" }
dagger = { algorithm = "dagger", gamma = 0.95, judge = 1.0, maxSteps = 100, calmExpert = true, generalize = "nearest", units = 20, measures = ["policy-value"], name = "DAgger" }
+++

## Story

::: step {run = "clone", at = 0, play = 1, pace = 320, arrows = false}
**A bridge of ice, with water on both sides.** The gem waits at the far end. The expert crosses it without ever slipping, as a skilled skater would: straight along the middle row, eight moves. Watch its demonstration.
:::

::: step {run = "clone", at = 1, fog = true, formula = 1}
**The clone learns from it.** It knows eight tiles, each with the expert's move: right. Under the fog, it copies the move of the nearest tile it was shown, as a network would. But the clone is no expert: one move in ten, it slides to a side. Played with those slides, the expert's own policy still reaches the gem 89% of the time, because it knows how to come back. The clone, 49%.
:::

::: step {run = "dagger", at = 1, play = 1, pace = 320, fog = true}
**Here is the clone on the ice.** Two tiles in, it slides onto the lower row, a tile no demonstration ever showed. The nearest tile it knows says right, so right it goes, along the edge and off the end into the water. More demonstrations would not help: they all walk the middle row, and none shows a recovery.
:::

::: step {run = "dagger", at = 2, fog = true, formula = 2}
**DAgger asks the expert about the clone's own walk.** For each tile the clone stood on, what would you do here? Six new labels, all along the lower row, all the same: up, back to the middle. The chance of reaching the gem rises from 49% to 66%.
:::

::: step {run = "dagger", at = 3, play = 1, pace = 320, fog = true}
**Two walks later, the same slide**, onto the same tile of the lower row. This time the clone knows what to do there, and steps back up. After six episodes, the first of them the demonstration, it has nineteen labels and reaches the gem 89% of the time, as often as the expert's own policy.
:::

::: step {run = "dagger", at = 20, curves = ["clone", "dagger"], metric = "policy-value", title = "Chance the learner reaches the gem"}
**A hundred runs each.** Behavior cloning stays at 49% however many demonstrations it watches: the expert never shows a recovery. DAgger averages 87% after 20 walks, and 96 runs of 100 pass 80%. The expert's time is spent where the clone goes wrong. [Race them in the Lab](lab:imitation-bridge).
:::

## Textbook

### Learning from demonstrations {#demos}

Writing a reward that captures what we want is hard ([[reward-design]]); showing what we want is often easy. A driver can drive, a surgeon can operate, a player can play. **Imitation learning** learns a policy from demonstrations: trajectories of states and the expert's actions, sometimes with no reward at all.

### Behavior cloning {#cloning}

The direct approach treats the demonstrations as labeled data: states as inputs, the expert's actions as labels, and a policy trained by supervised learning to predict them. ALVINN (Pomerleau, 1989) learned to steer a van this way from camera images, and behavior cloning remains the first thing to try.

Its weakness is **compounding error**. The expert's data covers the states the expert visits. The clone makes a small mistake, drifts slightly off the expert's path, and lands in a state the data never shows, where its mistakes are larger, which takes it further off. Ross, Gordon and Bagnell (2011) showed that with a per-step error $\epsilon$, the total cost of a cloned policy over a horizon $T$ can grow like $\epsilon T^2$, against $\epsilon T$ for a learner trained on its own states. ALVINN's designers met it directly: a car that never saw the expert recover from the road's edge had no idea how to recover, and they had to synthesize views of shifted positions with their corrective steering.

Cloning has had a revival in robotics. Demonstrations often contain several right ways to do a task, and a policy that averages them, reaching neither left nor right around an obstacle, fails. Expressive generative policies, such as diffusion models that output a short sequence of actions at a time (Chi et al., 2023), keep the alternatives apart, and with a few hundred demonstrations they learn many manipulation skills by cloning alone.

### DAgger: ask about your own mistakes {#dagger}

DAgger (dataset aggregation) fixes the mismatch by collecting labels where the learner goes. Train a policy on the demonstrations; run it; ask the expert what it would have done in each state the learner visited; add those labels to the dataset; retrain; repeat. The data then covers the learner's own distribution of states, including the recoveries, and the error grows only linearly with the horizon. The cost is an expert who can be queried, not just recorded.

::: example {#ex-bridge} An expert who never slips
On a bridge of ice three tiles wide, an expert walks the middle row without ever slipping; the learner slides to a side one move in ten. A clone that copies the expert's move where it has one, and the move of the nearest labeled tile elsewhere, reaches the gem 49% of the time, against 89% for the expert's own policy played with the learner's slides. Every further demonstration walks the same row, so cloning never gets better. After a slide, the clone's nearest label says "right", and it walks along the edge into the water. With DAgger, the expert labels the tiles the clone actually stood on, "up" along the lower row and "down" along the upper one; over 100 runs, the chance of reaching the gem averages 87% after 20 walks.
:::

### Inverse reinforcement learning {#irl}

Instead of copying actions, infer *why*: find a reward function under which the expert's behavior is optimal, then solve for a policy with RL (Ng and Russell, 2000). A recovered reward transfers to new situations where copied actions would not, and it explains the behavior. The problem is ill-posed, since many rewards make the same behavior optimal, including a reward of zero everywhere. Apprenticeship learning (Abbeel and Ng, 2004) matches the expert's feature counts instead of guessing one true reward; maximum entropy IRL (Ziebart et al., 2008) picks, among the behaviors that match, the most random one, which resolves the ambiguity in a principled way.

### Adversarial imitation {#gail}

GAIL (Ho and Ermon, 2016) skips the explicit reward: a discriminator learns to tell the expert's state-action pairs from the learner's, and the learner is trained with RL to fool it, using the discriminator's output as its reward. At the optimum, the learner's distribution of states and actions matches the expert's. Because the learner interacts with the world, it learns to recover from its own mistakes, which cloning cannot.

### Imitation and RL together {#together}

Demonstrations and rewards combine well. AlphaGo's policy network began by imitating human games before improving by self-play ([[mcts]]). Robots are often given a few demonstrations to start, then refined with RL. Large language models are first trained to imitate text and human-written answers, a form of behavior cloning, before being tuned with feedback ([[rlhf]]). Imitation gives a competent start; RL improves beyond the teacher.

## Card

### Idea

Learn from demonstrations instead of, or before, rewards: copy the expert's actions (behavior cloning), ask the expert about your own states (DAgger), infer the reward the expert pursues (inverse RL), or learn to look like the expert to a discriminator (GAIL).

::: analogy
Learning a dance by copying a video works until you misstep, and the video never shows how to recover. A teacher who watches you and corrects your own mistakes is DAgger.
:::

### The methods {#methods}

- Behavior cloning: supervised learning on the expert's state–action pairs.
- DAgger: the expert labels the states the learner visits.
- Inverse RL: recover a reward, then do RL.
- GAIL: RL against a discriminator that tells learner from expert.

### Pseudocode

::: pseudocode
Labels $\mathcal D \leftarrow \emptyset$; the learner's policy $\pol{\pi}$ fit to $\mathcal D$
Repeat for each episode:
  First episode (behavior cloning: every episode): the expert plays {#demo}
  Later episodes: the learner plays $\pol{\pi}$ {#play}
  For each state $s$ reached: ask the expert, $\mathcal D \leftarrow \mathcal D \cup \{(s, \pol{\pi_E(s)})\}$ {#label}
  Fit $\pol{\pi}$ to $\mathcal D$
:::

### Pitfalls

- Cloning without recovery data: errors compound off the expert's path.
- Demonstrations that hide the expert's information: the camera does not see what the driver heard.
- Expecting imitation to surpass the teacher: that needs a reward, and RL.

### Check yourself {#check}

::: question
Why can a behavior-cloned policy fail even when it matches the expert's actions 99% of the time on the demonstrations?
---
The 1% of mistakes take it to states the expert never visited, where it has no data and errs more often, which takes it further away. The errors compound over the horizon instead of staying at 1% per step.
:::

::: question
What does DAgger need that behavior cloning does not?
---
An expert who can be asked what to do in any state the learner reaches, not just a fixed set of recordings. That lets the dataset cover the learner's own states, including the recoveries.
:::

::: question
Why is recovering a reward from demonstrations ill-posed, and how does maximum entropy IRL respond?
---
Many rewards make the same behavior optimal, a reward of zero everywhere among them. Maximum entropy IRL chooses, among the behaviors consistent with the expert's feature counts, the least committed one, which picks a single answer without assuming more than the data shows.
:::
