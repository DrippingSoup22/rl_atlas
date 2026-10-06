+++
summary = "When the goal is hard to write down but easy to judge, learn the reward from people's judgments. People compare pairs of an agent's outputs; a reward model learns to predict their preferences; the agent is then optimized against that model, with a penalty for drifting far from where it started. This is how large language models are tuned to be helpful, and it inherits every weakness of a learned reward."
story_in = ["reward-design", "ppo"]
prereqs = ["reward-design", "ppo", "imitation"]
sources = [
  { text = "Christiano, Leike, Brown, Martic, Legg & Amodei (2017), Deep reinforcement learning from human preferences, NeurIPS", url = "https://arxiv.org/abs/1706.03741" },
  { text = "Ziegler et al. (2019), Fine-tuning language models from human preferences", url = "https://arxiv.org/abs/1909.08593" },
  { text = "Stiennon et al. (2020), Learning to summarize from human feedback, NeurIPS", url = "https://arxiv.org/abs/2009.01325" },
  { text = "Ouyang et al. (2022), Training language models to follow instructions with human feedback, NeurIPS", url = "https://arxiv.org/abs/2203.02155" },
  { text = "Bai et al. (2022), Constitutional AI: harmlessness from AI feedback", url = "https://arxiv.org/abs/2212.08073" },
  { text = "Rafailov, Sharma, Mitchell, Ermon, Manning & Finn (2023), Direct preference optimization: your language model is secretly a reward model, NeurIPS", url = "https://arxiv.org/abs/2305.18290" },
  { text = "Gao, Schulman & Hilton (2023), Scaling laws for reward model overoptimization, ICML", url = "https://arxiv.org/abs/2210.10760" },
]
+++

## Textbook

### When the reward is the hard part {#hard}

Every method in this guide assumes a reward. For many tasks nobody can write one down: a good summary, a helpful answer, a graceful backflip. People can still *judge* results, especially side by side. RL from human feedback learns the reward from those judgments, then optimizes against what it learned ([[reward-design]]).

### Learning a reward from comparisons {#reward-model}

Absolute scores from people are noisy and inconsistent; comparisons are easier and more reliable. So people are shown two outputs for the same situation, $y_1$ and $y_2$, and say which is better. A **reward model** $r_\phi$ is trained to explain those choices with the Bradley–Terry model: the chance that $y_1$ is preferred is
$$P(y_1 \succ y_2) = \frac{e^{r_\phi(x, y_1)}}{e^{r_\phi(x, y_1)} + e^{r_\phi(x, y_2)}},$$
fit by maximizing the likelihood of the recorded preferences. Christiano et al. (2017) showed the idea on simulated robots and Atari: with about an hour of a person's time comparing short clips, agents learned behaviors such as a backflip that would be hard to specify as a reward.

### Optimizing against it {#optimize}

For a language model, the setting is a contextual bandit with a very long action: the prompt $x$ is the state, the whole response $y$, token by token, is the action, and the reward model scores the finished response. The policy $\pol{\pi_{\boldsymbol\theta}}$ is trained to maximize the learned reward while staying close to a reference model $\pi_{\text{ref}}$, usually the model before this stage:
$$\max_{\boldsymbol\theta}\ \mathbb E_{x,\ y \sim \pi_{\boldsymbol\theta}}\big[r_\phi(x, y)\big] - \beta\,D_{\mathrm{KL}}\big(\pol{\pi_{\boldsymbol\theta}(\cdot \mid x)} \,\|\, \pi_{\text{ref}}(\cdot \mid x)\big).$$
The standard optimizer has been PPO ([[ppo]]), with a value network for advantages. The KL term keeps the outputs fluent and keeps the policy where the reward model was trained, since the reward model is only reliable near the data people judged. The full recipe, imitation of human-written answers first, then a reward model, then PPO, was developed for summarization (Stiennon et al., 2020) and applied to instruction following in InstructGPT (Ouyang et al., 2022).

### Overoptimization {#overoptimization}

The reward model is a learned approximation, and an optimizer will find its errors, as it finds the errors of any learned value ([[td3]]) or model ([[model-based-deep]]). Pushed hard, the policy produces outputs the reward model scores highly and people do not like: longer answers, flattery, confident errors. Gao, Schulman and Hilton (2023) measured it, with a large reward model standing in for people: as the policy moves further from its start, the learned reward keeps rising while the stand-in's score first rises, then falls. The KL penalty, early stopping, larger reward models and fresh human data all push the turning point further out. It is the same lesson as reward hacking, met with a reward that is itself learned.

### Variations {#variations}

- **Direct preference optimization** (DPO; Rafailov et al., 2023) observes that for the KL-regularized objective, the optimal policy and the reward are tied in closed form. It turns the preference data directly into a classification loss on the policy, with no reward model and no RL loop.
- **AI feedback.** Comparisons can come from a model guided by written principles instead of people, as in Constitutional AI (Bai et al., 2022). That makes feedback cheaper, and the principles explicit.
- **Verifiable rewards.** Where answers can be checked automatically, as in mathematics or code with tests, the reward needs no model of preferences at all, and RL can optimize correctness directly; policy-gradient methods in the PPO family are used there too.

## Card

### Idea

Learn the reward from human comparisons: a reward model predicts which of two outputs people prefer, then the policy is optimized against it with a KL penalty to stay near its starting point. It makes goals that can only be judged, not written, trainable, and it inherits the flaws of a learned reward.

::: analogy
A cook who cannot get a recipe, only diners' verdicts on pairs of dishes. Learning what they like works, but cooking only to please the learned taste model eventually produces dishes the model loves and the diners do not.
:::

### The recipe {#recipe}

1. Start from a model trained to imitate good examples.
2. Collect comparisons of its outputs; fit a reward model to them.
3. Optimize the policy against the reward model, with a KL penalty to the start (PPO), or skip the reward model with DPO.
4. Collect new comparisons on the new policy's outputs, and repeat.

### Pitfalls

- Overoptimizing a learned reward: the score rises while quality falls.
- Feedback that rewards style over substance: length and confidence are easy to prefer.
- Dropping the KL term: the policy leaves the region where the reward model is right.

### Check yourself {#check}

::: question
Why compare pairs of outputs instead of asking people for scores?
---
People are inconsistent about absolute scores but much more reliable about which of two is better. A model like Bradley–Terry turns many such comparisons into a consistent reward scale.
:::

::: question
What is the KL penalty for in RLHF?
---
It keeps the tuned policy close to the reference model. The reward model was trained on outputs near that model's; far from it, its scores are unreliable and easy to exploit. The penalty also keeps the outputs fluent.
:::

::: question
How is reward overoptimization related to overestimation in DDPG?
---
Both come from optimizing against a learned estimate: the optimizer moves toward wherever the estimate is too high. In DDPG the actor climbs the critic's errors; in RLHF the policy climbs the reward model's. In both, the cures limit how far the optimizer can move from where the estimate is trustworthy, or make the estimate more cautious.
:::
