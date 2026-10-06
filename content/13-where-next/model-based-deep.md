+++
summary = "Learn a model of the world from experience, then use it: to plan actions, to generate extra experience, or to train a policy entirely in imagination. A good model can save most of the real interaction a model-free method needs. The danger is that a policy learns to exploit the model's mistakes, so the best methods keep track of what the model does not know and trust it only a few steps ahead."
prereqs = ["models", "dyna-q", "mcts"]
sources = [
  { text = "Sutton (1991), Dyna, an integrated architecture for learning, planning, and reacting, SIGART Bulletin 2", url = "https://dl.acm.org/doi/10.1145/122344.122377" },
  { text = "Chua, Calandra, McAllister & Levine (2018), Deep reinforcement learning in a handful of trials using probabilistic dynamics models (PETS), NeurIPS", url = "https://arxiv.org/abs/1805.12114" },
  { text = "Ha & Schmidhuber (2018), World models", url = "https://arxiv.org/abs/1803.10122" },
  { text = "Janner, Fu, Zhang & Levine (2019), When to trust your model: model-based policy optimization (MBPO), NeurIPS", url = "https://arxiv.org/abs/1906.08253" },
  { text = "Hafner, Pasukonis, Ba & Lillicrap (2023), Mastering diverse domains through world models (DreamerV3)", url = "https://arxiv.org/abs/2301.04104" },
  { text = "Schrittwieser et al. (2020), Mastering Atari, Go, chess and shogi by planning with a learned model (MuZero), Nature 588", url = "https://www.nature.com/articles/s41586-020-03051-4" },
]
+++

## Textbook

### Why learn a model {#why}

The off-policy methods of Part 11 needed 6,000 to 9,000 steps of experience to swing up a pendulum; Atari agents use millions of frames. For a robot or any system where trying is slow or costly, that is too many. A **model** of the world, a function that predicts the next state and reward from a state and action, can be learned from the same experience and then queried as often as needed. Dyna showed the idea with tables: every real step also updates a model, and the model replays imagined steps to the learner ([[dyna-q]]). Deep model-based RL does the same with networks, in worlds too large for tables.

### Three ways to use a model {#uses}

1. **Plan with it.** At each step, search over action sequences in the model and take the first action of the best one, then plan again from the next state (model-predictive control). PETS (Chua et al., 2018) learned an ensemble of probabilistic networks and planned by sampling action sequences. On simulated robots it matched strong model-free methods with many times less experience: on one task, 8 times less than SAC. MuZero searches with a tree inside its learned model ([[mcts]]).
2. **Make more experience.** Use the model to generate short imagined transitions, starting from real states, and feed them to a model-free learner, as Dyna does. MBPO (Janner et al., 2019) branches rollouts of a few steps from states in the replay memory and trains SAC on the mix ([[sac]]).
3. **Learn in imagination.** Train the policy entirely inside the model, on long imagined trajectories in a learned compact state. World Models (Ha and Schmidhuber, 2018) trained a controller entirely inside a learned "dream" of a video game, then ran it in the real game. The Dreamer agents learn a latent model from pixels and train an actor–critic on its imagined trajectories; DreamerV3 (Hafner et al., 2023) used one set of settings across many domains and was the first to collect diamonds in Minecraft from scratch, without human data.

### What goes wrong {#wrong}

A model is wrong somewhere, and a policy optimized against it will find exactly where: an action the model wrongly thinks is excellent looks like the best plan. This is the same weakness as an overestimating critic ([[td3]]), and it grows with the length of imagined trajectories, because each predicted step starts from the previous prediction and errors compound. The main defenses:

- **Uncertainty.** An ensemble of models trained on different data disagrees where data is scarce; plans can avoid, or be penalized for, regions of disagreement.
- **Short horizons.** Trust the model for a few steps from real states and bootstrap with a value function beyond that, as MBPO does.
- **Learn what matters.** A model need not predict every pixel; it needs to predict what affects rewards and values. MuZero's model predicts only rewards, values and policies, never observations.

### Model-based or model-free? {#which}

Model-based methods win when experience is expensive and the world is regular enough to learn: robotics, control, games with clear dynamics. Model-free methods are simpler, and they win when experience is cheap or the dynamics are hard to predict but a good policy is simple. The line keeps moving; the strongest recent agents often combine both, a learned model for planning or imagination and model-free value learning on top ([[model-based-free]]).

## Card

### Idea

Learn to predict the next state and reward, then use the predictions: plan actions, generate extra experience for a model-free learner, or train the policy entirely in imagination. It saves real experience, if the policy is kept from exploiting the model's errors.

::: analogy
A pilot who trains in a flight simulator: thousands of hours of practice for the price of a few real flights, as long as the simulator behaves like the airplane where it matters.
:::

### The uses {#uses}

- Planning with the model at decision time (PETS, MuZero).
- Short imagined rollouts as extra data (Dyna, MBPO).
- Learning the policy in imagination (World Models, Dreamer).

### Pitfalls

- Optimizing against the model's errors: plans that only work in the model.
- Long imagined rollouts: prediction errors compound step by step.
- Spending the model's capacity on details that do not affect rewards.

### Check yourself {#check}

::: question
Why can a policy trained on a learned model look excellent in imagination and fail in reality?
---
The policy is optimized against the model, and finds actions where the model is wrong in its favor. Those errors do not exist in the real world. Ensembles that flag uncertainty, and short rollouts from real states, limit the damage.
:::

::: question
Why does MBPO imagine only a few steps from real states instead of whole episodes?
---
Prediction errors compound: each imagined step starts from the previous prediction. Short branches from real states keep the imagined data close to reality, and a value function bootstraps beyond them.
:::

::: question
What does MuZero's model predict, and what does it leave out?
---
From a learned hidden state, it predicts the reward, the value and the move probabilities after each action. It never predicts the next observation, so its capacity goes only to what matters for planning.
:::
