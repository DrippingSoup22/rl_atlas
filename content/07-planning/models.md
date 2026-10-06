+++
summary = "A model is anything the agent can ask “what would happen if I did this here?” Planning is learning from the model's answers instead of from the world: the same updates, applied to imagined experience."
story_in = ["dyna-q", "prioritized-sweeping"]
prereqs = ["model-based-free", "q-learning", "policy-evaluation"]
lab = "dyna-maze"
sources = [
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., §8.1, §8.5–8.6 and §8.8–8.11", url = "http://incompleteideas.net/book/the-book-2nd.html" },
  { text = "Sutton (1990), Integrated architectures for learning, planning, and reacting based on approximating dynamic programming, Proceedings of the 7th International Conference on Machine Learning", url = "https://doi.org/10.1016/B978-1-55860-141-3.50030-4" },
  { text = "Barto, Bradtke & Singh (1995), Learning to act using real-time dynamic programming, Artificial Intelligence 72", url = "https://doi.org/10.1016/0004-3702(94)00011-O" },
  { text = "Tesauro & Galperin (1997), On-line policy improvement using Monte-Carlo search, Advances in Neural Information Processing Systems 9" },
  { text = "Talvitie (2014), Model regularization for stable sample rollouts, Proceedings of the 30th Conference on Uncertainty in Artificial Intelligence" },
  { text = "Grimm, Barreto, Singh & Silver (2020), The value equivalence principle for model-based reinforcement learning, NeurIPS", url = "https://arxiv.org/abs/2011.03506" },
  { text = "Craik (1943), The Nature of Explanation, Cambridge University Press" },
]
+++

## Textbook

### What a model is {#idea}

Everything the agent has met so far learns from the world directly: act, see what happens, update. A **model** adds a second source. It is the agent's own version of the environment, a function it can query without acting: *if I were in this state and did this, what would happen next, and what would it pay?* The world answers by happening; the model answers on demand, as often as asked, for situations the agent is not in.

A model can answer in two ways, and the difference decides which algorithms can use it.

- A **distribution model** answers with every possible outcome and its probability, the full $p(s', r \mid s, a)$. This is what dynamic programming assumed it had ([[policy-evaluation]]).
- A **sample model** answers with one outcome, drawn with the right probabilities, like the real world does. A shuffled deck is a good picture: drawing a card is trivial, while writing down the chance of every five-card hand is a chore.

Samples are easy to get from a distribution model; the reverse requires many samples and gives only an estimate. A model can be handed to the agent (the rules of chess, a physics simulator) or **learned** from experience, by remembering what followed each action. Either way it produces *simulated experience*: transitions that did not happen, but could have, and that the agent can learn from at the price of computation instead of real interaction.

A learned model is never exact, and its mistakes behave differently from the noise in real experience. Asked one step ahead, it is off by a little; asked to imagine a long trajectory, it takes each of its own predictions as the next input, and the errors compound (Talvitie, 2014). Nor does it need to be a faithful picture of the world: it only has to get right what changes the values and the choice of action. Such *value-equivalent* models (Grimm et al., 2020) are what deep model-based agents like MuZero learn ([[model-based-deep]]).

### Planning {#planning}

In this atlas, **planning** means turning a model into a better policy by computation alone, without acting. The kind of planning used here works on states and actions: it estimates value functions, as every method so far did, and computes them with the same kinds of updates, only applied to transitions the model makes up. Planning and learning then differ in a single respect, where their transitions come from ([[model-based-free]]):

$$\text{model} \;\longrightarrow\; \text{simulated experience} \;\xrightarrow{\ \text{updates}\ }\; \text{values} \;\longrightarrow\; \text{policy}. \label{eq-planning}$$

Swap the model for the real environment and the same pipeline is learning. This is a useful way to read the whole subject: almost every learning method of Parts 3 to 6 becomes a planning method by feeding it the model's output, and almost every idea about planning (which states to update, in what order, with which target) has a counterpart in learning. The simplest example takes [[q-learning]] and lets it learn from a model queried at random:

::: algorithm {#alg-qplan} Q-planning on random samples from a model
Input: a sample model, a step size $\alp$ and a discount $\gam$
Repeat forever:
  Pick a state $S$ and an action $A$ at random
  Ask the sample model for a next state $S'$ and reward $\rew{R}$
  $\val{Q(S, A)} \leftarrow \val{Q(S, A)} + \alp\,[\rew{R} + \gam \max_a \val{Q(S', a)} - \val{Q(S, A)}]$
:::

If every state–action pair keeps being picked and the step sizes shrink as Q-learning requires, the values converge to the optimal values *of the model*: as good as the model, no better. Planning in such small, independent steps has two practical advantages. It can be stopped at any moment and still leave useful values, and it can be interleaved with acting and learning, a few updates between real steps, which is exactly what [[dyna-q]] does.

### Expected and sample updates {#updates}

An update can **average** over all possible next states, using a distribution model, or use one **sample**. For action values, the two versions are

$$\val{Q(s, a)} \leftarrow \sum_{s', r} p(s', r \mid s, a)\,\Big[\rew{r} + \gam \max_{a'} \val{Q(s', a')}\Big] \qquad \text{and} \qquad \val{Q(s, a)} \leftarrow \val{Q(s, a)} + \alp\,\Big[\rew{R} + \gam \max_{a'} \val{Q(S', a')} - \val{Q(s, a)}\Big]. \label{eq-updates}$$

The expected update leaves no randomness in its target, but it must look at every possible successor: its cost grows with the **branching factor** $b$, the number of next states with nonzero probability. The sample update looks at one successor, so it is $b$ times cheaper, but its target is noisy and only an average of many of them is reliable. The fair comparison is at equal computation. Take one pair whose estimate is off by 1, with $b$ equally likely successors whose values are already correct. One expected update, $b$ computations, removes the error completely. Sample updates with step sizes $1/t$ (each a running average of the targets seen) leave, after $t$ of them, an error of $\sqrt{(b-1)/(bt)}$, the usual square-root shrinking of an average of $t$ draws.

::: figure {#fig-updates}
{{expected-vs-sample}}
Error left in one estimate after expected and sample updates, against the number of computations, for branching factors 2, 10, 100 and 1000; the estimate starts with an error of 1 and the successors' values are exact. Worked out exactly. After Sutton & Barto, Figure 8.7.
:::

The curves (\ref{fig-updates}) fall steeply at first and slowly after: the first few samples do most of the work. With $b = 1000$, a hundred samples, a tenth of the cost of one expected update, already leave only 10% of the error. The rest of the budget is better spent on other pairs. In a real planning problem this effect is stronger still, because the successors' values are not correct yet: an expected update spends its whole budget averaging over values that will soon change, while cheap sample updates spread the same budget over many pairs and keep the successors' values improving too. Expected updates remain the better choice when the branching factor is small, or when an exact answer matters more than speed.

### Where to spend the updates {#distribution}

Dynamic programming sweeps every state in turn, as if all mattered equally. In a large problem most do not: a chess position with nine queens on the board can be legal and still never arise in a sensible game. Computation spent on such states is wasted, and two ideas steer it elsewhere.

- **Trajectory sampling**: instead of sweeping, imagine whole episodes with the model and the current policy, and update the states and actions along them. States then get attention in proportion to how often the current policy actually reaches them. Early on this is a large gain, since effort goes where play goes; later it can starve rarely visited states whose values are still wrong.
- **Real-time dynamic programming** (Barto, Bradtke & Singh, 1995) applies the same idea to value iteration: full expected updates, but only on the states that real or imagined trajectories pass through. On many problems it reaches a policy that is optimal wherever it matters, without ever touching most of the state space.

Within a fixed set of states, the order of the updates matters too: updates that change nothing waste computation. [[prioritized-sweeping]] works backward from the states whose values just changed.

### Planning at decision time {#decision-time}

The planning above runs in the background: it improves values for every state, and acting just reads them off. The opposite style waits until an action is actually needed and then thinks hard about **this state only**, usually throwing the result away once the move is made. This is **decision-time planning**, the way a chess player uses the clock.

- **Heuristic search** looks ahead from the current state through a tree of possible continuations, evaluates the leaves with a value function, and backs the values up to choose an action. Deeper search compensates for an imperfect value function.
- **Rollout algorithms** estimate each action's value by averaging the returns of many simulated episodes that start with that action and then follow a fixed rollout policy; acting greedily on those estimates improves on the rollout policy (Tesauro & Galperin, 1997).
- **Monte Carlo tree search** grows a search tree selectively, using rollouts to evaluate new nodes and statistics in the tree to decide where to look next. It is at the heart of game-playing programs such as AlphaGo and AlphaZero ([[mcts]]).

Which style fits depends on time and repetition. A robot that must react within milliseconds, or that meets the same situations again and again, gains from values prepared in advance. A game program with seconds per move, facing positions it will never see twice, gains from spending all of its effort on the one position in front of it. The strongest systems combine both: values learned in the background guide and cut short the search done at decision time.

### Historical remarks {#history}

The idea that an agent can carry a small-scale model of the world in its head, and use it to try out alternatives before acting, goes back to Craik (1943). The view of planning as learning from simulated experience, and its integration with acting in the Dyna architecture, is due to Sutton (1990). Real-time dynamic programming is due to Barto, Bradtke and Singh (1995); rollout algorithms in this form to Tesauro and Galperin (1997).

## Card

### Idea

A model answers “what would happen if I did this here?”. Planning is learning from those answers instead of from the world: the same updates as learning, fed with imagined experience. A learned model turns every real step into many imagined ones.

::: analogy
A chess player thinking before moving: the rules of the game are the model, and the moves tried in the head are simulated experience. Nothing on the board changes until the decision is made.
:::

### Two kinds of model {#kinds}

| | gives | needed by |
| --- | --- | --- |
| distribution model | every next state and reward, with its probability | dynamic programming, expected updates |
| sample model | one next state and reward, drawn at random | rollouts, sample updates, Dyna |

### Planning as learning {#formula}

$$\text{model} \;\to\; \text{simulated experience} \;\to\; \text{values} \;\to\; \text{policy}$$

Real experience can improve the model (model learning) and the values directly (direct RL); simulated experience improves the values (planning). See [[dyna-q]].

### Why it matters {#why}

- Real experience can be expensive or slow; computation is often cheap. A model turns each real step into many updates. [See it](lab:dyna-maze)
- Sample updates usually beat expected updates for the same computation when the branching factor is large.
- Focusing planning (trajectory sampling, [[prioritized-sweeping]], search from the current state) beats uniform sweeps.

### Pitfalls

- Trusting a wrong model: planning amplifies its errors ([[dyna-q-plus]]).
- Planning on states that never matter: uniform sweeps waste most of their effort in large problems.
- Long imagined trajectories from a learned model: its errors compound with every step ([[model-based-deep]]).

### Check yourself {#check}

::: question
What is the difference between a distribution model and a sample model, and which one does dynamic programming need?
---
A distribution model gives every possible outcome with its probability; a sample model gives one outcome drawn at random. Dynamic programming needs a distribution model, because its updates average over all outcomes.
:::

::: question
How does random-sample Q-planning differ from Q-learning?
---
Only in where its transitions come from: a model queried at random state–action pairs instead of the real environment. The update is the same.
:::

::: question
With a branching factor of 1000, why might sample updates beat expected updates for the same computation?
---
One expected update costs 1000 computations. In the same budget, 1000 sample updates can each be spent on a different state, and even a hundred samples on one state already remove about 90% of its error.
:::
