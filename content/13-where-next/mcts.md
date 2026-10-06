+++
summary = "When the rules of the world are known, an agent can think before it acts: grow a tree of possible futures from the current state, spending its effort on the moves that look most promising. Monte Carlo tree search does this with bandit-style exploration at every node. AlphaZero adds a network that suggests moves and judges positions, and trains that network on the results of its own searches."
prereqs = ["models", "ucb", "mc-control"]
sources = [
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., sections 8.8–8.11 and 16.6", url = "http://incompleteideas.net/book/the-book-2nd.html" },
  { text = "Kocsis & Szepesvári (2006), Bandit based Monte-Carlo planning, ECML", url = "https://link.springer.com/chapter/10.1007/11871842_29" },
  { text = "Silver et al. (2016), Mastering the game of Go with deep neural networks and tree search, Nature 529", url = "https://www.nature.com/articles/nature16961" },
  { text = "Silver et al. (2017), Mastering the game of Go without human knowledge, Nature 550", url = "https://www.nature.com/articles/nature24270" },
  { text = "Silver et al. (2018), A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play, Science 362", url = "https://www.science.org/doi/10.1126/science.aar6404" },
  { text = "Mankowitz et al. (2023), Faster sorting algorithms discovered using deep reinforcement learning, Nature 618", url = "https://www.nature.com/articles/s41586-023-06004-9" },
  { text = "Schrittwieser et al. (2020), Mastering Atari, Go, chess and shogi by planning with a learned model, Nature 588", url = "https://www.nature.com/articles/s41586-020-03051-4" },
]
+++

## Textbook

### Planning at decision time {#decision-time}

The methods of this guide mostly learn a policy or values in advance, then act by looking them up. With a model of the world, a simulator or the rules of a game, an agent can also plan *at decision time*: from the state it is in, imagine what each move would lead to, and choose with that look-ahead ([[models]]). In chess or Go the possible futures branch so fast that no search can follow them all; the question is where to spend a limited budget of simulated moves.

### Monte Carlo tree search {#mcts}

MCTS grows a tree rooted at the current state, one simulation at a time. Each node keeps, for each move, a visit count $N(s, a)$ and an average result $\val{Q(s, a)}$. A simulation has four phases:

1. **Selection.** From the root, repeatedly pick a move by a bandit rule that balances the moves with good averages against the moves tried least ([[ucb]]), until reaching a node not yet fully expanded.
2. **Expansion.** Add one or more children of that node to the tree.
3. **Evaluation.** Estimate the new node's value: classically, by playing the game out with a fast random or simple policy (a *rollout*) and noting who won.
4. **Backup.** Pass the result up the path, updating each node's count and average.

After the simulations, play the move visited most at the root, and keep the subtree below it for the next decision. With the bandit rule UCT, $\val{Q(s,a)} + c\sqrt{\ln N(s) / N(s, a)}$ (Kocsis and Szepesvári, 2006), the search concentrates on promising lines while still checking the others, and with enough simulations its choice converges to the best move. It needs no evaluation function, only the ability to simulate, which made it the method that first brought computer Go to strong amateur level.

### AlphaGo and AlphaZero {#alphazero}

Rollouts with a weak policy give noisy evaluations, and uniform priors waste simulations on obviously bad moves. AlphaZero (Silver et al., 2018) replaces both with one network that reads a position $s$ and outputs move probabilities $\mathbf p$ and a value $v$:

- **Selection** uses the network's probabilities as priors (PUCT): pick the move maximizing $\val{Q(s,a)} + c\,P(s,a)\sqrt{N(s)} / \big(1 + N(s,a)\big)$. Moves the network likes get explored first.
- **Evaluation** is the network's $v$ at the new leaf: no rollouts.

The network learns from self-play. In each position of each game, the search's visit counts at the root form a distribution $\boldsymbol\pi$, sharper and stronger than the network's own $\mathbf p$, and the game's final result $z$ is known at the end. The network is trained to predict both:
$$L = (z - v)^2 - \boldsymbol\pi^{\top}\ln \mathbf p + c\,\|\boldsymbol\theta\|^2.$$
The search improves the policy, and the network learns to imitate the improved policy, which makes the next searches stronger: a loop of policy evaluation and improvement, generalized policy iteration with a search as the improvement step ([[gpi]]).

The line of systems: AlphaGo (Silver et al., 2016) learned first from human games, then from self-play, and beat Lee Sedol 4–1 in 2016. AlphaGo Zero (2017) started from random play with no human data and surpassed it. AlphaZero (2018) used the same algorithm for chess, shogi and Go, and beat the strongest programs of each. MuZero (Schrittwieser et al., 2020) dropped the need for the rules: it learns a model that predicts rewards, values and policies a few steps ahead, and searches inside that model, mastering Atari games as well as the board games ([[model-based-deep]]).

### Why search helps {#why}

A network's evaluation of a position is a fast intuition, with errors. Search corrects it by looking at what actually follows: errors near the root matter less when the leaves are many moves ahead. Training the network on search results then pulls the intuition toward what search found. The same pairing, a learned policy that proposes and a search that verifies, has been used beyond games, for example to discover faster sorting routines (Mankowitz et al., 2023).

### Limits {#limits}

MCTS needs a model it can simulate many times per decision, and most of its successes are in deterministic, fully observed games with clear outcomes. Chance and hidden information (cards, dice) need extensions, and continuous actions need sampling-based variants. Each decision costs many network evaluations, so it is slow to act compared with a single forward pass.

## Card

### Idea

With a model of the world, search before acting: grow a tree of futures from the current state, choosing which branch to explore with a bandit rule, and back the results up. AlphaZero guides the search with a network's move priors and position values, and trains the network on the search's own choices and the games' outcomes.

::: analogy
A chess player who considers a handful of plausible moves, reads further into the most promising ones, and after the game compares their first instincts with what the analysis found.
:::

### The loop {#loop}

- Search: selection by a bandit rule, expansion, evaluation, backup.
- Play the move the search visited most.
- Train the network: policy toward the visit counts, value toward the game's result.
- Search again, now with the better network.

### Pitfalls

- Treating search as free: every decision needs many simulations.
- Searching with a wrong model: the plan is only as good as the simulator.
- Forgetting exploration in self-play: without noise at the root, the games repeat themselves.

### Check yourself {#check}

::: question
Why does MCTS use a bandit rule at each node instead of always following the best average?
---
The averages are estimates from few simulations. Always taking the best one would stop checking the others, and a move that was unlucky early would never get a second look. A bandit rule like UCT keeps trying the moves that are uncertain, as UCB does for arms.
:::

::: question
What does AlphaZero's policy network learn to predict, and why is that a good target?
---
The distribution of the search's visit counts at the root. The search, guided by the network, is stronger than the network alone, so imitating its choices improves the network, and the improved network makes the next search stronger.
:::

::: question
What did MuZero change compared with AlphaZero?
---
It no longer needs the rules: it learns a model that predicts, from a hidden state, the rewards, values and move probabilities after each move, and searches inside that learned model. That let the same method play Atari games, whose rules it was never given.
:::
