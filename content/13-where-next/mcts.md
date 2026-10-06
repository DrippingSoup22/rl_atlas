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
[story]
scene = "tree"
board = "O....X..."
c = 1.4
seed = 1
formula = '\step{1}{\text{back up: } N(s,a) \leftarrow N(s,a) + 1,\ \ \val{Q(s,a)} \leftarrow \text{average of the results}} \step{2}{\qquad \text{select: } \arg\max_a \Big[\val{Q(s,a)} + c\sqrt{\ln N(s) / N(s,a)}\,\Big]}'
+++

## Story

::: step {sims = 0}
**X to move.** O holds a corner, X the middle of the right edge, and seven cells are free. One of them wins by force; the others draw or lose against an opponent who does not blunder. The agent knows the rules, so it can try moves in its head before playing one. Which move, and how should it spend its thinking?
:::

::: step {sims = 0, flat = 3000}
The simplest plan: for each move, play 3000 games to the end at random and keep the average. **The center wins on average** (68%), the top-right corner comes second (65%). But the center only draws against careful play. Random games reward moves that a random opponent fumbles, and this opponent will not fumble. More random games only make the wrong answer more certain.
:::

::: step {sims = 0, play = 7, pace = 1300, formula = 1}
Monte Carlo tree search keeps a tree of the positions it has thought about. The first seven simulations each add one move to it, then finish the game at random (the faint moves), and write the result into the new node: its visit count and its average. One game per move says almost nothing yet.
:::

::: step {sims = 50, formula = 2}
From the eighth simulation on, every first move has been tried, so the search must choose which line to follow. At each node it picks by a bandit rule, as [[ucb]] picks an arm: the best average plus a bonus for moves tried rarely. After 50 simulations the tree is three moves deep, and the center leads with 11 visits: so far the search agrees with the random games.
:::

::: step {sims = 300, play = 3, pace = 1600, formula = 2}
Each simulation now walks down the tree by the bandit rule (the red path), adds one node at its end, and only then plays at random. Inside the tree the replies are not random any more: at O's nodes the rule picks what is best **for O**. After 300 simulations the center still leads, 77 visits to the corner's 40.
:::

::: step {sims = 1000, mark = 2}
After about 980 simulations the corner takes the lead for good. Under it, the tree has found O's only defense, blocking at the bottom right, and then X's reply in the center, which threatens two lines at once. The random games could not see that line: it needs O to answer correctly and X to follow up.
:::

::: step {sims = 2000, mark = 2}
After 2000 simulations, the corner has 1249 of the visits and wins 83% of its games; the center has stalled at 257 visits and 66%. The search spends its effort where the decision is made and checks the rest just enough. It plays the most visited move: the winning corner.
:::

::: step {sims = 2000, chart = true}
The same question asked of 100 searches, each with its own seed. With 1000 simulations, 89 of them choose the winning corner, and with 3000, all of them. Flat Monte Carlo with the same number of random games gets worse as it plays more: 3 of 100 at 3000. A search that models the opponent's best replies beats a pile of random games, and AlphaZero's networks make each simulation smarter still.
:::

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

On the tic-tac-toe position of the story (X to move, O in a corner, X on an edge), playing each move out at random ranks the center first, though only the top-right corner wins against best play: after it, O must block, and X's center then threatens two lines. The tree finds that line because inside the tree each side picks its own best reply.

::: figure {#fig-odds}
{{mcts-odds}}
How often a search from the story's position chooses the winning corner, against the number of simulations. Flat Monte Carlo spreads the same number of random games over the seven moves and grows more confident in the center; UCT's tree finds the corner after about a thousand simulations. Computed live from 100 seeded searches.
:::

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
