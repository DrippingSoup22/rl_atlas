+++
summary = "When several agents learn in the same world, each one's environment includes the others, and it changes as they learn. The Markov property of the single-agent world breaks, the right answer becomes an equilibrium rather than an optimum, and credit for a shared reward is hard to assign. Self-play, centralized training with decentralized execution, and leagues of past opponents are the main tools."
prereqs = ["mdp", "mcts", "on-off-policy"]
story_in = ["mcts"]
sources = [
  { text = "Littman (1994), Markov games as a framework for multi-agent reinforcement learning, ICML" },
  { text = "Yu, Velu, Vinitsky, Gao, Wang, Bayen & Wu (2022), The surprising effectiveness of PPO in cooperative multi-agent games, NeurIPS Datasets and Benchmarks", url = "https://arxiv.org/abs/2103.01955" },
  { text = "Tesauro (1995), Temporal difference learning and TD-Gammon, Communications of the ACM 38", url = "https://dl.acm.org/doi/10.1145/203330.203343" },
  { text = "Lowe, Wu, Tamar, Harb, Abbeel & Mordatch (2017), Multi-agent actor-critic for mixed cooperative-competitive environments (MADDPG), NeurIPS", url = "https://arxiv.org/abs/1706.02275" },
  { text = "Rashid, Samvelyan, de Witt, Farquhar, Foerster & Whiteson (2018), QMIX: monotonic value function factorisation for deep multi-agent reinforcement learning, ICML", url = "https://arxiv.org/abs/1803.11485" },
  { text = "Vinyals et al. (2019), Grandmaster level in StarCraft II using multi-agent reinforcement learning, Nature 575", url = "https://www.nature.com/articles/s41586-019-1724-z" },
  { text = "Berner et al. (2019), Dota 2 with large scale deep reinforcement learning (OpenAI Five)", url = "https://arxiv.org/abs/1912.06680" },
]
+++

## Textbook

### Other learners in the world {#others}

A single agent faces a world that, however random, follows fixed rules: the same state and action lead to the same distribution of outcomes ([[mdp]]). Put several learning agents in one world and that breaks. From any one agent's view, the others are part of the environment, and they keep changing their policies. What worked yesterday may fail today because the opponent adapted. The world is no longer stationary, and the guarantees of single-agent methods no longer apply.

The formal model is a **Markov game** (or stochastic game; Littman, 1994): a state, one action per agent, a joint transition, and one reward per agent.

### Cooperation, competition, and both {#kinds}

- **Fully cooperative**: all agents share one reward, as a team of robots or the units of one player. The question is coordination: who does what.
- **Competitive (zero-sum)**: one agent's gain is the other's loss, as in chess or Go. The natural goal is a minimax policy, the best against the opponent's best reply.
- **Mixed**: partly shared, partly conflicting interests, as in negotiation, traffic, markets. The goal becomes an **equilibrium**, a set of policies where no agent gains by changing its own alone (a Nash equilibrium). There can be many, and learning may cycle instead of settling.

### Self-play {#self-play}

In a symmetric game, an agent can train against copies of itself. TD-Gammon learned backgammon close to the level of the best human players this way (Tesauro, 1995); AlphaZero mastered chess, shogi and Go ([[mcts]]). The opponent improves exactly as fast as the learner, providing a curriculum of rising difficulty. The risk is forgetting: a policy that beats its current self may lose to an older strategy it no longer remembers how to counter, and play can chase itself in circles, as in rock–paper–scissors.

**Leagues** guard against that: keep a population of past and specialized agents, and train against a mix of them. AlphaStar reached Grandmaster level in StarCraft II with a league that included agents trained specifically to exploit the main agents' weaknesses (Vinyals et al., 2019). OpenAI Five learned Dota 2, a five-against-five team game, by self-play at enormous scale, and beat the reigning world champions in 2019 (Berner et al., 2019).

### Learning to cooperate {#cooperate}

Two problems dominate teams:

- **Credit assignment.** With one shared reward, which agent's action earned it? An agent that did nothing useful still sees the team's reward and may learn to keep doing nothing.
- **Partial observability.** Each agent sees only its own surroundings, but the right action depends on what the others see and do.

The common answer is **centralized training with decentralized execution**: during training, a critic may see everything, all observations and all actions, while each agent's policy uses only its own observations, so it can act alone afterward. MADDPG (Lowe et al., 2017) gives each agent a DDPG actor and a critic over everyone's actions ([[ddpg]]). QMIX (Rashid et al., 2018) learns per-agent values combined by a mixing network into a team value, constrained so that each agent maximizing its own value also maximizes the team's.

### Simpler options first {#simple}

Treating the other agents as part of the environment, each learning independently (independent Q-learning, independent PPO), ignores the non-stationarity but often works surprisingly well, especially with enough experience replayed recently or on-policy data that tracks the others' current behavior. It is the baseline every multi-agent method must beat, and a hard one: PPO with a value function that sees the whole team, and otherwise little change, matched or beat the specialized methods on standard cooperative benchmarks (Yu et al., 2022).

## Card

### Idea

With several learners in one world, each one's environment includes the others, and it changes as they learn. Goals become equilibria instead of optima. Self-play, leagues of past opponents, and centralized training with decentralized execution are the main tools.

::: analogy
Learning to play poker against friends who are also learning: as soon as you figure out one friend's bluff, they change it. The best you can aim for is a way of playing that holds up against however they adapt.
:::

### The settings {#settings}

- Cooperative: shared reward; coordination and credit assignment.
- Competitive: zero-sum; minimax, self-play.
- Mixed: equilibria, possibly many, possibly cycling.

### Pitfalls

- Assuming the others stand still: the environment is non-stationary.
- Self-play that forgets old strategies and cycles; keep a league of past opponents.
- A shared reward without credit assignment: lazy agents ride along.

### Check yourself {#check}

::: question
Why is the world non-stationary from one agent's point of view when the others are learning?
---
The others are part of its environment, and their policies change as they learn. The same state and action now lead to different outcomes than before, so the agent's experience goes stale.
:::

::: question
What does "centralized training, decentralized execution" mean, and why is it useful?
---
During training a critic can use everyone's observations and actions, which makes the others' behavior part of the critic's input instead of unexplained noise. At execution each agent acts only on its own observations, so it does not need the others' information once deployed.
:::

::: question
Why did AlphaStar train against a league instead of only its latest self?
---
Pure self-play can forget strategies it once beat and cycle between them. A league keeps old and specialized opponents, including agents trained to exploit the main agent's weaknesses, so the main agent must stay robust against all of them.
:::
