+++
summary = "An agent only learns about what it tries. Exploration strategies decide what to try beyond the current best guess: random dithering (ε-greedy, softmax, action noise), optimism about the unknown (optimistic starts, UCB, count bonuses), and randomness kept inside the policy (entropy bonuses, sampling from what it believes). They differ most on problems where a good discovery needs many deliberate steps."
prereqs = ["explore-exploit", "epsilon-greedy"]
lab = "mountain-car-optimism"
sources = [
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., chapter 2 and section 8.3", url = "http://incompleteideas.net/book/the-book-2nd.html" },
  { text = "Lai & Robbins (1985), Asymptotically efficient adaptive allocation rules, Advances in Applied Mathematics 6", url = "https://doi.org/10.1016/0196-8858(85)90002-8" },
  { text = "Auer, Cesa-Bianchi & Fischer (2002), Finite-time analysis of the multiarmed bandit problem, Machine Learning 47", url = "https://link.springer.com/article/10.1023/A:1013689704352" },
  { text = "Osband, Blundell, Pritzel & Van Roy (2016), Deep exploration via bootstrapped DQN, NeurIPS", url = "https://arxiv.org/abs/1602.04621" },
  { text = "Bellemare, Srinivasan, Ostrovski, Schaul, Saxton & Munos (2016), Unifying count-based exploration and intrinsic motivation, NeurIPS", url = "https://arxiv.org/abs/1606.01868" },
  { text = "Burda, Edwards, Storkey & Klimov (2019), Exploration by random network distillation, ICLR", url = "https://arxiv.org/abs/1810.12894" },
  { text = "Fortunato et al. (2018), Noisy networks for exploration, ICLR", url = "https://arxiv.org/abs/1706.10295" },
]

[story]
scene = "grid"
env = "two-gems"
seed = 3
average = 40

[story.runs]
dither = { algorithm = "q-learning", alpha = 0.5, epsilon = 0.1, gamma = 0.95, maxSteps = 500, units = 500, name = "ε-greedy, ε = 0.1" }
dither3 = { algorithm = "q-learning", alpha = 0.5, epsilon = 0.3, gamma = 0.95, maxSteps = 500, units = 500, name = "ε-greedy, ε = 0.3" }
optimism = { algorithm = "q-learning", alpha = 0.5, epsilon = 0.0, q0 = 1.0, gamma = 0.95, maxSteps = 500, units = 500, name = "optimistic start, Q = 1" }
+++

## Story

::: step {run = "dither", at = 0}
**An agent only learns about what it tries.** The two gems again: a small one two steps from the start pays $\rew{0.3}$, a big one seven steps away pays $\rew{1}$. Even discounted by $\gam = 0.95$ for the walk, the big gem is worth more than twice as much. A Q-learner starts with every value at 0 and has to find that out.
:::

::: step {run = "dither", at = 1}
**Exploring by chance.** This learner is ε-greedy: one move in ten at random, the rest the move that looks best. At first every move looks the same, so its first episode is a random walk, 73 steps long, and it happens to end on the big gem. One value learns from it: the last move, into the gem. The news has six more steps to travel back to the start.
:::

::: step {run = "dither", at = 5, path = true}
The small gem is two steps from the start, so random walks find it far more often, and its news reaches the start first. Once a move toward it is worth more than 0, the greedy choice takes it nine times in ten. After 5 episodes, the greedy path leads to the small gem.
:::

::: step {run = "dither", at = 500, path = true}
**After 500 episodes**, it still does. This run reached the big gem four times, in episodes 1, 3, 34 and 51, each time at the end of a string of random moves, and each visit carried the news a little further back. After 500 episodes, it has traveled two steps from the gem. Of 40 runs, 24 touch the big gem at least once, and none ends up going for it. More randomness hardly helps: with $\eps = 0.3$, 4 of 40 do.
:::

::: step {run = "optimism", at = 1, arrows = false}
**Exploring on purpose.** The same learner with no random moves at all, $\eps = 0$, but every value starts at 1, the most any move can be worth here: blue everywhere. A move that has been tried falls toward what it actually earned, so the untried moves look best, and the greedy choice tries them one after another. Its first episode ends at the small gem after 12 steps. Each move it tried dropped a little below 1, and the move into the small gem fell to 0.65: from now on, any move not yet tried looks better.
:::

::: step {run = "optimism", at = 3, arrows = false}
The small gem's 0.3 is far less than the 1 promised everywhere else, so the agent keeps looking. In its third episode, after 62 steps through tiles it had not seen, it reaches the big gem. From the seventh episode on, every episode ends there.
:::

::: step {run = "optimism", at = 200, path = true}
**After 200 episodes** the values have settled at what the moves are really worth: $\val{0.74}$ at the start, the big gem's 1 discounted for six moves. Since its 110th episode, every episode has taken the seven steps to the big gem. All 40 runs reach the big gem within five episodes, and from the tenth on, every episode of every run ends there.
:::

::: step {run = "optimism", at = 500, curves = ["dither", "dither3", "optimism"], metric = "return"}
**The return per episode, averaged over 40 runs.** Random moves spread exploration thinly everywhere, so a discovery several deliberate steps deep needs luck, and the news needs more luck to come back. Optimism sends the agent where it has not been. In a table, “not tried yet” is easy to know; in a large world, the methods below estimate it with counts, prediction errors or uncertainty. [Race them in the Lab](lab:gems-optimism).
:::

## Textbook

### Why exploring is hard {#why}

The trade between exploiting what looks best and trying something else ([[explore-exploit]]) has no free answer. Every try of a worse-looking option costs something now, and its payoff, learning that it was better after all, comes later and only sometimes. A method that never pays that cost can lock onto the first thing that works, as the guide's experiments show again and again:

- On the **two gems**, A2C without an entropy bonus settles on the small gem in all 20 of the Lab's runs ([[entropy-bonus]]).
- In **Mountain Car**, SARSA whose values start at $-150$ is still stuck in the valley at the end of all 20 runs. Starting at 0, all 20 end up getting out in under 300 steps ([[semi-gradient-sarsa]]).
- In the **shortcut maze**, Dyna-Q never finds the shortcut that opens halfway through, in 20 runs of 20 ([[dyna-q-plus]]).

Strategies differ in what they add to a greedy choice: noise, optimism, or uncertainty.

### Random dithering {#dithering}

The simplest strategies add randomness to the greedy choice, without regard to what has been tried:

- **ε-greedy** ([[epsilon-greedy]]): with probability $\eps$, a uniformly random action. On the 10-armed testbed, over 500 runs, the fraction of runs that end up pulling the best arm most of the time grows from 39% with $\eps = 0$ to 61% with $\eps = 0.01$ and 85% with $\eps = 0.1$. DQN decays $\eps$ from 1 to a small value, exploring most when it knows least ([[dqn]]).
- **Softmax (Boltzmann)**: actions chosen with probabilities $\propto e^{Q(s,a)/\tau}$, so a nearly-best action is tried far more often than a terrible one. The temperature $\tau$ plays the role of $\eps$.
- **Action noise** for continuous actions: Gaussian noise added to a deterministic action, as in [[ddpg]] and [[td3]].

Dithering is cheap and works when good actions are a few random steps away. It fails when a discovery needs a long, specific sequence. In a corridor of $N$ steps where only the far end pays and every wrong step sends the agent back, uniformly random moves reach the end with probability $2^{-N}$: about once in a million tries for $N = 20$. Exploration that reaches such places has to be *deep*: committed to a direction for many steps (Osband et al., 2016).

### Optimism {#optimism}

Optimism in the face of uncertainty treats what has not been tried as possibly great, so the agent goes to check, and keeps going until experience says otherwise. Unlike dithering, it is directed: it prefers what it knows least.

- **Optimistic initial values** ([[optimistic-init]]). Start every estimate above anything achievable: each action tried disappoints, so the agent moves on to the others. On the testbed, greedy with $Q_1 = 5$ ends up pulling the best arm most of the time in 89% of 500 runs. Mountain Car shows the same effect with features: values starting at 0, above every true value, drive the car out of the valley in every run.
- **Upper confidence bounds** ([[ucb]]). Add a bonus that shrinks with the number of tries: $\val{Q_t(a)} + c\sqrt{\ln t / N_t(a)}$. Over 500 runs on the testbed, UCB with $c = 2$ ends well in 89%, against 85% for ε-greedy with $\eps = 0.1$.
- **Bonuses for the unvisited**. Dyna-Q+ adds $\kappa\sqrt{\tau}$ to the reward of a transition not tried for $\tau$ steps. In the shortcut maze, all 20 runs find the shortcut ([[dyna-q-plus]]). With large state spaces, counts are replaced by measures of novelty: pseudo-counts from a density model (Bellemare et al., 2016), or the error of a network trained to predict a fixed random network's output, large on unfamiliar states (random network distillation; Burda et al., 2019).

### Randomness inside the policy {#policy}

Policy-gradient methods sample their actions from the policy, so they explore by construction, but a policy gradient also makes the policy more decisive with every update. Left alone, it can become nearly deterministic before it has found the best option.

- **An entropy bonus** pays the policy for staying uncertain ([[entropy-bonus]]). On the two gems, $\beta = 0.1$ finds the big gem in all 20 of the Lab's runs within 240 rounds, against none without a bonus. Too large a bonus keeps the policy random forever ([[hyperparameters]]).
- **Maximum entropy RL** makes that bonus part of the objective, with a weight tuned automatically, as [[sac]] does.
- **Sampling a belief.** Thompson sampling keeps a distribution over what each action is worth, samples one plausible world, and acts greedily in it. Each sample commits to one hypothesis for a while, which gives deep exploration. Bootstrapped DQN does this with several value heads trained on different subsets of the data, acting with one head per episode (Osband et al., 2016). Noisy networks put learned noise in the weights themselves (Fortunato et al., 2018).

### Choosing {#choosing}

| situation | a good first choice |
| --- | --- |
| few actions, rewards found by luck quickly | ε-greedy or softmax, decayed over time |
| bandits, short horizons | UCB or optimistic starts |
| values from features, rewards far away | optimistic initial values |
| policy gradients | an entropy bonus, tuned; or SAC |
| continuous actions, off-policy | Gaussian action noise (DDPG, TD3), or SAC |
| sparse rewards in a huge state space | novelty bonuses (counts, RND) or posterior sampling |
| the world changes | bonuses for what has not been tried lately (Dyna-Q+) |

### Historical remarks {#history}

ε-greedy and softmax selection are as old as reinforcement learning itself. UCB comes from the theory of bandits: Lai and Robbins (1985) showed how fast regret must grow, and Auer, Cesa-Bianchi and Fischer (2002) gave UCB1, which achieves that rate up to a constant. Thompson's sampling rule dates from 1933. In deep RL, Atari's Montezuma's Revenge, where rewards are rare and far apart, became the benchmark for exploration: pseudo-counts (Bellemare et al., 2016) made large progress there, and random network distillation (Burda et al., 2019) was the first method to beat the average human score without demonstrations or access to the game's internal state.

## Card

### Idea

An agent only learns about what it tries. Explore by adding noise to the greedy choice (cheap, undirected), by being optimistic about the unknown (directed: it goes to what it knows least), or by keeping the policy itself uncertain (entropy bonuses, sampling a belief).

::: analogy
Choosing restaurants in a new city. Dithering: now and then, walk into a random one. Optimism: assume every untried place might be the best in town, and go check. Sampling a belief: pick a theory, "the best food is near the station", and follow it for a week.
:::

### The families {#families}

- **Dithering**: ε-greedy, softmax, Gaussian action noise. Easy; shallow.
- **Optimism**: optimistic starts, UCB, novelty bonuses. Directed; needs a sense of what is unknown.
- **Policy randomness**: entropy bonus, SAC, Thompson sampling, bootstrapped heads, noisy nets.

### Pitfalls

- Turning exploration off too early: a greedy policy cannot correct what it never tries.
- Expecting dithering to find a reward that takes a long, specific sequence of moves.
- Pessimistic starting values with greedy action choice: the agent stops looking.
- Too large an entropy bonus or ε: exploring forever instead of using what was learned.

### Check yourself {#check}

::: question
Why do optimistic initial values make even a greedy agent explore?
---
Every action starts out looking better than anything achievable. Each one tried returns less than promised, so its estimate drops below the untried ones, and the greedy choice moves on, until every action has been tried enough to be judged on its merits.
:::

::: question
In a corridor of 20 steps where only the far end pays and a wrong step sends you back, why does ε-greedy struggle?
---
With uniform random moves, the chance of 20 right steps in a row is 2⁻²⁰, about one in a million. Dithering explores near the start; it does not commit to a direction for long. Directed methods (optimism, novelty bonuses, posterior sampling) push the agent toward the unvisited end.
:::

::: question
Mountain Car with values starting at −150 ends every run stuck in the valley. Why does starting at 0 fix it?
---
Every step pays −1, so every true value is below 0. Starting at 0 makes every untried state and action look better than what has been tried, so the agent keeps pushing into new territory until it finds the way out. Starting at −150 makes the familiar valley look as good as anything, and nothing pulls the agent elsewhere.
:::
