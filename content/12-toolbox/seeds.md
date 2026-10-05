+++
summary = "A seed fixes every random choice of a run: the moves it explores, the weights it starts from, what the world throws at it. Change the seed and the same settings can end somewhere else entirely, because what an agent learns decides what it sees next. Judge settings by many seeds, show the spread and not just the average, and compare settings on the same seeds."
prereqs = ["hyperparameters", "explore-exploit"]
lab = "seeds-gems"
sources = [
  { text = "Henderson, Islam, Bachman, Pineau, Precup & Meger (2018), Deep reinforcement learning that matters, AAAI", url = "https://arxiv.org/abs/1709.06560" },
  { text = "Colas, Sigaud & Oudeyer (2018), How many random seeds? Statistical power analysis in deep reinforcement learning experiments", url = "https://arxiv.org/abs/1806.08295" },
  { text = "Agarwal, Schwarzer, Castro, Courville & Bellemare (2021), Deep reinforcement learning at the edge of the statistical precipice, NeurIPS", url = "https://arxiv.org/abs/2108.13264" },
  { text = "Patterson, Neumann, White & White (2024), Empirical design in reinforcement learning, Journal of Machine Learning Research 25", url = "https://arxiv.org/abs/2304.01315" },
]

[story]
scene = "grid"
env = "two-gems"
seed = 19
average = 40

[story.runs]
small = { algorithm = "a2c", alpha = 2.0, alphaW = 0.3, workers = 4, n = 5, beta = 0.05, gamma = 0.95, maxSteps = 500, units = 200, seed = 19, name = "seed 19" }
big = { algorithm = "a2c", alpha = 2.0, alphaW = 0.3, workers = 4, n = 5, beta = 0.05, gamma = 0.95, maxSteps = 500, units = 200, seed = 22, name = "seed 22" }
mean = { algorithm = "a2c", alpha = 2.0, alphaW = 0.3, workers = 4, n = 5, beta = 0.05, gamma = 0.95, maxSteps = 500, units = 200, name = "β = 0.05, the average of 40 seeds" }
+++

## Story

::: step {run = "small", at = 0}
**One setting, two seeds.** The two gems of the [[hyperparameters|hyperparameter guide]]: a small one two steps from the start pays $\rew{0.3}$, a big one seven steps away pays $\rew{1}$. Four A2C workers learn with an entropy weight $\beta = 0.05$, halfway between too little and enough. Below are two runs with exactly the same settings. Only the seed differs, and the seed decides every random move the workers make.
:::

::: step {run = "small", at = 200, values = true}
**Seed 19.** In round 5 a worker reaches the big gem, but the small one is found far more often, and every visit makes the moves toward it likelier. The small gem wins. Over the last 20 rounds the episodes take 5 steps and earn $\rew{0.37}$. From the start, 70% of the policy points up or left, toward the small gem.
:::

::: step {run = "big", at = 80}
**Seed 22.** The same settings. Here a worker reaches the big gem in the very first round, and the next few finds come early enough to outweigh the small gem's pull. By rounds 61 to 80 the episodes already earn $\rew{0.81}$.
:::

::: step {run = "big", at = 200, values = true}
After 200 rounds, 97% of the policy at the start points right or down, toward the big gem. The episodes take 10.5 steps and earn $\rew{0.98}$. Same algorithm, same knobs, same world: a run that earns almost three times as much.
:::

::: step {run = "big", at = 200, curves = ["mean"], metric = "return"}
**The average of 40 seeds** climbs smoothly to $\rew{0.66}$. No run behaves like that. Of the 40 runs behind the line, 18 end below $\rew{0.4}$, on the small gem, and 18 above $\rew{0.9}$, on the big one; the last 4 are still switching. No run ends between $\rew{0.42}$ and $\rew{0.79}$, the band the average lands in. When outcomes split, the average describes none of them. Count them instead: here, 18 of 40 runs go for the big gem. [See ten seeds side by side in the Lab](lab:seeds-gems).
:::

## Textbook

### What a seed decides {#what}

Every random choice in a run comes from a pseudo-random number generator, and the seed is where that generator starts. It decides the moves an ε-greedy or stochastic policy tries, the starting weights of a network, which transitions a replay memory hands back, and everything random in the world itself: slippery tiles, dealt cards, starting positions. With the same seed and the same code, a run replays exactly. Every story in this guide relies on that: what you see is what the run did, and it can be played again. Change the seed and every random choice changes with it.

### Why runs drift apart {#drift}

In supervised learning the data is fixed, and different seeds mostly scatter the result a little around the same place. In reinforcement learning the agent collects its own data: what it has learned decides where it goes, and where it goes decides what it learns next. A small early difference, such as one worker stumbling on the big gem in round 1 rather than round 5, gets amplified by that loop. Runs then do not scatter around one result but split into different results: here one gem or the other, elsewhere a pole that is balanced or dropped, a car that leaves the valley or never does ([[semi-gradient-sarsa]]).

### What an average hides {#averages}

An average over seeds is the standard summary, and it is useful: it moves smoothly, and it estimates what to expect from one more run. But it hides how the runs are spread. On the two gems, the average of 40 runs ends at 0.66, a value no run comes near. Better summaries say how the runs are spread:

- **Thin lines.** Draw each run, faintly, under the average. A split shows at once.
- **The success rate.** Fix what counts as ending well, and count the runs that do. The Lab's odds work this way.
- **Robust averages.** The median, or the interquartile mean (the average of the middle half of the runs), is not dragged around by one run that diverged.
- **The whole distribution.** For each score, the fraction of runs that reach it: a performance profile (Agarwal et al., 2021).

### How many seeds {#how-many}

A success rate measured on $n$ runs is itself uncertain. With $k$ successes, a 95% confidence interval for the true rate is:

| runs that ended well | the true rate is likely between |
| --- | --- |
| 5 of 5 | 48% and 100% |
| 10 of 20 | 27% and 73% |
| 18 of 40 | 29% and 62% |
| 35 of 40 | 73% and 96% |

These are exact binomial (Clopper–Pearson) intervals. Five runs that all succeed are consistent with a setting that fails half the time. Comparisons need even more care. Fisher's exact test asks how likely two counts this different would be if both settings had the same true rate: for 5 of 5 against 3 of 5, the chance is 44%, so the difference proves nothing. For 5 of 5 against 0 of 5 it is under 1%. For 35 of 40 against 18 of 40, it is 0.01%.

Deep networks make seeds expensive. The recorded sweeps of Parts 9 and 10 train 5 seeds per setting on a CPU. That is enough to see a setting that fails every time, such as DQN without a target network, but not enough to rank two settings that both mostly work. The entries say how many runs each count rests on, so it can be read with this table in mind.

### Fair comparisons {#fair}

- **The same seeds for every setting.** If setting A gets seeds 1 to 20 and B gets 21 to 40, part of their difference is luck. Giving both the same seeds removes some of it, at least while the two runs still behave alike. The Lab's sweeps do this.
- **Tuning seeds are not test seeds.** Choosing settings on some seeds and then reporting those same seeds reports the luck you selected for. Tune on one set and report on fresh ones.
- **Report every run.** The best of five runs is the best of five, not what the method does. Say how many runs there were, how they were summarized, and what counts as success.

### Seeds in this guide {#guide}

Each story plays one run, on a seed chosen because it shows its point clearly. Where the point depends on luck, the story says how often it happens, and the Lab shows the odds: its tally runs 20 seeds per setting, and its sweeps give every value the same seeds. A story's seed is an example, chosen and said to be chosen. The odds are the evidence.

### Historical remarks {#history}

Henderson et al. (2018) showed that the same algorithm, with the same settings, averaged over two different sets of five seeds, gave learning curves different enough to pass a significance test. Colas, Sigaud and Oudeyer (2018) worked out how many seeds a comparison needs to detect a given difference reliably. Agarwal et al. (2021) showed that on a standard benchmark, with the few runs typical of deep RL papers, the uncertainty in published comparisons was often as large as the improvements they claimed. They proposed interval estimates, interquartile means and performance profiles as standard practice.

## Card

### Idea

The seed fixes every random choice of a run, and in RL different seeds can end in different places, because what an agent learns decides what it sees next. Judge settings by many seeds, look at how runs spread, and compare settings on the same seeds.

::: analogy
One coin toss tells you little about the coin. Ten tosses tell you a bit more, and you would still not bet much on 5 heads out of 5.
:::

### Reading many runs {#reading}

- Draw the runs, not just their average: splits show at once.
- Count the runs that end well; give the count with its total, as in "18 of 40".
- Prefer the median or the interquartile mean when a few runs diverge.
- Remember the uncertainty: 5 of 5 is consistent with a true rate of 48%.

### Fair play {#fair}

- The same seeds for every setting you compare.
- Separate seeds for tuning and for reporting.
- Every run reported, with how many there were.

### Pitfalls

- Reporting the best seed as if it were typical.
- Concluding from 5 of 5 against 3 of 5 that one setting is better.
- Trusting an average curve when the runs split into groups.
- Comparing settings that faced different seeds.

### Check yourself {#check}

::: question
Two runs with the same settings and different seeds end at 0.37 and 0.98. Is something broken?
---
Not necessarily. In RL each run collects its own data, so early luck can send runs to different solutions. On two gems with β = 0.05, 18 of 40 runs end on the big gem and 18 on the small one. The settings are not good or bad; they are good 45% of the time.
:::

::: question
Setting A succeeded in 5 of 5 runs, setting B in 3 of 5. Which is better?
---
You cannot tell yet. If both had the same true rate, counts this different would still turn up 44% of the time (Fisher's exact test). More seeds are needed, ideally the same seeds for both.
:::

::: question
The average of 40 runs ends at 0.66. Why might no single run end near 0.66?
---
Because the runs split: about half end near 0.3 and half near 1. The average of two groups lands between them, where no run is. Thin lines or a success rate show the split.
:::
