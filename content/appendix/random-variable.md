+++
title = "Random variables"
summary = "A number whose value is decided by chance, such as the next reward or the length of an episode. Capital letters name them; lower-case letters name the values they can take."
group = "Probability"
sources = [
  { text = "Blitzstein & Hwang (2019), Introduction to Probability, 2nd ed., CRC Press, chapter 3", url = "https://projects.iq.harvard.edu/stat110" },
]
+++

A **random variable** is a quantity whose value is not fixed in advance but decided by chance: the number rolled on a die, the reward after the next action, the number of steps before an episode ends. Before the experiment it has many possible values, each with a probability; after it, it has one.

### Capital and lower-case letters {#letters}

The atlas follows a convention common in probability. A **capital letter** names the random variable, the quantity that is not yet known; a **lower-case letter** names one particular value it might take.

- $S_t$ is the state at time $t$, random, since it depends on everything that happened before; $s$ is a specific state, such as “battery low”.
- $\rew{R_{t+1}}$ is the next reward, random; $\rew{r}$ is a possible value of it, such as $\rew{+2}$.
- $\Pr\{S_{t+1} = s'\}$ is the probability that the random next state turns out to be the particular state $s'$.

So in $\val{v_\pi(s)} = \mathbb{E}_\pi[\rew{G_t} \mid S_t = s]$, the capitals are things that chance decides, and the lower-case $s$ is the state we chose to ask about ([[value-functions]]). Estimates follow the same rule: $\val{V(s)}$ is the number stored for state $s$, and $\val{V(S_t)}$ is the number stored for whatever state the agent happens to be in at time $t$, itself random.

### Distribution {#distribution}

A random variable is described by its **distribution**: the probability of each value it can take ([[a:probability|probability]]). For the reward of searching with a low battery in the recycling robot ([[mdp]]),

| value $\rew{r}$ | $\rew{+2}$ | $\rew{-3}$ |
| --- | --- | --- |
| $\Pr\{\rew{R_{t+1}} = \rew{r}\}$ | 0.6 | 0.4 |

Two summaries of a distribution are used constantly: its average, the **expected value** ([[a:expectation|expectation]]), here $0.6 \times 2 + 0.4 \times (-3) = 0$; and its spread, the **variance** ([[a:variance|variance]]).

### Random variables built from others {#functions}

Anything computed from random variables is itself a random variable. The return $\rew{G_t} = \rew{R_{t+1}} + \gam\,\rew{R_{t+2}} + \cdots$ is a sum of random rewards, so it is random too ([[return]]). The time $T$ at which an episode ends depends on the states visited, so it is random as well ([[episodes]]). The atlas's methods mostly study averages of such quantities, because averages, unlike the quantities themselves, are fixed numbers that can be compared.

### Discrete and continuous {#kinds}

A random variable with a finite (or countable) list of possible values is **discrete**, like a state of a gridworld or a die roll. One that can take any value in an interval is **continuous**, like a torque or a position. For continuous variables, single values have probability 0, and probabilities come from a density instead ([[a:probability-density|densities]]).
