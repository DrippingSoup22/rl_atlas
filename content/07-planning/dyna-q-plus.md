+++
summary = "Dyna-Q with curiosity: in planning, every remembered move earns a small bonus that grows with the time since it was last tried for real. Stale parts of the model start to look attractive, so the agent goes back to check, and notices when the world has changed."
change = "In planning, use the reward R + κ√τ instead of R, where τ is the number of real steps since the move was last tried; moves never tried from a visited state may also be planned, as if they led back to it with reward 0."
prereqs = ["dyna-q", "explore-exploit"]
lab = "dyna-shortcut"
sources = [
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., §8.3 and Examples 8.2 and 8.3", url = "http://incompleteideas.net/book/the-book-2nd.html" },
  { text = "Sutton (1990), Integrated architectures for learning, planning, and reacting based on approximating dynamic programming, Proceedings of the 7th International Conference on Machine Learning", url = "https://doi.org/10.1016/B978-1-55860-141-3.50030-4" },
  { text = "Kaelbling (1993), Learning in Embedded Systems, MIT Press", url = "https://mitpress.mit.edu/9780262111744/" },
  { text = "Brafman & Tennenholtz (2002), R-max: a general polynomial time algorithm for near-optimal reinforcement learning, Journal of Machine Learning Research 3", url = "https://jmlr.org/papers/v3/brafman02a.html" },
]

[story]
scene = "grid"
env = "shortcut-maze"
seed = 1
average = 10
range = 0.8
formula = '\step{1}{\text{planning reward} = \rew{r} + \knob{\kappa}\sqrt{\tau}} \step{2}{\qquad \tau = \text{real steps since } (s, a) \text{ was last tried}}'

[story.runs]
q = { algorithm = "dyna-q", planning = 50, alpha = 1.0, epsilon = 0.1, gamma = 0.95, units = 400, name = "Dyna-Q" }
plus = { algorithm = "dyna-q-plus", planning = 50, kappa = 0.001, alpha = 1.0, epsilon = 0.1, gamma = 0.95, units = 400, name = "Dyna-Q+" }
+++

## Story

::: step {run = "q", at = 100}
**Dyna-Q has learned this maze**: from S at the bottom, around the left end of the wall, up to the gem. Its model knows every move along the way, and its planning keeps the values consistent with what the model says. Here it is after 100 episodes.
:::

::: step {run = "q", at = 250}
After 3000 steps, a **shortcut opens** at the right end of the wall. Nothing tells the agent. 130 episodes later, Dyna-Q still goes the long way round: its model says the right end is a wall, its plans agree, and its occasional random moves rarely get it all the way there.
:::

::: step {run = "plus", at = 250, formula = 2}
Dyna-Q+ plans with a twist: a remembered move that has not been tried for real in a long time earns a small **bonus** in planning, growing with the square root of the time since. The moves near the right end of the wall, untried for thousands of steps, start to look worth a visit. It went to check, found the gap, and now takes the shortcut.
:::

::: step {run = "plus", at = 400, curves = ["q", "plus"], metric = "steps"}
Steps per episode, averaged over 10 runs: both settle on the long way at first; after the shortcut opens, only Dyna-Q+ drops to the shorter path. The price is a little extra wandering beforehand, the checks that find nothing. [Watch it happen in the Lab](lab:dyna-shortcut), or see what happens when the way is [blocked instead](lab:dyna-blocking).
:::

## Textbook

### When the model is wrong {#idea}

[[dyna-q]] trusts its model. When the environment is stochastic and few samples have been seen, or when it changes, the model is wrong, and planning then computes a policy that is optimal for the wrong world. How bad this is depends on the direction of the error.

- **Optimistic errors correct themselves, eventually.** If the model believes in a path that no longer exists, the agent follows it, discovers the change by acting, updates the model, and replans. Finding the alternative may still take long, as the blocking maze shows below.
- **Pessimistic errors may never be corrected.** If the environment changes for the better and the model does not know it, the current policy keeps working, and the agent has no reason to go and find out.

This is the conflict between exploration and exploitation ([[explore-exploit]]) in a planning setting: to keep its model right, the agent must sometimes act in ways its model says are worse. ε-greedy exploration helps, but too slowly: the better option may require a long sequence of unlikely random moves.

### The exploration bonus {#bonus}

**Dyna-Q+** keeps track, for every state–action pair, of the number of real time steps $\tau$ since it was last tried, and adds a bonus to the reward of that pair when it is used in planning:

$$\val{Q(s, a)} \leftarrow \val{Q(s, a)} + \alp\,\Big[\,\rew{r} + \knob{\kappa}\sqrt{\tau(s, a)} + \gam \max_{a'} \val{Q(s', a')} - \val{Q(s, a)}\,\Big], \label{eq-bonus}$$

for a small constant $\kappa > 0$. The longer a move has gone untested, the more likely its model is out of date, and the more attractive it becomes. Planning spreads that attraction back along the paths that lead to it, so the agent eventually takes a real, possibly long, detour to test it. The real reward is never changed: only the imagined one.

Two more details make the bonus effective. Planning may consider actions never tried from a visited state, modeled as leading back to the same state with reward zero, so they too collect bonuses and get tried. And the direct Q-learning update from real experience uses the real reward, so the values always come back to reality when a move is tested.

::: algorithm {#alg-plus} Dyna-Q+ (changes to Dyna-Q)
Keep a clock of real steps, and for every pair the time it was last tried
On the first visit to a state, add its untried actions to the model as leading back to it with reward 0
After each real step from $S$ with $A$: record the time; update the model and $\val{Q}$ as in Dyna-Q
In each planning step on a remembered $(s, a)$ with $\rew{r}, s' \leftarrow \textit{Model}(s, a)$:
  $\tau \leftarrow$ the time since $(s, a)$ was last tried
  $\val{Q(s, a)} \leftarrow \val{Q(s, a)} + \alp\,[\rew{r} + \knob{\kappa}\sqrt{\tau} + \gam \max_{a'} \val{Q(s', a')} - \val{Q(s, a)}]$
:::

### Example: the blocking maze {#blocking}

::: example {#ex-blocking} Blocking maze (Sutton & Barto, Example 8.2)
A 6 × 9 maze with a wall across the middle and a gap at its right end. After 1000 time steps the gap moves to the left end; the path the agent has learned is blocked, and a longer one opens. The reward is $\rew{+1}$ at the goal and 0 otherwise.
:::

::: figure {#fig-blocking}
{{changing-maze blocking}}
The blocking maze before and after the change, and the cumulative reward of Dyna-Q and Dyna-Q+ against time steps (10 planning steps, $\alpha = 1$, $\varepsilon = 0.1$, $\gamma = 0.95$, $\kappa = 0.001$), averaged over 20 runs. The slope is the rate of reaching the goal. Computed by the Lab. After Sutton & Barto, Figure 8.4.
:::

Both agents find the short path in the first phase, and both stall when it is blocked: they keep bumping into the new wall. Dyna-Q+ finds the gap at the other end within a few hundred steps and its cumulative reward climbs again (\ref{fig-blocking}). Dyna-Q recovers much more slowly, and in many runs not at all within 2000 steps. Its values below the wall fade only gradually, since they are propagated from one another, and stay above the value 0 of the moves it has never made, such as stepping up through the new gap. Its greedy choices therefore lead back to the right, and only rare exploratory moves can take it through the gap. Dyna-Q+'s bonus is exactly what such untried moves need. It also explains Dyna-Q+'s faster start: its first episode is a more systematic search.

### Example: the shortcut maze {#shortcut}

::: example {#ex-shortcut} Shortcut maze (Sutton & Barto, Example 8.3)
The same maze with the gap at the left end. After 3000 time steps a second gap opens at the right end: a shorter path, while the old one stays open.
:::

::: figure {#fig-shortcut}
{{changing-maze shortcut}}
The shortcut maze before and after the change, and the cumulative reward of Dyna-Q and Dyna-Q+ (50 planning steps, otherwise as above), averaged over 20 runs. Computed by the Lab. After Sutton & Barto, Figure 8.5.
:::

Here the error is pessimistic. Dyna-Q never finds the shortcut: its model says the right end is a wall, and ε-greedy exploration practically never leads it there. Dyna-Q+ notices within a few hundred steps, and its cumulative reward climbs faster from then on (\ref{fig-shortcut}). In the first phase it does slightly worse than Dyna-Q, the cost of the checks that find nothing.

### Choosing κ, and where to put the bonus {#kappa}

The bonus must be small compared with real rewards, or the agent wanders forever; large enough that untested moves eventually win. With rewards of 1 and $\gamma = 0.95$, $\kappa = 0.001$ means that a move untested for 10 000 steps earns an extra 0.1 per planning update. Because the bonus is added to the values, it also inflates them: the greedy policy of Dyna-Q+'s values sometimes heads for untested moves rather than the goal, and its values no longer estimate returns.

An alternative is to leave the values alone and add the bonus only when choosing actions, for example acting greedily on $\val{Q(S, a)} + \kappa\sqrt{\tau(S, a)}$. This keeps the values honest but explores less effectively, since the attraction of a distant untested move is not propagated back by planning. The general idea, optimism about what is uncertain, also underlies [[ucb]] for bandits and model-based methods with guarantees such as R-max (Brafman & Tennenholtz, 2002), which treat every insufficiently tried pair as maximally rewarding.

### Historical remarks {#history}

Sutton (1990) introduced Dyna-Q+ and the blocking and shortcut mazes to show the problem of changing environments. Exploration bonuses based on the time since an action was tried, or on how often it has been tried, have a long history in reinforcement learning (Kaelbling, 1993); R-max (Brafman & Tennenholtz, 2002) turned optimism in the face of uncertainty into polynomial-time guarantees.

## Card

### Idea

Dyna-Q that gets curious about stale knowledge. In planning, every remembered move earns a bonus $\kappa\sqrt{\tau}$, growing with the time $\tau$ since it was last tried for real. Moves untested for long start to look good, the agent goes to check them, and finds out when the world has changed, for better or worse.

::: analogy
Taking a different street home now and then, just in case a new road has opened since the last time you looked.
:::

### The update {#update}

$$\val{Q(s,a)} \leftarrow \val{Q(s,a)} + \alp\,\big[\,\rew{r} + \knob{\kappa}\sqrt{\tau} + \gam \max_{a'} \val{Q(s',a')} - \val{Q(s,a)}\,\big]$$

In planning only; real transitions use the real reward. $\tau$: real steps since $(s, a)$ was last tried.

### One change from Dyna-Q {#change}

| | planning reward | planning may use |
| --- | --- | --- |
| [[dyna-q]] | $\rew{r}$, as remembered | moves already tried |
| Dyna-Q+ | $\rew{r} + \kappa\sqrt{\tau}$ | also untried moves from visited states |

### Backup diagram {#backup}

{{backup q-learning}}

The Q-learning backup, with a bonus added to the reward in planning.

### Pseudocode

::: pseudocode
Parameters: step size $\alp$, exploration $\eps$, discount $\gam$, planning steps $n$, bonus $\knob{\kappa}$
Set $\val{Q(s,a)}$ (here 0), an empty model and a clock of real steps {#init}
Repeat for each episode:
  Start: $S$ {#start}
  Repeat until $S$ is terminal:
    Choose $A$ $\eps$-greedily from $\val{Q}$ {#choose}
    Take $A$, observe $\rew{R}$ and $S'$; the clock ticks {#act}
    $\val{Q(S,A)} \leftarrow \val{Q(S,A)} + \alp\,[\rew{R} + \gam \max_a \val{Q(S',a)} - \val{Q(S,A)}]$ {#update}
    $\textit{Model}(S,A) \leftarrow \rew{R}, S'$; remember when $(S, A)$ was tried; on a first visit to $S$, its untried actions lead back to $S$ with reward 0 {#model}
    $n$ times: a remembered $(s, a)$ at random, $\rew{r}, s' \leftarrow \textit{Model}(s,a)$, and the update with reward $\rew{r} + \knob{\kappa}\sqrt{\tau}$ {#plan}
    $S \leftarrow S'$ {#next}
:::

### Perks

- Finds improvements in a changing world that Dyna-Q never notices. [See it](lab:dyna-shortcut)
- Recovers faster when the known path is blocked. [See it](lab:dyna-blocking)
- Exploration is directed: it goes where the model is most likely out of date.

### Flaws

- Explores even when nothing has changed: a little worse in a static world.
- The bonus inflates the values: they no longer estimate returns, and the greedy policy may chase untested moves.
- $\kappa$ must be tuned to the size of the rewards.

### Knobs

| Knob | Too low | Too high |
| --- | --- | --- |
| $\knob{\kappa}$ bonus | behaves like Dyna-Q, never checks | wanders after stale moves instead of exploiting |
| $n$ planning steps | the bonus spreads slowly | more computation per step |

### Pitfalls

- Adding the bonus to real rewards: it belongs in planning only.
- Measuring $\tau$ in planning steps instead of real steps: staleness is about the real world.
- Reading the values as returns: they include the bonuses.

### Check yourself {#check}

::: question
Why does Dyna-Q notice at once when its path is blocked, but never notice a shortcut that opens?
---
A blocked path is an optimistic error: the agent keeps trying the old path, sees that it fails, and corrects its model. A new shortcut is a pessimistic error: the current path still works, so nothing makes the agent try the moves that would reveal the shortcut. (Finding the new path around a block can still take long, for the same reason: untried moves look worthless.)
:::

::: question
With $\kappa = 0.001$, what bonus does a move untried for 10 000 steps earn in planning?
---
$0.001 \times \sqrt{10\,000} = 0.1$ per planning update.
:::

::: question
Why does Dyna-Q+ also plan with actions it has never tried, modeled as leading back to the same state?
---
So that they collect bonuses too: their bonus grows with the time since the state was first visited, and planning eventually makes them attractive enough to try for real.
:::
