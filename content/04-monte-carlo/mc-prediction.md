+++
summary = "Estimate how good a policy is without knowing the rules: play whole episodes, and average the returns that actually followed each state."
change = "Average the returns that actually followed each state in played episodes, instead of computing the expected return from a model of the world."
prereqs = ["policy-evaluation", "return", "incremental-mean"]
lab = "blackjack-prediction"
sources = [
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., §5.1, Example 5.1 and Figure 5.1", url = "http://incompleteideas.net/book/the-book-2nd.html" },
  { text = "Metropolis & Ulam (1949), The Monte Carlo method, Journal of the American Statistical Association 44", url = "https://doi.org/10.1080/01621459.1949.10483310" },
  { text = "Barto & Duff (1994), Monte Carlo matrix inversion and reinforcement learning, Advances in Neural Information Processing Systems 6" },
  { text = "Singh & Sutton (1996), Reinforcement learning with replacing eligibility traces, Machine Learning 22", url = "https://doi.org/10.1007/BF00114726" },
]

[story]
scene = "cards"
env = "blackjack"
seed = 3
average = 20
formula = '\step{1}{\rew{G_t} = \rew{R_{t+1}} + \gam\,\rew{R_{t+2}} + \cdots + \gam^{T-t-1}\rew{R_T} \qquad} \step{2}{\val{V(S_t)} \leftarrow \text{average of the returns that followed } S_t}'

[story.runs]
mc = { algorithm = "mc-prediction", policy = "stick-20", gamma = 1.0, units = 100000 }
chart = { algorithm = "mc-prediction", policy = "stick-20", gamma = 1.0, units = 20000, measures = ["error"], name = "Monte Carlo" }
+++

## Story

::: step {run = "mc", at = 0, tiles = "v"}
**Blackjack.** Get closer to 21 than the dealer without going over. The player sees its own sum (12 to 21), the dealer's face-up card, and whether it holds an ace that can count as 11: 200 situations, the two maps on the right. The policy to judge is simple: **stick on 20 or 21, otherwise hit**. How good is each situation under this policy?
:::

::: step {run = "mc", at = 0, play = 1, pace = 700, formula = 1}
Dynamic programming would need the probability of every outcome of every move, which is tedious to work out even for a card game. Monte Carlo just plays. First hand: 17 against a 7. The policy hits, draws an 8, and is bust: a return of $\rew{-1}$. The one situation it passed through takes that return as its first estimate.
:::

::: step {run = "mc", at = 1, play = 1, pace = 700, formula = 2}
Second hand: 15 against a 10. Hit, a 5: now 20, so stick. The dealer ends on 17, and the player wins, $\rew{+1}$. Walking back through the hand, **both** situations it passed through, 20 and 15, get credit for the win. Monte Carlo waits for the end of an episode and then credits every state with the return that followed it.
:::

::: step {run = "mc", at = 1000, tiles = "v"}
After 1000 hands the maps are a noisy patchwork. Each estimate is the average of the few returns that happened to follow its situation, and many situations, especially those with a usable ace, have been seen only a handful of times, or never.
:::

::: step {run = "mc", at = 10000, tiles = "v"}
After 10,000 hands the shape appears. Sums of 20 and 21 are good, since the policy sticks there and usually wins. Everything else is bad, because the policy keeps hitting on 17, 18 and 19 and goes bust. A 19 against a 10 is worth about $\val{-0.74}$.
:::

::: step {run = "mc", at = 100000, tiles = "v"}
After 100,000 hands the estimates are smooth, and close to the exact values worked out from the rules. The maps with a usable ace stay rougher: those situations come up about one time in nine, so they have fewer returns to average.
:::

::: step {run = "mc", at = 100000, tiles = "v", curves = ["chart"], metric = "error"}
The error against the exact values, averaged over 20 runs, falls like one over the square root of the number of hands: four times as many hands halve it. [Play it in the Lab](lab:blackjack-prediction).
:::

## Textbook

### Learning from episodes {#idea}

Dynamic programming computes values from a model ([[policy-evaluation]]). **Monte Carlo methods** compute them from experience instead: whole episodes of states, actions and rewards, collected by interacting with the world or with a simulator of it. Real interaction needs no knowledge of the dynamics at all. A simulator only has to *produce* outcomes, one at a time, which is often far easier than writing down their probabilities: a few lines of code can deal a hand of cards, while the probability of every way the hand can end is a long calculation (the Blackjack example below).

The idea is the definition of value itself. The value of a state is the expected return from it ([[value-functions]]),

$$\val{v_\pi(s)} = \mathbb{E}_\pi\big[\,\rew{G_t} \mid S_t = s\,\big], \qquad \rew{G_t} = \rew{R_{t+1}} + \gam\,\rew{R_{t+2}} + \cdots + \gam^{T-t-1}\,\rew{R_T}, \label{value}$$

so the natural estimate is an average: collect the returns that followed the state in many episodes and average them, and the law of large numbers does the rest. A return is complete only when its episode ends, so the Monte Carlo methods here are for episodic tasks ([[episodes]]), in which every episode eventually terminates, and they change their estimates and policies between episodes, never during one.

### First visits and every visit {#visits}

An episode may pass through the same state several times; each pass is a *visit*. **First-visit** MC takes one return per episode for each state, the one that follows its first visit. **Every-visit** MC takes the returns after all the visits.

::: theorem {#thm-mc} Convergence of Monte Carlo prediction
Both first-visit and every-visit Monte Carlo prediction converge to $\val{v_\pi(s)}$ as the number of visits to $s$ goes to infinity. For first-visit MC each return is an independent, identically distributed estimate of $\val{v_\pi(s)}$ with finite variance, so the average is unbiased and its standard deviation falls as $1/\sqrt{n}$, where $n$ is the number of returns averaged.
:::

::: proof Proof idea
Returns following first visits in different episodes are independent, because episodes are, and each has expectation $\val{v_\pi(s)}$ by \ref{value}. The strong law of large numbers gives convergence, and the variance of an average of $n$ independent samples is $\sigma^2 / n$ ([[incremental-mean]]). Returns following several visits within one episode are not independent, so every-visit estimates are biased for a finite number of episodes; they still converge, and the bias vanishes as the number of episodes grows (Singh & Sutton, 1996).
:::

In practice the two give similar estimates. The atlas uses first-visit MC, whose returns are independent and therefore easiest to analyze; every-visit MC uses more of the data, and it is the version that later carries over to function approximation and eligibility traces.

### The algorithm {#algorithm}

::: algorithm {#alg-mc} First-visit Monte Carlo prediction, for estimating $V \approx \val{v_\pi}$
Input: the policy $\pol{\pi}$ to evaluate
For every state $s$: set $\val{V(s)}$ to any value, and start an empty list $\textit{Returns}(s)$
Repeat for each episode:
  Play one episode with $\pol{\pi}$, recording $S_0, A_0, \rew{R_1}, S_1, \ldots, S_{T-1}, A_{T-1}, \rew{R_T}$
  $\rew{G} \leftarrow 0$
  For $t = T-1$ down to $0$:
    $\rew{G} \leftarrow \gam\,\rew{G} + \rew{R_{t+1}}$
    If this is the first visit to $S_t$ (it is none of $S_0, \ldots, S_{t-1}$):
      Add $\rew{G}$ to $\textit{Returns}(S_t)$, and set $\val{V(S_t)}$ to the mean of $\textit{Returns}(S_t)$
:::

Walking backward through the episode lets the return be accumulated with one multiplication and one addition per step, $\rew{G_t} = \rew{R_{t+1}} + \gam\,\rew{G_{t+1}}$ ([[return]]). The list of returns is not needed: the average can be kept incrementally, with a count $N(s)$ ([[incremental-mean]]):

$$\val{V(S_t)} \leftarrow \val{V(S_t)} + \frac{1}{N(S_t)}\,\big[\,\rew{G_t} - \val{V(S_t)}\,\big]. \label{incremental}$$

Replacing $1/N(S_t)$ by a constant step size $\alp$ gives **constant-α MC**, which weighs recent returns more ([[step-size]]); it suits problems that change over time and is the form compared with TD methods in [[mc-vs-td]].

### Backup diagram {#backup}

The backup diagram of Monte Carlo prediction is the path of one episode, from the state being updated down to the terminal state (\ref{fig-backup}). Beside the diagram of dynamic programming it looks like its opposite. Dynamic programming is wide and shallow: every possible transition, one step deep. Monte Carlo is narrow and deep: one sampled path, followed to the end. Its targets also contain no estimates. Each state's value is an average of returns alone, so the method does not **bootstrap** ([[bootstrapping]]).

::: figure {#fig-backup}
{{backup mc}}
The backup diagram of Monte Carlo prediction: one sampled episode, from the state being updated to the end. Compare the one-step, full-width diagram of dynamic programming in [[policy-evaluation]].
:::

Since no state's estimate leans on another's, one state can be evaluated on its own: start many episodes there, average their returns, and ignore everything else. The cost depends on how long the episodes are, not on how many states the world has. That makes Monte Carlo attractive when only a few states matter: a position in a game, a patient's condition, a configuration of a robot.

### Example: Blackjack {#example}

::: example {#ex-bj} Blackjack (Sutton & Barto, Example 5.1)
Each hand is an episode. The player and the dealer are dealt two cards each; one of the dealer's cards is face up. Cards are drawn from an infinite deck, so with replacement. Face cards count 10, and an ace counts 1 or 11; an ace that can count as 11 without going over 21 is *usable*. The player may *hit* (take another card) or *stick*. Going over 21, *bust*, loses. After the player sticks, the dealer hits until reaching 17 or more. Rewards are $\rew{+1}$, $\rew{-1}$ and $0$ for winning, losing and drawing, all given at the end; $\gam = 1$, so returns equal the final reward. The state is the player's sum (12 to 21; below 12 the player always hits), the dealer's face-up card, and whether the player has a usable ace: 200 states.
:::

Even with complete knowledge of the rules, dynamic programming would need the probabilities $p(s', r \mid s, a)$, for instance the chance of each final outcome given the player's sum and the dealer's showing card, and these are tedious and error-prone to compute. Generating sample hands is easy.

\ref{fig-values} shows the values of the policy that sticks only on 20 or 21, estimated by first-visit Monte Carlo after 10,000 and 500,000 hands, next to the exact values computed from the rules. Only the states with a sum of 20 or 21 have positive value: this policy keeps hitting on 17, 18 and 19, and loses about two hands in three. The estimates for states with a usable ace are rougher, because those states are visited about one time in nine; after 100,000 hands, the state with sum 13, a usable ace and a dealer's 2 has been visited fewer than a hundred times.

::: figure {#fig-values}
{{blackjack-values}}
The values of the policy “stick on 20 or 21” at Blackjack: estimated by first-visit Monte Carlo after 10,000 and after 500,000 hands, and exact. Each map has the player's sum up the side and the dealer's face-up card along the bottom. The Lab plays the hands when the figure comes into view. After Sutton & Barto, Figure 5.1.
:::

### Properties {#properties}

- **No model.** Only sample episodes are needed, from real interaction or from a simulator.
- **Unbiased, but noisy.** First-visit estimates are unbiased; their variance comes from everything that happens until the end of the episode, all the actions and all the transitions, and can be large ([[mc-vs-td]]).
- **No bootstrapping.** Estimates do not depend on other estimates, so errors do not propagate between states, and the method does not rely on the Markov property of the states ([[mdp]]).
- **Only at the end of episodes.** Nothing is learned until an episode ends, and the method does not apply to continuing tasks without modification.
- **Focus on what matters.** States can be evaluated one at a time, at a cost independent of the size of the state space.

### Historical remarks {#history}

The name *Monte Carlo*, for methods that estimate quantities by random sampling, was coined in the 1940s by the physicists who used such methods at Los Alamos (Metropolis & Ulam, 1949). Barto and Duff (1994) related Monte Carlo methods for solving linear systems to reinforcement learning. Singh and Sutton (1996) analyzed first-visit and every-visit methods. The Blackjack example is Sutton and Barto's Example 5.1.

## Card

### Idea

Play an episode to the end with the policy, then walk back through it and credit each state with the return that actually followed it. A state's value is the average of those returns. No model is needed, only complete episodes.

::: analogy
Rating a commute by its total travel time. You learn how good it is to be at a given junction by noting, every day you pass it, how long the rest of the trip actually took, and averaging.
:::

### The update {#update}

$$\val{V(S_t)} \leftarrow \val{V(S_t)} + \frac{1}{N(S_t)}\,\big[\,\rew{G_t} - \val{V(S_t)}\,\big] \qquad \rew{G_t} = \rew{R_{t+1}} + \gam\,\rew{G_{t+1}}$$

Applied after the episode ends, from the last step back to the first, at the first visit to each state. With a constant $\alp$ instead of $1/N$: constant-α MC.

### One change from policy evaluation {#change}

| | target of the update |
| --- | --- |
| [[policy-evaluation]] | $\sum_a \pol{\pi(a \mid s)} \sum_{s',r} p(s',r \mid s,a)\,[\rew{r} + \gam\,\val{V(s')}]$: all outcomes, from the model |
| Monte Carlo | $\rew{G_t}$: the return that actually followed, from one played episode |

### Backup diagram {#backup}

{{backup mc}}

From the state being updated along the one episode that was played, all the way to its end.

### Pseudocode

::: pseudocode
Parameters: the policy $\pol{\pi}$ to evaluate, discount $\gam$
Set $\val{V(s)} = 0$ and $N(s) = 0$ for every state {#init}
Repeat for each episode:
  Deal or reset: the first state $S_0$ {#start}
  Play the episode with $\pol{\pi}$ to the end: $S_0, A_0, \rew{R_1}, \ldots, S_{T-1}, A_{T-1}, \rew{R_T}$ {#generate}
  $\rew{G} \leftarrow 0$ {#g0}
  For $t = T-1$ down to $0$: $\rew{G} \leftarrow \gam\,\rew{G} + \rew{R_{t+1}}$ {#return}
    If $S_t$ was already visited earlier in the episode, skip it {#first}
    $N(S_t) \leftarrow N(S_t) + 1$ and $\val{V(S_t)} \leftarrow \val{V(S_t)} + \tfrac{1}{N(S_t)}\,[\rew{G} - \val{V(S_t)}]$ {#update}
:::

### Perks

- Needs no model: playing is enough, even when the rules are too complicated to write down as probabilities. [See it](lab:blackjack-prediction)
- Unbiased: each return is a fair sample of the value.
- Each state's estimate stands alone: no bootstrapping, and any single state can be evaluated without the others.

### Flaws

- High variance: a return depends on every random event until the end of the episode.
- Learns only when an episode ends, and needs episodes that end.
- Rarely visited states stay poorly estimated. [See it](lab:blackjack-prediction)

### Knobs

| Knob | Too low | Too high |
| --- | --- | --- |
| $\alp$ step size (constant-α MC) | slow, but averages well | follows the latest returns, which are noisy |
| episodes | estimates still noisy | (none: more episodes only help) |

### Pitfalls

- Accumulating the return forward: walk backward, $\rew{G} \leftarrow \gam\,\rew{G} + \rew{R_{t+1}}$, to do it in one pass.
- Mixing first-visit and every-visit by accident: check for earlier visits before counting a return.
- Expecting estimates for states the policy rarely visits: Monte Carlo only learns about states it sees.

### Check yourself {#check}

::: question
In Blackjack with $\gam = 1$, what is the return from every state of a hand?
---
The final reward: $+1$, $0$ or $-1$. Every intermediate reward is 0, so each state the hand passed through is credited with how the hand ended.
:::

::: question
Why does Monte Carlo prediction not need the Markov property?
---
It never uses the estimate of a next state. Each estimate is an average of real returns, whatever the states contain.
:::

::: question
How many times as many hands are needed to halve the error of the estimates?
---
About four times as many: the standard error of an average falls like $1/\sqrt{n}$.
:::
