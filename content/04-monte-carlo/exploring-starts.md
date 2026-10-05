+++
summary = "Learn to act from whole episodes: average the returns after each state and action, act greedily on the averages, and start every episode with a random state and move so that nothing goes untried."
change = "Estimate action values Q(s, a) and make the policy greedy with respect to them after every episode; start each episode from a random state and action, so that every pair keeps being tried."
prereqs = ["mc-prediction", "policy-improvement", "gpi"]
lab = "blackjack-control"
sources = [
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., §5.2–5.3, Example 5.3 and Figure 5.2", url = "http://incompleteideas.net/book/the-book-2nd.html" },
  { text = "Thorp (1966), Beat the Dealer, 2nd ed., Random House" },
  { text = "Tsitsiklis (2002), On the convergence of optimistic policy iteration, Journal of Machine Learning Research 3", url = "https://www.jmlr.org/papers/v3/tsitsiklis02a.html" },
]

[story]
scene = "cards"
env = "blackjack"
seed = 1
average = 10
formula = '\step{1}{\val{Q(S_t, A_t)} \leftarrow \text{average of the returns after } (S_t, A_t) \qquad} \step{2}{\pol{\pi(s)} = \operatorname*{arg\,max}_a \val{Q(s,a)}}'

[story.runs]
es = { algorithm = "exploring-starts", gamma = 1.0, units = 100000 }
chart = { algorithm = "exploring-starts", gamma = 1.0, units = 20000, measures = ["match"], name = "Monte Carlo ES" }
+++

## Story

::: step {run = "es", at = 0, tiles = "policy", formula = 1}
**Now the player must learn how to play.** It keeps two numbers for each situation, the average return after sticking and after hitting, $\val{Q(s, \text{stick})}$ and $\val{Q(s, \text{hit})}$, and plays the move whose number is higher. The maps show its current choice: dark for hit, light for stick, blank where the two are tied. Nothing has been played yet.
:::

::: step {run = "es", at = 0, play = 2, pace = 650, formula = 2}
There is a catch. A player that always picks the move that looks better never tries the other one, so it can never find out it was wrong. The fix here is **exploring starts**: every hand begins in a random situation with a random first move, and is played greedily from then on. The first hand starts at 18 with a usable ace against a 3, and its random first move is to hit.
:::

::: step {run = "es", at = 1000, tiles = "policy", formula = 2}
After 1000 hands the maps are a rough sketch: most situations with a high sum already stick, most low ones hit. About two situations in three already get the optimal move.
:::

::: step {run = "es", at = 10000, tiles = "policy"}
After 10,000 hands: close to nine in ten. The remaining mistakes sit where the two moves are worth almost the same, so their averages need many hands to separate.
:::

::: step {run = "es", at = 100000, tiles = "policy"}
After 100,000 hands the strategy is nearly the optimal one, computed exactly from the rules. Without a usable ace: stick on 17 or more, but against a dealer's 2 to 6 stick much earlier, because the dealer is likely to go bust. With a usable ace: keep hitting up to 17 or 18, since an ace that can drop back to 1 makes a bust impossible on the next card.
:::

::: step {run = "es", at = 100000, tiles = "policy", curves = ["chart"], metric = "match"}
The share of situations whose greedy move is optimal, averaged over 10 runs. Monte Carlo ES gets most situations right quickly and the last few slowly. [Learn it in the Lab](lab:blackjack-control).
:::

## Textbook

### Values of actions {#action-values}

State values are enough to act only with a model to look one step ahead: from $\val{v}$ and $p$, the best action is the one that leads to the best mix of reward and next-state value ([[policy-improvement]]). Without a model, knowing which next states are good does not say which action gets there. The values must be attached to the actions themselves: $\val{q_\pi(s,a)}$, the expected return after taking $a$ in $s$ and following $\pol{\pi}$ from then on. Estimating them needs nothing new ([[mc-prediction]]). Count a *visit* to the pair $(s, a)$ whenever $a$ is taken in $s$, and average the returns that follow first visits. As every pair is visited more and more, the averages converge to the true action values, with errors shrinking like one over the square root of the number of visits.

The catch is coverage. A deterministic policy takes the same action every time it passes through a state, so only that action's returns are ever observed. The other actions keep their initial estimates forever, and comparing them, the whole point of action values, never becomes possible. Some way of **maintaining exploration** is needed ([[explore-exploit]]).

### Exploring starts {#es}

The bluntest fix is to control how episodes begin. Choose the first state *and the first action* at random, giving every pair some chance of being the start; then, over infinitely many episodes, every pair is tried infinitely often, whatever the policy does afterwards. This is the assumption of **exploring starts**. It costs nothing in a simulator that can be reset to any situation. In the real world it is usually out of reach: neither a robot nor a patient can be placed in an arbitrary situation and made to try an arbitrary action. The alternatives, policies that keep exploring on their own, are the subject of [[mc-control]] and [[off-policy-mc]].

### Monte Carlo control {#control}

Control follows the loop of generalized policy iteration ([[gpi]]), with Monte Carlo doing the evaluation: keep a policy and a table of action values, let averages of returns pull the table toward the policy's values, and let the policy turn greedy for the table:

$$\pol{\pi_0} \xrightarrow{\;E\;} \val{q_{\pi_0}} \xrightarrow{\;I\;} \pol{\pi_1} \xrightarrow{\;E\;} \val{q_{\pi_1}} \xrightarrow{\;I\;} \pol{\pi_2} \xrightarrow{\;E\;} \cdots \xrightarrow{\;I\;} \pol{\pi_*} \xrightarrow{\;E\;} \val{q_*}. \label{sequence}$$

Improvement makes the policy greedy with respect to the current action values, $\pol{\pi(s)} = \operatorname*{arg\,max}_a \val{q(s,a)}$, which needs no model. The policy improvement theorem then applies to $\pol{\pi_k}$ and $\pol{\pi_{k+1}}$, because for all states

$$\val{q_{\pi_k}(s, \pol{\pi_{k+1}(s)})} = \val{q_{\pi_k}(s, \operatorname*{arg\,max}_a q_{\pi_k}(s,a))} = \max_a \val{q_{\pi_k}(s,a)} \ge \val{q_{\pi_k}(s, \pol{\pi_k(s)})} \ge \val{v_{\pi_k}(s)}. \label{improve}$$

Each new policy is therefore at least as good as the last, and if it is no better, both are optimal. Taken literally, the loop needs two things that never happen: exploring starts, and infinitely many episodes for each evaluation. The second is easy to drop, just as [[value-iteration]] drops complete evaluations: improve before the evaluation is finished. The natural unit for Monte Carlo is the episode, so the two steps take turns once per episode. The finished episode's returns go into the averages, and the policy is made greedy again in every state the episode passed through.

### The algorithm {#algorithm}

::: algorithm {#alg-es} Monte Carlo ES (Exploring Starts), for estimating $\pol{\pi} \approx \pol{\pi_*}$
For every state and action: pick any action $\pol{\pi(s)}$, any value $\val{Q(s,a)}$, and start an empty list $\textit{Returns}(s,a)$
Repeat for each episode:
  Draw a starting pair $S_0, A_0$ at random, every pair with a chance above 0
  Play the episode from there, following $\pol{\pi}$ after the first action: $S_0, A_0, \rew{R_1}, \ldots, S_{T-1}, A_{T-1}, \rew{R_T}$
  $\rew{G} \leftarrow 0$
  For $t = T-1$ down to $0$:
    $\rew{G} \leftarrow \gam\,\rew{G} + \rew{R_{t+1}}$
    If this is the first visit to the pair $S_t, A_t$ in the episode:
      Add $\rew{G}$ to $\textit{Returns}(S_t, A_t)$, and set $\val{Q(S_t, A_t)}$ to their mean
      $\pol{\pi(S_t)} \leftarrow \operatorname*{arg\,max}_a \val{Q(S_t, a)}$
:::

As in [[mc-prediction]], the averages can be kept incrementally with counts. Ties in the arg max are best broken at random, since early on, with all estimates equal, a fixed tie-breaking rule would make the policy systematic in a way that has nothing to do with the values.

### Convergence {#convergence}

Monte Carlo ES pools every return a pair has ever produced, under whichever policy was in force at the time. Why should that settle on an optimal policy? Suppose it settled on a worse one. From then on all new returns would come from that policy, the averages would drift toward its action values, and greedy improvement on those values would change the policy after all ([[policy-improvement]]). So the only place the algorithm can come to rest is where both the policy and the values are optimal. That it always gets there is a different claim: the argument rules out suboptimal resting points, not endless wandering. Proofs exist only for special cases and for closely related variants, such as the synchronous version analyzed by Tsitsiklis (2002); the general case is still open.

### Example: solving Blackjack {#example}

Blackjack ([[mc-prediction]]) suits exploring starts, because hands are simulated: it is simple to start each one by dealing a random player sum, dealer's card and usable-ace status, and choosing the first action at random. Starting from a policy and action values that know nothing, Monte Carlo ES learns a strategy close to the optimal one, which for Blackjack can be computed exactly from the rules (\ref{fig-policy}).

::: figure {#fig-policy}
{{blackjack-policy}}
The optimal Blackjack strategy, computed exactly from the rules, and the greedy strategy learned by Monte Carlo ES; cells where the two differ are outlined. Each map has the player's sum up the side and the dealer's face-up card along the bottom; dark is hit, light is stick. The Lab plays the hands when the figure comes into view. After Sutton & Barto, Figure 5.2.
:::

The optimal strategy is Thorp's *basic strategy* for this version of the game (Thorp, 1966). Without a usable ace, stick on 17 or more against an ace or a 7 to 10, but against a dealer's 2 or 3 stick from 13, and against a 4, 5 or 6 stick from 12: those dealers bust often, so the player should not risk it. With a usable ace, hit up to 17, or 18 against an ace, a 9 or a 10: an ace that can fall back to 1 makes it impossible to bust on the next card, so hitting costs less. The learned strategy agrees with it almost everywhere; where it does not, the two actions are worth almost the same, and their averages take many hands to separate.

### Properties {#properties}

- **Model-free control.** Action values make improvement possible without a model.
- **Exploring starts are a strong assumption.** They need control over how episodes start, which a simulator gives and the real world usually does not.
- **Averages across policies.** Each average mixes returns from all the policies that were in force, most of them worse than the current one. This slows learning; a constant step size, which forgets old returns, is a common alternative ([[nonstationary]]).
- **Only at the end of episodes,** like all Monte Carlo methods, and only for episodic tasks.
- **Finds a deterministic optimal policy,** greedy with respect to the action values.

### Historical remarks {#history}

Monte Carlo ES was introduced by Sutton and Barto (§5.3); their Blackjack example builds on one by Widrow, Gupta and Maitra (1973). The strategy it learns is the basic strategy derived by Thorp (1966) from careful computation of the odds. The convergence of Monte Carlo ES remains open in general; Tsitsiklis (2002) proved it for a version with synchronous updates.

## Card

### Idea

Average the returns that followed each state *and action*, and always play the action with the best average. To make sure every action keeps being tried, start every episode in a random state with a random first action. Episode by episode, the values and the greedy policy improve each other.

::: analogy
Learning a card game by dealing yourself random hands and forcing a random first move each time, then playing what seems best and keeping score of how each move in each situation turned out.
:::

### The update {#update}

$$\val{Q(S_t, A_t)} \leftarrow \val{Q(S_t, A_t)} + \frac{1}{N(S_t, A_t)}\,\big[\,\rew{G_t} - \val{Q(S_t, A_t)}\,\big] \qquad \pol{\pi(S_t)} \leftarrow \operatorname*{arg\,max}_a \val{Q(S_t, a)}$$

After every episode, at the first visit to each pair; each episode starts with a random state and action.

### One change from Monte Carlo prediction {#change}

| | estimates | policy |
| --- | --- | --- |
| [[mc-prediction]] | $\val{V(s)}$ of a fixed policy | given |
| Monte Carlo ES | $\val{Q(s,a)}$ | greedy with respect to $\val{Q}$, improved after every episode |

### Backup diagram {#backup}

{{backup mc-q}}

From the state–action pair being updated along the one episode that was played, to its end.

### Pseudocode

::: pseudocode
Parameters: discount $\gam$
Set $\val{Q(s,a)} = 0$ and $N(s,a) = 0$ for every state and action {#init}
Repeat for each episode:
  Start at a random state with a random first action {#start}
  Play the episode to the end, greedily with respect to $\val{Q}$ after the first action {#generate}
  $\rew{G} \leftarrow 0$ {#g0}
  For $t = T-1$ down to $0$: $\rew{G} \leftarrow \gam\,\rew{G} + \rew{R_{t+1}}$ {#return}
    If the pair $S_t, A_t$ was already visited earlier in the episode, skip it {#first}
    $N \leftarrow N + 1$ and $\val{Q(S_t,A_t)} \leftarrow \val{Q(S_t,A_t)} + \tfrac{1}{N}\,[\rew{G} - \val{Q(S_t,A_t)}]$; the policy is greedy with respect to $\val{Q}$ {#update}
:::

### Perks

- Learns an optimal policy from episodes alone: no model. [See it](lab:blackjack-control)
- Simple: averages and a greedy choice.
- Every estimate stands on real returns, with no bootstrapping.

### Flaws

- Needs exploring starts: control over where episodes begin, which real experience rarely allows.
- Averages returns from old, worse policies along with recent ones, which slows it down.
- Learns only at the end of episodes; useless for tasks that never end.

### Knobs

| Knob | Too low | Too high |
| --- | --- | --- |
| episodes | the boundary of the strategy is still noisy | (none: more episodes only help) |
| $\alp$ (if a constant step size replaces averages) | slow to forget the returns of old policies | estimates follow the latest returns, which are noisy |

### Pitfalls

- Starting episodes from a fixed state: then exploring starts are not exploring anything.
- Breaking ties toward one action: early on all values tie, and the “greedy” policy becomes systematic.
- Applying it to real interaction, where the starting state and action cannot be chosen.

### Check yourself {#check}

::: question
Why does model-free control estimate action values rather than state values?
---
Improving a policy from state values needs a one-step lookahead with the model. With action values, the best action can be read off directly.
:::

::: question
What would go wrong without exploring starts, if the policy is greedy?
---
In each state only the currently best-looking action would ever be taken, so the other actions' values would never be estimated, and a wrong first impression could never be corrected.
:::

::: question
Why can Monte Carlo ES not settle on a suboptimal policy?
---
If the policy stopped changing, the action values would converge to that policy's values, and greedy improvement would then change the policy, unless it was already optimal.
:::
