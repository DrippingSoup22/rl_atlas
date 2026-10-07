# World study: which algorithm learns where, and from what settings

The Lab's World panel lets every algorithm run on every shared world its kind allows (`lab/worlds.js`). This study
picks the settings a run starts from in each world, and finds the pairings that rarely end well, which the panel marks
**hard** instead of hiding them: a failing run is a result worth seeing.

## Method

1. **Rules** (who may run where): tabular methods on the grids, Taxi, Catch and Blackjack; dynamic programming where the
   rules are written out; prediction wherever true values can be worked out to judge it; linear SARSA on the continuous
   states (tile coding); policy-gradient methods on the grids and the continuous states (TRPO on the grids only: its
   natural gradient is worked out for one preference per state and action).
2. **First pass**, per world and family: a few shared settings (tabular α; policy α and critic αw; prediction α) tried
   on the family's main algorithms, 3 seeds each, with the world's episode budget. The setting under which most runs
   end well was kept.
3. **Second pass**, per algorithm where the family's setting failed: REINFORCE's much smaller step sizes, Monte Carlo
   control's exploration, A2C's larger steps, n-step methods' smaller steps.
4. **Verification**: every algorithm the panel offers in every world, with the settings it now starts from, 3 seeds.
   Pairings with at most 1 of 3 seeds ending well are marked hard (the tables below).

The pass mark of each world is its odds rule in the Lab. Prediction has no pass mark; its column shows the error left.

## Changes the study led to

- **Episode caps** on the grids (1,000 steps; Frozen Lake 100 and 200, as in Gymnasium): a policy that wanders no
  longer runs 5,000-step episodes.
- **Frozen Lake is judged by its chance of reaching the gem** (`judge = 1`), not by the discounted value, whose best
  possible is only 0.54 on the small lake. Pass marks: 60% (best possible 82%) and 50% on the large lake (best 89%).
- **Prediction on the grids** evaluates "mostly the best way" (the shortest way nine times in ten, a random move
  otherwise), with episodes starting anywhere: a policy's values do not depend on where its episodes start, and a
  policy that only walks its own path would leave most states unvisited.
- **Dynamic programming** uses γ = 0.9 on Cliff walking, the windy gridworld and Taxi, where evaluating the first,
  random policy with γ = 1 takes thousands of sweeps.
- **A racer's own step size** keeps its ratio to the lab's when it runs in another world (a racer with double the step
  size still has double, in the new world's units); settings tied to its home world (its own features, a behavior
  policy named there) are dropped.

## What the study found

- **REINFORCE**, with or without a baseline, fails on every grid but the large lake (where plain REINFORCE passes on 2
  of 3 seeds), on Catch (it improves, too slowly for the pass mark), and on Mountain Car and Acrobot. One long early episode's return (tens of thousands on the cliff) saturates the policy whatever the step
  size: the high variance its chapter warns about, at full scale.
- **Monte Carlo control** (on-policy, exploring starts, off-policy) fails on the cliff, the windy gridworld and Taxi:
  episodes that rarely end in the first place give it nothing to average. Off-policy MC learns only from the tail of
  each episode after the behavior's last non-greedy move.
- **Q-learning and the Dyna family on the cliff** fail the "while still exploring" pass mark: they learn the edge path
  and keep falling while they explore. This is the cliff's own lesson.
- **Actor–critic** is the most dependable policy method on the grids; **A2C, PPO and TRPO** solve the cliff, the windy
  gridworld, the Dyna maze and Catch, and fail on Taxi in their round budget.
- **The drifting testbed** defeats UCB and the gradient bandit, whose long memories lose track of moving arms.

## Results with the settings each run starts from

### Cliff walking

Ends well (control): end up losing less than 40 per episode on average, while still exploring.

| Algorithm | Measure, last tenth (mean of 3 seeds) | Seeds that end well | Marked hard |
|---|---|---|---|
| TD(0) | error 6.77 | — |  |
| SARSA | return −24.71 | 3 of 3 |  |
| Q-learning | return −58.09 | 0 of 3 | hard |
| Expected SARSA | return −24.67 | 3 of 3 |  |
| Double Q-learning | return −26.61 | 3 of 3 |  |
| Monte Carlo | error 10.66 | — |  |
| Monte Carlo ES | return −424 | 0 of 3 | hard |
| On-policy MC | return −1231 | 0 of 3 | hard |
| Off-policy MC | return −3871 | 0 of 3 | hard |
| Policy evaluation | delta 0.00 | — |  |
| Policy iteration | delta 0.31 | — |  |
| Value iteration | delta 0.00 | — |  |
| n-step TD | error 8.49 | — |  |
| n-step SARSA | return −22.63 | 3 of 3 |  |
| TD(λ) | error 8.07 | — |  |
| SARSA(λ) | return −23.15 | 3 of 3 |  |
| Offline λ-return | error 8.79 | — |  |
| Dyna-Q | return −44.60 | 1 of 3 | hard |
| Dyna-Q+ | return −44.05 | 1 of 3 | hard |
| Prioritized sweeping | return −48.26 | 1 of 3 | hard |
| REINFORCE | return −67000 | 0 of 3 | hard |
| REINFORCE with baseline | return −1000 | 0 of 3 | hard |
| Actor–critic | return −15.16 | 3 of 3 |  |
| A2C | return −19.44 | 3 of 3 |  |
| TRPO | return −15.06 | 3 of 3 |  |
| PPO | return −17.64 | 3 of 3 |  |

### Windy gridworld

Ends well (control): end with episodes under 25 steps on average, while still exploring.

| Algorithm | Measure, last tenth (mean of 3 seeds) | Seeds that end well | Marked hard |
|---|---|---|---|
| TD(0) | error 10.37 | — |  |
| SARSA | steps 21.25 | 3 of 3 |  |
| Q-learning | steps 17.80 | 3 of 3 |  |
| Expected SARSA | steps 18.31 | 3 of 3 |  |
| Double Q-learning | steps 24.12 | 2 of 3 |  |
| Monte Carlo | error 6.65 | — |  |
| Monte Carlo ES | steps 1000 | 0 of 3 | hard |
| On-policy MC | steps 731 | 0 of 3 | hard |
| Off-policy MC | steps 1000 | 0 of 3 | hard |
| Policy evaluation | delta 0.00 | — |  |
| Policy iteration | delta 0.00 | — |  |
| Value iteration | delta 0.00 | — |  |
| n-step TD | error 7.55 | — |  |
| n-step SARSA | steps 21.35 | 3 of 3 |  |
| TD(λ) | error 7.86 | — |  |
| SARSA(λ) | steps 20.22 | 3 of 3 |  |
| Offline λ-return | error 7.23 | — |  |
| Dyna-Q | steps 17.39 | 3 of 3 |  |
| Dyna-Q+ | steps 16.80 | 3 of 3 |  |
| Prioritized sweeping | steps 16.82 | 3 of 3 |  |
| REINFORCE | steps 1000 | 0 of 3 | hard |
| REINFORCE with baseline | steps 1000 | 0 of 3 | hard |
| Actor–critic | steps 17.82 | 3 of 3 |  |
| A2C | steps 18.46 | 3 of 3 |  |
| TRPO | steps 15.68 | 3 of 3 |  |
| PPO | steps 344 | 2 of 3 |  |

### Dyna maze

Ends well (control): end with episodes under 40 steps on average.

| Algorithm | Measure, last tenth (mean of 3 seeds) | Seeds that end well | Marked hard |
|---|---|---|---|
| TD(0) | error 0.58 | — |  |
| SARSA | steps 17.87 | 3 of 3 |  |
| Q-learning | steps 17.93 | 3 of 3 |  |
| Expected SARSA | steps 17.93 | 3 of 3 |  |
| Double Q-learning | steps 68.13 | 0 of 3 | hard |
| Monte Carlo | error 0.40 | — |  |
| Monte Carlo ES | steps 474 | 1 of 3 | hard |
| On-policy MC | steps 19.20 | 3 of 3 |  |
| Off-policy MC | steps 22.33 | 3 of 3 |  |
| Policy evaluation | delta 0.00 | — |  |
| Policy iteration | delta 0.00 | — |  |
| Value iteration | delta 0.00 | — |  |
| n-step TD | error 0.47 | — |  |
| n-step SARSA | steps 19.13 | 3 of 3 |  |
| TD(λ) | error 0.47 | — |  |
| SARSA(λ) | steps 19.93 | 3 of 3 |  |
| Offline λ-return | error 0.43 | — |  |
| Dyna-Q | steps 15.73 | 3 of 3 |  |
| Dyna-Q+ | steps 16.53 | 3 of 3 |  |
| Prioritized sweeping | steps 15.87 | 3 of 3 |  |
| REINFORCE | steps 471 | 0 of 3 | hard |
| REINFORCE with baseline | steps 479 | 0 of 3 | hard |
| Actor–critic | steps 545 | 0 of 3 | hard |
| A2C | steps 15.54 | 3 of 3 |  |
| TRPO | steps 14.43 | 3 of 3 |  |
| PPO | steps 14.81 | 3 of 3 |  |

### Frozen Lake

Ends well (control): end with a greedy policy that reaches the gem at least 60% of the time (82% is the best possible).

| Algorithm | Measure, last tenth (mean of 3 seeds) | Seeds that end well | Marked hard |
|---|---|---|---|
| TD(0) | error 0.02 | — |  |
| SARSA | greedy 0.71 | 3 of 3 |  |
| Q-learning | greedy 0.76 | 3 of 3 |  |
| Expected SARSA | greedy 0.74 | 3 of 3 |  |
| Double Q-learning | greedy 0.46 | 0 of 3 | hard |
| Monte Carlo | error 0.03 | — |  |
| Monte Carlo ES | greedy 0.26 | 0 of 3 | hard |
| On-policy MC | greedy 0.16 | 0 of 3 | hard |
| Off-policy MC | greedy 0.11 | 0 of 3 | hard |
| Policy evaluation | delta 0.00 | — |  |
| Policy iteration | delta 0.00 | — |  |
| Value iteration | delta 0.00 | — |  |
| n-step TD | error 0.04 | — |  |
| n-step SARSA | greedy 0.54 | 2 of 3 |  |
| TD(λ) | error 0.03 | — |  |
| SARSA(λ) | greedy 0.17 | 0 of 3 | hard |
| Offline λ-return | error 0.05 | — |  |
| Dyna-Q | greedy 0.39 | 0 of 3 | hard |
| Dyna-Q+ | greedy 0.40 | 0 of 3 | hard |
| Prioritized sweeping | greedy 0.30 | 0 of 3 | hard |
| REINFORCE | greedy 0.31 | 0 of 3 | hard |
| REINFORCE with baseline | greedy 0.30 | 1 of 3 | hard |
| Actor–critic | greedy 0.78 | 3 of 3 |  |
| A2C | greedy 0.21 | 0 of 3 | hard |
| TRPO | greedy 0.73 | 3 of 3 |  |
| PPO | greedy 0.29 | 0 of 3 | hard |

### Frozen Lake 8 × 8

Ends well (control): end with a greedy policy that reaches the gem at least half the time (89% is the best possible).

| Algorithm | Measure, last tenth (mean of 3 seeds) | Seeds that end well | Marked hard |
|---|---|---|---|
| TD(0) | error 0.02 | — |  |
| SARSA | greedy 0.88 | 3 of 3 |  |
| Q-learning | greedy 0.85 | 3 of 3 |  |
| Expected SARSA | greedy 0.86 | 3 of 3 |  |
| Double Q-learning | greedy 0.61 | 2 of 3 |  |
| Monte Carlo | error 0.04 | — |  |
| Monte Carlo ES | greedy 0.26 | 0 of 3 | hard |
| On-policy MC | greedy 0.03 | 0 of 3 | hard |
| Off-policy MC | greedy 0.00 | 0 of 3 | hard |
| Policy evaluation | delta 0.00 | — |  |
| Policy iteration | delta 0.00 | — |  |
| Value iteration | delta 0.00 | — |  |
| n-step TD | error 0.03 | — |  |
| n-step SARSA | greedy 0.60 | 3 of 3 |  |
| TD(λ) | error 0.03 | — |  |
| SARSA(λ) | greedy 0.32 | 0 of 3 | hard |
| Offline λ-return | error 0.04 | — |  |
| Dyna-Q | greedy 0.52 | 2 of 3 |  |
| Dyna-Q+ | greedy 0.27 | 0 of 3 | hard |
| Prioritized sweeping | greedy 0.18 | 0 of 3 | hard |
| REINFORCE | greedy 0.50 | 2 of 3 |  |
| REINFORCE with baseline | greedy 0.27 | 1 of 3 | hard |
| Actor–critic | greedy 0.95 | 3 of 3 |  |
| A2C | greedy 0.07 | 0 of 3 | hard |
| TRPO | greedy 0.53 | 2 of 3 |  |
| PPO | greedy 0.02 | 0 of 3 | hard |

### Taxi

Ends well (control): end up losing less than 5 per trip on average, while still exploring.

| Algorithm | Measure, last tenth (mean of 3 seeds) | Seeds that end well | Marked hard |
|---|---|---|---|
| TD(0) | error 4.42 | — |  |
| SARSA | return 1.80 | 3 of 3 |  |
| Q-learning | return 2.43 | 3 of 3 |  |
| Expected SARSA | return 2.49 | 3 of 3 |  |
| Double Q-learning | return 1.82 | 3 of 3 |  |
| Monte Carlo | error 2.51 | — |  |
| Monte Carlo ES | return −617 | 0 of 3 | hard |
| On-policy MC | return −1007 | 0 of 3 | hard |
| Off-policy MC | return −815 | 0 of 3 | hard |
| Policy evaluation | delta 0.00 | — |  |
| Policy iteration | delta 0.16 | — |  |
| Value iteration | delta 0.00 | — |  |
| n-step TD | error 2.46 | — |  |
| n-step SARSA | return −457 | 1 of 3 | hard |
| TD(λ) | error 2.55 | — |  |
| SARSA(λ) | return −269 | 2 of 3 |  |
| Offline λ-return | error 2.45 | — |  |
| Dyna-Q | return 2.13 | 3 of 3 |  |
| Dyna-Q+ | return 2.21 | 3 of 3 |  |
| Prioritized sweeping | return 2.37 | 3 of 3 |  |
| REINFORCE | return −416 | 0 of 3 | hard |
| REINFORCE with baseline | return −482 | 0 of 3 | hard |
| Actor–critic | return 6.76 | 3 of 3 |  |
| A2C | return −75.26 | 0 of 3 | hard |
| TRPO | return −130 | 0 of 3 | hard |
| PPO | return −116 | 0 of 3 | hard |

### Catch

Ends well (control): end up catching the ball at least four times in five on average, while still exploring.

| Algorithm | Measure, last tenth (mean of 3 seeds) | Seeds that end well | Marked hard |
|---|---|---|---|
| TD(0) | error 0.57 | — |  |
| SARSA | return 0.83 | 3 of 3 |  |
| Q-learning | return 0.81 | 3 of 3 |  |
| Expected SARSA | return 0.84 | 3 of 3 |  |
| Double Q-learning | return 0.83 | 3 of 3 |  |
| Monte Carlo | error 0.29 | — |  |
| Monte Carlo ES | return 1.00 | 3 of 3 |  |
| On-policy MC | return 0.95 | 3 of 3 |  |
| Off-policy MC | return 0.71 | 3 of 3 |  |
| Policy evaluation | delta 0.00 | — |  |
| Policy iteration | delta 1.33 | — |  |
| Value iteration | delta 0.00 | — |  |
| n-step TD | error 0.33 | — |  |
| n-step SARSA | return 0.85 | 3 of 3 |  |
| TD(λ) | error 0.35 | — |  |
| SARSA(λ) | return 0.85 | 3 of 3 |  |
| Offline λ-return | error 0.31 | — |  |
| Dyna-Q | return 0.86 | 3 of 3 |  |
| Dyna-Q+ | return 0.76 | 3 of 3 |  |
| Prioritized sweeping | return 0.81 | 3 of 3 |  |
| REINFORCE | return 0.43 | 0 of 3 | hard |
| REINFORCE with baseline | return 0.52 | 0 of 3 | hard |
| Actor–critic | return 1.00 | 3 of 3 |  |
| A2C | return 0.82 | 3 of 3 |  |
| TRPO | return 0.91 | 3 of 3 |  |
| PPO | return 0.97 | 3 of 3 |  |

### Blackjack

| Algorithm | Measure, last tenth (mean of 3 seeds) | Seeds that end well | Marked hard |
|---|---|---|---|
| TD(0) | error 0.07 | — |  |
| SARSA | return −0.14 | — |  |
| Q-learning | return −0.14 | — |  |
| Expected SARSA | return −0.14 | — |  |
| Double Q-learning | return −0.15 | — |  |
| Monte Carlo | error 0.10 | — |  |
| Monte Carlo ES | return −0.15 | — |  |
| On-policy MC | return −0.12 | — |  |
| Off-policy MC | return −0.14 | — |  |
| n-step TD | error 0.10 | — |  |
| n-step SARSA | return −0.14 | — |  |
| TD(λ) | error 0.09 | — |  |
| SARSA(λ) | return −0.14 | — |  |
| Offline λ-return | error 0.09 | — |  |
| Dyna-Q | return −0.18 | — |  |
| Dyna-Q+ | return −0.17 | — |  |
| Prioritized sweeping | return −0.18 | — |  |
| REINFORCE | return −0.10 | — |  |
| REINFORCE with baseline | return −0.10 | — |  |
| Actor–critic | return −0.11 | — |  |
| A2C | return −0.09 | — |  |
| TRPO | return −0.11 | — |  |
| PPO | return −0.12 | — |  |

### Mountain Car

Ends well (control): end with episodes under 150 steps on average.

| Algorithm | Measure, last tenth (mean of 3 seeds) | Seeds that end well | Marked hard |
|---|---|---|---|
| Semi-gradient SARSA | steps 115 | 3 of 3 |  |
| REINFORCE | steps 2000 | 0 of 3 | hard |
| REINFORCE with baseline | steps 2000 | 0 of 3 | hard |
| Actor–critic | steps 115 | 3 of 3 |  |
| A2C | steps 106 | 3 of 3 |  |
| PPO | steps 1370 | 1 of 3 | hard |

### CartPole

Ends well (control): end up keeping the pole up for more than 300 steps an episode on average.

| Algorithm | Measure, last tenth (mean of 3 seeds) | Seeds that end well | Marked hard |
|---|---|---|---|
| Semi-gradient SARSA | steps 393 | 3 of 3 |  |
| REINFORCE | steps 228 | 1 of 3 | hard |
| REINFORCE with baseline | steps 478 | 3 of 3 |  |
| Actor–critic | steps 434 | 3 of 3 |  |
| A2C | steps 429 | 3 of 3 |  |
| PPO | steps 438 | 3 of 3 |  |

### Acrobot

Ends well (control): end up getting the tip over the line in under 150 steps on average.

| Algorithm | Measure, last tenth (mean of 3 seeds) | Seeds that end well | Marked hard |
|---|---|---|---|
| Semi-gradient SARSA | steps 118 | 3 of 3 |  |
| REINFORCE | steps 500 | 0 of 3 | hard |
| REINFORCE with baseline | steps 500 | 0 of 3 | hard |
| Actor–critic | steps 94.09 | 3 of 3 |  |
| A2C | steps 101 | 3 of 3 |  |
| PPO | steps 228 | 2 of 3 |  |

### Random walk

| Algorithm | Measure, last tenth (mean of 3 seeds) | Seeds that end well | Marked hard |
|---|---|---|---|
| TD(0) | error 0.06 | — |  |
| Monte Carlo | error 0.08 | — |  |
| n-step TD | error 0.10 | — |  |
| TD(λ) | error 0.09 | — |  |
| Offline λ-return | error 0.09 | — |  |
| Gradient Monte Carlo | error 0.12 | — |  |
| Semi-gradient TD | error 0.06 | — |  |

### 19-state random walk

| Algorithm | Measure, last tenth (mean of 3 seeds) | Seeds that end well | Marked hard |
|---|---|---|---|
| TD(0) | error 0.41 | — |  |
| Monte Carlo | error 0.22 | — |  |
| n-step TD | error 0.18 | — |  |
| TD(λ) | error 0.15 | — |  |
| Offline λ-return | error 0.11 | — |  |
| Gradient Monte Carlo | error 0.47 | — |  |
| Semi-gradient TD | error 0.41 | — |  |

### 1000-state random walk

| Algorithm | Measure, last tenth (mean of 3 seeds) | Seeds that end well | Marked hard |
|---|---|---|---|
| TD(0) | ve 0.27 | — |  |
| Monte Carlo | ve 0.21 | — |  |
| n-step TD | ve 0.07 | — |  |
| TD(λ) | ve 0.06 | — |  |
| Offline λ-return | ve 0.06 | — |  |
| Gradient Monte Carlo | ve 0.24 | — |  |
| Semi-gradient TD | ve 0.27 | — |  |

### 10-armed testbed

Ends well (control): end up pulling the best arm most of the time.

| Algorithm | Measure, last tenth (mean of 3 seeds) | Seeds that end well | Marked hard |
|---|---|---|---|
| ε-greedy | optimal 0.89 | 3 of 3 |  |
| Optimistic greedy | optimal 0.89 | 3 of 3 |  |
| UCB | optimal 0.67 | 2 of 3 |  |
| Gradient bandit | optimal 0.99 | 3 of 3 |  |

### Drifting 10-armed testbed

Ends well (control): end up pulling the current best arm most of the time.

| Algorithm | Measure, last tenth (mean of 3 seeds) | Seeds that end well | Marked hard |
|---|---|---|---|
| ε-greedy | optimal 0.66 | 2 of 3 |  |
| Optimistic greedy | optimal 0.66 | 2 of 3 |  |
| UCB | optimal 0.12 | 0 of 3 | hard |
| Gradient bandit | optimal 0.33 | 1 of 3 | hard |
