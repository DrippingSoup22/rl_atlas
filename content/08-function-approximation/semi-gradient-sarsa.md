+++
summary = "SARSA with weights: estimate action values from features of the state and the action, act ε-greedily on them, and after each step move the weights toward the reward plus the estimate of the next state and action. It is how an underpowered car learns to climb out of a valley it cannot drive straight out of, from a landscape of expected costs that it digs itself."
change = "Learn action values q̂(s, a, w) instead of state values, choose actions ε-greedily on them, and use the next action's estimate in the target: R + γ q̂(S′, A′, w). Control instead of prediction."
prereqs = ["semi-gradient-td", "sarsa", "features"]
lab = "mountain-car"
sources = [
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., §10.1–10.5 and Example 10.1", url = "http://incompleteideas.net/book/the-book-2nd.html" },
  { text = "Rummery & Niranjan (1994), On-line Q-learning using connectionist systems, Technical Report CUED/F-INFENG/TR 166, Cambridge University" },
  { text = "Sutton (1996), Generalization in reinforcement learning: successful examples using sparse coarse coding, Advances in Neural Information Processing Systems 8", url = "https://papers.nips.cc/paper/1995/hash/8f1d43620bc6bb580df6e80b0dc05c48-Abstract.html" },
  { text = "Moore (1990), Efficient memory-based learning for robot control, PhD thesis, University of Cambridge" },
  { text = "Gordon (1996), Chattering in SARSA(λ), CMU Learning Lab Internal Report" },
  { text = "Mahadevan (1996), Average reward reinforcement learning: foundations, algorithms, and empirical results, Machine Learning 22", url = "https://doi.org/10.1007/BF00114727" },
]

[story]
scene = "car"
env = "mountain-car"
seed = 1
average = 10
formula = '\step{1}{\val{\hat q(s, a, \mathbf w)} = \mathbf w_a^\top \mathbf x(s)} \step{2}{\qquad \mathbf w \leftarrow \mathbf w + \alp\,\big[\rew{R} + \gam\,\val{\hat q(S^\prime, A^\prime, \mathbf w)} - \val{\hat q(S, A, \mathbf w)}\big]\,\nabla \val{\hat q(S, A, \mathbf w)}}'

[story.runs]
one = { algorithm = "semi-gradient-sarsa", features = "tiles", tilings = 8, cells = 8, alpha = 0.06, epsilon = 0.0, gamma = 1.0, units = 500, name = "one-step SARSA" }
eight = { algorithm = "semi-gradient-sarsa", features = "tiles", tilings = 8, cells = 8, alpha = 0.04, n = 8, epsilon = 0.0, gamma = 1.0, units = 500, name = "8-step SARSA" }
+++

## Story

::: step {run = "one", at = 0, path = false}
**A car stuck in a valley.** Its engine is too weak to drive up the right slope to the flag: gravity wins. The only way out is to back up the left slope first and come down with speed. Every step costs $\rew{-1}$, so the goal is to get out in as few steps as possible. The state is two numbers, position and speed, so there are infinitely many states: no table will do.
:::

::: step {run = "one", at = 0, path = false, formula = 1}
On the right, what the learner believes: for each position and speed, how many steps it expects before reaching the flag, the **cost-to-go**. It is computed from action values estimated with tile coding, eight tilings of 8 × 8 tiles over position and speed, one set of weights per action. All weights start at 0: the learner believes it is already there. The landscape is flat.
:::

::: step {run = "one", at = 0, play = 1, pace = 3, formula = 2}
The first episode. Every step costs 1 and disappoints the estimate it came from, so the moves the car has tried sink below the untried ones, which still promise 0. Greedy on that optimism, with no random exploration at all ($\varepsilon = 0$), it keeps trying what it has not tried: watch the swings widen and the landscape rise under the path, until a swing carries it over the top. 1358 steps.
:::

::: step {run = "one", at = 12}
After 12 episodes the landscape is a mountain, highest where the car sits at rest in the valley: from there it expects the longest trip. The path of the last episode spirals outward through the states: back, forward, back again higher. 235 steps.
:::

::: step {run = "one", at = 100, look = "map"}
The same landscape **seen from above**, position across and speed up and down. The episode's path spirals outward from the valley floor. States near the flag, or moving fast toward it, are cheap; slow states at the bottom are dear. 149 steps.
:::

::: step {run = "one", at = 500}
After 500 episodes: 105 steps, a short push forward, a long reverse up the left slope, then full throttle all the way. The learner was never told to back up first; the landscape it dug made it the cheapest way out.
:::

::: step {run = "one", at = 500, curves = ["one", "eight"], metric = "steps"}
Averaged over 10 runs: **eight-step SARSA** ([[n-step-sarsa]]) passes each disappointment eight steps back at once, and gets out faster from the first episodes on. [Drive it yourself in the Lab](lab:mountain-car), and turn the landscape with the mouse.
:::

## Textbook

### Action values with weights {#idea}

To control, not just predict, the learner needs action values, and with function approximation those become $\val{\hat q(s, a, \mathbf w)} \approx \val{q_\pi(s, a)}$. The simplest construction keeps the state features and gives each action its own copy of the weights:

$$\val{\hat q(s, a, \mathbf w)} = \mathbf w_a^\top \mathbf x(s), \label{eq-q}$$

so an update for action $a$ moves only $\mathbf w_a$, and generalization happens across states, not across actions. That suits a few discrete actions, like Mountain Car's three. When actions are many or similar to one another, features of the pair $\mathbf x(s, a)$ can let actions share what they learn too.

The update is [[semi-gradient-td]]'s, applied to action values with the [[sarsa|SARSA]] target, which uses the action actually taken next:

$$\mathbf w \leftarrow \mathbf w + \alp\,\big[\,\rew{R_{t+1}} + \gam\,\val{\hat q(S_{t+1}, A_{t+1}, \mathbf w)} - \val{\hat q(S_t, A_t, \mathbf w)}\,\big]\,\nabla \val{\hat q(S_t, A_t, \mathbf w)}. \label{eq-update}$$

At the end of an episode the target is just the last reward.

### The algorithm {#algorithm}

::: algorithm {#alg-sarsa} Episodic semi-gradient SARSA
Input: a differentiable estimate $\val{\hat q(s, a, \mathbf w)}$, a step size $\alp > 0$, an exploration rate $\eps > 0$
Set the weights $\mathbf w$ (for example, all 0)
Repeat for each episode:
  Start: $S$; choose $A$ $\eps$-greedily from $\val{\hat q(S, \cdot, \mathbf w)}$
  Repeat for each step:
    Take $A$, observe $\rew{R}$ and $S'$
    If $S'$ is terminal: $\mathbf w \leftarrow \mathbf w + \alp\,[\rew{R} - \val{\hat q(S, A, \mathbf w)}]\,\nabla \val{\hat q(S, A, \mathbf w)}$, and the episode ends
    Choose $A'$ $\eps$-greedily from $\val{\hat q(S', \cdot, \mathbf w)}$
    $\mathbf w \leftarrow \mathbf w + \alp\,[\rew{R} + \gam\,\val{\hat q(S', A', \mathbf w)} - \val{\hat q(S, A, \mathbf w)}]\,\nabla \val{\hat q(S, A, \mathbf w)}$
    $S \leftarrow S'$, $A \leftarrow A'$
:::

### Control with approximate values {#control}

The recipe is [[gpi|generalized policy iteration]] as before: the values follow the policy, the policy follows the values. What changes is the guarantee. Even with linear features and on-policy data, nothing ensures that semi-gradient SARSA converges: each change of the weights can flip greedy actions in many states at once, which changes the data, which changes the weights. The weights can end up **chattering**, wandering inside a region instead of settling (Gordon, 1996). In practice the region is usually a good one, and the method works well on many problems; it is the workhorse of control with linear features.

Exploration has a familiar ally. When every step costs $-1$ and all weights start at 0, every estimate starts **optimistic**: the learner believes it is one step from the goal. Each step it takes disappoints the estimates of the states and actions it used, and the untried ones look better by comparison, so a purely greedy learner explores systematically, as optimistic initial values did for bandits ([[optimistic-init]]). With function approximation the optimism also fades from the neighbors of visited states, through the shared tiles, so it explores regions rather than single states. Start the estimates below every real value and the ally turns against you. With $\gam = 0.99$, no state is worth less than $-100$, the value of paying 1 forever; estimates that start at $-150$ rise with every move the car tries, so it keeps repeating them and stays at the bottom. In 19 of 20 runs of 200 episodes it never got out at all, while all 20 that started at 0 did ([the odds, in the Lab](lab:mountain-car-optimism)).

### Example: Mountain Car {#example}

::: example {#ex-car} Mountain Car
An underpowered car must reach the top of a hill on the right of a valley. The state is its position $x \in [-1.2, 0.5]$ and speed $\dot x \in [-0.07, 0.07]$; the actions are full throttle backward, zero throttle and full throttle forward ($A \in \{-1, 0, +1\}$). Each step,

$$\dot x \leftarrow \operatorname{clip}\big(\dot x + 0.001\,A - 0.0025 \cos(3x)\big), \qquad x \leftarrow \operatorname{clip}(x + \dot x),$$

where gravity, the cosine term, outweighs the engine. Hitting the left wall stops the car; reaching $x = 0.5$ ends the episode. Every step pays $\rew{-1}$, and episodes start at rest at a random position in $[-0.6, -0.4]$, at the bottom of the valley.
:::

The task is hard in an instructive way: the best first move is *away* from the goal, and things get worse (higher up the wrong slope) before they get better. A learner must assign value to building speed, which pays only much later. The Lab uses tile coding with eight tilings of 8 × 8 tiles over position and speed (with the shifts, each tiling has 9 × 9 tiles: 648 weights per action), step size $\alpha = 0.06$ per weight, about $0.5/8$, and $\varepsilon = 0$, relying on the optimism of zero estimates.

::: figure {#fig-surfaces}
{{car-surfaces}}
The cost-to-go, $-\max_a \hat q(s, a, \mathbf w)$, after 1, 12, 104 and 1000 episodes of one run. It grows from flat to a mountain whose ridge spirals around the valley floor: the states from which the trip out is longest. Computed live by the Lab; after Sutton & Barto, Figure 10.1.
:::

::: figure {#fig-curves}
{{car-curves}}
Steps per episode for three step sizes (per weight; there are eight tilings). The largest of the three learns fastest. Computed live by the Lab; after Sutton & Barto, Figure 10.2.
:::

### $n$-step semi-gradient SARSA {#n-step}

The [[n-step-sarsa|$n$-step]] target replaces the one reward by $n$ of them and bootstraps from the action value $n$ steps later:

$$\rew{G_{t:t+n}} = \rew{R_{t+1}} + \gam\,\rew{R_{t+2}} + \dots + \gam^{n-1}\,\rew{R_{t+n}} + \gam^n\,\val{\hat q(S_{t+n}, A_{t+n}, \mathbf w)}, \label{eq-n}$$

with the update made at time $t + n$. On Mountain Car, where the consequences of a push show up dozens of steps later, passing each surprise several steps back at once speeds learning markedly, and an intermediate $n$ does best.

::: figure {#fig-n}
{{car-curves n}}
One-step and eight-step semi-gradient SARSA on Mountain Car, each with a good step size. Computed live by the Lab; after Sutton & Barto, Figure 10.3.
:::

### Tasks that never end {#average-reward}

In a continuing task with function approximation, discounting runs into a conceptual problem: with states that cannot be told apart exactly, there is no clear sense in which one policy is better than another state by state, and the discounted objective, averaged over the states the policy visits, turns out to rank policies exactly as their **average reward per step** does, whatever $\gamma$. That makes the average reward $r(\pi)$ the natural objective, with **differential** values measuring how much better than average things will go from a state. The SARSA update then replaces discounting by subtracting a running estimate $\bar R$ of the average reward:

$$\del = \rew{R_{t+1}} - \bar R + \val{\hat q(S_{t+1}, A_{t+1}, \mathbf w)} - \val{\hat q(S_t, A_t, \mathbf w)}, \qquad \bar R \leftarrow \bar R + \beta\,\del, \qquad \mathbf w \leftarrow \mathbf w + \alp\,\del\,\nabla \val{\hat q(S_t, A_t, \mathbf w)}. \label{eq-diff}$$

Episodic tasks, like Mountain Car, keep the familiar form. Policy-gradient methods meet the average reward again ([[pg-theorem]]).

### Historical remarks {#history}

SARSA with function approximation was introduced by Rummery and Niranjan (1994), who trained neural networks with it under the name modified connectionist Q-learning. Mountain Car comes from Moore's thesis (1990); Sutton (1996) solved it with SARSA and tile coding, the setup the Lab reproduces, at a time when such problems were thought to defeat function approximation. Gordon (1996) observed that the weights of SARSA can chatter without converging. Average-reward reinforcement learning was surveyed by Mahadevan (1996).

## Card

### Idea

SARSA with weights: action values computed from features, ε-greedy actions, and after each step the weights move toward the reward plus the estimate of the next state and action. Linear features and tile coding make it the standard way to control in continuous state spaces.

::: analogy
Learning to get a stuck car out of a ditch by rocking it: no one tells you to reverse first, but after a few tries you know which positions and speeds lead out quickly.
:::

### The update {#update}

$$\mathbf w \leftarrow \mathbf w + \alp\,\big[\,\rew{R} + \gam\,\val{\hat q(S', A', \mathbf w)} - \val{\hat q(S, A, \mathbf w)}\,\big]\,\nabla \val{\hat q(S, A, \mathbf w)}$$

With one copy of the weights per action, $\nabla \val{\hat q(S, A, \mathbf w)}$ is $\mathbf x(S)$ in $A$'s copy and 0 elsewhere.

### One change from semi-gradient TD {#change}

| | estimates | target | purpose |
| --- | --- | --- | --- |
| [[semi-gradient-td]] | $\val{\hat v(s, \mathbf w)}$ | $\rew{R} + \gam\,\val{\hat v(S', \mathbf w)}$ | prediction |
| Semi-gradient SARSA | $\val{\hat q(s, a, \mathbf w)}$ | $\rew{R} + \gam\,\val{\hat q(S', A', \mathbf w)}$ | control, ε-greedy |

### Backup diagram {#backup}

{{backup sarsa}}

SARSA's backup: one step, the next action chosen by the same policy; the update writes to the weights.

### Pseudocode

::: pseudocode
Parameters: step size $\alp$, exploration $\eps$, discount $\gam$; features $\mathbf x(s)$ and one weight vector $\mathbf w_a$ per action
Set $\mathbf w$ (here 0)
Repeat for each episode:
  Start: $S$ {#start}
  Choose $A$ $\eps$-greedily from $\val{\hat q(S, \cdot, \mathbf w)}$ {#choose}
  Repeat until $S$ is terminal:
    Take $A$, observe $\rew{R}$ and $S'$ {#act}
    Choose $A'$ $\eps$-greedily from $\val{\hat q(S', \cdot, \mathbf w)}$ (none at the end) {#choose-next}
    $\mathbf w_A \leftarrow \mathbf w_A + \alp\,[\rew{R} + \gam\,\val{\hat q(S', A', \mathbf w)} - \val{\hat q(S, A, \mathbf w)}]\,\mathbf x(S)$, then $S, A \leftarrow S', A'$ {#update}
:::

### Perks

- Control in continuous or huge state spaces, online, step by step. [See it](lab:mountain-car)
- With tile coding: cheap, fast and reliable on many classic problems.
- Optimistic zero estimates explore systematically when steps cost.

### Flaws

- No convergence guarantee, even linear and on-policy: the weights may chatter.
- The features must be designed; poor ones make the task unlearnable.
- On-policy: it learns the value of the exploring policy, as tabular SARSA does.

### Knobs

| Knob | Too low | Too high |
| --- | --- | --- |
| $\alp$ step size (per weight) | slow | unstable; scale it by the number of tilings |
| $\eps$ exploration | relies on optimism alone | wasted steps, and the values of a sloppier policy |
| $n$ steps ahead | slow credit for delayed effects | high variance, longer waits |
| tilings and tile size | coarse control | slow generalization, many weights |

### Pitfalls

- Pessimistic initial weights with $\varepsilon = 0$: the car never discovers the way out.
- A step size that ignores the number of active features.
- Expecting the weights to settle: watch the steps per episode instead.

### Check yourself {#check}

::: question
On Mountain Car with all weights at 0 and $\varepsilon = 0$, why does the car explore at all?
---
Every estimate starts at 0, more than any real value (every step costs $-1$). The moves the car tries lose value, so untried moves, still at 0, look better and get chosen: optimism drives exploration.
:::

::: question
With one copy of the weights per action, which weights does an update for action $A$ in state $S$ change?
---
Only $A$'s copy, and in it only the weights of the features active in $S$ (with tile coding, one tile per tiling).
:::

::: question
Why does $n$-step SARSA help on Mountain Car?
---
The value of a push shows up many steps later. With $n$ steps, each update passes the observed costs $n$ steps back at once instead of one step per episode visit, so the cost-to-go landscape fills in sooner.
:::
