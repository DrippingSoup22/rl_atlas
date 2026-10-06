+++
summary = "The surprise of one step: how much better or worse things turned out than predicted, measured as the reward plus the next prediction, minus the old prediction."
prereqs = ["td0", "bellman"]
lab = "random-walk"
sources = [
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., §6.1–6.2, §13.5 and §15.4", url = "http://incompleteideas.net/book/the-book-2nd.html" },
  { text = "Dabney, Kurth-Nelson, Uchida, Starkweather, Hassabis, Munos & Botvinick (2020), A distributional code for value in dopamine-based reinforcement learning, Nature 577", url = "https://doi.org/10.1038/s41586-019-1924-6" },
  { text = "Sutton (1988), Learning to predict by the methods of temporal differences, Machine Learning 3", url = "https://doi.org/10.1007/BF00115009" },
  { text = "Schultz, Dayan & Montague (1997), A neural substrate of prediction and reward, Science 275", url = "https://doi.org/10.1126/science.275.5306.1593" },
  { text = "Schulman, Moritz, Levine, Jordan & Abbeel (2016), High-dimensional continuous control using generalized advantage estimation, International Conference on Learning Representations", url = "https://arxiv.org/abs/1506.02438" },
]

[story]
scene = "surprise"
delay = 4
alpha = 0.4
gamma = 1.0
formula = '''\step{1}{\del_t = \rew{R_{t+1}} + \gam\,\val{V(S_{t+1})} - \val{V(S_t)}} \step{2}{\qquad \val{V(S_t)} \leftarrow \val{V(S_t)} + \alp\,\del_t}'''
+++

## Story

::: step {trials = [1], formula = 1}
**A light comes on; four moments later, a reward.** The light comes at a moment nobody can foresee, so before it every prediction is 0. The learner, TD(0) with a step size of 0.4, predicts at each moment how much reward is still to come, and starts knowing nothing. On the first trial nothing surprises it until the reward itself: one TD error, $\del = 1 + 0 - 0 = +1$, at the moment of the reward.
:::

::: step {trials = [1, 2], values = 2, formula = 2}
**The surprise teaches the moment before it.** That $+1$ raised the prediction at the last moment before the reward from 0 to 0.4 (the blue dot). On trial 2 the reward is less of a surprise, $+0.6$, and the step into that moment is a new one: $0 + 0.4 - 0 = +0.4$, one step earlier.
:::

::: step {trials = [1, 2, 4, 6, 10, 20]}
**The surprise travels back toward the light.** Each trial, every prediction moves a step's worth toward the next one, so the bump of TD errors slides earlier, trial after trial, and the reward itself surprises less and less. By trial 20 almost all of the surprise, $+0.98$, comes when the light comes on, and almost none when the reward arrives.
:::

::: step {trials = [20, 60], values = 60}
**After learning, only the light is news.** Every moment after it predicts the reward exactly (the dots, all at 1), so every step along the way has a TD error of 0, the reward included. The light still surprises, $+1.00$: it comes when nothing predicted it, and it announces a reward. The errors of a trial add up to the gap between what was first expected, 0, and what came, 1: at first all of it at the reward, at the end all of it at the light.
:::

::: step {trials = [60], omit = 60}
**And if the reward does not come?** The light still brings its $+1$. Then, at the very moment the reward was due, nothing arrives where 1 was predicted: $\del = 0 + 0 - 1 = -1$, a disappointment, on time. In the 1990s, recordings of dopamine neurons in monkeys showed this exact pattern: a burst for an unexpected reward, a burst that moves to the cue as the cue is learned, and a dip, on time, when a predicted reward fails to come.
:::

## Textbook

### A prediction checks itself {#idea}

A value function is a prediction: $\val{V(S_t)}$ predicts the return that will follow $S_t$. One step later the agent knows a little more. It has received the reward $\rew{R_{t+1}}$ and reached $S_{t+1}$, which carries its own prediction $\val{V(S_{t+1})}$. The **temporal-difference error** compares the two predictions:

$$\del_t = \rew{R_{t+1}} + \gam\,\val{V(S_{t+1})} - \val{V(S_t)}. \label{def}$$

The first two terms are the newer prediction of the return from $S_t$: the part already received, plus the discounted prediction of the rest. The last term is the older prediction. A positive $\del_t$ means things went better than predicted; a negative one, worse. In the atlas, $\del$ is drawn in red, the color of surprise. If $S_{t+1}$ is terminal, $\val{V(S_{t+1})} = 0$.

The TD error is the error that every temporal-difference method corrects. TD(0) moves $\val{V(S_t)}$ a fraction $\alp$ of the way: $\val{V(S_t)} \leftarrow \val{V(S_t)} + \alp\,\del_t$ ([[td0]]).

### Example: a commute {#example}

::: example {#ex-commute} Cycling to work
Every morning you predict how long the ride to work will take; the quantity predicted is the total travel time, and the minutes of each leg play the role of rewards. One morning:

| Situation | Minutes elapsed | Predicted minutes to go | Predicted total |
| --- | --- | --- | --- |
| leaving home | 0 | 25 | 25 |
| at the bridge, into a headwind | 6 | 22 | 28 |
| through the park | 15 | 10 | 25 |
| stuck at roadworks | 20 | 9 | 29 |
| arriving | 29 | 0 | 29 |
:::

The TD error of each leg is the change in the predicted total: the leg's minutes plus the new prediction of the time to go, minus the old one. On the first leg it is $6 + 22 - 25 = +3$; then $9 + 10 - 22 = -3$, $5 + 9 - 10 = +4$, and finally $9 + 0 - 9 = 0$. They add up to $+4$, the difference between the actual total, 29 minutes, and the first prediction, 25.

A Monte Carlo method would wait until arrival and then move every prediction toward the actual outcome: each would be off by the remaining time actually taken. A TD method moves each prediction toward the next one as soon as it is made. When the headwind hits at the bridge, there is no need to wait until arriving to learn that “leaving home” predicted too little; the next prediction already says so.

### The errors add up to the Monte Carlo error {#telescope}

The example's errors summed exactly to the overall error, and that is no accident.

::: lemma {#lem-sum} Telescoping
If the predictions $\val{V}$ do not change during an episode, then $\rew{G_t} - \val{V(S_t)} = \sum_{k=t}^{T-1} \gam^{\,k-t}\,\del_k$.
:::

::: proof
Write $\rew{G_t} = \rew{R_{t+1}} + \gam\,\rew{G_{t+1}}$. Then
$$\rew{G_t} - \val{V(S_t)} = \rew{R_{t+1}} + \gam\,\val{V(S_{t+1})} - \val{V(S_t)} + \gam\,\big(\rew{G_{t+1}} - \val{V(S_{t+1})}\big) = \del_t + \gam\,\big(\rew{G_{t+1}} - \val{V(S_{t+1})}\big),$$
and repeating the step until the terminal state, where $\rew{G_T} = \val{V(S_T)} = 0$, gives the sum.
:::

So the Monte Carlo error of a prediction is the discounted sum of the TD errors that follow it. Monte Carlo methods apply the whole sum at once, at the end of the episode; TD(0) applies only the first term, immediately. Methods in between, which apply the first $n$ terms or a weighted sum of all of them, are $n$-step TD and TD(λ) ([[n-step-td]], [[td-lambda]]). When the predictions do change during the episode, as in TD(0), the identity holds only approximately, the more closely the smaller the step size.

### What a TD error is worth on average {#expectation}

If the predictions are exactly right, $\val{V} = \val{v_\pi}$, the TD error is zero *on average*, though not on every step:

$$\mathbb{E}_\pi\big[\,\del_t \mid S_t = s\,\big] = \mathbb{E}_\pi\big[\,\rew{R_{t+1}} + \gam\,\val{v_\pi(S_{t+1})} \mid S_t = s\,\big] - \val{v_\pi(s)} = 0, \label{zero}$$

because the first term is the right-hand side of the Bellman equation ([[bellman]]). For estimates that are not yet right, the expected TD error is the **Bellman error** of the estimates at $s$: how far they are from satisfying the Bellman equation there. TD learning drives the Bellman error toward zero, one sampled transition at a time.

Conditioned on the action as well, the expected TD error of the true values is the **advantage** of the action ([[value-functions]]):

$$\mathbb{E}_\pi\big[\,\del_t \mid S_t = s,\ A_t = a\,\big] = \val{q_\pi(s,a)} - \val{v_\pi(s)}. \label{advantage}$$

This is why actor–critic methods use the TD error to judge the action just taken: on average, a positive TD error means the action was better than the policy's typical action in that state ([[actor-critic]]). Generalized advantage estimation combines discounted sums of TD errors to trade bias against variance ([[gae]]; Schulman et al., 2016).

### Variants {#variants}

The TD error takes slightly different forms in different methods, always as a target minus the current estimate:

| Method | TD error |
| --- | --- |
| [[td0]] | $\rew{R_{t+1}} + \gam\,\val{V(S_{t+1})} - \val{V(S_t)}$ |
| [[sarsa]] | $\rew{R_{t+1}} + \gam\,\val{Q(S_{t+1}, A_{t+1})} - \val{Q(S_t, A_t)}$ |
| [[expected-sarsa]] | $\rew{R_{t+1}} + \gam \sum_a \pol{\pi(a \mid S_{t+1})}\,\val{Q(S_{t+1}, a)} - \val{Q(S_t, A_t)}$ |
| [[q-learning]] | $\rew{R_{t+1}} + \gam \max_a \val{Q(S_{t+1}, a)} - \val{Q(S_t, A_t)}$ |
| [[actor-critic]] | $\rew{R_{t+1}} + \gam\,\val{V(S_{t+1})} - \val{V(S_t)}$, used to update both the critic and the actor |

### A signal in the brain {#dopamine}

In the 1990s, recordings from dopamine neurons in the midbrains of monkeys showed a striking pattern. The neurons fired a burst when a reward arrived unexpectedly. Once the animal had learned that a cue predicted the reward, they fired at the cue instead, and not at the predicted reward. And when a predicted reward failed to arrive, their activity dipped below its baseline at the moment the reward should have come. This is exactly how a TD error behaves: positive for an unpredicted reward, transferred to the earliest reliable predictor, and negative when a prediction is disappointed (Schultz, Dayan & Montague, 1997). The *reward prediction error hypothesis* of dopamine is one of the best-known points of contact between reinforcement learning and neuroscience. The story has continued: individual dopamine neurons turn out to differ in how optimistic they are, some signalling surprise relative to a high prediction and some to a low one, so that together they represent the whole distribution of possible rewards, as distributional reinforcement learning does ([[return]]; Dabney et al., 2020).

## Card

### Idea

The TD error is the surprise of one step: the reward plus the new prediction of what follows, minus the old prediction. Positive: better than expected. Negative: worse. Every TD method learns by nudging its prediction a fraction of the way to cancel it.

::: analogy
Your phone's arrival-time estimate on a drive. Each time it revises the estimate, the revision is a TD error: the minutes since the last update plus the new estimate, minus the old one.
:::

### In symbols {#formula}

$$\del_t = \rew{R_{t+1}} + \gam\,\val{V(S_{t+1})} - \val{V(S_t)} \qquad \rew{G_t} - \val{V(S_t)} = \sum_{k=t}^{T-1} \gam^{\,k-t}\,\del_k \ \text{(if } \val{V} \text{ stays fixed)}$$

With the true values, the TD error is zero on average, and its average given the action is the advantage $\val{q_\pi(s,a)} - \val{v_\pi(s)}$.

### Why it matters {#why}

It is the learning signal of TD(0), SARSA, Q-learning and actor–critic methods, and the building block of n-step returns, TD(λ) and GAE. [See it](lab:random-walk)

### Pitfalls

- Forgetting that the terminal state's value is 0 when computing the last TD error of an episode.
- Reading a single TD error as a verdict: it is noisy; only its average says whether the estimate is wrong.
- Mixing up the sign: the target comes first, $\del = \textit{target} - \textit{estimate}$.

### Check yourself {#check}

::: question
The estimate at a state is 10. The next reward is 2, the next state's estimate is 9, and $\gam = 1$. What is the TD error?
---
$\del = 2 + 9 - 10 = +1$: things look one unit better than predicted.
:::

::: question
If all estimates are correct, is every TD error zero?
---
No. Only their average is zero. Individual transitions still differ from the average because of the randomness of rewards and next states.
:::

::: question
How do the TD errors of an episode relate to the Monte Carlo error of its first prediction?
---
If the estimates stay fixed during the episode, the Monte Carlo error is the discounted sum of all the TD errors that follow.
:::
