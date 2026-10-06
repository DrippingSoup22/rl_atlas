+++
summary = "With fewer weights than states, some estimates must be wrong; the value error decides which mistakes matter, weighing each state's squared error by how often the agent is there. Learning is gradient descent on it, one state at a time. With the return as target that is true stochastic gradient descent; with a bootstrapped target it becomes a semi-gradient method, which ignores that the target moves with the weights."
story_in = ["gradient-mc"]
prereqs = ["why-approximate", "features", "bootstrapping"]
lab = "walk-aggregation"
sources = [
  { text = "Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., §9.2–9.4", url = "http://incompleteideas.net/book/the-book-2nd.html" },
  { text = "Robbins & Monro (1951), A stochastic approximation method, Annals of Mathematical Statistics 22", url = "https://doi.org/10.1214/aoms/1177729586" },
  { text = "Widrow & Hoff (1960), Adaptive switching circuits, IRE WESCON Convention Record 4" },
  { text = "Tsitsiklis & Van Roy (1997), An analysis of temporal-difference learning with function approximation, IEEE Transactions on Automatic Control 42", url = "https://doi.org/10.1109/9.580874" },
  { text = "Bertsekas (2012), Dynamic Programming and Optimal Control, Vol. II, 4th ed., Athena Scientific, §6.3" },
  { text = "Barnard (1993), Temporal-difference methods and Markov models, IEEE Transactions on Systems, Man, and Cybernetics 23", url = "https://doi.org/10.1109/21.229471" },
]
+++

## Textbook

### Which mistakes matter {#ve}

A table can, given time, make every estimate right. A function with fewer weights than states cannot: moving the weights to fix one state shifts others, and in general no setting of the weights is right everywhere ([[why-approximate]]). The learner must trade errors between states, and needs a rule for the trade. The natural rule is to care about a state in proportion to how often it comes up. Let $\mu(s) \ge 0$, with $\sum_s \mu(s) = 1$, be the share of its time the agent spends in state $s$ while following its policy. The **value error** of a set of weights is the squared error of each state, weighted by that share:

$$\overline{\text{VE}}(\mathbf w) = \sum_{s} \mu(s)\,\big[\val{v_\pi(s)} - \val{\hat v(s, \mathbf w)}\big]^2. \label{eq-ve}$$

Its square root, $\sqrt{\overline{\text{VE}}}$, is a typical error in the units of the values, which is how the Lab's charts report it. States the agent rarely visits barely count: being wrong there costs little, because the agent is rarely there to act on the mistake.

::: definition {#def-mu} The on-policy distribution
In a task that never ends, $\mu$ is the stationary distribution of the states under $\pi$: the long-run share of time spent in each. In an episodic task, let $h(s)$ be the probability that an episode starts in $s$ and $\eta(s)$ the expected number of steps spent in $s$ per episode. A visit to $s$ is either a start there or a step from some state $\bar s$ that leads there:

$$\eta(s) = h(s) + \sum_{\bar s} \eta(\bar s) \sum_a \pol{\pi(a \mid \bar s)}\, p(s \mid \bar s, a), \qquad \mu(s) = \frac{\eta(s)}{\sum_{s'} \eta(s')}. \label{eq-mu}$$
:::

On the 1000-state walk, solving \ref{eq-mu} gives an average episode of 83 steps and a $\mu$ shaped like a hill around the start, with a spike at state 500 itself (every episode begins there) and little weight near the ends, which the walk reaches only rarely. With discounting, $\mu$ can also be weighted toward early steps; we keep the plain version.

The value error is not the final goal, a good policy is, and there are reasons to prefer other objectives (some come up with the [[deadly-triad]]). But for learning the values of a fixed policy it is the clearest yardstick, and the methods below descend it.

### Gradient descent, one state at a time {#sgd}

If the true values were known, the obvious learner would follow the gradient of the error on each state it visits: move the weights a little in the direction that most reduces that state's squared error,

$$\mathbf w_{t+1} = \mathbf w_t - \tfrac12 \alp\, \nabla \big[\val{v_\pi(S_t)} - \val{\hat v(S_t, \mathbf w_t)}\big]^2 \qquad = \mathbf w_t + \alp\,\big[\val{v_\pi(S_t)} - \val{\hat v(S_t, \mathbf w_t)}\big]\,\nabla \val{\hat v(S_t, \mathbf w_t)}. \label{eq-sgd}$$

This is **stochastic gradient descent** (SGD): stochastic because each step uses one state, drawn by the agent's own experience, and so with frequency $\mu$, which is exactly why the steps descend $\overline{\text{VE}}$ on average. The step is kept small on purpose. Removing one state's error completely would undo what was learned about the states that share its weights; small steps let the errors of many states be balanced. With step sizes that shrink at the usual rate ($\sum_t \alpha_t = \infty$, $\sum_t \alpha_t^2 < \infty$, after Robbins and Monro, 1951), SGD converges to a **local minimum** of the error, which for linear methods (§\ref{linear}) is a global one. For a weighted sum of features the gradient is the feature vector, and \ref{eq-sgd} is the least-mean-square rule of Widrow and Hoff (1960), the "delta rule".

### Targets instead of true values {#targets}

The true value $\val{v_\pi(S_t)}$ is what we are trying to learn, so the update uses a target $U_t$ in its place, any of the targets of Parts 3 to 7:

$$\mathbf w_{t+1} = \mathbf w_t + \alp\,\big[\,U_t - \val{\hat v(S_t, \mathbf w_t)}\,\big]\,\nabla \val{\hat v(S_t, \mathbf w_t)}. \label{eq-general}$$

Whether this is still gradient descent depends on the target.

- **An unbiased target**, one whose expectation is the true value, $\mathbb E[U_t \mid S_t = s] = \val{v_\pi(s)}$, keeps it SGD: on average each step points where \ref{eq-sgd} would. The return $\rew{G_t}$ is such a target, and the resulting method, [[gradient-mc]], inherits the guarantee: it converges to a local minimum of $\overline{\text{VE}}$.
- **A bootstrapped target**, such as $\rew{R_{t+1}} + \gam\,\val{\hat v(S_{t+1}, \mathbf w_t)}$, is built from the current estimates. It is biased (until the estimates are right), and it depends on the weights. The true gradient of $[U_t - \hat v(S_t, \mathbf w)]^2$ would include the target's own dependence on $\mathbf w$; the update \ref{eq-general} takes only the half that comes from $\hat v(S_t, \mathbf w)$ and treats the target as a fixed number. It is a **semi-gradient** method ([[semi-gradient-td]]).

Semi-gradient methods give up the guarantee of SGD, but not for nothing. They learn online, from each step, with no wait for the end of the episode; they work in tasks that never end; and like TD in the tabular case they usually learn faster, since bootstrapping cuts the variance of the target. In the linear, on-policy case they still converge, though not to the minimum of $\overline{\text{VE}}$.

### The linear case {#linear}

With $\hat v = \mathbf w^\top \mathbf x(s)$, the value error is a quadratic bowl in $\mathbf w$: one minimum (or a flat valley of equally good weights when features are redundant), no false local minima. [[gradient-mc|Gradient Monte Carlo]] finds its bottom.

[[semi-gradient-td|Semi-gradient TD(0)]] settles elsewhere. Write $\mathbf x_t = \mathbf x(S_t)$. In expectation over the on-policy distribution, its update is

$$\mathbb E[\mathbf w_{t+1} \mid \mathbf w_t] = \mathbf w_t + \alp\,(\mathbf b - \mathbf A \mathbf w_t), \qquad \mathbf A = \mathbb E\big[\mathbf x_t (\mathbf x_t - \gam\,\mathbf x_{t+1})^\top\big], \quad \mathbf b = \mathbb E\big[\rew{R_{t+1}}\, \mathbf x_t\big], \label{eq-ab}$$

which stops moving where $\mathbf A \mathbf w = \mathbf b$: the **TD fixed point** $\mathbf w_{\text{TD}} = \mathbf A^{-1} \mathbf b$.

::: theorem {#thm-td} Linear TD converges near the best fit
With linear features, on-policy sampling and suitably shrinking step sizes, semi-gradient TD(0) converges with probability 1 to the TD fixed point, and its error there is at most a factor $1/(1-\gamma)$ worse than the best possible:

$$\overline{\text{VE}}(\mathbf w_{\text{TD}}) \le \frac{1}{1-\gamma}\,\min_{\mathbf w} \overline{\text{VE}}(\mathbf w). \label{eq-bound}$$
:::

::: proof Proof idea
On-policy, the matrix $\mathbf A$ is positive definite: the expected update always has a component pointing toward $\mathbf w_{\text{TD}}$, so the weights are drawn there and cannot run away. The bound comes from viewing the TD solution $\hat v_{\text{TD}} = \hat v(\cdot, \mathbf w_{\text{TD}})$ as the fixed point of a Bellman backup followed by a projection $\Pi$ onto what the features can represent, with distances weighted by $\mu$. Its error splits into two perpendicular parts: the error of the best fit, $\Pi v_\pi - v_\pi$, and the gap from the best fit to $\hat v_{\text{TD}}$. The projected backup shrinks distances by $\gamma$, so the gap is at most $\gamma$ times the whole error, and Pythagoras gives $\overline{\text{VE}}(\mathbf w_{\text{TD}}) \le \min_{\mathbf w} \overline{\text{VE}}(\mathbf w) + \gamma^2\, \overline{\text{VE}}(\mathbf w_{\text{TD}})$. That is a factor $1/(1-\gamma^2)$, slightly sharper than \ref{eq-bound}, since $1-\gamma^2 \ge 1-\gamma$ (Tsitsiklis and Van Roy, 1997; this form after Bertsekas).
:::

With $\gamma$ close to 1 the factor is large (100 for $\gamma = 0.99$), and even the sharper factor is about 50, so the bound alone is not reassuring; in practice TD's fixed point is usually much closer than that, and TD gets there with far less variance. The positive definiteness of $\mathbf A$ is where on-policy sampling matters: weighted by another distribution, $\mathbf A$ can lose it, and then nothing stops the weights from diverging ([[deadly-triad]]).

::: figure {#fig-fit}
{{walk-fit}}
The 1000-state walk with 10 groups: what gradient Monte Carlo and semi-gradient TD settle on. Monte Carlo gives each group the $\mu$-weighted average of its true values, the best staircase in the sense of $\overline{\text{VE}}$. TD's staircase is flatter at the ends: each group bootstraps from neighbors whose estimates are averages too, pulling the extreme groups toward the middle. Computed live by the Lab; after Sutton & Barto, Figures 9.1 and 9.2.
:::

### Historical remarks {#history}

Stochastic approximation, the theory of following noisy estimates of a gradient, began with Robbins and Monro (1951). Widrow and Hoff (1960) introduced the least-mean-square rule for adaptive linear elements, the template of every update in this part. The convergence of linear TD and the bound \ref{eq-bound} are due to Tsitsiklis and Van Roy (1997), after earlier partial results by Sutton (1988) and Dayan (1992); the term semi-gradient is Sutton and Barto's. Barnard (1993) showed that TD's update is not the gradient of any objective function.

## Card

### Idea

With fewer weights than states, errors must be traded between states. The value error weighs each state's squared error by how often it is visited, and learning descends it one state at a time. An unbiased target (the return) makes this true gradient descent; a bootstrapped target makes it a semi-gradient method.

::: analogy
A shop that cannot stock everything stocks what customers ask for most often: a few unhappy customers for rare requests, many happy ones overall.
:::

### The objective and the update {#formula}

$$\overline{\text{VE}}(\mathbf w) = \sum_s \mu(s)\,\big[\val{v_\pi(s)} - \val{\hat v(s, \mathbf w)}\big]^2$$

$$\mathbf w \leftarrow \mathbf w + \alp\,\big[\,U_t - \val{\hat v(S_t, \mathbf w)}\,\big]\,\nabla \val{\hat v(S_t, \mathbf w)}$$

### Which target {#targets}

| target $U_t$ | unbiased? | method | in the linear case converges to |
| --- | --- | --- | --- |
| return $\rew{G_t}$ | yes | [[gradient-mc]] (true SGD) | the minimum of $\overline{\text{VE}}$ |
| $\rew{R} + \gam\,\val{\hat v(S', \mathbf w)}$ | no | [[semi-gradient-td]] | the TD fixed point, within $1/(1-\gamma)$ of it |

### Why it matters {#why}

- It says which states the estimates should get right: the ones the agent actually visits. [See it](lab:walk-aggregation)
- It explains why Monte Carlo and TD settle on different values with the same features.
- On-policy sampling keeps linear TD stable; leaving it is the first step toward divergence.

### Pitfalls

- Reading $\sqrt{\overline{\text{VE}}}$ as the error everywhere: rarely visited states may be far worse.
- Steps too large: each update undoes what the shared weights learned for other states.
- Calling semi-gradient TD gradient descent: it is not the gradient of any objective.

### Check yourself {#check}

::: question
Why is the value error weighted by $\mu$ rather than by every state equally?
---
With fewer weights than states, errors must be traded between states. Weighting by $\mu$ trades them in favor of the states the agent actually visits, where a wrong estimate would actually be used.
:::

::: question
Why is the return a valid target for stochastic gradient descent, and $R + \gamma \hat v(S', \mathbf w)$ not?
---
The return's expectation is the true value, so on average the update is the true gradient step. The bootstrapped target is biased while the estimates are wrong, and it depends on the weights, a dependence the update ignores.
:::

::: question
How far from the best possible can linear semi-gradient TD(0) end, on-policy?
---
Its value error at the TD fixed point is at most $1/(1-\gamma)$ times the smallest possible value error, though in practice it is usually much closer.
:::
