+++
title = "Derivatives and gradients"
summary = "The derivative is the slope of a function: how fast it changes when its input changes. The gradient collects the slopes in every direction and points uphill, so taking small steps along it climbs to better parameters."
group = "Calculus and optimization"
sources = [
  { text = "Stewart, Calculus: Early Transcendentals, §2.7 (derivatives) and §14.6 (gradients)" },
  { text = "Goodfellow, Bengio & Courville (2016), Deep Learning, MIT Press, §4.3 (gradient-based optimization)", url = "https://www.deeplearningbook.org/" },
]
+++

Many methods in the atlas improve something by nudging numbers in the right direction: the weights of a value function, the parameters of a policy. The derivative says which direction is right.

### The derivative {#derivative}

The **derivative** $f'(x)$ of a function $f$ at a point $x$ is the slope of its graph there: how much $f$ changes per unit change of $x$, for very small changes. Formally it is the limit ([[a:limits|limits]])

$$f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}. \label{derivative}$$

For small steps, the change of $f$ is approximately the slope times the step: $f(x + h) \approx f(x) + f'(x)\,h$. A positive slope means $f$ increases as $x$ increases, a negative one that it decreases, and a slope of zero marks a flat spot, such as the top of a hill.

A few derivatives cover most cases: the derivative of $x^2$ is $2x$, of $x^n$ is $n x^{n-1}$, of $e^x$ is $e^x$ ([[a:exponential-logarithm|exponentials]]), of $\ln x$ is $1/x$, and of a constant is 0. Derivatives respect sums and constant factors, and the **chain rule** handles functions of functions: the derivative of $f(g(x))$ is $f'(g(x))\,g'(x)$.

### Partial derivatives and the gradient {#gradient}

A function of several numbers, $f(\theta_1, \ldots, \theta_d)$, has a slope in each direction. The **partial derivative** $\partial f / \partial \theta_i$ is the derivative with respect to $\theta_i$ alone, all the other inputs held fixed. Collected into a vector ([[a:vectors-matrices|vectors]]), the partial derivatives form the **gradient**:

$$\nabla f(\boldsymbol\theta) = \Big(\frac{\partial f}{\partial \theta_1},\ \ldots,\ \frac{\partial f}{\partial \theta_d}\Big). \label{gradient}$$

The gradient points in the direction in which $f$ increases fastest, and its length is the rate of increase in that direction. Its opposite, $-\nabla f$, points steepest downhill.

::: example {#ex-gradient}
$f(\theta_1, \theta_2) = \theta_1^2 + 3\,\theta_2$ has $\partial f/\partial \theta_1 = 2\theta_1$ and $\partial f/\partial \theta_2 = 3$, so at $(1, 0)$ the gradient is $(2, 3)$: to increase $f$ fastest from there, move mostly in the second coordinate.
:::

### Gradient ascent and descent {#ascent}

To make $f$ larger, take a small step along the gradient; to make it smaller, against it:

$$\boldsymbol\theta \leftarrow \boldsymbol\theta + \alp\,\nabla f(\boldsymbol\theta) \quad \text{(ascent)}, \qquad \boldsymbol\theta \leftarrow \boldsymbol\theta - \alp\,\nabla f(\boldsymbol\theta) \quad \text{(descent)}. \label{steps}$$

The **step size** $\alp$ is a small positive number. Too small, and progress is slow; too large, and a step overshoots the hill it was climbing. Repeated, the steps climb to a point where the gradient is zero, usually a local peak, not necessarily the highest one.

In the atlas, gradient descent fits value functions with parameters by reducing their squared error, and gradient ascent improves parameterized policies by increasing their expected return ([[policy]], [[prediction-control]]). When the gradient can only be estimated from samples, as it nearly always can in reinforcement learning, the method is called *stochastic* gradient ascent or descent: each step follows a noisy estimate of the gradient, and the noise averages out over many small steps.
