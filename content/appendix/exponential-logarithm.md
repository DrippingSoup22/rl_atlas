+++
title = "Exponentials and logarithms"
summary = "Repeated multiplication, and its undoing. Discount weights fall exponentially, the logarithm tells how many steps until they halve, and the special number e makes both convenient."
group = "Calculus and optimization"
sources = [
  { text = "Stewart, Calculus: Early Transcendentals, §1.4 and §1.5 (exponential and logarithmic functions)" },
]
+++

### Powers and exponential decay {#powers}

For a number $a > 0$ and a whole number $k$, the power $a^k$ is $a$ multiplied by itself $k$ times, with $a^0 = 1$. Powers obey

$$a^{j+k} = a^j\,a^k, \qquad (a^j)^k = a^{jk}, \qquad a^{-k} = \frac{1}{a^k}.$$

A quantity that is multiplied by the same factor at every step grows or decays **exponentially**. The discount weights $\gam^k$ are the example of the atlas ([[discount]]): with $\gam = 0.9$ they go $1, 0.9, 0.81, 0.73, \ldots$, each a tenth smaller than the one before. Exponential decay is fast in the long run: $0.9^{50} \approx 0.005$. Exponential growth, with a factor above 1, is explosive ([[a:growth-rates|growth rates]]).

### The number e {#e}

One base is special: $e \approx 2.718$. The **exponential function** $e^x$, also written $\exp(x)$, is defined for every real $x$, is always positive, and grows faster and faster. Its defining property is that its rate of change equals its value, which makes it the natural base for anything that grows or decays in proportion to its size. Any other base can be written with it, $a^x = e^{x \ln a}$.

The exponential function appears in the atlas wherever scores must become positive weights: the softmax policy gives each action a weight $e^{\val{Q(s,a)}/\tau}$ ([[a:softmax|softmax]]), and the normal density has $e^{-(x-\mu)^2/2\sigma^2}$ at its heart ([[a:probability-density|densities]]).

### Logarithms {#log}

The **logarithm** undoes the power: $\log_a y$ is the exponent to which $a$ must be raised to give $y$. So $\log_{10} 1000 = 3$ and $\log_2 8 = 3$. The **natural logarithm** $\ln$ uses the base $e$: $\ln e^x = x$ and $e^{\ln y} = y$ for $y > 0$.

Logarithms turn multiplication into addition, which is why they make long products manageable:

$$\ln(xy) = \ln x + \ln y, \qquad \ln\frac{x}{y} = \ln x - \ln y, \qquad \ln(x^k) = k\,\ln x.$$

The logarithm grows very slowly: $\ln 10 \approx 2.3$, $\ln 1000 \approx 6.9$, $\ln 1\,000\,000 \approx 13.8$. That slowness is what makes a regret growing like $\ln t$ so much better than one growing like $t$ ([[explore-exploit]]), and why the exploration bonus $\sqrt{\ln t / N}$ of UCB shrinks for an action as soon as it is tried a few times.

### Solving for a number of steps {#solve}

The logarithm answers questions of the form “after how many steps has the weight fallen to a given level?”. For the discount weights to fall to one half, $\gam^k = \tfrac12$. Taking logarithms of both sides, $k \ln \gam = \ln \tfrac12 = -\ln 2$, so

$$k = \frac{\ln 2}{\ln(1/\gam)}. \label{half-life}$$

For $\gam = 0.9$ this is $0.693 / 0.105 \approx 6.6$ steps; for $\gam = 0.99$, about 69 steps. This **half-life** is a second way, with the effective horizon $1/(1 - \gam)$, to say how far ahead a discounted agent looks. For $\gam$ close to 1, $\ln(1/\gam) \approx 1 - \gam$, so the half-life is about $0.69 / (1-\gam)$: roughly 70% of the effective horizon.
