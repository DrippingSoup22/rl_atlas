"""Small neural networks in NumPy: fully connected layers, their gradients, and the Adam optimizer.

The networks the recorder trains have two hidden layers of a few dozen units, small enough that plain NumPy is fast,
and every result can be reproduced bit for bit from a seed.
"""

from __future__ import annotations

import numpy as np


class MLP:
    """x → hidden layers (ReLU or tanh) → a linear output. Weights are lists of arrays, so copies are cheap."""

    def __init__(self, sizes: list[int], rng: np.random.Generator, act: str = "relu", out_scale: float = 1.0) -> None:
        self.act = act
        self.W, self.b = [], []
        for i, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            last = i == len(sizes) - 2
            # He initialization for ReLU, Glorot for tanh; the output layer can start small (a policy close to uniform)
            scale = (np.sqrt(2 / n_in) if act == "relu" else np.sqrt(1 / n_in)) * (out_scale if last else 1.0)
            self.W.append(rng.normal(0.0, scale, (n_in, n_out)))
            self.b.append(np.zeros(n_out))

    @property
    def params(self) -> list[np.ndarray]:
        return self.W + self.b

    def copy_from(self, other: MLP) -> None:
        for mine, theirs in zip(self.params, other.params):
            mine[...] = theirs

    def soft_update(self, other: MLP, tau: float) -> None:
        for mine, theirs in zip(self.params, other.params):
            mine *= 1 - tau
            mine += tau * theirs

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        """The output, and what backward needs: the input of every layer."""
        h, inputs = x, []
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            inputs.append(h)
            h = h @ W + b
            if i < len(self.W) - 1:
                h = np.maximum(h, 0.0) if self.act == "relu" else np.tanh(h)
        return h, inputs

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)[0]

    def backward(self, inputs: list[np.ndarray], dy: np.ndarray) -> list[np.ndarray]:
        """Gradients of a loss with gradient dy at the output, in the order of params (all W, then all b)."""
        dW, db = [None] * len(self.W), [None] * len(self.W)
        g = dy
        for i in range(len(self.W) - 1, -1, -1):
            x = inputs[i]
            dW[i] = x.T @ g
            db[i] = g.sum(axis=0)
            if i > 0:
                g = g @ self.W[i].T
                h = inputs[i]  # the activation that fed layer i
                g = g * (h > 0) if self.act == "relu" else g * (1 - h**2)
        return dW + db

    def input_grad(self, inputs: list[np.ndarray], dy: np.ndarray) -> np.ndarray:
        """The gradient with respect to the network's input, for a gradient dy at the output: how a critic's value moves
        with the action it was given, which is what an actor climbs."""
        g = dy
        for i in range(len(self.W) - 1, -1, -1):
            g = g @ self.W[i].T
            if i > 0:
                h = inputs[i]
                g = g * (h > 0) if self.act == "relu" else g * (1 - h**2)
        return g


class Adam:
    """Adam (Kingma & Ba, 2015) with an optional limit on the norm of each gradient."""

    def __init__(self, params: list[np.ndarray], lr: float, clip: float | None = None) -> None:
        self.params, self.lr, self.clip = params, lr, clip
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads: list[np.ndarray], lr: float | None = None) -> None:
        if self.clip:
            norm = np.sqrt(sum(float((g**2).sum()) for g in grads))
            if norm > self.clip:
                grads = [g * (self.clip / norm) for g in grads]
        self.t += 1
        lr = self.lr if lr is None else lr
        c1, c2 = 1 - 0.9**self.t, 1 - 0.999**self.t
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m *= 0.9
            m += 0.1 * g
            v *= 0.999
            v += 0.001 * g * g
            p -= lr * (m / c1) / (np.sqrt(v / c2) + 1e-8)
