"""
Minimal numpy-only MLP with manual backprop + Adam.
No torch is available in this environment, so everything (forward, backward,
optimizer) is implemented by hand. This is the shared backbone used for both
the Flow Matching VelocityNet and the DDPM epsilon-network baseline.
"""
import numpy as np


def silu(x):
    s = 1.0 / (1.0 + np.exp(-x))
    return x * s, s


def silu_grad(x, s):
    # d/dx [x*sigmoid(x)] = sigmoid(x) + x*sigmoid(x)*(1-sigmoid(x))
    return s + x * s * (1 - s)


class MLP:
    """
    Simple fully-connected net: Linear -> SiLU -> ... -> Linear (no output act).
    dims: e.g. [3, 128, 128, 128, 2]
    """

    def __init__(self, dims, seed=0):
        rng = np.random.default_rng(seed)
        self.dims = dims
        self.W = []
        self.b = []
        for i in range(len(dims) - 1):
            fan_in = dims[i]
            # He init, good match for SiLU/ReLU-family activations
            W = rng.normal(0, np.sqrt(2.0 / fan_in), size=(dims[i], dims[i + 1]))
            b = np.zeros(dims[i + 1])
            self.W.append(W)
            self.b.append(b)
        # Adam state
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]
        self.t = 0

    def forward(self, x):
        """x: (B, dims[0]). Returns output (B, dims[-1]) and cache for backward."""
        cache = {"a": [x], "z": [], "s": []}
        h = x
        n_layers = len(self.W)
        for i in range(n_layers):
            z = h @ self.W[i] + self.b[i]
            cache["z"].append(z)
            if i < n_layers - 1:
                h, s = silu(z)
                cache["s"].append(s)
            else:
                h = z  # linear output layer
            cache["a"].append(h)
        return h, cache

    def backward(self, cache, dout):
        """dout: gradient wrt output (B, dims[-1]). Returns grads for W, b."""
        n_layers = len(self.W)
        dW = [None] * n_layers
        db = [None] * n_layers
        dh = dout
        for i in reversed(range(n_layers)):
            if i < n_layers - 1:
                dz = dh * silu_grad(cache["z"][i], cache["s"][i])
            else:
                dz = dh
            a_prev = cache["a"][i]
            # ``dout`` supplied by the training loops is already the gradient
            # of a batch-mean loss (it contains the 1 / batch_size factor).
            # Do not average here a second time.
            dW[i] = a_prev.T @ dz
            db[i] = dz.sum(axis=0)
            dh = dz @ self.W[i].T
        return dW, db

    def adam_step(self, dW, db, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        for i in range(len(self.W)):
            self.mW[i] = beta1 * self.mW[i] + (1 - beta1) * dW[i]
            self.vW[i] = beta2 * self.vW[i] + (1 - beta2) * (dW[i] ** 2)
            mhat = self.mW[i] / (1 - beta1 ** self.t)
            vhat = self.vW[i] / (1 - beta2 ** self.t)
            self.W[i] -= lr * mhat / (np.sqrt(vhat) + eps)

            self.mb[i] = beta1 * self.mb[i] + (1 - beta1) * db[i]
            self.vb[i] = beta2 * self.vb[i] + (1 - beta2) * (db[i] ** 2)
            mhat_b = self.mb[i] / (1 - beta1 ** self.t)
            vhat_b = self.vb[i] / (1 - beta2 ** self.t)
            self.b[i] -= lr * mhat_b / (np.sqrt(vhat_b) + eps)

    def predict(self, x):
        out, _ = self.forward(x)
        return out
