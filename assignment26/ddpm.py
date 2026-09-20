import numpy as np
from nn import MLP
from data import sample_8gaussians


def cosine_schedule(T):
    s = 0.008
    steps = np.arange(T + 1)
    f = np.cos(((steps / T) + s) / (1 + s) * np.pi / 2) ** 2
    alphas_cumprod = f / f[0]
    return np.clip(alphas_cumprod, 1e-5, 1.0)


class DDPM:
    def __init__(self, T=1000):
        self.T = T
        self.alphas_cumprod = cosine_schedule(T)  # length T+1, index 0..T


def train_ddpm(n_steps=8000, batch_size=512, lr=1e-3, hidden=128, T=1000, seed=0):
    """
    Standard DDPM: predict the noise eps added to x0 at timestep t.
        x_t = sqrt(abar_t) * x0 + sqrt(1-abar_t) * eps,  eps ~ N(0,I)
    Loss: E || eps_theta(x_t, t/T) - eps ||^2
    Net input is (x_t, t/T) so we can reuse the exact same MLP architecture as
    the Flow Matching model for a fair comparison.
    """
    ddpm = DDPM(T=T)
    net = MLP([3, hidden, hidden, hidden, 2], seed=seed)
    rng = np.random.default_rng(seed + 1)
    losses = []
    for step in range(n_steps):
        x0 = sample_8gaussians(batch_size, seed=None)
        t_idx = rng.integers(1, T + 1, size=batch_size)
        abar = ddpm.alphas_cumprod[t_idx][:, None]
        eps = rng.normal(size=(batch_size, 2))
        xt = np.sqrt(abar) * x0 + np.sqrt(1 - abar) * eps

        t_norm = (t_idx / T)[:, None]
        inp = np.concatenate([xt, t_norm], axis=1)
        pred_eps, cache = net.forward(inp)

        diff = pred_eps - eps
        loss = np.mean(np.sum(diff ** 2, axis=1))
        dout = 2 * diff / batch_size

        dW, db = net.backward(cache, dout)
        net.adam_step(dW, db, lr=lr)
        losses.append(loss)
    return net, ddpm, losses


def ddim_sample(net, ddpm, n_samples, n_steps, seed=0, x_start=None):
    """
    Deterministic DDIM sampling (eta=0), going from t=T (pure noise) to t=0 (data)
    using n_steps evenly spaced timesteps. This gives a genuine ODE trajectory,
    directly comparable to the Flow Matching Euler trajectory.
    Returns traj shaped (n_steps+1, n_samples, 2) ordered so index 0 = noise,
    index -1 = data (matching the Flow Matching trajectory convention).
    """
    rng = np.random.default_rng(seed)
    T = ddpm.T
    abar = ddpm.alphas_cumprod
    if x_start is None:
        x = rng.normal(size=(n_samples, 2))
    else:
        x = x_start.copy()

    # timesteps from T down to 0 (n_steps+1 points)
    ts = np.linspace(T, 0, n_steps + 1).round().astype(int)
    traj = np.zeros((n_steps + 1, n_samples, 2))
    traj[0] = x
    for i in range(n_steps):
        t_cur = max(ts[i], 1)
        t_next = ts[i + 1]
        a_cur = abar[t_cur]
        a_next = abar[t_next] if t_next > 0 else 1.0

        t_norm = np.full((n_samples, 1), t_cur / T)
        inp = np.concatenate([x, t_norm], axis=1)
        eps = net.predict(inp)

        x0_pred = (x - np.sqrt(1 - a_cur) * eps) / np.sqrt(a_cur)
        x = np.sqrt(a_next) * x0_pred + np.sqrt(max(1 - a_next, 0.0)) * eps
        traj[i + 1] = x
    return traj
