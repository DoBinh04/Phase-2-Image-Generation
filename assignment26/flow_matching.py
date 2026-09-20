import numpy as np
from nn import MLP
from data import sample_8gaussians


def train_flow_matching(n_steps=8000, batch_size=512, lr=1e-3, hidden=128, seed=0):
    """
    Conditional Flow Matching with the (rectified-flow-style) linear / OT path:
        x0 ~ N(0, I)                     (noise)
        x1 ~ data                        (target sample)
        x_t = (1 - t) * x0 + t * x1,  t ~ U(0, 1)
        target velocity: u_t = x1 - x0    (constant along the path!)
    Loss: E || v_theta(x_t, t) - (x1 - x0) ||^2
    """
    net = MLP([3, hidden, hidden, hidden, 2], seed=seed)
    rng = np.random.default_rng(seed + 1)
    losses = []
    for step in range(n_steps):
        x1 = sample_8gaussians(batch_size, seed=None)
        x0 = rng.normal(size=(batch_size, 2))
        t = rng.uniform(0, 1, size=(batch_size, 1))

        xt = (1 - t) * x0 + t * x1
        target_v = x1 - x0

        inp = np.concatenate([xt, t], axis=1)
        pred_v, cache = net.forward(inp)

        diff = pred_v - target_v
        loss = np.mean(np.sum(diff ** 2, axis=1))
        dout = 2 * diff / batch_size  # dL/dpred, L = mean_b sum_d diff^2

        dW, db = net.backward(cache, dout)
        net.adam_step(dW, db, lr=lr)
        losses.append(loss)
    return net, losses


def euler_sample(net, n_samples, n_steps, seed=0, x_start=None):
    """
    Integrate dx/dt = v_theta(x, t) from t=0 to t=1 with n_steps Euler steps.
    Returns full trajectory: array of shape (n_steps+1, n_samples, 2)
    and also the (x0, x1) endpoint pairs.
    """
    rng = np.random.default_rng(seed)
    if x_start is None:
        x = rng.normal(size=(n_samples, 2))
    else:
        x = x_start.copy()
    x0 = x.copy()
    dt = 1.0 / n_steps
    traj = np.zeros((n_steps + 1, n_samples, 2))
    traj[0] = x
    t = 0.0
    for i in range(n_steps):
        tt = np.full((n_samples, 1), t)
        inp = np.concatenate([x, tt], axis=1)
        v = net.predict(inp)
        x = x + v * dt
        t += dt
        traj[i + 1] = x
    return traj, x0, x
