import numpy as np


def straightness_deviation(traj):
    """
    traj: (n_steps+1, n_samples, 2)
    For each sample path, compare it to the straight line between its start
    and end point. Returns the mean (over samples and interior time steps) of
    the perpendicular-ish deviation, normalized by the chord length, i.e.
    a scale-free curvature score. 0 = perfectly straight.
    """
    x_start = traj[0]          # (N,2)
    x_end = traj[-1]           # (N,2)
    chord = x_end - x_start    # (N,2)
    chord_len = np.linalg.norm(chord, axis=1, keepdims=True)
    chord_len = np.maximum(chord_len, 1e-6)
    n_steps = traj.shape[0] - 1

    devs = []
    for i in range(1, n_steps):
        frac = i / n_steps
        straight_pt = x_start + frac * chord
        actual_pt = traj[i]
        dev = np.linalg.norm(actual_pt - straight_pt, axis=1)
        devs.append(dev / chord_len.squeeze())
    if not devs:
        return 0.0
    devs = np.stack(devs, axis=0)  # (n_steps-1, N)
    return devs.mean()


def path_length_ratio(traj):
    """
    Ratio of actual traversed path length to straight-line chord length.
    1.0 = perfectly straight path (minimal possible).
    """
    diffs = traj[1:] - traj[:-1]              # (n_steps, N, 2)
    seg_len = np.linalg.norm(diffs, axis=2)   # (n_steps, N)
    path_len = seg_len.sum(axis=0)            # (N,)
    chord_len = np.linalg.norm(traj[-1] - traj[0], axis=1)
    chord_len = np.maximum(chord_len, 1e-6)
    return (path_len / chord_len).mean()
