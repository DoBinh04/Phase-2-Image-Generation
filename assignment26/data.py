import numpy as np


def sample_8gaussians(n, radius=3.0, std=0.15, seed=None):
    rng = np.random.default_rng(seed)
    k = rng.integers(0, 8, size=n)
    angles = k * (2 * np.pi / 8)
    centers = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
    return centers + std * rng.normal(size=(n, 2))


def all_centers(radius=3.0):
    angles = np.arange(8) * (2 * np.pi / 8)
    return np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
