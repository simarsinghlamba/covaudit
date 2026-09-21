"""Synthetic data with one deliberately hard group (the truth is known)."""
import numpy as np


def make_hard_group_data(n, seed=0, hard_share=0.2, easy_noise=0.5, hard_noise=2.5):
    """Group 1 has much noisier labels, so the model is less sure there.

    Returns (X, y, groups): X has columns [x0, x1, group], y is 0/1.
    """
    rng = np.random.default_rng(seed)
    groups = rng.choice([0, 1], size=n, p=[1 - hard_share, hard_share])
    x = rng.normal(size=(n, 2))
    noise = np.where(groups == 1, hard_noise, easy_noise)
    y = (x[:, 0] + noise * rng.normal(size=n) > 0).astype(int)
    X = np.column_stack([x, groups])
    return X, y, groups
