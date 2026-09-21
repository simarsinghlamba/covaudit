import numpy as np

from covaudit.data import split_indices


def test_split_sizes_add_up():
    s = split_indices(1000, seed=0)
    assert len(s["train"]) + len(s["cal"]) + len(s["test"]) == 1000


def test_splits_never_overlap():
    s = split_indices(1000, seed=0)
    assert not set(s["train"]) & set(s["cal"])
    assert not set(s["train"]) & set(s["test"])
    assert not set(s["cal"]) & set(s["test"])


def test_same_seed_same_split_and_different_seed_different_split():
    a, b, c = split_indices(500, 1), split_indices(500, 1), split_indices(500, 2)
    assert np.array_equal(a["test"], b["test"])
    assert not np.array_equal(a["test"], c["test"])
