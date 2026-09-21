"""Coverage, set size, confidence intervals and the per-group audit table."""
import numpy as np


def is_covered(y, sets, classes):
    """True where the true label is inside the prediction set."""
    col = np.searchsorted(np.asarray(classes), np.asarray(y))
    return sets[np.arange(len(col)), col]


def coverage(y, sets, classes):
    """Fraction of people whose prediction set contains their true label."""
    return float(np.mean(is_covered(y, sets, classes)))


def average_set_size(sets):
    """Average number of labels per prediction set."""
    return float(np.mean(sets.sum(axis=1)))
