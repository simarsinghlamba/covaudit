"""Coverage, set size, confidence intervals and the per-group audit table."""
import numpy as np
import pandas as pd
from scipy.stats import binomtest


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


def clopper_pearson(k, n, confidence=0.95):
    """Exact confidence interval for k successes out of n.

    Returns (low, high). An empty group (n = 0) gives (0.0, 1.0): nothing is known.
    """
    if n == 0:
        return (0.0, 1.0)
    ci = binomtest(int(k), int(n)).proportion_ci(confidence_level=confidence,
                                                 method="exact")
    return (float(ci.low), float(ci.high))


def coverage_status(cov, ci_high, alpha):
    """FAIL if even the interval's upper end is below 1 - alpha;
    LOW if coverage is below 1 - alpha but chance could explain it; else OK.
    """
    target = 1 - alpha
    if ci_high < target:
        return "FAIL"
    if cov < target:
        return "LOW"
    return "OK"


def group_coverage_table(y, sets, groups, classes, alpha=0.1, confidence=0.95):
    """One row per group plus an 'ALL' row: n, covered, coverage, interval,
    average set size and FAIL/LOW/OK status.
    """
    covered = is_covered(y, sets, classes)
    sizes = sets.sum(axis=1)
    groups = np.asarray(groups)
    rows = []
    for name, mask in [("ALL", np.ones(len(groups), bool))] + [
        (g, groups == g) for g in np.unique(groups)
    ]:
        n = int(mask.sum())
        k = int(covered[mask].sum())
        cov = k / n if n else float("nan")
        low, high = clopper_pearson(k, n, confidence)
        rows.append({
            "group": name, "n": n, "covered": k, "coverage": cov,
            "ci_low": low, "ci_high": high,
            "avg_set_size": float(sizes[mask].mean()) if n else float("nan"),
            "status": coverage_status(cov, high, alpha) if n else "EMPTY",
        })
    return pd.DataFrame(rows)
