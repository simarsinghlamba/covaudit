"""Coverage, set size, confidence intervals and the per-group audit table."""
import numpy as np
import pandas as pd
from scipy.stats import binomtest


def is_covered(y, sets, classes):
    """True where the true label is inside the prediction set."""
    col = np.searchsorted(np.asarray(classes), np.asarray(y))
    return sets[np.arange(len(col)), col]


def coverage(y, sets, classes):
    """Fraction of people whose prediction set contains their true label.

    Args:
        y: true labels.
        sets: boolean prediction sets, shape (n_people, n_classes).
        classes: class labels in column order (``model.classes_``).

    Returns:
        Coverage as a float between 0 and 1.
    """
    return float(np.mean(is_covered(y, sets, classes)))


def average_set_size(sets):
    """Average number of labels per prediction set (smaller is more useful).

    Args:
        sets: boolean prediction sets, shape (n_people, n_classes).

    Returns:
        Mean set size as a float; between 0 and 2 for yes/no problems.
    """
    return float(np.mean(sets.sum(axis=1)))


def clopper_pearson(k, n, confidence=0.95):
    """Exact (Clopper-Pearson) confidence interval for k successes out of n.

    Guaranteed to reach at least the stated confidence for every true value, so it
    is conservative for small groups.

    Args:
        k: number of successes (people covered).
        n: number of trials (people in the group).
        confidence: confidence level, default 0.95.

    Returns:
        (low, high). An empty group (n = 0) gives (0.0, 1.0): nothing is known.
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
    """The per-group audit: one row per group plus an 'ALL' row.

    Status rule: FAIL if the whole interval is below 1 - alpha; LOW if coverage
    is below 1 - alpha but the interval reaches it; otherwise OK.

    Args:
        y: true labels.
        sets: boolean prediction sets, shape (n_people, n_classes).
        groups: group of each person.
        classes: class labels in column order (``model.classes_``).
        alpha: miscoverage level, default 0.1.
        confidence: interval confidence level, default 0.95.

    Returns:
        DataFrame with columns group, n, covered, coverage, ci_low, ci_high,
        avg_set_size, status.
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


def worst_group_gap(table, alpha=0.1):
    """How far the lowest-covered group falls below 1 - alpha (0 if none).

    Ignores the ALL row and empty groups. Uses raw coverage, not intervals,
    so it is sensitive to very small groups: read it next to the status column.
    """
    groups = table[(table["group"] != "ALL") & (table["n"] > 0)]
    return float(max(0.0, (1 - alpha) - groups["coverage"].min()))
