"""Split and Mondrian conformal prediction for classification."""
import numpy as np


def conformal_threshold(scores, alpha):
    """k-th smallest score with k = ceil((n + 1)(1 - alpha)).

    Returns infinity when k > n: too little data, so every label is kept.
    """
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if n == 0 or k > n:
        return np.inf
    return float(np.sort(scores)[k - 1])


def true_label_scores(model, X, y):
    """Nonconformity score = 1 - probability the model gave the TRUE label."""
    proba = model.predict_proba(X)
    col = np.searchsorted(model.classes_, np.asarray(y))
    return 1.0 - proba[np.arange(len(col)), col]


class SplitConformal:
    """One threshold shared by everybody."""

    def __init__(self, model, alpha=0.1):
        self.model = model
        self.alpha = alpha

    def calibrate(self, X_cal, y_cal):
        """Compute the shared threshold from the calibration pile; returns self."""
        scores = true_label_scores(self.model, X_cal, y_cal)
        self.threshold_ = conformal_threshold(scores, self.alpha)
        return self

    def predict_sets(self, X):
        """Boolean array (n_people, n_classes): True = label is in the set."""
        proba = self.model.predict_proba(X)
        return (1.0 - proba) <= self.threshold_


class MondrianConformal:
    """One threshold per group, each computed from that group's data only."""

    def __init__(self, model, alpha=0.1):
        self.model = model
        self.alpha = alpha

    def calibrate(self, X_cal, y_cal, groups_cal):
        """Compute one threshold per group from its own calibration scores; returns self."""
        scores = true_label_scores(self.model, X_cal, y_cal)
        groups_cal = np.asarray(groups_cal)
        self.thresholds_ = {
            g: conformal_threshold(scores[groups_cal == g], self.alpha)
            for g in np.unique(groups_cal)
        }
        return self

    def predict_sets(self, X, groups):
        """Boolean array (n_people, n_classes) using each person's group threshold."""
        proba = self.model.predict_proba(X)
        groups = np.asarray(groups)
        # Unseen group -> infinite threshold -> full set (honest, never a crash).
        t = np.array([self.thresholds_.get(g, np.inf) for g in groups])
        return (1.0 - proba) <= t[:, None]
