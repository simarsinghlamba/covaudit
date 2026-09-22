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
    """Split conformal prediction: one threshold shared by everybody.

    Guarantees at least 1 - alpha coverage on average over everyone, but not
    within any particular group.

    Args:
        model: fitted classifier with ``predict_proba`` and ``classes_``.
        alpha: miscoverage level; 0.1 promises at least 90% coverage.

    Attributes:
        threshold_: set by ``calibrate``; ``inf`` if the calibration pile is too
            small for the promise (then every label is kept).
    """

    def __init__(self, model, alpha=0.1):
        """Store the fitted model and the miscoverage level."""
        self.model = model
        self.alpha = alpha

    def calibrate(self, X_cal, y_cal):
        """Compute the shared threshold from the calibration pile.

        Args:
            X_cal: calibration features (never used for training).
            y_cal: true 0/1 labels of the calibration pile.

        Returns:
            self, so calls can be chained.
        """
        scores = true_label_scores(self.model, X_cal, y_cal)
        self.threshold_ = conformal_threshold(scores, self.alpha)
        return self

    def predict_sets(self, X):
        """Build a prediction set for every person.

        Args:
            X: features of the people to predict for.

        Returns:
            Boolean array of shape (n_people, n_classes); True means the label is
            in the set. Columns follow ``model.classes_``.
        """
        proba = self.model.predict_proba(X)
        return (1.0 - proba) <= self.threshold_


class MondrianConformal:
    """Mondrian conformal prediction: one threshold per group.

    Each group's threshold is computed from that group's calibration people only,
    so each group gets its own 1 - alpha guarantee (on average over calibration
    draws). The group must be known at prediction time.

    Args:
        model: fitted classifier with ``predict_proba`` and ``classes_``.
        alpha: miscoverage level; 0.1 promises at least 90% coverage per group.

    Attributes:
        thresholds_: dict {group: threshold}, set by ``calibrate``. Groups with
            too few calibration people get ``inf`` (every label is kept).
    """

    def __init__(self, model, alpha=0.1):
        """Store the fitted model and the miscoverage level."""
        self.model = model
        self.alpha = alpha

    def calibrate(self, X_cal, y_cal, groups_cal):
        """Compute one threshold per group from that group's calibration scores.

        Args:
            X_cal: calibration features (never used for training).
            y_cal: true 0/1 labels of the calibration pile.
            groups_cal: group of each calibration person (same length as y_cal).

        Returns:
            self, so calls can be chained.
        """
        scores = true_label_scores(self.model, X_cal, y_cal)
        groups_cal = np.asarray(groups_cal)
        self.thresholds_ = {
            g: conformal_threshold(scores[groups_cal == g], self.alpha)
            for g in np.unique(groups_cal)
        }
        return self

    def predict_sets(self, X, groups):
        """Build a prediction set for every person using their group's threshold.

        Args:
            X: features of the people to predict for.
            groups: group of each person. A group never seen in calibration gets
                an infinite threshold: a full set, never an error.

        Returns:
            Boolean array of shape (n_people, n_classes); True means the label is
            in the set. Columns follow ``model.classes_``.
        """
        proba = self.model.predict_proba(X)
        groups = np.asarray(groups)
        # Unseen group -> infinite threshold -> full set (honest, never a crash).
        t = np.array([self.thresholds_.get(g, np.inf) for g in groups])
        return (1.0 - proba) <= t[:, None]
