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
