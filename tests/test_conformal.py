import numpy as np
from sklearn.linear_model import LogisticRegression

from covaudit.conformal import SplitConformal, conformal_threshold
from covaudit.metrics import coverage
from covaudit.synthetic import make_hard_group_data


def test_threshold_matches_hand_example():
    scores = [0.05, 0.10, 0.15, 0.20, 0.30, 0.35, 0.45, 0.60, 0.70, 0.85]
    assert conformal_threshold(scores, alpha=0.2) == 0.70  # k = ceil(11*0.8) = 9


def test_threshold_is_infinite_when_too_little_data():
    assert conformal_threshold(np.arange(8), alpha=0.1) == np.inf
    assert np.isfinite(conformal_threshold(np.arange(9), alpha=0.1))


def test_split_coverage_is_about_90_percent_on_average():
    covs = []
    for seed in range(20):
        X, y, _ = make_hard_group_data(3000, seed=seed)
        model = LogisticRegression().fit(X[:1000], y[:1000])
        cp = SplitConformal(model, alpha=0.1).calibrate(X[1000:2000], y[1000:2000])
        covs.append(coverage(y[2000:], cp.predict_sets(X[2000:]), model.classes_))
    assert 0.88 <= np.mean(covs) <= 0.93
