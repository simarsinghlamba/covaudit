import numpy as np
from sklearn.linear_model import LogisticRegression

from covaudit.conformal import MondrianConformal, SplitConformal
from covaudit.metrics import is_covered
from covaudit.synthetic import make_hard_group_data


def _run(seed):
    X, y, g = make_hard_group_data(30000, seed=seed)
    tr, ca, te = slice(0, 5000), slice(5000, 10000), slice(10000, None)
    model = LogisticRegression().fit(X[tr], y[tr])
    split = SplitConformal(model, 0.1).calibrate(X[ca], y[ca])
    mond = MondrianConformal(model, 0.1).calibrate(X[ca], y[ca], g[ca])
    c_split = is_covered(y[te], split.predict_sets(X[te]), model.classes_)
    c_mond = is_covered(y[te], mond.predict_sets(X[te], g[te]), model.classes_)
    hard = g[te] == 1
    return c_split[hard].mean(), c_mond[hard].mean(), c_mond[~hard].mean()


def test_mondrian_repairs_the_hard_group_on_average():
    results = np.array([_run(seed) for seed in range(10)])
    split_hard, mond_hard, mond_easy = results.mean(axis=0)
    assert split_hard < 0.80  # split hides a failure
    assert mond_hard >= 0.88  # Mondrian restores it
    assert mond_easy >= 0.88


def test_unseen_or_tiny_group_gets_full_set():
    X, y, _ = make_hard_group_data(2000, seed=0)
    model = LogisticRegression().fit(X[:1000], y[:1000])
    tiny = np.array([0] * 995 + [7] * 5)  # group 7 has only 5 people
    mond = MondrianConformal(model, 0.1).calibrate(X[1000:2000], y[1000:2000], tiny)
    sets = mond.predict_sets(X[:3], np.array([7, 7, 99]))  # 99 never seen
    assert sets.all()
