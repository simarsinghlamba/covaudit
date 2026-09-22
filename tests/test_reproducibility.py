import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from covaudit.conformal import MondrianConformal, SplitConformal
from covaudit.data import split_indices
from covaudit.metrics import group_coverage_table
from covaudit.model import train_model
from covaudit.synthetic import make_hard_group_data


def _pipeline(seed):
    X, y, g = make_hard_group_data(3000, seed=seed)
    s = split_indices(len(X), seed=seed)
    tr, ca, te = s["train"], s["cal"], s["test"]
    model = LogisticRegression().fit(X[tr], y[tr])
    split = SplitConformal(model, 0.1).calibrate(X[ca], y[ca])
    mond = MondrianConformal(model, 0.1).calibrate(X[ca], y[ca], g[ca])
    t1 = group_coverage_table(y[te], split.predict_sets(X[te]), g[te], model.classes_)
    t2 = group_coverage_table(y[te], mond.predict_sets(X[te], g[te]), g[te], model.classes_)
    t1.insert(0, "method", "split")
    t2.insert(0, "method", "mondrian")
    return pd.concat([t1, t2], ignore_index=True)


def test_same_seed_gives_identical_results():
    pd.testing.assert_frame_equal(_pipeline(3), _pipeline(3))


def test_different_seed_gives_different_results():
    assert not np.allclose(_pipeline(3)["coverage"], _pipeline(4)["coverage"])


def test_base_model_is_seeded():
    # 20,000 rows: gradient boosting turns on early stopping, which is random unless seeded.
    X, y, _ = make_hard_group_data(20000, seed=0)
    a = train_model(X, y, seed=0).predict_proba(X[:500])
    b = train_model(X, y, seed=0).predict_proba(X[:500])
    assert np.array_equal(a, b)
