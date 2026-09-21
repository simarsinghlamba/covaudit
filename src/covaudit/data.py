"""Download ACSIncome data and split it into train / calibration / test."""
import numpy as np

RACE_NAMES = {
    1: "White", 2: "Black", 3: "American Indian", 4: "Alaska Native",
    5: "AI/AN tribes", 6: "Asian", 7: "Native Hawaiian/PI",
    8: "Some other race", 9: "Two or more races",
}
SEX_NAMES = {1: "Male", 2: "Female"}


def load_acs_income(state="CA", year="2018", root="data"):
    """Return (X, y) for ACSIncome. X is a DataFrame of features, y is 0/1."""
    from folktables import ACSDataSource, ACSIncome

    source = ACSDataSource(survey_year=str(year), horizon="1-Year",
                           survey="person", root_dir=str(root))
    raw = source.get_data(states=[state], download=True)
    X, y, _ = ACSIncome.df_to_pandas(raw)
    X = X.reset_index(drop=True)
    y = y.iloc[:, 0].astype(int).reset_index(drop=True)
    return X, y


def split_indices(n, seed=0, fractions=(0.6, 0.2, 0.2)):
    """Shuffle row numbers 0..n-1 and cut them into train / cal / test."""
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("fractions must add up to 1")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(fractions[0] * n)
    n_cal = int(fractions[1] * n)
    return {"train": idx[:n_train],
            "cal": idx[n_train:n_train + n_cal],
            "test": idx[n_train + n_cal:]}
