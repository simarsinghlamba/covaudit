"""Multi-seed repair experiment (split vs Mondrian) and, later, the shift experiment."""
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import pandas as pd

from covaudit import __version__
from covaudit.conformal import MondrianConformal, SplitConformal
from covaudit.data import (RACE_NAMES, SEX_NAMES, file_checksum, load_acs_income,
                           split_indices)
from covaudit.metrics import group_coverage_table, worst_group_gap
from covaudit.model import train_model

NAMES = {"RAC1P": RACE_NAMES, "SEX": SEX_NAMES}
LIBRARIES = ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "pyyaml",
             "folktables"]


def _code(g):
    try:
        f = float(g)
    except (TypeError, ValueError):
        return g
    return int(f) if f.is_integer() else g


def _add_names(table, column):
    """Integer group codes plus a readable 'name' column."""
    names = NAMES.get(column, {})
    table = table.copy()
    codes = [g if g == "ALL" else _code(g) for g in table["group"]]
    table["group"] = codes
    table.insert(1, "name", ["everyone" if g == "ALL" else names.get(g, str(g))
                             for g in codes])
    return table


def _run_info(cfg, n_rows, root, year, seconds):
    versions = {}
    for lib in LIBRARIES:
        try:
            versions[lib] = importlib.metadata.version(lib)
        except importlib.metadata.PackageNotFoundError:
            versions[lib] = "not installed"
    checksums = {p.name: file_checksum(p)
                 for p in sorted(Path(root, str(year)).rglob("*.csv"))}
    return {
        "covaudit_version": __version__,
        "config": cfg,
        "data_rows": n_rows,
        "data_sha256": checksums,
        "runtime_seconds": round(seconds, 1),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "libraries": versions,
    }


def _conformal_sets(method, model, alpha, X_cal, y_cal, g_cal, X_test, g_test):
    if method == "mondrian":
        cp = MondrianConformal(model, alpha).calibrate(X_cal, y_cal, g_cal)
        return cp.predict_sets(X_test, g_test)
    cp = SplitConformal(model, alpha).calibrate(X_cal, y_cal)
    return cp.predict_sets(X_test)


def summarise_repair(groups, alpha):
    """Per method: mean and std over seeds of overall coverage, worst-group gap and
    average set size, plus the number of seeds with any FAIL group."""
    per_seed = []
    for (method, seed), t in groups.groupby(["method", "seed"], sort=False):
        everyone = t[t["group"] == "ALL"].iloc[0]
        per_seed.append({
            "method": method, "seed": seed,
            "coverage": everyone["coverage"], "avg_set_size": everyone["avg_set_size"],
            "worst_group_gap": worst_group_gap(t, alpha),
            "any_fail": bool((t[t["group"] != "ALL"]["status"] == "FAIL").any()),
        })
    per_seed = pd.DataFrame(per_seed)
    return per_seed.groupby("method", sort=False).agg(
        n_seeds=("seed", "count"),
        coverage_mean=("coverage", "mean"), coverage_std=("coverage", "std"),
        worst_gap_mean=("worst_group_gap", "mean"), worst_gap_std=("worst_group_gap", "std"),
        set_size_mean=("avg_set_size", "mean"), set_size_std=("avg_set_size", "std"),
        seeds_with_fail=("any_fail", "sum"),
    ).reset_index()


def run_repair_experiment(cfg, out_dir):
    """Split vs Mondrian over many seeds. Writes group_coverage.csv, summary.csv and
    run_info.json into out_dir; returns (groups, summary, info)."""
    start = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    d = cfg["dataset"]
    X, y = load_acs_income(d["state"], d["year"], d["root"])
    column, alpha = cfg["group_column"], float(cfg["alpha"])
    confidence = float(cfg.get("confidence", 0.95))
    fractions = tuple(cfg.get("split_fractions", (0.6, 0.2, 0.2)))

    frames = []
    for seed in cfg["seeds"]:
        s = split_indices(len(X), seed=seed, fractions=fractions)
        model = train_model(X.iloc[s["train"]], y.iloc[s["train"]], seed=seed)
        cal, test = s["cal"], s["test"]
        for method in cfg["methods"]:
            sets = _conformal_sets(method, model, alpha,
                                   X.iloc[cal], y.iloc[cal], X[column].iloc[cal],
                                   X.iloc[test], X[column].iloc[test])
            t = group_coverage_table(y.iloc[test], sets, X[column].iloc[test],
                                     model.classes_, alpha=alpha, confidence=confidence)
            t = _add_names(t, column)
            t.insert(0, "method", method)
            t.insert(0, "seed", seed)
            frames.append(t)

    groups = pd.concat(frames, ignore_index=True)
    summary = summarise_repair(groups, alpha)
    groups.to_csv(out / "group_coverage.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    info = _run_info(cfg, len(X), d["root"], d["year"], time.time() - start)
    (out / "run_info.json").write_text(json.dumps(info, indent=2, default=str))
    return groups, summary, info


def summarise_shift(shift, alpha):
    """One row per target x method: overall coverage, set size, worst-group gap, FAIL count."""
    rows = []
    for (target, method), t in shift.groupby(["target", "method"], sort=False):
        everyone = t[t["group"] == "ALL"].iloc[0]
        rows.append({
            "target": target, "shifted": bool(everyone["shifted"]), "method": method,
            "n": int(everyone["n"]), "coverage": everyone["coverage"],
            "avg_set_size": everyone["avg_set_size"],
            "worst_group_gap": worst_group_gap(t, alpha),
            "fail_groups": int(((t["group"] != "ALL") & (t["status"] == "FAIL")).sum()),
        })
    return pd.DataFrame(rows)


def run_shift_experiment(cfg, out_dir):
    """Calibrate on the source state, then use the SAME thresholds on other states.
    Writes shift_coverage.csv and run_info.json; returns (shift, info)."""
    start = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    src, root = cfg["source"], cfg["root"]
    year = src["year"]
    column, alpha = cfg["group_column"], float(cfg["alpha"])
    confidence = float(cfg.get("confidence", 0.95))
    seed = int(cfg.get("seed", 0))

    X, y = load_acs_income(src["state"], year, root)
    s = split_indices(len(X), seed=seed)
    model = train_model(X.iloc[s["train"]], y.iloc[s["train"]], seed=seed)
    cal = s["cal"]
    calibrated = {}
    for method in cfg["methods"]:
        if method == "mondrian":
            calibrated[method] = MondrianConformal(model, alpha).calibrate(
                X.iloc[cal], y.iloc[cal], X[column].iloc[cal])
        else:
            calibrated[method] = SplitConformal(model, alpha).calibrate(X.iloc[cal], y.iloc[cal])

    frames = []

    def audit(target, shifted, X_t, y_t):
        """Append the group tables of one state under every calibrated method."""
        g_t = X_t[column]
        for method, cp in calibrated.items():
            sets = cp.predict_sets(X_t, g_t) if method == "mondrian" else cp.predict_sets(X_t)
            t = group_coverage_table(y_t, sets, g_t, model.classes_, alpha=alpha,
                                     confidence=confidence)
            t = _add_names(t, column)
            t.insert(0, "method", method)
            t.insert(0, "shifted", shifted)
            t.insert(0, "target", target)
            frames.append(t)

    rows = {src["state"]: len(X)}
    audit(src["state"], False, X.iloc[s["test"]], y.iloc[s["test"]])
    del X, y
    for state in cfg["targets"]:
        X_t, y_t = load_acs_income(state, year, root)
        rows[state] = len(X_t)
        audit(state, True, X_t, y_t)
        del X_t, y_t

    shift = pd.concat(frames, ignore_index=True)
    shift.to_csv(out / "shift_coverage.csv", index=False)
    info = _run_info(cfg, rows, root, year, time.time() - start)
    (out / "run_info.json").write_text(json.dumps(info, indent=2, default=str))
    return shift, info
