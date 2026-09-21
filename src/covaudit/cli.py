"""Command-line interface: the `covaudit` command."""
import argparse
from pathlib import Path

from covaudit import __version__
from covaudit.conformal import SplitConformal
from covaudit.data import (RACE_NAMES, SEX_NAMES, file_checksum, load_acs_income,
                           split_indices)
from covaudit.metrics import (average_set_size, coverage, group_coverage_table,
                              worst_group_gap)
from covaudit.model import train_model

GROUP_NAMES = {"RAC1P": RACE_NAMES, "SEX": SEX_NAMES}


def _code(g):
    """Turn 2.0 into 2 for display; leave anything else unchanged."""
    try:
        f = float(g)
    except (TypeError, ValueError):
        return g
    return int(f) if f.is_integer() else g


def _readable(table, group_column):
    """Integer codes plus a human-readable name column (display and CSV only)."""
    names = GROUP_NAMES.get(group_column, {})
    table = table.copy()
    codes = [g if g == "ALL" else _code(g) for g in table["group"]]
    table["group"] = codes
    table.insert(1, "name", ["everyone" if g == "ALL" else names.get(g, str(g))
                             for g in codes])
    return table


def cmd_data(args):
    """Download (or reuse) the data and print a short summary."""
    X, y = load_acs_income(args.state, args.year, args.root)
    print(f"state {args.state} | year {args.year} | rows {len(X)} | "
          f"positive rate {y.mean():.4f}")
    for path in sorted(Path(args.root, str(args.year)).rglob("*.csv")):
        print(f"sha256 {path.name}: {file_checksum(path)}")
    return 0


def cmd_audit(args):
    """Split conformal prediction on one seed; print and save the per-group audit."""
    X, y = load_acs_income(args.state, args.year, args.root)
    if args.group not in X.columns:
        print(f"error: group column '{args.group}' not found; "
              f"choose from {list(X.columns)}")
        return 2
    s = split_indices(len(X), seed=args.seed)
    model = train_model(X.iloc[s["train"]], y.iloc[s["train"]], seed=args.seed)
    cp = SplitConformal(model, alpha=args.alpha)
    cp.calibrate(X.iloc[s["cal"]], y.iloc[s["cal"]])
    sets = cp.predict_sets(X.iloc[s["test"]])
    y_test = y.iloc[s["test"]]
    groups = X[args.group].iloc[s["test"]]

    print(f"state {args.state} {args.year} | seed {args.seed} | alpha {args.alpha} | "
          f"train {len(s['train'])} / cal {len(s['cal'])} / test {len(s['test'])}")
    print(f"threshold {cp.threshold_:.4f}")
    print(f"coverage {coverage(y_test, sets, model.classes_):.4f} "
          f"(target >= {1 - args.alpha:.2f}) | "
          f"avg set size {average_set_size(sets):.4f}")

    table = group_coverage_table(y_test, sets, groups, model.classes_, alpha=args.alpha)
    table = _readable(table, args.group)
    print(f"\nper-group audit by {args.group} (method split):")
    print(table.round(4).to_string(index=False))
    print(f"\nworst-group gap: {worst_group_gap(table, args.alpha):.4f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"audit_split_{args.group}.csv"
    table.to_csv(path, index=False)
    print(f"saved {path}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="covaudit", description=__doc__)
    parser.add_argument("--version", action="version",
                        version=f"covaudit {__version__}")
    sub = parser.add_subparsers(dest="command")

    p_data = sub.add_parser("data", help="download (or reuse) data and summarise it")
    p_data.add_argument("--state", default="CA")
    p_data.add_argument("--year", default="2018")
    p_data.add_argument("--root", default="data")

    p_audit = sub.add_parser("audit", help="audit conformal coverage by group")
    p_audit.add_argument("--state", default="CA")
    p_audit.add_argument("--year", default="2018")
    p_audit.add_argument("--root", default="data")
    p_audit.add_argument("--alpha", type=float, default=0.1)
    p_audit.add_argument("--seed", type=int, default=0)
    p_audit.add_argument("--group", default="RAC1P")
    p_audit.add_argument("--out", default="outputs/audit")

    args = parser.parse_args(argv)
    if args.command == "data":
        return cmd_data(args)
    if args.command == "audit":
        return cmd_audit(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
