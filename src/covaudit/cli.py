"""Command-line interface: the `covaudit` command."""
import argparse
from pathlib import Path

from covaudit import __version__
from covaudit.conformal import SplitConformal
from covaudit.data import file_checksum, load_acs_income, split_indices
from covaudit.metrics import average_set_size, coverage
from covaudit.model import train_model


def cmd_data(args):
    """Download (or reuse) the data and print a short summary."""
    X, y = load_acs_income(args.state, args.year, args.root)
    print(f"state {args.state} | year {args.year} | rows {len(X)} | "
          f"positive rate {y.mean():.4f}")
    for path in sorted(Path(args.root, str(args.year)).rglob("*.csv")):
        print(f"sha256 {path.name}: {file_checksum(path)}")
    return 0


def cmd_audit(args):
    """Split conformal prediction on one seed; print overall coverage and set size."""
    X, y = load_acs_income(args.state, args.year, args.root)
    s = split_indices(len(X), seed=args.seed)
    model = train_model(X.iloc[s["train"]], y.iloc[s["train"]], seed=args.seed)
    cp = SplitConformal(model, alpha=args.alpha).calibrate(X.iloc[s["cal"]], y.iloc[s["cal"]])
    sets = cp.predict_sets(X.iloc[s["test"]])
    y_test = y.iloc[s["test"]]
    print(f"state {args.state} {args.year} | seed {args.seed} | alpha {args.alpha} | "
          f"train {len(s['train'])} / cal {len(s['cal'])} / test {len(s['test'])}")
    print(f"threshold {cp.threshold_:.4f}")
    print(f"coverage {coverage(y_test, sets, model.classes_):.4f} "
          f"(target >= {1 - args.alpha:.2f}) | avg set size {average_set_size(sets):.4f}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="covaudit", description=__doc__)
    parser.add_argument("--version", action="version", version=f"covaudit {__version__}")
    sub = parser.add_subparsers(dest="command")

    p_data = sub.add_parser("data", help="download (or reuse) ACSIncome data and summarise it")
    p_data.add_argument("--state", default="CA")
    p_data.add_argument("--year", default="2018")
    p_data.add_argument("--root", default="data")

    p_audit = sub.add_parser("audit", help="run split conformal prediction and report coverage")
    p_audit.add_argument("--state", default="CA")
    p_audit.add_argument("--year", default="2018")
    p_audit.add_argument("--root", default="data")
    p_audit.add_argument("--alpha", type=float, default=0.1)
    p_audit.add_argument("--seed", type=int, default=0)

    args = parser.parse_args(argv)
    if args.command == "data":
        return cmd_data(args)
    if args.command == "audit":
        return cmd_audit(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
