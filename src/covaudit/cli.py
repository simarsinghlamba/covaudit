"""Command-line interface: the `covaudit` command."""
import argparse
from pathlib import Path

from covaudit import __version__
from covaudit.conformal import MondrianConformal, SplitConformal
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
    """Split or Mondrian CP on one seed; print and save the per-group audit."""
    X, y = load_acs_income(args.state, args.year, args.root)
    if args.group not in X.columns:
        print(f"error: group column '{args.group}' not found; "
              f"choose from {list(X.columns)}")
        return 2
    s = split_indices(len(X), seed=args.seed)
    model = train_model(X.iloc[s["train"]], y.iloc[s["train"]], seed=args.seed)
    y_test = y.iloc[s["test"]]
    groups = X[args.group].iloc[s["test"]]
    if args.method == "mondrian":
        cp = MondrianConformal(model, alpha=args.alpha)
        cp.calibrate(X.iloc[s["cal"]], y.iloc[s["cal"]], X[args.group].iloc[s["cal"]])
        sets = cp.predict_sets(X.iloc[s["test"]], groups)
    else:
        cp = SplitConformal(model, alpha=args.alpha)
        cp.calibrate(X.iloc[s["cal"]], y.iloc[s["cal"]])
        sets = cp.predict_sets(X.iloc[s["test"]])

    print(f"state {args.state} {args.year} | seed {args.seed} | alpha {args.alpha} | "
          f"train {len(s['train'])} / cal {len(s['cal'])} / test {len(s['test'])}")
    if args.method == "mondrian":
        names = GROUP_NAMES.get(args.group, {})
        th = ", ".join(f"{names.get(_code(g), _code(g))} {t:.4f}"
                       for g, t in cp.thresholds_.items())
        print(f"thresholds per group: {th}")
    else:
        print(f"threshold {cp.threshold_:.4f}")
    print(f"coverage {coverage(y_test, sets, model.classes_):.4f} "
          f"(target >= {1 - args.alpha:.2f}) | "
          f"avg set size {average_set_size(sets):.4f}")

    table = group_coverage_table(y_test, sets, groups, model.classes_, alpha=args.alpha)
    table = _readable(table, args.group)
    print(f"\nper-group audit by {args.group} (method {args.method}):")
    print(table.round(4).to_string(index=False))
    print(f"\nworst-group gap: {worst_group_gap(table, args.alpha):.4f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"audit_{args.method}_{args.group}.csv"
    table.to_csv(path, index=False)
    print(f"saved {path}")
    return 0


def cmd_run(args):
    """Run the multi-seed repair experiment from a YAML config."""
    from covaudit.config import load_config, load_default_config
    from covaudit.experiment import run_repair_experiment

    cfg = load_config(args.config) if args.config else load_default_config("repair")
    if args.root:
        cfg["dataset"]["root"] = args.root
    print(f"repair experiment: {len(cfg['seeds'])} seeds x {cfg['methods']} "
          f"| group {cfg['group_column']} | alpha {cfg['alpha']}")
    groups, summary, info = run_repair_experiment(cfg, args.out)
    print(summary.round(4).to_string(index=False))
    print(f"runtime {info['runtime_seconds']} s | wrote {args.out}/group_coverage.csv, "
          f"summary.csv, run_info.json")
    return 0


def cmd_shift(args):
    """Calibrate on the source state and measure coverage in other states."""
    from covaudit.config import load_config, load_default_config
    from covaudit.experiment import run_shift_experiment, summarise_shift

    cfg = load_config(args.config) if args.config else load_default_config("shift")
    if args.root:
        cfg["root"] = args.root
    print(f"shift experiment: calibrate on {cfg['source']['state']} -> "
          f"{cfg['targets']} | group {cfg['group_column']} | alpha {cfg['alpha']}")
    shift, info = run_shift_experiment(cfg, args.out)
    print(summarise_shift(shift, float(cfg["alpha"])).round(4).to_string(index=False))
    print(f"runtime {info['runtime_seconds']} s | wrote {args.out}/shift_coverage.csv, "
          f"run_info.json")
    return 0


def cmd_report(args):
    """Draw figures and write report.md from whatever result CSVs exist."""
    from covaudit.report import write_report

    if not Path(args.results).is_dir():
        print(f"error: results folder '{args.results}' does not exist")
        return 1
    path, notes = write_report(args.results, args.out, alpha=args.alpha)
    for note in notes:
        print(f"note: {note}")
    for png in sorted(Path(args.out).glob("*.png")):
        print(f"figure {png}")
    print(f"wrote {path}")
    return 0


def main(argv=None):
    """Entry point of the covaudit command; argv defaults to the real command line."""
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
    p_audit.add_argument("--method", choices=["split", "mondrian"], default="split")
    p_audit.add_argument("--out", default="outputs/audit")

    p_run = sub.add_parser("run", help="multi-seed split vs Mondrian experiment")
    p_run.add_argument("--config", default=None,
                       help="YAML settings (default: built-in, same as configs/repair.yaml)")
    p_run.add_argument("--out", default="outputs/repair")
    p_run.add_argument("--root", default=None, help="override the data folder")

    p_shift = sub.add_parser("shift", help="calibrate on one state, test on others")
    p_shift.add_argument("--config", default=None,
                         help="YAML settings (default: built-in, same as configs/shift.yaml)")
    p_shift.add_argument("--out", default="outputs/shift")
    p_shift.add_argument("--root", default=None, help="override the data folder")

    p_report = sub.add_parser("report", help="draw figures and write report.md")
    p_report.add_argument("--results", default="outputs")
    p_report.add_argument("--out", default="outputs/report")
    p_report.add_argument("--alpha", type=float, default=0.1)

    args = parser.parse_args(argv)
    if args.command == "data":
        return cmd_data(args)
    if args.command == "audit":
        return cmd_audit(args)
    if args.command == "run":
        return cmd_run(args)
    if args.command == "shift":
        return cmd_shift(args)
    if args.command == "report":
        return cmd_report(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
