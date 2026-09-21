"""Command-line interface: the `covaudit` command."""
import argparse
from pathlib import Path

from covaudit import __version__
from covaudit.data import file_checksum, load_acs_income


def cmd_data(args):
    """Download (or reuse) the data and print a short summary."""
    X, y = load_acs_income(args.state, args.year, args.root)
    print(f"state {args.state} | year {args.year} | rows {len(X)} | "
          f"positive rate {y.mean():.4f}")
    for path in sorted(Path(args.root, str(args.year)).rglob("*.csv")):
        print(f"sha256 {path.name}: {file_checksum(path)}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="covaudit", description=__doc__)
    parser.add_argument("--version", action="version", version=f"covaudit {__version__}")
    sub = parser.add_subparsers(dest="command")

    p_data = sub.add_parser("data", help="download (or reuse) ACSIncome data and summarise it")
    p_data.add_argument("--state", default="CA")
    p_data.add_argument("--year", default="2018")
    p_data.add_argument("--root", default="data")

    args = parser.parse_args(argv)
    if args.command == "data":
        return cmd_data(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
