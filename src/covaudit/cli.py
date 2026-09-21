"""Command-line interface: the `covaudit` command."""
import argparse

from covaudit import __version__


def main(argv=None):
    parser = argparse.ArgumentParser(prog="covaudit", description=__doc__)
    parser.add_argument("--version", action="version", version=f"covaudit {__version__}")
    parser.parse_args(argv)
    print(f"covaudit {__version__}. Use --help to see commands.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
