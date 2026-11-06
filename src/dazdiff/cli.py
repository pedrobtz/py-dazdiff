"""Command line interface: ``dazdiff X.csv Y.csv [options]``."""

from __future__ import annotations

import argparse
import sys
import warnings
from collections.abc import Sequence

import pandas as pd

from . import __version__
from .core import DiffError, _canon, diff_summary, diff_table, is_equal


def _align_numeric(x: pd.DataFrame, y: pd.DataFrame) -> None:
    """CSV carries no types: where one file reads a column as int and the other as
    float, read both as float so values are not truncated by the cast."""
    ny = dict(zip([_canon(c) for c in y.columns], y.columns, strict=True))
    for cx, nx in zip(x.columns, [_canon(c) for c in x.columns], strict=True):
        cy = ny.get(nx)
        if cy is None:
            continue
        kinds = {x[cx].dtype.kind, y[cy].dtype.kind}
        if kinds == {"i", "f"}:
            x[cx] = x[cx].astype(float)
            y[cy] = y[cy].astype(float)


def _csv_list(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dazdiff",
        description="Compare two CSV files value by value. "
        "Exit status: 0 if the tables match, 1 if they differ, 2 on error.",
    )
    p.add_argument("x", help="first CSV file (the benchmark in benchmark mode)")
    p.add_argument("y", help="second CSV file (the tested table in benchmark mode)")
    p.add_argument("-b", "--by", type=_csv_list, help="comma-separated key columns")
    p.add_argument(
        "-m",
        "--measures",
        default="numeric",
        help='"numeric" (default), "numeric+integer", or comma-separated column names',
    )
    p.add_argument("--mode", choices=["benchmark", "equal"], default="benchmark")
    p.add_argument("-t", "--tolerance", type=float, default=0.0, help="absolute tolerance")
    p.add_argument("-r", "--rel-tol", type=float, default=0.0, help="relative tolerance")
    p.add_argument(
        "--no-truncate",
        dest="truncate",
        action="store_false",
        help="refuse casts that would truncate values to integers",
    )
    p.add_argument("--duplicates", choices=["pair", "error"], default="pair")
    p.add_argument("--sep", default=",", help="CSV field separator (default ',')")
    p.add_argument("-o", "--output", help="write the full comparison to this CSV file")
    p.add_argument("-s", "--summary", action="store_true", help="print the per-measure summary")
    p.add_argument(
        "-d", "--only-diffs", action="store_true", help="print only rows that are not 'same'"
    )
    p.add_argument("-q", "--quiet", action="store_true", help="print nothing; use the exit status")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    measures = args.measures
    if measures not in ("numeric", "numeric+integer"):
        measures = _csv_list(measures)

    def show_warning(message, category, *_args, **_kwargs):
        if not args.quiet:
            print(f"dazdiff: {message}", file=sys.stderr)

    try:
        x = pd.read_csv(args.x, sep=args.sep)
        y = pd.read_csv(args.y, sep=args.sep)
        _align_numeric(x, y)
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.showwarning = show_warning
            d = diff_table(
                x,
                y,
                by=args.by,
                measures=measures,
                mode=args.mode,
                tolerance=args.tolerance,
                rel_tol=args.rel_tol,
                truncate=args.truncate,
                duplicates=args.duplicates,
            )
    except (DiffError, OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as e:
        print(f"dazdiff: error: {e}", file=sys.stderr)
        return 2

    if args.output:
        d.to_csv(args.output, index=False)
    if not args.quiet:
        if args.summary:
            print(diff_summary(d).to_string(index=False))
        elif not args.output:
            shown = d[d["status"] != "same"] if args.only_diffs else d
            print(shown.to_string(index=False) if len(shown) else "No differences.")
    return 0 if is_equal(d) else 1


if __name__ == "__main__":
    sys.exit(main())
