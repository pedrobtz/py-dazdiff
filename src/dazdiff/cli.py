"""Command line interface: ``dazdiff X.csv Y.csv [options]``."""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .core import _canon, diff_explain, diff_summary, diff_table, is_matching
from .report import diff_report


def _align_numeric(x: pd.DataFrame, y: pd.DataFrame) -> None:
    """CSV carries no types: where one file reads a column as int and the other as
    float with a fraction, read both as float so the cast does not truncate it.

    A float column of whole numbers is float only because it has an empty value; it
    casts to int exactly, so it is left alone and keeps its role: widening it would
    turn an integer key into a "numeric" measure.
    """
    ny = dict(zip([_canon(c) for c in y.columns], y.columns, strict=True))
    for cx, nx in zip(x.columns, [_canon(c) for c in x.columns], strict=True):
        cy = ny.get(nx)
        if cy is None:
            continue
        kinds = {x[cx].dtype.kind, y[cy].dtype.kind}
        if kinds == {"i"} and x[cx].dtype != y[cy].dtype:  # int64 vs Int64: both nullable
            x[cx] = x[cx].astype("Int64")
            y[cy] = y[cy].astype("Int64")
            continue
        if kinds == {"i", "f"} and _has_fraction(x[cx] if x[cx].dtype.kind == "f" else y[cy]):
            x[cx] = x[cx].astype(float)
            y[cy] = y[cy].astype(float)


def _printable(text: str, stream) -> bool:
    """Whether ``stream`` can encode ``text`` (a Windows console often cannot)."""
    encoding = getattr(stream, "encoding", None)
    if not encoding:
        return False
    try:
        text.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return False
    return True


_LEADING_ZERO = re.compile(r"\s*[+-]?0[0-9]")


def _read(path: str, sep: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The file as pandas types it, and as text."""
    return pd.read_csv(path, sep=sep), pd.read_csv(path, sep=sep, dtype=str)


def _keep_codes(x, x_text, y, y_text) -> None:
    """A column written with a leading zero in either file (``007``) holds codes, not
    numbers: read it as text in both, so ``007`` and ``7`` stay apart (#47)."""
    y_by = {_canon(c): c for c in y.columns}
    for cx in x.columns:
        cy = y_by.get(_canon(cx))
        cols = [(x, x_text, cx)] + ([(y, y_text, cy)] if cy is not None else [])
        if not any(df[c].dtype.kind in "iuf" for df, _, c in cols):
            continue
        if any(_has_leading_zero(text[c]) for _, text, c in cols):
            for df, text, c in cols:
                df[c] = text[c]


def _has_leading_zero(s: pd.Series) -> bool:
    return bool(s.dropna().map(lambda v: bool(_LEADING_ZERO.match(v))).any())


def _restore_ints(df: pd.DataFrame) -> None:
    """pandas reads an integer column with an empty cell as float; read it as a
    nullable integer instead, so its role and type follow its values (#46)."""
    for c in df.columns:
        s = df[c]
        if s.dtype.kind != "f" or not s.isna().any() or s.notna().sum() == 0:
            continue
        v = s.dropna().to_numpy()
        if np.isfinite(v).all() and (v == np.trunc(v)).all() and (np.abs(v) < 2.0**63).all():
            df[c] = s.astype("Int64")


def _has_fraction(s: pd.Series) -> bool:
    v = s.dropna().to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    return bool((v != np.trunc(v)).any())


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
        "--exclude",
        type=_csv_list,
        help="comma-separated columns to leave out of both files",
    )
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
        "--no-cast",
        dest="cast",
        action="store_const",
        const=False,
        default=None,
        help="do not cast the second file to the first's types; they must agree",
    )
    p.add_argument(
        "--no-truncate",
        dest="truncate",
        action="store_false",
        help="refuse casts that would truncate values to integers",
    )
    p.add_argument(
        "--duplicates",
        choices=["disambiguate", "aggregate", "error"],
        default="disambiguate",
        help="repeated keys: pair in arrival order (default), compare as multisets, or refuse",
    )
    p.add_argument(
        "--row-key",
        choices=["none", "position", "sorted"],
        default="none",
        help="add a row_number key: rows as they come, or after sorting both files",
    )
    p.add_argument("--sep", default=",", help="CSV field separator (default ',')")
    p.add_argument("-o", "--output", help="write the full comparison to this CSV file")
    p.add_argument("-s", "--summary", action="store_true", help="print the per-measure summary")
    p.add_argument(
        "-R", "--report", action="store_true", help="print a text report of the comparison"
    )
    p.add_argument("--x-name", help="the first file's name in the report (default: its file name)")
    p.add_argument("--y-name", help="the second file's name in the report (default: its file name)")
    p.add_argument(
        "--ascii",
        action="store_true",
        help="draw the report in ASCII (the default when the terminal cannot show Unicode)",
    )
    p.add_argument(
        "-e",
        "--explain",
        nargs="?",
        const="",
        metavar="COLUMN",
        help="print which key values split same from not same; with COLUMN, per value of it",
    )
    p.add_argument(
        "--joined",
        action="store_true",
        help="with --explain, count only rows found in both files",
    )
    p.add_argument(
        "-d", "--only-diffs", action="store_true", help="print only rows that are not 'same'"
    )
    p.add_argument("-q", "--quiet", action="store_true", help="print nothing; use the exit status")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _run(args)
    except Exception as e:  # every failure is exit 2: 1 means "the tables differ"
        print(f"dazdiff: error: {e}", file=sys.stderr)
        return 2


def _run(args: argparse.Namespace) -> int:
    def show_warning(message, category, *_args, **_kwargs):
        if not args.quiet:
            print(f"dazdiff: {message}", file=sys.stderr)

    x, x_text = _read(args.x, args.sep)
    y, y_text = _read(args.y, args.sep)
    _keep_codes(x, x_text, y, y_text)
    _restore_ints(x)
    _restore_ints(y)
    measures = args.measures
    if measures not in ("numeric", "numeric+integer"):
        measures = _csv_list(measures)
    _align_numeric(x, y)
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.showwarning = show_warning
        d = diff_table(
            x,
            y,
            by=args.by,
            exclude=args.exclude,
            cast=args.cast,
            row_key=args.row_key,
            measures=measures,
            mode=args.mode,
            tolerance=args.tolerance,
            rel_tol=args.rel_tol,
            truncate=args.truncate,
            duplicates=args.duplicates,
        )

    if args.output:
        d.to_csv(args.output, index=False)
    if not args.quiet:
        if args.report:
            r = diff_report(d, args.x_name or Path(args.x).name, args.y_name or Path(args.y).name)
            text = r.format(unicode=True)
            if args.ascii or not _printable(text, sys.stdout):
                text = r.format(unicode=False)
            print(text)
        elif args.explain is not None:
            e = diff_explain(d, args.explain or None, among="joined" if args.joined else "all")
            print(e.to_string(index=False))
        elif args.summary:
            print(diff_summary(d).to_string(index=False))
        elif not args.output:
            shown = d[d["status"] != "same"] if args.only_diffs else d
            print(shown.to_string(index=False) if len(shown) else "No differences.")
    return 0 if is_matching(d) else 1


if __name__ == "__main__":
    sys.exit(main())
