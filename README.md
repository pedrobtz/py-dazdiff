# dazdiff

[![CI](https://github.com/pedrobtz/py-dazdiff/actions/workflows/ci.yml/badge.svg)](https://github.com/pedrobtz/py-dazdiff/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/dazdiff)](https://pypi.org/project/dazdiff/)
[![Python](https://img.shields.io/pypi/pyversions/dazdiff)](https://pypi.org/project/dazdiff/)

`dazdiff` compares two tables value by value, from Python or from the command
line. It is a small Python take on the R package
[pedrobtz/dazdiff](https://github.com/pedrobtz/dazdiff).

Its main function, `diff_table()`, answers one of two questions:

- **Does this table match the benchmark?** (`mode="benchmark"`, the default.)
  The benchmark's column types are the truth: the tested table is cast to them
  and its extra columns are dropped.
- **Are these two tables equal?** (`mode="equal"`.) Both tables must have the
  same columns with the same types; nothing is cast.

The result has one row per key and measure, with `row_id_x`, `row_id_y`,
`value_x`, `value_y`, `diff`, `abs_diff`, `rel_diff` and a `status` of `same`,
`changed`, `only_x` or `only_y`.

## Installation

```sh
pip install dazdiff        # the library and the CLI
uv tool install dazdiff    # the CLI on its own
```

It needs Python 3.10 or later and pandas 2.2.2 or later. The development
version installs from GitHub:
`pip install git+https://github.com/pedrobtz/py-dazdiff`.

## Command line

```sh
$ dazdiff examples/benchmark.csv examples/stored.csv --by id,region --only-diffs
dazdiff: Dropped column(s) of `y` that the benchmark `x` does not have: loaded_at.
 ID REGION metric  row_id_x  row_id_y  value_x  value_y  diff  abs_diff  rel_diff  status
  1  north  UNITS         1         1     10.0     10.4   0.4       0.4  0.038462 changed
  2  south AMOUNT         2         2    200.0    200.5   0.5       0.5  0.002494 changed
  4   west AMOUNT         4      <NA>    400.0      NaN   NaN       NaN       NaN  only_x
  ...
```

| Option | Meaning |
| --- | --- |
| `-b, --by COLS` | comma-separated key columns (default: every non-measure column) |
| `--exclude COLS` | comma-separated columns to leave out of both files |
| `-m, --measures` | `numeric` (default), `numeric+integer`, or comma-separated names |
| `--mode` | `benchmark` (default) or `equal` |
| `-t, --tolerance` | absolute tolerance |
| `-r, --rel-tol` | relative tolerance (relative to the larger magnitude) |
| `--no-cast` | do not cast the second file to the first's types; they must agree |
| `--no-truncate` | refuse casts that would truncate values to integers |
| `--duplicates` | `disambiguate` repeated keys in arrival order (default), `aggregate` them as multisets, or `error` |
| `--row-key` | `position` or `sorted`: add a `row_number` key when every column is a measure |
| `--sep` | CSV separator |
| `-o, --output FILE` | write the full comparison to a CSV file |
| `-s, --summary` | print the per-measure summary instead |
| `-R, --report` | print a text report; `--x-name`, `--y-name` label the files, `--ascii` avoids box-drawing characters |
| `-e, --explain [COLUMN]` | print which key values split same from not same; with a column, per value of it |
| `--joined` | with `--explain`, count only rows found in both files |
| `-d, --only-diffs` | print only rows that are not `same` |
| `-q, --quiet` | print nothing |

The exit status is `0` when the tables match, `1` when they differ and `2` on
error, so `dazdiff` works in scripts and CI. Column names are normalized (upper
case, other characters turned into `_`), so `id` and `ID` pair up. Because CSV
files carry no types, a column read as integer in one file and as decimals in
the other is compared as float; one that is float only because of an empty
value stays an integer.

`python -m dazdiff` works too.

## Python

```python
import pandas as pd
from dazdiff import diff_report, diff_summary, diff_table, is_matching

benchmark = pd.read_csv("examples/benchmark.csv")
stored = pd.read_csv("examples/stored.csv")

d = diff_table(benchmark, stored, by=["id", "region"], measures="numeric+integer")
d[d["status"] != "same"]
diff_summary(d)
assert not is_matching(d)  # rows 4 and 5 are each in one table only
```

`is_matching(d, tolerance=1)` judges the values again at another threshold,
without running the comparison again.

`diff_explain(d)` says which key columns separate the rows that match from
those that do not, and `diff_explain(d, "region")` counts each status per
region, most differences first.

`exclude=` leaves technical columns (load stamps, run ids) out of both tables.
When a number of the tested table is cast to the benchmark's text, it is
written with the decimals the benchmark uses, so `2` meets `"2.0"`.

With no measure columns, rows are compared as wholes, and a repeated row counts
once per copy. Values truncated when casting to the benchmark's integer
columns are recorded in `d.attrs["truncated"]`.

### Report

```python
print(diff_report(d, "benchmark", "stored"))
```

draws the columns compared, how the rows of the two tables align, and per
changed measure its match rate and largest differences.
`diff_report(d).format(unicode=False)` draws it in ASCII.

### In tests

```python
from dazdiff import assert_table_equal

expected = pd.DataFrame({"id": [1, 2], "amount": [100.0, 200.0]})
actual = pd.DataFrame({"id": [1, 2], "amount": [100.0, 200.0 + 1e-12]})
assert_table_equal(actual, expected)  # within a relative 1.5e-8
```

`assert_table_equal()` tests `actual` against `expected` as the benchmark and
raises `AssertionError` with `diff_summary()` and the first records that
differ, so it works under pytest, unittest or a plain script. A repeated key
is an error unless `duplicates=` says how to treat it.

## Development

```sh
uv sync
uv run pytest
```
