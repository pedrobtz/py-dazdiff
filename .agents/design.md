# dazdiff (Python): design

Status: draft 2026-11-06; brought in line with the R package's main
(`2bb27b2`) on 2026-11-13. The order of work is in [roadmap.md](roadmap.md).

## 1 Purpose and scope

`dazdiff` compares two tables value by value and says, for every row and
numeric measure, whether the value is `same`, `changed`, `only_x` or `only_y`.
It runs as a library on pandas DataFrames and as a CLI on two CSV files.

It answers one of two questions:

- **Does this table match the benchmark?** (`mode="benchmark"`) `x` is the
  source of truth; `y` is cast to its types and its extra columns dropped.
- **Are these two tables equal?** (`mode="equal"`) The tables are peers:
  same columns, same types, nothing cast. The result is symmetric.

Out of scope for the package as a whole: fuzzy row matching, schema
migration, comparing more than two tables.

## 2 Relationship to dazdiff (R)

This is a Python port of the R package
[`dazdiff`](https://github.com/pedrobtz/dazdiff), called `daffiz` until its
rename (R #46, #59; noted 2026-11-13). Its
`.agents/plan.md` is the behavioural reference for 0.1.0, and its
`.agents/roadmap.md` (the features dropped in its core rebuild) is the
reference for what comes after.

- Behaviour matches the R package unless this document records a reason
  not to. Each such divergence is listed in §2.1.
- The API is Pythonic: snake_case arguments, exceptions and warnings instead
  of R conditions, pandas instead of data.table.
- Features the R package decided against (e.g. `duplicate_keys = "pair"`
  by sorting values, the plot) are not ported.

### 2.1 Divergences from the R package

| R | Python | Reason |
| --- | --- | --- |
| `nan_is_na` argument | none; NaN and NA are always the same missing value | pandas float columns cannot tell NaN from NA, and CSV has neither |
| messages for side effects | `DiffNote` warnings (§4.9) | Python has no message condition; warnings can be filtered and caught in tests |
| `normalize_dt()`, `cast_rules()`, `compare_columns()` exported | internal | not needed for 0.1.0; revisit if users ask |
| `date_format` argument | none | CSV dates arrive as text and stay identity columns; Date measures are deferred |
| `"groups"` shape: `value_key_x/y` (`%a` hex text) and `n_na_x/y` | `values_x/y`, sorted tuples | a tuple of floats is exact and readable; no text encoding is needed to compare multisets in pandas |
| condition classes (`dazdiff_error_*`, …) | three classes: `DiffError`, `DiffWarning`, `DiffNote` | the message names what happened; finer classes only if users ask |
| result keyed by the identity | sorted by the identity, default `RangeIndex` | pandas has no keys; sorting gives the same reading order |
| CLI | first-class, in 0.1.0 | the R roadmap marks it L; for Python it is the main entry point |
| one empty table and one not: an error (no key aligns) | its rows are `only_x` or `only_y` (§4.4) | decided 2026-11-06; an empty load is a finding to report, not a question that cannot be answered (recorded here 2026-11-13) |

## 3 API

```python
diff_table(x, y, *,
           mode="benchmark",              # or "equal"
           by=None,                       # key column(s); default: every non-measure column
           exclude=None,                  # columns left out of the comparison
           row_key="none",                # or "position", "sorted"
           measures="numeric",            # or "numeric+integer", or column names
           tolerance=0.0,
           rel_tol=0.0,
           duplicates="disambiguate",     # or "aggregate", "error"
           cast=None,                     # None: True in benchmark mode; equal mode never casts
           truncate=True) -> pd.DataFrame

diff_summary(d) -> pd.DataFrame           # counts per measure, plus magnitudes
diff_report(d, x_name="x", y_name="y", *, sample_k=5) -> DiffReport   # str(report) is the text
is_matching(d, *, tolerance=None, rel_tol=None) -> bool
diff_explain(d, column=None, *, among="all") -> pd.DataFrame   # or among="joined"
assert_table_equal(actual, expected, *, rel_tol=1.5e-8, max_rows=10, **kwargs) -> pd.DataFrame
normalize_names(names) -> list[str]

class DiffError(ValueError)               # the question cannot be answered
class DiffWarning(UserWarning)            # the answer may be wrong
class DiffNote(UserWarning)               # an expected side effect
```

Arguments after `y` are keyword-only. Everything else is internal.

## 4 Behaviour of `diff_table()`

### 4.1 Modes

- **benchmark**: `y`'s extra columns are dropped (note). A column missing
  from `y` is an error. `y` is cast to `x`'s types (§4.3).
- **equal**: every column must be on both sides and have the same dtype,
  else an error listing them. Nothing is cast.
- `cast=False` in benchmark mode keeps the column checks but requires the
  compared columns to have `x`'s dtypes already; `cast=True` in equal mode
  is an error. (Added 2026-11-13, as R's `cast` argument.)

### 4.1a Excluded columns

`exclude` names columns, in either spelling, that are removed from both
tables before anything else, in either mode. An excluded column need not be
in both tables, is not announced as dropped or ignored, and does not stop
equal mode from calling the rest equal. A name in neither table is a
`DiffWarning`, so one list of technical columns can serve many tables; a
name that `by` or `measures` also names is a `DiffError`. The removed
columns, as written, are in `attrs["excluded"]`. (Added 2026-11-13, R #65,
#68; was an "After 0.1.0" item.)

### 4.2 Column names

Both tables' names are normalized: upper-case, every run of characters other
than `A-Z` and `0-9` replaced by one `_`, `_` trimmed from both ends.

- A name that normalizes to empty is an error.
- Two names that normalize to the same thing are an error. In benchmark mode
  this applies to `y` only for the columns `x` uses.
- The result's own columns are lowercase (`metric`, `key_seq`, `row_id_x`,
  …), so they can never collide with a normalized name.
- Errors name columns as the user wrote them.

### 4.3 Casting (benchmark only)

Roles (§4.4) are decided first, on `x`, and only the compared columns of
`y` (keys and measures) are cast; a column the call ignores is neither cast
nor type-checked. (Decided 2026-11-13, after R: casting first let an
ignored column refuse the comparison or report a truncation.)

`y`'s column is cast to `x`'s dtype. Supported target types: integer, float,
bool, string, datetime.

- Float → integer truncates when `truncate=True`, with one note listing each
  column, its count and an example; `truncate=False` refuses with an error.
  Truncations are in `result.attrs["truncated"]`.
- A value that cannot be cast is an error naming the column and the value.
- To integer, exactly (decided 2026-11-13, Stage 9 review, #25): integer
  dtypes and integer text become Python ints, so a value above 2**53 keeps
  every digit; other numbers go through float and truncate. A value outside
  the target dtype's range (int8 … uint64) is an error, never wrapped.
- To date-time: numbers and booleans are refused (#27); pandas would read
  them as nanoseconds since 1970.
- Both pandas 2 (object strings) and pandas 3 (`str` dtype) are supported.
- To bool: numbers must be 0 or 1; text is `true`/`t`/`yes`/`1` or
  `false`/`f`/`no`/`0`, case-insensitive. To string: see the number-to-text
  rule below. Categorical and other dtypes are refused as targets.
  (Decided 2026-11-06, Stage 1.)
- `attrs["truncated"]` is keyed by the normalized column name, as the result
  is; notes and errors use the name as written.
- **Number to text** (decided 2026-11-13, after R #65): the benchmark's
  decimals are read off every distinct value of `x`'s column. When all are
  plain decimal text with one number of decimals, `y`'s numbers are written
  with that many, so `2` meets `"2.0"`, announced by a `DiffNote`. A value is
  reformatted only when the text reads back exactly, so `1.25` never becomes
  `"1.3"`. Otherwise integral floats lose their `.0`, as before. A value of
  `x` equal to one of `y`'s as a number but not as text is a `DiffWarning`
  that names the format used, since those rows will not align.
- A date-time cast to text is written `%Y-%m-%d %H:%M:%S`, midnight included,
  with its fraction to the microsecond when it has one.
- A column of `y` with no values at all (every value missing, whatever its
  dtype) casts to any target type.

### 4.4 Measures and identity

- `measures` picks numeric columns of `x`: `"numeric"` (float),
  `"numeric+integer"`, or names. A named non-numeric column is an error.
- The identity is `by`, or every column that is not a measure.
- With `by` given, left-over columns are ignored with a note in benchmark
  mode and are an error in equal mode.
- A table needs at least one key column, unless `row_key` adds one (§4.7).
  It may have no measures: then the question is "do they have the same
  rows?", and the result has one row per source row (§4.8).
- If the two tables have keys but no key in common, it is an error: the
  comparison would say nothing useful. The error names the key columns whose
  values the two tables do not share, with examples, and suggests `exclude`
  for those holding one value in each table (2026-11-13, after R #65). An
  empty table is not disjoint: its rows are all `only_x` or `only_y`
  (decided 2026-11-06).

### 4.5 Values and tolerance

A cell is `same` when both values are missing, or both are equal (including
equal infinities), or, for a finite pair, `abs_diff <= tolerance` or
`rel_diff <= rel_tol`. Otherwise it is `changed`.

- `diff = value_y - value_x`; `abs_diff = |diff|`.
- `rel_diff = abs_diff / max(|value_x|, |value_y|)`, 0 when both are 0,
  computed by scaling before subtracting so it cannot overflow.
- Integer measures are compared as integers (decided 2026-11-13, Stage 9
  review, #26): `status` and `diff` are exact above 2**53, `diff` computed in
  int64 with overflow detected. `value_x`/`value_y` are still float64.

### 4.6 Duplicate keys

- **disambiguate** (default): rows sharing a key are paired in arrival order
  (`key_seq`), with a `DiffWarning`. No warning when there are no measures,
  since identical rows are interchangeable.
- **aggregate**: compares the multiset of values per key and measure;
  `tolerance` and `rel_tol` must be 0.
- **error**: refuses, also for repeated rows when there are no measures,
  since the caller asked for it (decided 2026-11-06, Stage 2).
- Aggregated values are sorted tuples (`values_x`, `values_y`), missing
  values last; `-0.0` and `0.0` are the same value.

### 4.7 Virtual row keys

When every column is a measure, `row_key` adds a `row_number` key:
`"position"` numbers rows as they come; `"sorted"` numbers them after sorting
both tables by their compared columns.

- Non-measure columns, if any, stay in the identity next to `row_number`.
- `row_key` cannot be combined with `by` (decided 2026-11-06, Stage 2: `by`
  names the identity, `row_key` invents one).

### 4.8 Result

A DataFrame sorted by the identity, with `status` a categorical of `same`,
`changed`, `only_x`, `only_y`. Its shape is in `attrs["shape"]`:

| Shape | When | One row per |
| --- | --- | --- |
| `cells` | measures exist | row × measure: `metric`, `row_id_x/y`, `value_x/y`, `diff`, `abs_diff`, `rel_diff` |
| `groups` | `duplicates="aggregate"` | key × measure: `n_rows_x/y`, `values_x/y` |
| `rows` | no measures | source row: `key_seq`, `row_id_x/y` |

`attrs` also holds `by`, `measures`, `mode`, `truncated`, `paired` (whether
repeated keys were paired, since `cells` drops `key_seq`; added 2026-11-13,
Stage 5, for the report), `tolerance`
(`{"abs": …, "rel": …}`), `name_map` (each result column's name as written
in `x` and in `y`), `dropped` (columns of `y` dropped in benchmark mode),
`excluded` and `ignored` (columns in neither role), names as written. The
extra keys were added 2026-11-13, as R's attributes. Row ids are
1-based, as in the R package. The caller's DataFrames are never modified.

### 4.9 Errors, warnings, notes

- `DiffError` when the question cannot be answered.
- `DiffWarning` when the answer may be wrong: invented pairing.
- `DiffNote` for expected side effects: truncation, dropped or ignored
  columns, no measures.

## 4a Reading a result

- `is_matching(d)` is `True` when every row is `same`; two empty tables
  match. With `tolerance` or `rel_tol` it judges the values again at those
  thresholds, looser or stricter, from the stored `abs_diff` and `rel_diff`,
  by `diff_table()`'s rule (equal or both missing always match; tolerances
  apply to finite pairs only; when one is given, the other is 0). `only_x`
  and `only_y` never match. A tolerance above 0 on the `groups` shape is an
  error (multisets compare exactly); the `rows` shape has no values, so its
  answer does not change. (2026-11-13: replaces `is_equal()`,
  for parity with R; nothing was released.)
- `diff_explain(d)` returns one row per key column (and per `metric` when
  there are measures): `column`, `n_values`, `same_only`, `differ_only`,
  `mixed`, `separates`, `identifies`. A value is `same` when every record
  holding it matches, `differ` when none does, `mixed` otherwise; a column
  separates when no value is mixed. A column with a different value on
  every row is marked `identifies` and ranked last; the others by fewest
  mixed, then fewest values. For a key column the unit is the row (a row
  matches when all its measures do), for `groups` the key, for `metric` the
  value. A row is its pair of `row_id_x`/`row_id_y`, since the `cells` result
  has no `key_seq`, so paired duplicates count as rows of their own. `diff_explain(d, column)` returns one row per value of that
  column: `n`, `n_same`, `n_changed`, `n_only_x`, `n_only_y`, `class`, most
  differences first. `among="joined"` counts only rows found in both
  tables. (Added 2026-11-13, after R #66.)

## 5 Report

`diff_report(d, x_name, y_name)` lays out the comparison as text, after
the R package's `diff_report()`: identity and mode, columns compared/dropped/
truncated, a diagram of how the rows align, a table per measure (match rate,
max |diff|, max rel diff), and the keys only in each table. No "Top changes"
column (dropped 2026-11-13, as in R #65: it made the table too wide), and
with it `top_k`. Sections are bounded by `sample_k`.

- `str(report)` is Unicode; `report.format(unicode=False)` is ASCII, and the
  CLI falls back to it when stdout cannot encode Unicode (Windows consoles).
- No colour and no width argument: the report is as wide as its widest line,
  at least 40 columns, as in R (decided 2026-11-13, Stage 5; the earlier
  `width` argument had no R counterpart).
- The text is the R package's, byte for byte, for the same comparison: its
  report snapshots are this package's test fixtures. Numbers are written as
  R's `format(digits=)` writes them.
- "(...and N more)" counts the distinct keys not shown, not rows (#30;
  the R package counts rows, so a repeated key inflates it there).
- `DiffReport` holds the computed report as data (`rows`, `by_measure`,
  `only_x`, `only_y`, …) apart from its drawing.

## 6 Test helper

`assert_table_equal(actual, expected, **kwargs)` runs
`diff_table(expected, actual)`, `expected` as the benchmark, and raises
`AssertionError` with `diff_summary()` and the first `max_rows` records that
differ; it returns the comparison when they match. It is plain Python,
usable from pytest or unittest, with no pytest plugin.

- `rel_tol` defaults to `sqrt(float eps)` (about 1.5e-8), as the R package's
  `expect_table_equal()`; with `duplicates="aggregate"`, which compares
  exactly, it defaults to 0.
- Repeated keys follow the R package (decided 2026-11-13, Stage 6, replacing
  `duplicates="error"` as the default): the pairing `DiffWarning` becomes a
  `DiffError` unless `duplicates` is given, so an assertion never decides
  identity by arrival order; repeated whole rows (no measures) are paired,
  since they are identical, where `"error"` would have refused them.
- Tables that share no key fail the assertion; every other `DiffError`
  (missing columns, uncastable values) is raised as is, since the question
  cannot be answered.

## 7 CLI

`dazdiff X.csv Y.csv [options]` reads both files, runs `diff_table()`, and
prints the result, `--summary`, or `--report`.

- Every `diff_table()` argument has a flag; list values are comma-separated.
- Exit status: 0 when every row is `same`, 1 when they differ, 2 on any
  error, with a one-line message and no traceback (#29).
- Warnings and notes go to stderr; `--quiet` silences everything.
- CSV carries no types: a column read as integer in one file and as float
  holding a fraction in the other is read as float in both, so the cast does
  not truncate it. A float column of whole numbers (float only because of an
  empty value) is left alone, so an integer key stays a key (decided
  2026-11-13, Stage 9 review, #28).
- `--output FILE` writes the full result as CSV.
- `--report` prints `diff_report()`, naming the tables by their file names
  unless `--x-name`/`--y-name` say otherwise; `--ascii` forces ASCII
  (2026-11-13, Stage 5).
- `--exclude` and `--no-cast` map to `exclude` and `cast=False`;
  `--explain [COLUMN]` prints `diff_explain()`, with `--joined` for
  `among="joined"` (added 2026-11-13).
- `python -m dazdiff` is the same program.

## 8 Dependencies and supported versions

- Runtime: `pandas>=2.2.2` (and numpy through it). No other hard dependency.
  Decided 2026-11-06: 2.2.2 is the first pandas that loads against numpy 2;
  with `pandas>=2.0` the lowest-dependencies resolution installed pandas 2.0
  with numpy 2 and failed to import.
- Python: 3.10 to 3.14. Python 3.10 resolves to pandas 2.x, which keeps the
  pandas 2 path tested.
- Dev: `pytest`, `ruff`; managed with `uv`.

## 9 Packaging and release

- Build backend: hatchling; `src/` layout; `py.typed` shipped.
- License: MIT.
- Published to PyPI under the name from §11, via a GitHub Actions workflow
  using PyPI trusted publishing on a `v*` tag. TestPyPI first.
- `.github/workflows/release.yml` builds, runs `twine check` and the built
  wheel's CLI on the examples, then publishes: a manual run to TestPyPI
  (environment `testpypi`), a `v*` tag to PyPI (environment `pypi`), the tag
  checked against `project.version`. No API tokens.
- The version has one source, `project.version` in `pyproject.toml`;
  `dazdiff.__version__` reads it from the installed metadata (decided
  2026-11-13, Stage 8).
- `.agents/` is excluded from the sdist.

## 10 After 0.1.0

What comes later is listed in [roadmap.md](roadmap.md#after-010).

## 11 Open questions

- ~~Distribution name.~~ Decided 2026-11-13: `dazdiff` for the
  distribution, import and CLI. The R package was renamed `dazdiff` too
  (R #46, #59), so both share the name.
- ~~Argument name `duplicates="pair"`.~~ Decided 2026-11-06, Stage 2:
  renamed to `"disambiguate"` for parity with daffiz; nothing was released.
