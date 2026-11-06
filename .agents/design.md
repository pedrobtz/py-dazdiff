# dazdiff (Python): design

Status: draft 2026-11-06. The order of work is in [roadmap.md](roadmap.md).

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

## 2 Relationship to daffiz (R)

This is a Python port of the R package
[`daffiz`](https://github.com/pedrobtz/dazdiff) (repository `dazdiff`). Its
`.agents/plan.md` is the behavioural reference for 0.1.0, and its
`.agents/roadmap.md` (the features dropped in its core rebuild) is the
reference for what comes after.

- Behaviour matches daffiz unless this document records a reason not to.
  Each such divergence is listed in §2.1.
- The API is Pythonic: snake_case arguments, exceptions and warnings instead
  of R conditions, pandas instead of data.table.
- Features daffiz decided against (e.g. `duplicate_keys = "pair"` by sorting
  values, the plot) are not ported.

### 2.1 Divergences from daffiz

| daffiz | dazdiff | Reason |
| --- | --- | --- |
| `nan_is_na` argument | none; NaN and NA are always the same missing value | pandas float columns cannot tell NaN from NA, and CSV has neither |
| messages for side effects | `DiffNote` warnings (§4.9) | Python has no message condition; warnings can be filtered and caught in tests |
| `normalize_dt()`, `cast_rules()`, `compare_columns()` exported | internal | not needed for 0.1.0; revisit if users ask |
| `date_format` argument | none | CSV dates arrive as text and stay identity columns; Date measures are deferred |
| result keyed by the identity | sorted by the identity, default `RangeIndex` | pandas has no keys; sorting gives the same reading order |
| CLI | first-class, in 0.1.0 | the R roadmap marks it L; for Python it is the main entry point |

## 3 API

```python
diff_table(x, y, *,
           mode="benchmark",              # or "equal"
           by=None,                       # key column(s); default: every non-measure column
           row_key="none",                # or "position", "sorted"
           measures="numeric",            # or "numeric+integer", or column names
           tolerance=0.0,
           rel_tol=0.0,
           duplicates="disambiguate",     # or "aggregate", "error"
           truncate=True) -> pd.DataFrame

diff_summary(d) -> pd.DataFrame           # counts per measure, plus magnitudes
diff_report(d, x_name="x", y_name="y") -> DiffReport   # str(report) is the text
is_equal(d) -> bool
assert_table_equal(actual, expected, *, rel_tol=1.5e-8, **kwargs) -> None
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

`y`'s column is cast to `x`'s dtype. Supported target types: integer, float,
bool, string, datetime.

- Float → integer truncates when `truncate=True`, with one note listing each
  column, its count and an example; `truncate=False` refuses with an error.
  Truncations are in `result.attrs["truncated"]`.
- A value that cannot be cast is an error naming the column and the value.
- Both pandas 2 (object strings) and pandas 3 (`str` dtype) are supported.
- To bool: numbers must be 0 or 1; text is `true`/`t`/`yes`/`1` or
  `false`/`f`/`no`/`0`, case-insensitive. To string: integral floats lose
  their `.0`, so a key read as `1.0` matches `"1"`. Categorical and other
  dtypes are refused as targets. (Decided 2026-11-06, Stage 1.)
- `attrs["truncated"]` is keyed by the normalized column name, as the result
  is; notes and errors use the name as written.

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
  comparison would say nothing useful. An empty table is not disjoint: its
  rows are all `only_x` or `only_y` (decided 2026-11-06).

### 4.5 Values and tolerance

A cell is `same` when both values are missing, or both are equal (including
equal infinities), or, for a finite pair, `abs_diff <= tolerance` or
`rel_diff <= rel_tol`. Otherwise it is `changed`.

- `diff = value_y - value_x`; `abs_diff = |diff|`.
- `rel_diff = abs_diff / max(|value_x|, |value_y|)`, 0 when both are 0,
  computed by scaling before subtracting so it cannot overflow.
- Integer measures are compared as float.

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

`attrs` also holds `by`, `measures`, `mode` and `truncated`. Row ids are
1-based, as in daffiz. The caller's DataFrames are never modified.

### 4.9 Errors, warnings, notes

- `DiffError` when the question cannot be answered.
- `DiffWarning` when the answer may be wrong: invented pairing.
- `DiffNote` for expected side effects: truncation, dropped or ignored
  columns, no measures.

## 5 Report

`diff_report(d, x_name, y_name)` lays out the comparison as text, after
daffiz's `diff_report()`: identity and mode, columns compared/dropped/
truncated, a diagram of how the rows align, a table per measure (match rate,
max |diff|, max rel diff, top changes with an example key), and the keys
only in each table. Sections are bounded (`top_k`, `sample_k`).

- `str(report)` is Unicode; `report.format(unicode=False)` is ASCII, and the
  CLI falls back to it when stdout cannot encode Unicode (Windows consoles).
- No colour, no terminal width detection beyond a `width` argument.

## 6 Test helper

`assert_table_equal(actual, expected, **kwargs)` runs `diff_table(actual,
expected-as-benchmark)` and raises `AssertionError` with `diff_summary()`
and the first differing rows. Defaults: `rel_tol=1.5e-8`
(`sqrt(float eps)`, as daffiz's `expect_table_equal()`), and
`duplicates="error"` so a repeated key fails rather than being paired.
It is plain Python, usable from pytest or unittest, with no pytest plugin.

## 7 CLI

`dazdiff X.csv Y.csv [options]` reads both files, runs `diff_table()`, and
prints the result, `--summary`, or `--report`.

- Every `diff_table()` argument has a flag; list values are comma-separated.
- Exit status: 0 when every row is `same`, 1 when they differ, 2 on error.
- Warnings and notes go to stderr; `--quiet` silences everything.
- CSV carries no types: a column read as integer in one file and float in
  the other is read as float in both, so the cast does not truncate it.
- `--output FILE` writes the full result as CSV.
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
- `.agents/` is excluded from the sdist.

## 10 After 0.1.0

What comes later is listed in [roadmap.md](roadmap.md#after-010).

## 11 Open questions

- **Distribution name.** The R package is `daffiz`; this repository and its
  import name are `dazdiff`. Both are free on PyPI (checked 2026-11-06).
  Proposed: keep `dazdiff` for the distribution, import and CLI, and reserve
  `daffiz` only if the R and Python packages should share a name. Blocks the
  v0.1.0 readiness stage.
- ~~Argument name `duplicates="pair"`.~~ Decided 2026-11-06, Stage 2:
  renamed to `"disambiguate"` for parity with daffiz; nothing was released.
