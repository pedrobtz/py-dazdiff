# Changelog

## 0.1.0 (unreleased)

First release.


- `diff_table()` compares two DataFrames value by value, in benchmark or
  equal mode; `diff_summary()` and `is_equal()` condense the result.
- The `dazdiff` command compares two CSV files and exits 0 when they match,
  1 when they differ and 2 on error.
- Column names are normalized by collapsing runs of other characters into
  one `_` and trimming `_` from the ends; empty or clashing names are an
  error, and messages name columns as you wrote them.
- With `by`, other non-measure columns are ignored with a note in benchmark
  mode and are an error in equal mode.
- Comparing tables that share no key is an error.
- Equal infinities are `same`; tolerances apply to finite pairs only;
  `rel_diff` no longer overflows.
- Side effects (dropped columns, truncation, no measures) are `DiffNote`
  warnings; `DiffWarning` is kept for answers that may be wrong.
- `y` can be cast to bool, string and date-time columns of the benchmark.
- Arguments after `x` and `y` are keyword-only; the result's `attrs` hold
  `shape` and `mode`.
- Requires pandas 2.2.2 or later.
- `duplicates="pair"` is now `duplicates="disambiguate"`, as in daffiz.
- `duplicates="aggregate"` compares the multiset of values per key and
  measure (`attrs["shape"] == "groups"`).
- `row_key="position"` and `"sorted"` add a `row_number` key, so tables
  whose columns are all measures can be compared; the CLI has `--row-key`.
- `exclude=` (`--exclude`) leaves columns out of both tables; a name in
  neither table is a warning, one that `by` or `measures` names an error.
- `cast=False` (`--no-cast`) compares without casting; the compared columns'
  types must agree.
- Only the compared columns of `y` are cast, so an ignored column can no
  longer refuse the comparison or be truncated.
- A number cast to the benchmark's text is written with the decimals the
  benchmark uses, so `2` meets `"2.0"`; a value that matches only as a
  number warns.
- A date-time cast to text is written `%Y-%m-%d %H:%M:%S`, with its fraction
  of a second when it has one; a column with no values casts to any type.
- When no key aligns, the error names the key columns that share no value,
  with examples, and suggests `exclude` for those holding one value each.
- `attrs` also hold `tolerance`, `name_map`, `dropped`, `excluded` and
  `ignored`.
- `is_equal()` is now `is_matching()`, which can judge a result again at
  other `tolerance` and `rel_tol` thresholds.
- `diff_explain()` (`--explain [COLUMN]`, `--joined`) says which key values
  split the result into same and not same: per key column and per measure,
  or per value of one column.
- `diff_report()` (`--report`) draws a bounded text report: the columns
  compared, a picture of how the rows align, and per changed measure its
  match rate and largest differences, in Unicode or ASCII (`--ascii`, and
  automatically when the terminal cannot show Unicode).
- `attrs["paired"]` says whether repeated keys were paired.
- `assert_table_equal(actual, expected)` asserts that a table matches a
  reference within a relative `1.5e-8`, failing with `diff_summary()` and
  the first records that differ.
- A cast to an integer column is exact above 2**53 and refuses values
  outside the column's range, instead of wrapping them around.
- Integer measures are compared exactly, so values above 2**53 that differ
  are `changed`.
- A number cast to a date-time column is refused instead of being read as
  nanoseconds since 1970.
- The CLI no longer turns an integer key column into a measure when the
  other file reads it as float only because of an empty value.
- Every CLI error exits with status 2 and a one-line message, including
  errors writing `--output` and undecodable files.
- The report's "(...and N more)" counts keys not shown, not rows.
- Extra columns of `y` whose names normalize to nothing or clash are listed
  as dropped, in `attrs["dropped"]` and the note.
