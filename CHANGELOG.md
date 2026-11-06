# Changelog

## Unreleased

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
