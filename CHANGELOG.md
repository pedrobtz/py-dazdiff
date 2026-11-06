# Changelog

## Unreleased

- `diff_table()` compares two DataFrames value by value, in benchmark or
  equal mode; `diff_summary()` and `is_equal()` condense the result.
- The `dazdiff` command compares two CSV files and exits 0 when they match,
  1 when they differ and 2 on error.
