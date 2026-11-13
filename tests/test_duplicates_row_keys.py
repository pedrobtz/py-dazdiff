import warnings

import numpy as np
import pandas as pd
import pytest

from dazdiff import DiffError, DiffNote, DiffWarning, diff_summary, diff_table, is_matching


def kv(k, v):
    return pd.DataFrame({"k": k, "v": v})


def by_key(d, metric="V"):
    d = d[d["metric"] == metric]
    return dict(zip(d["K"], d["status"].astype(str), strict=True))


# --- duplicates="disambiguate" (default) -------------------------------------


def test_disambiguate_pairs_in_arrival_order():
    with pytest.warns(DiffWarning, match="repeated key"):
        d = diff_table(kv([1, 1, 2], [1.0, 2.0, 3.0]), kv([1, 1, 2], [2.0, 1.0, 3.0]))
    assert d.attrs["shape"] == "cells"
    assert d["status"].astype(str).tolist() == ["changed", "changed", "same"]


def test_disambiguate_rows_shape_counts_copies_without_warning():
    x = pd.DataFrame({"k": ["a", "a", "b"]})
    with warnings.catch_warnings():
        warnings.simplefilter("error", DiffWarning)
        with pytest.warns(DiffNote, match="as wholes"):
            d = diff_table(x, x.iloc[[0, 2]])
    assert d.attrs["shape"] == "rows"
    assert d["status"].astype(str).tolist() == ["same", "only_x", "same"]


# --- duplicates="error" ------------------------------------------------------


def test_error_refuses_repeated_keys():
    with pytest.raises(DiffError, match="`x` has 1 row"):
        diff_table(kv([1, 1], [1.0, 2.0]), kv([1, 2], [1.0, 2.0]), duplicates="error")


def test_error_refuses_repeated_rows_too():
    x = pd.DataFrame({"k": ["a", "a"]})
    with pytest.raises(DiffError, match="repeated key"), warnings.catch_warnings():
        warnings.simplefilter("ignore", DiffNote)
        diff_table(x, x, duplicates="error")


def test_error_accepts_unique_keys():
    assert is_matching(
        diff_table(kv([1, 2], [1.0, 2.0]), kv([2, 1], [2.0, 1.0]), duplicates="error")
    )


# --- duplicates="aggregate" --------------------------------------------------


def test_aggregate_compares_multisets():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DiffWarning)  # no pairing, so no warning
        d = diff_table(
            kv([1, 1, 2], [1.0, 2.0, 3.0]), kv([1, 1, 2], [2.0, 1.0, 3.0]), duplicates="aggregate"
        )
    assert d.attrs["shape"] == "groups"
    assert list(d.columns) == [
        "K", "metric", "n_rows_x", "n_rows_y", "values_x", "values_y", "status",
    ]  # fmt: skip
    assert is_matching(d)
    assert d["values_x"].tolist() == [(1.0, 2.0), (3.0,)]
    assert d["n_rows_x"].tolist() == [2, 1]


@pytest.mark.parametrize(
    ("y_k", "y_v", "expected"),
    [
        ([1, 1, 2], [1.0, 5.0, 3.0], {1: "changed", 2: "same"}),
        ([1, 2], [1.0, 3.0], {1: "changed", 2: "same"}),  # a copy missing
        ([1, 1, 3], [1.0, 2.0, 3.0], {1: "same", 2: "only_x", 3: "only_y"}),
    ],
    ids=["value", "count", "keys"],
)
def test_aggregate_statuses(y_k, y_v, expected):
    d = diff_table(kv([1, 1, 2], [1.0, 2.0, 3.0]), kv(y_k, y_v), duplicates="aggregate")
    assert by_key(d) == expected


def test_aggregate_missing_values_and_signed_zero():
    x = kv([1, 1, 2], [np.nan, 1.0, 0.0])
    y = kv([1, 1, 2], [1.0, np.nan, -0.0])
    assert is_matching(diff_table(x, y, duplicates="aggregate"))
    y = kv([1, 1, 2], [1.0, 1.0, 0.0])
    assert by_key(diff_table(x, y, duplicates="aggregate")) == {1: "changed", 2: "same"}


def test_aggregate_per_measure():
    x = pd.DataFrame({"k": [1, 1], "a": [1.0, 2.0], "b": [5.0, 6.0]})
    y = pd.DataFrame({"k": [1, 1], "a": [2.0, 1.0], "b": [5.0, 7.0]})
    d = diff_table(x, y, duplicates="aggregate")
    assert dict(zip(d["metric"], d["status"].astype(str), strict=True)) == {
        "A": "same",
        "B": "changed",
    }


@pytest.mark.parametrize("tol", [{"tolerance": 0.1}, {"rel_tol": 0.1}])
def test_aggregate_refuses_tolerances(tol):
    with pytest.raises(DiffError, match="must be 0"):
        diff_table(kv([1], [1.0]), kv([1], [1.0]), duplicates="aggregate", **tol)


def test_aggregate_disjoint_is_an_error():
    with pytest.raises(DiffError, match="in common"):
        diff_table(kv([1], [1.0]), kv([2], [1.0]), duplicates="aggregate")


def test_aggregate_without_measures_compares_rows():
    x = pd.DataFrame({"k": ["a", "a", "b"]})
    with pytest.warns(DiffNote, match="as wholes"):
        d = diff_table(x, x, duplicates="aggregate")
    assert d.attrs["shape"] == "rows"
    assert is_matching(d)


def test_summary_of_groups_shape():
    d = diff_table(kv([1, 1, 2], [1.0, 2.0, 3.0]), kv([1, 2], [1.0, 3.0]), duplicates="aggregate")
    s = diff_summary(d)
    assert s[["metric", "n", "n_same", "n_changed"]].values.tolist() == [["V", 2, 1, 1]]
    assert "max_abs_diff" not in s


# --- row_key -----------------------------------------------------------------


def measures_only():
    return pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})


def test_all_measure_table_needs_a_row_key():
    with pytest.raises(DiffError, match="row_key"):
        diff_table(measures_only(), measures_only())


def test_row_key_position_pairs_rows_as_they_come():
    x = measures_only()
    y = x.copy()
    y.loc[1, "b"] = 25.0
    d = diff_table(x, y, row_key="position")
    assert d.attrs["by"] == ["row_number"]
    assert "row_number" in d.columns
    changed = d[d["status"] == "changed"]
    assert changed[["row_number", "metric"]].values.tolist() == [[2, "B"]]


def test_row_key_position_extra_rows():
    x = measures_only()
    d = diff_table(x, x.iloc[:2], row_key="position")
    assert set(d.loc[d["row_number"] == 3, "status"]) == {"only_x"}


def test_row_key_sorted_ignores_row_order():
    x = measures_only()
    y = x.iloc[[2, 0, 1]]
    assert not is_matching(diff_table(x, y, row_key="position"))
    d = diff_table(x, y, row_key="sorted")
    assert is_matching(d)
    rows = d[d["metric"] == "A"]
    assert rows["row_id_y"].tolist() == [2, 3, 1]  # traced back to y's source rows


def test_row_key_sorted_sorts_on_all_compared_columns():
    x = pd.DataFrame({"a": [1.0, 1.0], "b": [9.0, 8.0]})
    y = pd.DataFrame({"a": [1.0, 1.0], "b": [8.0, 9.0]})
    assert is_matching(diff_table(x, y, row_key="sorted"))


def test_row_key_keeps_other_key_columns():
    x = pd.DataFrame({"g": ["a", "b"], "v": [1.0, 2.0]})
    d = diff_table(x, x, row_key="position")
    assert d.attrs["by"] == ["G", "row_number"]


def test_row_key_cannot_combine_with_by():
    with pytest.raises(DiffError, match="cannot be combined"):
        diff_table(measures_only(), measures_only(), row_key="position", by=["a"])


def test_row_key_must_be_known():
    with pytest.raises(DiffError, match="row_key"):
        diff_table(measures_only(), measures_only(), row_key="nope")
