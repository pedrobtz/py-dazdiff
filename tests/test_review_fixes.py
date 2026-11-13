"""Fixes from the Stage 9 code review (#9)."""

import numpy as np
import pandas as pd
import pytest

from dazdiff import DiffError, diff_table, is_matching

BIG = 2**53


# --- #25: casts to integer -------------------------------------------------------


@pytest.mark.parametrize("dtype", ["int8", "int16", "int32", "uint8", "uint32"])
def test_a_value_outside_the_integer_range_is_refused(dtype):
    x = pd.DataFrame({"id": [1], "v": pd.Series([1], dtype=dtype)})
    y = pd.DataFrame({"id": [1], "v": [float(np.iinfo(dtype).max) + 1]})
    with pytest.raises(DiffError, match="outside the range"):
        diff_table(x, y, by="id", measures="numeric+integer")


def test_a_negative_value_is_outside_an_unsigned_range():
    x = pd.DataFrame({"id": [1], "v": pd.Series([1], dtype="uint8")})
    y = pd.DataFrame({"id": [1], "v": [-1]})
    with pytest.raises(DiffError, match="outside the range"):
        diff_table(x, y, by="id", measures="numeric+integer")


def test_integer_text_casts_exactly_above_2_53():
    x = pd.DataFrame({"id": [BIG, BIG + 2], "v": [1.0, 2.0]})
    y = pd.DataFrame({"id": [str(BIG + 1), str(BIG + 2)], "v": [1.0, 2.0]})
    d = diff_table(x, y)
    assert sorted(d["status"].astype(str)) == ["only_x", "only_y", "same"]
    assert d.loc[d["status"] == "only_y", "ID"].tolist() == [BIG + 1]


def test_integer_dtypes_cast_exactly_above_2_53():
    x = pd.DataFrame({"id": np.array([BIG], dtype="int64"), "v": [1.0]})
    y = pd.DataFrame({"id": pd.Series([BIG + 1], dtype="Int64"), "v": [1.0]})
    with pytest.raises(DiffError, match="key in common"):
        diff_table(x, y)


def test_integer_casts_keep_missing_values_and_truncate_as_before():
    x = pd.DataFrame({"id": [1, 2, 3], "n": pd.Series([1, 2, 3], dtype="int16"), "v": [1.0] * 3})
    y = pd.DataFrame({"id": [1, 2, 3], "n": [1.5, None, 3.0], "v": [1.0] * 3})
    with pytest.warns(UserWarning):
        d = diff_table(x, y, measures=["v"], by=["id", "n"])
    assert d.attrs["truncated"] == {"N": [(1.5, 1.0)]}
    assert set(d.loc[d["ID"] == 2, "status"].astype(str)) == {"only_x", "only_y"}
    with pytest.raises(DiffError, match="truncate"):
        diff_table(x, y, measures=["v"], by=["id", "n"], truncate=False)


def test_integer_text_with_a_fraction_truncates():
    x = pd.DataFrame({"id": [1, 2], "v": [1.0, 2.0]})
    y = pd.DataFrame({"id": ["1", "2.7"], "v": [1.0, 2.0]})
    with pytest.warns(UserWarning, match="Truncated"):
        d = diff_table(x, y)
    assert is_matching(d)


def test_text_that_is_not_a_number_is_still_refused():
    x = pd.DataFrame({"id": [1], "v": [1.0]})
    y = pd.DataFrame({"id": ["one"], "v": [1.0]})
    with pytest.raises(DiffError, match="'one' is not one"):
        diff_table(x, y)


# --- #26: integer measures compared exactly ---------------------------------------


def test_integer_measures_above_2_53_are_compared_exactly():
    x = pd.DataFrame({"id": [1, 2], "v": [BIG, BIG]})
    y = pd.DataFrame({"id": [1, 2], "v": [BIG + 1, BIG]})
    d = diff_table(x, y, by="id", measures="numeric+integer")
    assert d["status"].astype(str).tolist() == ["changed", "same"]
    assert d["diff"].tolist() == [1.0, 0.0]
    assert d["abs_diff"].tolist() == [1.0, 0.0]


def test_integer_measures_with_missing_values():
    x = pd.DataFrame({"id": [1, 2, 3], "v": pd.Series([BIG, None, 5], dtype="Int64")})
    y = pd.DataFrame({"id": [1, 2, 3], "v": pd.Series([BIG + 1, None, 5], dtype="Int64")})
    d = diff_table(x, y, by="id", measures="numeric+integer", mode="equal")
    assert d["status"].astype(str).tolist() == ["changed", "same", "same"]


def test_integer_measures_at_the_int64_limits_do_not_overflow():
    lo, hi = np.iinfo("int64").min, np.iinfo("int64").max
    x = pd.DataFrame({"id": [1], "v": np.array([lo], dtype="int64")})
    y = pd.DataFrame({"id": [1], "v": np.array([hi], dtype="int64")})
    d = diff_table(x, y, by="id", measures="numeric+integer")
    assert d["diff"].iloc[0] == float(hi) - float(lo)
    assert d["status"].astype(str).tolist() == ["changed"]


def test_integer_measures_within_a_tolerance():
    x = pd.DataFrame({"id": [1], "v": [100]})
    y = pd.DataFrame({"id": [1], "v": [101]})
    assert is_matching(diff_table(x, y, by="id", measures="numeric+integer", tolerance=1))


# --- #27: numbers are not date-times ------------------------------------------------


@pytest.mark.parametrize("values", [[20240101], [1.5], [True]])
def test_a_number_is_not_cast_to_a_datetime(values):
    x = pd.DataFrame({"t": pd.to_datetime(["2024-01-01"]), "v": [1.0]})
    y = pd.DataFrame({"t": values, "v": [1.0]})
    with pytest.raises(DiffError, match="date-time"):
        diff_table(x, y)


def test_text_still_casts_to_a_datetime():
    x = pd.DataFrame({"t": pd.to_datetime(["2024-01-01"]), "v": [1.0]})
    y = pd.DataFrame({"t": ["2024-01-01"], "v": [1.0]})
    assert is_matching(diff_table(x, y))
