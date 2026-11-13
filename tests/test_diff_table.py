import warnings

import numpy as np
import pandas as pd
import pytest

from dazdiff import (
    DiffError,
    DiffNote,
    DiffWarning,
    diff_summary,
    diff_table,
    is_matching,
    normalize_names,
)


def quiet(*args, **kwargs):
    """Run diff_table, ignoring notes but not warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DiffNote)
        return diff_table(*args, **kwargs)


def status_of(d, metric, **key):
    sel = d["metric"] == metric
    for k, v in key.items():
        sel &= d[k] == v
    return d.loc[sel, "status"].item()


# --- names -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("amount (EUR)", "AMOUNT_EUR"),
        ("__a  b__", "A_B"),
        ("a.b-c", "A_B_C"),
        ("Région", "R_GION"),
        ("x1", "X1"),
    ],
)
def test_normalize_names(given, expected):
    assert normalize_names([given]) == [expected]


def test_empty_name_is_an_error():
    with pytest.raises(DiffError, match="empty"):
        normalize_names(["ok", "--"])


def test_clashing_names_are_an_error_in_the_users_spelling():
    with pytest.raises(DiffError, match="a b, a_b"):
        normalize_names(["a b", "a_b"])


def test_benchmark_ignores_clashes_in_unused_y_columns():
    x = pd.DataFrame({"k": [1], "v": [1.0]})
    y = pd.DataFrame({"k": [1], "v": [1.0], "extra 1": [0], "extra_1": [0]})
    assert is_matching(quiet(x, y))


def test_benchmark_refuses_clashes_in_used_y_columns():
    x = pd.DataFrame({"k": [1], "v": [1.0]})
    y = pd.DataFrame({"k": [1], "v": [1.0], "V": [2.0]})
    with pytest.raises(DiffError, match="v, V"):
        diff_table(x, y)


def test_errors_name_columns_as_written(benchmark):
    with pytest.raises(DiffError, match="region"):
        diff_table(benchmark, benchmark.drop(columns="region"))


def test_user_column_named_like_an_internal_one():
    x = pd.DataFrame({"key_seq": [1, 2], "v": [1.0, 2.0]})
    d = diff_table(x, x)
    assert "KEY_SEQ" in d.columns
    assert is_matching(d)


# --- modes, measures, identity -----------------------------------------------


def changed_copy(df):
    y = df.copy()
    y.loc[1, "amount"] = 250.0
    return y


@pytest.mark.parametrize("mode", ["benchmark", "equal"])
@pytest.mark.parametrize("measures", ["numeric", "numeric+integer", ["amount"]])
@pytest.mark.parametrize("by", [None, ["id", "region"]])
def test_mode_measures_by(benchmark, mode, measures, by):
    y = changed_copy(benchmark)
    units_is_measure = measures == "numeric+integer"
    if by is not None and not units_is_measure and mode == "equal":
        with pytest.raises(DiffError, match="neither keys nor measures"):
            diff_table(benchmark, y, mode=mode, by=by, measures=measures)
        return
    if by is not None and not units_is_measure:
        with pytest.warns(DiffNote, match="Ignored column"):
            d = diff_table(benchmark, y, mode=mode, by=by, measures=measures)
    else:
        d = diff_table(benchmark, y, mode=mode, by=by, measures=measures)

    if units_is_measure:  # the integer key ID is a measure too, unless `by` names it
        expected_measures = ["AMOUNT", "UNITS"] if by else ["ID", "AMOUNT", "UNITS"]
        expected_by = ["ID", "REGION"] if by else ["REGION"]
    else:
        expected_measures = ["AMOUNT"]
        expected_by = ["ID", "REGION"] if by else ["ID", "REGION", "UNITS"]
    assert d.attrs["measures"] == expected_measures
    assert d.attrs["by"] == expected_by
    assert d.attrs["mode"] == mode
    assert d.attrs["shape"] == "cells"
    assert len(d) == 4 * len(expected_measures)
    assert status_of(d, "AMOUNT", REGION="south") == "changed"
    assert (d["status"] == "changed").sum() == 1


def test_equal_mode_is_symmetric(benchmark):
    a = benchmark
    b = changed_copy(benchmark).iloc[[0, 1, 2]]
    b = pd.concat([b, pd.DataFrame({"id": [9], "region": ["x"], "amount": [1.0], "units": [1]})])
    ab = diff_table(a, b, mode="equal")
    ba = diff_table(b, a, mode="equal")
    swap = {"only_x": "only_y", "only_y": "only_x"}
    assert ab["status"].astype(str).replace(swap).tolist() == ba["status"].astype(str).tolist()
    np.testing.assert_array_equal(ab["value_x"], ba["value_y"])
    np.testing.assert_array_equal(ab["diff"], -ba["diff"])
    np.testing.assert_array_equal(ab["rel_diff"], ba["rel_diff"])


def test_equal_mode_requires_same_columns(benchmark, stored):
    with pytest.raises(DiffError, match="same columns"):
        diff_table(benchmark, stored, mode="equal")


def test_equal_mode_requires_same_types(benchmark):
    y = benchmark.assign(units=benchmark["units"].astype(float))
    with pytest.raises(DiffError, match="same column types: units"):
        diff_table(benchmark, y, mode="equal")


def test_equal_mode_ignores_row_order(benchmark):
    d = diff_table(benchmark, benchmark.iloc[[3, 0, 1, 2]], mode="equal")
    assert is_matching(d)
    assert d["row_id_y"].tolist() == [2, 3, 4, 1]


def test_no_key_is_an_error():
    x = pd.DataFrame({"a": [1.0], "b": [2.0]})
    with pytest.raises(DiffError, match="no key"):
        diff_table(x, x)


def test_no_key_in_common_is_an_error():
    x = pd.DataFrame({"k": [1, 2], "v": [1.0, 2.0]})
    y = pd.DataFrame({"k": [3, 4], "v": [1.0, 2.0]})
    with pytest.raises(DiffError, match="in common"):
        diff_table(x, y)


def test_an_empty_table_is_not_a_disjoint_one():
    x = pd.DataFrame({"k": [1, 2], "v": [1.0, 2.0]})
    d = diff_table(x, x.iloc[:0])
    assert set(d["status"]) == {"only_x"}


def test_named_measure_must_exist_and_be_numeric(benchmark):
    with pytest.raises(DiffError, match="not found in `x`: nope"):
        diff_table(benchmark, benchmark, measures=["nope"])
    with pytest.raises(DiffError, match="not numeric: region"):
        diff_table(benchmark, benchmark, measures=["region"])


def test_arguments_after_y_are_keyword_only(benchmark):
    with pytest.raises(TypeError):
        diff_table(benchmark, benchmark, "equal")


def test_inputs_are_not_modified(benchmark, stored):
    x0, y0 = benchmark.copy(), stored.copy()
    quiet(benchmark, stored, by=["id", "region"])
    pd.testing.assert_frame_equal(benchmark, x0)
    pd.testing.assert_frame_equal(stored, y0)


@pytest.mark.parametrize("bad", [{"mode": "nope"}, {"duplicates": "nope"}, {"rel_tol": -1}])
def test_bad_arguments(benchmark, bad):
    with pytest.raises(DiffError):
        diff_table(benchmark, benchmark, **bad)


# --- benchmark casting -------------------------------------------------------


def test_benchmark_casts_drops_and_truncates(benchmark, stored):
    with pytest.warns(DiffNote) as notes:
        d = diff_table(benchmark, stored, by=["id", "region"], measures="numeric+integer")
    messages = [str(n.message) for n in notes]
    assert any("Dropped column(s) of `y`" in m and "loaded_at" in m for m in messages)
    assert any("Truncated 1 value(s)" in m and "Units" in m for m in messages)
    assert not any(issubclass(n.category, DiffWarning) for n in notes)
    assert d.attrs["truncated"] == {"UNITS": [(10.4, 10.0)]}
    assert status_of(d, "AMOUNT", ID=2) == "changed"
    assert status_of(d, "UNITS", ID=1) == "same"  # 10.4 truncated to 10
    assert status_of(d, "AMOUNT", ID=4) == "only_x"
    assert status_of(d, "AMOUNT", ID=5) == "only_y"
    row = d[(d["ID"] == 2) & (d["metric"] == "AMOUNT")].iloc[0]
    assert row["diff"] == pytest.approx(0.5)
    assert row["rel_diff"] == pytest.approx(0.5 / 200.5)


def test_no_truncate_refuses_cast(benchmark, stored):
    with pytest.raises(DiffError, match="truncate"):
        quiet(benchmark, stored, by=["id", "region"], measures="numeric+integer", truncate=False)


def test_missing_column_in_y_is_an_error(benchmark):
    with pytest.raises(DiffError, match="lacks column"):
        diff_table(benchmark, benchmark.drop(columns="units"))


def k_v(v):
    return pd.DataFrame({"k": [1, 2, 3], "v": v})


@pytest.mark.parametrize(
    ("x_values", "y_values"),
    [
        ([True, False, True], ["true", "False", "1"]),
        ([True, False, True], [1, 0, 1]),
        (["1", "2", "3"], [1, 2, 3]),
        (["1", "2", "3"], [1.0, 2.0, 3.0]),
        (
            pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
            ["2026-01-01", "2026-01-02", "2026-01-03"],
        ),
        ([1.5, 2.0, 3.0], ["1.5", "2", "3"]),
        (pd.array(["a", "b", "c"], dtype="string"), ["a", "b", "c"]),
        (["a", "b", "c"], pd.array(["a", "b", "c"], dtype="string")),
    ],
    ids=[
        "bool<-str",
        "bool<-int",
        "str<-int",
        "str<-float",
        "datetime<-str",
        "float<-str",
        "string<-object",
        "object<-string",
    ],
)
@pytest.mark.filterwarnings("ignore::dazdiff.DiffWarning")  # bool keys repeat
def test_benchmark_casts_keys_to_the_benchmark_type(x_values, y_values):
    # The cast column is a key, so a wrong cast shows up as unmatched rows.
    x = pd.DataFrame({"key": x_values, "m": [1.0, 2.0, 3.0]})
    y = pd.DataFrame({"key": y_values, "m": [1.0, 2.0, 3.0]})
    assert is_matching(diff_table(x, y, by=["key"]))


@pytest.mark.parametrize(
    ("x_values", "y_values", "message"),
    [
        ([True, False, True], ["true", "maybe", "1"], "'maybe' is not true or false"),
        ([True, False, True], [1, 2, 0], "2 is not 0 or 1"),
        ([1.5, 2.0, 3.0], ["1.5", "abc", "3"], "'abc' is not one"),
        (pd.to_datetime(["2026-01-01"] * 3), ["2026-01-01", "soon", "x"], "'soon' is not one"),
        (pd.Categorical(["a", "b", "c"]), ["a", "b", "c"], "supported target types"),
    ],
    ids=["bool-text", "bool-number", "float", "datetime", "category"],
)
def test_failed_casts_name_the_value(x_values, y_values, message):
    x = pd.DataFrame({"key": x_values, "m": [1.0, 2.0, 3.0]})
    y = pd.DataFrame({"key": y_values, "m": [1.0, 2.0, 3.0]})
    with pytest.raises(DiffError, match=message):
        diff_table(x, y, by=["key"])


def test_integer_cast_keeps_missing_values():
    x = pd.DataFrame({"k": [1, 2, 3], "n": [1, 2, 3]})
    y = pd.DataFrame({"k": [1, 2, 3], "n": [1.0, None, 3.0]})
    d = diff_table(x, y, by=["k"], measures="numeric+integer")
    assert status_of(d, "N", K=2) == "changed"
    assert d["status"].tolist().count("same") == 2


# --- values ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("vx", "vy", "status"),
    [
        (np.inf, np.inf, "same"),
        (-np.inf, -np.inf, "same"),
        (np.inf, -np.inf, "changed"),
        (np.inf, 5.0, "changed"),
        (np.nan, np.nan, "same"),
        (np.nan, 1.0, "changed"),
        (0.0, -0.0, "same"),
        (1.0, 1.0 + 1e-12, "changed"),
    ],
)
def test_value_status(vx, vy, status):
    d = diff_table(k_v([vx, 0.0, 0.0]), k_v([vy, 0.0, 0.0]))
    assert status_of(d, "V", K=1) == status


def test_tolerances_apply_to_finite_pairs_only():
    d = diff_table(k_v([np.inf, 1.0, 0.0]), k_v([-np.inf, 1.5, 0.0]), tolerance=1e308)
    assert status_of(d, "V", K=1) == "changed"
    assert status_of(d, "V", K=2) == "same"


def test_rel_diff_does_not_overflow():
    d = diff_table(k_v([1e308, 1.0, 0.0]), k_v([-1e308, 1.0, 0.0]))
    row = d.iloc[0]
    assert row["rel_diff"] == pytest.approx(2.0)
    assert np.isinf(row["abs_diff"])


def test_rel_diff_both_zero_is_zero():
    d = diff_table(k_v([0.0, 1.0, 2.0]), k_v([0.0, 1.0, 2.0]))
    assert d["rel_diff"].tolist() == [0.0, 0.0, 0.0]


def test_tolerance_and_rel_tol(benchmark, stored):
    d = quiet(benchmark, stored, by=["id", "region"], measures="numeric+integer", rel_tol=0.01)
    assert set(d["status"]) == {"same", "only_x", "only_y"}
    d = quiet(benchmark, stored, by=["id", "region"], tolerance=0.5)
    assert status_of(d, "AMOUNT", ID=2) == "same"


# --- duplicates and whole rows -----------------------------------------------


def test_repeated_keys_are_paired_with_a_warning():
    x = pd.DataFrame({"k": [1, 1], "v": [1.0, 2.0]})
    with pytest.warns(DiffWarning, match="repeated key"):
        d = diff_table(x, x)
    assert is_matching(d)


def test_duplicates_error():
    x = pd.DataFrame({"k": [1, 1], "v": [1.0, 2.0]})
    with pytest.raises(DiffError, match="repeated key"):
        diff_table(x, x, duplicates="error")


def test_rows_compared_as_wholes_count_copies(benchmark):
    y = pd.DataFrame({"ID": [3, 1, 2, 2], "REGION": ["east", "north", "south", "south"]})
    with warnings.catch_warnings():
        warnings.simplefilter("error", DiffWarning)  # copies are not a repeated-key warning
        with pytest.warns(DiffNote, match="as wholes"):
            d = diff_table(benchmark[["id", "region"]], y)
    assert d.attrs["shape"] == "rows"
    assert list(d.columns) == ["ID", "REGION", "key_seq", "row_id_x", "row_id_y", "status"]
    assert d["status"].value_counts().to_dict() == {
        "same": 3,
        "only_x": 1,
        "only_y": 1,
        "changed": 0,
    }


# --- summary -----------------------------------------------------------------


def test_summary(benchmark, stored):
    d = quiet(benchmark, stored, by=["id", "region"], measures="numeric+integer")
    s = diff_summary(d).set_index("metric")
    counts = ["n", "n_same", "n_changed", "n_only_x", "n_only_y"]
    assert s.loc["AMOUNT", counts].tolist() == [5, 2, 1, 1, 1]
    assert s.loc["AMOUNT", "max_abs_diff"] == pytest.approx(0.5)


def test_summary_of_rows_shape(benchmark):
    d = quiet(benchmark[["id"]], benchmark[["id"]].iloc[:3])
    s = diff_summary(d)
    assert s["metric"].tolist() == ["<rows>"]
    assert s[["n", "n_same", "n_only_x"]].iloc[0].tolist() == [4, 3, 1]


def test_warnings_point_at_the_caller():
    x = pd.DataFrame({"k": [1, 1], "v": [1.0, 2.0]})
    with pytest.warns(DiffWarning) as record:
        diff_table(x, x)
    assert record[0].filename == __file__
