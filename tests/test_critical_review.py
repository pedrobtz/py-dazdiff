"""Fixes from the critical review of 171acc1 (#40-#49)."""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dazdiff import (
    DiffError,
    DiffWarning,
    assert_table_equal,
    diff_explain,
    diff_report,
    diff_summary,
    diff_table,
    is_matching,
)
from dazdiff.cli import main

BIG = 2**53


def quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


# --- #40: aggregate compares integers exactly ---------------------------------------


def test_aggregate_compares_integer_measures_exactly():
    x = pd.DataFrame({"k": ["a", "a"], "n": np.array([BIG, BIG + 1], dtype="int64")})
    y = pd.DataFrame({"k": ["a", "a"], "n": np.array([BIG, BIG], dtype="int64")})
    d = diff_table(x, y, measures="numeric+integer", duplicates="aggregate", mode="equal")
    assert d["status"].astype(str).tolist() == ["changed"]
    assert d["values_x"].iloc[0] == (BIG, BIG + 1)
    assert d["values_y"].iloc[0] == (BIG, BIG)
    assert not is_matching(d)
    with pytest.raises(AssertionError):
        assert_table_equal(y, x, measures="numeric+integer", duplicates="aggregate")


def test_aggregate_integer_multisets_with_missing_values():
    x = pd.DataFrame({"k": ["a", "a", "a"], "n": pd.Series([3, None, 1], dtype="Int64")})
    y = pd.DataFrame({"k": ["a", "a", "a"], "n": pd.Series([None, 1, 3], dtype="Int64")})
    d = diff_table(x, y, measures="numeric+integer", duplicates="aggregate", mode="equal")
    assert d["status"].astype(str).tolist() == ["same"]
    assert d["values_x"].iloc[0][:2] == (1, 3)
    assert pd.isna(d["values_x"].iloc[0][2])


# --- #41: a result is one comparison ---------------------------------------------------


@pytest.fixture
def joined():
    d1 = diff_table(pd.DataFrame({"id": ["a", "b"], "v": [1.0, 2.0]}),
                    pd.DataFrame({"id": ["a", "b"], "v": [1.0, 2.0]}))  # fmt: skip
    d2 = diff_table(pd.DataFrame({"id": ["c", "d"], "v": [1.0, 2.0]}),
                    pd.DataFrame({"id": ["c", "d"], "v": [1.0, 5.0]}))  # fmt: skip
    return pd.concat([d1, d2], ignore_index=True)


@pytest.mark.parametrize("fn", [diff_explain, diff_report])
def test_concatenated_results_are_refused(joined, fn):
    with pytest.raises(DiffError, match="more than one comparison"):
        fn(joined)


def test_a_result_concatenated_with_itself_is_refused():
    d = quiet(diff_table, pd.DataFrame({"id": ["a"]}), pd.DataFrame({"id": ["a"]}))
    with pytest.raises(DiffError, match="more than one comparison"):
        quiet(diff_report, pd.concat([d, d], ignore_index=True))


def test_aggregated_results_concatenated_are_refused():
    x = pd.DataFrame({"id": ["a", "a"], "v": [1.0, 2.0]})
    g = diff_table(x, x.copy(), duplicates="aggregate")
    with pytest.raises(DiffError, match="more than one comparison"):
        diff_explain(pd.concat([g, g], ignore_index=True))


def test_a_filtered_result_is_still_one_comparison(joined):
    d = diff_table(pd.DataFrame({"id": ["c", "d"], "v": [1.0, 2.0]}),
                   pd.DataFrame({"id": ["c", "d"], "v": [1.0, 5.0]}))  # fmt: skip
    assert len(diff_explain(d[d["status"] == "changed"], "id")) == 1


# --- #42: object columns are cast ---------------------------------------------------------


def test_object_text_keys_meet_object_integer_keys():
    x = pd.DataFrame({"id": pd.Series(["1", "2"], dtype=object), "v": [1.0, 2.0]})
    y = pd.DataFrame({"id": pd.Series([1, 2], dtype=object), "v": [1.0, 2.0]})
    d = diff_table(x, y)
    assert is_matching(d)
    assert d["ID"].tolist() == ["1", "2"]


def test_object_columns_of_text_on_both_sides_are_left_alone():
    x = pd.DataFrame({"id": pd.Series(["a", None], dtype=object), "v": [1.0, 2.0]})
    assert is_matching(diff_table(x, x.copy()))


# --- #43: by and measures do not overlap ---------------------------------------------------


def test_a_measure_that_by_also_names_is_refused():
    x = pd.DataFrame({"k": [1, 2], "v": [1.0, 2.0]})
    with pytest.raises(DiffError, match="`by` and `measures` both name: v"):
        diff_table(x, x.copy(), by=["k", "v"], measures=["v"])


def test_a_measure_keyword_still_yields_to_by():
    x = pd.DataFrame({"k": [1.0, 2.0], "v": [1.0, 2.0]})
    d = diff_table(x, x.copy(), by="k")
    assert d.attrs["by"] == ["K"] and d.attrs["measures"] == ["V"]


# --- #44: every public function raises DiffError --------------------------------------------


@pytest.fixture
def d():
    return diff_table(pd.DataFrame({"id": [1], "v": [1.0]}), pd.DataFrame({"id": [1], "v": [2.0]}))


@pytest.mark.parametrize(
    ("x", "y"), [({"a": [1]}, pd.DataFrame({"a": [1]})), (pd.DataFrame({"a": [1]}), [1, 2])]
)
def test_diff_table_refuses_what_is_not_a_dataframe(x, y):
    with pytest.raises(DiffError, match="must be a pandas DataFrame"):
        diff_table(x, y)


@pytest.mark.parametrize("bad", [None, "1", True, np.inf, np.nan, -1, [1]])
@pytest.mark.parametrize("arg", ["tolerance", "rel_tol"])
def test_diff_table_checks_its_thresholds(arg, bad):
    x = pd.DataFrame({"id": [1], "v": [1.0]})
    with pytest.raises(DiffError, match=f"`{arg}` must be a single finite number >= 0"):
        diff_table(x, x.copy(), **{arg: bad})


@pytest.mark.parametrize("bad", [None, 1, ["v", 2], {"v"}])
def test_diff_table_checks_measures(bad):
    x = pd.DataFrame({"id": [1], "v": [1.0]})
    with pytest.raises(DiffError, match="`measures` must be"):
        diff_table(x, x.copy(), measures=bad)


@pytest.mark.parametrize("bad", [1, [1], ["id", None], []])
def test_diff_table_checks_by(bad):
    x = pd.DataFrame({"id": [1], "v": [1.0]})
    with pytest.raises(DiffError, match="`by` must be"):
        diff_table(x, x.copy(), by=bad)


@pytest.mark.parametrize("fn", [diff_summary, is_matching, diff_explain, diff_report])
@pytest.mark.parametrize("bad", [pd.DataFrame({"a": [1]}), [1, 2], None])
def test_every_reader_refuses_what_is_not_a_result(fn, bad):
    with pytest.raises(DiffError, match="must be a result of diff_table"):
        fn(bad)


def test_mixed_utc_offsets_are_a_diff_error():
    x = pd.DataFrame({"t": pd.to_datetime(["2024-01-01", "2024-01-02"]).tz_localize("UTC"),
                      "v": [1.0, 2.0]})  # fmt: skip
    y = pd.DataFrame({"t": ["2024-01-01T00:00:00+01:00", "2024-01-02T00:00:00+02:00"],
                      "v": [1.0, 2.0]})  # fmt: skip
    with pytest.raises(DiffError, match="Cannot cast `t` to a date-time"):
        diff_table(x, y)


# --- #45: assert_table_equal() keeps diff_table()'s warnings when it fails -------------------


def test_assert_table_equal_passes_warnings_on_when_it_fails():
    expected = pd.DataFrame({"id": [1, 2], "v": [1.0, 2.0]})
    actual = pd.DataFrame({"id": [8, 9], "v": [1.0, 2.0]})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(AssertionError):
            assert_table_equal(actual, expected, exclude=["runid"])
    assert any(w.category is DiffWarning and "runid" in str(w.message) for w in caught)


def test_assert_table_equal_passes_warnings_on_on_a_mismatch():
    expected = pd.DataFrame({"id": [1, 2], "v": [1.0, 2.0]})
    actual = expected.assign(v=[1.0, 3.0], loaded="t")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(AssertionError):
            assert_table_equal(actual, expected)
    assert any("loaded" in str(w.message) for w in caught)


# --- #46, #47: the CLI reads integers and codes as they are written ------------------------


def write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


def test_cli_an_empty_cell_in_the_benchmark_integer_key(tmp_path, capsys):
    x = write(tmp_path, "x.csv", "id,g,v\n1,a,1.0\n2,b,2.0\n,c,3.0\n")
    y = write(tmp_path, "y.csv", "id,g,v\n1,a,1.0\n2,b,2.0\n3,c,3.0\n")
    assert main([x, y, "--only-diffs"]) == 1
    out = capsys.readouterr().out
    assert "only_x" in out and "only_y" in out and "changed" not in out


def test_cli_equal_mode_with_an_empty_integer_cell(tmp_path, capsys):
    a = write(tmp_path, "a.csv", "id,n,v\n1,10,1.5\n2,20,2.5\n")
    b = write(tmp_path, "b.csv", "id,n,v\n1,10,1.5\n2,,2.5\n")
    assert main([a, b, "--mode", "equal", "-q"]) == 1
    assert main([a, a, "--mode", "equal", "-q"]) == 0


def test_cli_keeps_leading_zeros(tmp_path, capsys):
    x = write(tmp_path, "x.csv", "code,v\n007,1.5\n010,2.5\n")
    y = write(tmp_path, "y.csv", "code,v\n7,1.5\n10,2.5\n")
    assert main([x, y, "-q"]) == 2  # no key in common: "007" is not "7"
    assert main([x, x, "-q"]) == 0
    z = write(tmp_path, "z.csv", "code,v\n007,1.5\n010,2.5\n")
    assert main([x, z, "--mode", "equal", "-q"]) == 0


def test_cli_a_lone_zero_is_still_a_number(tmp_path):
    x = write(tmp_path, "x.csv", "id,v\n0,1.5\n1,2.5\n")
    y = write(tmp_path, "y.csv", "id,v\n0.0,1.5\n1.0,2.5\n")
    assert main([x, y, "-q"]) == 0


# --- #49: the guards hand mutation found untested ---------------------------------------------


def test_uint64_measures_above_2_63_are_compared_exactly():
    v = np.array([2**63 + 1], dtype="uint64")
    x = pd.DataFrame({"id": [1], "v": v})
    y = pd.DataFrame({"id": [1], "v": v + np.uint64(2)})
    d = diff_table(x, y, by="id", measures="numeric+integer")
    assert d["status"].astype(str).tolist() == ["changed"]
    assert d["diff"].tolist() == [2.0]


def test_cast_must_be_a_bool():
    x = pd.DataFrame({"id": [1], "v": [1.0]})
    with pytest.raises(DiffError, match="`cast` must be None, True or False"):
        diff_table(x, x.copy(), cast="yes")


def test_is_matching_on_groups_refuses_a_relative_tolerance():
    x = pd.DataFrame({"id": [1, 1], "v": [1.0, 2.0]})
    g = diff_table(x, x.copy(), duplicates="aggregate")
    with pytest.raises(DiffError, match="exactly"):
        is_matching(g, rel_tol=0.1)


def test_aggregate_against_an_empty_table():
    x = pd.DataFrame({"id": ["a", "a"], "v": [1.0, 2.0]})
    d = diff_table(x, x.iloc[:0], duplicates="aggregate")
    assert d["status"].astype(str).tolist() == ["only_x"]
    assert d["n_rows_x"].tolist() == [2]


def test_the_text_nan_is_not_cast_to_an_integer():
    x = pd.DataFrame({"id": [1, 2], "v": [1.0, 2.0]})
    y = pd.DataFrame({"id": ["1", "nan"], "v": [1.0, 2.0]})
    with pytest.raises(DiffError, match="'nan' is not one"):
        diff_table(x, y)


def test_the_readme_examples_exist():
    # #48: paths built from this file, so the suite runs from any directory.
    assert (Path(__file__).parent.parent / "examples" / "benchmark.csv").exists()
