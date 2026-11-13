"""Stage 3: parity with the R package's main as of 2026-11-13 (design §4.1a, §4.3,
§4.4, §4.8, §4a)."""

import warnings

import numpy as np
import pandas as pd
import pytest

from dazdiff import DiffError, DiffNote, DiffWarning, diff_table, is_matching
from dazdiff.core import _cast_to


def run(x, y, **kwargs):
    """diff_table() with every warning recorded, returned beside the result."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        d = diff_table(x, y, **kwargs)
    return d, caught


def messages(caught, category):
    return [str(w.message) for w in caught if w.category is category]


# --- roles before casting ----------------------------------------------------


def test_an_ignored_column_is_not_cast():
    x = pd.DataFrame({"id": [1, 2], "amt": [1.0, 2.0], "note": [1, 2]})
    y = pd.DataFrame({"id": [1, 2], "amt": [1.0, 2.0], "note": ["a", "b"]})
    d, _caught = run(x, y, by="id", measures=["amt"])
    assert is_matching(d)
    assert d.attrs["ignored"] == ["note"]


def test_an_ignored_column_is_not_truncated():
    x = pd.DataFrame({"id": [1, 2], "amt": [1.0, 2.0], "n": [1, 2]})
    y = pd.DataFrame({"id": [1, 2], "amt": [1.0, 2.0], "n": [1.5, 2.5]})
    d, caught = run(x, y, by="id", measures=["amt"])
    assert d.attrs["truncated"] == {}
    assert not any("Truncated" in m for m in messages(caught, DiffNote))


# --- cast= -------------------------------------------------------------------


def test_cast_false_requires_matching_types():
    x = pd.DataFrame({"id": [1, 2], "amt": [1.0, 2.0]})
    y = pd.DataFrame({"id": ["1", "2"], "amt": [1.0, 2.0]})
    with pytest.raises(DiffError, match=r"cast=False.*id \(int64 vs"):
        diff_table(x, y, cast=False)


def test_cast_false_passes_when_types_agree_and_checks_compared_columns_only():
    x = pd.DataFrame({"id": [1, 2], "amt": [1.0, 2.0], "note": [1, 2]})
    y = pd.DataFrame({"id": [1, 2], "amt": [1.0, 2.0], "note": ["a", "b"]})
    d, _ = run(x, y, by="id", cast=False)
    assert is_matching(d)


def test_cast_true_in_equal_mode_is_an_error():
    x = pd.DataFrame({"id": [1], "amt": [1.0]})
    with pytest.raises(DiffError, match="never casts"):
        diff_table(x, x, mode="equal", cast=True)


def test_cast_false_in_equal_mode_is_allowed():
    x = pd.DataFrame({"id": [1], "amt": [1.0]})
    assert is_matching(diff_table(x, x.copy(), mode="equal", cast=False))


# --- exclude= ----------------------------------------------------------------


def test_exclude_removes_a_column_from_both_tables(benchmark, stored):
    d, caught = run(benchmark, stored, exclude=["loaded_at", "units"])
    assert d.attrs["excluded"] == ["units", "Units", "loaded_at"]
    assert "UNITS" not in set(d["metric"])
    # Not announced as dropped.
    assert not any("loaded_at" in m for m in messages(caught, DiffNote))


def test_exclude_in_either_spelling_and_a_column_in_one_table_only():
    x = pd.DataFrame({"id": [1, 2], "amt": [1.0, 2.0]})
    y = pd.DataFrame({"id": [1, 2], "amt": [1.0, 2.0], "Load Stamp": ["a", "b"]})
    d, _caught = run(x, y, mode="equal", exclude="LOAD_STAMP")
    assert is_matching(d)
    assert d.attrs["excluded"] == ["Load Stamp"]


def test_exclude_settles_a_name_clash():
    x = pd.DataFrame({"id": [1], "amt": [1.0], "a b": [1], "a_b": [2]})
    d, _ = run(x, x.copy(), exclude="A_B")
    assert is_matching(d)


def test_exclude_of_an_unknown_name_is_a_warning():
    x = pd.DataFrame({"id": [1], "amt": [1.0]})
    d, caught = run(x, x.copy(), exclude=["run_id"])
    assert is_matching(d)
    assert any("run_id" in m for m in messages(caught, DiffWarning))


@pytest.mark.parametrize("kwargs", [{"by": "id"}, {"measures": ["amt"]}])
def test_exclude_named_by_by_or_measures_is_an_error(kwargs):
    x = pd.DataFrame({"id": [1], "amt": [1.0]})
    with pytest.raises(DiffError, match="`exclude` names column"):
        diff_table(x, x.copy(), exclude=["Id", "amt"], **kwargs)


def test_exclude_with_a_measure_keyword_is_fine():
    x = pd.DataFrame({"id": [1], "amt": [1.0], "x": [2.0]})
    d, _ = run(x, x.copy(), measures="numeric", exclude="x")
    assert list(d["metric"]) == ["AMT"]


# --- number to text -----------------------------------------------------------


def test_numbers_cast_to_text_follow_the_benchmark_decimals():
    x = pd.DataFrame({"k": ["1.0", "2.0", "2.5"], "v": [1.0, 2.0, 3.0]})
    y = pd.DataFrame({"k": [1.0, 2.0, 2.5], "v": [1.0, 2.0, 3.0]})
    d, caught = run(x, y)
    assert is_matching(d)
    notes = messages(caught, DiffNote)
    assert any("1 decimal place" in m and "'1.0'" in m for m in notes)


def test_integers_cast_to_text_follow_the_benchmark_decimals():
    x = pd.DataFrame({"k": ["1.00", "2.00"], "v": [1.0, 2.0]})
    y = pd.DataFrame({"k": [1, 2], "v": [1.0, 2.0]})
    d, _ = run(x, y)
    assert is_matching(d)


def test_a_value_needing_more_decimals_keeps_its_digits():
    # 1.25 at one decimal would read "1.2" or "1.3" and join a different row.
    x = pd.DataFrame({"k": ["1.2", "1.3"], "v": [1.0, 2.0]})
    y = pd.DataFrame({"k": [1.2, 1.25], "v": [1.0, 2.0]})
    d, caught = run(x, y)
    assert sorted(d.loc[d["status"] == "only_y", "K"]) == ["1.25"]
    assert any("kept their own digits" in m for m in messages(caught, DiffNote))


def test_mixed_decimals_warn_about_a_format_skew():
    x = pd.DataFrame({"k": ["1.0", "2.50", "3"], "v": [1.0, 2.0, 3.0]})
    y = pd.DataFrame({"k": [1.0, 2.5, 3.0], "v": [1.0, 2.0, 3.0]})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        d = diff_table(x, y)
    warned = messages(caught, DiffWarning)
    assert any("'1.0' vs '1'" in m and "1, 2 decimal places" in m for m in warned)
    assert d.loc[d["K"] == "3", "status"].tolist() == ["same"]


def test_negative_zero_is_written_as_zero():
    x = pd.DataFrame({"k": ["0.0", "1.0"], "v": [1.0, 2.0]})
    y = pd.DataFrame({"k": [-0.0, 1.0], "v": [1.0, 2.0]})
    assert is_matching(run(x, y)[0])


# --- date-times and empty columns ---------------------------------------------


def test_a_datetime_cast_to_text_has_one_format():
    x = pd.DataFrame(
        {
            "t": ["2024-01-01 00:00:00", "2024-01-01 10:00:00.5", "2024-01-02 00:00:00.000001"],
            "v": [1.0, 2.0, 3.0],
        }
    )
    stamps = ["2024-01-01", "2024-01-01 10:00:00.5", "2024-01-02 00:00:00.000001"]
    y = pd.DataFrame({"t": pd.to_datetime(stamps, format="ISO8601"), "v": [1.0, 2.0, 3.0]})
    assert is_matching(run(x, y)[0])


@pytest.mark.parametrize(
    "target",
    [
        pd.Series([1, 2]),
        pd.Series([1.0, 2.0]),
        pd.Series([True, False]),
        pd.Series(["a", "b"]),
        pd.Series(pd.to_datetime(["2024-01-01", "2024-01-02"])),
    ],
    ids=["int", "float", "bool", "text", "datetime"],
)
@pytest.mark.parametrize(
    "empty",
    [[np.nan, np.nan], [None, None], pd.Series([pd.NA, pd.NA], dtype="boolean")],
    ids=["float", "object", "boolean"],
)
def test_an_all_missing_column_casts_to_any_type(target, empty):
    out = _cast_to(pd.Series(empty), target, "c", True, {}, [])
    assert out.isna().all()
    assert len(out) == 2
    if target.dtype.kind not in "iub":  # int and bool become their nullable types
        assert out.dtype == target.dtype


# --- signed zero keys -----------------------------------------------------------


def test_negative_and_positive_zero_are_one_key():
    x = pd.DataFrame({"k": [0.0, 1.0], "v": [1.0, 2.0]})
    y = pd.DataFrame({"k": [-0.0, 1.0], "v": [1.0, 2.0]})
    assert is_matching(diff_table(x, y, by="k"))
    both = pd.DataFrame({"k": [0.0, -0.0], "v": [1.0, 2.0]})
    with pytest.raises(DiffError, match="repeated key"):
        diff_table(both, both.copy(), by="k", duplicates="error")


# --- disjoint keys --------------------------------------------------------------


def test_the_disjoint_error_names_the_columns_and_suggests_exclude():
    x = pd.DataFrame({"id": [1, 2], "run": ["r1", "r1"], "v": [1.0, 2.0]})
    y = pd.DataFrame({"id": [1, 2], "run": ["r2", "r2"], "v": [1.0, 2.0]})
    with pytest.raises(DiffError) as e:
        diff_table(x, y)
    text = str(e.value)
    assert "share no value between the tables: run (x: 'r1'; y: 'r2')" in text
    assert "exclude=['run']" in text


def test_the_disjoint_error_shows_three_examples():
    x = pd.DataFrame({"k": ["1.0", "2.0", "3.0", "4.0"], "v": [1.0] * 4})
    y = pd.DataFrame({"k": ["1", "2", "3", "4"], "v": [1.0] * 4})
    with pytest.raises(DiffError) as e:
        diff_table(x, y)
    assert "k (x: '1.0', '2.0', '3.0', ...; y: '1', '2', '3', ...)" in str(e.value)
    assert "exclude" not in str(e.value)  # no other key would be left


def test_the_disjoint_error_with_aggregate():
    x = pd.DataFrame({"id": [1, 1], "run": ["r1", "r1"], "v": [1.0, 2.0]})
    y = pd.DataFrame({"id": [1, 1], "run": ["r2", "r2"], "v": [1.0, 2.0]})
    with pytest.raises(DiffError, match="run \\(x: 'r1'; y: 'r2'\\)"):
        diff_table(x, y, duplicates="aggregate")


# --- attrs ----------------------------------------------------------------------


def test_attrs_record_what_was_done(benchmark, stored):
    d, _ = run(benchmark, stored, by=["id", "region"], tolerance=0.5, rel_tol=0.1)
    assert d.attrs["tolerance"] == {"abs": 0.5, "rel": 0.1}
    assert d.attrs["dropped"] == ["loaded_at"]
    assert d.attrs["excluded"] == []
    assert d.attrs["ignored"] == ["units"]
    assert d.attrs["name_map"]["ID"] == {"x": "id", "y": "ID"}
    assert d.attrs["name_map"]["LOADED_AT"] == {"x": None, "y": "loaded_at"}


# --- is_matching ------------------------------------------------------------------


@pytest.fixture
def near():
    x = pd.DataFrame({"id": [1, 2, 3], "amount": [100.0, 200.0, 300.0]})
    y = pd.DataFrame({"id": [1, 2, 3], "amount": [100.0, 200.5, 300.0]})
    return diff_table(x, y)


def test_is_matching_rejudges_at_other_thresholds(near):
    assert not is_matching(near)
    assert is_matching(near, tolerance=1)
    assert not is_matching(near, tolerance=0.1)
    assert is_matching(near, rel_tol=0.01)
    assert not is_matching(near, rel_tol=0.001)


def test_is_matching_can_be_stricter_than_the_comparison():
    x = pd.DataFrame({"id": [1], "amount": [100.0]})
    y = pd.DataFrame({"id": [1], "amount": [100.5]})
    d = diff_table(x, y, tolerance=1)
    assert is_matching(d)
    assert not is_matching(d, tolerance=0)


def test_is_matching_keeps_missing_and_infinite_pairs_and_membership():
    x = pd.DataFrame({"id": [1, 2, 3], "a": [np.nan, np.inf, 1.0]})
    y = pd.DataFrame({"id": [1, 2, 4], "a": [np.nan, np.inf, 1.0]})
    d = diff_table(x, y)
    assert not is_matching(d, tolerance=1e9)
    assert is_matching(d[d["status"] == "same"], tolerance=0)


def test_is_matching_on_groups_and_rows():
    x = pd.DataFrame({"id": [1, 1], "a": [1.0, 2.0]})
    g = diff_table(x, x.copy(), duplicates="aggregate")
    assert is_matching(g, tolerance=0)
    with pytest.raises(DiffError, match="exactly"):
        is_matching(g, tolerance=1)
    r, _ = run(x[["id"]], x[["id"]].copy())
    assert is_matching(r, tolerance=1)


@pytest.mark.parametrize("bad", [-1, np.inf, np.nan, "1", [1]])
def test_is_matching_checks_its_thresholds(near, bad):
    with pytest.raises(DiffError, match="single finite number"):
        is_matching(near, tolerance=bad)


def test_two_empty_tables_match():
    x = pd.DataFrame({"id": pd.Series([], dtype="int64"), "a": pd.Series([], dtype=float)})
    assert is_matching(diff_table(x, x.copy()))


# --- cases ported from the R package's tests ---------------------------------------


def test_an_empty_exclude_excludes_nothing(benchmark, stored):
    a, _ = run(benchmark, stored, by=["id", "region"])
    b, _ = run(benchmark, stored, by=["id", "region"], exclude=[])
    pd.testing.assert_frame_equal(a, b)
    assert b.attrs["excluded"] == []


def test_an_excluded_key_is_not_aligned_on(benchmark, stored):
    d, _ = run(benchmark, stored, exclude=["region", "units"])
    assert d.attrs["by"] == ["ID"]
    assert set(d.loc[d["ID"] == 4, "status"]) == {"only_x"}


def test_an_excluded_column_is_never_cast():
    x = pd.DataFrame({"id": [1], "v": [1.0], "n": [1]})
    y = pd.DataFrame({"id": [1], "v": [1.0], "n": ["not a number"]})
    assert is_matching(run(x, y, exclude="n")[0])


def test_an_ignored_column_of_an_unsupported_type_is_not_refused():
    x = pd.DataFrame({"id": [1], "v": [1.0], "c": pd.Categorical(["a"])})
    y = pd.DataFrame({"id": [1], "v": [1.0], "c": ["a"]})
    assert is_matching(run(x, y, by="id")[0])
    with pytest.raises(DiffError, match="supported target types"):
        diff_table(x, y)


def test_a_datetime_is_written_in_its_own_zone():
    utc = pd.to_datetime(["2024-06-01 11:30:00"]).tz_localize("UTC")  # 12:30 in Lisbon
    y = pd.DataFrame({"t": utc.tz_convert("Europe/Lisbon")})
    x = pd.DataFrame({"t": ["2024-06-01 12:30:00"]})
    assert is_matching(run(x, y)[0])


def test_a_large_double_is_written_in_full_as_text():
    x = pd.DataFrame({"k": ["100000000000000000000"], "v": [1.0]})
    y = pd.DataFrame({"k": [1e20], "v": [1.0]})
    assert is_matching(run(x, y)[0])


def test_no_threshold_matches_a_pair_that_is_not_finite():
    x = pd.DataFrame({"id": [1, 2], "a": [1.0, np.nan]})
    y = pd.DataFrame({"id": [1, 2], "a": [np.inf, 1.0]})
    d = diff_table(x, y)
    assert not is_matching(d, tolerance=1e308, rel_tol=1e308)
