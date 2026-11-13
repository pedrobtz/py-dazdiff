"""diff_report(), ported from the R package's test-report.R (design §5).

The ``r_*`` snapshots are the R package's own (``_snaps/report.md`` at
``2bb27b2``): the same inputs must draw the same text. The ``py_*`` snapshots are
this package's; regenerate them with ``DAZDIFF_UPDATE_SNAPSHOTS=1`` and read the
diff.
"""

import os
import warnings
from pathlib import Path

import pandas as pd
import pytest

from dazdiff import DiffError, diff_report, diff_table
from dazdiff.report import _count, _num, _pct

SNAPSHOTS = Path(__file__).parent / "snapshots"


def quiet(x, y, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return diff_table(x, y, **kwargs)


def expect_snapshot(text: str, name: str) -> None:
    path = SNAPSHOTS / f"{name}.txt"
    if name.startswith("py_") and os.environ.get("DAZDIFF_UPDATE_SNAPSHOTS"):
        path.write_text(text + "\n", encoding="utf-8")
    assert text + "\n" == path.read_text(encoding="utf-8")


def report_x():
    return pd.DataFrame({"id": ["a", "b", "c"], "value": [1.0, 2.0, 3.0]})


def report_y():
    return pd.DataFrame({"id": ["a", "b", "d"], "value": [1.0, 2.5, 4.0]})


# --- the R package's drawings ---------------------------------------------------


def test_the_drawing():
    expect_snapshot(
        diff_report(diff_table(report_x(), report_y())).format(unicode=False), "r_mismatch"
    )
    expect_snapshot(
        diff_report(diff_table(report_x(), report_x())).format(unicode=False), "r_match"
    )


def test_the_drawing_with_names_dropped_and_ignored_columns():
    bench = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "region": ["n", "s", "n", "s", "e", "w"],
            "amount": [10.0, 20, 30, 40, 50, 60],
            "rate": [1.0, 1, 1, 2, 2, 2],
            "note": "z",
        }
    )
    tested = pd.DataFrame(
        {
            "ID": ["1", "2", "3", "4", "5", "7", "8"],
            "Region": ["n", "s", "n", "s", "e", "w", "w"],
            "Amount": [10.0, 21, 30, 41, 50, 70, 80],
            "rate": [1.0, 1, 1, 2.5, 2, 2, 2],
            "note": "z",
            "loaded": "t",
        }
    )
    d = quiet(bench, tested, by=["id", "region"], measures=["amount", "rate"])
    r = diff_report(d, x_name="bench", y_name="tested")
    expect_snapshot(r.format(unicode=False), "r_bench_tested")


def test_the_drawing_in_box_drawing_characters():
    r = diff_report(diff_table(report_x(), report_y()))
    expect_snapshot(r.format(unicode=True), "r_mismatch_unicode")
    assert str(r) == r.format(unicode=True)


def test_the_drawing_of_paired_rows_and_a_truncating_cast():
    x = pd.DataFrame({"id": ["a", "a", "b"], "n": [1, 1, 3], "v": [1.0, 2, 3]})
    y = pd.DataFrame({"id": ["a", "a", "b"], "n": [1.7, 1, 3], "v": [1.0, 2, 4]})
    d = quiet(x, y)
    expect_snapshot(diff_report(d).format(unicode=False), "r_paired_truncated")


def test_the_drawing_of_a_change_in_one_measure_of_several():
    x = pd.DataFrame({"id": [1, 2, 3], "a": [1.0, 2, 3], "b": [4.0, 5, 6], "c": [7.0, 8, 9]})
    y = pd.DataFrame({"id": [1, 2, 3], "a": [1.0, 2, 3], "b": [4.0, 5, 60], "c": [7.0, 8, 9]})
    expect_snapshot(diff_report(diff_table(x, y)).format(unicode=False), "r_one_measure_of_several")


def test_the_drawing_of_an_exact_match_that_dropped_a_column():
    x = pd.DataFrame({"id": [1, 2], "v": [1.0, 2.0]})
    y = pd.DataFrame({"id": [1, 2], "v": [1.0, 2.0], "loaded_at": "t"})
    r = diff_report(quiet(x, y))
    assert r.equal and r.dropped == ["loaded_at"]
    expect_snapshot(r.format(unicode=False), "r_match_dropped")


def test_the_drawing_of_an_exact_match_with_an_excluded_column():
    x = pd.DataFrame({"id": [1, 2], "v": [1.0, 2.0], "note": ["p", "q"]})
    y = pd.DataFrame({"id": [1, 2], "v": [1.0, 2.0], "note": ["r", "s"]})
    r = diff_report(diff_table(x, y, mode="equal", exclude="note"))
    assert r.equal and r.excluded == ["note"]
    expect_snapshot(r.format(unicode=False), "r_match_excluded")


def test_the_drawing_of_a_result_filtered_down_to_y_own_rows():
    d = diff_table(report_x(), report_y())
    r = diff_report(d[d["status"] == "only_y"])
    assert (r.rows["n_x"], r.rows["n_y"], r.rows["n_only_y"]) == (0, 1, 1)
    expect_snapshot(r.format(unicode=False), "r_filtered_only_y")


# --- this package's drawings: the README example and the other shapes ------------


def test_the_readme_example(benchmark, stored):
    d = quiet(benchmark, stored, by=["id", "region"], measures="numeric+integer")
    r = diff_report(d, "benchmark", "stored")
    expect_snapshot(r.format(unicode=False), "py_readme_ascii")
    expect_snapshot(r.format(unicode=True), "py_readme_unicode")


def test_the_groups_shape():
    x = pd.DataFrame({"id": ["a", "a", "b"], "v": [1.0, 2, 3]})
    y = pd.DataFrame({"id": ["a", "b", "b"], "v": [1.0, 3, 3]})
    r = diff_report(diff_table(x, y, duplicates="aggregate"))
    assert r.unit == "keys"
    assert (r.rows["n_x"], r.rows["n_changed"]) == (2, 2)
    expect_snapshot(r.format(unicode=False), "py_groups")


def test_the_rows_shape():
    rows = quiet(pd.DataFrame({"k": ["a", "b"]}), pd.DataFrame({"k": ["b", "c", "a"]}))
    r = diff_report(rows)
    assert r.measures == []
    assert (r.rows["n_same"], r.rows["n_only_y"]) == (2, 1)
    expect_snapshot(r.format(unicode=False), "py_rows")


def test_tolerance_and_row_number():
    x = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    y = pd.DataFrame({"a": [1.0, 2.05, 3.5]})
    r = diff_report(diff_table(x, y, row_key="position", tolerance=0.1))
    assert r.identity == ["row_number"]
    expect_snapshot(r.format(unicode=False), "py_tolerance_row_number")


# --- the numbers ------------------------------------------------------------------


def test_diff_report_counts_rows_by_alignment_a_row_changed_if_any_measure_is():
    x = pd.DataFrame({"id": [1, 2, 3, 4], "a": [1.0, 2, 3, 4], "b": [1.0, 2, 3, 4]})
    y = pd.DataFrame({"id": [1, 2, 3, 5], "a": [1.0, 9, 3, 5], "b": [1.0, 9, 7, 5]})
    r = diff_report(diff_table(x, y))
    assert r.rows == {
        "n_x": 4,
        "n_y": 4,
        "n_only_x": 1,
        "n_same": 1,
        "n_changed": 2,
        "n_only_y": 1,
    }
    assert r.by_measure["measure"].tolist() == ["a", "b"]
    assert r.by_measure["match_rate"].tolist() == [2 / 3, 1 / 3]
    assert not r.equal


def test_the_identity_is_named_in_the_callers_spelling():
    x = pd.DataFrame({"Id": [1, 2], "Sub Key": ["u", "v"], "v": [1.0, 2]})
    y = pd.DataFrame({"id": [1, 2], "sub_key": ["u", "v"], "v": [1.0, 3]})
    assert diff_report(diff_table(x, y)).identity == ["Id", "Sub Key"]


def test_only_in_one_table_keys_are_sampled_bounded_by_sample_k():
    x = pd.DataFrame({"id": range(1, 11), "v": 1.0})
    y = pd.DataFrame({"id": [1], "v": [1.0]})
    r = diff_report(diff_table(x, y), sample_k=3)
    assert r.only_x["id"].tolist() == [2, 3, 4]
    assert len(r.only_y) == 0
    assert "(...and 6 more)" in r.format(unicode=False)
    text = diff_report(diff_table(x, y), sample_k=0).format(unicode=False)
    assert "Rows only in x" not in text


def test_diff_report_reads_a_filtered_or_resorted_result():
    d = diff_table(report_x(), report_y())
    r = diff_report(d.sort_values("value_x", ascending=False))
    assert r.rows == diff_report(d).rows


def test_two_empty_tables_report_as_equal_without_warnings():
    e = pd.DataFrame({"id": pd.Series([], dtype=str), "v": pd.Series([], dtype=float)})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = diff_report(diff_table(e, e.copy()))
    assert r.equal
    assert "both empty" in r.format(unicode=False)


def test_counts_past_2_31_and_shares_are_formatted():
    assert _count(3e9) == "3,000,000,000"
    assert [_pct(p) for p in (0, 1e-9, 1 - 1e-9, 1)] == ["0.00%", "0.01%", "99.99%", "100.00%"]


@pytest.mark.parametrize(
    ("v", "digits", "text"),
    [
        (0.0476190476, 4, "0.04762"),
        (0.5, 4, "0.5"),
        (1.0, 4, "1"),
        (54.0, 4, "54"),
        (123456.0, 4, "123456"),
        (1e-10, 4, "1e-10"),
        (1.5e10, 4, "1.5e+10"),
        (-2.25, 7, "-2.25"),
        (float("inf"), 4, "Inf"),
        # format(v, digits = 4) and digits = 7 in R:
        (123456.7, 4, "123457"),
        (123456.7, 7, "123456.7"),
        (99999.99, 4, "1e+05"),
        (99999.99, 7, "99999.99"),
        (1e15, 4, "1e+15"),
        (123456789012.0, 4, "1.235e+11"),
        (123456789012.0, 7, "123456789012"),
        (0.0001234567890, 4, "0.0001235"),
        (2.5e-5, 4, "2.5e-05"),
    ],
)
def test_numbers_are_written_as_r_writes_them(v, digits, text):
    assert _num(v, digits) == text


def test_diff_report_validates_its_arguments():
    d = diff_table(report_x(), report_y())
    with pytest.raises(DiffError, match="result of diff_table"):
        diff_report(pd.DataFrame({"a": [1]}))
    for bad in (-1, 1.5, True):
        with pytest.raises(DiffError, match="sample_k"):
            diff_report(d, sample_k=bad)
    with pytest.raises(DiffError, match="x_name"):
        diff_report(d, x_name="")
    with pytest.raises(DiffError, match="unicode"):
        diff_report(d).format(unicode=None)
