"""Gaps found by the Stage 9 test review (#36). Each test fails against a mutant of
the source that the rest of the suite let through."""

import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dazdiff import (
    DiffError,
    DiffNote,
    DiffWarning,
    assert_table_equal,
    diff_summary,
    diff_table,
    is_matching,
)
from dazdiff.cli import main


def recorded(fn, *args, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = fn(*args, **kwargs)
    return out, [(w.category, str(w.message)) for w in caught]


# 1. aggregate with a missing key value


def test_aggregate_compares_groups_whose_key_is_missing():
    x = pd.DataFrame({"k": ["a", None, None], "v": [1.0, 2.0, 3.0]})
    y = pd.DataFrame({"k": ["a", None, None], "v": [1.0, 9.0, 9.0]})
    d = diff_table(x, y, duplicates="aggregate")
    assert len(d) == 2
    assert d.loc[d["K"].isna(), "status"].astype(str).tolist() == ["changed"]
    assert not is_matching(d)


# 2. assert_table_equal() passes diff_table()'s warnings on


def test_assert_table_equal_passes_notes_on():
    expected = pd.DataFrame({"id": [1, 2], "v": [1.0, 2.0]})
    actual = expected.assign(loaded_at="t")
    _, caught = recorded(assert_table_equal, actual, expected)
    note = "Dropped column(s) of `y` that the benchmark `x` does not have: loaded_at."
    assert (DiffNote, note) in caught


# 3. diff_summary()'s magnitudes, against hand-computed values


def test_diff_summary_magnitudes():
    x = pd.DataFrame({"id": [1, 2, 3, 4], "a": [1.0, 2.0, 3.0, 5.0]})
    y = pd.DataFrame({"id": [1, 2, 3, 5], "a": [1.0, 4.0, 6.0, 7.0]})
    s = diff_summary(diff_table(x, y)).iloc[0]
    # The joined pairs differ by 0, 2 and 3; rows in one table only carry no difference.
    counts = (s["n"], s["n_same"], s["n_changed"], s["n_only_x"], s["n_only_y"])
    assert counts == (5, 1, 2, 1, 1)
    assert s["max_abs_diff"] == 3.0
    assert s["mean_abs_diff"] == pytest.approx(5 / 3)
    assert s["rmse"] == pytest.approx(math.sqrt(13 / 3))
    assert s["max_rel_diff"] == pytest.approx(0.5)


# 4. CLI options reach diff_table()


@pytest.fixture
def near(tmp_path):
    x, y = tmp_path / "x.csv", tmp_path / "y.csv"
    x.write_text("id,a,b\n1,100.0,1.0\n2,200.0,2.0\n")
    y.write_text("id,a,b\n1,100.4,1.0\n2,200.0,2.0\n")
    return str(x), str(y)


def test_cli_tolerance_reaches_the_comparison(near):
    assert main([*near, "-q"]) == 1
    assert main([*near, "--tolerance", "0.5", "-q"]) == 0
    assert main([*near, "--tolerance", "0.1", "-q"]) == 1


def test_cli_rel_tol_reaches_the_comparison(near):
    # 0.4 against 100.4 is a relative difference of about 0.004.
    assert main([*near, "--rel-tol", "0.01", "-q"]) == 0
    assert main([*near, "--rel-tol", "0.001", "-q"]) == 1


def test_cli_measures_by_name(near, capsys):
    assert main([*near, "--by", "id", "--measures", "b", "-s"]) == 0
    out = capsys.readouterr().out
    assert " B " in f" {out} " and " A " not in f" {out} "


def test_cli_sep_reaches_the_reader(tmp_path, capsys):
    x = tmp_path / "x.csv"
    x.write_text("id;a\n1;1.5\n2;2.5\n")
    assert main([str(x), str(x), "--sep", ";", "-s"]) == 0
    assert "A" in capsys.readouterr().out.split()


def test_cli_no_truncate_reaches_diff_table(near, monkeypatch):
    # CSV numbers never truncate through the CLI (a fractional column is read as
    # float in both files), so check that the flag reaches the library.
    import dazdiff.cli as cli

    seen = {}
    real = cli.diff_table

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(cli, "diff_table", spy)
    main([*near, "-q"])
    assert seen["truncate"] is True
    main([*near, "--no-truncate", "-q"])
    assert seen["truncate"] is False


def test_cli_output_holds_the_comparison(csvs_example, tmp_path):
    out = tmp_path / "out.csv"
    assert main([*csvs_example, "-b", "id,region", "-o", str(out), "-q"]) == 1
    written = pd.read_csv(out)
    assert list(written.columns[:3]) == ["ID", "REGION", "metric"]
    assert sorted(written["status"].unique()) == ["changed", "only_x", "only_y", "same"]


@pytest.fixture
def csvs_example():
    root = Path(__file__).parent.parent / "examples"
    return str(root / "benchmark.csv"), str(root / "stored.csv")


# 5. a key column that is not there


def test_a_missing_by_column_is_a_diff_error():
    x = pd.DataFrame({"id": [1], "v": [1.0]})
    with pytest.raises(DiffError, match=r"Key column\(s\) not found in `x`: nope"):
        diff_table(x, x.copy(), by=["id", "nope"])


# 6. an infinite value cast to an integer column


def test_an_infinite_value_is_not_cast_to_an_integer():
    x = pd.DataFrame({"id": [1, 2], "n": [1, 2], "v": [1.0, 2.0]})
    y = pd.DataFrame({"id": [1, 2], "n": [1.0, np.inf], "v": [1.0, 2.0]})
    with pytest.raises(DiffError, match="infinite"):
        diff_table(x, y, by=["id", "n"])


# 7. warnings the parity tests recorded and never checked


def test_exclude_of_a_column_only_in_y_does_not_warn():
    x = pd.DataFrame({"id": [1, 2], "amt": [1.0, 2.0]})
    y = pd.DataFrame({"id": [1, 2], "amt": [1.0, 2.0], "Load Stamp": ["a", "b"]})
    _, caught = recorded(diff_table, x, y, mode="equal", exclude="LOAD_STAMP")
    assert caught == []


def test_an_ignored_column_is_announced_not_cast():
    x = pd.DataFrame({"id": [1, 2], "amt": [1.0, 2.0], "note": [1, 2]})
    y = pd.DataFrame({"id": [1, 2], "amt": [1.0, 2.0], "note": ["a", "b"]})
    _, caught = recorded(diff_table, x, y, by="id", measures=["amt"])
    note = "Ignored column(s) that are neither keys nor measures: note."
    assert caught == [(DiffNote, note)]


def test_excluding_a_clashing_name_is_silent():
    x = pd.DataFrame({"id": [1], "amt": [1.0], "a b": [1], "a_b": [2]})
    _, caught = recorded(diff_table, x, x.copy(), exclude="A_B")
    assert [c for c, _ in caught if c is DiffWarning] == []


# 8. a column named twice


def test_a_column_named_twice_is_compared_once():
    x = pd.DataFrame({"id": [1, 2], "amount": [1.0, 2.0]})
    d = diff_table(x, x.copy(), by=["id", "ID"], measures=["amount", "Amount"])
    assert d.attrs["by"] == ["ID"]
    assert d["metric"].tolist() == ["AMOUNT", "AMOUNT"] and len(d) == 2


# 9. assert_table_equal()'s message


@pytest.fixture
def off():
    expected = pd.DataFrame({"id": [1, 2], "v": [1.0, 2.0]})
    return expected.assign(v=[5.0, 6.0]), expected


def test_assert_message_with_every_record_shown(off):
    with pytest.raises(AssertionError) as e:
        assert_table_equal(*off)
    text = str(e.value)
    assert "2 of 2 records differ" in text
    assert "First" not in text and "5.0" in text and "6.0" in text


def test_assert_message_with_no_records_shown(off):
    with pytest.raises(AssertionError) as e:
        assert_table_equal(*off, max_rows=0)
    text = str(e.value)
    assert "First" not in text and "value_x" not in text
    assert "n_changed" in text


# 10. a bool column of y cast to a number


def test_a_bool_column_casts_to_a_number():
    x = pd.DataFrame({"id": [1, 2, 3], "v": [1.0, 0.0, 1.0]})
    y = pd.DataFrame({"id": [1, 2, 3], "v": [True, False, False]})
    d = diff_table(x, y)
    assert d["value_y"].tolist() == [1.0, 0.0, 0.0]
    assert d["status"].astype(str).tolist() == ["same", "same", "changed"]
