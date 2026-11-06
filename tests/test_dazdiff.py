import pandas as pd
import pytest

from dazdiff import DiffError, DiffWarning, diff_summary, diff_table, is_equal
from dazdiff.cli import main

BENCHMARK = pd.DataFrame(
    {
        "id": [1, 2, 3, 4],
        "region": ["north", "south", "east", "west"],
        "amount": [100.0, 200.0, 300.0, 400.0],
        "units": [10, 20, 30, 40],
    }
)
STORED = pd.DataFrame(
    {
        "ID": ["1", "2", "3", "5"],
        "Region": ["north", "south", "east", "north"],
        "Amount": [100, 200.5, 300, 500],
        "Units": [10.4, 20, 30, 50],
        "loaded_at": "2026-09-29",
    }
)


def compare(**kw):
    with pytest.warns(DiffWarning):
        return diff_table(BENCHMARK, STORED, by=["id", "region"], measures="numeric+integer", **kw)


def status_of(d, id_, metric):
    return d.loc[(d["ID"] == id_) & (d["metric"] == metric), "status"].item()


def test_benchmark_mode_casts_drops_and_flags():
    d = compare()
    assert list(d.columns[:3]) == ["ID", "REGION", "metric"]
    assert len(d) == 10
    assert status_of(d, 2, "AMOUNT") == "changed"
    assert status_of(d, 1, "UNITS") == "same"  # 10.4 truncated to 10
    assert status_of(d, 4, "AMOUNT") == "only_x"
    assert status_of(d, 5, "AMOUNT") == "only_y"
    assert d.attrs["truncated"] == {"UNITS": [(10.4, 10.0)]}
    row = d[(d["ID"] == 2) & (d["metric"] == "AMOUNT")].iloc[0]
    assert row["diff"] == pytest.approx(0.5)
    assert row["rel_diff"] == pytest.approx(0.5 / 200.5)


def test_no_truncate_refuses_cast():
    with pytest.raises(DiffError, match="truncate"), pytest.warns(DiffWarning):
        diff_table(BENCHMARK, STORED, by=["id", "region"], truncate=False)


def test_rel_tol_absorbs_small_changes():
    d = compare(rel_tol=0.01)
    assert set(d["status"]) == {"same", "only_x", "only_y"}


def test_equal_mode_requires_same_columns():
    with pytest.raises(DiffError, match="same columns"):
        diff_table(BENCHMARK, STORED, mode="equal")


def test_equal_mode_ignores_row_order():
    d = diff_table(BENCHMARK, BENCHMARK.iloc[[3, 0, 1, 2]], mode="equal")
    assert is_equal(d)
    assert d["row_id_y"].tolist() == [2, 3, 4, 1]


def test_rows_compared_as_wholes_count_copies():
    y = pd.DataFrame({"ID": [3, 1, 2, 2], "REGION": ["east", "north", "south", "south"]})
    with pytest.warns(DiffWarning, match="as wholes"):
        d = diff_table(BENCHMARK[["id", "region"]], y)
    assert d["status"].value_counts().to_dict() == {
        "same": 3,
        "only_x": 1,
        "only_y": 1,
        "changed": 0,
    }


def test_duplicates_error():
    x = pd.DataFrame({"k": [1, 1], "v": [1.0, 2.0]})
    with pytest.raises(DiffError, match="repeated key"):
        diff_table(x, x, duplicates="error")


def test_both_missing_is_same():
    x = pd.DataFrame({"k": [1], "v": [float("nan")]})
    assert is_equal(diff_table(x, x))


def test_summary():
    s = diff_summary(compare()).set_index("metric")
    assert s.loc["AMOUNT", ["n", "n_same", "n_changed", "n_only_x", "n_only_y"]].tolist() == [
        5,
        2,
        1,
        1,
        1,
    ]
    assert s.loc["AMOUNT", "max_abs_diff"] == pytest.approx(0.5)


@pytest.fixture
def csvs(tmp_path):
    x, y = tmp_path / "x.csv", tmp_path / "y.csv"
    BENCHMARK.assign(amount=[100, 200, 300, 400]).to_csv(x, index=False)  # ints, as CSV would infer
    STORED.to_csv(y, index=False)
    return str(x), str(y)


def test_cli_reports_differences(csvs, capsys):
    assert main([*csvs, "--by", "id,region", "--only-diffs"]) == 1
    out = capsys.readouterr()
    assert "200.5" in out.out  # int column in x was not used to truncate y
    assert "LOADED_AT" in out.err


def test_cli_equal_files_exit_zero(csvs, capsys):
    assert main([csvs[0], csvs[0], "-q"]) == 0
    assert capsys.readouterr().out == ""


def test_cli_error_exit_two(csvs, capsys):
    assert main([*csvs, "--mode", "equal"]) == 2
    assert "same columns" in capsys.readouterr().err


def test_cli_writes_output(csvs, tmp_path):
    out = tmp_path / "out.csv"
    main([*csvs, "-b", "id,region", "-o", str(out), "-q"])
    assert len(pd.read_csv(out)) == 10
