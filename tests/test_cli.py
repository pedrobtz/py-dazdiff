import pandas as pd
import pytest

from dazdiff.cli import main


@pytest.fixture
def csvs(tmp_path, benchmark, stored):
    x, y = tmp_path / "x.csv", tmp_path / "y.csv"
    benchmark.assign(amount=[100, 200, 300, 400]).to_csv(x, index=False)  # ints, as CSV infers
    stored.to_csv(y, index=False)
    return str(x), str(y)


def test_cli_reports_differences(csvs, capsys):
    assert main([*csvs, "--by", "id,region", "--only-diffs"]) == 1
    out = capsys.readouterr()
    assert "200.5" in out.out  # the int column in x did not truncate y
    assert "loaded_at" in out.err


def test_cli_equal_files_exit_zero(csvs, capsys):
    assert main([csvs[0], csvs[0], "-q"]) == 0
    assert capsys.readouterr().out == ""


def test_cli_error_exit_two(csvs, capsys):
    assert main([*csvs, "--mode", "equal"]) == 2
    assert "same columns" in capsys.readouterr().err


def test_cli_missing_file_exit_two(tmp_path, capsys):
    assert main([str(tmp_path / "nope.csv"), str(tmp_path / "nope.csv")]) == 2
    assert "error" in capsys.readouterr().err


def test_cli_writes_output(csvs, tmp_path):
    out = tmp_path / "out.csv"
    main([*csvs, "-b", "id,region", "-o", str(out), "-q"])
    assert len(pd.read_csv(out)) == 10


def test_cli_summary(csvs, capsys):
    main([*csvs, "-b", "id,region", "-s"])
    assert "max_abs_diff" in capsys.readouterr().out


def test_cli_ignores_clashes_in_columns_it_drops(tmp_path, capsys):
    x, y = tmp_path / "x.csv", tmp_path / "y.csv"
    x.write_text("k,v\n1,1.5\n")
    y.write_text("k,v,extra 1,extra_1\n1,1.5,0,0\n")
    assert main([str(x), str(y)]) == 0


def test_cli_row_key_sorted(tmp_path):
    x, y = tmp_path / "x.csv", tmp_path / "y.csv"
    x.write_text("a,b\n1.5,2.5\n3.5,4.5\n")
    y.write_text("a,b\n3.5,4.5\n1.5,2.5\n")
    assert main([str(x), str(y), "-q"]) == 2  # no key
    assert main([str(x), str(y), "--row-key", "position", "-q"]) == 1
    assert main([str(x), str(y), "--row-key", "sorted", "-q"]) == 0


def test_cli_duplicates_aggregate(tmp_path):
    x, y = tmp_path / "x.csv", tmp_path / "y.csv"
    x.write_text("k,v\n1,1.5\n1,2.5\n")
    y.write_text("k,v\n1,2.5\n1,1.5\n")
    assert main([str(x), str(y), "-q"]) == 1
    assert main([str(x), str(y), "--duplicates", "aggregate", "-q"]) == 0
    assert main([str(x), str(y), "--duplicates", "error", "-q"]) == 2


def test_cli_exclude(tmp_path, capsys):
    x, y = tmp_path / "x.csv", tmp_path / "y.csv"
    x.write_text("id,run,v\n1,r1,1.5\n2,r1,2.5\n")
    y.write_text("id,run,v\n1,r2,1.5\n2,r2,2.5\n")
    assert main([str(x), str(y)]) == 2
    assert "exclude=['run']" in capsys.readouterr().err
    assert main([str(x), str(y), "--exclude", "run"]) == 0


def test_cli_no_cast(tmp_path, capsys):
    x, y = tmp_path / "x.csv", tmp_path / "y.csv"
    x.write_text("id,v\n1,1.5\n")
    y.write_text("id,v\na,1.5\n")
    assert main([str(x), str(y), "--no-cast"]) == 2
    assert "cast=False" in capsys.readouterr().err


def test_cli_explain(csvs, capsys):
    assert main([*csvs, "-b", "id,region", "--explain"]) == 1
    out = capsys.readouterr().out
    assert out.splitlines()[0].split()[:3] == ["column", "n_values", "same_only"]
    assert main([*csvs, "-b", "id,region", "--explain", "region", "--joined"]) == 1
    out = capsys.readouterr().out
    assert "REGION" in out and "west" not in out


def test_cli_explain_refuses_a_measure(csvs, capsys):
    assert main([*csvs, "-b", "id,region", "--explain", "amount"]) == 2
    assert "key column of the result" in capsys.readouterr().err
