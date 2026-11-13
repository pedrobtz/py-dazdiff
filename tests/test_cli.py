import io
import sys

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


def test_cli_report(csvs, capsys):
    assert main([*csvs, "-b", "id,region", "--report"]) == 1
    out = capsys.readouterr().out
    assert "dazdiff report" in out
    assert "x.csv is tested against" not in out  # x is the benchmark
    assert "y.csv is tested against x.csv" in out
    assert main([*csvs, "-b", "id,region", "-R", "--x-name", "bench", "--ascii"]) == 1
    out = capsys.readouterr().out
    assert "y.csv is tested against bench" in out
    assert out.isascii()


def test_cli_report_falls_back_to_ascii(csvs, monkeypatch):
    raw = io.BytesIO()
    console = io.TextIOWrapper(raw, encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", console)
    assert main([*csvs, "-b", "id,region", "--report"]) == 1
    console.flush()
    text = raw.getvalue().decode("cp1252")
    assert "dazdiff report" in text
    assert text.isascii()


def test_python_dash_m_runs_the_cli(monkeypatch, capsys):
    import runpy

    monkeypatch.setattr(sys, "argv", ["dazdiff", "--version"])
    with pytest.raises(SystemExit) as e:
        runpy.run_module("dazdiff", run_name="__main__")
    assert e.value.code == 0
    assert capsys.readouterr().out.startswith("dazdiff ")


def test_version_comes_from_the_package_metadata():
    from importlib.metadata import version

    import dazdiff

    assert dazdiff.__version__ == version("dazdiff")


# --- Stage 9 review fixes ----------------------------------------------------------


def test_cli_widening_keeps_an_integer_key_a_key(tmp_path, capsys):
    # One empty id makes the second file read `id` as float; it must stay the key.
    x, y = tmp_path / "x.csv", tmp_path / "y.csv"
    x.write_text("id,v\n1,1.5\n2,2.5\n")
    y.write_text("id,v\n1,1.5\n,2.5\n")
    assert main([str(x), str(y), "--only-diffs"]) == 1
    captured = capsys.readouterr()
    assert "Every column is a measure" not in captured.err
    assert "only_x" in captured.out and "only_y" in captured.out


def test_cli_widening_with_excluded_and_integer_measures(tmp_path):
    x, y = tmp_path / "x.csv", tmp_path / "y.csv"
    x.write_text("id,n,v,stamp\n1,10,1.5,a\n2,20,2.5,b\n")
    y.write_text("id,n,v,stamp\n1,10.0,1.5,c\n2,20.5,2.5,d\n")
    # n is an integer measure read as float in y: widened, so 20.5 is not truncated.
    assert main([str(x), str(y), "-m", "numeric+integer", "--exclude", "stamp,v", "-q"]) == 2
    assert (
        main([str(x), str(y), "-m", "numeric+integer", "--by", "id", "--exclude", "stamp", "-q"])
        == 1
    )


def test_cli_an_output_error_exits_two(csvs, tmp_path, capsys):
    out = tmp_path / "missing" / "out.csv"
    assert main([*csvs, "-b", "id,region", "-o", str(out)]) == 2
    err = capsys.readouterr().err
    assert "dazdiff: error:" in err and "Traceback" not in err


def test_cli_an_undecodable_file_exits_two(tmp_path, capsys):
    x = tmp_path / "x.csv"
    x.write_bytes(b"id,v\n1,\xff\xfe\n")
    assert main([str(x), str(x)]) == 2
    assert capsys.readouterr().err.startswith("dazdiff: error:")
