"""diff_explain(), ported from the R package's test-explain.R (design §4a)."""

import warnings

import numpy as np
import pandas as pd
import pytest

from dazdiff import DiffError, diff_explain, diff_table


def quiet(x, y, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return diff_table(x, y, **kwargs)


@pytest.fixture
def d():
    """The README's comparison: ID 1 and 3 match, 2 changed, 4 and 5 are in one
    table only, and REGION north is mixed."""
    x = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "region": ["north", "south", "east", "west"],
            "amount": [100.0, 200.0, 300.0, 400.0],
            "units": [10.0, 20.0, 30.0, 40.0],
        }
    )
    y = pd.DataFrame(
        {
            "id": [1, 2, 3, 5],
            "region": ["north", "south", "east", "north"],
            "amount": [100.0, 200.5, 300.0, 500.0],
            "units": [10.0, 20.0, 30.0, 50.0],
        }
    )
    return diff_table(x, y, by=["id", "region"])


def test_a_row_matches_only_when_every_measure_does(d):
    e = diff_explain(d, "id")
    assert e["ID"].tolist() == [2, 4, 5, 1, 3]
    assert e["class"].tolist() == ["differ", "differ", "differ", "same", "same"]
    assert e["n"].tolist() == [1] * 5
    assert e.loc[e["ID"] == 2, "n_changed"].tolist() == [1]
    assert list(e.columns) == ["ID", "n", "n_same", "n_changed", "n_only_x", "n_only_y", "class"]


def test_the_overview_classes_every_value_and_ranks_identifiers_last(d):
    e = diff_explain(d)
    assert e["column"].tolist() == ["REGION", "metric", "ID"]
    region = e.set_index("column").loc["REGION"]
    assert [region.same_only, region.differ_only, region.mixed] == [1, 2, 1]
    ident = e.set_index("column").loc["ID"]
    assert bool(ident.separates) and bool(ident.identifies)
    metric = e.set_index("column").loc["metric"]
    assert [metric.n_values, metric.mixed] == [2, 2]
    assert not metric.identifies
    assert list(e.columns) == [
        "column",
        "n_values",
        "same_only",
        "differ_only",
        "mixed",
        "separates",
        "identifies",
    ]


def test_a_column_that_separates_without_identifying_leads():
    x = pd.DataFrame({"id": [1, 2, 3, 4], "grp": ["a", "a", "b", "b"], "v": [1.0, 2, 3, 4]})
    y = pd.DataFrame({"id": [1, 2, 3, 4], "grp": ["a", "a", "b", "b"], "v": [1.0, 2, 9, 9]})
    e = diff_explain(diff_table(x, y))
    assert e["column"].iloc[0] == "GRP"
    assert e["separates"].iloc[0] and not e["identifies"].iloc[0]
    g = diff_explain(diff_table(x, y), "grp")
    assert g["GRP"].tolist() == ["b", "a"]
    assert g["class"].tolist() == ["differ", "same"]


def test_metric_counts_each_value_and_joined_leaves_out_membership(d):
    assert diff_explain(d, "metric")["class"].tolist() == ["mixed", "mixed"]
    joined = diff_explain(d, "metric", among="joined")
    assert joined["metric"].tolist() == ["AMOUNT", "UNITS"]
    assert joined["class"].tolist() == ["mixed", "same"]
    assert joined["n"].tolist() == [3, 3]
    r = diff_explain(d, "region", among="joined")
    assert r["REGION"].tolist() == ["south", "east", "north"]
    assert int((r["n_only_x"] + r["n_only_y"]).sum()) == 0


def test_membership_comes_from_row_id_not_from_the_value():
    x = pd.DataFrame({"id": [1, 2], "v": [1.0, np.nan]})
    y = pd.DataFrame({"id": [1, 2, 3], "v": [1.0, np.nan, 3.0]})
    e = diff_explain(diff_table(x, y), "id")
    assert e.loc[e["ID"] == 2, "class"].tolist() == ["same"]
    assert e.loc[e["ID"] == 3, "n_only_y"].tolist() == [1]


def test_paired_duplicates_are_rows_of_their_own():
    x = pd.DataFrame({"k": ["a", "a"], "v": [1.0, 2.0]})
    y = pd.DataFrame({"k": ["a", "a"], "v": [1.0, 5.0]})
    e = diff_explain(quiet(x, y), "k")
    assert e[["n", "n_same", "n_changed", "class"]].values.tolist() == [[2, 1, 1, "mixed"]]


def test_it_explains_every_shape():
    x = pd.DataFrame({"k": ["a", "b", "b"], "v": [1.0, 2.0, 2.0]})
    y = pd.DataFrame({"k": ["a", "b"], "v": [1.0, 5.0]})
    g = diff_explain(diff_table(x, y, duplicates="aggregate"), "k")
    assert g["class"].tolist() == ["differ", "same"]

    rows = quiet(pd.DataFrame({"k": ["a", "b"]}), pd.DataFrame({"k": ["a", "c"]}))
    assert diff_explain(rows)["column"].tolist() == ["K"]
    assert diff_explain(rows, "k")["class"].tolist() == ["differ", "differ", "same"]

    p = diff_table(x.iloc[:2][["v"]], y[["v"]], row_key="position")
    e = diff_explain(p).set_index("column")
    assert bool(e.loc["row_number", "separates"]) and bool(e.loc["row_number", "identifies"])


def test_an_empty_or_filtered_result_still_explains(d):
    assert len(diff_explain(d.iloc[:0], "region")) == 0
    assert diff_explain(d.iloc[:0])["n_values"].tolist() == [0, 0, 0]
    assert diff_explain(d[d["status"] == "same"])["separates"].all()


def test_missing_key_values_are_values_too():
    x = pd.DataFrame({"k": ["a", None], "v": [1.0, 2.0]})
    y = pd.DataFrame({"k": ["a", None], "v": [1.0, 3.0]})
    e = diff_explain(diff_table(x, y), "k")
    assert e["class"].tolist() == ["differ", "same"]
    assert e["K"].isna().tolist() == [True, False]


def test_it_refuses_what_is_not_a_key_column(d):
    with pytest.raises(DiffError, match="result of diff_table"):
        diff_explain(pd.DataFrame({"a": [1]}))
    with pytest.raises(DiffError, match="single column name"):
        diff_explain(d, ["id", "region"])
    with pytest.raises(DiffError, match=r"key column of the result.*amount"):
        diff_explain(d, "amount")
    with pytest.raises(DiffError, match="among"):
        diff_explain(d, among="some")
    pd.testing.assert_frame_equal(diff_explain(d, "Region"), diff_explain(d, "REGION"))
