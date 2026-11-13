"""Fixes from the Stage 9 code review (#9): report counts, dropped names,
assert_table_equal()'s pairing signal, and the shared result readers."""

import warnings

import pandas as pd
import pytest

from dazdiff import DiffError, assert_table_equal, diff_report, diff_table, is_matching
from dazdiff.core import _result_ids, _result_shape


def quiet(x, y, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return diff_table(x, y, **kwargs)


# --- #30 ------------------------------------------------------------------------


def test_the_report_counts_the_keys_it_did_not_show():
    x = pd.DataFrame({"k": ["a", "a", "a", "b"], "v": [1.0, 2.0, 3.0, 4.0]})
    y = pd.DataFrame({"k": ["b"], "v": [4.0]})
    text = diff_report(quiet(x, y)).format(unicode=False)
    assert "more)" not in text  # 'a' is the only key only in x
    x2 = pd.DataFrame({"k": list("abcdefg"), "v": [1.0] * 7})
    text = diff_report(quiet(x2, y.assign(k=["g"]).assign(v=[1.0])), sample_k=2).format(
        unicode=False
    )
    assert "(...and 4 more)" in text


# --- #31 ------------------------------------------------------------------------


def test_extra_y_columns_with_empty_or_clashing_names_are_listed_as_dropped():
    x = pd.DataFrame({"id": [1], "v": [1.0]})
    y = pd.DataFrame({"id": [1], "v": [1.0], "!!": [0], "a b": [0], "a_b": [0]})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        d = diff_table(x, y)
    assert d.attrs["dropped"] == ["!!", "a b", "a_b"]
    assert any("!!, a b, a_b" in str(w.message) for w in caught)


# --- #32 ------------------------------------------------------------------------


def test_assert_refuses_pairing_whatever_the_warning_says(monkeypatch):
    import dazdiff.core as core

    t = pd.DataFrame({"id": [1, 1], "v": [1.0, 2.0]})
    real = core.diff_table

    def reworded(*args, **kwargs):  # the pairing warning's text must not matter
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return real(*args, **kwargs)

    monkeypatch.setattr(core, "diff_table", reworded)
    with pytest.raises(DiffError, match="does not pair duplicate keys"):
        assert_table_equal(t.copy(), t)


# --- shared result readers -----------------------------------------------------------


def test_the_result_readers_agree_without_attrs():
    x = pd.DataFrame({"id": [1, 1], "v": [1.0, 2.0]})
    g = diff_table(x, x.copy(), duplicates="aggregate")
    c = quiet(x.assign(id=[1, 2]), x.assign(id=[1, 2], v=[1.0, 3.0]))
    r = quiet(x[["id"]], x[["id"]].copy())
    for d, shape in ((g, "groups"), (c, "cells"), (r, "rows")):
        bare = pd.DataFrame(d)
        bare.attrs = {}
        assert _result_shape(bare) == shape
        assert _result_ids(bare) == ["ID"]
    bare = pd.DataFrame(g)
    bare.attrs = {}
    with pytest.raises(DiffError, match="exactly"):
        is_matching(bare, tolerance=1)
