"""assert_table_equal(), after the R package's expect_table_equal() (design §6)."""

import warnings

import pandas as pd
import pytest

from dazdiff import DiffError, assert_table_equal


@pytest.fixture
def expected():
    return pd.DataFrame({"id": [1, 2, 3], "amount": [100.0, 200.0, 300.0]})


def test_it_passes_and_returns_the_comparison(expected):
    d = assert_table_equal(expected.copy(), expected)
    assert (d["status"] == "same").all()


def test_it_allows_a_relative_difference_of_sqrt_eps(expected):
    actual = expected.assign(amount=[100.0, 200.0 * (1 + 1e-9), 300.0])
    assert_table_equal(actual, expected)
    with pytest.raises(AssertionError):
        assert_table_equal(actual, expected, rel_tol=0)


def test_it_fails_on_a_change_with_a_bounded_message(expected):
    actual = expected.assign(amount=[100.0, 250.0, 301.0])
    with pytest.raises(AssertionError) as e:
        assert_table_equal(actual, expected, max_rows=1)
    text = str(e.value)
    assert text.startswith("`actual` does not match `expected`: 2 of 3 records differ.")
    assert "n_changed" in text  # diff_summary()
    assert "First 1 records that differ:" in text
    assert "250.0" in text and "301.0" not in text


def test_actual_is_tested_against_expected(expected):
    # expected is the benchmark: actual's extra column is dropped, its types cast.
    actual = expected.assign(id=["1", "2", "3"], extra="z")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert_table_equal(actual, expected)
    with pytest.raises(DiffError, match="lacks column"):  # cannot be answered
        assert_table_equal(expected[["id"]], expected)


def test_a_repeated_key_is_an_error_unless_duplicates_accepts_it():
    t = pd.DataFrame({"id": [1, 1], "v": [1.0, 2.0]})
    with pytest.raises(DiffError, match="does not pair duplicate keys"):
        assert_table_equal(t.copy(), t)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert_table_equal(t.copy(), t, duplicates="disambiguate")
    assert_table_equal(t.copy(), t, duplicates="aggregate")  # rel_tol steps aside


def test_repeated_whole_rows_are_paired():
    t = pd.DataFrame({"id": [1, 1, 2]})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        assert_table_equal(t.copy(), t)


def test_an_explicit_rel_tol_with_aggregate_is_refused():
    t = pd.DataFrame({"id": [1, 1], "v": [1.0, 2.0]})
    with pytest.raises(DiffError, match="exactly"):
        assert_table_equal(t.copy(), t, duplicates="aggregate", rel_tol=1e-8)


def test_tables_that_share_no_key_fail(expected):
    with pytest.raises(AssertionError, match="key in common"):
        assert_table_equal(expected.assign(id=[7, 8, 9]), expected)


def test_it_checks_max_rows(expected):
    for bad in (-1, 1.5, True):
        with pytest.raises(DiffError, match="max_rows"):
            assert_table_equal(expected, expected, max_rows=bad)
