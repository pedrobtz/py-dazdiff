"""dazdiff: compare two tables value by value."""

from .core import (
    DiffError,
    DiffNote,
    DiffWarning,
    assert_table_equal,
    diff_explain,
    diff_summary,
    diff_table,
    is_matching,
    normalize_names,
)
from .report import DiffReport, diff_report

__version__ = "0.1.0.dev0"
__all__ = [
    "DiffError",
    "DiffNote",
    "DiffReport",
    "DiffWarning",
    "assert_table_equal",
    "diff_explain",
    "diff_report",
    "diff_summary",
    "diff_table",
    "is_matching",
    "normalize_names",
]
