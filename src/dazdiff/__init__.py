"""dazdiff: compare two tables value by value."""

from importlib.metadata import PackageNotFoundError, version

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

try:  # one source: the version in pyproject.toml, read from the installed metadata
    __version__ = version("dazdiff")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0+unknown"
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
