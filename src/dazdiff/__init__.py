"""dazdiff: compare two tables value by value."""

from .core import (
    DiffError,
    DiffNote,
    DiffWarning,
    diff_summary,
    diff_table,
    is_matching,
    normalize_names,
)

__version__ = "0.1.0.dev0"
__all__ = [
    "DiffError",
    "DiffNote",
    "DiffWarning",
    "diff_summary",
    "diff_table",
    "is_matching",
    "normalize_names",
]
