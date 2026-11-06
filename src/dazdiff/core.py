"""Compare two tables value by value."""

from __future__ import annotations

import re
import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd

STATUSES = ["same", "changed", "only_x", "only_y"]
_MODES = ("benchmark", "equal")
_DUPLICATES = ("pair", "error")


class DiffError(ValueError):
    """The two tables cannot be compared as asked."""


class DiffWarning(UserWarning):
    """Something was changed or assumed so the comparison could go ahead."""


def normalize_names(names: Sequence[str]) -> list[str]:
    """Upper-case column names, turning anything but letters and digits into ``_``."""
    out = [re.sub(r"[^A-Z0-9]", "_", str(n).upper()) for n in names]
    dupes = sorted({n for n in out if out.count(n) > 1})
    if dupes:
        raise DiffError(f"Column names clash after normalization: {', '.join(dupes)}.")
    return out


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = normalize_names(df.columns)
    return df.reset_index(drop=True)


def _is_float(s: pd.Series) -> bool:
    return pd.api.types.is_float_dtype(s)


def _is_int(s: pd.Series) -> bool:
    return pd.api.types.is_integer_dtype(s) and not pd.api.types.is_bool_dtype(s)


def _cast_to(y: pd.Series, x: pd.Series, truncate: bool, truncated: dict) -> pd.Series:
    """Cast column ``y`` to the dtype of the benchmark column ``x``."""
    if y.dtype == x.dtype:
        return y
    name = y.name
    try:
        if _is_int(x) or _is_float(x):
            num = pd.to_numeric(y, errors="raise")
            if _is_int(x):
                frac = num.notna() & (num != np.trunc(num))
                if frac.any():
                    if not truncate:
                        raise DiffError(
                            f"Casting `{name}` to integer would truncate {int(frac.sum())} value(s)."
                        )
                    truncated[name] = list(
                        zip(num[frac].tolist(), np.trunc(num[frac]).tolist(), strict=True)
                    )
                num = np.trunc(num)
                return num.astype("Int64" if num.isna().any() else x.dtype)
            return num.astype(x.dtype)
        if pd.api.types.is_bool_dtype(x):
            return (
                y.map({"True": True, "False": False, "true": True, "false": False})
                .fillna(y)
                .astype(bool)
            )
        if pd.api.types.is_datetime64_any_dtype(x):
            return pd.to_datetime(y)
        return y.astype(x.dtype)
    except DiffError:
        raise
    except (ValueError, TypeError) as e:
        raise DiffError(f"Cannot cast `{name}` from {y.dtype} to {x.dtype}: {e}") from e


def _select_measures(x: pd.DataFrame, measures) -> list[str]:
    if isinstance(measures, str) and measures in ("numeric", "numeric+integer"):
        keep = _is_float if measures == "numeric" else (lambda s: _is_float(s) or _is_int(s))
        return [c for c in x.columns if keep(x[c])]
    names = [measures] if isinstance(measures, str) else list(measures)
    names = normalize_names(names)
    missing = [m for m in names if m not in x.columns]
    if missing:
        raise DiffError(f"Measure column(s) not found: {', '.join(missing)}.")
    bad = [m for m in names if not (_is_float(x[m]) or _is_int(x[m]))]
    if bad:
        raise DiffError(f"Measure column(s) are not numeric: {', '.join(bad)}.")
    return names


def _add_seq(df: pd.DataFrame, by: list[str], duplicates: str, label: str) -> pd.DataFrame:
    seq = df.groupby(by, dropna=False, sort=False).cumcount() + 1
    n_dup = int((seq > 1).sum())
    if n_dup and duplicates != "ignore":
        if duplicates == "error":
            raise DiffError(f"`{label}` has {n_dup} row(s) with a repeated key.")
        warnings.warn(
            f"`{label}` has {n_dup} row(s) with a repeated key; they are paired in order of arrival.",
            DiffWarning,
            stacklevel=4,
        )
    df = df.copy()
    df["KEY_SEQ"] = seq.astype(int)
    return df


def diff_table(
    x: pd.DataFrame,
    y: pd.DataFrame,
    by: Sequence[str] | str | None = None,
    measures: Sequence[str] | str = "numeric",
    mode: str = "benchmark",
    tolerance: float = 0.0,
    rel_tol: float = 0.0,
    truncate: bool = True,
    duplicates: str = "pair",
) -> pd.DataFrame:
    """Compare two tables value by value.

    Parameters
    ----------
    x, y
        The tables. In ``mode="benchmark"``, ``x`` is the benchmark: ``y`` is cast to
        its column types and ``y``'s extra columns are dropped. In ``mode="equal"`` the
        tables must have the same columns with the same types.
    by
        Key columns that identify a row. Defaults to every column that is not a measure.
    measures
        ``"numeric"`` (float columns), ``"numeric+integer"``, or column names.
    tolerance, rel_tol
        A value is ``same`` when ``abs_diff <= tolerance`` or
        ``rel_diff <= rel_tol``.
    truncate
        Allow truncating non-integral values when casting to an integer column.
    duplicates
        ``"pair"`` pairs repeated keys in arrival order (with a warning);
        ``"error"`` refuses them.

    Returns
    -------
    pandas.DataFrame
        One row per key and measure, with ``row_id_x``, ``row_id_y``, ``value_x``,
        ``value_y``, ``diff``, ``abs_diff``, ``rel_diff`` and ``status``. With no
        measures, one row per row of either table. Values truncated by the cast are
        in ``result.attrs["truncated"]``.
    """
    if mode not in _MODES:
        raise DiffError(f"`mode` must be one of {_MODES}, not {mode!r}.")
    if duplicates not in _DUPLICATES:
        raise DiffError(f"`duplicates` must be one of {_DUPLICATES}, not {duplicates!r}.")
    if tolerance < 0 or rel_tol < 0:
        raise DiffError("`tolerance` and `rel_tol` must not be negative.")

    x, y = _normalize(x), _normalize(y)
    only_x = [c for c in x.columns if c not in y.columns]
    only_y = [c for c in y.columns if c not in x.columns]
    truncated: dict = {}

    if mode == "equal":
        if only_x or only_y:
            raise DiffError(
                'mode = "equal" needs the same columns in both tables.\n'
                f"  only in `x`: {', '.join(only_x) or '<none>'}\n"
                f"  only in `y`: {', '.join(only_y) or '<none>'}"
            )
        y = y[list(x.columns)]
        differ = [
            f"{c} ({x[c].dtype} vs {y[c].dtype})" for c in x.columns if x[c].dtype != y[c].dtype
        ]
        if differ:
            raise DiffError('mode = "equal" needs the same column types: ' + ", ".join(differ))
    else:
        if only_x:
            raise DiffError(f"`y` lacks column(s) of the benchmark `x`: {', '.join(only_x)}.")
        if only_y:
            warnings.warn(
                f"Dropped column(s) of `y` that the benchmark `x` does not have: {', '.join(only_y)}.",
                DiffWarning,
                stacklevel=2,
            )
        y = pd.DataFrame({c: _cast_to(y[c], x[c], truncate, truncated) for c in x.columns})
        if truncated:
            lines = [f"  {c}: {len(v)} ({v[0][0]:g} -> {v[0][1]:g})" for c, v in truncated.items()]
            n = sum(len(v) for v in truncated.values())
            warnings.warn(
                f"Truncated {n} value(s) while casting to the reference types:\n"
                + "\n".join(lines),
                DiffWarning,
                stacklevel=2,
            )

    if by is not None:
        by = normalize_names([by] if isinstance(by, str) else list(by))
        missing = [b for b in by if b not in x.columns]
        if missing:
            raise DiffError(f"Key column(s) not found: {', '.join(missing)}.")
        meas = [m for m in _select_measures(x, measures) if m not in by]
    else:
        meas = _select_measures(x, measures)
        by = [c for c in x.columns if c not in meas]
    if not by:
        raise DiffError(
            "Every column is a measure; pass `by` or fewer `measures` so rows have a key."
        )

    # With no measures a row is its own key, so repeats are copies, not clashes.
    dup_rule = duplicates if meas else "ignore"
    x = _add_seq(x, by, dup_rule, "x")
    y = _add_seq(y, by, dup_rule, "y")
    keys = [*by, "KEY_SEQ"]
    x["row_id_x"] = np.arange(1, len(x) + 1)
    y["row_id_y"] = np.arange(1, len(y) + 1)

    if not meas:
        warnings.warn(
            "No measure columns, so rows are compared as wholes: do the two tables have the same rows?",
            DiffWarning,
            stacklevel=2,
        )
        m = x[[*keys, "row_id_x"]].merge(y[[*keys, "row_id_y"]], on=keys, how="outer")
        status = np.select(
            [m["row_id_x"].isna(), m["row_id_y"].isna()], ["only_y", "only_x"], "same"
        )
        out = m.assign(status=pd.Categorical(status, categories=STATUSES))
        out = out.rename(columns={"KEY_SEQ": "key_seq"})
        keys = [*by, "key_seq"]
    else:
        m = x[[*keys, "row_id_x", *meas]].merge(
            y[[*keys, "row_id_y", *meas]], on=keys, how="outer", suffixes=("_x", "_y")
        )
        parts = []
        for metric in meas:
            p = m[[*keys, "row_id_x", "row_id_y"]].copy()
            p.insert(len(keys), "metric", metric)
            p["value_x"] = m[f"{metric}_x"].astype(float)
            p["value_y"] = m[f"{metric}_y"].astype(float)
            parts.append(p)
        out = pd.concat(parts, ignore_index=True)
        out["diff"] = out["value_y"] - out["value_x"]
        out["abs_diff"] = out["diff"].abs()
        scale = np.maximum(out["value_x"].abs(), out["value_y"].abs())
        out["rel_diff"] = np.where(scale > 0, out["abs_diff"] / scale.where(scale > 0, 1), 0.0)
        out.loc[out["abs_diff"].isna(), "rel_diff"] = np.nan

        missing_x, missing_y = out["row_id_x"].isna(), out["row_id_y"].isna()
        both_na = out["value_x"].isna() & out["value_y"].isna()
        close = (out["abs_diff"] <= tolerance) | (out["rel_diff"] <= rel_tol)
        status = np.select(
            [missing_x, missing_y, both_na | close], ["only_y", "only_x", "same"], "changed"
        )
        out["status"] = pd.Categorical(status, categories=STATUSES)
        out = out.drop(columns="KEY_SEQ")
        keys = by

    for c in ("row_id_x", "row_id_y"):
        out[c] = out[c].astype("Int64")
    sort = keys + (["metric"] if meas else [])
    out = out.sort_values(sort, kind="stable", ignore_index=True)
    out.attrs["by"] = by
    out.attrs["measures"] = meas
    out.attrs["truncated"] = truncated
    return out


def diff_summary(d: pd.DataFrame) -> pd.DataFrame:
    """Condense a :func:`diff_table` result per measure."""
    counts = (
        pd.crosstab(d["metric"], d["status"]).reindex(columns=STATUSES, fill_value=0)
        if "metric" in d
        else pd.crosstab(pd.Series("<rows>", index=d.index, name="metric"), d["status"]).reindex(
            columns=STATUSES, fill_value=0
        )
    )
    counts.columns = [f"n_{s}" for s in STATUSES]
    counts.insert(0, "n", counts.sum(axis=1))
    if "abs_diff" in d:
        g = d.groupby("metric")
        stats = pd.DataFrame(
            {
                "max_abs_diff": g["abs_diff"].max(),
                "mean_abs_diff": g["abs_diff"].mean(),
                "rmse": g["abs_diff"].apply(lambda s: float(np.sqrt((s.dropna() ** 2).mean()))),
                "max_rel_diff": g["rel_diff"].max(),
            }
        )
        counts = counts.join(stats)
    counts.columns.name = None
    return counts.reset_index()


def is_equal(d: pd.DataFrame) -> bool:
    """``True`` when every row of a :func:`diff_table` result is ``same``."""
    return bool((d["status"] == "same").all())
