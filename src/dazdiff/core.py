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
_TRUE = {"true", "t", "yes", "1"}
_FALSE = {"false", "f", "no", "0"}


class DiffError(ValueError):
    """The two tables cannot be compared as asked."""


class DiffWarning(UserWarning):
    """The answer may be wrong, e.g. rows were paired by arrival order."""


class DiffNote(UserWarning):
    """An expected side effect, e.g. a column was dropped or a value truncated."""


# --- names -------------------------------------------------------------------


def _canon(name) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", str(name).upper()).strip("_")


def _join(names) -> str:
    return ", ".join(str(n) for n in names)


def normalize_names(names: Sequence[str]) -> list[str]:
    """Normalize column names as :func:`diff_table` does.

    Upper-case, every run of characters other than ``A-Z`` and ``0-9`` replaced by
    one ``_``, and ``_`` trimmed from both ends. A name that ends up empty, or two
    names that end up the same, are an error.
    """
    out = [_canon(n) for n in names]
    _check_names(list(names), out, label="names")
    return out


def _check_names(original: list, canon: list[str], label: str) -> None:
    empty = [o for o, c in zip(original, canon, strict=True) if not c]
    if empty:
        raise DiffError(f"Column name(s) of {label} are empty once normalized: {empty!r}.")
    groups: dict[str, list] = {}
    for o, c in zip(original, canon, strict=True):
        groups.setdefault(c, []).append(o)
    clashes = [g for g in groups.values() if len(g) > 1]
    if clashes:
        listed = "; ".join(_join(g) for g in clashes)
        raise DiffError(f"Column names of {label} clash once normalized: {listed}.")


# --- casting -----------------------------------------------------------------


def _is_float(s: pd.Series) -> bool:
    return pd.api.types.is_float_dtype(s)


def _is_int(s: pd.Series) -> bool:
    return pd.api.types.is_integer_dtype(s) and not pd.api.types.is_bool_dtype(s)


def _is_bool(s: pd.Series) -> bool:
    return pd.api.types.is_bool_dtype(s)


def _is_datetime(s: pd.Series) -> bool:
    return pd.api.types.is_datetime64_any_dtype(s)


def _is_text(s: pd.Series) -> bool:
    return pd.api.types.is_object_dtype(s) or isinstance(s.dtype, pd.StringDtype)


def _show(v) -> str:
    """A value as an error message shows it: strings quoted, numbers plain."""
    if isinstance(v, str):
        return repr(v)
    return str(v.item() if isinstance(v, np.generic) else v)


def _first_bad(original: pd.Series, converted: pd.Series):
    bad = original.notna() & converted.isna()
    return original[bad].iloc[0]


def _to_text(v):
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def _cast_to(y: pd.Series, x: pd.Series, name: str, truncate: bool, truncated: dict) -> pd.Series:
    """Cast column ``y`` to the dtype of the benchmark column ``x``."""
    if y.dtype == x.dtype:
        return y

    if _is_int(x) or _is_float(x):
        if _is_bool(y):
            num = y.astype(float)
        else:
            num = pd.to_numeric(y, errors="coerce")
            if (y.notna() & num.isna()).any():
                v = _first_bad(y, num)
                raise DiffError(f"Cannot cast `{name}` to a number: {_show(v)} is not one.")
            num = num.astype(float)
        if _is_float(x):
            return num.astype(x.dtype)
        frac = num.notna() & np.isfinite(num) & (num != np.trunc(num))
        if frac.any():
            if not truncate:
                raise DiffError(
                    f"Casting `{name}` to integer would truncate {int(frac.sum())} value(s), "
                    f"e.g. {_show(num[frac].iloc[0])}; pass truncate=True to allow it."
                )
            pairs = zip(num[frac].tolist(), np.trunc(num[frac]).tolist(), strict=True)
            truncated[name] = list(pairs)
        if (~np.isfinite(num.fillna(0))).any():
            raise DiffError(f"Cannot cast `{name}` to integer: it holds infinite values.")
        num = np.trunc(num)
        return num.astype("Int64" if num.isna().any() else x.dtype)

    if _is_bool(x):
        if _is_int(y) or _is_float(y):
            ok = y.isna() | y.isin([0, 1])
            if not ok.all():
                raise DiffError(
                    f"Cannot cast `{name}` to bool: {_show(y[~ok].iloc[0])} is not 0 or 1."
                )
            out = y.map(lambda v: v if pd.isna(v) else bool(v))
        else:
            low = y.map(lambda v: v if pd.isna(v) else str(v).strip().lower())
            out = low.map(lambda v: True if v in _TRUE else False if v in _FALSE else None)
            if (y.notna() & out.isna()).any():
                v = _first_bad(y, out)
                raise DiffError(f"Cannot cast `{name}` to bool: {_show(v)} is not true or false.")
        return out.astype("boolean" if out.isna().any() else x.dtype)

    if _is_datetime(x):
        conv = pd.to_datetime(y, errors="coerce")
        if (y.notna() & conv.isna()).any():
            v = _first_bad(y, conv)
            raise DiffError(f"Cannot cast `{name}` to a date-time: {_show(v)} is not one.")
        try:
            return conv.astype(x.dtype)
        except (TypeError, ValueError) as e:
            raise DiffError(f"Cannot cast `{name}` to {x.dtype}: {e}") from e

    if _is_text(x):
        return y.map(lambda v: v if pd.isna(v) else _to_text(v)).astype(x.dtype)

    raise DiffError(
        f"Cannot cast `{name}` from {y.dtype} to {x.dtype}: supported target types are "
        "integer, float, bool, string and date-time."
    )


# --- measures and identity ---------------------------------------------------


def _select_measures(x: pd.DataFrame, measures, x_names: dict) -> list[str]:
    if isinstance(measures, str) and measures in ("numeric", "numeric+integer"):
        if measures == "numeric":
            return [c for c in x.columns if _is_float(x[c])]
        return [c for c in x.columns if _is_float(x[c]) or _is_int(x[c])]
    given = [measures] if isinstance(measures, str) else list(measures)
    canon = [_canon(m) for m in given]
    missing = [g for g, c in zip(given, canon, strict=True) if c not in x.columns]
    if missing:
        raise DiffError(f"Measure column(s) not found in `x`: {_join(missing)}.")
    bad = [x_names[c] for c in canon if not (_is_float(x[c]) or _is_int(x[c]))]
    if bad:
        raise DiffError(f"Measure column(s) are not numeric: {_join(bad)}.")
    return list(dict.fromkeys(canon))


def _add_seq(df: pd.DataFrame, by: list[str], duplicates: str, label: str, alerts: list):
    seq = df.groupby(by, dropna=False, sort=False).cumcount() + 1
    n_dup = int((seq > 1).sum())
    if n_dup and duplicates != "ignore":
        if duplicates == "error":
            raise DiffError(f"`{label}` has {n_dup} row(s) with a repeated key.")
        alerts.append(
            (
                DiffWarning,
                f"`{label}` has {n_dup} row(s) with a repeated key; "
                "they are paired in order of arrival.",
            )
        )
    df["key_seq"] = seq.astype("int64").to_numpy()
    return df


# --- values ------------------------------------------------------------------


def _classify(vx: pd.Series, vy: pd.Series, tolerance: float, rel_tol: float):
    vx = vx.to_numpy(dtype=float, na_value=np.nan)
    vy = vy.to_numpy(dtype=float, na_value=np.nan)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        equal = vx == vy  # also equal infinities
        finite = np.isfinite(vx) & np.isfinite(vy)
        diff = np.where(equal, 0.0, vy - vx)
        abs_diff = np.abs(diff)
        scale = np.maximum(np.abs(vx), np.abs(vy))
        safe = np.where(scale > 0, scale, 1.0)
        rel = np.abs(vy / safe - vx / safe)  # scaled first, so it cannot overflow
        rel_diff = np.where(equal, 0.0, np.where(finite, rel, np.nan))
    both_na = np.isnan(vx) & np.isnan(vy)
    close = finite & ((abs_diff <= tolerance) | (rel_diff <= rel_tol))
    same = both_na | equal | close
    return diff, abs_diff, rel_diff, same


# --- diff_table --------------------------------------------------------------


def diff_table(
    x: pd.DataFrame,
    y: pd.DataFrame,
    *,
    mode: str = "benchmark",
    by: Sequence[str] | str | None = None,
    measures: Sequence[str] | str = "numeric",
    tolerance: float = 0.0,
    rel_tol: float = 0.0,
    duplicates: str = "pair",
    truncate: bool = True,
) -> pd.DataFrame:
    """Compare two tables value by value.

    Parameters
    ----------
    x, y
        The tables. In ``mode="benchmark"``, ``x`` is the benchmark: ``y`` is cast to
        its column types and ``y``'s extra columns are dropped. In ``mode="equal"`` the
        tables must have the same columns with the same types.
    mode
        ``"benchmark"`` (default) or ``"equal"``.
    by
        Key columns that identify a row. Defaults to every column that is not a measure.
        With ``by`` given, other non-measure columns are ignored with a note in
        benchmark mode, and are an error in equal mode.
    measures
        ``"numeric"`` (float columns), ``"numeric+integer"``, or column names.
    tolerance, rel_tol
        For a finite pair, a value is ``same`` when ``abs_diff <= tolerance`` or
        ``rel_diff <= rel_tol``.
    duplicates
        ``"pair"`` pairs repeated keys in arrival order (with a warning);
        ``"error"`` refuses them.
    truncate
        Allow truncating non-integral values when casting to an integer column.

    Returns
    -------
    pandas.DataFrame
        Sorted by the key. With measures (``attrs["shape"] == "cells"``), one row per
        key and measure, with ``metric``, ``row_id_x``, ``row_id_y``, ``value_x``,
        ``value_y``, ``diff``, ``abs_diff``, ``rel_diff`` and ``status``. With no
        measures (``"rows"``), one row per source row, with ``key_seq``. ``attrs``
        also holds ``mode``, ``by``, ``measures`` and ``truncated``.

    Raises
    ------
    DiffError
        When the question cannot be answered: missing columns, type differences in
        equal mode, clashing names, no key, or no key in common.

    Warns
    -----
    DiffWarning
        When the answer may be wrong: repeated keys paired by arrival order.
    DiffNote
        For expected side effects: dropped or ignored columns, truncation, no measures.
    """
    if mode not in _MODES:
        raise DiffError(f"`mode` must be one of {_MODES}, not {mode!r}.")
    if duplicates not in _DUPLICATES:
        raise DiffError(f"`duplicates` must be one of {_DUPLICATES}, not {duplicates!r}.")
    if not (tolerance >= 0 and rel_tol >= 0):
        raise DiffError("`tolerance` and `rel_tol` must be non-negative numbers.")

    alerts: list[tuple[type[Warning], str]] = []
    try:
        out = _diff_table(
            x, y, mode, by, measures, tolerance, rel_tol, duplicates, truncate, alerts
        )
    finally:
        for category, message in alerts:
            warnings.warn(message, category, stacklevel=2)
    return out


def _diff_table(x, y, mode, by, measures, tolerance, rel_tol, duplicates, truncate, alerts):
    # Names: x is checked whole; y only for the columns that are compared.
    x_canon = [_canon(c) for c in x.columns]
    _check_names(list(x.columns), x_canon, "`x`")
    y_canon = [_canon(c) for c in y.columns]
    if mode == "equal":
        _check_names(list(y.columns), y_canon, "`y`")
    else:
        used = [(o, c) for o, c in zip(y.columns, y_canon, strict=True) if c in set(x_canon)]
        _check_names([o for o, _ in used], [c for _, c in used], "`y`")
    x_names = dict(zip(x_canon, x.columns, strict=True))
    y_names = {c: o for o, c in zip(y.columns, y_canon, strict=True) if c}

    x = x.copy()
    x.columns = x_canon
    x = x.reset_index(drop=True)
    y_keep = [i for i, c in enumerate(y_canon) if c in y_names and y_names[c] == y.columns[i]]
    y = y.iloc[:, y_keep].copy()
    y.columns = [y_canon[i] for i in y_keep]
    y = y.reset_index(drop=True)

    only_x = [c for c in x.columns if c not in y.columns]
    only_y = [c for c in y.columns if c not in x.columns]
    truncated: dict = {}

    if mode == "equal":
        if only_x or only_y:
            raise DiffError(
                'mode = "equal" needs the same columns in both tables.\n'
                f"  only in `x`: {_join(x_names[c] for c in only_x) or '<none>'}\n"
                f"  only in `y`: {_join(y_names[c] for c in only_y) or '<none>'}"
            )
        y = y[list(x.columns)]
        differ = [
            f"{x_names[c]} ({x[c].dtype} vs {y[c].dtype})"
            for c in x.columns
            if x[c].dtype != y[c].dtype
        ]
        if differ:
            raise DiffError('mode = "equal" needs the same column types: ' + ", ".join(differ))
    else:
        if only_x:
            missing = _join(x_names[c] for c in only_x)
            raise DiffError(f"`y` lacks column(s) of the benchmark `x`: {missing}.")
        dropped = [y_names.get(c, c) for c in only_y]
        if only_y:
            alerts.append(
                (
                    DiffNote,
                    f"Dropped column(s) of `y` that the benchmark `x` does not have: "
                    f"{_join(dropped)}.",
                )
            )
        y = pd.DataFrame(
            {c: _cast_to(y[c], x[c], y_names[c], truncate, truncated) for c in x.columns}
        )
        if truncated:
            lines = [f"  {c}: {len(v)} ({v[0][0]:g} -> {v[0][1]:g})" for c, v in truncated.items()]
            n = sum(len(v) for v in truncated.values())
            alerts.append(
                (
                    DiffNote,
                    f"Truncated {n} value(s) while casting to the reference types:\n"
                    + "\n".join(lines),
                )
            )

    if by is not None:
        given = [by] if isinstance(by, str) else list(by)
        by = list(dict.fromkeys(_canon(b) for b in given))
        missing = [g for g in given if _canon(g) not in x.columns]
        if missing:
            raise DiffError(f"Key column(s) not found in `x`: {_join(missing)}.")
        meas = [m for m in _select_measures(x, measures, x_names) if m not in by]
        rest = [c for c in x.columns if c not in by and c not in meas]
        if rest:
            if mode == "equal":
                raise DiffError(
                    f"Column(s) that are neither keys nor measures: {_join(x_names[c] for c in rest)}. "
                    'mode = "equal" compares every column; add them to `by` or `measures`.'
                )
            alerts.append(
                (
                    DiffNote,
                    f"Ignored column(s) that are neither keys nor measures: "
                    f"{_join(x_names[c] for c in rest)}.",
                )
            )
    else:
        meas = _select_measures(x, measures, x_names)
        by = [c for c in x.columns if c not in meas]
    if not by:
        raise DiffError(
            "Every column is a measure, so rows have no key; pass `by` or fewer `measures`."
        )

    if not meas:
        alerts.append(
            (
                DiffNote,
                "No measure columns, so rows are compared as wholes: "
                "do the two tables have the same rows?",
            )
        )

    # With no measures a row is its own key, so repeats are copies, not clashes.
    dup_rule = duplicates if meas else "ignore"
    x_names_y = {c: y_names.get(c) for c in x.columns}
    x = _add_seq(x[by + meas].copy(), by, dup_rule, "x", alerts)
    y = _add_seq(y[by + meas].copy(), by, dup_rule, "y", alerts)
    keys = [*by, "key_seq"]
    x["row_id_x"] = np.arange(1, len(x) + 1)
    y["row_id_y"] = np.arange(1, len(y) + 1)

    m = x.merge(y, on=keys, how="outer", suffixes=("_x", "_y"), sort=False)
    matched = m["row_id_x"].notna() & m["row_id_y"].notna()
    if len(x) and len(y) and not matched.any():
        raise DiffError(
            f"No row of `x` has a key in common with `y` (key: {_join(x_names[c] for c in by)})."
        )
    for c in ("row_id_x", "row_id_y"):
        m[c] = m[c].astype("Int64")
    only_in_y, only_in_x = m["row_id_x"].isna().to_numpy(), m["row_id_y"].isna().to_numpy()

    if not meas:
        status = np.select([only_in_y, only_in_x], ["only_y", "only_x"], "same")
        out = m[[*keys, "row_id_x", "row_id_y"]].copy()
        out["status"] = pd.Categorical(status, categories=STATUSES)
        sort = keys
        shape = "rows"
    else:
        parts = []
        for metric in meas:
            p = m[[*keys, "row_id_x", "row_id_y"]].copy()
            p.insert(len(keys), "metric", metric)
            vx, vy = m[f"{metric}_x"], m[f"{metric}_y"]
            diff, abs_diff, rel_diff, same = _classify(vx, vy, tolerance, rel_tol)
            p["value_x"] = vx.to_numpy(dtype=float, na_value=np.nan)
            p["value_y"] = vy.to_numpy(dtype=float, na_value=np.nan)
            p["diff"], p["abs_diff"], p["rel_diff"] = diff, abs_diff, rel_diff
            status = np.select(
                [only_in_y, only_in_x, same], ["only_y", "only_x", "same"], "changed"
            )
            p["status"] = pd.Categorical(status, categories=STATUSES)
            parts.append(p)
        out = pd.concat(parts, ignore_index=True).drop(columns="key_seq")
        sort = [*by, "metric"]
        shape = "cells"

    out = out.sort_values(sort, kind="stable", ignore_index=True, na_position="last")
    out.attrs.update(
        {
            "shape": shape,
            "mode": mode,
            "by": by,
            "measures": meas,
            "truncated": {
                c: truncated[x_names_y[c]] for c in by + meas if x_names_y.get(c) in truncated
            },
        }
    )
    return out


def diff_summary(d: pd.DataFrame) -> pd.DataFrame:
    """Condense a :func:`diff_table` result per measure."""
    groups = d["metric"] if "metric" in d else pd.Series("<rows>", index=d.index, name="metric")
    counts = pd.crosstab(groups, d["status"]).reindex(columns=STATUSES, fill_value=0)
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
    counts.index.name = "metric"
    return counts.reset_index()


def is_equal(d: pd.DataFrame) -> bool:
    """``True`` when every row of a :func:`diff_table` result is ``same``."""
    return bool((d["status"] == "same").all())
