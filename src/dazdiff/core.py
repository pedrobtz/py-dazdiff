"""Compare two tables value by value."""

from __future__ import annotations

import re
import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd

STATUSES = ["same", "changed", "only_x", "only_y"]
_MODES = ("benchmark", "equal")
_DUPLICATES = ("disambiguate", "aggregate", "error")
_ROW_KEYS = ("none", "position", "sorted")
_MEASURE_KEYWORDS = ("numeric", "numeric+integer")
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
        return str(int(v + 0.0))  # + 0.0 writes -0.0 as "0"
    return str(v)


def _examples(values, n: int = 3) -> str:
    shown = [_show(v) if not pd.isna(v) else "NA" for v in list(values)[:n]]
    return ", ".join(shown) + (", ..." if len(values) > n else "")


_PLAIN_DECIMAL = re.compile(r"-?(0|[1-9][0-9]*)(\.[0-9]+)?")


def _text_decimals(ref: pd.Series) -> tuple[int | None, list[int], bool]:
    """The one number of decimals every value of the text column ``ref`` uses.

    Read off every distinct value, not a sample: one ``"1.00"`` among a million
    ``"1.0"`` is the value a sample misses. Only plain decimal text counts; a sign,
    leading zeros, separators or an exponent mean there is no format to follow.
    Returns the decimals (or None), the counts of decimals seen, and whether every
    value is plain decimal text.
    """
    u = [str(v) for v in pd.unique(ref.dropna())]
    plain = [bool(_PLAIN_DECIMAL.fullmatch(v)) for v in u]
    seen = sorted({len(v.partition(".")[2]) for v, p in zip(u, plain, strict=True) if p})
    k = seen[0] if u and all(plain) and len(seen) == 1 else None
    return k, seen, all(plain)


def _number_as_text_like(v: pd.Series, out: pd.Series, ref: pd.Series, name: str, alerts: list):
    """Write ``y``'s numbers ``v`` (already text in ``out``) as the benchmark ``ref``
    writes them, where every value of ``ref`` has the same number of decimals.

    A value is reformatted only when its text reads back as the same number: 1.25 at
    one decimal would become "1.2" and join the benchmark's 1.2, a silent false
    match, so it keeps its own text and shows as unmatched instead.
    """
    k, seen, plain = _text_decimals(ref)
    num = v.to_numpy(dtype=float, na_value=np.nan)
    finite = np.isfinite(num)
    fits = np.zeros(len(v), dtype=bool)
    kept = np.zeros(len(v), dtype=bool)
    out = out.copy()
    if k:
        txt = np.array([f"{n + 0.0:.{k}f}" if f else "" for n, f in zip(num, finite, strict=True)])
        fits = finite & np.array(
            [f and float(t) == n for t, n, f in zip(txt, num, finite, strict=True)]
        )
        kept = finite & ~fits
        out[fits] = txt[fits]

    # Values of x that are numbers y has, written differently: those rows will not align.
    first_text = {}
    for n, t in zip(num, out, strict=True):
        if np.isfinite(n):
            first_text.setdefault(n + 0.0, t)
    have = set(out.dropna())
    u = [str(s) for s in pd.unique(ref.dropna())]
    u_num = pd.to_numeric(pd.Series(u, dtype=object), errors="coerce").to_numpy(dtype=float)
    skew = [
        (s, first_text[n + 0.0])
        for s, n in zip(u, u_num, strict=True)
        if np.isfinite(n) and n + 0.0 in first_text and s not in have
    ]
    if skew:
        pairs = ", ".join(f"{a!r} vs {b!r}" for a, b in skew[:3]) + (
            ", ..." if len(skew) > 3 else ""
        )
        if k:
            why = f"`y`'s numbers were written with {k} decimal place{'s' if k > 1 else ''}, as `x` writes them"
        elif not plain:
            why = "no decimal places were inferred, since not every value of `x` is plain decimal text"
        elif len(seen) > 1:
            why = f"no decimal places were inferred, since `x` writes them with {_join(seen)} decimal places"
        else:
            why = "`x` writes them with no decimal places"
        alerts.append(
            (
                DiffWarning,
                f"Column `{name}`: {len(skew)} value(s) of `x` equal a number of `y` but not its "
                f"text ({pairs}), so those rows will not align; {why}; cast the column of `x` "
                "to a number to compare by value.",
            )
        )
    elif fits.any():
        firsts = pd.Series(num[fits]).drop_duplicates().index[:3]
        shown = [(num[fits][i], out[fits].iloc[i]) for i in firsts]
        listed = ", ".join(f"{n:g} -> {t!r}" for n, t in shown)
        more = ", ..." if int(fits.sum()) > len(shown) else ""
        extra = ""
        if kept.any():
            kept_text = ", ".join(repr(t) for t in pd.unique(out[kept])[:3])
            extra = f"; {int(kept.sum())} value(s) needed more decimals and kept their own digits ({kept_text})"
        alerts.append(
            (
                DiffNote,
                f"Wrote column `{name}` of `y` as text with {k} decimal place{'s' if k > 1 else ''}, "
                f"as `x` writes it: {int(fits.sum())} value(s) ({listed}{more}){extra}.",
            )
        )
    return out


def _stamp_to_text(y: pd.Series) -> pd.Series:
    """Date-times as ``%Y-%m-%d %H:%M:%S``, midnight included, with the fraction of a
    second to the microsecond, without trailing zeros, when there is one."""
    s = y.dt.round("us")
    txt = s.dt.strftime("%Y-%m-%d %H:%M:%S")
    us = s.dt.microsecond
    frac = s.notna() & (us > 0)
    txt[frac] = txt[frac] + "." + us[frac].map(lambda u: f"{int(u):06d}".rstrip("0"))
    return txt


def _cast_to(
    y: pd.Series, x: pd.Series, name: str, truncate: bool, truncated: dict, alerts: list
) -> pd.Series:
    """Cast column ``y`` to the dtype of the benchmark column ``x``."""
    if y.dtype == x.dtype:
        return y

    # A column with no values (how a reader types an empty column) casts to anything.
    if (
        len(y)
        and y.isna().all()
        and (_is_int(x) or _is_float(x) or _is_bool(x) or _is_datetime(x) or _is_text(x))
    ):
        target = "Int64" if _is_int(x) else "boolean" if _is_bool(x) else x.dtype
        missing = np.nan if pd.api.types.is_object_dtype(x) or _is_float(x) else pd.NA
        return pd.Series([missing] * len(y), index=y.index, dtype=object).astype(target)

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
        if _is_datetime(y):
            return _stamp_to_text(y).astype(x.dtype)
        out = y.map(lambda v: v if pd.isna(v) else _to_text(v)).astype(object)
        if _is_int(y) or _is_float(y):
            out = _number_as_text_like(y, out, x, name, alerts)
        return out.astype(x.dtype)

    raise DiffError(
        f"Cannot cast `{name}` from {y.dtype} to {x.dtype}: supported target types are "
        "integer, float, bool, string and date-time."
    )


# --- measures and identity ---------------------------------------------------


def _select_measures(x: pd.DataFrame, measures, x_names: dict) -> list[str]:
    if isinstance(measures, str) and measures in _MEASURE_KEYWORDS:
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
    exclude: Sequence[str] | str | None = None,
    row_key: str = "none",
    measures: Sequence[str] | str = "numeric",
    tolerance: float = 0.0,
    rel_tol: float = 0.0,
    duplicates: str = "disambiguate",
    cast: bool | None = None,
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
    exclude
        Columns, in either spelling, removed from both tables before anything else, in
        either mode. They need not be in both tables and are not announced as dropped
        or ignored. A name in neither table is a :class:`DiffWarning`; a name ``by`` or
        ``measures`` also names is an error.
    row_key
        ``"none"`` (default), or a virtual ``row_number`` key added to the identity:
        ``"position"`` numbers rows as they come, ``"sorted"`` numbers them after
        sorting both tables by their compared columns. Use it when every column is a
        measure. It cannot be combined with ``by``.
    measures
        ``"numeric"`` (float columns), ``"numeric+integer"``, or column names.
    tolerance, rel_tol
        For a finite pair, a value is ``same`` when ``abs_diff <= tolerance`` or
        ``rel_diff <= rel_tol``.
    duplicates
        ``"disambiguate"`` (default) pairs repeated keys in arrival order, with a
        warning; ``"aggregate"`` compares the multiset of values per key and measure
        (tolerances must be 0); ``"error"`` refuses repeated keys.
    cast
        Whether to cast the compared columns of ``y`` to ``x``'s types. ``None``
        (default) casts in benchmark mode; ``False`` requires the types to agree.
        Equal mode never casts, so ``cast=True`` with it is an error. A number cast
        to text is written with the decimals ``x`` uses, when every value of ``x``'s
        column uses the same number of them.
    truncate
        Allow truncating non-integral values when casting to an integer column.

    Returns
    -------
    pandas.DataFrame
        Sorted by the key. With measures (``attrs["shape"] == "cells"``), one row per
        key and measure, with ``metric``, ``row_id_x``, ``row_id_y``, ``value_x``,
        ``value_y``, ``diff``, ``abs_diff``, ``rel_diff`` and ``status``. With
        ``duplicates="aggregate"`` (``"groups"``), one row per key and measure, with
        ``n_rows_x``, ``n_rows_y``, ``values_x`` and ``values_y`` (sorted tuples). With no
        measures (``"rows"``), one row per source row, with ``key_seq``. ``attrs``
        also holds ``mode``, ``by``, ``measures``, ``tolerance``, ``name_map``,
        ``dropped``, ``excluded``, ``ignored`` and ``truncated``.

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
    if row_key not in _ROW_KEYS:
        raise DiffError(f"`row_key` must be one of {_ROW_KEYS}, not {row_key!r}.")
    if cast is not None and not isinstance(cast, bool):
        raise DiffError("`cast` must be None, True or False.")
    if mode == "equal" and cast:
        raise DiffError('mode = "equal" never casts: neither table is the reference for the other.')
    if cast is None:
        cast = mode == "benchmark"
    if row_key != "none" and by is not None:
        raise DiffError("`row_key` adds a key of its own, so it cannot be combined with `by`.")
    if not (tolerance >= 0 and rel_tol >= 0):
        raise DiffError("`tolerance` and `rel_tol` must be non-negative numbers.")
    if duplicates == "aggregate" and (tolerance or rel_tol):
        raise DiffError(
            'duplicates = "aggregate" compares multisets exactly; `tolerance` and `rel_tol` '
            "must be 0."
        )

    alerts: list[tuple[type[Warning], str]] = []
    try:
        out = _diff_table(
            x,
            y,
            mode,
            by,
            exclude,
            row_key,
            measures,
            tolerance,
            rel_tol,
            duplicates,
            cast,
            truncate,
            alerts,
        )
    finally:
        for category, message in alerts:
            warnings.warn(message, category, stacklevel=2)
    return out


def _exclude(x, y, exclude, by, measures, alerts):
    """Remove the ``exclude`` columns from both tables, before anything else looks at
    them: they are neither required, dropped with a note, nor checked for clashes."""
    given = [exclude] if isinstance(exclude, str) else list(exclude)
    canon = [_canon(e) for e in given]
    cols_x = [_canon(c) for c in x.columns]
    cols_y = [_canon(c) for c in y.columns]
    # A warning, not an error: one list of technical columns (load stamps, run ids)
    # serves many tables, not all of which have each one. Still a warning, since a
    # typo leaves a column compared.
    unknown = [g for g, c in zip(given, canon, strict=True) if c not in cols_x + cols_y]
    if unknown:
        alerts.append(
            (
                DiffWarning,
                f"`exclude` names column(s) in neither table, so nothing was excluded for "
                f"them: {_join(unknown)}.",
            )
        )
    named = [] if by is None else [by] if isinstance(by, str) else list(by)
    if not (isinstance(measures, str) and measures in _MEASURE_KEYWORDS):
        named += [measures] if isinstance(measures, str) else list(measures)
    named_canon = {_canon(n) for n in named}
    both = [g for g, c in zip(given, canon, strict=True) if c in named_canon]
    if both:
        raise DiffError(
            f"`exclude` names column(s) that `by` or `measures` also name: {_join(both)}."
        )
    drop = set(canon)
    excluded = list(
        dict.fromkeys(
            [o for o, c in zip(x.columns, cols_x, strict=True) if c in drop]
            + [o for o, c in zip(y.columns, cols_y, strict=True) if c in drop]
        )
    )
    x = x.iloc[:, [i for i, c in enumerate(cols_x) if c not in drop]]
    y = y.iloc[:, [i for i, c in enumerate(cols_y) if c not in drop]]
    return x, y, excluded


def _diff_table(
    x,
    y,
    mode,
    by,
    exclude,
    row_key,
    measures,
    tolerance,
    rel_tol,
    duplicates,
    cast,
    truncate,
    alerts,
):
    excluded: list = []
    if exclude is not None:
        x, y, excluded = _exclude(x, y, exclude, by, measures, alerts)

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
    name_map = {c: {"x": o, "y": y_names.get(c)} for c, o in x_names.items()}
    name_map.update({c: {"x": None, "y": o} for c, o in y_names.items() if c not in x_names})

    x = x.copy()
    x.columns = x_canon
    x = x.reset_index(drop=True)
    y_keep = [i for i, c in enumerate(y_canon) if c in y_names and y_names[c] == y.columns[i]]
    y = y.iloc[:, y_keep].copy()
    y.columns = [y_canon[i] for i in y_keep]
    y = y.reset_index(drop=True)

    only_x = [c for c in x.columns if c not in y.columns]
    only_y = [c for c in y.columns if c not in x.columns]
    dropped: list = []

    if mode == "equal":
        if only_x or only_y:
            raise DiffError(
                'mode = "equal" needs the same columns in both tables.\n'
                f"  only in `x`: {_join(x_names[c] for c in only_x) or '<none>'}\n"
                f"  only in `y`: {_join(y_names[c] for c in only_y) or '<none>'}"
            )
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
    y = y[list(x.columns)]

    # Roles are decided on x, whose types a cast never changes, and before casting,
    # so a column the call ignores can neither refuse the comparison nor be truncated.
    rest: list = []
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
    compared = by + meas

    # Types, of the compared columns only.
    truncated: dict = {}
    if cast:
        y = y.assign(
            **{c: _cast_to(y[c], x[c], y_names[c], truncate, truncated, alerts) for c in compared}
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
    else:
        differ = [
            f"{x_names[c]} ({x[c].dtype} vs {y[c].dtype})"
            for c in compared
            if x[c].dtype != y[c].dtype
        ]
        if differ:
            lead = (
                'mode = "equal" needs the same column types'
                if mode == "equal"
                else ("cast=False, but x and y disagree on the type of")
            )
            raise DiffError(f"{lead}: " + ", ".join(differ))

    if row_key != "none":
        x = x.assign(row_number=_row_numbers(x, row_key, compared))
        y = y.assign(row_number=_row_numbers(y, row_key, compared))
        x_names["row_number"] = "row_number"
        by = [*by, "row_number"]
    if not by:
        raise DiffError(
            "Every column is a measure, so rows have no key; pass `by`, fewer `measures`, "
            'or `row_key="position"` or `"sorted"`.'
        )

    if not meas:
        alerts.append(
            (
                DiffNote,
                "No measure columns, so rows are compared as wholes: "
                "do the two tables have the same rows?",
            )
        )

    x_names_y = {c: y_names.get(c) for c in x.columns}
    x, y = x[by + meas].copy(), y[by + meas].copy()
    if duplicates == "aggregate" and meas:
        out = _aggregate(x, y, by, meas, x_names)
        sort, shape = [*by, "metric"], "groups"
    else:
        # With no measures a row is its own key, so repeats are copies, not clashes,
        # unless the caller asked for repeats to be refused.
        dup_rule = duplicates if meas or duplicates == "error" else "ignore"
        out, sort, shape = _pairwise(x, y, by, meas, dup_rule, tolerance, rel_tol, x_names, alerts)

    out = out.sort_values(sort, kind="stable", ignore_index=True, na_position="last")
    out.attrs.update(
        {
            "shape": shape,
            "mode": mode,
            "by": by,
            "measures": meas,
            "tolerance": {"abs": float(tolerance), "rel": float(rel_tol)},
            "name_map": name_map,
            "dropped": dropped,
            "excluded": excluded,
            "ignored": [x_names[c] for c in rest],
            "truncated": {
                c: truncated[x_names_y[c]] for c in by + meas if x_names_y.get(c) in truncated
            },
        }
    )
    return out


def _row_numbers(df: pd.DataFrame, row_key: str, compared: list[str]) -> np.ndarray:
    n = len(df)
    if row_key == "position" or n == 0:
        return np.arange(1, n + 1)
    order = df.sort_values(compared, kind="stable", na_position="last").index.to_numpy()
    numbers = np.empty(n, dtype="int64")
    numbers[order] = np.arange(1, n + 1)
    return numbers


def _no_overlap(x: pd.DataFrame, y: pd.DataFrame, x_names: dict, by: list[str]) -> DiffError:
    """The error for tables that share no key, naming the key columns that alone share
    no value: most often one that labels the table (a run id, a source) and belongs in
    ``exclude``."""
    msg = (
        f"No row of `x` has a key in common with `y` (key: {_join(x_names[c] for c in by)}); "
        "check `by`, and the format of the key values"
    )
    apart, constant = [], []
    many = max(len(x), len(y)) > 1
    for c in by:
        ux, uy = pd.Series(pd.unique(x[c])), pd.Series(pd.unique(y[c]))
        if not ux.isin(uy).any():
            apart.append(f"{x_names[c]} (x: {_examples(ux)}; y: {_examples(uy)})")
            if many and len(ux) == 1 and len(uy) == 1:
                constant.append(x_names[c])
    if apart:
        msg += f"; key column(s) that share no value between the tables: {', '.join(apart)}"
    # Offered only when a key is left once they are excluded.
    if constant and len(constant) < len(by):
        each = "each hold" if len(constant) > 1 else "holds"
        msg += (
            f"; {_join(constant)} {each} one value in each table, so "
            f"{'they label' if len(constant) > 1 else 'it labels'} the table rather than its "
            f"rows: leave {'them' if len(constant) > 1 else 'it'} out with exclude={constant!r}"
        )
    return DiffError(msg + ".")


def _pairwise(x, y, by, meas, dup_rule, tolerance, rel_tol, x_names, alerts):
    x = _add_seq(x, by, dup_rule, "x", alerts)
    y = _add_seq(y, by, dup_rule, "y", alerts)
    keys = [*by, "key_seq"]
    x["row_id_x"] = np.arange(1, len(x) + 1)
    y["row_id_y"] = np.arange(1, len(y) + 1)

    m = x.merge(y, on=keys, how="outer", suffixes=("_x", "_y"), sort=False)
    matched = m["row_id_x"].notna() & m["row_id_y"].notna()
    if len(x) and len(y) and not matched.any():
        raise _no_overlap(x, y, x_names, by)
    for c in ("row_id_x", "row_id_y"):
        m[c] = m[c].astype("Int64")
    only_in_y, only_in_x = m["row_id_x"].isna().to_numpy(), m["row_id_y"].isna().to_numpy()

    if not meas:
        status = np.select([only_in_y, only_in_x], ["only_y", "only_x"], "same")
        out = m[[*keys, "row_id_x", "row_id_y"]].copy()
        out["status"] = pd.Categorical(status, categories=STATUSES)
        return out, keys, "rows"

    parts = []
    for metric in meas:
        p = m[[*keys, "row_id_x", "row_id_y"]].copy()
        p.insert(len(keys), "metric", metric)
        vx, vy = m[f"{metric}_x"], m[f"{metric}_y"]
        diff, abs_diff, rel_diff, same = _classify(vx, vy, tolerance, rel_tol)
        p["value_x"] = vx.to_numpy(dtype=float, na_value=np.nan)
        p["value_y"] = vy.to_numpy(dtype=float, na_value=np.nan)
        p["diff"], p["abs_diff"], p["rel_diff"] = diff, abs_diff, rel_diff
        status = np.select([only_in_y, only_in_x, same], ["only_y", "only_x", "same"], "changed")
        p["status"] = pd.Categorical(status, categories=STATUSES)
        parts.append(p)
    out = pd.concat(parts, ignore_index=True).drop(columns="key_seq")
    return out, [*by, "metric"], "cells"


def _multiset(values: pd.Series) -> tuple:
    """The values of one group as a sorted tuple; missing values last, as NaN."""
    v = values.to_numpy(dtype=float, na_value=np.nan)
    v = np.where(v == 0, 0.0, v)  # -0.0 and 0.0 are the same value
    return tuple(np.sort(v).tolist())  # np.sort puts NaN last


def _same_multiset(a: tuple, b: tuple) -> bool:
    if len(a) != len(b):
        return False
    return all(u == v or (u != u and v != v) for u, v in zip(a, b, strict=True))


def _aggregate(x, y, by, meas, x_names):
    def groups(df, side):
        g = df.groupby(by, dropna=False, sort=False)
        out = g[meas].agg(_multiset)
        out.columns = [f"{m}_{side}" for m in meas]
        out.insert(0, f"n_rows_{side}", g.size())
        return out.reset_index()

    gx, gy = groups(x, "x"), groups(y, "y")
    m = gx.merge(gy, on=by, how="outer", sort=False)
    in_x, in_y = m["n_rows_x"].notna().to_numpy(), m["n_rows_y"].notna().to_numpy()
    if len(x) and len(y) and not (in_x & in_y).any():
        raise _no_overlap(x, y, x_names, by)
    for c in ("n_rows_x", "n_rows_y"):
        m[c] = m[c].astype("Int64")

    parts = []
    for metric in meas:
        p = m[[*by, "n_rows_x", "n_rows_y"]].copy()
        p.insert(len(by), "metric", metric)
        vx, vy = m[f"{metric}_x"], m[f"{metric}_y"]
        p["values_x"], p["values_y"] = vx.to_numpy(), vy.to_numpy()
        same = np.array(
            [a == a and b == b and _same_multiset(a, b) for a, b in zip(vx, vy, strict=True)],
            dtype=bool,
        )
        status = np.select([~in_x, ~in_y, same], ["only_y", "only_x", "same"], "changed")
        p["status"] = pd.Categorical(status, categories=STATUSES)
        parts.append(p)
    return pd.concat(parts, ignore_index=True)


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


def is_matching(
    d: pd.DataFrame, *, tolerance: float | None = None, rel_tol: float | None = None
) -> bool:
    """``True`` when every row of a :func:`diff_table` result is ``same``: no value
    changed and no row is found in only one table. Two empty tables match.

    With ``tolerance`` or ``rel_tol`` the values are judged again at those thresholds,
    looser or stricter, from the stored ``abs_diff`` and ``rel_diff``, by
    :func:`diff_table`'s rule: equal or both-missing values always match, and the
    tolerances apply to finite pairs only. When only one is given, the other is 0.
    A row found in only one table never matches.
    """
    for arg, v in (("tolerance", tolerance), ("rel_tol", rel_tol)):
        ok = v is None or (
            isinstance(v, (int, float, np.integer, np.floating))
            and not isinstance(v, bool)
            and np.isfinite(v)
            and v >= 0
        )
        if not ok:
            raise DiffError(f"`{arg}` must be None or a single finite number >= 0.")
    status = d["status"].astype(str).to_numpy()
    if tolerance is None and rel_tol is None:
        return bool((status == "same").all())
    tolerance = 0.0 if tolerance is None else float(tolerance)
    rel_tol = 0.0 if rel_tol is None else float(rel_tol)

    shape = d.attrs.get("shape", "cells" if "abs_diff" in d else None)
    # The multisets of "aggregate" are compared exactly and hold no difference to
    # re-judge, as diff_table() refuses a tolerance with them.
    if shape == "groups" and (tolerance > 0 or rel_tol > 0):
        raise DiffError(
            'A result of duplicates = "aggregate" compares values exactly; '
            "`tolerance` and `rel_tol` must be 0."
        )
    if shape != "cells":
        return bool((status == "same").all())

    joined = np.isin(status, ["same", "changed"])
    # rel_diff is defined where both values are finite, which is where the
    # tolerances apply. Elsewhere a value matched by equality or by being missing on
    # both sides, which no threshold changes.
    abs_diff = d["abs_diff"].to_numpy(dtype=float, na_value=np.nan)
    rel_diff = d["rel_diff"].to_numpy(dtype=float, na_value=np.nan)
    finite = ~np.isnan(rel_diff)
    with np.errstate(invalid="ignore"):
        close = (abs_diff <= tolerance) | (rel_diff <= rel_tol)
    same = joined & np.where(finite, close, status == "same")
    return bool(same.all())


# --- diff_explain -----------------------------------------------------------------

_AMONG = ("all", "joined")
_OVERVIEW = ["column", "n_values", "same_only", "differ_only", "mixed", "separates", "identifies"]


def diff_explain(d: pd.DataFrame, column: str | None = None, *, among: str = "all") -> pd.DataFrame:
    """Which values of the key columns separate the rows that match from those that
    do not.

    With no ``column``, one row per key column (and per ``metric`` when the result
    has measures): ``column``, ``n_values``, and how many of its values are
    ``same_only`` (every record holding it matches), ``differ_only`` (none does) or
    ``mixed``. A column ``separates`` when no value is mixed; one with a different
    value on every record ``identifies`` and is ranked last. The others come first,
    fewest mixed values first, then fewest values.

    With a ``column`` (a key column in either spelling, or ``"metric"``), one row per
    value of it: ``n``, ``n_same``, ``n_changed``, ``n_only_x``, ``n_only_y`` and
    ``class``, most differences first.

    For a key column the unit is the row, which matches only when every one of its
    measures does; with ``duplicates="aggregate"`` it is the key; for ``metric`` it
    is the value. ``among="joined"`` counts only rows found in both tables, so the
    explanation is of the changed values alone.
    """
    if not isinstance(d, pd.DataFrame) or "status" not in d:
        raise DiffError("`d` must be a result of diff_table().")
    if among not in _AMONG:
        raise DiffError(f"`among` must be one of {_AMONG}, not {among!r}.")
    # User columns are normalized to upper case; internal ones hold a lowercase
    # letter, and row_number is the one internal key. Read off the names, so a
    # filtered or re-sorted result still explains.
    ids = [c for c in d.columns if not re.search("[a-z]", str(c)) or c == "row_number"]
    has_metric = "metric" in d
    candidates = ids + (["metric"] if has_metric else [])

    if column is not None:
        if not isinstance(column, str):
            raise DiffError("`column` must be None or a single column name.")
        given = column
        column = column if column in candidates else _canon(column)
        if column not in candidates:
            raise DiffError(
                "`column` must be a key column of the result"
                + (' or "metric"' if has_metric else "")
                + f"; got {given}; the key columns are {_join(ids)}."
            )

    shape = d.attrs.get("shape") or ("groups" if "n_rows_x" in d else "cells")
    units = _explain_units(d, ids, shape)
    cells = (
        pd.DataFrame({"metric": d["metric"].astype(str), "status": d["status"].astype(str)})
        if has_metric
        else None
    )
    if among == "joined":
        units = units[units["status"].isin(["same", "changed"])]
        if cells is not None:
            cells = cells[cells["status"].isin(["same", "changed"])]

    def records(col):
        return cells if col == "metric" else units

    if column is not None:
        return _explain_values(records(column), column)

    rows = []
    for col in candidates:
        r = records(col)
        v = _explain_values(r, col)
        rows.append(
            {
                "column": col,
                "n_values": len(v),
                "same_only": int((v["class"] == "same").sum()),
                "differ_only": int((v["class"] == "differ").sum()),
                "mixed": int((v["class"] == "mixed").sum()),
                "separates": not (v["class"] == "mixed").any(),
                # metric is never an identifier: it repeats on every row.
                "identifies": col != "metric" and len(v) > 1 and len(v) == len(r),
            }
        )
    out = pd.DataFrame(rows, columns=_OVERVIEW)
    out = out.astype(
        {
            "n_values": "int64",
            "same_only": "int64",
            "differ_only": "int64",
            "mixed": "int64",
            "separates": bool,
            "identifies": bool,
        }
    )
    return out.sort_values(["identifies", "mixed", "n_values"], kind="stable", ignore_index=True)


def _explain_units(d: pd.DataFrame, ids: list[str], shape: str) -> pd.DataFrame:
    """One record per row (per key with ``"groups"``), with the status of the row as
    a whole: ``only_x`` or ``only_y`` by membership, which comes from ``row_id`` or
    ``n_rows`` and never from the value; otherwise ``changed`` when any of its
    measures changed."""
    present = "n_rows" if shape == "groups" else "row_id"
    # A row is its pair of source rows, which tells paired duplicates apart.
    unit = ids if shape == "groups" else ["row_id_x", "row_id_y"]
    w = d[list(dict.fromkeys(ids + unit))].copy()
    w["_in_x"] = d[f"{present}_x"].notna().to_numpy()
    w["_in_y"] = d[f"{present}_y"].notna().to_numpy()
    w["_changed"] = (d["status"].astype(str) == "changed").to_numpy()
    if len(w):
        agg = {c: "first" for c in ids if c not in unit}
        agg.update({"_in_x": "max", "_in_y": "max", "_changed": "max"})
        w = w.groupby(unit, dropna=False, sort=False).agg(agg).reset_index()
    status = np.select(
        [~w["_in_y"].astype(bool), ~w["_in_x"].astype(bool), w["_changed"].astype(bool)],
        ["only_x", "only_y", "changed"],
        "same",
    )
    out = w[ids].reset_index(drop=True)
    out["status"] = status
    return out


def _explain_values(r: pd.DataFrame, col: str) -> pd.DataFrame:
    """The counts of each status per value of ``col``, with the value's class: most
    differences first, then the values that only differ before the mixed ones, since
    they are the cleaner lead."""
    s = r["status"].to_numpy()
    v = pd.DataFrame(
        {
            col: r[col].to_numpy(),
            "n": 1,
            "n_same": s == "same",
            "n_changed": s == "changed",
            "n_only_x": s == "only_x",
            "n_only_y": s == "only_y",
        }
    )
    out = (
        v.groupby(col, dropna=False, sort=False)
        .sum()
        .reset_index()
        .astype({k: "int64" for k in ["n", "n_same", "n_changed", "n_only_x", "n_only_y"]})
    )
    out["class"] = np.select(
        [out["n_same"] == out["n"], out["n_same"] == 0], ["same", "differ"], "mixed"
    )
    out["_d"] = out["n"] - out["n_same"]
    out["_r"] = out["class"].map({"differ": 0, "mixed": 1, "same": 2})
    out = out.sort_values(["_r", col], kind="stable", na_position="last")
    out = out.sort_values("_d", ascending=False, kind="stable", ignore_index=True)
    return out.drop(columns=["_d", "_r"])
