"""A text report of a comparison, after the R package's ``diff_report()``.

The layout follows the summary of the Python package diffly (Quantco): a section
each for the columns, the rows and the measures, and a picture of how the rows of
the two tables align. The report is computed first, as plain data, and drawn by
:meth:`DiffReport.format`, so the numbers can be tested apart from the drawing.
"""

from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .core import DiffError, _explain_units, _result_ids, _result_shape, diff_summary

_TITLE = "dazdiff report"


@dataclass
class DiffReport:
    """What :func:`diff_report` found; ``str(report)`` draws it in Unicode and
    ``report.format(unicode=False)`` in ASCII."""

    x_name: str
    y_name: str
    mode: str
    tolerance: dict
    shape: str
    unit: str
    identity: list[str]
    measures: list[str]
    dropped: list[str]
    excluded: list[str]
    ignored: list[str]
    truncated: dict
    paired: bool
    equal: bool
    rows: dict
    by_measure: pd.DataFrame | None
    only_x: pd.DataFrame = field(repr=False)
    only_y: pd.DataFrame = field(repr=False)
    # Distinct keys found in one table only: "(...and N more)" counts these, not
    # rows, since a key can repeat.
    n_keys_only_x: int = 0
    n_keys_only_y: int = 0

    def format(self, unicode: bool = True) -> str:
        """The report as text: box-drawing characters, or ASCII with
        ``unicode=False``."""
        if not isinstance(unicode, bool):
            raise DiffError("`unicode` must be True or False.")
        return "\n".join(_draw(self, _GLYPHS if unicode else _ASCII))

    def __str__(self) -> str:
        return self.format(unicode=True)


def diff_report(
    d: pd.DataFrame, x_name: str = "x", y_name: str = "y", *, sample_k: int = 5
) -> DiffReport:
    """Lay out a :func:`diff_table` result as a bounded text report.

    The identity and mode; the columns compared, dropped, excluded, ignored and
    truncated; a picture of how the rows of the two tables align; per measure that
    changed, its match rate and largest differences; and up to ``sample_k`` keys
    found in only one table. ``x_name`` and ``y_name`` label the two tables.
    """
    if not isinstance(d, pd.DataFrame) or "status" not in d:
        raise DiffError("`d` must be a result of diff_table().")
    for arg, v in (("x_name", x_name), ("y_name", y_name)):
        if not isinstance(v, str) or not v:
            raise DiffError(f"`{arg}` must be a single non-empty string.")
    if isinstance(sample_k, bool) or not isinstance(sample_k, (int, np.integer)) or sample_k < 0:
        raise DiffError("`sample_k` must be a single whole number >= 0.")

    attrs = d.attrs
    shape = _result_shape(d)
    name_map = attrs.get("name_map", {})

    def label(c):
        return (name_map.get(c) or {}).get("x") or c

    ids = _result_ids(d)
    measures: list[str] = []
    if "metric" in d:
        present = list(dict.fromkeys(d["metric"].astype(str)))
        order = [m for m in attrs.get("measures", []) if m in present]
        measures = order + [m for m in present if m not in order]

    # Rows: one per source row (per key for "groups"), changed if any measure is.
    # Membership comes from row_id or n_rows, never from the value.
    units = _explain_units(d, ids, shape)
    status = units["status"].to_numpy()
    rows = {
        "n_x": int((status != "only_y").sum()),
        "n_y": int((status != "only_x").sum()),
        "n_only_x": int((status == "only_x").sum()),
        "n_same": int((status == "same").sum()),
        "n_changed": int((status == "changed").sum()),
        "n_only_y": int((status == "only_y").sum()),
    }

    by_measure = None
    if measures:
        s = diff_summary(d).set_index("metric").reindex(measures)
        joined = s["n_same"] + s["n_changed"]
        by_measure = pd.DataFrame(
            {
                "measure": [label(m) for m in measures],
                "n_same": s["n_same"].to_numpy(),
                "n_changed": s["n_changed"].to_numpy(),
                "match_rate": np.where(
                    joined > 0, s["n_same"] / joined.where(joined > 0, 1), np.nan
                ),
            }
        )
        if shape == "cells":
            by_measure["max_abs_diff"] = s["max_abs_diff"].to_numpy()
            by_measure["max_rel_diff"] = s["max_rel_diff"].to_numpy()

    def keys(side):
        keep = units["status"] == f"only_{side}"
        return units.loc[keep, ids].drop_duplicates()

    def sample(side):
        out = keys(side).head(sample_k).reset_index(drop=True)
        return out.rename(columns={c: label(c) for c in ids})

    return DiffReport(
        x_name=x_name,
        y_name=y_name,
        mode=attrs.get("mode", "benchmark"),
        tolerance=attrs.get("tolerance", {"abs": 0.0, "rel": 0.0}),
        shape=shape,
        unit="keys" if shape == "groups" else "rows",
        identity=[label(c) for c in ids],
        measures=[label(m) for m in measures],
        dropped=list(attrs.get("dropped", [])),
        excluded=list(attrs.get("excluded", [])),
        ignored=list(attrs.get("ignored", [])),
        truncated={label(c): v for c, v in attrs.get("truncated", {}).items()},
        paired=bool(attrs.get("paired", False)),
        equal=bool((d["status"].astype(str) == "same").all()),
        rows=rows,
        by_measure=by_measure,
        only_x=sample("x"),
        only_y=sample("y"),
        n_keys_only_x=len(keys("x")),
        n_keys_only_y=len(keys("y")),
    )


# --- drawing --------------------------------------------------------------------


def _draw(x: DiffReport, g: dict) -> list[str]:
    r = x.rows

    def noun(n, one, many=None):
        return f"{_count(n)} {one if n == 1 else (many or one + 's')}"

    intro = [
        f"Identity: {', '.join(x.identity)}",
        f"Mode:     {x.mode}"
        + (f": {x.y_name} is tested against {x.x_name}" if x.mode == "benchmark" else ""),
    ]
    if x.paired and r["n_x"] + r["n_y"] > 0:
        intro.append("Repeated keys were paired in arrival order (key_seq).")
    if x.shape == "groups":
        intro.append("Repeated keys: each key's values are compared as a multiset.")
    tol = x.tolerance or {}
    if tol.get("abs", 0) > 0 or tol.get("rel", 0) > 0:
        intro.append(f"Tolerance: {_num(tol['abs'], 7)} absolute, {_num(tol['rel'], 7)} relative.")

    columns = [
        f"Compared {noun(len(x.identity) + len(x.measures), 'column')}: "
        f"{noun(len(x.identity), 'key')}, {noun(len(x.measures), 'measure')}."
    ]
    if x.dropped:
        columns.append(f"Only in {x.y_name}, dropped: {', '.join(x.dropped)}.")
    if x.excluded:
        columns.append(f"Excluded: {', '.join(x.excluded)}.")
    if x.ignored:
        columns.append(f"In neither the identity nor the measures: {', '.join(x.ignored)}.")
    if x.truncated:
        columns.append(f"Truncated by the cast to {x.x_name}'s types:")
        for col, pairs in x.truncated.items():
            shown = ", ".join(f"{_num(a, 7)} -> {_num(b, 7)}" for a, b in pairs[:3])
            columns.append(f"  {col}: {noun(len(pairs), 'value')} ({shown})")
    notes = bool(x.dropped or x.excluded or x.ignored or x.truncated)

    body = _indent(intro)
    if x.equal:
        if r["n_x"] + r["n_y"] == 0:
            verdict = f"--- {x.x_name} and {x.y_name} are both empty ---"
        else:
            verdict = (
                f"--- {x.x_name} and {x.y_name} match exactly "
                f"({noun(r['n_x'], x.unit[:-1], x.unit)}, {noun(len(x.measures), 'measure')}) ---"
            )
        body += ["", *_indent([verdict])]
        if notes:
            body += _section("Columns", columns, g)
    else:
        body += _section("Columns", columns, g)
        diagram = _row_diagram(x, g)
        body += _section("Rows", _row_counts(x) + (["", *diagram] if diagram else []), g)
        if x.measures:
            body += _section("Measures", _measure_lines(x, g), g)
        body += _sample_lines(x, "x", g) + _sample_lines(x, "y", g)

    body = [line.rstrip() for line in body]
    width = max(40, *(_width(line) for line in body))
    left = (width - 2 - len(_TITLE)) // 2
    header = [
        g["htl"] + g["hh"] * (width - 2) + g["htr"],
        g["hv"] + " " * left + _TITLE + " " * (width - 2 - len(_TITLE) - left) + g["hv"],
        g["hbl"] + g["hh"] * (width - 2) + g["hbr"],
    ]
    return header + body


def _indent(lines: list[str], n: int = 3) -> list[str]:
    return [" " * n + line if line else line for line in lines]


def _section(title: str, lines: list[str], g: dict) -> list[str]:
    return ["", f" {title}", " " + g["under"] * len(title), *_indent(lines)]


def _center(s: str, width: int) -> str:
    gap = width - _width(s)
    left = gap // 2
    return " " * left + s + " " * (gap - left)


def _row_counts(x: DiffReport) -> list[str]:
    """``x rows   (+33.33%)   y rows`` over the two counts, each column centered."""
    r = x.rows
    if r["n_x"] == 0:
        change = ""
    elif r["n_y"] == r["n_x"]:
        change = "(no change)"
    else:
        sign = "+" if r["n_y"] > r["n_x"] else "-"
        change = f"({sign}{_pct(abs(r['n_y'] / r['n_x'] - 1), clamp=False)})"
    top = [f"{x.x_name} {x.unit}", "", f"{x.y_name} {x.unit}"]
    bottom = [_count(r["n_x"]), change, _count(r["n_y"])]
    widths = [
        max(_width(a), _width(b)) + extra
        for a, b, extra in zip(top, bottom, (0, 4, 0), strict=False)
    ]
    return [
        "".join(_center(s, w) for s, w in zip(top, widths, strict=False)),
        "".join(_center(s, w) for s, w in zip(bottom, widths, strict=False)),
    ]


def _row_diagram(x: DiffReport, g: dict) -> list[str]:
    """Two stacks of boxes, x on the left and y on the right, offset so that the rows
    found in both tables sit side by side: x's own rows above them, y's own below.
    Each band is labelled with its count and share: of x's rows for the rows only in
    x, of the joined rows for ``same`` and ``changed``, of y's rows for the rows only
    in y."""
    r = x.rows
    n_joined = r["n_same"] + r["n_changed"]
    # A filtered result can have no joined band.
    if n_joined == 0:
        return []
    bands = {
        "only_x": (r["n_only_x"], f"only in {x.x_name}", r["n_only_x"] / max(r["n_x"], 1)),
        "same": (r["n_same"], "same", r["n_same"] / n_joined),
        "changed": (r["n_changed"], "changed", r["n_changed"] / n_joined),
        "only_y": (r["n_only_y"], f"only in {x.y_name}", r["n_only_y"] / max(r["n_y"], 1)),
    }
    bands = {k: v for k, v in bands.items() if v[0] > 0}
    counts = {k: _count(v[0]) for k, v in bands.items()}
    kinds = {k: v[1] for k, v in bands.items()}
    pcts = {k: f"({_pct(v[2])})" for k, v in bands.items()}
    wc = max(len(c) for c in counts.values())
    wk = max(_width(k) for k in kinds.values())
    wp = max(len(p) for p in pcts.values())
    labels = {
        k: f"{counts[k]:>{wc}} {kinds[k] + ' ' * (wk - _width(kinds[k]))} {pcts[k]:>{wp}}"
        for k in bands
    }
    label_width = max(_width(v) for v in labels.values())

    def strip(left, mid, right, fill):
        return left + mid.join([fill] * 5) + right

    top = strip(g["btl"], g["btt"], g["btr"], g["bh"])
    sep = strip(g["bml"], g["bmx"], g["bmr"], g["bmh"])
    bottom = strip(g["bbl"], g["bbt"], g["bbr"], g["bh"])

    def cells(ch):
        return strip(g["bv"], g["biv"], g["bv"], ch)

    blank = " " * _width(top)
    dashes = g["dash"] * 3
    run = g["dash"] * (label_width + 4)

    def label(band):
        return "  " + labels[band] + " " * (label_width - _width(labels[band])) + "  "

    out = []
    if "only_x" in bands:
        out.append(top)
        out.append(cells("-") + "   " + blank + label("only_x"))
    out.append((sep if "only_x" in bands else top) + dashes + top + run + g["open"])
    if "same" in bands:
        out.append(cells(" ") + " = " + cells(" ") + label("same") + g["bar"])
    if "same" in bands and "changed" in bands:
        out.append(sep + dashes + sep + run + g["tee"] + "  " + _count(n_joined) + " joined")
    if "changed" in bands:
        out.append(cells(" ") + " " + g["ne"] + " " + cells(" ") + label("changed") + g["bar"])
    out.append(bottom + dashes + (sep if "only_y" in bands else bottom) + run + g["close"])
    if "only_y" in bands:
        out.append(blank + "   " + cells("+") + label("only_y"))
        out.append(blank + "   " + bottom)
    return out


def _measure_lines(x: DiffReport, g: dict) -> list[str]:
    m = x.by_measure
    changed = m[m["n_changed"] > 0]
    rest = len(m) - len(changed)
    if not len(changed):
        return [f"Every measure matches on every row found in both tables ({_count(len(m))})."]
    cells = x.shape == "cells"
    header = ["Measure", "Match rate"] + (["Max |diff|", "Max rel diff"] if cells else [])
    rows = []
    for _, c in changed.iterrows():
        row = [str(c["measure"]), _pct(c["match_rate"])]
        if cells:
            row += [_size(c["max_abs_diff"]), _size(c["max_rel_diff"])]
        rows.append(row)
    align = ["left", "right"] + (["right", "right"] if cells else [])
    out = _table(header, rows, align, g)
    if rest:
        what = "other measure matches" if rest == 1 else "other measures match"
        out.append(f"{_count(rest)} {what} on every row found in both tables.")
    return out


def _sample_lines(x: DiffReport, side: str, g: dict) -> list[str]:
    keys = x.only_x if side == "x" else x.only_y
    n = x.n_keys_only_x if side == "x" else x.n_keys_only_y
    if not len(keys) or not n:
        return []
    max_cols = 5
    shown = list(keys.columns[:max_cols])
    more_cols = len(keys.columns) > max_cols
    header = shown + (["..."] if more_cols else [])
    rows = [
        [_cell(keys[c].iloc[i]) for c in shown] + (["..."] if more_cols else [])
        for i in range(len(keys))
    ]
    name = x.x_name if side == "x" else x.y_name
    lines = _table(header, rows, ["left"] * len(header), g)
    if n > len(keys):
        lines.append(f"(...and {_count(n - len(keys))} more)")
    return _section(f"{'Keys' if x.unit == 'keys' else 'Rows'} only in {name}", lines, g)


def _table(header: list[str], rows: list[list[str]], align: list[str], g: dict) -> list[str]:
    widths = [max(_width(header[j]), *(_width(r[j]) for r in rows)) for j in range(len(header))]

    def pad(s, w, a):
        gap = " " * (w - _width(s))
        return gap + s if a == "right" else s + gap

    def rule(left, mid, right):
        return left + mid.join(g["th"] * (w + 2) for w in widths) + right

    def line(cells):
        return (
            g["tv"]
            + g["tv"].join(
                f" {pad(c, w, a)} " for c, w, a in zip(cells, widths, align, strict=False)
            )
            + g["tv"]
        )

    out = [rule(g["ttl"], g["ttt"], g["ttr"]), line(header), rule(g["tml"], g["tmx"], g["tmr"])]
    out += [line(r) for r in rows]
    return [*out, rule(g["tbl"], g["tbt"], g["tbr"])]


# --- formatting -----------------------------------------------------------------


def _width(s: str) -> int:
    """Display width: wide East Asian characters take two cells, combining marks
    none."""
    return sum(
        0 if unicodedata.combining(ch) else 2 if unicodedata.east_asian_width(ch) in "WF" else 1
        for ch in s
    )


def _count(n) -> str:
    return f"{int(n):,}"


def _pct(p, clamp: bool = True) -> str:
    """A share as a percentage with two decimals. With ``clamp``, it reads 0% or 100%
    only when it is exactly that: one changed row in a billion is not 0.00%."""
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "-"
    p = 100 * float(p)
    if clamp and 0 < p < 100:
        p = min(max(p, 0.01), 99.99)
    return f"{p:.2f}%"


def _num(v, digits: int) -> str:
    """A number as R's ``format(v, digits=)`` writes it: ``digits`` significant
    digits, trailing zeros dropped, fixed notation unless scientific is shorter."""
    v = float(v)
    if math.isnan(v):
        return "NA"
    if math.isinf(v):
        return "Inf" if v > 0 else "-Inf"
    if v == 0:
        return "0"
    mantissa, exp = f"{v:.{digits - 1}e}".split("e")
    if "." in mantissa:
        mantissa = mantissa.rstrip("0").rstrip(".")
    sci = f"{mantissa}e{exp[0]}{exp[1:].zfill(2)}"
    # The significant digits the rounded value needs set the decimals; fixed notation
    # then writes the value itself, keeping every digit before the point.
    sig = len(mantissa.lstrip("-").replace(".", ""))
    decimals = max(0, sig - 1 - int(exp))
    fixed = f"{v:.{decimals}f}"
    return fixed if len(fixed) <= len(sci) else sci


def _size(v) -> str:
    return "-" if v is None or pd.isna(v) else _num(v, 4)


def _cell(v) -> str:
    """One key value: text quoted, numbers with their own digits."""
    if v is None or (not isinstance(v, (list, tuple)) and pd.isna(v)):
        return "NA"
    if isinstance(v, str):
        return f'"{v}"'
    if isinstance(v, (float, np.floating)):
        return _num(v, 7)
    if isinstance(v, (np.integer, np.bool_)):
        return str(v.item())
    return str(v)


_GLYPHS = {
    # the title: heavy
    "htl": "┏", "hh": "━", "htr": "┓", "hv": "┃", "hbl": "┗", "hbr": "┛",
    # the row diagram: heavy edges, light inside
    "btl": "┏", "btt": "┯", "btr": "┓", "bh": "━", "bv": "┃", "biv": "│",
    "bml": "┠", "bmx": "┼", "bmr": "┨", "bmh": "─",
    "bbl": "┗", "bbt": "┷", "bbr": "┛",
    "dash": "╌", "open": "╮", "bar": "│", "tee": "├╴", "close": "╯", "ne": "≠",
    # tables: light
    "ttl": "┌", "ttt": "┬", "ttr": "┐", "th": "─", "tv": "│",
    "tml": "├", "tmx": "┼", "tmr": "┤", "tbl": "└", "tbt": "┴", "tbr": "┘",
    "under": "▔",
}  # fmt: skip

_ASCII = {
    "htl": "+", "hh": "=", "htr": "+", "hv": "|", "hbl": "+", "hbr": "+",
    "btl": "+", "btt": "+", "btr": "+", "bh": "=", "bv": "|", "biv": "|",
    "bml": "+", "bmx": "+", "bmr": "+", "bmh": "-",
    "bbl": "+", "bbt": "+", "bbr": "+",
    "dash": ".", "open": "+", "bar": "|", "tee": "+-", "close": "+", "ne": "!",
    "ttl": "+", "ttt": "+", "ttr": "+", "th": "-", "tv": "|",
    "tml": "+", "tmx": "+", "tmr": "+", "tbl": "+", "tbt": "+", "tbr": "+",
    "under": "-",
}  # fmt: skip
