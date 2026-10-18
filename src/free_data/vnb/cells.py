"""Text cells of the operators' files: encodings, numbers, voltage levels.

Encodings: UTF-8 (with or without BOM) and cp1252 both occur; `decode`
tries them in that order.

Numbers are written in three ways, sometimes with trailing spaces:

    761.832      Berlin, MITNETZ, SWM yearly: whole kW, "." groups thousands
    12.351,10    Neubrandenburg: "," is the decimal mark, "." groups thousands
    196580,1691  SWM monthly: "," is the decimal mark, no grouping

So `numbers` decides per column, from all its cells: a "," anywhere makes
"," the decimal mark; otherwise "." groups thousands if every cell with a
"." is written as grouped digits (1.234, 12.345.678) and no cell without
one has four digits or more; otherwise "." is the decimal mark. A column
of grouped-looking numbers that are really three decimals (761.832 kW)
can't be told apart; a spec then says `decimal`.

Voltage levels are named by everyone differently ("Höchstspannung",
"HöS", "Hoes", "Umspannung HS/MS", "Lastverlauf_MS-NS"); `level` reads
them into one of LEVELS.
"""

from __future__ import annotations

import re

import numpy as np

ENCODINGS = ("utf-8-sig", "cp1252")
EMPTY = {"", "-", "--", "n/a", "na", "nan", "null"}

# From the highest voltage down; a transformation between two levels is "upper/lower".
LEVELS = ("HöS", "HöS/HS", "HS", "HS/MS", "MS", "MS/NS", "NS")
_RANK = {"HöS": 0, "HS": 1, "MS": 2, "NS": 3}
_WORDS = {
    "hoechst": "HöS", "hoes": "HöS", "hoesp": "HöS", "hoechstspannung": "HöS",
    "hoch": "HS", "hs": "HS", "hsp": "HS", "hochspannung": "HS",
    "mittel": "MS", "ms": "MS", "msp": "MS", "mittelspannung": "MS",
    "nieder": "NS", "ns": "NS", "nsp": "NS", "niederspannung": "NS",
}  # fmt: skip
_WORD = re.compile(r"\b(hoechst|hoch|mittel|nieder)(?:spannung\w*)?\b|\b(hoesp|hoes|hsp|hs|msp|ms|nsp|ns)\b")
_GROUPED = re.compile(r"-?[1-9]\d{0,2}(\.\d{3})+")
_PLAIN = re.compile(r"-?\d+")


def decode(raw: bytes) -> str:
    """The text of a file, from the first encoding in ENCODINGS that fits."""
    for encoding in ENCODINGS:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1")  # every byte is a character; never fails


def decimal_mark(cells: list[str]) -> str:
    """The decimal mark, "," or ".", of a column of number cells (see the module's text)."""
    texts = [c.strip() for c in cells if c is not None and c.strip().lower() not in EMPTY]
    if any("," in t for t in texts):
        return ","
    dotted = [t for t in texts if "." in t]
    if dotted and all(_GROUPED.fullmatch(t) for t in dotted):
        plain = [t for t in texts if "." not in t]
        if all(_PLAIN.fullmatch(t) and len(t.lstrip("-")) <= 3 for t in plain):
            return ","
    return "."


def numbers(cells: list, decimal: str | None = None) -> np.ndarray:
    """A column of cells as floats; empty cells and markers ("-", "n/a") are NaN.

    Cells that are already numbers (from a spreadsheet) are kept. Text cells
    are read with `decimal` as the decimal mark, or the one `decimal_mark`
    finds in the column's text cells.
    """
    texts = [c for c in cells if isinstance(c, str)]
    mark = decimal or (decimal_mark(texts) if texts else ".")
    out = np.full(len(cells), np.nan)
    for i, c in enumerate(cells):
        if c is None or isinstance(c, bool):
            continue
        if isinstance(c, int | float):
            out[i] = c
            continue
        t = str(c).strip().replace(" ", "").replace(" ", "")
        if t.lower() in EMPTY:
            continue
        t = t.replace(".", "").replace(",", ".") if mark == "," else t.replace(",", "")
        try:
            out[i] = float(t)
        except ValueError as e:
            raise ValueError(f"cell {i}: {c!r} isn't a number") from e
    return out


def level(text: str) -> str | None:
    """The voltage level (one of LEVELS) a name or file name refers to, or None.

    One level word gives that level, two neighbouring ones the transformation
    between them; anything else is None.
    """
    t = text.lower().replace("ö", "oe").replace("ä", "ae").replace("ü", "ue")
    t = re.sub(r"[^a-z0-9]+", " ", t)
    found: list[str] = []
    for m in _WORD.finditer(t):
        name = _WORDS[m[1] or m[2]]
        if name not in found:
            found.append(name)
    if len(found) == 1:
        return found[0]
    if len(found) == 2:
        upper, lower = sorted(found, key=_RANK.__getitem__)
        if _RANK[lower] - _RANK[upper] == 1:
            return f"{upper}/{lower}"
    return None
