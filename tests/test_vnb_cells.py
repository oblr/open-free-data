"""Cells of the operators' files: encodings, numbers and voltage levels as the survey found them."""

from __future__ import annotations

import math

import pytest

from free_data.vnb.cells import LEVELS, decimal_mark, decode, level, numbers


def test_decode():
    assert decode("﻿Netzbetreibername;Mitnetz".encode()) == "Netzbetreibername;Mitnetz"
    assert decode("Umspannung Höchst-/Hochspannung".encode("cp1252")) == "Umspannung Höchst-/Hochspannung"


@pytest.mark.parametrize(
    ("cells", "mark"),
    [
        (["761.832", "698.930", "999"], ","),  # Berlin: "." groups thousands
        (["4.471.779", "3.902.114"], ","),  # Netze BW
        (["349.468 ", "350.001 "], ","),  # SWM yearly, trailing spaces
        (["12.351,10", "998,00"], ","),  # Neubrandenburg
        (["196580,1691", "-12,5"], ","),  # SWM monthly
        (["761.83", "698.9"], "."),
        (["0.832", "1.5"], "."),
        (["761.832", "1234"], "."),  # a plain cell of four digits: "." can't group thousands here
        (["1234", "17"], "."),
        (["", "-"], "."),
    ],
)
def test_decimal_mark(cells, mark):
    assert decimal_mark(cells) == mark


def test_numbers():
    assert numbers(["761.832", "1.002.003", "999", ""]).tolist()[:3] == [761_832, 1_002_003, 999]
    assert numbers(["12.351,10", "-0,5", " 3 "]).tolist() == [12_351.1, -0.5, 3.0]
    assert numbers(["761.832", "1.5"]).tolist() == [761.832, 1.5]
    assert numbers(["761.832"], decimal=".").tolist() == [761.832]
    assert numbers([1.5, 2, None, "n/a", "1.000"]).tolist()[:2] == [1.5, 2.0]
    assert all(math.isnan(x) for x in numbers([None, "", "-", "n/a"]))
    with pytest.raises(ValueError, match="'#BEZUG!' isn't a number"):
        numbers(["1", "#BEZUG!"])


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Höchstspannung", "HöS"),
        ("HöS", "HöS"),
        ("Lastverlauf_Hoes.csv", "HöS"),
        ("Umspannung Höchst-/Hochspannung", "HöS/HS"),
        ("lastverlauf-zeitreihe-hoes-hs.CSV", "HöS/HS"),
        ("Lastverlauf_HoeS-HS.csv", "HöS/HS"),
        ("jahreshoechstlast-2025-hochspannung.csv", "HS"),
        ("Netz- oder Umspannebene: Hochspannungsnetz", "HS"),
        ("Umspannung HS/MS", "HS/MS"),
        ("Mittelspannung", "MS"),
        ("Lastverlauf_MS-NS.csv", "MS/NS"),
        ("Umspannebene Mittel-/Niederspannung", "MS/NS"),
        ("summe-einspeisungen-niederspannung-2025.csv", "NS"),
        ("Einspeisungen_in_NS.csv", "NS"),
        ("enwg_23c_3_nr_1_ns.csv", "NS"),
        ("2025_23c-abs-3-nr-1-enwg-ns.csv", "NS"),
        ("NS/MS", "MS/NS"),
        ("Jahreshöchstlast", None),
        ("Einspeisungen", None),
        ("HS und NS", None),
        ("Summenlast SLP", None),
    ],
)
def test_level(text, expected):
    assert level(text) == expected
    assert expected is None or expected in LEVELS
