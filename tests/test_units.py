"""The unit table is consistent, and units convert only where that makes sense."""

from __future__ import annotations

import numpy as np
import pytest

from free_data import sources, units
from free_data.units import DIMENSIONS, UNITS


def test_units_are_well_formed():
    spellings = [s for symbol, u in UNITS.items() for s in (symbol, *u.spellings)]
    assert len(spellings) == len(set(spellings)), "a spelling names two units"
    assert all(s == s.strip() and s for s in spellings)
    assert all(u.dimension in DIMENSIONS for u in UNITS.values())


def test_every_dimension_has_its_base_unit():
    for key, d in DIMENSIONS.items():
        assert d.rate == "" or d.rate in DIMENSIONS, key
        assert any(u.dimension == key for u in UNITS.values()), f"{key} has no units"
        if d.base:
            base = UNITS[d.base]
            assert (base.dimension, base.factor, base.offset) == (key, 1.0, 0.0), key


def test_the_vocabulary_re_exports_the_units():
    from free_data import vocabulary

    assert vocabulary.UNITS is UNITS and vocabulary.convert is units.convert


def test_spellings_become_the_symbol():
    assert units.unit("Grad") == units.unit("°") == "°"
    assert units.unit(" m3/s ") == "m³/s"
    assert units.unit("μS/cm") == units.unit("uS/cm") == "µS/cm"  # Greek mu, ASCII u, micro sign
    assert units.unit("m. ü. NHN") == "m+NHN"
    assert units.unit("furlongs") == "furlongs"  # unknown: kept; test_every_dataset_uses_the_vocabulary flags it
    assert units.dimension("Grad") == "angle" and units.dimension("furlongs") == ""
    assert sources.Dataset("x", "X", unit="TE/F").unit == "FTU"


def test_convert_within_a_dimension():
    assert units.convert(1.5, "GW", "MW") == pytest.approx(1500)
    assert units.convert(250, "cm", "m") == pytest.approx(2.5)
    assert units.convert(np.array([1.0, 2.0]), "mS/cm", "uS/cm").tolist() == pytest.approx([1000, 2000])
    assert units.convert(0.0, "°C", "K") == pytest.approx(273.15)
    assert units.convert(300.0, "K", "°C") == pytest.approx(26.85)
    assert units.convert(36.0, "km/h", "m/s") == pytest.approx(10)
    assert units.convert(5.0, "Grad", "°") == 5.0
    assert units.convert(3.0, "furlongs", "furlongs") == 3.0


def test_convert_an_amount_to_its_rate():
    # SMARD: MWh per quarter hour; éCO2mix: average MW
    assert units.convert(250.0, "MWh", "MW", hours=0.25) == pytest.approx(1000)
    assert units.convert(1.0, "GW", "MWh", hours=24) == pytest.approx(24_000)
    days = units.convert(np.array([2300.0, 2400.0, 2500.0]), "MWh", "MW", hours=np.array([23, 24, 25]))
    assert days.tolist() == pytest.approx([100, 100, 100])
    with pytest.raises(ValueError, match="interval"):
        units.convert(250.0, "MWh", "MW")


@pytest.mark.parametrize("a, b", [
    ("cm", "m+NN"),  # above the gauge zero vs. above sea level
    ("m+NN", "m+NHN"),  # two datums
    ("FNU", "NTU"),  # different methods
    ("MGN", "°"),  # magnetic vs. true north
    ("MW", "m³/s"),
    ("°C", "furlongs"),
])
def test_what_does_not_convert(a, b):
    assert not units.convertible(a, b)
    with pytest.raises(ValueError):
        units.convert(1.0, a, b, hours=1)
