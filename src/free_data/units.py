"""Units: how sources write them, what they measure, and how they convert.

Three tables, each pointing into the one before it:

    DIMENSIONS  what a unit measures, its base unit, and the rate an amount
                per interval averages to ("energy" -> "power")
    UNITS       canonical symbol -> label, dimension, factor to the base
                unit, and the other spellings sources use
    MEASURES    (in vocabulary.py) what kind of number a dataset is, by the
                dimensions its unit may have

A plugin may write any spelling ("Grad", "m3/s", "TE/F"); Dataset keeps the
symbol (see unit()), so the page draws one axis per unit and not one per
spelling. A new spelling goes into its unit's `spellings`, a new unit into
UNITS, a new kind of quantity into DIMENSIONS; tests/test_vocabulary.py
checks that they fit together.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Dimension:
    """What a unit measures. Units of one dimension convert into each other through `base`."""

    label: str
    base: str  # the symbol in UNITS with factor 1 and offset 0; "" when the units don't convert at all
    rate: str = ""  # for an amount per interval, the dimension it averages to: rate = amount / hours


@dataclass(frozen=True)
class Unit:
    """A unit, by its canonical symbol in UNITS: value in the base unit = value * factor + offset."""

    label: str
    dimension: str  # a key of DIMENSIONS
    factor: float = 1.0
    offset: float = 0.0
    spellings: tuple[str, ...] = ()  # how sources write it, besides the symbol


DIMENSIONS: dict[str, Dimension] = {
    "power": Dimension("Power", "MW"),
    "energy": Dimension("Energy", "MWh", rate="power"),  # 250 MWh in a quarter hour is 1000 MW
    "price": Dimension("Price of energy", "EUR/MWh"),
    "intensity": Dimension("Emissions per energy", "g/kWh"),
    "temperature": Dimension("Temperature", "°C"),
    "fraction": Dimension("Fraction", "%"),
    "pressure": Dimension("Pressure", "hPa"),
    "speed": Dimension("Speed", "m/s"),
    "angle": Dimension("Angle from true north", "°"),
    "angle_magnetic": Dimension("Angle from magnetic north", "°M"),  # the declination varies by place and year
    "time": Dimension("Time", "s"),
    "length": Dimension("Length; a water level above the gauge's own zero (PNP)", "m"),
    "height_nn": Dimension("Height above Normalnull", "m+NN"),
    "height_nhn": Dimension("Height above Normalhöhennull", "m+NHN"),
    "flow": Dimension("Volume per time", "m³/s"),
    "volume": Dimension("Volume", "m³"),
    "conductivity": Dimension("Electrical conductivity", "µS/cm"),
    "concentration": Dimension("Mass concentration, in water or air", "mg/l"),  # mg/l = g/m³
    "turbidity": Dimension("Turbidity; its units measure differently and only roughly agree", ""),
    "dimensionless": Dimension("No unit", "-"),
}

UNITS: dict[str, Unit] = {
    # electricity
    "kW": Unit("kilowatt", "power", 1e-3),
    "MW": Unit("megawatt", "power"),
    "GW": Unit("gigawatt", "power", 1e3),
    "kWh": Unit("kilowatt hour", "energy", 1e-3),
    "MWh": Unit("megawatt hour", "energy"),
    "GWh": Unit("gigawatt hour", "energy", 1e3),
    "TWh": Unit("terawatt hour", "energy", 1e6),
    "EUR/MWh": Unit("euro per megawatt hour", "price", spellings=("€/MWh", "EUR / MWh")),
    "EUR/kWh": Unit("euro per kilowatt hour", "price", 1e3, spellings=("€/kWh",)),
    "ct/kWh": Unit("euro cent per kilowatt hour", "price", 10.0),
    "g/kWh": Unit("gram per kilowatt hour", "intensity", spellings=("gCO2/kWh", "gCO2eq/kWh")),
    "kg/MWh": Unit("kilogram per megawatt hour", "intensity"),
    # weather
    "°C": Unit("degree Celsius", "temperature", spellings=("degC", "Grad C")),
    "K": Unit("kelvin", "temperature", 1.0, -273.15),
    "%": Unit("percent", "fraction", spellings=("percent", "Prozent")),
    "hPa": Unit("hectopascal", "pressure", spellings=("mbar",)),
    "Pa": Unit("pascal", "pressure", 1e-2),
    "m/s": Unit("metre per second", "speed"),
    "cm/s": Unit("centimetre per second", "speed", 1e-2),
    "km/h": Unit("kilometre per hour", "speed", 1 / 3.6),
    "kn": Unit("knot", "speed", 1852 / 3600, spellings=("kt",)),
    "°": Unit("degree", "angle", spellings=("Grad", "deg", "degree")),
    "°M": Unit("degree from magnetic north", "angle_magnetic", spellings=("MGN",)),
    # time
    "s": Unit("second", "time"),
    "cs": Unit("centisecond", "time", 1e-2, spellings=("1/100s", "1/100 s")),
    "min": Unit("minute", "time", 60.0),
    "h": Unit("hour", "time", 3600.0),
    # lengths
    "mm": Unit("millimetre", "length", 1e-3),
    "cm": Unit("centimetre", "length", 1e-2),
    "m": Unit("metre", "length", spellings=("m+PNP",)),
    "km": Unit("kilometre", "length", 1e3),
    "m+NN": Unit("metre above Normalnull", "height_nn", spellings=("m ü. NN", "m. ü. NN", "mNN")),
    "m+NHN": Unit("metre above Normalhöhennull", "height_nhn", spellings=("m ü. NHN", "m. ü. NHN", "mNHN")),
    # water
    "m³/s": Unit("cubic metre per second", "flow", spellings=("m3/s", "m^3/s")),
    "l/s": Unit("litre per second", "flow", 1e-3, spellings=("L/s",)),
    "m³": Unit("cubic metre", "volume", spellings=("m3", "m^3")),
    "µS/cm": Unit("microsiemens per centimetre", "conductivity", spellings=("μS/cm", "uS/cm")),
    "mS/cm": Unit("millisiemens per centimetre", "conductivity", 1e3),
    "mg/l": Unit("milligram per litre", "concentration", spellings=("mg/L",)),
    "µg/l": Unit("microgram per litre", "concentration", 1e-3, spellings=("μg/l", "ug/l", "µg/L")),
    # air
    "mg/m³": Unit("milligram per cubic metre", "concentration", 1e-3, spellings=("mg/m3",)),
    "µg/m³": Unit("microgram per cubic metre", "concentration", 1e-6, spellings=("μg/m³", "µg/m3", "ug/m3")),
    "ng/m³": Unit("nanogram per cubic metre", "concentration", 1e-9, spellings=("ng/m3",)),
    "FNU": Unit("formazin nephelometric unit", "turbidity"),
    "FTU": Unit("formazin turbidity unit", "turbidity", spellings=("TE/F",)),
    "NTU": Unit("nephelometric turbidity unit", "turbidity"),
    "-": Unit("no unit, e.g. pH", "dimensionless", spellings=("--", "1")),
}

_SPELLINGS = {spelling: symbol for symbol, u in UNITS.items() for spelling in (symbol, *u.spellings)}


def unit(spelling: str) -> str:
    """The canonical symbol for a unit as a source writes it; an unknown one stays as it is."""
    spelling = spelling.strip()
    return _SPELLINGS.get(spelling, spelling)


def dimension(spelling: str) -> str:
    """The dimension of a unit, "" for an unknown one."""
    found = UNITS.get(unit(spelling))
    return found.dimension if found else ""


def convertible(a: str, b: str) -> bool:
    """Whether values in unit `a` convert to unit `b` (an amount to its rate needs the interval)."""
    a, b = unit(a), unit(b)
    if a == b:
        return True
    if a not in UNITS or b not in UNITS:
        return False
    da, db = DIMENSIONS[UNITS[a].dimension], DIMENSIONS[UNITS[b].dimension]
    if not (da.base and db.base):
        return False
    return da == db or da.rate == UNITS[b].dimension or db.rate == UNITS[a].dimension


def convert(values, a: str, b: str, hours=None):
    """`values` (a number or numpy array) in unit `a`, converted to unit `b`.

    Between an amount per interval and its rate (MWh and MW), `hours` is the
    length of each interval: a number, or an array as long as `values` when
    the intervals differ (days with a clock change). Raises ValueError for
    units that don't convert.
    """
    if not convertible(a, b):
        raise ValueError(f"can't convert {a!r} to {b!r}")
    a, b = unit(a), unit(b)
    if a == b:
        return values
    ua, ub = UNITS[a], UNITS[b]
    base = values * ua.factor + ua.offset
    if ua.dimension != ub.dimension:
        if hours is None:
            raise ValueError(f"converting {a!r} to {b!r} needs the interval length in hours")
        base = base / hours if DIMENSIONS[ua.dimension].rate == ub.dimension else base * hours
    return (base - ub.offset) / ub.factor
