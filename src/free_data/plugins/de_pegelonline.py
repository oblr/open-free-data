"""PEGELONLINE gauges of the German federal waterways, read through pegelonline.py."""

from __future__ import annotations

from datetime import datetime

from free_data import pegelonline
from free_data.sources import Dataset, Option, Place, Plugin, Table

# Series short name -> (variable, measure) in the terms of vocabulary.py.
VARIABLES = {
    "W": ("water.level", "level"),
    "Q": ("water.discharge", "flow"),
    "WT": ("water.temperature", "temperature"),
    "LF": ("water.conductivity", "conductivity"),
    "O2": ("water.oxygen", "concentration"),
    "PH": ("water.ph", "ph"),
    "TR": ("water.turbidity", "turbidity"),
    "VA": ("water.flow_velocity", "speed"),
    "R": ("water.flow_direction", "direction"),
    "RV": ("water.flow_direction", "direction"),
    "DFH": ("water.clearance_height", "level"),
    "GRU": ("water.groundwater_level", "level"),
    "MAXH": ("water.waves", "level"),
    "SIGH": ("water.waves", "level"),
    "TP": ("water.waves", "period"),
    "DIRTP": ("water.waves", "direction"),
    "V": ("water.other", "volume"),
    "SW": ("water.other", "other"),  # Sichtweite: how deep one can see
    "LT": ("weather.air_temperature", "temperature"),
    "HL": ("weather.relative_humidity", "humidity"),
    "WG": ("weather.wind_speed", "speed"),
    "WR": ("weather.wind_direction", "direction"),
}


class Pegelonline(Plugin):
    id = pegelonline.SOURCE
    title = "PEGELONLINE waterway gauges"
    timezone = "Europe/Berlin"
    options = (
        Option("resolution", "Resolution", pegelonline.RESOLUTIONS, "hour",
               help="Only the last 31 days are available."),
    )

    def datasets(self) -> list[Dataset]:
        """One per station and series, grouped by what is measured and the water it is on."""
        rows = []
        for station in sorted(pegelonline.stations(), key=lambda s: (s["water"]["longname"], s.get("km") or 0)):
            name, water = _title(station["shortname"]), _title(station["water"]["longname"])
            where = f"{name} on the {water}" + (f", km {station['km']:g}" if station.get("km") is not None else "")
            place = Place(country="DE", name=name, lat=station.get("latitude"), lon=station.get("longitude"),
                          water=water)
            for series in station["timeseries"]:
                kind = pegelonline.PARAMETERS.get(series["shortname"], _title(series["longname"]))
                zero = series.get("gaugeZero")
                rows.append(Dataset(
                    id=f"{station['number']}/{series['shortname']}",
                    label=name,
                    group=f"{kind} / {water}",
                    unit=series["unit"],
                    description=" ".join((
                        f"{kind} ({series['longname']}) at {where}; gauge {station['number']},"
                        f" operated by {_title(station['agency'])}, every {series['equidistance']} min.",
                        f"Gauge zero {zero['value']:g} {zero['unit']}." if zero else "",
                        "Raw data; only the last 31 days are available.",
                    )).replace("  ", " "),
                    link=pegelonline.STATION_PAGE.format(number=station["number"]),
                    variable=VARIABLES.get(series["shortname"], ("water.other", "other"))[0],
                    measure=VARIABLES.get(series["shortname"], ("water.other", "other"))[1],
                    place=place,
                    resolution=_step(series["equidistance"]),
                    quality="raw",
                ))
        return rows

    def read(
        self, datasets: list[str], start: datetime | None, end: datetime | None, options: dict[str, str]
    ) -> Table:
        return pegelonline.read(datasets, start, end, options["resolution"])


def _step(minutes: int) -> str:
    """15 -> "15min", 60 -> "1h", 1440 -> "1d"."""
    if minutes and minutes % 1440 == 0:
        return f"{minutes // 1440}d"
    if minutes and minutes % 60 == 0:
        return f"{minutes // 60}h"
    return f"{minutes}min" if minutes else ""


def _title(text: str) -> str:
    """PEGELONLINE's upper-case names in title case: "UNTERE HAVEL-WASSERSTRASSE" -> "Untere Havel-Wasserstrasse"."""
    return text.title().replace("_", " ")


PLUGIN = Pegelonline()
