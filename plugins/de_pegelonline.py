"""PEGELONLINE gauges of the German federal waterways, read through pegelonline.py."""

from __future__ import annotations

from datetime import datetime

import pegelonline
from sources import Dataset, Option, Plugin, Table


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
                ))
        return rows

    def read(
        self, datasets: list[str], start: datetime | None, end: datetime | None, options: dict[str, str]
    ) -> Table:
        return pegelonline.read(datasets, start, end, options["resolution"])


def _title(text: str) -> str:
    """PEGELONLINE's upper-case names in title case: "UNTERE HAVEL-WASSERSTRASSE" -> "Untere Havel-Wasserstrasse"."""
    return text.title().replace("_", " ")


PLUGIN = Pegelonline()
