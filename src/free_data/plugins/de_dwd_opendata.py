"""DWD hourly weather observations, read through dwd.py."""

from __future__ import annotations

from datetime import datetime

from free_data import dwd
from free_data.sources import Dataset, Option, Place, Plugin, Table
from free_data.vocabulary import REGIONS, STATE_TO_NUTS

# DWD variable -> (variable, measure) in the terms of vocabulary.py.
VARIABLES = {
    "TT_TU": ("weather.air_temperature", "temperature"),
    "RF_TU": ("weather.relative_humidity", "humidity"),
    "R1": ("weather.precipitation", "amount"),
    "F": ("weather.wind_speed", "speed"),
    "D": ("weather.wind_direction", "direction"),
    "SD_SO": ("weather.sunshine_duration", "duration"),
    "P": ("weather.air_pressure", "pressure"),
}


class Dwd(Plugin):
    id = dwd.SOURCE
    title = "DWD weather stations"
    timezone = "Europe/Berlin"
    options = (
        Option("resolution", "Resolution", dwd.RESOLUTIONS, "hour",
               help="Day is the mean per Berlin day; precipitation and sunshine are summed."),
    )

    def datasets(self) -> list[Dataset]:
        """One per reporting station and variable, grouped by variable and state."""
        rows = []
        for group in dict.fromkeys(v.group for v in dwd.VARIABLES.values()):
            for station in sorted(dwd.reporting(group), key=lambda s: s["name"]):
                since = f"{station['from'][:4]}-{station['from'][4:6]}-{station['from'][6:]}"
                region = STATE_TO_NUTS.get(station["state"], "")
                place = Place(country=REGIONS[region].country if region else "DE", region=region,
                              name=station["name"], lat=station["lat"], lon=station["lon"])
                for key, variable in dwd.VARIABLES.items():
                    if variable.group != group:
                        continue
                    rows.append(Dataset(
                        id=f"{station['id']}/{key}",
                        label=station["name"],
                        group=f"{variable.name} / {station['state'] or 'Other'}",
                        unit=variable.unit,
                        description=(f"{variable.name} ({key}) at {station['name']}, {station['state']}, "
                                     f"{station['height']:g} m, DWD station {station['id']}; "
                                     f"hourly since {since}."),
                        link=f"{dwd.BASE}/{group}/",
                        variable=VARIABLES[key][0],
                        measure=VARIABLES[key][1],
                        place=place,
                        resolution="1h",
                        since=since,
                        quality="mixed",  # quality-checked history, recent data checked less
                    ))
        return rows

    def read(
        self, datasets: list[str], start: datetime | None, end: datetime | None, options: dict[str, str]
    ) -> Table:
        return dwd.read(datasets, start, end, options["resolution"])


PLUGIN = Dwd()
