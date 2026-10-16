"""DWD hourly weather observations, read through dwd.py."""

from __future__ import annotations

from datetime import datetime

import dwd
from sources import Dataset, Option, Plugin, Table


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
                                     f"hourly since {station['from'][:4]}-{station['from'][4:6]}-{station['from'][6:]}."),
                        link=f"{dwd.BASE}/{group}/",
                    ))
        return rows

    def read(
        self, datasets: list[str], start: datetime | None, end: datetime | None, options: dict[str, str]
    ) -> Table:
        return dwd.read(datasets, start, end, options["resolution"])


PLUGIN = Dwd()
