"""UBA air quality stations, read through uba.py."""

from __future__ import annotations

from datetime import datetime

import numpy as np

from free_data import uba
from free_data.sources import Dataset, Option, Place, Plugin, Table
from free_data.vocabulary import REGIONS, STATE_CODE_TO_NUTS

# Component -> variable in the terms of vocabulary.py; every one is a concentration.
VARIABLES = {
    "PM10": "air.pm10",
    "PM2": "air.pm2_5",
    "NO2": "air.no2",
    "O3": "air.o3",
    "SO2": "air.so2",
    "CO": "air.co",
}

# The UBA's network codes of the states are their two-letter codes; the UBA's own network ("UB") spans several.
NETWORK_TO_NUTS = STATE_CODE_TO_NUTS


class Uba(Plugin):
    id = uba.SOURCE
    title = "UBA air quality"
    timezone = "Europe/Berlin"
    options = (
        Option("resolution", "Resolution", uba.RESOLUTIONS, "native",
               help="Native is hourly, or daily where a station only has daily means; day is the mean per "
                    "Berlin day."),
    )

    def datasets(self) -> list[Dataset]:
        """One per reporting station and component, grouped by component and state."""
        rows = []
        for s in uba.catalog():
            if not s.reporting:
                continue
            region = NETWORK_TO_NUTS.get(s.network, "")
            network = REGIONS[region].name if region else "Umweltbundesamt"
            since = str(np.datetime64(s.first + uba.CET_MS, "ms").astype("datetime64[D]"))
            name = uba.COMPONENTS[s.component]
            rows.append(Dataset(
                id=s.name,
                label=s.station_name,
                group=f"{name} / {network}",
                unit=s.unit,
                description=(f"{name} ({s.component}) at {s.station_name}, {s.city}: {s.setting} {s.type} station "
                             f"{s.station} of the {s.network_name} network; {s.scope_name.lower()} "
                             f"({s.scope}) since {since}. Data of the current year are preliminary until June "
                             f"of the next."),
                variable=VARIABLES[s.component],
                measure="concentration",
                place=Place(country="DE", region=region, name=s.station_name, lat=s.lat, lon=s.lon),
                resolution="1d" if s.scope in uba.DAILY else "1h",
                since=since,
                quality="mixed",  # validated history, preliminary current year
            ))
        return rows

    def read(
        self, datasets: list[str], start: datetime | None, end: datetime | None, options: dict[str, str]
    ) -> Table:
        return uba.read(datasets, start, end, options["resolution"])


PLUGIN = Uba()
