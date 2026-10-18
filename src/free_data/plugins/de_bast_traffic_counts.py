"""BASt automatic traffic counting stations, read through bast.py."""

from __future__ import annotations

from datetime import datetime

from free_data import bast
from free_data.sources import Dataset, Place, Plugin, Table
from free_data.vocabulary import REGIONS, STATE_CODE_TO_NUTS

ROADS = {"A": "Motorways", "B": "Federal roads"}


class Bast(Plugin):
    id = bast.SOURCE
    title = "BASt traffic counts"
    timezone = "Europe/Berlin"

    def datasets(self) -> list[Dataset]:
        """One per station and variable, grouped by road class and state."""
        last = max(bast.years())
        rows = []
        for s in bast.stations():
            region = STATE_CODE_TO_NUTS.get(s.state, "")
            state = REGIONS[region].name if region else s.state
            towards = {"KFZ_R1": f", towards {s.direction1}" if s.direction1 else "",
                       "KFZ_R2": f", towards {s.direction2}" if s.direction2 else ""}
            for key, (_, label, what) in bast.VARIABLES.items():
                rows.append(Dataset(
                    id=f"{s.id}/{key}",
                    label=f"{s.name}, {s.road}{towards.get(key, '')}",
                    group=f"{label} / {ROADS[s.road_class]} / {state}",
                    unit="count",
                    description=(f"Vehicles per hour at the counting station {s.id} {s.name} on the {s.road} "
                                 f"({state}): {what}. Direction 1 leads to {s.direction1 or '?'}, direction 2 to "
                                 f"{s.direction2 or '?'}. Counts the devices missed are estimated by the BASt. "
                                 f"Published once a year, up to {last}; the first read of a year downloads "
                                 f"and reduces that year's file, which takes a minute or two."
                                 + (" Which vehicles count as trucks depends on the device, so trucks compare "
                                    "poorly between stations." if key == "LKW" else "")),
                    variable="traffic.vehicles.trucks" if key == "LKW" else "traffic.vehicles",
                    measure="count",
                    place=Place(country="DE", region=region, name=s.name, lat=s.lat, lon=s.lon),
                    resolution="1h",
                    quality="mixed",  # raw counts and the BASt's estimates for gaps
                ))
        return rows

    def read(
        self, datasets: list[str], start: datetime | None, end: datetime | None, options: dict[str, str]
    ) -> Table:
        return bast.read(datasets, start, end)


PLUGIN = Bast()
