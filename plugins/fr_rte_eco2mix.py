"""RTE éCO2mix, France's national electricity mix, read through eco2mix.py."""

from __future__ import annotations

from datetime import datetime

import eco2mix
from sources import Dataset, Option, Plugin, Table


class Eco2mix(Plugin):
    id = eco2mix.SOURCE
    title = "RTE éCO2mix electricity mix"
    timezone = "Europe/Paris"
    options = (
        Option("resolution", "Resolution", eco2mix.RESOLUTIONS, "hour",
               help="Hour and day are mean power; the history before the real-time data is half-hourly."),
    )

    def datasets(self) -> list[Dataset]:
        return [
            Dataset(id=field, label=label, group=group, unit=unit,
                    description=" ".join(filter(None, (description, f"éCO2mix field {field}."))))
            for field, (group, label, unit, description) in eco2mix.FIELDS.items()
        ]

    def read(
        self, datasets: list[str], start: datetime | None, end: datetime | None, options: dict[str, str]
    ) -> Table:
        return eco2mix.read(datasets, start, end, options["resolution"])


PLUGIN = Eco2mix()
