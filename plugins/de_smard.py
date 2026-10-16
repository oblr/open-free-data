"""SMARD (Bundesnetzagentur) electricity market data, read through smard.py."""

from __future__ import annotations

from datetime import datetime

import smard
from sources import Dataset, Option, Plugin, Table, combine

PRICE_REGION = "DE-LU"  # prices exist only for the bidding zone
MARKET_DATA = "https://www.smard.de/en/marktdaten"  # SMARD's interactive charts, where every series can be picked

# Where each filter sits in the browser's tree, and its label there.
LAYOUT: dict[int, tuple[str, str]] = {
    4067: ("Generation / Renewable", "Wind onshore"),
    1225: ("Generation / Renewable", "Wind offshore"),
    4068: ("Generation / Renewable", "Solar"),
    1226: ("Generation / Renewable", "Hydro"),
    4066: ("Generation / Renewable", "Biomass"),
    1228: ("Generation / Renewable", "Other renewable"),
    1223: ("Generation / Conventional", "Lignite"),
    4069: ("Generation / Conventional", "Hard coal"),
    4071: ("Generation / Conventional", "Natural gas"),
    1224: ("Generation / Conventional", "Nuclear"),
    4070: ("Generation / Conventional", "Pumped storage"),
    1227: ("Generation / Conventional", "Other conventional"),
    410: ("Consumption", "Total (grid load)"),
    4359: ("Consumption", "Residual load"),
    4387: ("Consumption", "Pumped storage"),
    122: ("Forecast / Generation", "Total"),
    123: ("Forecast / Generation", "Wind onshore"),
    3791: ("Forecast / Generation", "Wind offshore"),
    125: ("Forecast / Generation", "Solar"),
    5097: ("Forecast / Generation", "Wind and solar"),
    715: ("Forecast / Generation", "Other"),
    411: ("Forecast / Consumption", "Total (grid load)"),
    4362: ("Forecast / Consumption", "Residual load"),
    4169: ("Day-ahead price", "Germany/Luxembourg"),
    5078: ("Day-ahead price", "Neighbours of DE/LU (average)"),
    254: ("Day-ahead price", "France"),
}


class Smard(Plugin):
    id = smard.SOURCE
    title = "SMARD electricity market"
    timezone = "Europe/Berlin"
    options = (
        Option("region", "Region", smard.REGIONS, "DE", help=f"Prices are always read for {PRICE_REGION}."),
        Option("resolution", "Resolution", smard.RESOLUTIONS, "hour"),
    )

    def datasets(self) -> list[Dataset]:
        rows = []
        order = [*LAYOUT, *(key for key in smard.FILTERS if key not in LAYOUT)]  # the tree's order
        for key in order:
            name = smard.FILTERS[key]
            price = name.startswith("price")
            group, label = LAYOUT.get(key, ("Other", name))
            rows.append(Dataset(
                id=str(key),
                label=label,
                group=group,
                unit="EUR/MWh" if price else "MWh",
                description=f"SMARD filter {key} ({name})" + (f", region {PRICE_REGION}" if price else ""),
                link=MARKET_DATA,
            ))
        return rows

    def read(
        self, datasets: list[str], start: datetime | None, end: datetime | None, options: dict[str, str]
    ) -> Table:
        tables = []
        for dataset in datasets:
            key = int(dataset)
            price = smard.FILTERS.get(key, "").startswith("price")
            region = PRICE_REGION if price else options["region"]
            tables.append(smard.read(key, region, options["resolution"], start, end))
        combined = combine(tables)
        return Table(tuple(datasets), combined.time, combined.values)


PLUGIN = Smard()
