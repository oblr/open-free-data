"""SMARD (Bundesnetzagentur) electricity market data, read through smard.py."""

from __future__ import annotations

from datetime import datetime

import smard
from sources import Dataset, Option, Place, Plugin, Table, combine

PRICE_REGION = "DE-LU"  # prices exist only for the bidding zone
MARKET_DATA = "https://www.smard.de/en/marktdaten"  # SMARD's interactive charts, where every series can be picked

# What each filter measures, in the terms of vocabulary.py. Forecasts share
# the variable of what they forecast and differ in `kind`.
VARIABLES: dict[int, str] = {
    1223: "electricity.generation.coal.lignite",
    1224: "electricity.generation.nuclear",
    1225: "electricity.generation.wind.offshore",
    1226: "electricity.generation.hydro",
    1227: "electricity.generation.other_conventional",
    1228: "electricity.generation.other_renewable",
    4066: "electricity.generation.bioenergy.biomass",
    4067: "electricity.generation.wind.onshore",
    4068: "electricity.generation.solar",
    4069: "electricity.generation.coal.hard_coal",
    4070: "electricity.generation.hydro.pumped_storage",
    4071: "electricity.generation.gas",
    410: "electricity.consumption",
    4359: "electricity.residual_load",
    4387: "electricity.storage.pumping",
    411: "electricity.consumption",
    4362: "electricity.residual_load",
    122: "electricity.generation",
    123: "electricity.generation.wind.onshore",
    125: "electricity.generation.solar",
    3791: "electricity.generation.wind.offshore",
    715: "electricity.generation.other",
    5097: "electricity.generation.wind_and_solar",
    4169: "electricity.price.day_ahead",
    5078: "electricity.price.day_ahead",
    254: "electricity.price.day_ahead",
}

# Where the series apply, for the default region "DE"; prices have their own.
GERMANY = Place(country="DE", name="Germany")
PRICE_PLACES = {
    4169: Place(country="DE", name="Germany/Luxembourg bidding zone"),
    5078: Place(country="", name="Neighbours of the Germany/Luxembourg bidding zone"),
    254: Place(country="FR", name="France"),
}

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
                variable=VARIABLES[key],
                measure="price" if price else "energy",
                kind="forecast" if name.startswith("forecast") else "observation",
                place=PRICE_PLACES.get(key, GERMANY),
                resolution="15min",
                quality="mixed",  # recent values are provisional until consolidated
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
