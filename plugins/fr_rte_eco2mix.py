"""RTE éCO2mix, France's national electricity mix, read through eco2mix.py."""

from __future__ import annotations

from datetime import datetime

import eco2mix
from sources import Dataset, Option, Place, Plugin, Table

# What each field measures, in the terms of vocabulary.py.
VARIABLES = {
    "consommation": "electricity.consumption",
    "prevision_j1": "electricity.consumption",
    "prevision_j": "electricity.consumption",
    "nucleaire": "electricity.generation.nuclear",
    "eolien": "electricity.generation.wind",
    "eolien_terrestre": "electricity.generation.wind.onshore",
    "eolien_offshore": "electricity.generation.wind.offshore",
    "solaire": "electricity.generation.solar",
    "hydraulique": "electricity.generation.hydro",
    "hydraulique_fil_eau_eclusee": "electricity.generation.hydro.run_of_river",
    "hydraulique_lacs": "electricity.generation.hydro.reservoir",
    "hydraulique_step_turbinage": "electricity.generation.hydro.pumped_storage",
    "bioenergies": "electricity.generation.bioenergy",
    "bioenergies_dechets": "electricity.generation.bioenergy.waste",
    "bioenergies_biomasse": "electricity.generation.bioenergy.biomass",
    "bioenergies_biogaz": "electricity.generation.bioenergy.biogas",
    "gaz": "electricity.generation.gas",
    "gaz_ccg": "electricity.generation.gas.combined_cycle",
    "gaz_tac": "electricity.generation.gas.combustion_turbine",
    "gaz_cogen": "electricity.generation.gas.cogeneration",
    "gaz_autres": "electricity.generation.gas.other",
    "charbon": "electricity.generation.coal",
    "fioul": "electricity.generation.oil",
    "fioul_tac": "electricity.generation.oil.combustion_turbine",
    "fioul_cogen": "electricity.generation.oil.cogeneration",
    "fioul_autres": "electricity.generation.oil.other",
    "pompage": "electricity.storage.pumping",
    "stockage_batterie": "electricity.storage.battery_charging",
    "destockage_batterie": "electricity.storage.battery_discharging",
    "ech_physiques": "electricity.exchange.physical",
    "ech_comm_angleterre": "electricity.exchange.commercial",
    "ech_comm_espagne": "electricity.exchange.commercial",
    "ech_comm_italie": "electricity.exchange.commercial",
    "ech_comm_suisse": "electricity.exchange.commercial",
    "ech_comm_allemagne_belgique": "electricity.exchange.commercial",
    "taux_co2": "electricity.co2_intensity",
}
REALTIME_ONLY = {"eolien_terrestre", "eolien_offshore", "stockage_batterie", "destockage_batterie"}
FRANCE = Place(country="FR", name="Mainland France")


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
                    description=" ".join(filter(None, (description, f"éCO2mix field {field}."))),
                    variable=VARIABLES[field],
                    measure="intensity" if unit == "g/kWh" else "power",
                    kind="forecast" if field.startswith("prevision") else "estimate" if field == "taux_co2"
                    else "observation",
                    place=FRANCE,
                    resolution="15min",  # the consolidated history is half-hourly for most fields
                    since="" if field in REALTIME_ONLY else f"{eco2mix.FIRST_YEAR}-01-01",
                    quality="mixed")  # consolidated and final history, then real time
            for field, (group, label, unit, description) in eco2mix.FIELDS.items()
        ]

    def read(
        self, datasets: list[str], start: datetime | None, end: datetime | None, options: dict[str, str]
    ) -> Table:
        return eco2mix.read(datasets, start, end, options["resolution"])


PLUGIN = Eco2mix()
