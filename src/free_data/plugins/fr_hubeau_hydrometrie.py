"""Hub'eau hydrometry, the river gauges of France, read through hubeau.py."""

from __future__ import annotations

from datetime import datetime

from free_data import hubeau
from free_data.sources import Dataset, Option, Place, Plugin, Table
from free_data.vocabulary import INSEE_TO_NUTS

# Series -> (variable, measure) in the terms of vocabulary.py.
VARIABLES = {"H": ("water.level", "level"), "Q": ("water.discharge", "flow")}

STATION_PAGE = "https://www.hydro.eaufrance.fr/stationhydro/{code}/fiche"


class Hubeau(Plugin):
    id = hubeau.SOURCE
    title = "Hub'eau river gauges"
    timezone = "Europe/Paris"
    options = (
        Option("resolution", "Resolution", hubeau.RESOLUTIONS, "day",
               help="Only day reaches back years, for flows; native and hour cover the last 30 days."),
    )

    def datasets(self) -> list[Dataset]:
        """One per station and series, grouped by what is measured and the river it is on."""
        rows = []
        for station in sorted(hubeau.stations(), key=lambda s: s["libelle_station"]):
            code, river = station["code_station"], station["libelle_cours_eau"] or "Other waters"
            opened = (station.get("date_ouverture_station") or "")[:10]
            place = Place(country="FR", region=INSEE_TO_NUTS.get(station.get("code_region") or "", ""),
                          name=station["libelle_station"], lat=station.get("latitude_station"),
                          lon=station.get("longitude_station"), water=station["libelle_cours_eau"] or "")
            for kind in hubeau.TYPES.get(station["type_station"], ()):
                name, unit, _ = hubeau.SERIES[kind]
                history = ("Daily mean flows reach back" + (f" to {opened[:4]}" if opened else "") + "; "
                           if kind == "Q" else "No history; ")
                rows.append(Dataset(
                    id=f"{code}/{kind}",
                    label=station["libelle_station"],
                    group=f"{name} / {river}",
                    unit=unit,
                    description=(f"{name} at {station['libelle_station']} ({station['libelle_commune'] or '?'}, "
                                 f"département {station['code_departement']}), station {code}. "
                                 f"{history}the real-time data covers the last 30 days."),
                    link=STATION_PAGE.format(code=code),
                    variable=VARIABLES[kind][0],
                    measure=VARIABLES[kind][1],
                    place=place,
                    since=opened if kind == "Q" else "",  # levels only have the rolling 30 days
                    quality="mixed" if kind == "Q" else "raw",  # validated daily flows, raw real time
                ))
        return rows

    def read(
        self, datasets: list[str], start: datetime | None, end: datetime | None, options: dict[str, str]
    ) -> Table:
        return hubeau.read(datasets, start, end, options["resolution"])


PLUGIN = Hubeau()
