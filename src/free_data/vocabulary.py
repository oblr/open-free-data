"""The shared vocabulary that makes series from different sources comparable.

Every dataset names a `variable` from VARIABLES, a `measure` from MEASURES,
a unit from UNITS (units.py, re-exported here) and, through its Place, a
country and a region from REGIONS. Plugins fill these in;
tests/test_vocabulary.py checks that they only use terms defined here, so
"solar generation" means the same in SMARD and in éCO2mix.

Variable ids are dotted and nest: "electricity.generation.coal.lignite" is
a kind of "electricity.generation.coal", so French coal (one series) and
German lignite and hard coal (two) still meet one level up. Every prefix of
an id is itself a variable.

Where a standard exists, a variable names its term there:
    cf:<name>      CF Standard Name (cfconventions.org), for weather and water
    entsoe:<code>  ENTSO-E production type (B01 ... B20), for generation
"""

from __future__ import annotations

from dataclasses import dataclass, field

from free_data.units import (  # noqa: F401  (re-exported: the units are part of the vocabulary)
    DIMENSIONS,
    UNITS,
    Dimension,
    Unit,
    convert,
    convertible,
    dimension,
    unit,
)


@dataclass(frozen=True)
class Variable:
    label: str  # English
    synonyms: tuple[str, ...] = ()  # German, French and other English words people search for
    standard: str = ""  # "cf:air_temperature", "entsoe:B16"


@dataclass(frozen=True)
class Region:
    name: str
    country: str  # ISO 3166-1 alpha-2
    synonyms: tuple[str, ...] = field(default=())


def _v(label: str, *synonyms: str, standard: str = "") -> Variable:
    return Variable(label, synonyms, standard)


VARIABLES: dict[str, Variable] = {
    # ---- electricity ----
    "electricity": _v("Electricity", "Strom", "électricité"),
    "electricity.consumption": _v("Electricity consumption", "load", "demand", "Verbrauch", "Netzlast",
                                  "consommation"),
    "electricity.residual_load": _v("Residual load", "Residuallast", "charge résiduelle"),
    "electricity.generation": _v("Electricity generation", "production", "Erzeugung", "génération"),
    "electricity.generation.nuclear": _v("Nuclear", "Kernenergie", "nucléaire", standard="entsoe:B14"),
    "electricity.generation.coal": _v("Coal", "Kohle", "charbon"),
    "electricity.generation.coal.lignite": _v("Lignite", "brown coal", "Braunkohle", "lignite",
                                              standard="entsoe:B02"),
    "electricity.generation.coal.hard_coal": _v("Hard coal", "Steinkohle", "houille", standard="entsoe:B05"),
    "electricity.generation.gas": _v("Gas", "natural gas", "Erdgas", "gaz", standard="entsoe:B04"),
    "electricity.generation.gas.combined_cycle": _v("Gas, combined cycle", "CCGT", "GuD", "cycle combiné"),
    "electricity.generation.gas.combustion_turbine": _v("Gas, combustion turbines", "OCGT", "Gasturbine", "TAC"),
    "electricity.generation.gas.cogeneration": _v("Gas, cogeneration", "CHP", "KWK", "cogénération"),
    "electricity.generation.gas.other": _v("Gas, other"),
    "electricity.generation.oil": _v("Oil", "Öl", "fioul", standard="entsoe:B06"),
    "electricity.generation.oil.combustion_turbine": _v("Oil, combustion turbines", "TAC"),
    "electricity.generation.oil.cogeneration": _v("Oil, cogeneration", "CHP", "KWK", "cogénération"),
    "electricity.generation.oil.other": _v("Oil, other"),
    "electricity.generation.wind": _v("Wind", "Windkraft", "éolien"),
    "electricity.generation.wind.onshore": _v("Wind onshore", "Wind an Land", "éolien terrestre",
                                              standard="entsoe:B19"),
    "electricity.generation.wind.offshore": _v("Wind offshore", "Wind auf See", "éolien en mer",
                                               standard="entsoe:B18"),
    "electricity.generation.solar": _v("Solar", "photovoltaics", "PV", "Photovoltaik", "solaire",
                                       standard="entsoe:B16"),
    "electricity.generation.wind_and_solar": _v("Wind and solar", "variable renewables", "Wind und Solar"),
    "electricity.generation.hydro": _v("Hydro", "Wasserkraft", "hydraulique"),
    "electricity.generation.hydro.run_of_river": _v("Hydro, run-of-river", "Laufwasser", "fil de l'eau",
                                                    standard="entsoe:B11"),
    "electricity.generation.hydro.reservoir": _v("Hydro, reservoirs", "Speicherwasser", "lacs",
                                                 standard="entsoe:B12"),
    "electricity.generation.hydro.pumped_storage": _v("Hydro, pumped storage", "Pumpspeicher", "STEP",
                                                      standard="entsoe:B10"),
    "electricity.generation.bioenergy": _v("Bioenergy", "Bioenergie", "bioénergies"),
    "electricity.generation.bioenergy.biomass": _v("Biomass", "Biomasse", "biomasse", standard="entsoe:B01"),
    "electricity.generation.bioenergy.biogas": _v("Biogas", "Biogas", "biogaz"),
    "electricity.generation.bioenergy.waste": _v("Waste", "Abfall", "déchets", standard="entsoe:B17"),
    "electricity.generation.other_renewable": _v("Other renewables", "sonstige Erneuerbare",
                                                 standard="entsoe:B15"),
    "electricity.generation.other_conventional": _v("Other conventional", "sonstige Konventionelle",
                                                    standard="entsoe:B20"),
    "electricity.generation.other": _v("Other generation"),
    "electricity.storage": _v("Electricity storage", "Speicher", "stockage"),
    "electricity.storage.pumping": _v("Pumping", "Pumpspeicher Verbrauch", "pompage"),
    "electricity.storage.battery_charging": _v("Battery charging", "Batterie laden", "stockage batterie"),
    "electricity.storage.battery_discharging": _v("Battery discharging", "Batterie entladen",
                                                  "déstockage batterie"),
    "electricity.exchange": _v("Cross-border exchange", "imports", "exports", "Austausch", "échanges"),
    "electricity.exchange.physical": _v("Physical exchange", "physikalischer Fluss", "échanges physiques"),
    "electricity.exchange.commercial": _v("Commercial exchange", "kommerzieller Austausch",
                                          "échanges commerciaux"),
    "electricity.price": _v("Electricity price", "Strompreis", "prix"),
    "electricity.price.day_ahead": _v("Day-ahead price", "spot price", "Day-Ahead-Preis", "prix spot"),
    "electricity.co2_intensity": _v("CO2 intensity of electricity", "carbon intensity", "emissions",
                                    "CO2-Intensität", "taux de CO2"),
    # ---- weather ----
    "weather": _v("Weather", "Wetter", "météo"),
    "weather.air_temperature": _v("Air temperature", "Lufttemperatur", "température de l'air",
                                  standard="cf:air_temperature"),
    "weather.relative_humidity": _v("Relative humidity", "Luftfeuchte", "humidité relative",
                                    standard="cf:relative_humidity"),
    "weather.precipitation": _v("Precipitation", "rain", "Niederschlag", "Regen", "précipitations", "pluie",
                                standard="cf:precipitation_amount"),
    "weather.wind_speed": _v("Wind speed", "Windgeschwindigkeit", "vitesse du vent", standard="cf:wind_speed"),
    "weather.wind_direction": _v("Wind direction", "Windrichtung", "direction du vent",
                                 standard="cf:wind_from_direction"),
    "weather.sunshine_duration": _v("Sunshine duration", "Sonnenscheindauer", "durée d'ensoleillement",
                                    standard="cf:duration_of_sunshine"),
    "weather.air_pressure": _v("Air pressure at sea level", "Luftdruck", "pression atmosphérique",
                               standard="cf:air_pressure_at_mean_sea_level"),
    # ---- water ----
    "water": _v("Water", "Gewässer", "eau"),
    "water.level": _v("Water level", "stage", "gauge height", "Wasserstand", "Pegel", "hauteur d'eau",
                      standard="cf:water_surface_height_above_reference_datum"),
    "water.discharge": _v("River discharge", "flow", "Abfluss", "débit",
                          standard="cf:water_volume_transport_in_river_channel"),
    "water.temperature": _v("Water temperature", "Wassertemperatur", "température de l'eau"),
    "water.conductivity": _v("Electrical conductivity of water", "Leitfähigkeit", "conductivité"),
    "water.oxygen": _v("Dissolved oxygen", "Sauerstoff", "oxygène dissous"),
    "water.ph": _v("pH of water", "pH-Wert"),
    "water.turbidity": _v("Turbidity", "Trübung", "turbidité"),
    "water.flow_velocity": _v("Flow velocity", "Fließgeschwindigkeit", "vitesse du courant"),
    "water.flow_direction": _v("Flow direction", "Strömungsrichtung"),
    "water.clearance_height": _v("Clearance height under bridges", "Durchfahrtshöhe"),
    "water.waves": _v("Waves", "Wellen", "vagues"),
    "water.other": _v("Other water measurements"),
    "water.groundwater_level": _v("Groundwater level", "Grundwasser", "nappe phréatique"),
    # ---- air quality ----
    "air": _v("Air quality", "air pollution", "Luftqualität", "Luftschadstoffe", "qualité de l'air", "pollution"),
    "air.pm10": _v("Particulate matter PM10", "fine dust", "Feinstaub", "particules fines", "PM10",
                   standard="cf:mass_concentration_of_pm10_ambient_aerosol_particles_in_air"),
    "air.pm2_5": _v("Particulate matter PM2.5", "fine dust", "Feinstaub", "particules fines", "PM2.5",
                    standard="cf:mass_concentration_of_pm2p5_ambient_aerosol_particles_in_air"),
    "air.no2": _v("Nitrogen dioxide", "NO2", "Stickstoffdioxid", "dioxyde d'azote",
                  standard="cf:mass_concentration_of_nitrogen_dioxide_in_air"),
    "air.o3": _v("Ozone", "O3", "Ozon", "ozone", standard="cf:mass_concentration_of_ozone_in_air"),
    "air.so2": _v("Sulphur dioxide", "sulfur dioxide", "SO2", "Schwefeldioxid", "dioxyde de soufre",
                  standard="cf:mass_concentration_of_sulfur_dioxide_in_air"),
    "air.co": _v("Carbon monoxide", "CO", "Kohlenmonoxid", "monoxyde de carbone",
                 standard="cf:mass_concentration_of_carbon_monoxide_in_air"),
    # ---- finance ----
    "finance": _v("Financial markets", "stock exchange", "Börse", "Finanzmarkt", "bourse", "marchés financiers"),
    "finance.share_price": _v("Share price", "stock price", "Aktienkurs", "Kurs", "cours de l'action", "action"),
    "finance.index_level": _v("Stock index", "Aktienindex", "Index", "indice boursier"),
    "finance.trading_volume": _v("Trading volume", "Handelsvolumen", "Umsatz", "volume des échanges"),
}

@dataclass(frozen=True)
class Measure:
    label: str
    dimensions: tuple[str, ...] = ()  # keys of units.DIMENSIONS its unit may have; () for any

    @property
    def units(self) -> tuple[str, ...]:
        """The base units of its dimensions, for people choosing it: "MW", or "m", "m+NN", "m+NHN"."""
        return tuple(DIMENSIONS[d].base for d in self.dimensions if DIMENSIONS[d].base)


# What a number is, independent of the subject: comparing a power with an
# energy, or a level with a flow, needs care even for the same variable.
MEASURES: dict[str, Measure] = {
    "power": Measure("Average power over the interval", ("power",)),
    "energy": Measure("Energy per interval", ("energy",)),
    "price": Measure("Price", ("price", "money")),
    "index": Measure("Index level", ("index",)),
    "count": Measure("Number per interval, e.g. shares traded", ("count",)),
    "intensity": Measure("Emissions per unit of energy", ("intensity",)),
    "temperature": Measure("Temperature", ("temperature",)),
    "humidity": Measure("Relative humidity", ("fraction",)),
    "amount": Measure("Amount per interval, e.g. precipitation", ("length",)),
    "duration": Measure("Duration per interval, e.g. sunshine", ("time",)),
    "speed": Measure("Speed", ("speed",)),
    "direction": Measure("Direction", ("angle", "angle_magnetic")),
    "pressure": Measure("Pressure", ("pressure",)),
    "level": Measure("Height or level", ("length", "height_nn", "height_nhn")),
    "flow": Measure("Volume per time", ("flow",)),
    "concentration": Measure("Concentration", ("concentration",)),
    "conductivity": Measure("Electrical conductivity", ("conductivity",)),
    "ph": Measure("pH", ("dimensionless",)),
    "turbidity": Measure("Turbidity", ("turbidity",)),
    "volume": Measure("Volume", ("volume",)),
    "period": Measure("Period", ("time",)),
    "other": Measure("Anything else"),
}

KINDS = ("observation", "forecast", "estimate")
QUALITIES = ("", "raw", "mixed", "validated")  # "" when the source doesn't say

# ISO 3166-1 alpha-2 -> name, and what people may type instead.
COUNTRIES: dict[str, Region] = {
    "DE": Region("Germany", "DE", ("Deutschland", "Allemagne")),
    "FR": Region("France", "FR", ("Frankreich",)),
    "AT": Region("Austria", "AT", ("Österreich", "Autriche")),
    "LU": Region("Luxembourg", "LU", ("Luxemburg",)),
}

# NUTS level 1 (2021): the German states and the French regions.
REGIONS: dict[str, Region] = {
    "DE1": Region("Baden-Württemberg", "DE"),
    "DE2": Region("Bayern", "DE", ("Bavaria",)),
    "DE3": Region("Berlin", "DE"),
    "DE4": Region("Brandenburg", "DE"),
    "DE5": Region("Bremen", "DE"),
    "DE6": Region("Hamburg", "DE"),
    "DE7": Region("Hessen", "DE", ("Hesse",)),
    "DE8": Region("Mecklenburg-Vorpommern", "DE"),
    "DE9": Region("Niedersachsen", "DE", ("Lower Saxony",)),
    "DEA": Region("Nordrhein-Westfalen", "DE", ("North Rhine-Westphalia",)),
    "DEB": Region("Rheinland-Pfalz", "DE", ("Rhineland-Palatinate",)),
    "DEC": Region("Saarland", "DE"),
    "DED": Region("Sachsen", "DE", ("Saxony",)),
    "DEE": Region("Sachsen-Anhalt", "DE", ("Saxony-Anhalt",)),
    "DEF": Region("Schleswig-Holstein", "DE"),
    "DEG": Region("Thüringen", "DE", ("Thuringia",)),
    "FR1": Region("Île-de-France", "FR"),
    "FRB": Region("Centre-Val de Loire", "FR"),
    "FRC": Region("Bourgogne-Franche-Comté", "FR"),
    "FRD": Region("Normandie", "FR", ("Normandy",)),
    "FRE": Region("Hauts-de-France", "FR"),
    "FRF": Region("Grand Est", "FR"),
    "FRG": Region("Pays de la Loire", "FR"),
    "FRH": Region("Bretagne", "FR", ("Brittany",)),
    "FRI": Region("Nouvelle-Aquitaine", "FR"),
    "FRJ": Region("Occitanie", "FR"),
    "FRK": Region("Auvergne-Rhône-Alpes", "FR"),
    "FRL": Region("Provence-Alpes-Côte d'Azur", "FR"),
    "FRM": Region("Corse", "FR", ("Corsica",)),
    "FRY1": Region("Guadeloupe", "FR"),
    "FRY2": Region("Martinique", "FR"),
    "FRY3": Region("Guyane", "FR", ("French Guiana",)),
    "FRY4": Region("La Réunion", "FR", ("Réunion",)),
    "FRY5": Region("Mayotte", "FR"),
    "AT3": Region("Westösterreich", "AT", ("Western Austria", "Tirol")),
}

STATE_TO_NUTS = {region.name: code for code, region in REGIONS.items() if region.country == "DE"}
STATE_TO_NUTS["Tirol"] = "AT3"  # a few DWD stations are across the border

# INSEE region code (Hub'eau's code_region) -> NUTS 1
INSEE_TO_NUTS = {
    "11": "FR1", "24": "FRB", "27": "FRC", "28": "FRD", "32": "FRE", "44": "FRF", "52": "FRG", "53": "FRH",
    "75": "FRI", "76": "FRJ", "84": "FRK", "93": "FRL", "94": "FRM",
    "01": "FRY1", "02": "FRY2", "03": "FRY3", "04": "FRY4", "06": "FRY5",
}


def broader(variable: str) -> list[str]:
    """The variable and every variable it is a kind of, most specific first."""
    parts = variable.split(".")
    return [".".join(parts[:n]) for n in range(len(parts), 0, -1)]
