"""Which series relate to which, computed from their metadata (see vocabulary.py).

Nothing here names a single dataset: every relation follows from the
datasets' variables, kinds and places, so a new plugin's series relate to
the existing ones as soon as it fills those fields in.

    same_place      other variables measured at the same station
    forecast        a forecast and what it forecasts, at the same place
    same_variable   the same variable (or a broader or narrower one) elsewhere:
                    the nearest stations, or other countries
    driver          series whose variables LINKS connect, at the same place,
                    nearby (stations within NEARBY_KM) or in the same country

Each relation comes with a reason in words, for the page to show.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from free_data.sources import Dataset
from free_data.vocabulary import COUNTRIES, VARIABLES, broader

NEARBY_KM = 50.0  # how far a station's drivers may be from it
SAME_SPOT_KM = 0.05  # closer than this is the same station
LIMITS = {"same_place": 12, "forecast": 4, "same_variable": 8, "driver": 12}  # per selected series
ORDER = ("same_place", "forecast", "same_variable", "driver")

# Variables that drive or explain each other. Each also covers the variables
# below it: "electricity.generation.wind" includes onshore and offshore.
LINKS: tuple[tuple[str, str, str], ...] = (
    ("weather.sunshine_duration", "electricity.generation.solar", "Sunshine drives solar generation"),
    ("weather.wind_speed", "electricity.generation.wind", "Wind speed drives wind generation"),
    ("weather.air_temperature", "electricity.consumption", "Cold and heat drive electricity demand"),
    ("weather.precipitation", "water.discharge", "Rain feeds river flow"),
    ("weather.precipitation", "water.level", "Rain raises water levels"),
    ("weather.air_temperature", "water.temperature", "Air temperature warms rivers"),
    ("water.discharge", "electricity.generation.hydro", "River flow drives hydro generation"),
    ("water.temperature", "electricity.generation.nuclear", "Warm rivers limit the cooling of nuclear plants"),
    ("weather.wind_speed", "air", "Wind disperses air pollution"),
    ("weather.sunshine_duration", "air.o3", "Sunshine forms ozone"),
    ("electricity.generation.wind", "electricity.residual_load", "Wind lowers the residual load"),
    ("electricity.generation.solar", "electricity.residual_load", "Solar lowers the residual load"),
    ("electricity.consumption", "electricity.residual_load", "Residual load is consumption minus wind and solar"),
    ("electricity.residual_load", "electricity.price.day_ahead", "Residual load sets the price"),
    ("electricity.generation.wind_and_solar", "electricity.price.day_ahead", "Wind and solar push prices down"),
    ("electricity.generation.coal", "electricity.co2_intensity", "Coal raises the CO2 intensity"),
    ("electricity.generation.gas", "electricity.co2_intensity", "Gas raises the CO2 intensity"),
    ("electricity.price.day_ahead", "electricity.exchange", "Price differences drive cross-border exchange"),
)


@dataclass(frozen=True)
class Entry:
    source: str
    dataset: Dataset

    @property
    def key(self) -> str:
        return f"{self.source}:{self.dataset.id}"


@dataclass(frozen=True)
class Relation:
    key: str  # the related dataset, "<source>:<id>"
    to: str  # the selected dataset it relates to
    relation: str  # one of ORDER
    reason: str
    distance_km: float | None = None


def related(targets: list[Entry], entries: list[Entry]) -> list[Relation]:
    """What relates to the targets, best first per kind; none of the targets themselves."""
    selected = {t.key for t in targets}
    seen: dict[str, Relation] = {}
    for target in targets:
        for found in _related(target, entries):
            if found.key not in selected and found.key not in seen:
                seen[found.key] = found
    return sorted(seen.values(), key=lambda r: ORDER.index(r.relation))


def _related(target: Entry, entries: list[Entry]) -> list[Relation]:
    t = target.dataset
    if not t.variable:
        return []
    found: dict[str, list[tuple[tuple, Relation]]] = defaultdict(list)  # relation -> (rank, relation)
    t_broader = broader(t.variable)
    for entry in entries:
        d = entry.dataset
        if entry.key == target.key or not d.variable:
            continue
        km = _distance(t, d)
        same_spot = km is not None and km < SAME_SPOT_KM
        hits: list[tuple[str, tuple, str]] = []  # (relation, rank, reason)
        add = hits.append

        if d.variable == t.variable and d.kind != t.kind and _same_area(t, d, same_spot):
            add(("forecast", (0,), f"{d.kind.capitalize()} of the same"))
        elif same_spot:
            add(("same_place", (0, d.variable), f"Also measured at {d.place.name}"))
        elif d.kind == t.kind and _comparable(t.variable, d.variable) and not _same_area(t, d, same_spot):
            exact = d.variable == t.variable
            what = _label(d.variable) if exact else f"{_label(d.variable)}, comparable to {_label(t.variable)}"
            if km is not None:
                river = d.place.water and d.place.water == t.place.water
                add(("same_variable", (not exact, not river, km),
                     f"{what}, {km:.0f} km away" + (" on the same river" if river else "")))
            elif _is_area(t) and _is_area(d):
                add(("same_variable", (not exact, entry.source == target.source, d.place.name),
                     f"{what} in {d.place.name or _country(d.place.country)}"))

        why = _link(t_broader, d.variable)
        if why:
            rank = _driver_rank(t, d, km)
            if rank is not None:
                suffix = f", {km:.0f} km away" if km is not None and not same_spot else ""
                add(("driver", rank, f"{why}{suffix}"))
        distance = None if km is None else round(km, 1)
        for kind, rank, reason in hits:
            found[kind].append((rank, Relation(entry.key, target.key, kind, reason, distance)))

    results = []
    for kind in ORDER:
        ranked = sorted(found[kind], key=lambda pair: pair[0])
        if kind == "driver" and _is_area(t):
            ranked = _spread(ranked, entries)  # a country-wide series: a few stations across the country
        results += [relation for _, relation in ranked[:LIMITS[kind]]]
    return results


def _label(variable: str) -> str:
    return VARIABLES[variable].label if variable in VARIABLES else variable


def _country(code: str) -> str:
    return COUNTRIES[code].name if code in COUNTRIES else code or "several countries"


def _comparable(a: str, b: str) -> bool:
    """The same variable, or one a kind of the other below the top two levels ("electricity.generation")."""
    if a == b:
        return True
    shorter, longer = sorted((a, b), key=len)
    return shorter in broader(longer) and shorter.count(".") >= 2


def _is_area(d: Dataset) -> bool:
    """A series for a whole area, like a country, rather than a station (which may lack coordinates)."""
    return d.place is not None and d.place.lat is None and not d.place.water


def _same_area(t: Dataset, d: Dataset, same_spot: bool) -> bool:
    """The same station, or the same area (same country and name)."""
    if same_spot:
        return True
    if not (_is_area(t) and _is_area(d)):
        return False
    return (t.place.country, t.place.name) == (d.place.country, d.place.name)


def _link(t_broader: list[str], variable: str) -> str:
    d_broader = broader(variable)
    for a, b, why in LINKS:
        if (a in t_broader and b in d_broader) or (b in t_broader and a in d_broader):
            return why
    return ""


def _driver_rank(t: Dataset, d: Dataset, km: float | None) -> tuple | None:
    """How good a linked series is as a driver or effect of the target; None if too far."""
    if not (t.place and d.place):
        return None
    if km is not None:
        return (0, km) if km <= NEARBY_KM else None
    if (_is_area(t) or _is_area(d)) and t.place.country and t.place.country == d.place.country:
        # an area and something in it: other area series first, then stations with the longest history
        return (1, not _is_area(d), d.since or "9999", d.place.name)
    return None


def _spread(ranked: list[tuple[tuple, Relation]], entries: list[Entry]) -> list[tuple[tuple, Relation]]:
    """Every area series, then at most one station per region (or river), regions with many stations first."""
    datasets = {e.key: e.dataset for e in entries}
    areas = [pair for pair in ranked if _is_area(datasets[pair[1].key])]
    stations = [pair for pair in ranked if not _is_area(datasets[pair[1].key])]

    def group(r: Relation) -> str:
        place = datasets[r.key].place
        return place.region or place.water or place.name if place else ""

    sizes: dict[str, int] = defaultdict(int)
    for _, r in stations:
        sizes[group(r)] += 1
    best: dict[str, tuple[tuple, Relation]] = {}
    for rank, r in stations:
        best.setdefault(group(r), (rank, r))
    return areas + sorted(best.values(), key=lambda pair: (-sizes[group(pair[1])], pair[0]))


def _distance(a: Dataset, b: Dataset) -> float | None:
    """Kilometres between two stations; None unless both have coordinates."""
    if not (a.place and b.place) or None in (a.place.lat, a.place.lon, b.place.lat, b.place.lon):
        return None
    lat1, lon1, lat2, lon2 = map(math.radians, (a.place.lat, a.place.lon, b.place.lat, b.place.lon))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(min(1.0, h)))
