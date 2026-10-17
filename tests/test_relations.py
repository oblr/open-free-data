"""relations.py on a few hand-made datasets."""

from __future__ import annotations

from free_data import relations
from free_data.relations import Entry
from free_data.sources import Dataset, Place

GERMANY = Place(country="DE", name="Germany")
FRANCE = Place(country="FR", name="Mainland France")


def station(name: str, lat: float, lon: float, region: str = "DE4", water: str = "") -> Place:
    return Place(country="DE", region=region, name=name, lat=lat, lon=lon, water=water)


POTSDAM = station("Potsdam", 52.381, 13.062)
BERLIN = station("Berlin", 52.465, 13.302, region="DE3")  # 19 km from Potsdam
MUNICH = station("München", 48.137, 11.575, region="DE2")  # far
KOELN = station("Köln", 50.937, 6.963, region="DEA", water="Rhein")
BONN = station("Bonn", 50.737, 7.098, region="DEA", water="Rhein")
NO_COORDS = Place(country="DE", name="Gauge without coordinates", water="Spree")


def entry(source: str, id: str, variable: str, place: Place, kind: str = "observation", since: str = "") -> Entry:
    return Entry(source, Dataset(id=id, label=id, variable=variable, place=place, kind=kind, since=since))


ENTRIES = [
    entry("smard", "solar", "electricity.generation.solar", GERMANY),
    entry("smard", "solar-forecast", "electricity.generation.solar", GERMANY, kind="forecast"),
    entry("smard", "wind-onshore", "electricity.generation.wind.onshore", GERMANY),
    entry("smard", "residual", "electricity.residual_load", GERMANY),
    entry("smard", "price", "electricity.price.day_ahead", Place(country="DE", name="Germany/Luxembourg")),
    entry("eco2mix", "solar", "electricity.generation.solar", FRANCE),
    entry("eco2mix", "wind", "electricity.generation.wind", FRANCE),
    entry("dwd", "potsdam/sun", "weather.sunshine_duration", POTSDAM, since="1893-01-01"),
    entry("dwd", "potsdam/temp", "weather.air_temperature", POTSDAM),
    entry("dwd", "berlin/sun", "weather.sunshine_duration", BERLIN, since="1990-01-01"),
    entry("dwd", "berlin/temp", "weather.air_temperature", BERLIN),
    entry("dwd", "munich/sun", "weather.sunshine_duration", MUNICH, since="1950-01-01"),
    entry("dwd", "munich/temp", "weather.air_temperature", MUNICH),
    entry("dwd", "koeln/rain", "weather.precipitation", station("Köln-Stammheim", 50.96, 6.99, region="DEA")),
    entry("gauges", "koeln/Q", "water.discharge", KOELN),
    entry("gauges", "koeln/W", "water.level", KOELN),
    entry("gauges", "bonn/Q", "water.discharge", BONN),
    entry("gauges", "nowhere/WT", "water.temperature", NO_COORDS),
]
BY_KEY = {e.key: e for e in ENTRIES}


def related(*keys: str) -> dict[str, relations.Relation]:
    return {r.key: r for r in relations.related([BY_KEY[k] for k in keys], ENTRIES)}


def test_country_wide_series():
    found = related("smard:solar")
    assert found["smard:solar-forecast"].relation == "forecast"
    assert found["eco2mix:solar"].relation == "same_variable"
    assert found["eco2mix:solar"].reason == "Solar in Mainland France"
    assert found["smard:residual"].relation == "driver"
    # stations for a country-wide series: one per region, longest history first within a region
    sun = [k for k, r in found.items() if r.relation == "driver" and k.startswith("dwd:")]
    assert sorted(sun) == ["dwd:berlin/sun", "dwd:munich/sun", "dwd:potsdam/sun"]
    assert "dwd:potsdam/temp" not in found  # temperature doesn't drive solar


def test_broader_and_narrower_variables_compare():
    found = related("eco2mix:wind")
    assert found["smard:wind-onshore"].reason == "Wind onshore, comparable to Wind in Germany"
    assert "smard:solar" not in found  # siblings under electricity.generation are not comparable


def test_stations_nearby_and_at_the_same_place():
    found = related("dwd:potsdam/temp")
    assert found["dwd:potsdam/sun"].relation == "same_place"
    assert found["dwd:berlin/temp"].relation == "same_variable"
    assert found["dwd:berlin/temp"].reason == "Air temperature, 19 km away"
    assert found["dwd:berlin/temp"].distance_km == 18.8
    assert "dwd:munich/temp" in found  # the same variable has no distance limit, only an order
    assert "gauges:nowhere/WT" not in found  # a station without coordinates is not an area


def test_drivers_of_a_station_are_nearby():
    found = related("gauges:koeln/Q")
    assert found["gauges:koeln/W"].relation == "same_place"
    assert found["gauges:bonn/Q"].reason == "River discharge, 24 km away on the same river"
    assert found["dwd:koeln/rain"].reason == "Rain feeds river flow, 3 km away"
    assert not any(k.startswith("smard:") for k in found)


def test_order_limits_and_no_selected_series():
    found = relations.related([BY_KEY["smard:solar"], BY_KEY["eco2mix:solar"]], ENTRIES)
    keys = [r.key for r in found]
    assert "smard:solar" not in keys and "eco2mix:solar" not in keys
    assert len(keys) == len(set(keys))
    order = [relations.ORDER.index(r.relation) for r in found]
    assert order == sorted(order)


def test_datasets_without_a_variable_have_no_relations():
    lonely = Entry("x", Dataset(id="y", label="y"))
    assert relations.related([lonely], [*ENTRIES, lonely]) == []
