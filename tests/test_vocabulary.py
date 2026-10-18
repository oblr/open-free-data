"""The vocabulary is consistent, and every plugin describes every dataset with it."""

from __future__ import annotations

import json
import re
from dataclasses import asdict

import pytest

from free_data import sources, vocabulary
from free_data.units import DIMENSIONS, UNITS
from free_data.vocabulary import KINDS, MEASURES, QUALITIES, REGIONS, VARIABLES


def test_every_broader_variable_exists():
    for variable in VARIABLES:
        for broader in vocabulary.broader(variable):
            assert broader in VARIABLES, f"{variable} needs {broader}"


def test_variables_are_well_formed():
    for key, variable in VARIABLES.items():
        assert re.fullmatch(r"[a-z0-9_]+(\.[a-z0-9_]+)*", key), key
        assert variable.label and all(s.strip() for s in variable.synonyms), key
        assert variable.standard == "" or re.fullmatch(r"(cf:[a-z0-9_]+|entsoe:B\d\d)", variable.standard), key


def test_region_tables_point_into_regions():
    assert all(re.fullmatch(r"[A-Z]{2}", r.country) for r in REGIONS.values())
    assert all(code in REGIONS for code in vocabulary.STATE_TO_NUTS.values())
    assert all(code in REGIONS for code in vocabulary.INSEE_TO_NUTS.values())
    assert len(vocabulary.STATE_TO_NUTS) == 17  # the 16 states, and Tirol


def test_broader():
    assert vocabulary.broader("electricity.generation.coal.lignite") == [
        "electricity.generation.coal.lignite", "electricity.generation.coal", "electricity.generation", "electricity"]


def test_measures_point_into_dimensions():
    for key, measure in MEASURES.items():
        assert all(d in DIMENSIONS for d in measure.dimensions), key
    assert MEASURES["power"].units == ("MW",)
    assert MEASURES["level"].units == ("m", "m+NN", "m+NHN")


@pytest.fixture
def every_dataset(
    pegelonline_stations, hubeau_files, dwd_files, uba_files, bast_files
) -> list[tuple[str, sources.Dataset]]:
    return [(source, d) for source, plugin in sources.discover().items() for d in plugin.datasets()]


def test_every_dataset_uses_the_vocabulary(every_dataset):
    assert len({source for source, _ in every_dataset}) == len(sources.discover())
    for source, d in every_dataset:
        where = f"{source}:{d.id}"
        assert d.variable in VARIABLES, where
        assert d.measure in MEASURES, where
        assert d.unit in UNITS, where
        allowed = MEASURES[d.measure].dimensions
        assert not allowed or UNITS[d.unit].dimension in allowed, f"{where}: {d.unit} isn't a {d.measure}"
        assert d.kind in KINDS, where
        assert d.quality in QUALITIES, where
        assert d.resolution == "" or re.fullmatch(r"\d+(min|h|d)", d.resolution), where
        assert d.since == "" or re.fullmatch(r"\d{4}-\d{2}-\d{2}", d.since), where
        place = d.place
        assert place is not None, where
        assert place.country == "" or re.fullmatch(r"[A-Z]{2}", place.country), where
        if place.region:
            assert REGIONS[place.region].country == place.country, where
        if place.lat is not None or place.lon is not None:
            assert -90 <= place.lat <= 90 and -180 <= place.lon <= 180, where


def test_the_same_thing_has_the_same_variable_across_sources(every_dataset):
    by_key = {f"{source}:{d.id}": d for source, d in every_dataset}
    assert by_key["de-smard:4068"].variable == by_key["fr-rte-eco2mix:solaire"].variable
    assert by_key["de-pegelonline:2730010/Q"].variable == by_key["fr-hubeau-hydrometrie:F700000103/Q"].variable
    assert by_key["de-dwd-opendata:03987/TT_TU"].variable == "weather.air_temperature"
    # French coal is one series, German coal two; they meet one level up
    lignite, coal = by_key["de-smard:1223"].variable, by_key["fr-rte-eco2mix:charbon"].variable
    assert coal in vocabulary.broader(lignite)
    # a forecast is the same variable as what it forecasts
    assert by_key["de-smard:411"].variable == by_key["de-smard:410"].variable
    assert (by_key["de-smard:411"].kind, by_key["de-smard:410"].kind) == ("forecast", "observation")


def test_places(every_dataset):
    by_key = {f"{source}:{d.id}": d for source, d in every_dataset}
    potsdam = by_key["de-dwd-opendata:03987/TT_TU"].place
    assert (potsdam.country, potsdam.region, potsdam.lat) == ("DE", "DE4", 52.3813)
    paris = by_key["fr-hubeau-hydrometrie:F700000103/Q"].place
    assert (paris.region, paris.water, paris.lat) == ("FR1", "La Seine", 48.8447)
    assert by_key["fr-hubeau-hydrometrie:F700000103/Q"].since == "2006-01-26"
    assert by_key["de-pegelonline:2730010/W"].place.water == "Rhein"
    assert by_key["de-smard:254"].place.country == "FR"  # the French price, read from SMARD


def test_the_metadata_reaches_the_page(every_dataset):
    """browser.py sends asdict(Dataset): the place is a nested object, and it all serialises."""
    row = asdict(next(d for _, d in every_dataset if d.place and d.place.lat is not None))
    assert set(row["place"]) == {"country", "region", "name", "lat", "lon", "water"}
    json.dumps(row)
