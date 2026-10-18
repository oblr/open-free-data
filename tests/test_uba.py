"""UBA air data against the uba_files fixture (see conftest); "now" is 2024-01-10 12:00 Berlin."""

from __future__ import annotations

import json

import numpy as np
import pytest

from conftest import uba_rows
from free_data import cache, uba
from free_data.cache import FetchError
from free_data.plugins.de_uba_luftdaten import PLUGIN


def utc(text: str) -> int:
    return int(np.datetime64(text, "ms").astype(np.int64))


def test_one_scope_per_station_and_component(uba_files):
    found = {s.name: s for s in uba.catalog()}
    assert set(found) == {"DEBB021/NO2", "DEBB021/PM10", "DEUB001/PM10", "DEUB001/CO", "DEUB001/SO2"}
    # the hourly mean before the daily one, the daily before the 8-hour one; never a maximum
    assert [found[k].scope for k in ("DEBB021/NO2", "DEBB021/PM10", "DEUB001/PM10", "DEUB001/CO")] == [
        "1SMW", "1SMW", "1TMW", "8SMW"]
    assert found["DEBB021/PM10"].first == utc("2018-02-28T23:00")  # CET midnight
    assert not found["DEUB001/SO2"].reporting and found["DEUB001/CO"].reporting


def test_hourly_values_across_years_in_cet(uba_files):
    table = uba.read(["DEBB021/NO2"], "2023-12-31T22:00+01:00", "2024-01-01T02:00+01:00")
    assert table.columns == ("DEBB021/NO2",)
    # labelled by the start of the hour, CET: 2024-01-01 00:00 CET is 2023-12-31 23:00 UTC
    assert table.time.astype(np.int64).tolist() == [utc(f"2023-12-31T{h}:00") for h in (21, 22, 23)] + [
        utc("2024-01-01T00:00"), utc("2024-01-01T01:00")]
    np.testing.assert_array_equal(table.values[:, 0], [20, 21, 30, np.nan, 32])  # null is NaN


def test_cet_all_year_even_in_summer(uba_files, monkeypatch):
    (uba_files / "DEBB021/NO2/1SMW/2024.json").unlink()
    monkeypatch.setattr(uba, "_now", lambda: utc("2024-07-10T10:00"))
    monkeypatch.setattr(cache, "_download", lambda url: json.dumps(
        uba_rows("21", 5, 2, ["2024-07-01 12:00:00"], [40])).encode())
    table = uba.read(["DEBB021/NO2"], "2024-07-01", "2024-07-01T23:59")
    assert table.time.astype(np.int64).tolist() == [utc("2024-07-01T11:00")]  # 13:00 in Berlin, CEST


def test_daily_means_at_the_start_of_the_cet_day(uba_files):
    table = uba.read(["deub001/pm10"], "2024-01-01", "2024-01-02T23:59")  # names in any case
    assert table.time.astype(np.int64).tolist() == [utc("2023-12-31T23:00"), utc("2024-01-01T23:00")]
    assert table.values[:, 0].tolist() == [18, 6]


def test_two_series_and_daily_means(uba_files):
    table = uba.read(["DEBB021/NO2", "DEUB001/PM10"], "2024-01-01", "2024-01-01T23:59", "day")
    assert table.time.astype(np.int64).tolist() == [utc("2023-12-31T23:00")]
    assert table.values.tolist() == [[(30 + 32 + 33) / 3, 18]]


def test_only_the_years_asked_for_are_read(uba_files):
    # 2023 is the only cached year before 2024; a download of 2022 would fail the test
    assert len(uba.read(["DEBB021/NO2"], "2023-06-01", "2023-12-31T23:59").time) == 2


def test_last_year_is_final_from_july(uba_files, monkeypatch):
    fetched = []

    def download(url: str) -> bytes:
        fetched.append(url)
        return json.dumps(uba_rows("21", 5, 2, ["2023-12-31 23:00:00"], [99])).encode()

    monkeypatch.setattr(cache, "_download", download)
    monkeypatch.setattr(uba, "_now", lambda: utc("2024-07-01T10:00"))
    table = uba.read(["DEBB021/NO2"], "2023-12-31", "2023-12-31T23:59")
    assert table.values[:, 0].tolist() == [99]  # not the preliminary 21
    assert "date_from=2023-01-01" in fetched[0] and "station=21&component=5&scope=2" in fetched[0]
    assert (uba_files / "DEBB021/NO2/1SMW/2023.json").exists()


def test_unknown_series(uba_files):
    with pytest.raises(uba.UbaError, match="no series 'O3' at station 'DEBB021'"):
        uba.read(["DEBB021/O3"], "2024-01-01", "2024-01-02")
    with pytest.raises(uba.UbaError):
        uba.read(["DEUB001/PM10BAP"], "2024-01-01", "2024-01-02")  # in the meta data, but not offered
    with pytest.raises(ValueError, match="station code"):
        uba.read(["DEBB021"])


def test_the_api_refusing_a_series(uba_files, monkeypatch):
    def refused(url: str) -> bytes:
        raise FetchError(url, 409, "HTTP 409")

    monkeypatch.setattr(cache, "_download", refused)
    with pytest.raises(uba.UbaError, match="no CO at DEUB001 in 2024"):
        uba.read(["DEUB001/CO"], "2024-01-01", "2024-01-02")


def test_plugin_datasets(uba_files):
    datasets = {d.id: d for d in PLUGIN.datasets()}
    assert "DEUB001/SO2" not in datasets  # no longer reporting
    no2 = datasets["DEBB021/NO2"]
    assert (no2.label, no2.group, no2.unit) == ("Potsdam-Zentrum", "Nitrogen dioxide / Brandenburg", "µg/m³")
    assert (no2.variable, no2.measure, no2.resolution, no2.since) == ("air.no2", "concentration", "1h", "2016-01-01")
    assert (no2.place.region, no2.place.lat) == ("DE4", 52.4014)
    pm10 = datasets["DEUB001/PM10"]
    assert (pm10.group, pm10.resolution, pm10.place.region) == ("Particulate matter PM10 / Umweltbundesamt", "1d", "")
    assert datasets["DEUB001/CO"].unit == "mg/m³"


def test_plugin_read(uba_files, monkeypatch):
    monkeypatch.setattr(cache, "_download", lambda url: b'{"data": {}}')  # the years not cached
    table = PLUGIN.read(["DEBB021/NO2"], None, None, PLUGIN.resolve({}))
    assert table.values.shape == (6, 1)  # every year since 2016 is asked for, but only the cached ones exist

