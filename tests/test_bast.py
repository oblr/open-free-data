"""BASt traffic counts against the bast_files fixture (see conftest) and yearly zips made here."""

from __future__ import annotations

import json

import numpy as np
import pytest

from conftest import bast_row, bast_zip
from free_data import bast, cache
from free_data.plugins.de_bast_traffic_counts import PLUGIN


def utc(text: str) -> int:
    return int(np.datetime64(text, "ms").astype(np.int64))


OK = [(100, "-"), (80, "-"), (10, "-"), (8, "-")]

# Köln-Nord: three hours of 2 January (CET); Eppelborn is on a federal road, so not in a motorway file.
JANUARY = [bast_row(5113, "240102", h, [(100 * h, "-"), (80 * h, "-"), (10, "-"), (8, "-")]) for h in (1, 2, 3)]


def test_stations_from_the_station_list(bast_files):
    found = {s.id: s for s in bast.stations()}
    assert set(found) == {"5113", "0223"}  # four digits; not the Landesstraße, not the empty row
    koeln = found["5113"]
    assert (koeln.name, koeln.state, koeln.road_class, koeln.road) == ("Köln-Nord (O)", "NW", "A", "A 1")
    assert (koeln.lat, koeln.lon, koeln.direction1, koeln.direction2) == (51.0123, 6.9456, "Dortmund", "Koblenz")


def test_years_and_road_classes(bast_files):
    assert bast.years() == {2023: ("A",), 2024: ("A", "B")}


def test_clock_changes():
    rows = [
        bast_row(5113, "240331", 2, OK),  # 01:00 CET
        bast_row(5113, "240331", 3, [(0, "z")] * 4),  # 02:00 to 03:00 doesn't exist
        bast_row(5113, "240331", 4, OK),  # 03:00 CEST
        bast_row(5113, "241027", 2, OK),  # 01:00 CEST
        bast_row(5113, "241027", 3, [(150, "z")] * 4),  # the two hours from 02:00 in one row
        bast_row(5113, "241027", 4, OK),  # 03:00 CET
    ]
    time, values = bast.parse_year(bast_zip(rows), 2024)["5113"]
    assert time.tolist() == [utc("2024-03-31T00:00"), utc("2024-03-31T01:00"),
                             utc("2024-10-26T23:00"), utc("2024-10-27T02:00")]
    assert values.tolist() == [[100, 80, 10, 8]] * 4


def test_missing_faulty_and_uncounted_are_nan_estimates_kept():
    rows = [bast_row(5113, "240102", 1, [(100, "a"), (80, "d"), (-1, "x"), (8, "s")]),
            bast_row(5113, "240102", 2, [(100, "u"), (80, "k"), (10, "-"), (8, "-")])]
    _, values = bast.parse_year(bast_zip(rows), 2024)["5113"]
    np.testing.assert_array_equal(values, [[np.nan, np.nan, np.nan, 8], [100, 80, 10, 8]])


def test_rows_that_are_not_fixed_width_read_the_same():
    fixed = bast.parse_year(bast_zip(JANUARY + [bast_row(223, "240102", 1, OK)]), 2024)
    loose = bast.parse_year(bast_zip([r.replace("  ", "") for r in JANUARY + [bast_row(223, "240102", 1, OK)]]),
                            2024)
    assert set(fixed) == set(loose) == {"5113", "0223"}
    for station in fixed:
        np.testing.assert_array_equal(fixed[station][0], loose[station][0])
        np.testing.assert_array_equal(fixed[station][1], loose[station][1])


def test_blocks_split_mid_file(monkeypatch):
    monkeypatch.setattr(bast, "BLOCK", 300)  # a little over a row: every block ends inside one
    time, values = bast.parse_year(bast_zip(JANUARY), 2024)["5113"]
    assert time.tolist() == [utc("2024-01-01T23:00"), utc("2024-01-02T00:00"), utc("2024-01-02T01:00")]
    assert values[:, 0].tolist() == [100, 200, 300]


def test_read_reduces_a_year_once(bast_files, monkeypatch):
    urls = []

    def download(url):
        urls.append(url)
        return bast_zip(JANUARY)

    monkeypatch.setattr(cache, "_download", download)
    table = bast.read(["5113/KFZ", "5113/kfz_r2", "5113/LKW"], "2024-01-02", "2024-01-02T01:59")
    assert urls == ["https://www.bast.de/videos/2024_A_S.zip"]
    assert table.columns == ("5113/KFZ", "5113/kfz_r2", "5113/LKW")
    assert table.time.astype(np.int64).tolist() == [utc("2024-01-01T23:00"), utc("2024-01-02T00:00")]
    assert table.values.tolist() == [[180, 80, 18], [360, 160, 18]]
    # the download is gone, one file per station and a list of the stations stay
    assert not (bast_files / "2024/A.zip").exists()
    assert json.loads((bast_files / "2024/A.json").read_text()) == ["5113"]
    assert (bast_files / "2024/5113.npz").exists()
    again = bast.read(["5113/KFZ"], "2024-01-02", "2024-01-02T23:59")
    assert urls == ["https://www.bast.de/videos/2024_A_S.zip"]
    assert again.values[:, 0].tolist() == [180, 360, 540]


def test_only_published_years_of_the_road_class_are_read(bast_files, monkeypatch):
    urls = []

    def download(url):
        urls.append(url)
        return bast_zip([bast_row(5113, "230615", 12, OK)], "2023_A_S.txt") if "2023" in url else bast_zip([])

    monkeypatch.setattr(cache, "_download", download)
    table = bast.read(["5113/KFZ"], "2022-01-01", "2025-12-31")
    assert urls == ["https://www.bast.de/videos/2023_A_S.zip", "https://www.bast.de/videos/2024_A_S.zip"]
    assert table.values[:, 0].tolist() == [180]
    assert bast.read(["0223/KFZ"], "2023-01-01", "2023-12-31").values.size == 0  # no federal roads in 2023
    assert len(urls) == 2


def test_a_station_missing_from_its_years_file_has_no_values(bast_files, monkeypatch):
    monkeypatch.setattr(cache, "_download", lambda url: bast_zip(JANUARY))  # the B file without Eppelborn
    table = bast.read(["5113/KFZ", "0223/KFZ"], "2024-01-02", "2024-01-02T23:59")
    assert table.columns == ("5113/KFZ", "0223/KFZ")
    assert table.values[:, 0].tolist() == [180, 360, 540]
    assert np.isnan(table.values[:, 1]).all()


def test_offline_without_the_year_says_so(bast_files):
    with cache.refreshing("offline"), pytest.raises(cache.NotCached):
        bast.read(["5113/KFZ"], "2024-01-01", "2024-01-31")


@pytest.mark.parametrize("name", ["5113", "5113/XYZ", "/KFZ"])
def test_bad_names(bast_files, name):
    with pytest.raises(ValueError, match="station"):
        bast.read([name], "2024-01-01", "2024-01-02")


def test_unknown_station(bast_files):
    with pytest.raises(ValueError, match="no BASt counting station 9999"):
        bast.read(["9999/KFZ"], "2024-01-01", "2024-01-02")


def test_plugin_datasets(bast_files):
    found = {d.id: d for d in PLUGIN.datasets()}
    assert len(found) == 2 * len(bast.VARIABLES)
    kfz, lkw, r1 = found["5113/KFZ"], found["5113/LKW"], found["5113/KFZ_R1"]
    assert (kfz.variable, kfz.measure, kfz.unit, kfz.resolution) == ("traffic.vehicles", "count", "count", "1h")
    assert lkw.variable == "traffic.vehicles.trucks"
    assert kfz.group == "All vehicles / Motorways / Nordrhein-Westfalen"
    assert found["0223/KFZ"].group == "All vehicles / Federal roads / Saarland"
    assert r1.label == "Köln-Nord (O), A 1, towards Dortmund"
    assert (kfz.place.region, kfz.place.lat) == ("DEA", 51.0123)
    assert "up to 2024" in kfz.description
