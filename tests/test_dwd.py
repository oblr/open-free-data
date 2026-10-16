"""DWD against the files of the dwd_files fixture (see conftest)."""

from __future__ import annotations

import numpy as np
import pytest

import cache
import dwd
from cache import FetchError
from plugins.de_dwd_opendata import PLUGIN


def utc(text: str) -> int:
    return int(np.datetime64(text, "ms").astype(np.int64))


def test_historical_and_recent_join_and_the_historical_wins(dwd_files):
    table = dwd.read(["03987/TT_TU"], "2023-12-31T22:00+00:00", "2024-01-01T01:00+00:00")
    assert table.time.astype(np.int64).tolist() == [utc(f"2023-12-31T{h}") for h in ("22", "23")] + [
        utc("2024-01-01T00"), utc("2024-01-01T01")]
    np.testing.assert_array_equal(table.values[:, 0], [1.0, np.nan, 3.0, 11.0])  # -999 is NaN; 3.0 not 99.0


def test_two_variables_of_one_file(dwd_files):
    table = dwd.read(["3987/RF_TU", "03987/TT_TU"], "2024-01-01T01:00+00:00", "2024-01-01T01:00+00:00")
    assert table.columns == ("3987/RF_TU", "03987/TT_TU") and table.values.tolist() == [[90.0, 11.0]]


def test_old_range_needs_no_recent_file(dwd_files):
    (dwd_files / "air_temperature/recent/03987.zip").unlink()  # a download would fail the test
    assert len(dwd.read(["03987/TT_TU"], "2023-12-31T00:00+00:00", "2023-12-31T23:00+00:00").time) == 2


def test_daily_sums_means_and_direction(dwd_files):
    table = dwd.read(["03987/R1", "03987/F", "03987/D"], "2024-01-02", "2024-01-02T23:59", "day")
    rain, speed, direction = table.values[0]
    assert rain == 2.0 and speed == 3.0
    assert min(direction, 360 - direction) < 1e-9  # 350° and 10° average to north, not 180°


def test_a_station_without_a_recent_file_just_has_no_recent_data(dwd_files, monkeypatch):
    def missing(url: str) -> bytes:
        raise FetchError(url, 404, "HTTP 404")

    monkeypatch.setattr(cache, "_download", missing)
    (dwd_files / "sun/recent/03987.zip").unlink() if (dwd_files / "sun/recent/03987.zip").exists() else None
    assert dwd.read(["03987/SD_SO"], "2024-01-02").values.shape == (0, 1)


def test_reporting_stations_and_their_list(dwd_files):
    rows = dwd.stations("wind")
    assert [(r["id"], r["name"], r["state"]) for r in rows] == [
        ("00003", "Aachen", "Nordrhein-Westfalen"), ("03987", "Potsdam Telegrafenberg", "Brandenburg")]
    assert [r["id"] for r in dwd.reporting("wind")] == ["03987"]


@pytest.mark.parametrize(
    ("args", "message"),
    [((["03987/XX"],), "variable"), ((["03987"],), "variable"),
     ((["03987/TT_TU"], None, None, "week"), "resolution"),
     ((["03987/TT_TU"], "2024-01-02", "2024-01-01"), "after")],
)
def test_bad_arguments(args, message):
    with pytest.raises(ValueError, match=message):
        dwd.read(*args)


def test_cli_list(dwd_files, capsys):
    dwd.main(["--list", "potsdam"])
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 1 and out[0].split()[0] == "03987" and out[0].endswith("TT_TU RF_TU R1 F D SD_SO P")


def test_plugin_datasets(dwd_files):
    datasets = {d.id: d for d in PLUGIN.datasets()}
    assert len(datasets) == len(dwd.VARIABLES)  # Potsdam only: Aachen no longer reports
    temperature = datasets["03987/TT_TU"]
    assert (temperature.label, temperature.group, temperature.unit) == (
        "Potsdam Telegrafenberg", "Air temperature / Brandenburg", "°C")
    assert "since 1893-01-01" in temperature.description
