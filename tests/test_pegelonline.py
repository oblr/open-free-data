"""PEGELONLINE against a two-station list and hand-made measurements; "now" is 2024-01-10."""

from __future__ import annotations

import numpy as np
import pytest

import cache
import pegelonline
from cache import FetchError
from conftest import STATIONS, put
from plugins.de_pegelonline import PLUGIN
from sources import epoch_ms



def ms(text: str) -> int:
    return epoch_ms(text, pegelonline.TZ)


def stamps(start: str, count: int, minutes: int) -> list[str]:
    """Berlin-time ISO stamps as PEGELONLINE writes them (CET in January)."""
    return [str(np.datetime64(ms(start) + 3_600_000 + i * minutes * 60_000, "ms").astype("datetime64[s]")) + "+01:00"
            for i in range(count)]


@pytest.fixture
def files(cache_root, pegelonline_stations, monkeypatch):
    monkeypatch.setattr(pegelonline, "_now", lambda: ms("2024-01-10T12:00"))
    level = [{"timestamp": t, "value": 100.0 + i} for i, t in enumerate(stamps("2024-01-09T00:00", 8, 15))]
    level[2]["value"] = None
    put(cache_root, pegelonline.SOURCE, "2730010/W.json", level)
    flow = [{"timestamp": t, "value": 500.0} for t in stamps("2024-01-09T00:30", 2, 30)]
    put(cache_root, pegelonline.SOURCE, "2730010/Q.json", flow)


def test_native_values_on_the_union_of_times(files):
    table = pegelonline.read(["2730010/W", "2730010/Q"], "2024-01-09T00:15", "2024-01-09T01:00", "native")
    assert table.columns == ("2730010/W", "2730010/Q")
    assert (table.time.astype(np.int64) - ms("2024-01-09T00:15")).tolist() == [0, 900_000, 1_800_000, 2_700_000]
    np.testing.assert_array_equal(table.values, [[101, np.nan], [np.nan, 500], [103, np.nan], [104, 500]])


def test_hourly_and_daily_means(files):
    hourly = pegelonline.read(["2730010/W"], "2024-01-09", None)
    assert hourly.values[:, 0].tolist() == [(100 + 101 + 103) / 3, 105.5]  # the null at 00:30 is skipped
    daily = pegelonline.read(["2730010/W"], None, None, "day")
    assert daily.time.astype(np.int64).tolist() == [ms("2024-01-09")]


def test_ranges_older_than_pegelonline_keeps_are_not_fetched(cache_root, monkeypatch):
    monkeypatch.setattr(pegelonline, "_now", lambda: ms("2024-01-10T12:00"))
    table = pegelonline.read(["2730010/W"], "2023-11-01", "2023-11-30", "native")  # nothing cached either
    assert table.values.shape == (0, 1)


def test_unknown_series(files, monkeypatch):
    def missing(url: str) -> bytes:
        raise FetchError(url, 404, "HTTP 404")

    monkeypatch.setattr(cache, "_download", missing)
    with pytest.raises(pegelonline.PegelonlineError, match="no series 'XX' at station '2730010'"):
        pegelonline.read(["2730010/XX"])


@pytest.mark.parametrize(
    ("args", "message"),
    [((["KÖLN"],), "station number"), ((["2730010/W"], None, None, "minute"), "resolution"),
     ((["2730010/W"], "2024-01-09", "2024-01-08"), "after")],
)
def test_bad_arguments(args, message):
    with pytest.raises(ValueError, match=message):
        pegelonline.read(*args)


def test_resolve_stations_by_number_name_or_uuid():
    assert pegelonline.resolve("köln/Q", STATIONS) == "2730010/Q"
    assert pegelonline.resolve("2710080", STATIONS) == "2710080/W"
    assert pegelonline.resolve("u-bonn/XYZ", STATIONS) == "2710080/XYZ"
    with pytest.raises(ValueError, match="no station"):
        pegelonline.resolve("BERLIN/W", STATIONS)
    with pytest.raises(ValueError, match="ambiguous"):
        pegelonline.resolve("KÖLN", [*STATIONS, {**STATIONS[0], "number": "1", "uuid": "other"}])


def test_cli_list_and_read(files, capsys):
    pegelonline.main(["--list", "rhein köln"])
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 1 and out[0].split() == ["2730010", "KÖLN", "RHEIN", "W", "Q"]
    pegelonline.main(["KÖLN/W", "--start", "2024-01-09", "--end", "2024-01-09"])
    assert capsys.readouterr().out.splitlines()[1] == f"2024-01-09T00:00:00+01:00,{(100 + 101 + 103) / 3!r}"


def test_plugin_datasets(files):
    datasets = {d.id: d for d in PLUGIN.datasets()}
    assert list(datasets) == ["2710080/XYZ", "2730010/W", "2730010/Q"]  # along the river
    level = datasets["2730010/W"]
    assert (level.label, level.group, level.unit) == ("Köln", "Water level / Rhein", "cm")
    assert "km 688" in level.description and "Gauge zero 35.038" in level.description
    assert level.link.endswith("pegelnr=2730010")
    assert datasets["2710080/XYZ"].group == "Sichtweite / Rhein"  # unknown kinds keep PEGELONLINE's name
