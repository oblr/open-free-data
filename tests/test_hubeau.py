"""Hub'eau against the files of the hubeau_files fixture (see conftest)."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import paris_ms
from free_data import hubeau
from free_data.plugins.fr_hubeau_hydrometrie import PLUGIN


def test_native_is_sorted_and_in_cm_and_cubic_metres(hubeau_files):
    table = hubeau.read(["F700000103/Q", "F700000103/H"], "2024-01-08T00:00", "2024-01-08T00:30", "native")
    assert table.time.astype(np.int64).tolist() == [paris_ms("2024-01-08T00:00"), paris_ms("2024-01-08T00:30")]
    assert table.values.tolist() == [[100.0, 100.0], [100.001, 100.1]]


def test_hourly_means(hubeau_files):
    table = hubeau.read(["F700000103/H"], "2024-01-08T00:00", "2024-01-08T01:59", "hour")
    assert table.values[:, 0].tolist() == [100.05, 100.25]


def test_daily_flows_are_history_then_real_time(hubeau_files):
    table = hubeau.read(["F700000103/Q"], "2024-01-07", "2024-01-09T23:59", "day")
    assert table.time.astype(np.int64).tolist() == [paris_ms(d) for d in ("2024-01-07", "2024-01-08", "2024-01-09")]
    # published daily means for the 7th and 8th (not the real-time mean of the 8th), real time for the 9th
    assert table.values[:, 0].tolist() == [50.008, 50.009, (100_048 + 100_095) / 2 / 1000]


def test_daily_levels_have_no_history(hubeau_files):
    table = hubeau.read(["F700000103/H"], "2024-01-01", "2024-01-09T23:59", "day")
    assert len(table.time) == 2


def test_old_range_needs_no_real_time(hubeau_files, monkeypatch):
    (hubeau_files / "F700000103/Q.json").unlink()  # a download would fail the test
    monkeypatch.setattr(hubeau, "_now", lambda: paris_ms("2024-03-01"))
    table = hubeau.read(["F700000103/Q"], "2023-12-30", "2023-12-31T23:59", "day")
    assert table.values[:, 0].tolist() == [50.0, 50.001]


@pytest.mark.parametrize(
    ("args", "message"),
    [((["F700000103/X"],), "station code"), ((["F700000103"],), "station code"),
     ((["F700000103/Q"], None, None, "week"), "resolution"),
     ((["F700000103/Q"], "2024-01-02", "2024-01-01"), "after")],
)
def test_bad_arguments(args, message):
    with pytest.raises(ValueError, match=message):
        hubeau.read(*args)


def test_stations_follow_the_pages(hubeau_files):
    assert [s["code_station"] for s in hubeau.stations()] == ["F700000103", "D0000001", "C0000001"]


def test_plugin_datasets_by_station_type(hubeau_files):
    datasets = {d.id: d for d in PLUGIN.datasets()}
    assert sorted(datasets) == ["D0000001/Q", "F700000103/H", "F700000103/Q"]  # computed stations left out
    flow = datasets["F700000103/Q"]
    assert (flow.group, flow.unit) == ("Flow / La Seine", "m³/s")
    assert "reach back to 2006" in flow.description and flow.link.endswith("/F700000103/fiche")
    assert datasets["D0000001/Q"].group == "Flow / Other waters"
    assert "No history" in datasets["F700000103/H"].description
