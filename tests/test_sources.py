from __future__ import annotations

import numpy as np
import pytest

import sources
from sources import Option, Plugin, Table


def table(columns: tuple[str, ...], times: list[int], values: list[list[float]]) -> Table:
    return Table(columns, np.array(times, dtype="datetime64[ms]"), np.array(values, dtype=np.float64))


def test_combine_lines_up_on_the_union_of_times():
    a = table(("x",), [1, 3], [[1.0], [3.0]])
    b = table(("y", "z"), [2, 3], [[20.0, 200.0], [30.0, 300.0]])
    combined = sources.combine([a, b])
    assert combined.columns == ("x", "y", "z")
    np.testing.assert_array_equal(combined.time.astype(np.int64), [1, 2, 3])
    np.testing.assert_array_equal(
        combined.values, [[1.0, np.nan, np.nan], [np.nan, 20.0, 200.0], [3.0, 30.0, 300.0]]
    )


def test_combine_nothing():
    combined = sources.combine([])
    assert combined.columns == () and combined.values.shape == (0, 0)


def test_merge_prefixes_the_source():
    merged = sources.merge([("s1", table(("x",), [1], [[1.0]])), ("s2", table(("x",), [2], [[2.0]]))])
    assert merged.columns == ("s1:x", "s2:x")
    assert merged.values.shape == (2, 2)


class Fake(Plugin):
    id = "zz-fake"
    title = "Fake"
    options = (Option("size", "Size", ("s", "l"), "s"),)

    def datasets(self):
        return []

    def read(self, datasets, start, end, options):
        raise NotImplementedError


def test_resolve_fills_defaults_and_checks():
    assert Fake().resolve({}) == {"size": "s"}
    assert Fake().resolve({"size": "l", "other": "ignored"}) == {"size": "l"}
    with pytest.raises(ValueError, match="size must be one of"):
        Fake().resolve({"size": "xl"})


def test_catalog_lists_every_source_plugged_in_first():
    rows = sources.catalog()
    ids = [row["id"] for row in rows]
    assert len(ids) == len(set(ids)) == len(list(sources.DATA.glob("*.json")))
    plugged = [row["plugin"] for row in rows]
    assert plugged == sorted(plugged, reverse=True)
    smard = next(row for row in rows if row["id"] == "de-smard")
    assert smard["plugin"] and smard["country"] == "Germany"


def test_every_plugin_has_unique_dataset_ids(pegelonline_stations):
    for plugin in sources.discover().values():
        ids = [d.id for d in plugin.datasets()]
        assert len(ids) == len(set(ids)), plugin.id
        assert all(o.default in o.choices for o in plugin.options), plugin.id


def test_means_per_local_day_across_the_clock_change():
    zone = "Europe/Paris"
    hours = np.arange(sources.epoch_ms("2024-03-30T22:00", zone), sources.epoch_ms("2024-04-01T02:00", zone), 3_600_000)
    values = np.ones((len(hours), 1))
    values[0] = np.nan
    labels, means = sources.means(hours, values, "day", zone)
    assert labels.tolist() == [sources.epoch_ms(d, zone) for d in ("2024-03-30", "2024-03-31", "2024-04-01")]
    assert means[:, 0].tolist() == [1.0, 1.0, 1.0]  # the NaN is skipped; the 23-hour day is one bin


def test_means_per_hour_and_empty_input():
    time = np.array([0, 900_000, 3_600_000], np.int64)
    labels, means = sources.means(time, np.array([[1.0], [3.0], [np.nan]]), "hour", "UTC")
    assert labels.tolist() == [0, 3_600_000]
    assert means[0, 0] == 2.0 and np.isnan(means[1, 0])
    empty = np.array([], np.int64)
    assert sources.means(empty, np.empty((0, 1)), "day", "UTC")[0].size == 0
    with pytest.raises(ValueError):
        sources.means(time, np.ones((3, 1)), "week", "UTC")
