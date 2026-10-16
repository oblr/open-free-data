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


def test_every_plugin_has_unique_dataset_ids():
    for plugin in sources.discover().values():
        ids = [d.id for d in plugin.datasets()]
        assert len(ids) == len(set(ids)), plugin.id
        assert all(o.default in o.choices for o in plugin.options), plugin.id
