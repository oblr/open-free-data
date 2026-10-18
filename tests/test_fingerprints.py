"""Measurements run once per window and version, and land in each dataset's fingerprint."""

from __future__ import annotations

import numpy as np
import pytest

from free_data import cache as cache_module
from free_data import fingerprints
from free_data import sources as sources_module
from free_data.fingerprints import Measurement, Series, Store, Window
from free_data.sources import Dataset

WINDOW = Window("2024-01-01", "2024-01-31", "day")


def series(key: str, days: list[int], values: list[float]) -> Series:
    time = np.array(days, "datetime64[D]").astype("datetime64[ms]")
    return Series(key, Dataset(key.split(":")[1], key), time, np.array(values), "MW", "UTC")


DATA = {
    "s:a": series("s:a", [0, 1, 2, 3], [1.0, 2.0, 3.0, 4.0]),
    "s:b": series("s:b", [1, 2, 3, 4], [10.0, 20.0, 30.0, 40.0]),
    "s:c": series("s:c", [0, 1, 2], [np.nan, 5.0, 5.0]),
}


class Reader:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, keys: list[str], window: Window) -> dict[str, Series]:
        self.calls.append(keys)
        return {k: DATA[k] for k in keys}


MEAN = Measurement("mean", 1, lambda s: np.nanmean(s.values))
SHARED = Measurement("shared", 1, lambda a, b: len(a.time))  # both are aligned: the times they share
FIRST_RATIO = Measurement("first_ratio", 1, lambda a, b: b.values[0] / a.values[0], symmetric=False)


def run(reader, pairs=(), features=(MEAN,), comparisons=(SHARED,), keys=("s:a",), window=WINDOW):
    return fingerprints.precompute(keys, window, pairs, features=features, comparisons=comparisons,
                                   store=Store(), reader=reader)


def test_features_and_comparisons_are_stored_per_dataset():
    prints = run(Reader(), [("s:a", "s:b"), ("s:a", "s:c")], comparisons=(SHARED, FIRST_RATIO))
    assert prints["s:a"].feature("mean") == 2.5 and prints["s:b"].feature("mean") == 25.0
    assert prints["s:a"].compared("shared") == {"s:b": 3, "s:c": 3}
    assert prints["s:b"].compared("shared") == {"s:a": 3}  # symmetric: stored on both sides
    assert prints["s:a"].compared("first_ratio") == {"s:b": 5.0, "s:c": None}  # NaN is stored as None
    assert prints["s:b"].compared("first_ratio") == {}  # asymmetric: only the way it was asked
    loaded = Store().load("s:a")
    assert loaded == prints["s:a"]


def test_nothing_is_read_or_computed_twice():
    reader = Reader()
    run(reader, [("s:a", "s:b")])
    run(reader, [("s:a", "s:b"), ("s:b", "s:a")])
    assert reader.calls == [["s:a", "s:b"]]
    run(reader, [("s:a", "s:c")])  # only the new pair is read
    assert reader.calls[-1] == ["s:a", "s:c"]


def test_a_new_version_or_window_recomputes():
    reader = Reader()
    run(reader)
    run(reader, features=(Measurement("mean", 2, lambda s: 0.0),))
    assert Store().load("s:a").features["mean"] == {"version": 2, "value": 0.0}
    prints = run(reader, window=Window("2024-02-01", "2024-02-29", "day"))
    assert len(reader.calls) == 3 and prints["s:a"].window.start == "2024-02-01"


def test_registered_measurements_are_the_default(monkeypatch):
    monkeypatch.setattr(fingerprints, "FEATURES", {})
    monkeypatch.setattr(fingerprints, "COMPARISONS", {})

    @fingerprints.feature("last")
    def last(s: Series) -> float:
        return float(s.values[-1])

    prints = fingerprints.precompute(["s:b"], WINDOW, store=Store(), reader=Reader())
    assert prints["s:b"].feature("last") == 40.0
    with pytest.raises(ValueError, match="two measurements named 'last'"):
        fingerprints.feature("last")(lambda s: 0.0)


def test_store_paths_stay_in_their_folder():
    assert Store().path("de-dwd-opendata:03987/TT_TU").name == "03987__TT_TU.json"
    with pytest.raises(ValueError):
        Store().path("../../etc:passwd")


def test_reads_through_the_plugins(smard_files):
    window = Window("2024-01-01T00:00", "2024-01-01T06:00", "hour")
    found = fingerprints.read(["de-smard:410"], window)
    assert found["de-smard:410"].values[:2].tolist() == [1.0, 2.0] and found["de-smard:410"].zone == "Europe/Berlin"


def test_unreadable_datasets_are_skipped_and_tried_again_later():
    reader = Reader()
    readable = lambda keys, window: {k: DATA[k] for k in keys if k != "s:b"}  # noqa: E731
    prints = run(readable, keys=("s:a", "s:b"))
    assert prints["s:a"].feature("mean") == 2.5 and prints["s:b"].features == {}
    assert Store().load("s:b") is None  # nothing stored for it
    run(reader, keys=("s:a", "s:b"))
    assert reader.calls == [["s:b"]]  # only the one that was missing


def test_read_reads_chunks_one_by_one_when_one_dataset_fails(monkeypatch):
    class Plugin(sources_module.Plugin):
        id, title, timezone = "xx", "X", "UTC"

        def datasets(self):
            return [Dataset(i, i, unit="MW", measure="power") for i in ("a", "b")]

        def read(self, datasets, start, end, options):
            if "b" in datasets:
                raise cache_module.NotCached("u", "b.json")
            time = np.array(["2025-01-01T00", "2025-01-01T01"], "datetime64[ms]")
            return sources_module.Table(tuple(datasets), time, np.ones((2, len(datasets))))

    monkeypatch.setattr(sources_module, "discover", lambda: {"xx": Plugin()})
    found = fingerprints.read(["xx:a", "xx:b"], Window("2025-01-01", "2025-01-01", "hour"))
    assert list(found) == ["xx:a"]


def test_store_lists_every_fingerprint_and_skips_broken_files(tmp_path):
    store = Store(tmp_path)
    store.save(fingerprints.Fingerprint("s:b", WINDOW))
    store.save(fingerprints.Fingerprint("t:a/1", WINDOW))
    (store.dir / "s" / "broken.json").write_text("{")
    assert [f.key for f in store.all()] == ["s:b", "t:a/1"]


def test_compare_on_the_command_line_compares_every_stored_pair(monkeypatch, capsys):
    calls = []

    def fake(keys, window, pairs=(), **kwargs):
        calls.append(list(pairs))
        return {k: fingerprints.Fingerprint(k, window) for pair in pairs for k in pair}

    window = Window("2025-01-01", "2025-12-31")
    for key in ("s:a", "s:b", "t:c"):
        Store().save(fingerprints.Fingerprint(key, window, {"coverage": {"version": 1, "value": 1.0}}))
    Store().save(fingerprints.Fingerprint("s:old", Window("2024-01-01", "2024-12-31"), {"coverage": {}}))
    monkeypatch.setattr(fingerprints, "precompute", fake)
    fingerprints.main(["--compare", "--start", "2025-01-01", "--end", "2025-12-31"])
    assert calls == [[("s:a", "s:b"), ("s:a", "t:c"), ("s:b", "t:c")]]  # each once, other windows left out
    assert "comparing 3 pairs" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        fingerprints.main([])
