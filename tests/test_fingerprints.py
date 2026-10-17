"""Measurements run once per window and version, and land in each dataset's fingerprint."""

from __future__ import annotations

import numpy as np
import pytest

from free_data import fingerprints
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
