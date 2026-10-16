from __future__ import annotations

import io

import numpy as np
import pytest

import cache
import smard
from cache import FetchError
from conftest import HOUR_MS, T0


def hours(*offsets: int) -> np.ndarray:
    return np.array([T0 + o * HOUR_MS for o in offsets], dtype="datetime64[ms]")


def test_read_joins_chunks_and_trims_padding(smard_files):
    table = smard.read(410)
    assert table.columns == ("consumption_total",)
    np.testing.assert_array_equal(table.time, hours(1, 2, 3, 4, 5))
    # the later chunk wins at T0+3h, the gap inside stays NaN
    np.testing.assert_array_equal(table.values[:, 0], [1.0, 2.0, 30.0, np.nan, 5.0])


def test_read_skips_chunks_outside_the_range(smard_files):
    table = smard.read(410, end=hours(2)[0])
    np.testing.assert_array_equal(table.values[:, 0], [1.0, 2.0])


def test_read_trims_leading_nan_inside_the_range(smard_files):
    table = smard.read(410, start=hours(4)[0])
    np.testing.assert_array_equal(table.time, hours(5))


def test_naive_times_are_berlin_time(smard_files):
    table = smard.read(410, start="2024-01-01T02:00", end="2024-01-01T03:00")  # CET = UTC+1
    np.testing.assert_array_equal(table.time, hours(1, 2))


def test_unknown_series_is_a_smard_error(monkeypatch):
    def missing(url: str) -> bytes:
        raise FetchError(url, 404, "HTTP 404")

    monkeypatch.setattr(cache, "_download", missing)
    with pytest.raises(smard.SmardError, match="not found"):
        smard.read(999)


def test_bad_resolution():
    with pytest.raises(ValueError, match="resolution"):
        smard.read(410, resolution="minute")


def test_read_frame_and_csv(smard_files):
    table = smard.read_frame([410, 410], end=hours(2)[0])
    assert table.values.shape == (2, 2)
    out = io.StringIO()
    smard.write_csv(table, out)
    assert out.getvalue().splitlines() == [
        "timestamp,consumption_total,consumption_total",
        "2024-01-01T02:00:00+01:00,1.0,1.0",
        "2024-01-01T03:00:00+01:00,2.0,2.0",
    ]
