"""Shared fixtures: a temporary cache and no network.

Every test runs against its own cache folder ($FREE_DATA_CACHE) and fails if
anything tries to download; tests that want a download patch
cache._download themselves.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import cache

HOUR_MS = 3_600_000
T0 = 1_704_067_200_000  # 2024-01-01T00:00Z


# Two PEGELONLINE stations, as its station list describes them.
STATIONS = [
    {"uuid": "u-koeln", "number": "2730010", "shortname": "KÖLN", "km": 688.0, "agency": "STANDORT KÖLN",
     "water": {"shortname": "RHEIN", "longname": "RHEIN"},
     "timeseries": [
         {"shortname": "W", "longname": "WASSERSTAND ROHDATEN", "unit": "cm", "equidistance": 15,
          "gaugeZero": {"unit": "m. ü. NHN", "value": 35.038, "validFrom": "2019-11-01"}},
         {"shortname": "Q", "longname": "ABFLUSS_ROHDATEN", "unit": "m³/s", "equidistance": 15},
     ]},
    {"uuid": "u-bonn", "number": "2710080", "shortname": "BONN", "km": 654.8, "agency": "STANDORT BONN",
     "water": {"shortname": "RHEIN", "longname": "RHEIN"},
     "timeseries": [{"shortname": "XYZ", "longname": "SICHTWEITE", "unit": "m", "equidistance": 60}]},
]


@pytest.fixture(autouse=True)
def cache_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "cache"
    monkeypatch.setenv("FREE_DATA_CACHE", str(root))

    def offline(url: str) -> bytes:
        raise AssertionError(f"unexpected download: {url}")

    monkeypatch.setattr(cache, "_download", offline)
    return root


def put(root: Path, source: str, key: str, body: object) -> Path:
    """A cached file, fresh as of now."""
    path = root / source / key
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body))
    return path


@pytest.fixture
def smard_files(cache_root: Path) -> dict[str, list[float | None]]:
    """Two hourly chunks of filter 410 in region DE, overlapping at T0+3h.

    chunk A (T0):    T0 pad, 1, 2, 3
    chunk B (T0+3h): 30 (overrides A), gap, 5, trailing pad
    """
    a, b = T0, T0 + 3 * HOUR_MS
    chunks = {
        a: [[a + i * HOUR_MS, v] for i, v in enumerate([None, 1.0, 2.0, 3.0])],
        b: [[b + i * HOUR_MS, v] for i, v in enumerate([30.0, None, 5.0, None])],
    }
    folder = "410/DE/hour"
    put(cache_root, "de-smard", f"{folder}/index_hour.json", {"timestamps": [b, a]})
    for chunk, series in chunks.items():
        put(cache_root, "de-smard", f"{folder}/410_DE_hour_{chunk}.json", {"meta_data": {}, "series": series})
    return chunks


@pytest.fixture
def pegelonline_stations(cache_root: Path) -> list[dict]:
    put(cache_root, "de-pegelonline", "stations.json", STATIONS)
    return STATIONS
