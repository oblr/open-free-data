"""éCO2mix against small CSV files in ODRÉ's export format.

"Now" is 2024-01-10. The history covers 2023-12-31 and 2024-01-01 (Paris),
the real-time data starts 2024-01-02 00:00 Paris (2024-01-01T23:00Z), so
the history's own rows from then on must be ignored.
"""

from __future__ import annotations

import numpy as np
import pytest

import browser
import cache
import eco2mix
from sources import epoch_ms

HEADER = "perimetre;nature;date;heure;date_heure;consommation;eolien;ech_comm_allemagne_belgique"
QUARTER = 900_000


def ms(text: str) -> int:
    return epoch_ms(text, eco2mix.TZ)


def csv_rows(start: str, count: int, value) -> str:
    """`count` quarter-hour rows from `start` (Paris), values from value(i) -> (conso, wind, exchange)."""
    lines = []
    for i in range(count):
        t = np.datetime64(ms(start) + i * QUARTER, "ms").astype("datetime64[s]")
        lines.append(";".join(("France", "x", "", "", f"{t}+00:00", *value(i))))
    return "\n".join(lines)


def history(i: int) -> tuple[str, str, str]:
    # most fields only on the half hour; the exchange every quarter hour, sometimes as a marker
    half = i % 2 == 0
    return (str(1000 + i) if half else "", "5" if half else "", "ND" if i == 3 else str(-i))


def realtime(i: int) -> tuple[str, str, str]:
    return (str(2000 + i), "7", str(i)) if i < 8 else ("", "", "")  # trailing empty rows


@pytest.fixture
def files(cache_root, monkeypatch):
    monkeypatch.setattr(eco2mix, "_now", lambda: ms("2024-01-10T12:00"))
    source = cache_root / eco2mix.SOURCE
    source.joinpath(eco2mix.HISTORY).mkdir(parents=True)
    # 2023: the last day only; 2024: two days, the second already covered by real time
    (source / eco2mix.HISTORY / "2023.csv").write_text(f"﻿{HEADER}\n{csv_rows('2023-12-31', 96, history)}\n")
    (source / eco2mix.HISTORY / "2024.csv").write_text(f"﻿{HEADER}\n{csv_rows('2024-01-01', 192, history)}\n")
    (source / f"{eco2mix.REALTIME}.csv").write_text(f"﻿{HEADER}\n{csv_rows('2024-01-02', 12, realtime)}\n")
    return source


def test_quarterhour_joins_history_and_real_time(files):
    table = eco2mix.read(["consommation"], "2024-01-01T23:00", "2024-01-02T00:30", "quarterhour")
    times = table.time.astype(np.int64)
    # the history's empty quarter hours are dropped, the real time is every quarter hour
    assert (times - ms("2024-01-01T23:00")).tolist() == [0, 2 * QUARTER, 4 * QUARTER, 5 * QUARTER, 6 * QUARTER]
    assert table.values[:, 0].tolist() == [1092.0, 1094.0, 2000.0, 2001.0, 2002.0]


def test_hourly_means_trim_empty_rows_at_the_end(files):
    table = eco2mix.read(["consommation", "eolien"], "2024-01-02")
    assert table.values.tolist() == [[2001.5, 7.0], [2005.5, 7.0]]  # rows 8-11 are empty
    assert table.time.astype(np.int64).tolist() == [ms("2024-01-02T00:00"), ms("2024-01-02T01:00")]


def test_daily_means_per_paris_day(files):
    table = eco2mix.read(["eolien"], "2023-12-31", "2024-01-01", "day")
    assert table.time.astype(np.int64).tolist() == [ms("2023-12-31"), ms("2024-01-01")]
    assert table.values[:, 0].tolist() == [5.0, 5.0]


def test_text_numbers_and_markers(files):
    table = eco2mix.read(["ech_comm_allemagne_belgique"], "2023-12-31T00:00", "2023-12-31T01:00", "quarterhour")
    assert table.values[:, 0].tolist() == [0.0, -1.0, -2.0, -4.0]  # "ND" at 00:45 is dropped


def test_fields_missing_from_a_file_are_nan(files):
    table = eco2mix.read(["consommation", "solaire"], "2024-01-02T00:00", "2024-01-02T00:15", "quarterhour")
    assert table.values[:, 0].tolist() == [2000.0, 2001.0]
    assert np.isnan(table.values[:, 1]).all()


def test_real_time_range_needs_no_history(files, monkeypatch):
    (files / eco2mix.HISTORY / "2024.csv").unlink()  # a download would fail the test
    assert len(eco2mix.read(["consommation"], "2024-01-02").time) == 2


def test_old_range_needs_no_real_time(files, monkeypatch):
    (files / f"{eco2mix.REALTIME}.csv").unlink()
    monkeypatch.setattr(eco2mix, "_now", lambda: ms("2024-12-01"))
    assert len(eco2mix.read(["consommation"], "2023-12-31", "2023-12-31T23:59").time) == 24


def test_history_url_covers_one_paris_year(monkeypatch):
    urls = []

    def fake(url: str) -> bytes:
        urls.append(url)
        return f"{HEADER}\n".encode()

    monkeypatch.setattr(cache, "_download", fake)
    monkeypatch.setattr(eco2mix, "_now", lambda: ms("2030-01-01"))
    eco2mix.read(["consommation"], "2024-06-01", "2024-06-02")
    assert len(urls) == 1 and "eco2mix-national-cons-def" in urls[0]
    assert "2023-12-31T23%3A00%3A00Z" in urls[0] and "2024-12-31T23%3A00%3A00Z" in urls[0]


@pytest.mark.parametrize(
    ("args", "message"),
    [((["nope"],), "unknown fields"), ((["consommation"], None, None, "week"), "resolution"),
     ((["consommation"], "2024-01-02", "2024-01-01"), "after")],
)
def test_bad_arguments(args, message):
    with pytest.raises(ValueError, match=message):
        eco2mix.read(*args)


def test_side_by_side_with_smard_in_the_browser(files, smard_files):
    columns, table, on_grid = browser.series({
        "d": ["fr-rte-eco2mix:consommation", "de-smard:410"],
        "start": ["2024-01-01T00:00+00:00"], "end": ["2024-01-01T05:00+00:00"],
    })
    assert [c["unit"] for c in columns] == ["MW", "MWh"]
    assert table.columns == ("fr-rte-eco2mix:consommation", "de-smard:410")
    assert on_grid[:, 0].all() and on_grid[1:, 1].all()  # both hourly; SMARD's 00:00Z is padding
    assert table.values[1].tolist() == [1009.0, 1.0]  # 01:00Z: mean of rows 8 and 10 of 2024; SMARD 1.0
