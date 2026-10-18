"""Rows of an operator's file placed in time, under the label conventions found in the survey."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta

import numpy as np
import pytest

from free_data.vnb import timing
from free_data.vnb.timing import TZ, TimingError, align, label

Y25 = (date(2025, 1, 1), date(2026, 1, 1))


def wall(times: np.ndarray, standard: bool = False) -> list[datetime]:
    """Naive wall-clock starts of each quarter hour, in local time or in UTC+1 all year."""
    if standard:
        return [datetime(1970, 1, 1) + timedelta(milliseconds=int(t) + timing.HOUR) for t in times]
    return [datetime.fromtimestamp(int(t) / 1000, TZ).replace(tzinfo=None) for t in times]


def labels(start: date, end: date, kind: str) -> list[datetime]:
    """Labels of each row, as one operator or another writes them."""
    s = wall(timing.grid(start, end), standard=kind == "standard")
    q = timedelta(minutes=15)
    return {
        "start": s,
        "standard": s,
        "end": [x + q for x in s],
        "end-date, start-time": [datetime.combine((x + q).date(), x.time()) for x in s],  # MITNETZ, SWM
        "start-date, end-time": [datetime.combine(x.date(), (x + q).time()) for x in s],  # Berlin
    }[kind]


def test_grid_counts_local_quarter_hours():
    assert len(timing.grid(*Y25)) == 35_040
    assert len(timing.grid(date(2024, 1, 1), date(2025, 1, 1))) == 35_136
    assert len(timing.grid(date(2025, 3, 30), date(2025, 3, 31))) == 92
    assert len(timing.grid(date(2025, 10, 26), date(2025, 10, 27))) == 100
    assert timing.grid(*Y25)[0] == np.datetime64("2024-12-31T23:00", "ms").astype(np.int64)


def test_spring_gap_is_found():
    times = timing.grid(*Y25)
    (gap,) = timing.springs(times)
    assert np.datetime64(int(times[gap]), "ms") == np.datetime64("2025-03-30T01:00")  # 03:00 local
    assert wall(times[gap - 1 : gap + 1]) == [datetime(2025, 3, 30, 1, 45), datetime(2025, 3, 30, 3, 0)]


@pytest.mark.parametrize("kind", ["start", "end", "end-date, start-time", "start-date, end-time", "standard"])
def test_every_label_convention_passes(kind):
    values = np.arange(35_040.0)
    a = align(values, labels(*Y25, kind), *Y25)
    assert a.matched == a.labelled == 35_040
    assert a.clock == ("standard" if kind == "standard" else "local")
    assert a.values.tolist() == values.tolist()
    assert a.time[0] == np.datetime64("2024-12-31T23:00")


def test_october_hour_counts_twice():
    start, end = date(2025, 10, 26), date(2025, 10, 27)
    a = align(np.ones(100), labels(start, end, "start"), start, end)
    t = a.time.astype(np.int64)
    assert np.all(np.diff(t) == timing.QUARTER)  # even steps in UTC, through the repeated hour
    assert wall(t[8:16]) == [datetime(2025, 10, 26, 2, m) for m in (0, 15, 30, 45)] * 2


def test_zero_padded_march_hour_is_dropped():
    start, end = date(2025, 3, 30), date(2025, 3, 31)
    s = wall(timing.grid(start, end))
    padded = s[:8] + [datetime(2025, 3, 30, 2, m) for m in (0, 15, 30, 45)] + s[8:]
    values = np.r_[np.arange(8.0), [0, 0, 0, np.nan], np.arange(8.0, 92.0)]
    a = align(values[:, None], padded, start, end)
    assert a.padded == (8, 9, 10, 11)
    assert a.values[:, 0].tolist() == list(range(92))
    assert a.matched == 92


def test_padding_with_values_fails():
    start, end = date(2025, 3, 30), date(2025, 3, 31)
    values = np.r_[np.arange(8.0), [0, 5, 0, 0], np.arange(8.0, 92.0)]
    with pytest.raises(TimingError, match="pad the March hour"):
        align(values, None, start, end)


def test_padding_over_the_whole_year():
    values = np.ones(35_044)
    values[88 * 96 + 8 : 88 * 96 + 12] = 0  # 30 March is day 89; its third hour
    a = align(values, None, *Y25)
    assert a.padded == tuple(range(88 * 96 + 8, 88 * 96 + 12))
    assert len(a.time) == 35_040 and a.clock == ""


@pytest.mark.parametrize("n", [35_039, 35_041, 35_036])
def test_wrong_row_count_fails(n):
    with pytest.raises(TimingError, match=f"{n} rows"):
        align(np.ones(n), None, *Y25)


def test_broken_labels_are_skipped():
    lab = labels(*Y25, "start-date, end-time")
    for i in range(120 * 96, 121 * 96):  # Berlin, 1 May 2025: "#BEZUG!"
        lab[i] = label("#BEZUG!", "#BEZUG!")
    a = align(np.ones(35_040), lab, *Y25)
    assert a.labelled == 35_040 - 96 and a.matched == a.labelled


def test_labels_an_hour_off_in_summer_fail():
    lab = labels(*Y25, "start")
    lab = [x + timedelta(hours=1) if 4 <= x.month <= 9 else x for x in lab]
    with pytest.raises(TimingError, match="first mismatch: row .* is 2025-04-01 00:00 CEST"):
        align(np.ones(35_040), lab, *Y25)


def test_too_few_labels_fail():
    lab: list = [None] * 35_040
    lab[0] = datetime(2025, 1, 1)
    with pytest.raises(TimingError, match="only 1 of 35040 rows"):
        align(np.ones(35_040), lab, *Y25)


@pytest.mark.parametrize(
    ("day", "clock", "expected"),
    [
        ("01.01.2025", "00:15", datetime(2025, 1, 1, 0, 15)),
        ("1.1.2025", "0:15:00", datetime(2025, 1, 1, 0, 15)),
        ("2025-01-01", "23:45", datetime(2025, 1, 1, 23, 45)),
        ("31.12.2025", "24:00", datetime(2026, 1, 1)),
        ("01.01.2025 00:00 bis 00:15", None, datetime(2025, 1, 1)),  # Pforzheim
        ("31.12.2025 23:45:00", None, datetime(2025, 12, 31, 23, 45)),
        (datetime(2025, 1, 1, 0, 15), None, datetime(2025, 1, 1, 0, 15)),
        (datetime(2025, 3, 2), time(13, 0), datetime(2025, 3, 2, 13)),
        (datetime(2025, 3, 2), datetime(1899, 12, 30, 13, 0), datetime(2025, 3, 2, 13)),
        (date(2025, 3, 2), "13:00", datetime(2025, 3, 2, 13)),
        ("01.01.2025", None, datetime(2025, 1, 1)),
        ("#BEZUG!", "#BEZUG!", None),
        ("31.02.2025", "00:00", None),
        ("01.01.2025", "25:00", None),
        (None, None, None),
        (45658.0, None, None),
    ],
)
def test_label(day, clock, expected):
    assert label(day, clock) == expected
