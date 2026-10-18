"""Times of the rows in an operator's file, from their position.

The operators label their quarter hours in at least five ways (start or end
time, the date of the start or of the end next to either), some labels are
broken (`#BEZUG!` in Berlin's 2025 files), and some files pad the hour that
doesn't exist in March with four rows of 0 (SWM, Netze BW, Pforzheim). So
labels aren't parsed into times. Instead:

1. The file covers a known period in local time, whole days from `start` up
   to `end` (exclusive), from its header, its name or its spec.
2. The n-th row is the n-th quarter hour of that period. The quarter hours
   are even steps in UTC, so the October hour that repeats in local time is
   simply four more rows, and the March hour that's left out four fewer.
3. A file with four rows more per March change is zero-padded: those rows
   (the third hour of that day, counted in rows) are dropped, and must hold
   only 0 or nothing.
4. The labels are then checked: a label matches its row if it is the start
   or the end of the quarter hour, with the date of either. A file passes
   when almost all its readable labels match, in local time or in standard
   time (UTC+1) all year.

Any other row count fails, as do labels that don't match: a wrong guess
about a file stops loudly instead of shifting its data. What the labels
can't tell apart is a file that starts one row late from one labelled by
the end of each quarter hour; the row count catches that.

    align(values, labels, date(2025, 1, 1), date(2026, 1, 1)) -> Alignment
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np

from free_data.sources import epoch_ms

TZ = ZoneInfo("Europe/Berlin")
QUARTER = 900_000  # ms
HOUR = 3_600_000
DAY = 86_400_000
STANDARD = HOUR  # Central European standard time, UTC+1
MATCHED = 0.99  # share of readable labels that must match their row
LABELLED = 0.5  # share of rows that must have a readable label, when labels are given


class TimingError(ValueError):
    """The rows of a file don't fit the period it is said to cover."""


@dataclass(frozen=True)
class Alignment:
    """Rows of a file placed in time."""

    time: np.ndarray  # UTC datetime64[ms], the start of each kept row's quarter hour
    values: np.ndarray  # the kept rows, padding dropped
    rows: np.ndarray  # index of each kept row in the file
    padded: tuple[int, ...]  # rows dropped as padding of the March hour
    labelled: int  # kept rows with a readable label
    matched: int  # of those, the ones that match their row
    clock: str  # what the labels count in: "local", "standard", or "" without labels


def grid(start: date, end: date) -> np.ndarray:
    """UTC ms of every quarter hour from local midnight of `start` to that of `end` (exclusive)."""
    first = epoch_ms(datetime.combine(start, time()), TZ)
    last = epoch_ms(datetime.combine(end, time()), TZ)
    return np.arange(first, last, QUARTER, dtype=np.int64)


def offsets(ms: np.ndarray) -> np.ndarray:
    """Local UTC offset in ms at each UTC ms (on quarter hours; changes happen on the hour)."""
    hours = ms - ms % HOUR
    unique, inverse = np.unique(hours, return_inverse=True)
    off = [datetime.fromtimestamp(h / 1000, TZ).utcoffset() // timedelta(milliseconds=1) for h in unique.tolist()]
    return np.asarray(off, dtype=np.int64)[inverse]


def springs(times: np.ndarray) -> list[int]:
    """Indices in `times` of the first quarter hour after each March change (offset grows)."""
    off = offsets(times)
    return (np.flatnonzero(np.diff(off) > 0) + 1).tolist()


def align(
    values: np.ndarray,
    labels: list[datetime | None] | None,
    start: date,
    end: date,
) -> Alignment:
    """Place the rows of `values` (one per row, any columns) in the period; see the module's text."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    times = grid(start, end)
    gaps = springs(times)
    if n == len(times):
        rows, padded = np.arange(n), ()
    elif n == len(times) + 4 * len(gaps) and gaps:
        # in the padded file each gap's four rows sit where the gap is, after earlier gaps' rows
        padded = tuple(g + 4 * k + j for k, g in enumerate(gaps) for j in range(4))
        rows = np.setdiff1d(np.arange(n), padded)
        pad = values[list(padded)]
        if np.any(pad[np.isfinite(pad)] != 0):
            raise TimingError(f"rows {padded[0]}-{padded[-1]} pad the March hour but hold values other than 0")
    else:
        expect = f"{len(times)}" + (f" (or {len(times) + 4 * len(gaps)} with the March hour padded)" if gaps else "")
        raise TimingError(f"{n} rows, but {start} to {end} has {expect} quarter hours")
    labelled = matched = 0
    clock = ""
    if labels is not None:
        if len(labels) != n:
            raise TimingError(f"{len(labels)} labels for {n} rows")
        kept = [labels[i] for i in rows.tolist()]
        labelled, matched, clock = _check(kept, times)
        if labelled < LABELLED * len(rows):
            raise TimingError(f"only {labelled} of {len(rows)} rows have a readable time label")
        if matched < MATCHED * labelled:
            raise TimingError(_mismatch(kept, times, labelled, matched))
    return Alignment(times.astype("datetime64[ms]"), values[rows], rows, padded, labelled, matched, clock)


def _naive_ms(when: datetime) -> int:
    """A label's wall-clock time as ms, as if it were UTC (labels carry no offset)."""
    return int(when.replace(tzinfo=UTC).timestamp() * 1000)


def _candidates(times: np.ndarray, clock: str) -> tuple[np.ndarray, ...]:
    """Wall-clock ms a label may show for each quarter hour: start, end, and each with the other's date."""
    start = times + (offsets(times) if clock == "local" else STANDARD)
    end = start + QUARTER
    day_start, day_end = start - start % DAY, end - end % DAY
    return start, end, day_end + start % DAY, day_start + end % DAY


def _matches(kept: list[datetime | None], times: np.ndarray, clock: str) -> np.ndarray:
    """Per kept row: 1 if its label matches, 0 if not, -1 without a label."""
    have = np.array([x is not None for x in kept])
    label = np.array([_naive_ms(x) if x is not None else 0 for x in kept], dtype=np.int64)
    ok = np.zeros(len(kept), dtype=bool)
    for c in _candidates(times, clock):
        ok |= label == c
    return np.where(have, ok.astype(int), -1)


def _check(kept: list[datetime | None], times: np.ndarray) -> tuple[int, int, str]:
    labelled = sum(x is not None for x in kept)
    if not labelled:
        return 0, 0, ""
    best = max(("local", "standard"), key=lambda c: int((_matches(kept, times, c) == 1).sum()))
    return labelled, int((_matches(kept, times, best) == 1).sum()), best


def _mismatch(kept: list[datetime | None], times: np.ndarray, labelled: int, matched: int) -> str:
    m = _matches(kept, times, "local")
    first = int(np.flatnonzero(m == 0)[0])
    when = datetime.fromtimestamp(int(times[first]) / 1000, TZ)
    return (
        f"only {matched} of {labelled} time labels match their row; "
        f"first mismatch: row {first} is {when:%Y-%m-%d %H:%M %Z}, labelled {kept[first]}"
    )


# Labels


_DATE = re.compile(r"(\d{1,2})\.(\d{1,2})\.(\d{4})|(\d{4})-(\d{2})-(\d{2})")
_TIME = re.compile(r"(\d{1,2}):(\d{2})(?::\d{2})?")


def label(day: object, clock: object = None) -> datetime | None:
    """A row's time label from its date cell and time cell, or None if they don't read as one.

    Accepts strings ("01.01.2025", "2025-01-01", "00:15", "00:15:00", also both
    in one cell), dates, times and datetimes as spreadsheets hold them. "24:00"
    is midnight of the next day.
    """
    if isinstance(day, datetime):
        when = day
    elif isinstance(day, date):
        when = datetime.combine(day, time())
    elif isinstance(day, str) and (m := _DATE.search(day)):
        d, mo, y = (m[1], m[2], m[3]) if m[1] else (m[6], m[5], m[4])
        try:
            when = datetime(int(y), int(mo), int(d))
        except ValueError:
            return None
        rest = day[m.end() :]
        if clock is None and _TIME.search(rest):
            clock = rest
    else:
        return None
    if clock is None:
        return when
    if isinstance(clock, time):
        return datetime.combine(when.date(), clock)
    if isinstance(clock, datetime):  # spreadsheets hold a bare time as a datetime on 1899-12-30 or 1900-01-01
        return datetime.combine(when.date(), clock.time())
    if isinstance(clock, str) and (m := _TIME.search(clock)):
        h, mi = int(m[1]), int(m[2])
        if h == 24 and mi == 0:
            return datetime.combine(when.date(), time()) + timedelta(days=1)
        if h < 24 and mi < 60:
            return datetime.combine(when.date(), time(h, mi))
    return None
