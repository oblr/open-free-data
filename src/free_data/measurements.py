"""The features fingerprints.py stores for every series, registered on import.

    coverage        share of the window's steps that have a value
    summary         min, p5, median, p95, max, mean, std (directions: circular mean and spread)
    zero_share      share of the values that are 0: night for solar, dry hours for rain
    profile.day     24 numbers: the mean per local hour, as z-scores of the whole series
    profile.year    12 numbers: the same per local month
    cycle.strength  how much of the variance the hour of day, the day of week and the month explain
    autocorr        the series against itself 1 h, 24 h and 168 h later
    trend           seasonal Theil-Sen slope of the daily means, in unit per year

Nothing here names a source or a variable; what differs follows the
dataset's measure. A direction (350° is next to 10°) only gets a summary.
A feature that needs more data than the series has is None: profile.day
an hourly step and 7 values per hour, profile.year and cycle.strength.year
300 days in all 12 months, trend 2 years.

Profiles are z-scores, (mean of the hour - mean) / std, so a temperature in
°C, a balance that changes sign and a generation in MW compare alike; how
strong the cycle is against the rest is cycle.strength.
"""

from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import numpy as np

from free_data import sources
from free_data.fingerprints import Series, feature

HOUR_MS = 3_600_000
DAY_MS = 24 * HOUR_MS
MIN_PER_HOUR = 7  # values per hour of day for a daily profile: a week
MIN_DAYS_YEAR = 300  # days with a value for anything about the year
MIN_DAYS_TREND = 730
MIN_PAIRS = 30  # value pairs for an autocorrelation
MOST_PAIRS = 50_000  # pairs of days per month for the trend; more are sampled (fixed seed)
LAGS = {"1h": 1, "24h": 24, "168h": 168}  # in hours
CIRCULAR = ("direction",)  # measures whose values are angles in degrees
ZERO = 1e-9


# ---- helpers ----


def _ms(s: Series) -> np.ndarray:
    return s.time.astype("datetime64[ms]").astype(np.int64)


def _finite(s: Series) -> tuple[np.ndarray, np.ndarray]:
    """The times (epoch ms) and values that are there."""
    ok = np.isfinite(s.values)
    return _ms(s)[ok], s.values[ok]


def _step_ms(s: Series) -> float:
    """The series' time step: the window's when it has one, else the median gap."""
    step = s.window.step if s.window else ""
    if step == "hour":
        return HOUR_MS
    if step == "day":
        return DAY_MS
    if step == "month":
        return 30.4 * DAY_MS
    ms = _ms(s)
    return float(np.median(np.diff(ms))) if len(ms) > 1 else np.nan


def _local(ms: np.ndarray, zone: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Local hour (0-23), weekday (0 = Monday), month (1-12) and day (days since 1970) of each time."""
    tz = ZoneInfo(zone)
    hours = ms - ms % HOUR_MS
    unique, back = np.unique(hours, return_inverse=True)  # the offset changes on whole hours at most
    offsets = np.array([datetime.fromtimestamp(t / 1000, UTC).astimezone(tz).utcoffset().total_seconds() * 1000
                        for t in unique.tolist()], np.int64)
    local = (ms + offsets[back]).astype("datetime64[ms]")
    days = local.astype("datetime64[D]")
    hour = (local - days).astype("timedelta64[h]").astype(int)
    weekday = (days.astype(np.int64) + 3) % 7  # 1970-01-01 was a Thursday
    month = days.astype("datetime64[M]").astype(int) % 12 + 1
    return hour, weekday, month, days.astype(np.int64)


def _group_means(values: np.ndarray, groups: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean and count of `values` per group 0..n-1."""
    counts = np.bincount(groups, minlength=n)
    sums = np.bincount(groups, weights=values, minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        return sums / counts, counts


def _explained(values: np.ndarray, groups: np.ndarray, n: int) -> float:
    """R² of fitting each value by its group's mean: the share of the variance the grouping explains."""
    means, _ = _group_means(values, groups, n)
    total = np.sum((values - values.mean()) ** 2)
    if total <= 0:
        return np.nan
    return float(1 - np.sum((values - means[groups]) ** 2) / total)


def _year_covered(days: np.ndarray, month: np.ndarray) -> bool:
    return len(np.unique(days)) >= MIN_DAYS_YEAR and len(np.unique(month)) == 12


def _circular(s: Series) -> bool:
    return s.dataset.measure in CIRCULAR


# ---- features ----


@feature("coverage", version=1)
def coverage(s: Series) -> float | None:
    """Values there / steps in the window (or between the first and last time, without a window)."""
    ms, _ = _finite(s)
    expected = _expected_steps(s)
    if not expected:
        return None
    return min(1.0, len(ms) / expected)


def _expected_steps(s: Series) -> int:
    if s.window:
        start = sources.moment(s.window.start, s.zone, end=False)
        end = sources.moment(s.window.end, s.zone, end=True)
        if s.window.step in ("day", "month"):
            unit = "D" if s.window.step == "day" else "M"
            first, last = (np.datetime64(t.date(), unit) for t in (start, end))
            return int((last - first).astype(int)) + 1
        return int(np.ceil((end.timestamp() - start.timestamp()) * 1000 / HOUR_MS))
    ms, step = _ms(s), _step_ms(s)
    if not len(ms) or not np.isfinite(step):
        return len(ms)
    return int(round((ms[-1] - ms[0]) / step)) + 1


@feature("summary", version=1)
def summary(s: Series) -> dict | None:
    _, v = _finite(s)
    if not len(v):
        return None
    if _circular(s):
        rad = np.deg2rad(v)
        c, si = np.cos(rad).mean(), np.sin(rad).mean()
        r = min(1.0, float(np.hypot(c, si)))
        spread = np.rad2deg(np.sqrt(-2 * np.log(r))) if r > 0 else np.inf
        return {"mean": float(np.rad2deg(np.arctan2(si, c)) % 360), "std": float(spread), "n": len(v)}
    p5, median, p95 = np.percentile(v, [5, 50, 95])
    return {"min": float(v.min()), "p5": float(p5), "median": float(median), "p95": float(p95),
            "max": float(v.max()), "mean": float(v.mean()), "std": float(v.std()), "n": len(v)}


@feature("zero_share", version=1)
def zero_share(s: Series) -> float | None:
    _, v = _finite(s)
    if not len(v) or _circular(s):
        return None
    return float(np.mean(np.abs(v) < ZERO))


@feature("profile.day", version=1)
def profile_day(s: Series) -> list[float] | None:
    ms, v = _finite(s)
    if _circular(s) or not len(v) or _step_ms(s) > HOUR_MS * 1.01:
        return None
    hour, *_ = _local(ms, s.zone)
    means, counts = _group_means(v, hour, 24)
    if counts.min() < MIN_PER_HOUR or v.std() <= 0:
        return None
    return ((means - v.mean()) / v.std()).tolist()


@feature("profile.year", version=1)
def profile_year(s: Series) -> list[float] | None:
    ms, v = _finite(s)
    if _circular(s) or not len(v) or v.std() <= 0:
        return None
    _, _, month, days = _local(ms, s.zone)
    if not _year_covered(days, month):
        return None
    # each day weighs the same, whatever its number of values
    day_means, _ = _daily(v, days)
    day_month = month[np.unique(days, return_index=True)[1]]
    means, _ = _group_means(day_means, day_month - 1, 12)
    return ((means - day_means.mean()) / day_means.std()).tolist() if day_means.std() > 0 else None


def _daily(v: np.ndarray, days: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The mean per day, in the order of np.unique(days), and each value's day index."""
    unique, index = np.unique(days, return_inverse=True)
    means, _ = _group_means(v, index, len(unique))
    return means, index


@feature("cycle.strength", version=1)
def cycle_strength(s: Series) -> dict | None:
    """R² of the means per local hour of day (day), weekday (week) and month (year), each on its own."""
    ms, v = _finite(s)
    if _circular(s) or len(v) < 2:
        return None
    hour, weekday, month, days = _local(ms, s.zone)
    step = _step_ms(s)
    span_days = len(np.unique(days))
    return {
        "day": _explained(v, hour, 24) if step <= HOUR_MS * 1.01 and span_days >= MIN_PER_HOUR else None,
        "week": _explained(v, weekday, 7) if step <= DAY_MS * 1.01 and span_days >= 14 else None,
        "year": _explained(v, month - 1, 12) if _year_covered(days, month) else None,
    }


@feature("autocorr", version=1)
def autocorr(s: Series) -> dict | None:
    """Pearson r of each value with the one LAGS later, on the series' regular grid; None for a lag
    that isn't a whole number of steps."""
    if _circular(s):
        return None
    ms, step = _ms(s), _step_ms(s)
    if len(ms) < 2 or not np.isfinite(step) or step > DAY_MS * 1.01:
        return None
    index = np.round((ms - ms[0]) / step).astype(np.int64)  # day steps of 23 or 25 hours still round right
    grid = np.full(index[-1] + 1, np.nan)
    grid[index] = s.values
    found = {}
    for name, lag_h in LAGS.items():
        k = lag_h * HOUR_MS / step
        found[name] = _lagged_r(grid, int(round(k))) if k >= 1 and abs(k - round(k)) < 1e-6 else None
    return found


def _lagged_r(grid: np.ndarray, k: int) -> float | None:
    if k >= len(grid):
        return None
    a, b = grid[:-k], grid[k:]
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < MIN_PAIRS:
        return None
    a, b = a[ok], b[ok]
    if a.std() <= 0 or b.std() <= 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


@feature("trend", version=1)
def trend(s: Series) -> float | None:
    """Seasonal Theil-Sen slope (Hirsch's Seasonal Kendall estimator) of the daily means, in unit per year:
    the median slope over pairs of days in the same calendar month of different years, so the seasons
    neither make nor hide a trend."""
    ms, v = _finite(s)
    if _circular(s) or not len(v):
        return None
    _, _, month, days = _local(ms, s.zone)
    unique, first = np.unique(days, return_index=True)
    if unique[-1] - unique[0] + 1 < MIN_DAYS_TREND or len(unique) < MIN_DAYS_YEAR:
        return None
    day_means, _ = _daily(v, days)
    day_month, day_year = month[first], unique.astype("datetime64[D]").astype("datetime64[Y]").astype(int)
    slopes = []
    for mo in range(1, 13):
        here = np.flatnonzero(day_month == mo)
        i, j = np.triu_indices(len(here), 1)
        i, j = here[i], here[j]
        apart = day_year[i] != day_year[j]
        i, j = i[apart], j[apart]
        if len(i) > MOST_PAIRS:
            pick = np.random.default_rng(mo).choice(len(i), MOST_PAIRS, replace=False)
            i, j = i[pick], j[pick]
        slopes.append((day_means[j] - day_means[i]) / (unique[j] - unique[i]))
    slopes = np.concatenate(slopes)
    return float(np.median(slopes)) * 365.25 if len(slopes) else None
