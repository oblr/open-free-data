"""The features fingerprints.py stores for every series, registered on import.

    coverage        share of the window's steps that have a value
    summary         min, p5, median, p95, max, mean, std (directions: circular mean and spread)
    zero_share      share of the values that are 0: night for solar, dry hours for rain
    profile.day     24 numbers: the mean per local hour, as z-scores of the whole series
    profile.year    12 numbers: the same per local month
    cycle.strength  how much of the variance the hour of day, the day of week and the month explain
    autocorr        the series against itself 1 h, 24 h and 168 h later
    trend           seasonal Theil-Sen slope of the daily means, in unit per year

What the calendar doesn't explain, the remainder (hourly series only):

    remainder.share       share of the variance left after the calendar model
    remainder.memory      its autocorrelation at 1 h to a week, and its decorrelation time
    remainder.bands       how its variance splits over timescales: < 6 h, 6 h-2 d, 2-10 d, > 10 d
    remainder.shape       skewness, excess kurtosis, the share beyond 3 sigma and how much of it is up
    remainder.ramps       the 99th percentile of the hour-to-hour change, and rises against falls
    remainder.episodes    how long it stays above or below the calendar, and how often |z| > 2
    remainder.clustering  whether large surprises come in spells: |z| against |z| a day later
    remainder.spread.day  its spread per local hour, relative to its overall spread

and one comparison of two series:

    remainder.lag         how the two remainders move together, and with what lag

The calendar model is the mean per local month x day type (workday,
Saturday, Sunday) x hour; a cell with under MIN_CELL values falls back to
month x hour, then to hour. Public holidays count as the weekday they fall
on, so they end up in the remainder. The share is adjusted for the number of
cells, as a year has only about 10 values per cell. All but the share and
the spread use the standardized remainder, z: each value divided by the
typical size of the remainder at its local month and hour (the median of
|remainder|, so a few spikes don't hide themselves), so a surprise at noon
in June and one at night in January weigh the same. Where that typical size
is under NO_SPREAD of the whole remainder's (solar at night), there is no z.

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

from dataclasses import dataclass
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import numpy as np

from free_data import sources
from free_data.fingerprints import Series, comparison, feature

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


# ---- the remainder: what the calendar doesn't explain ----

MIN_DAYS_REMAINDER = 28
MIN_CELL = 4  # values for a cell's mean; with fewer, a value falls back to a coarser cell
MEMORY_LAGS = {"1h": 1, "6h": 6, "24h": 24, "72h": 72, "168h": 168}
MAX_DECORRELATION_H = 14 * 24
BANDS = {"<6h": 6, "6h-2d": 48, "2d-10d": 240, ">10d": None}  # name -> the moving mean (h) that ends it
NO_SPREAD = 0.01  # a typical |remainder| under this share of the whole one's is none (solar at night)
MAD_TO_SIGMA = 1.4826  # the median |x| of a normal distribution times this is its sigma


@dataclass(frozen=True)
class Remainder:
    """The remainder on the hourly grid from the series' first value; NaN for gaps."""

    raw: np.ndarray  # value - its cell's mean, in the series' unit
    z: np.ndarray  # raw / its typical size at that month and hour; also NaN where that is none
    hour: np.ndarray  # local hour of each step, -1 for gaps
    share: float | None
    start_h: int  # hours since 1970 of the grid's first step


REMAINDERS: dict[tuple, Remainder | None] = {}  # the features of a series, and its comparisons, share its model
MOST_REMAINDERS = 256


def _remainder(s: Series) -> Remainder | None:
    """The series' remainder model, fitted once for the same values at the same times."""
    key = (s.key, s.zone, s.window, len(s.time), hash(s.time.tobytes()), hash(s.values.tobytes()))
    if key not in REMAINDERS:
        if len(REMAINDERS) >= MOST_REMAINDERS:
            del REMAINDERS[next(iter(REMAINDERS))]  # the oldest
        REMAINDERS[key] = _fit_remainder(s)
    return REMAINDERS[key]


def _fit_remainder(s: Series) -> Remainder | None:
    if _circular(s) or _step_ms(s) > HOUR_MS * 1.01:
        return None
    ms, v = _finite(s)
    if len(v) < 2:
        return None
    hour, weekday, month, days = _local(ms, s.zone)
    if len(np.unique(days)) < MIN_DAYS_REMAINDER:
        return None
    daytype = np.clip(weekday - 4, 0, 2)  # 0 workday, 1 Saturday, 2 Sunday
    levels = [((month - 1) * 3 + daytype) * 24 + hour, (month - 1) * 24 + hour, hour]
    expected = np.full(len(v), np.nan)
    cell = np.full(len(v), -1)
    offset = 0
    for groups, n in zip(levels, (12 * 3 * 24, 12 * 24, 24), strict=True):
        means, counts = _group_means(v, groups, n)
        use = np.isnan(expected) & (counts[groups] >= MIN_CELL)
        expected[use] = means[groups[use]]
        cell[use] = groups[use] + offset
        offset += n
    rest = np.isnan(expected)  # an hour with under MIN_CELL values in all: the overall mean
    expected[rest], cell[rest] = v.mean(), offset
    raw = v - expected

    total = np.sum((v - v.mean()) ** 2)
    cells = len(np.unique(cell))
    share = None
    if total > 0 and len(v) > cells:
        share = float(np.clip(np.sum(raw**2) / (len(v) - cells) / (total / (len(v) - 1)), 0, 1))
    spread = _typical_size(np.abs(raw), month, hour) * MAD_TO_SIGMA
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(spread > NO_SPREAD * raw.std(), raw / spread, np.nan)

    index = np.round((ms - ms[0]) / HOUR_MS).astype(np.int64)
    grid = np.full((2, index[-1] + 1), np.nan)
    grid[0, index], grid[1, index] = raw, z
    hours = np.full(index[-1] + 1, -1)
    hours[index] = hour
    return Remainder(grid[0], grid[1], hours, share, int(round(ms[0] / HOUR_MS)))


def _typical_size(size: np.ndarray, month: np.ndarray, hour: np.ndarray) -> np.ndarray:
    """The median of `size` at each value's local month and hour, or at its hour where the month has too few."""
    by_month = _group_medians(size, (month - 1) * 24 + hour, 12 * 24)
    by_hour = _group_medians(size, hour, 24)
    _, counts = _group_means(size, (month - 1) * 24 + hour, 12 * 24)
    fine = (month - 1) * 24 + hour
    return np.where(counts[fine] >= 2 * MIN_CELL, by_month[fine], by_hour[hour])


def _group_medians(values: np.ndarray, groups: np.ndarray, n: int) -> np.ndarray:
    """Median of `values` per group 0..n-1; NaN for an empty group."""
    order = np.lexsort((values, groups))
    v, g = values[order], groups[order]
    counts = np.bincount(g, minlength=n)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    medians = np.full(n, np.nan)
    has = counts > 0
    lo = starts[has] + (counts[has] - 1) // 2
    hi = starts[has] + counts[has] // 2
    medians[has] = (v[lo] + v[hi]) / 2
    return medians


@feature("remainder.share", version=1)
def remainder_share(s: Series) -> float | None:
    r = _remainder(s)
    return r.share if r else None


@feature("remainder.memory", version=1)
def remainder_memory(s: Series) -> dict | None:
    """r of z with itself at MEMORY_LAGS, and decorrelation_h: the lag (interpolated) where it falls below 1/e;
    None if it doesn't within two weeks."""
    r = _remainder(s)
    if not r:
        return None
    found: dict = {name: _lagged_r(r.z, k) for name, k in MEMORY_LAGS.items()}
    found["decorrelation_h"] = None
    previous = 1.0
    for k in range(1, MAX_DECORRELATION_H + 1):
        rk = _lagged_r(r.z, k)
        if rk is None:
            break
        if rk < 1 / np.e:
            found["decorrelation_h"] = k - 1 + (previous - 1 / np.e) / (previous - rk)
            break
        previous = rk
    return found


def _moving_mean(x: np.ndarray, width: int) -> np.ndarray:
    """Centred mean over `width` steps; NaN where under half of them have a value."""
    ok = np.isfinite(x)
    counts = np.concatenate([[0], np.cumsum(ok)])
    sums = np.concatenate([[0], np.cumsum(np.where(ok, x, 0))])
    lo = np.clip(np.arange(len(x)) - width // 2, 0, len(x))
    hi = np.clip(lo + width, 0, len(x))
    n = counts[hi] - counts[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (sums[hi] - sums[lo]) / n
    mean[n < width / 2] = np.nan
    return mean


@feature("remainder.bands", version=1)
def remainder_bands(s: Series) -> dict | None:
    """Shares of z's variance by timescale: the differences between moving means of 6 h, 2 d and 10 d."""
    r = _remainder(s)
    if not r:
        return None
    levels = [r.z] + [_moving_mean(r.z, w) for w in BANDS.values() if w]
    parts = [a - b for a, b in zip(levels, levels[1:], strict=False)] + [levels[-1]]
    power = np.array([np.nanmean(p**2) if np.isfinite(p).any() else np.nan for p in parts])
    if not np.isfinite(power).all() or power.sum() <= 0:
        return None
    return dict(zip(BANDS, (power / power.sum()).tolist(), strict=True))


@feature("remainder.shape", version=1)
def remainder_shape(s: Series) -> dict | None:
    """Skewness, excess kurtosis, tail (share beyond 3 sigma) and tail_up (the share of the tail that is up)."""
    r = _remainder(s)
    z = r.z[np.isfinite(r.z)] if r else np.array([])
    if len(z) < MIN_PAIRS or z.std() <= 0:
        return None
    u = (z - z.mean()) / z.std()
    tail = float(np.mean(np.abs(u) > 3))
    return {"skewness": float(np.mean(u**3)), "kurtosis": float(np.mean(u**4) - 3), "tail": tail,
            "tail_up": float(np.mean(u > 3)) / tail if tail else None}


@feature("remainder.ramps", version=1)
def remainder_ramps(s: Series) -> dict | None:
    """p99: the 99th percentile of |z(t+1h) - z(t)|; rise_fall: that of the rises over that of the falls."""
    r = _remainder(s)
    if not r:
        return None
    d = np.diff(r.z)
    d = d[np.isfinite(d)]
    up, down = d[d > 0], -d[d < 0]
    if len(up) < MIN_PAIRS or len(down) < MIN_PAIRS:
        return None
    return {"p99": float(np.percentile(np.abs(d), 99)),
            "rise_fall": float(np.percentile(up, 99) / np.percentile(down, 99))}


@feature("remainder.episodes", version=1)
def remainder_episodes(s: Series) -> dict | None:
    """How long z stays on one side of 0 (mean_h, longest_above_h, longest_below_h; a gap ends a run, and the
    runs at the ends are cut by the window) and extreme: the share of hours with |z| > 2."""
    r = _remainder(s)
    if not r or not np.isfinite(r.z).any():
        return None
    sign = np.where(np.isfinite(r.z), np.sign(r.z), 0)
    starts = np.concatenate([[0], np.flatnonzero(np.diff(sign)) + 1])
    lengths = np.diff(np.concatenate([starts, [len(sign)]]))
    above, below = lengths[sign[starts] > 0], lengths[sign[starts] < 0]
    if not len(above) or not len(below):
        return None
    z = r.z[np.isfinite(r.z)]
    return {"mean_h": float(np.concatenate([above, below]).mean()), "longest_above_h": int(above.max()),
            "longest_below_h": int(below.max()), "extreme": float(np.mean(np.abs(z) > 2))}


@feature("remainder.clustering", version=1)
def remainder_clustering(s: Series) -> float | None:
    """r of |z| with |z| a day later: high when calm and turbulent spells last."""
    r = _remainder(s)
    return _lagged_r(np.abs(r.z), 24) if r else None


@feature("remainder.spread.day", version=1)
def remainder_spread_day(s: Series) -> list[float] | None:
    """24 numbers: the remainder's root mean square per local hour over its overall one (1 = average)."""
    r = _remainder(s)
    if not r:
        return None
    ok = r.hour >= 0
    mean_square, counts = _group_means(r.raw[ok] ** 2, r.hour[ok], 24)
    overall = np.mean(r.raw[ok] ** 2)
    if counts.min() < MIN_PER_HOUR or overall <= 0:
        return None
    return np.sqrt(mean_square / overall).tolist()


# ---- comparing remainders ----

MAX_LAG_H = 72
MIN_OVERLAP_H = 14 * 24  # hours both have, at a lag, for its r


@comparison("remainder.lag", version=2, symmetric=False)
def remainder_lag(a: Series, b: Series) -> dict | None:
    """How the standardized remainders of a and b move together, within MAX_LAG_H hours.

        r0, r, lag_h        Pearson r at no lag, and at the lag where |r| is largest;
                            lag_h > 0 means b follows a by that many hours
        r_change, lag_change_h   the same for the hour-to-hour changes: slow swings they
                            share weigh less, so the timing is sharper for persistent series
        n                   hours both have, at no lag
        n_eff               about how many of them are independent: n (1 - pa pb) / (1 + pa pb), with pa
                            and pb each remainder's r an hour later (Bretherton et al. 1999). An |r| under
                            about 2 / sqrt(n_eff) could be chance, more so as the best of many lags

    Asymmetric: the pair (b, a) has the lags the other way round, so storing it under a
    is enough. A lag of +-MAX_LAG_H means the best is probably further out.
    """
    ra, rb = _remainder(a), _remainder(b)
    if not ra or not rb:
        return None
    start = min(ra.start_h, rb.start_h)
    length = max(ra.start_h + len(ra.z), rb.start_h + len(rb.z)) - start
    x, y = np.full(length, np.nan), np.full(length, np.nan)
    x[ra.start_h - start: ra.start_h - start + len(ra.z)] = ra.z
    y[rb.start_h - start: rb.start_h - start + len(rb.z)] = rb.z
    level, n = _lagged_correlations(x, y)
    change, _ = _lagged_correlations(np.diff(x), np.diff(y))
    if not np.isfinite(level).any():
        return None
    pa, pb = _lagged_r(x, 1) or 0.0, _lagged_r(y, 1) or 0.0
    found = {"r0": _finite_or_none(level[MAX_LAG_H]), "n": int(n),
             "n_eff": float(n * (1 - pa * pb) / (1 + pa * pb)) if pa * pb < 1 else 0.0}
    for name, rs in (("", level), ("_change", change)):
        if np.isfinite(rs).any():
            best = int(np.nanargmax(np.abs(rs) - 1e-9 * np.abs(np.arange(len(rs)) - MAX_LAG_H)))  # ties: smaller lag
            found[f"r{name}"], found[f"lag{name}_h"] = float(rs[best]), best - MAX_LAG_H
        else:
            found[f"r{name}"] = found[f"lag{name}_h"] = None
    return found


def _finite_or_none(v: float) -> float | None:
    return float(v) if np.isfinite(v) else None


def _lagged_correlations(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, int]:
    """Pearson r of x(t) with y(t + k) for k = -MAX_LAG_H..MAX_LAG_H over the pairs both have (NaN where they
    have under MIN_OVERLAP_H), and the number of pairs at k = 0. Exact, through six cross-correlations by FFT."""
    mx, my = np.isfinite(x), np.isfinite(y)
    x0, y0 = np.where(mx, x, 0.0), np.where(my, y, 0.0)
    size = 1 << int(np.ceil(np.log2(2 * len(x))))

    def cross(u: np.ndarray, v: np.ndarray) -> np.ndarray:  # sum over t of u(t) v(t + k)
        c = np.fft.irfft(np.conj(np.fft.rfft(u, size)) * np.fft.rfft(v, size), size)
        return np.concatenate([c[size - MAX_LAG_H:], c[:MAX_LAG_H + 1]])

    m1, m2 = mx.astype(float), my.astype(float)
    n = np.round(cross(m1, m2))
    sx, sy = cross(x0, m2), cross(m1, y0)
    sxx, syy, sxy = cross(x0**2, m2), cross(m1, y0**2), cross(x0, y0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (n * sxy - sx * sy) / np.sqrt((n * sxx - sx**2) * (n * syy - sy**2))
    r[(n < MIN_OVERLAP_H) | ~np.isfinite(r)] = np.nan
    return np.clip(r, -1, 1), int(n[MAX_LAG_H])
