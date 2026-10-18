"""Which series look alike, from the profiles in their stored fingerprints.

Two series are compared by their daily and their yearly profile
(measurements.py: the mean per local hour and per month, as z-scores):

    r0      Pearson r of the two profiles as they are
    r, lag  the r of largest size with the second profile shifted by up to
            MAX_LAG steps, and that shift: lag > 0 means the second series
            follows the first by `lag` hours (or months)

A negative r is a relation too (temperature and heating). The shift stays
under a quarter of the period, so "the opposite" and "half a period later"
don't get mixed up. A period is only compared when both series have a cycle
in it (cycle.strength at least MIN_STRENGTH); a flat profile is noise.

    score   the mean |r| over the periods the first series has a cycle in;
            one the other has no cycle in counts as 0

This is only a hint: it sees the shape of the average day and year, not
whether the two move together on the same days. Nothing is read from the
data, so it answers at once for every stored fingerprint.

moves_with() is the answer from the data: the remainder.lag comparisons
(measurements.py) that free-data-fingerprints --compare stores, how the two
series' surprises move together and with what lag. A pair is stored under
one of the two, so it is read from either side, its lags turned round.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from free_data.fingerprints import Fingerprint

PERIODS = {"day": ("profile.day", 6), "year": ("profile.year", 3)}  # name -> (feature, MAX_LAG in its steps)
MIN_STRENGTH = 0.05  # cycle.strength: share of the variance the period explains


@dataclass(frozen=True)
class Match:
    r0: float
    r: float
    lag: int


@dataclass(frozen=True)
class Similar:
    key: str
    score: float
    day: Match | None
    year: Match | None


def compare(a: list[float], b: list[float], max_lag: int) -> Match | None:
    """The profiles a and b as they are, and with b shifted by up to max_lag steps."""
    x, y = np.asarray(a, float), np.asarray(b, float)
    if len(x) != len(y) or len(x) < 3 or not (np.isfinite(x).all() and np.isfinite(y).all()):
        return None
    if x.std() <= 0 or y.std() <= 0:
        return None
    found = {}
    for lag in range(-max_lag, max_lag + 1):
        shifted = np.roll(y, -lag)  # shifted[t] = y[t + lag]: b `lag` steps after a lines up with a
        found[lag] = float(np.corrcoef(x, shifted)[0, 1])
    best = max(found, key=lambda lag: (round(abs(found[lag]), 6), -abs(lag)))  # ties: the smaller shift
    return Match(found[0], found[best], best)


def _profile(fp: Fingerprint, period: str) -> list[float] | None:
    feature, _ = PERIODS[period]
    strength = (fp.feature("cycle.strength") or {}).get(period)
    profile = fp.feature(feature)
    if strength is None or strength < MIN_STRENGTH or not isinstance(profile, list):
        return None
    return profile


def similar(target: Fingerprint, others: list[Fingerprint], limit: int = 50) -> list[Similar]:
    """The others most like the target, best first; those sharing no period with it are left out."""
    mine = {period: _profile(target, period) for period in PERIODS}
    periods = [period for period, profile in mine.items() if profile]
    found = []
    for other in others:
        if other.key == target.key or not periods:
            continue
        matches = dict.fromkeys(PERIODS)
        for period in periods:
            theirs = _profile(other, period)
            matches[period] = compare(mine[period], theirs, PERIODS[period][1]) if theirs else None
        compared = [m for m in matches.values() if m]
        if compared:
            found.append(Similar(other.key, sum(abs(m.r) for m in compared) / len(periods), **matches))
    found.sort(key=lambda s: -s.score)
    return found[:limit]


LAG = "remainder.lag"


def moves_with(target: Fingerprint, others: list[Fingerprint]) -> list[dict]:
    """Every stored remainder.lag of the target, from its side: lag_h > 0 means the other follows it.
    Strongest |r| first."""
    found = {k: v for k, v in target.compared(LAG).items() if v and v.get("r") is not None}
    for other in others:
        v = other.compared(LAG).get(target.key)
        if other.key != target.key and other.key not in found and v and v.get("r") is not None:
            found[other.key] = {**v, **{k: -v[k] for k in ("lag_h", "lag_change_h") if v.get(k) is not None}}
    return sorted(({"key": k, **v} for k, v in found.items()), key=lambda x: -abs(x["r"]))
