"""Series compared by the profiles in their fingerprints."""

from __future__ import annotations

import numpy as np
import pytest

from free_data.fingerprints import Fingerprint, Window
from free_data.similarity import compare, moves_with, similar

WINDOW = Window("2025-01-01", "2025-12-31")
HOURS = np.arange(24)


def day(peak: float) -> list[float]:
    return np.cos(2 * np.pi * (HOURS - peak) / 24).tolist()


def fp(key: str, profile: list[float] | None, strength: float = 0.5) -> Fingerprint:
    return Fingerprint(key, WINDOW, {"profile.day": {"version": 1, "value": profile},
                                     "cycle.strength": {"version": 1, "value": {"day": strength, "year": None}}})


def test_a_later_peak_is_a_positive_lag():
    match = compare(day(12), day(15), max_lag=6)
    assert match.lag == 3 and match.r == pytest.approx(1)
    assert match.r0 == pytest.approx(np.cos(2 * np.pi * 3 / 24))
    assert compare(day(15), day(12), max_lag=6).lag == -3


def test_the_opposite_is_a_negative_r_not_half_a_day_later():
    match = compare(day(12), day(0), max_lag=6)
    assert match.lag == 0 and match.r == pytest.approx(-1)


def test_flat_or_odd_profiles_are_not_compared():
    assert compare([1.0] * 24, day(12), max_lag=6) is None
    assert compare(day(12), day(12)[:12], max_lag=6) is None


def test_similar_ranks_and_skips_what_has_no_cycle():
    target = fp("s:a", day(12))
    others = [target, fp("s:near", day(13)), fp("s:far", day(16)), fp("s:weak", day(12), strength=0.01),
              fp("s:none", None)]
    found = similar(target, others)
    assert [s.key for s in found] == ["s:near", "s:far"]
    assert found[0].day.lag == 1 and found[0].year is None and found[0].score == pytest.approx(1)


def test_a_period_only_the_target_has_lowers_the_score():
    year = np.cos(2 * np.pi * np.arange(12) / 12).tolist()
    both = Fingerprint("s:a", WINDOW, {"profile.day": {"version": 1, "value": day(12)},
                                       "profile.year": {"version": 1, "value": year},
                                       "cycle.strength": {"version": 1, "value": {"day": 0.5, "year": 0.5}}})
    found = similar(both, [fp("s:day-only", day(12))])
    assert found[0].score == pytest.approx(0.5) and found[0].year is None


def test_moves_with_reads_a_pair_from_either_side():
    lag = {"r0": 0.5, "r": -0.8, "lag_h": 3, "r_change": 0.4, "lag_change_h": 2, "n": 100}
    a = Fingerprint("s:a", WINDOW, {}, {"remainder.lag": {"version": 1, "values": {"s:b": lag, "s:c": None}}})
    b, c = Fingerprint("s:b", WINDOW), Fingerprint("s:c", WINDOW)
    assert moves_with(a, [a, b, c]) == [{"key": "s:b", **lag}]
    assert moves_with(b, [a, b, c]) == [{"key": "s:a", **lag, "lag_h": -3, "lag_change_h": -2}]
    assert moves_with(c, [a, b, c]) == []
