"""The registered features, on synthetic series with known answers."""

from __future__ import annotations

import numpy as np
import pytest

from free_data import measurements as m
from free_data.fingerprints import FEATURES, Series, Window
from free_data.sources import Dataset

YEAR = Window("2025-01-01", "2025-12-31", "hour")


def hourly(values, start="2025-01-01T00", zone="UTC", measure="power", window=YEAR) -> Series:
    time = np.datetime64(start, "h") + np.arange(len(values)).astype("timedelta64[h]")
    return Series("s:x", Dataset("x", "x", measure=measure), time.astype("datetime64[ms]"),
                  np.asarray(values, float), "MW", zone, window)


HOURS = np.arange(365 * 24)
DAILY_SINE = np.sin(2 * np.pi * (HOURS % 24) / 24)  # peaks at 06 UTC


def test_all_eight_are_registered():
    assert {"coverage", "summary", "zero_share", "profile.day", "profile.year", "cycle.strength", "autocorr",
            "trend"} <= set(FEATURES)


def test_coverage_counts_the_steps_of_the_window():
    values = np.ones(365 * 24)
    values[: 365 * 12] = np.nan
    assert m.coverage(hourly(values)) == pytest.approx(0.5)
    assert m.coverage(hourly(np.ones(24 * 31), window=Window("2025-01-01", "2025-01-31", "day"))) == 1.0
    assert m.coverage(hourly(np.ones(10), window=None)) == 1.0


def test_summary_and_its_circular_form():
    s = m.summary(hourly(np.arange(101.0)))
    assert (s["min"], s["median"], s["max"], s["p5"]) == (0.0, 50.0, 100.0, 5.0)
    d = m.summary(hourly([350.0, 10.0, 350.0, 10.0], measure="direction"))
    assert d["mean"] == pytest.approx(0.0, abs=1e-6) or d["mean"] == pytest.approx(360.0)
    assert d["std"] < 15 and "min" not in d


def test_zero_share():
    assert m.zero_share(hourly([0, 0, 1, np.nan, 2])) == 0.5
    assert m.zero_share(hourly([0, 90], measure="direction")) is None


def test_profile_day_is_in_local_hours():
    p = m.profile_day(hourly(DAILY_SINE))
    assert len(p) == 24 and int(np.argmax(p)) == 6
    berlin = m.profile_day(hourly(DAILY_SINE, zone="Europe/Berlin"))  # 06 UTC is 07 or 08 there
    assert int(np.argmax(berlin)) in (7, 8)
    assert m.profile_day(hourly(DAILY_SINE[: 24 * 5])) is None  # under a week


def test_profile_day_needs_an_hourly_step():
    days = Series("s:x", Dataset("x", "x"), np.arange(400).astype("datetime64[D]").astype("datetime64[ms]"),
                  np.ones(400), "MW", "UTC", Window("1970-01-01", "1971-02-04", "day"))
    assert m.profile_day(days) is None


def test_profile_year_and_cycle_strength():
    season = -np.cos(2 * np.pi * HOURS / len(HOURS))  # low in January, high in July
    p = m.profile_year(hourly(season))
    assert len(p) == 12 and int(np.argmax(p)) in (5, 6) and int(np.argmin(p)) in (0, 11)
    c = m.cycle_strength(hourly(DAILY_SINE))
    assert c["day"] == pytest.approx(1.0) and c["week"] < 0.01 and c["year"] < 0.01
    assert m.cycle_strength(hourly(season))["year"] > 0.9
    short = m.cycle_strength(hourly(DAILY_SINE[: 24 * 30]))
    assert short["year"] is None and short["day"] == pytest.approx(1.0)


def test_autocorr():
    a = m.autocorr(hourly(DAILY_SINE))
    assert a["24h"] == pytest.approx(1.0) and a["168h"] == pytest.approx(1.0)
    assert a["1h"] == pytest.approx(np.cos(2 * np.pi / 24), abs=1e-3)
    noise = m.autocorr(hourly(np.random.default_rng(1).normal(size=2000)))
    assert abs(noise["1h"]) < 0.1


def test_trend_needs_two_years_and_ignores_the_seasons():
    hours = np.arange(3 * 365 * 24)
    rising = 2.0 * hours / (365.25 * 24) + 10 * np.sin(2 * np.pi * hours / (365.25 * 24))  # 2 per year
    s = hourly(rising, window=Window("2023-01-01", "2025-12-31", "hour"))
    assert m.trend(s) == pytest.approx(2.0, rel=0.05)
    assert m.trend(hourly(DAILY_SINE)) is None


def test_directions_only_get_a_summary():
    s = hourly(np.random.default_rng(2).uniform(0, 360, 24 * 400), measure="direction")
    assert m.summary(s) is not None
    assert all(f(s) is None for f in (m.zero_share, m.profile_day, m.profile_year, m.cycle_strength,
                                      m.autocorr, m.trend))
