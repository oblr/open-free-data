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


# ---- the remainder ----

RNG = np.random.default_rng(7)
WEEKLY = np.where((HOURS // 24 + 2) % 7 >= 5, -1.0, 0.0)  # 2025-01-01 was a Wednesday: weekends lower
CALENDAR = 10 * DAILY_SINE + 3 * WEEKLY + 5 * np.sin(2 * np.pi * HOURS / len(HOURS))


def ar1(n: int, hours: float, seed: int = 3) -> np.ndarray:
    """Red noise whose autocorrelation falls to 1/e after `hours`."""
    phi, rng = np.exp(-1 / hours), np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal()
    return x


def test_a_series_the_calendar_explains_leaves_nothing():
    s = hourly(CALENDAR)
    assert m.remainder_share(s) < 0.01  # the yearly sine isn't flat within a month: a little is left
    week = 10 * DAILY_SINE + 3 * WEEKLY  # flat within each month, so all that is left is the noise
    noisy = week + RNG.normal(size=len(HOURS))
    assert m.remainder_share(hourly(noisy)) == pytest.approx(1 / np.var(noisy), rel=0.1)


def test_memory_finds_the_decorrelation_time():
    s = hourly(CALENDAR + 5 * ar1(len(HOURS), 24))
    memory = m.remainder_memory(s)
    assert memory["decorrelation_h"] == pytest.approx(24, rel=0.2)
    assert memory["1h"] > 0.9 and memory["168h"] < 0.1
    white = m.remainder_memory(hourly(RNG.normal(size=len(HOURS))))
    assert white["decorrelation_h"] < 1 and abs(white["1h"]) < 0.05


def test_bands_put_slow_swings_in_the_slow_band():
    fast, slow = (m.remainder_bands(hourly(CALENDAR + 5 * ar1(len(HOURS), h))) for h in (1, 24 * 20))
    assert sum(fast.values()) == pytest.approx(1)
    assert max(fast, key=fast.get) == "<6h" and max(slow, key=slow.get) == ">10d"


def test_shape_and_ramps_see_spikes_upwards():
    rng = np.random.default_rng(11)
    values = CALENDAR + rng.normal(size=len(HOURS))
    values[rng.choice(len(HOURS), 100, replace=False)] += 30
    shape = m.remainder_shape(hourly(values))
    assert shape["skewness"] > 1 and shape["kurtosis"] > 10
    assert shape["tail_up"] > 0.6  # the noise has some 3-sigma values too, as many down as up
    ramps = m.remainder_ramps(hourly(values))
    assert ramps["p99"] > 3 and ramps["rise_fall"] == pytest.approx(1, rel=0.3)  # a spike goes up and down


def test_episodes_count_runs_on_one_side():
    wave = np.where(HOURS % 96 < 48, 1.0, -1.0) + 0.01 * RNG.normal(size=len(HOURS))  # 2 days up, 2 down
    episodes = m.remainder_episodes(hourly(wave))
    assert episodes["mean_h"] == pytest.approx(48, rel=0.15) and episodes["longest_above_h"] >= 48


def test_clustering_and_spread_by_hour():
    calm_and_storm = RNG.normal(size=len(HOURS)) * np.where((HOURS // (24 * 7)) % 2, 5, 1)  # alternating weeks
    assert m.remainder_clustering(hourly(calm_and_storm)) > 0.1
    assert abs(m.remainder_clustering(hourly(RNG.normal(size=len(HOURS))))) < 0.05
    noon_noise = CALENDAR + RNG.normal(size=len(HOURS)) * np.where(HOURS % 24 == 12, 5, 1)
    spread = m.remainder_spread_day(hourly(noon_noise))
    assert len(spread) == 24 and int(np.argmax(spread)) == 12


def test_the_remainder_needs_an_hourly_step_and_four_weeks():
    assert m.remainder_share(hourly(CALENDAR[: 24 * 20])) is None
    days = Series("s:x", Dataset("x", "x"), np.arange(400).astype("datetime64[D]").astype("datetime64[ms]"),
                  np.arange(400.0), "MW", "UTC", Window("1970-01-01", "1971-02-04", "day"))
    assert m.remainder_memory(days) is None
    s = hourly(RNG.normal(size=24 * 60), measure="direction")
    assert m.remainder_share(s) is None and m.remainder_bands(s) is None


def test_features_of_one_series_share_its_model(monkeypatch):
    fits = []
    monkeypatch.setattr(m, "REMAINDERS", {})  # the fake models stay in this test
    monkeypatch.setattr(m, "_fit_remainder", lambda s: fits.append(s) or None)
    s = hourly(CALENDAR + RNG.normal(size=len(HOURS)))
    m.remainder_share(s)
    m.remainder_memory(hourly(s.values.copy()))  # the same values at the same times: the same model
    m.remainder_memory(hourly(s.values + 1))
    assert len(fits) == 2


def test_remainder_lag_finds_who_follows_whom():
    weather = ar1(len(HOURS) + 10, 24, seed=5)
    a = hourly(CALENDAR + 5 * weather[10:] + RNG.normal(size=len(HOURS)))
    b = hourly(-3 * CALENDAR + 4 * weather[5:-5] + RNG.normal(size=len(HOURS)))  # the same weather, 5 h later
    found = m.remainder_lag(a, b)
    assert found["lag_h"] == 5 and found["lag_change_h"] == 5 and found["r"] > 0.8
    assert found["r0"] < found["r"] and found["n"] == len(HOURS)
    back = m.remainder_lag(b, a)
    assert back["lag_h"] == -5 and back["r"] == pytest.approx(found["r"])
    opposite = m.remainder_lag(a, hourly(-b.values))
    assert opposite["lag_h"] == 5 and opposite["r"] == pytest.approx(-found["r"])


def test_remainder_lag_with_gaps_and_unrelated_series():
    a = CALENDAR + 5 * ar1(len(HOURS), 24, seed=8)
    a[1000:3000] = np.nan
    found = m.remainder_lag(hourly(a), hourly(CALENDAR + 5 * ar1(len(HOURS), 24, seed=9)))
    assert abs(found["r_change"]) < 0.1 and found["n"] == len(HOURS) - 2000
    assert 50 < found["n_eff"] < 500  # a day's memory: some 6760 hours, a few hundred independent ones
    assert abs(found["r"]) < 2.5 / np.sqrt(found["n_eff"])  # unrelated: within chance
    assert m.remainder_lag(hourly(a), hourly(np.arange(24.0 * 10))) is None  # too short for a remainder
