"""Series convert between units and aggregate over time before the page shows them."""

from __future__ import annotations

import numpy as np
import pytest

from free_data import sources, transforms
from free_data.sources import Dataset, Table
from free_data.transforms import Transform

QUARTER = 900_000
DATASETS = {
    "solar_mwh": Dataset("solar_mwh", "Solar DE", unit="MWh", measure="energy"),
    "solar_mw": Dataset("solar_mw", "Solar FR", unit="MW", measure="power"),
    "rain": Dataset("rain", "Rain", unit="mm", measure="amount"),
    "temp": Dataset("temp", "Temperature", unit="°C", measure="temperature"),
}


def table(columns: tuple[str, ...], time_ms: np.ndarray, values: np.ndarray) -> Table:
    return Table(columns, time_ms.astype("datetime64[ms]"), values)


def test_interval_lengths_take_the_next_time_and_the_last_the_one_before():
    time = np.array([0, QUARTER, 2 * QUARTER, 4 * QUARTER], np.int64)
    assert transforms.hours(time).tolist() == [0.25, 0.25, 0.5, 0.5]
    assert np.isnan(transforms.hours(np.array([0], np.int64))).all()


def test_energy_per_quarter_hour_becomes_power_and_back():
    t = table(("solar_mwh", "solar_mw"), np.arange(4) * QUARTER, np.array([[250.0, 1000.0]] * 4))
    out, units = transforms.apply(t, DATASETS, Transform({"solar_mwh": "MW", "solar_mw": "MWh"}), "UTC")
    assert units == ["MW", "MWh"]
    assert out.values[:, 0].tolist() == [1000.0] * 4  # 250 MWh in a quarter hour is 1000 MW
    assert out.values[:, 1].tolist() == [250.0] * 4


def test_units_of_one_dimension_need_no_interval():
    t = table(("solar_mwh",), np.array([0], np.int64), np.array([[1500.0]]))
    out, units = transforms.apply(t, DATASETS, Transform({"solar_mwh": "GWh"}), "UTC")
    assert units == ["GWh"] and out.values[0, 0] == 1.5


def test_daily_steps_sum_amounts_and_average_the_rest():
    hours = np.arange(48) * 3_600_000
    values = np.column_stack([np.ones(48), np.full(48, 2.0), np.arange(48.0)])
    t = table(("solar_mwh", "rain", "temp"), hours, values)
    out, units = transforms.apply(t, DATASETS, Transform(step="day"), "UTC")
    assert units == ["MWh", "mm", "°C"]
    assert out.time.astype(np.int64).tolist() == [0, 86_400_000]
    assert out.values[:, 0].tolist() == [24.0, 24.0]  # energy adds up
    assert out.values[:, 1].tolist() == [48.0, 48.0]  # so does precipitation
    assert out.values[:, 2].tolist() == [11.5, 35.5]  # temperature averages


def test_power_turned_into_energy_adds_up_per_day_and_energy_into_power_averages():
    quarters = np.arange(96) * QUARTER
    t = table(("solar_mw", "solar_mwh"), quarters, np.column_stack([np.full(96, 1000.0), np.full(96, 250.0)]))
    out, units = transforms.apply(t, DATASETS, Transform({"solar_mw": "GWh", "solar_mwh": "MW"}, "day"), "UTC")
    assert units == ["GWh", "MW"]
    assert out.values.tolist() == [[24.0, 1000.0]]  # 1 GW for a day is 24 GWh; the mean power stays 1000 MW


def test_months_begin_in_the_source_zone():
    zone = "Europe/Paris"
    days = np.array([sources.epoch_ms(d, zone) for d in ("2024-01-31", "2024-02-01", "2024-02-29")], np.int64)
    t = table(("rain",), days, np.array([[1.0], [2.0], [3.0]]))
    out, _ = transforms.apply(t, DATASETS, Transform(step="month"), zone)
    assert out.time.astype(np.int64).tolist() == [sources.epoch_ms(d, zone) for d in ("2024-01-01", "2024-02-01")]
    assert out.values[:, 0].tolist() == [1.0, 5.0]


def test_impossible_transformations_are_refused():
    with pytest.raises(ValueError, match="can't convert MWh to °C"):
        transforms.check(Transform({"solar_mwh": "°C"}), DATASETS)
    with pytest.raises(ValueError, match="step must be one of"):
        transforms.check(Transform(step="fortnight"), DATASETS)
