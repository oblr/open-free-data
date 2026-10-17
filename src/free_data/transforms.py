"""Transformations of a series before it is shown or downloaded.

    unit    another unit of the same dimension (MWh -> GWh), or between an
            amount per interval and its rate (MWh per quarter hour <-> MW),
            through units.convert
    step    totals or means per hour, day or month in the source's timezone

The unit comes first, then the step, so MW per quarter hour turned into MWh
and then aggregated per day is the energy of each day. Amounts per interval
(energy, precipitation, sunshine) are summed, everything else is averaged;
a sum over a period with gaps covers only the values that are there.

The interval of a value is the time to the next one, the last value taking
the one before: times label the start of their interval, and a day with a
clock change has 23 or 25 hours.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from free_data import sources, units

STEPS = ("hour", "day", "month")
SUMMED = ("energy", "amount", "duration")  # measures (vocabulary.MEASURES) that are an amount per interval


@dataclass(frozen=True)
class Transform:
    units: dict[str, str] = field(default_factory=dict)  # dataset id -> unit to convert to
    step: str = ""  # one of STEPS; "" keeps the source's own time step


def hours(time: np.ndarray) -> np.ndarray:
    """The length in hours of each interval starting at `time` (epoch ms or datetime64[ms], increasing)."""
    ms = time.astype(np.int64)
    if len(ms) < 2:
        return np.full(len(ms), np.nan)
    steps = np.diff(ms) / 3_600_000
    return np.append(steps, steps[-1])


def summed(measure: str, unit: str, target: str) -> bool:
    """Whether values of `measure` in `unit`, converted to `target`, add up over time (else they average)."""
    if units.dimension(unit) == units.dimension(target):
        return measure in SUMMED
    return bool(units.DIMENSIONS[units.dimension(target)].rate)  # MW -> MWh adds up, MWh -> MW averages


def check(transform: Transform, datasets: dict[str, sources.Dataset]) -> None:
    """Raises ValueError for a step or unit the datasets can't be transformed to."""
    if transform.step and transform.step not in STEPS:
        raise ValueError(f"step must be one of {STEPS}, got {transform.step!r}")
    for dataset, target in transform.units.items():
        unit = datasets[dataset].unit
        if not units.convertible(unit, target):
            raise ValueError(f"{datasets[dataset].label}: can't convert {unit or 'no unit'} to {target}")


def apply(
    table: sources.Table, datasets: dict[str, sources.Dataset], transform: Transform, zone: str
) -> tuple[sources.Table, list[str]]:
    """The table transformed, and the unit of each of its columns afterwards.

    `datasets` describes the table's columns, by id; `zone` is the source's
    timezone, where days and months begin.
    """
    check(transform, datasets)
    lengths = hours(table.time)
    columns, column_units, sums = [], [], []
    for i, dataset in enumerate(table.columns):
        d = datasets[dataset]
        target = units.unit(transform.units.get(dataset) or d.unit)
        values = table.values[:, i]
        if target != d.unit:
            with np.errstate(invalid="ignore"):
                values = units.convert(values, d.unit, target, hours=lengths)
        columns.append(values)
        column_units.append(target)
        sums.append(summed(d.measure, d.unit, target))
    values = np.column_stack(columns) if columns else table.values
    time = table.time
    if transform.step:
        ms = table.time.astype(np.int64)
        parts = []
        for how in ("sum", "mean"):
            wanted = np.array([s == (how == "sum") for s in sums], bool)
            labels, part = sources.aggregate(ms, values[:, wanted], transform.step, zone, how)
            parts.append((wanted, part))
        aggregated = np.full((len(labels), len(columns)), np.nan)
        for wanted, part in parts:
            aggregated[:, wanted] = part
        time, values = labels.astype("datetime64[ms]"), aggregated
    return sources.Table(table.columns, time, values), column_units
