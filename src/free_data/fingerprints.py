"""Precomputed descriptions of series and comparisons between them.

Two kinds of measurement, both plain functions registered under a name:

    Feature     one series -> a value         a daily profile, the share of zeros
    Comparison  two series -> a value         a correlation, the best lag

Nothing here knows what is measured. A module registers its measurements
when it is imported (a co-movement module, for example):

    from free_data.fingerprints import comparison, feature

    @feature("profile.day", version=1)
    def daily_profile(s: Series) -> list[float]: ...

    @comparison("anomaly_r", version=1)
    def anomaly_r(a: Series, b: Series) -> float | None: ...   # a and b share their times

`precompute(keys, window, pairs)` reads the series once over a reference
window, brought to one time step (transforms.py), runs every feature on
each series and every comparison on the given pairs, and stores the results
as one Fingerprint per dataset:

    <cache root>/_fingerprints/<source>/<dataset>.json

A stored value is reused until its measurement's version changes or the
fingerprint was made over another window; only what is missing or out of
date is computed, and nothing is read when nothing is. Bump a
measurement's version when its result changes. The pairs to compare come
from the caller, e.g. related_pairs(), so the metadata (relations.py)
proposes candidates and the data measures them.

Values are what JSON holds: a number, None, a list or a dict of numbers.
NaN and infinities are stored as None.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from free_data import cache, relations, sources, transforms

FOLDER = "_fingerprints"  # in the cache root, next to the sources' folders

Value = float | int | None | list | dict


@dataclass(frozen=True)
class Series:
    """One dataset's values over a window: what a measurement gets."""

    key: str  # "<source>:<dataset>"
    dataset: sources.Dataset
    time: np.ndarray  # datetime64[ms], UTC, increasing
    values: np.ndarray  # float64, NaN for gaps
    unit: str  # after the window's step; the dataset's own unit
    zone: str  # the source's timezone, where its days begin


@dataclass(frozen=True)
class Window:
    """The reference period measurements are made over."""

    start: str  # ISO date or date-time, read in each source's timezone
    end: str
    step: str = "hour"  # one of transforms.STEPS, so series of different sources meet


@dataclass(frozen=True)
class Measurement:
    name: str
    version: int
    function: Callable[..., Value]
    symmetric: bool = True  # comparisons only: f(a, b) == f(b, a), so it is stored on both sides


@dataclass
class Fingerprint:
    """What has been measured for one dataset, over one window."""

    key: str
    window: Window
    # name -> {"version": int, "value": Value}
    features: dict[str, dict] = field(default_factory=dict)
    # name -> {"version": int, "values": {other key: Value}}
    comparisons: dict[str, dict] = field(default_factory=dict)

    def feature(self, name: str) -> Value:
        """A stored feature's value, None if it hasn't been computed."""
        return self.features.get(name, {}).get("value")

    def compared(self, name: str) -> dict[str, Value]:
        """A comparison's values against other datasets, by their key."""
        return self.comparisons.get(name, {}).get("values", {})


FEATURES: dict[str, Measurement] = {}
COMPARISONS: dict[str, Measurement] = {}


def feature(name: str, version: int = 1) -> Callable:
    """Registers a function of one Series as a feature."""
    def register(function: Callable[[Series], Value]) -> Callable[[Series], Value]:
        _register(FEATURES, Measurement(name, version, function))
        return function
    return register


def comparison(name: str, version: int = 1, symmetric: bool = True) -> Callable:
    """Registers a function of two Series on the same times as a comparison."""
    def register(function: Callable[[Series, Series], Value]) -> Callable[[Series, Series], Value]:
        _register(COMPARISONS, Measurement(name, version, function, symmetric))
        return function
    return register


def _register(table: dict[str, Measurement], m: Measurement) -> None:
    if m.name in table and table[m.name].function is not m.function:
        raise ValueError(f"two measurements named {m.name!r}")
    table[m.name] = m


# ---- storage ----


class Store:
    """Fingerprints as JSON files, one per dataset, in the cache."""

    def __init__(self, base: Path | None = None) -> None:
        self.dir = (base or cache.root()) / FOLDER

    def path(self, key: str) -> Path:
        source, _, dataset = key.partition(":")
        target = (self.dir / source / f"{dataset.replace('/', '__')}.json").resolve()
        if not target.is_relative_to(self.dir.resolve()):
            raise ValueError(f"not a dataset key: {key!r}")
        return target

    def load(self, key: str) -> Fingerprint | None:
        path = self.path(key)
        if not path.exists():
            return None
        raw = json.loads(path.read_text())
        return Fingerprint(raw["key"], Window(**raw["window"]), raw["features"], raw["comparisons"])

    def save(self, fingerprint: Fingerprint) -> None:
        path = self.path(fingerprint.key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(fingerprint), ensure_ascii=False))
        tmp.replace(path)


# ---- reading ----

Reader = Callable[[list[str], Window], dict[str, Series]]


def read(keys: list[str], window: Window) -> dict[str, Series]:
    """The datasets over the window, through their plugins and the cache, at the window's step."""
    wanted: dict[str, list[str]] = {}
    for key in keys:
        source, _, dataset = key.partition(":")
        wanted.setdefault(source, []).append(dataset)
    found = {}
    for source, ids in wanted.items():
        plugin = sources.discover()[source]
        described = {d.id: d for d in plugin.datasets()}
        start = sources.moment(window.start, plugin.timezone, end=False)
        end = sources.moment(window.end, plugin.timezone, end=True)
        table = plugin.read(ids, start, end, plugin.resolve({}))
        table, column_units = transforms.apply(table, described, transforms.Transform(step=window.step),
                                               plugin.timezone)
        for i, (dataset, unit) in enumerate(zip(table.columns, column_units, strict=True)):
            found[f"{source}:{dataset}"] = Series(f"{source}:{dataset}", described[dataset], table.time,
                                                  table.values[:, i], unit, plugin.timezone)
    return found


def align(a: Series, b: Series) -> tuple[Series, Series]:
    """Both series on the times they share; values may still be NaN."""
    _, ia, ib = np.intersect1d(a.time, b.time, assume_unique=True, return_indices=True)
    return (Series(a.key, a.dataset, a.time[ia], a.values[ia], a.unit, a.zone),
            Series(b.key, b.dataset, b.time[ib], b.values[ib], b.unit, b.zone))


def related_pairs(targets: list[relations.Entry], entries: list[relations.Entry]) -> list[tuple[str, str]]:
    """Each target with the series relations.py relates it to: the candidates worth comparing."""
    return [(r.to, r.key) for r in relations.related(targets, entries)]


# ---- computing ----


def precompute(
    keys: Iterable[str],
    window: Window,
    pairs: Iterable[tuple[str, str]] = (),
    *,
    features: Iterable[Measurement] | None = None,
    comparisons: Iterable[Measurement] | None = None,
    store: Store | None = None,
    reader: Reader = read,
) -> dict[str, Fingerprint]:
    """The fingerprints of `keys` and of every dataset in `pairs`, with what was missing computed and stored.

    `features` and `comparisons` default to the registered ones. A pair
    (a, b) of an asymmetric comparison is stored under a only; pass (b, a)
    as well for the other direction.
    """
    store = store or Store()
    features = list(FEATURES.values() if features is None else features)
    comparisons = list(COMPARISONS.values() if comparisons is None else comparisons)
    pairs = list(dict.fromkeys((a, b) for a, b in pairs if a != b))
    keys = list(dict.fromkeys([*keys, *(k for pair in pairs for k in pair)]))

    prints = {}
    for key in keys:
        found = store.load(key)
        prints[key] = found if found and found.window == window else Fingerprint(key, window)

    todo_features = {k: [m for m in features if _stale(prints[k].features.get(m.name), m)] for k in keys}
    todo_pairs = [(a, b, m) for a, b in pairs for m in comparisons
                  if _stale(prints[a].comparisons.get(m.name), m, b)
                  and not (m.symmetric and (b, a) in pairs and pairs.index((b, a)) < pairs.index((a, b)))]
    needed = {k for k, ms in todo_features.items() if ms} | {k for a, b, _ in todo_pairs for k in (a, b)}
    if not needed:
        return prints

    series = reader(sorted(needed), window)
    changed = set()
    for key, ms in todo_features.items():
        for m in ms:
            prints[key].features[m.name] = {"version": m.version, "value": _clean(m.function(series[key]))}
            changed.add(key)
    for a, b, m in todo_pairs:
        value = _clean(m.function(*align(series[a], series[b])))
        sides = [(a, b), (b, a)] if m.symmetric else [(a, b)]
        for this, other in sides:
            entry = prints[this].comparisons.get(m.name)
            if not entry or entry["version"] != m.version:  # an old version's values against others go
                entry = prints[this].comparisons[m.name] = {"version": m.version, "values": {}}
            entry["values"][other] = value
            changed.add(this)
    for key in changed:
        store.save(prints[key])
    return prints


def _stale(entry: dict | None, m: Measurement, other: str | None = None) -> bool:
    if not entry or entry["version"] != m.version:
        return True
    return other is not None and other not in entry["values"]


def _clean(value: object) -> Value:
    """A measurement's result as JSON holds it: numpy numbers become floats, NaN becomes None."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
