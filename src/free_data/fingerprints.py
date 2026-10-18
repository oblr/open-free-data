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
import logging
import math
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from free_data import cache, relations, sources, transforms

log = logging.getLogger(__name__)

CHUNK = 25  # datasets per plugin.read; a chunk that fails is read one by one
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
    window: Window | None = None  # what it was read over; None when made by hand


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

    def all(self) -> list[Fingerprint]:
        """Every stored fingerprint, by source and dataset; a file that can't be read is skipped."""
        found = []
        for path in sorted(self.dir.glob("*/*.json")):
            try:
                raw = json.loads(path.read_text())
                found.append(Fingerprint(raw["key"], Window(**raw["window"]), raw["features"], raw["comparisons"]))
            except (OSError, ValueError, KeyError, TypeError) as error:
                log.warning("%s: skipping, can't read it: %s", path, error)
        return found

    def save(self, fingerprint: Fingerprint) -> None:
        path = self.path(fingerprint.key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(fingerprint), ensure_ascii=False))
        tmp.replace(path)


# ---- reading ----

Reader = Callable[[list[str], Window], dict[str, Series]]


def read(keys: list[str], window: Window) -> dict[str, Series]:
    """The datasets over the window, through their plugins and the cache, at the window's step.

    A dataset that can't be read (under cache.refreshing("offline"): one
    whose files aren't all cached) is left out and logged; the rest are read.
    """
    wanted: dict[str, list[str]] = {}
    for key in keys:
        source, _, dataset = key.partition(":")
        wanted.setdefault(source, []).append(dataset)
    found = {}
    for source, ids in wanted.items():
        plugin = sources.discover()[source]
        try:
            described = {d.id: d for d in plugin.datasets()}
        except Exception as error:  # noqa: BLE001 (e.g. its dataset list isn't cached)
            log.warning("%s: can't list its datasets, skipping %d: %s", source, len(ids), error)
            continue
        start = sources.moment(window.start, plugin.timezone, end=False)
        end = sources.moment(window.end, plugin.timezone, end=True)
        for i in range(0, len(ids), CHUNK):
            for table in _tables(plugin, ids[i:i + CHUNK], start, end):
                table, column_units = transforms.apply(table, described, transforms.Transform(step=window.step),
                                                       plugin.timezone)
                for j, (dataset, unit) in enumerate(zip(table.columns, column_units, strict=True)):
                    found[f"{source}:{dataset}"] = Series(f"{source}:{dataset}", described[dataset], table.time,
                                                          table.values[:, j], unit, plugin.timezone, window)
    return found


def _tables(plugin: sources.Plugin, ids: list[str], start, end) -> list[sources.Table]:
    """The chunk as one table, or, if that fails, a table per dataset that can be read."""
    try:
        return [plugin.read(ids, start, end, plugin.resolve({}))]
    except Exception as error:  # noqa: BLE001 (one unreadable dataset doesn't stop the others)
        if len(ids) == 1:
            log.info("%s: skipping %s: %s", plugin.id, ids[0], error)
            return []
    return [t for dataset in ids for t in _tables(plugin, [dataset], start, end)]


def align(a: Series, b: Series) -> tuple[Series, Series]:
    """Both series on the times they share; values may still be NaN."""
    _, ia, ib = np.intersect1d(a.time, b.time, assume_unique=True, return_indices=True)
    return (Series(a.key, a.dataset, a.time[ia], a.values[ia], a.unit, a.zone, a.window),
            Series(b.key, b.dataset, b.time[ib], b.values[ib], b.unit, b.zone, b.window))


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
        if key not in series:  # couldn't be read: nothing stored, so a later run tries again
            continue
        for m in ms:
            try:
                value = _clean(m.function(series[key]))
            except Exception as error:  # noqa: BLE001 (a failing measurement is logged, the others still run)
                log.warning("%s: %s failed: %s", key, m.name, error)
                continue
            prints[key].features[m.name] = {"version": m.version, "value": value}
            changed.add(key)
    for a, b, m in todo_pairs:
        if a not in series or b not in series:
            continue
        try:
            value = _clean(m.function(*align(series[a], series[b])))
        except Exception as error:  # noqa: BLE001 (as for features)
            log.warning("%s, %s: %s failed: %s", a, b, m.name, error)
            continue
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


# ---- command line ----


def main(argv: list[str] | None = None) -> None:
    """free-data-fingerprints: compute the registered features of cached series, and show them; with --compare,
    also how their remainders move with every other stored series."""
    import argparse  # noqa: PLC0415 (only the command line needs these)
    import sys  # noqa: PLC0415
    from datetime import date  # noqa: PLC0415

    from free_data import ingest, measurements  # noqa: F401, PLC0415 (measurements registers the features)

    last_year = date.today().year - 1
    parser = argparse.ArgumentParser(description="Compute fingerprints of series, from the cache by default.")
    parser.add_argument("source", nargs="?", help="a source id with a plugin, e.g. de-smard")
    parser.add_argument("datasets", nargs="*", help="dataset ids")
    parser.add_argument("--all", action="store_true", help="every dataset of the source; those not cached are skipped")
    parser.add_argument("--match", nargs="+", metavar="WORD", help="the datasets whose names contain these words")
    parser.add_argument("--start", default=f"{last_year}-01-01", help="window start (default: last year)")
    parser.add_argument("--end", default=f"{last_year}-12-31", help="window end, inclusive")
    parser.add_argument("--step", choices=transforms.STEPS, default="hour")
    parser.add_argument("--download", action="store_true",
                        help="download what isn't cached (throttled) instead of skipping it")
    parser.add_argument("--compare", action="store_true",
                        help="also compare them with every stored fingerprint over the same window (remainder.lag); "
                             "without a source, compare every stored pair")
    parser.add_argument("--json", action="store_true", help="the fingerprints as JSON, one per line")
    parser.add_argument("-v", "--verbose", action="store_true", help="also why each skipped dataset was skipped")
    args = parser.parse_args(argv)

    logging.basicConfig(format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S", level=logging.WARNING)
    logging.getLogger("free_data").setLevel(logging.INFO if args.verbose else logging.WARNING)
    window = Window(args.start, args.end, args.step)
    if args.source is None:
        if not args.compare:
            parser.error("name a source, or use --compare alone to compare every stored fingerprint")
        with cache.refreshing("never" if args.download else "offline"):
            _compare([], window)
        return
    if args.source not in sources.discover():
        parser.error(f"no plugin for source {args.source!r}")
    plugin = sources.discover()[args.source]
    with cache.refreshing("never" if args.download else "offline"):
        try:
            datasets = {d.id: d for d in plugin.datasets()}
        except cache.NotCached as error:
            parser.error(f"{args.source}: its dataset list isn't cached ({error}); run with --download once")
        ids = list(args.datasets)
        if args.match:
            ids += ingest._matching(args.source, args.match)
        if args.all:
            ids = list(datasets)
        if not ids:
            parser.error("name datasets, or use --match or --all")
        if unknown := [i for i in ids if i not in datasets]:
            parser.error(f"unknown datasets of {args.source}: {', '.join(unknown[:5])}")
        keys = [f"{args.source}:{i}" for i in dict.fromkeys(ids)]
        prints = precompute(keys, window)
        if args.compare:
            _compare([k for k in keys if prints[k].features], window)

    found = [prints[k] for k in keys if prints[k].features]
    if args.json:
        for fp in found:
            print(json.dumps(asdict(fp), ensure_ascii=False))
    else:
        _table(found, datasets)
    skipped = len(keys) - len(found)
    print(f"{len(found)} fingerprints over {window.start} to {window.end} per {window.step}"
          + (f"; {skipped} datasets skipped, not cached (--download fetches them, -v says why)" if skipped else "")
          + f"\nstored in {Store().dir / args.source}", file=sys.stderr)
    if not found:
        sys.exit(1)


RELATIONS_SHOWN = 25


def _compare(keys: list[str], window: Window) -> None:
    """Compares `keys` (all stored, if empty) with every stored fingerprint over the window; prints the strongest."""
    import sys  # noqa: PLC0415

    stored = [fp.key for fp in Store().all() if fp.window == window and fp.features]
    pairs = sorted({tuple(sorted((a, b))) for a in (keys or stored) for b in stored if a != b})
    if not pairs:
        print(f"nothing to compare: no other fingerprints over {window.start} to {window.end}", file=sys.stderr)
        return
    print(f"comparing {len(pairs)} pairs ...", file=sys.stderr)
    prints = precompute([], window, pairs)  # each pair once, stored under its first: the lag reads the other way
    found = [(a, b, prints[a].compared("remainder.lag").get(b)) for a, b in pairs]
    found = sorted((x for x in found if x[2] and x[2].get("r") is not None), key=lambda x: -abs(x[2]["r"]))
    names = _names({k for a, b, _ in found[:RELATIONS_SHOWN] for k in (a, b)})
    print(f"\n{'r':>5} {'lag':>5} {'r Δ':>5} {'lag Δ':>5}  series (lag > 0: the second follows the first)")
    for a, b, v in found[:RELATIONS_SHOWN]:
        change = "-" if v["r_change"] is None else f"{v['r_change']:.2f}"
        print(f"{v['r']:5.2f} {v['lag_h']:+4d}h {change:>5} {v['lag_change_h'] or 0:+4d}h  {names[a]}  ~  {names[b]}")
    print(f"{len(found)} of {len(pairs)} pairs related over {window.start} to {window.end}"
          + (f"; the strongest {RELATIONS_SHOWN} shown" if len(found) > RELATIONS_SHOWN else ""), file=sys.stderr)


def _names(keys: set[str]) -> dict[str, str]:
    """"<source> <group> / <label>" for each key, or the key where the source's datasets can't be listed."""
    names = {k: k for k in keys}
    for source in {k.partition(":")[0] for k in keys}:
        try:
            datasets = {d.id: d for d in sources.discover()[source].datasets()}
        except Exception:  # noqa: BLE001, S112 (a name is a nicety)
            continue
        for k in keys:
            d = datasets.get(k.partition(":")[2]) if k.startswith(source + ":") else None
            if d:
                names[k] = f"{source} {d.group} / {d.label}" if d.group else f"{source} {d.label}"
    return names


def _table(found: list[Fingerprint], datasets: dict[str, sources.Dataset]) -> None:
    def num(value: Value, digits: int = 3) -> str:
        return "-" if value is None else f"{value:.{digits}g}"

    def pct(value: Value) -> str:
        return "-" if value is None else f"{100 * value:.0f}%"

    head = (f"{'dataset':<42} {'cover':>5} {'mean':>9} {'std':>9} {'unit':<8} {'zero':>4} "
            f"{'day':>4} {'week':>4} {'year':>4} {'ac24h':>5} {'trend/y':>8} {'left':>4} {'memory':>6}")
    print(head)
    print("-" * len(head))
    for fp in found:
        d = datasets[fp.key.partition(":")[2]]
        s = fp.feature("summary") or {}
        c = fp.feature("cycle.strength") or {}
        a = fp.feature("autocorr") or {}
        decorrelation = (fp.feature("remainder.memory") or {}).get("decorrelation_h")
        memory = "-" if not fp.feature("remainder.memory") else ">2wk" if decorrelation is None else f"{decorrelation:.0f}h"
        name = f"{d.id} {d.label}"
        print(f"{name[:42]:<42} {pct(fp.feature('coverage')):>5} {num(s.get('mean')):>9} {num(s.get('std')):>9} "
              f"{d.unit[:8]:<8} {pct(fp.feature('zero_share')):>4} {pct(c.get('day')):>4} {pct(c.get('week')):>4} "
              f"{pct(c.get('year')):>4} {num(a.get('24h'), 2):>5} {num(fp.feature('trend')):>8} "
              f"{pct(fp.feature('remainder.share')):>4} {memory:>6}")


if __name__ == "__main__":
    main()
