"""Plugin interface for the free data sources.

A data source is described by data/<id>.json (id, link, description). A
plugin makes it readable: a module in plugins/ with a module-level `PLUGIN`,
an instance of `Plugin` whose `id` matches the JSON file.

    class MySource(Plugin):
        id = "fr-rte-eco2mix"
        title = "RTE eCO2mix"
        timezone = "Europe/Paris"
        options = (Option("resolution", "Resolution", ("15min", "hour"), "hour"),)

        def datasets(self) -> list[Dataset]: ...
        def read(self, datasets, start, end, options) -> Table: ...

    PLUGIN = MySource()

`discover()` imports every module in plugins/ and `catalog()` lists all
sources in data/, with or without a plugin. browser.py serves both.
"""

from __future__ import annotations

import importlib
import json
import pkgutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, tzinfo
from functools import cache
from pathlib import Path
from typing import TextIO
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
DATA = HERE / "data"
GROUP_SEPARATOR = " / "
CLI_DAYS = 30  # default span of a reader's command line
COUNTRIES = {  # source id prefix -> top level of the browser's tree
    "de": "Germany",
    "fr": "France",
    "eu": "European Union",
    "uk": "United Kingdom",
    "us": "United States",
}


Time = str | datetime | np.datetime64 | None


@dataclass(frozen=True)
class Dataset:
    """One series a plugin can read."""

    id: str  # unique within the plugin
    label: str
    group: str = ""  # where it sits in the browser's tree; " / " nests, e.g. "Generation / Renewable"
    unit: str = ""
    description: str = ""  # what the series measures, for people choosing it
    link: str = ""  # the series on the source's website; empty means the source's link in data/


@dataclass(frozen=True)
class Option:
    """A choice that applies to every dataset read from a plugin, e.g. a region."""

    name: str
    label: str
    choices: tuple[str, ...]
    default: str
    help: str = ""


@dataclass(frozen=True)
class Table:
    columns: tuple[str, ...]  # dataset ids, in the order they were asked for
    time: np.ndarray  # datetime64[ms], UTC, strictly increasing
    values: np.ndarray  # float64, shape (len(time), len(columns)), NaN for gaps


class Plugin(ABC):
    id: str  # matches data/<id>.json
    title: str
    timezone: str = "UTC"  # how the source's own times are shown, and how bare dates are read
    options: tuple[Option, ...] = ()

    @abstractmethod
    def datasets(self) -> list[Dataset]:
        """Everything this source offers."""

    @abstractmethod
    def read(
        self, datasets: list[str], start: datetime | None, end: datetime | None, options: dict[str, str]
    ) -> Table:
        """The datasets between `start` and `end` (inclusive, timezone-aware).

        `options` holds a valid value for every declared option.
        """

    def resolve(self, given: dict[str, str]) -> dict[str, str]:
        """The declared options filled in with defaults and checked."""
        resolved = {}
        for option in self.options:
            value = given.get(option.name, option.default)
            if value not in option.choices:
                raise ValueError(f"{self.id}: {option.name} must be one of {option.choices}, got {value!r}")
            resolved[option.name] = value
        return resolved


@cache
def discover() -> dict[str, Plugin]:
    """Every plugin in plugins/, by source id."""
    import plugins

    found: dict[str, Plugin] = {}
    for module in pkgutil.iter_modules(plugins.__path__):
        plugin = getattr(importlib.import_module(f"plugins.{module.name}"), "PLUGIN", None)
        if not isinstance(plugin, Plugin):
            continue
        if plugin.id in found:
            raise RuntimeError(f"two plugins for source {plugin.id!r}")
        found[plugin.id] = plugin
    return found


def catalog() -> list[dict]:
    """All sources in data/, plus plugins without a JSON file, plugged-in first."""
    plugins = discover()
    entries = {}
    for path in sorted(DATA.glob("*.json")):
        entry = json.loads(path.read_text())
        entries[entry["data-source-id"]] = entry
    for key in plugins:
        entries.setdefault(key, {"data-source-id": key, "link": "", "description": ""})
    rows = [
        {
            "id": key,
            "title": plugins[key].title if key in plugins else _title(entry, key),
            "link": entry.get("link", ""),
            "description": entry.get("description", ""),
            "plugin": key in plugins,
            "country": COUNTRIES.get(key.split("-", 1)[0], "Other"),
        }
        for key, entry in entries.items()
    ]
    return sorted(rows, key=lambda row: (not row["plugin"], row["id"]))


def _title(entry: dict, key: str) -> str:
    """A name for a source without a plugin: its description up to the first colon."""
    head, colon, _ = entry.get("description", "").partition(":")
    return head.strip() if colon and 0 < len(head) <= 60 else key


def combine(tables: list[Table]) -> Table:
    """Tables side by side on the union of their times, NaN where one has no value."""
    times = [table.time for table in tables]
    time = np.unique(np.concatenate(times)) if times else np.array([], "datetime64[ms]")
    columns = [column for table in tables for column in table.columns]
    values = np.full((len(time), len(columns)), np.nan)
    at = 0
    for table in tables:
        width = len(table.columns)
        values[np.searchsorted(time, table.time), at : at + width] = table.values
        at += width
    return Table(tuple(columns), time, values)


def merge(tables: list[tuple[str, Table]]) -> Table:
    """Tables from several sources side by side, columns named "<source>:<dataset>"."""
    combined = combine([table for _, table in tables])
    columns = tuple(f"{source}:{column}" for source, table in tables for column in table.columns)
    return Table(columns, combined.time, combined.values)


def moment(text: str | None, zone: str | tzinfo, end: bool) -> datetime | None:
    """A date or date-time typed by a person; without an offset it is in `zone`.

    A bare end date means the whole day. Raises ValueError for anything else.
    """
    if not text:
        return None
    try:
        value = datetime.fromisoformat(text)
    except ValueError as error:
        raise ValueError(f"not a date: {text!r}") from error
    if len(text) == 10 and end:
        value = datetime.combine(value.date(), datetime.max.time())
    return value if value.tzinfo else value.replace(tzinfo=_zone(zone))


def span(start: str | None, end: str | None, zone: str | tzinfo, days: int = CLI_DAYS) -> tuple[datetime, datetime]:
    """A reader's --start and --end: the end defaults to now, the start to `days` before the end."""
    last = moment(end, zone, end=True) or datetime.now(_zone(zone))
    first = moment(start, zone, end=False) or last - timedelta(days=days)
    if first > last:
        raise ValueError(f"start {first} is after end {last}")
    return first, last


def epoch_ms(value: Time, zone: str | tzinfo) -> int | None:
    """Milliseconds since 1970 UTC; strings and datetimes without an offset are in `zone`."""
    if value is None:
        return None
    if isinstance(value, np.datetime64):
        return int(value.astype("datetime64[ms]").astype(np.int64))  # numpy times are UTC
    when = datetime.fromisoformat(value) if isinstance(value, str) else value
    if when.tzinfo is None:
        when = when.replace(tzinfo=_zone(zone))
    return int(when.astimezone(timezone.utc).timestamp() * 1000)


def write_csv(table: Table, out: TextIO, zone: str | tzinfo) -> None:
    """The table as CSV with ISO timestamps in `zone`."""
    tz = _zone(zone)
    out.write(",".join(("timestamp", *table.columns)) + "\n")
    for t, row in zip(table.time.astype(np.int64), table.values, strict=True):
        stamp = datetime.fromtimestamp(t / 1000, tz).isoformat()
        out.write(",".join((stamp, *("" if np.isnan(v) else repr(float(v)) for v in row))) + "\n")


def _zone(zone: str | tzinfo) -> tzinfo:
    return ZoneInfo(zone) if isinstance(zone, str) else zone


def aggregate(
    time: np.ndarray, values: np.ndarray, resolution: str, zone: str | tzinfo, how: str = "mean"
) -> tuple[np.ndarray, np.ndarray]:
    """Means (or sums) per hour or per day in `zone`, labelled with its start; NaN where it has no value.

    `time` is epoch ms (UTC), sorted; `values` has one column per series.
    Hours are UTC hours, which are local hours in any zone a whole number
    of hours off UTC.
    """
    if resolution not in ("hour", "day"):
        raise ValueError(f"aggregates are per hour or day, not {resolution!r}")
    if how not in ("mean", "sum"):
        raise ValueError(f"how is 'mean' or 'sum', not {how!r}")
    if not len(time):
        return time, values
    if resolution == "hour":
        bins = time - time % 3_600_000
    else:
        tz = _zone(zone)
        first, last = (datetime.fromtimestamp(t / 1000, tz).date() for t in (time[0], time[-1]))
        days = np.arange(np.datetime64(first, "D"), np.datetime64(last, "D") + np.timedelta64(2, "D"))
        edges = np.array([epoch_ms(datetime.fromisoformat(str(d)), tz) for d in days], np.int64)
        bins = edges[np.searchsorted(edges, time, side="right") - 1]
    labels, group = np.unique(bins, return_inverse=True)
    present = ~np.isnan(values)
    sums = np.zeros((len(labels), values.shape[1]))
    counts = np.zeros((len(labels), values.shape[1]))
    np.add.at(sums, group, np.where(present, values, 0.0))
    np.add.at(counts, group, present)
    if how == "sum":
        return labels, np.where(counts > 0, sums, np.nan)
    with np.errstate(invalid="ignore"):
        return labels, np.where(counts > 0, sums / counts, np.nan)
