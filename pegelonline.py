"""Reader for PEGELONLINE, the gauges of the German federal waterways (WSV).

About 800 stations on rivers, canals and the coast, most with a water level
(W), some with discharge (Q), water temperature (WT) and more, as raw data
straight from the gauges. No API key is needed; the licence is DL-DE Zero 2.0.

    https://www.pegelonline.wsv.de/webservices/rest-api/v2/stations.json?includeTimeseries=true
        -> [{"uuid", "number", "shortname", "km", "water": {...}, "timeseries": [...]}, ...]
    https://www.pegelonline.wsv.de/webservices/rest-api/v2/stations/{number}/{series}/measurements.json?start=P31D
        -> [{"timestamp": "2026-10-16T21:15:00+02:00", "value": 56.0}, ...]

PEGELONLINE keeps only about the last 31 days: older ranges come back empty.
A series is named "<station number>/<series>", e.g. "2730010/W" (Köln, water
level); the command line also takes the station's name, e.g. "KÖLN/W".

Times are returned as UTC `datetime64[ms]`. Date arguments without a UTC
offset are read as Berlin time, and the CSV output is in Berlin time.

Downloads go through the free-data cache (cache.py), source "de-pegelonline":

    station list                1 day
    a series' last 31 days     15 min

    python pegelonline.py --list rhein
    python pegelonline.py KÖLN/W BONN/W --resolution native --start 2026-10-01
"""

from __future__ import annotations

import argparse
import json
import sys
import time as clock
from datetime import datetime
from pathlib import Path
from urllib.parse import quote
from zoneinfo import ZoneInfo

import numpy as np

import sources
from cache import DAY, MINUTE, Cache, FetchError
from sources import Table, Time

BASE = "https://www.pegelonline.wsv.de/webservices/rest-api/v2"
STATION_PAGE = "https://www.pegelonline.wsv.de/gast/stammdaten?pegelnr={number}"
SOURCE = "de-pegelonline"  # the id in data/ and the cache folder
STATIONS_MAX_AGE = DAY
SERIES_MAX_AGE = 15 * MINUTE
KEPT = 31 * DAY  # how far back PEGELONLINE has data
TZ = ZoneInfo("Europe/Berlin")
RESOLUTIONS = ("native", "hour", "day")

# Series short name -> English name; others keep PEGELONLINE's own long name.
PARAMETERS = {
    "W": "Water level",
    "Q": "Discharge",
    "WT": "Water temperature",
    "LT": "Air temperature",
    "LF": "Conductivity",
    "O2": "Oxygen",
    "PH": "pH",
    "TR": "Turbidity",
    "WG": "Wind speed",
    "WR": "Wind direction",
    "GRU": "Groundwater level",
    "DFH": "Clearance height",
    "VA": "Flow velocity",
}


class PegelonlineError(RuntimeError):
    pass


def stations(cache: Path | None = None) -> list[dict]:
    """Every station with its time series, as PEGELONLINE describes them."""
    url = f"{BASE}/stations.json?includeTimeseries=true"
    return json.loads(Cache(SOURCE, cache).fetch(url, "stations.json", STATIONS_MAX_AGE))


def read(
    series: list[str] | tuple[str, ...],
    start: Time = None,
    end: Time = None,
    resolution: str = "hour",
    cache: Path | None = None,
) -> Table:
    """The series ("<station number>/<series>") between `start` and `end` (both inclusive).

    "hour" and "day" are means over each Berlin hour or day of the values
    there are; "native" is every value as measured (1 to 60 minutes apart).
    """
    if resolution not in RESOLUTIONS:
        raise ValueError(f"resolution must be one of {RESOLUTIONS}, got {resolution!r}")
    lo, hi = sources.epoch_ms(start, TZ), sources.epoch_ms(end, TZ)
    if lo is not None and hi is not None and lo > hi:
        raise ValueError(f"start {start} is after end {end}")
    store = Cache(SOURCE, cache)
    reachable = hi is None or hi >= _now() - (KEPT + DAY) * 1000  # skip what PEGELONLINE no longer has

    tables = []
    for name in series:
        time, values = _measurements(store, name) if reachable else (np.array([], np.int64), np.array([]))
        keep = np.ones(len(time), bool)
        if lo is not None:
            keep &= time >= lo
        if hi is not None:
            keep &= time <= hi
        tables.append(Table((name,), time[keep], values[keep].reshape(-1, 1)))
    combined = sources.combine(tables)
    time = combined.time.astype(np.int64)
    values = combined.values
    if resolution != "native":
        time, values = sources.aggregate(time, values, resolution, TZ)
    return Table(tuple(series), time.astype("datetime64[ms]"), values)


def _measurements(store: Cache, name: str) -> tuple[np.ndarray, np.ndarray]:
    number, slash, series = name.partition("/")
    if not (number and slash and series):
        raise ValueError(f"a series is named '<station number>/<series>', e.g. '2730010/W', not {name!r}")
    url = f"{BASE}/stations/{quote(number)}/{quote(series)}/measurements.json?start=P31D"
    try:
        rows = json.loads(store.fetch(url, f"{number}/{series}.json", SERIES_MAX_AGE))
    except FetchError as error:
        if error.status == 404:
            raise PegelonlineError(f"no series {series!r} at station {number!r}") from error
        raise
    time = np.array([_epoch(row["timestamp"]) for row in rows], np.int64)
    values = np.array([np.nan if row["value"] is None else row["value"] for row in rows], np.float64)
    order = np.argsort(time, kind="stable")
    return time[order], values[order]


def _epoch(stamp: str) -> int:
    return sources.epoch_ms(datetime.fromisoformat(stamp), TZ)


def _now() -> int:
    return int(clock.time() * 1000)


def resolve(spec: str, known: list[dict]) -> str:
    """"<station>/<series>" with the station given by number, name or uuid -> "<number>/<series>"."""
    station, slash, series = spec.partition("/")
    if not slash:
        series = "W"  # a bare station means its water level
    wanted = station.casefold()
    matches = [s for s in known if wanted in (s["number"].casefold(), s["shortname"].casefold(), s["uuid"])]
    if len(matches) != 1:
        raise ValueError(f"no station {station!r}" if not matches else f"station name {station!r} is ambiguous")
    return f"{matches[0]['number']}/{series}"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download PEGELONLINE gauge series to CSV.")
    parser.add_argument("series", nargs="*", help="<station>/<series>, e.g. KÖLN/W or 2730010/Q; see --list")
    parser.add_argument("--resolution", default="hour", choices=RESOLUTIONS)
    parser.add_argument("--start", help=f"ISO date or time, Berlin time unless an offset is given "
                        f"(default: {sources.CLI_DAYS} days before the end)")
    parser.add_argument("--end", help="ISO date or time, inclusive; a bare date means the whole day (default: now)")
    parser.add_argument("--out", help="CSV file (default: stdout)")
    parser.add_argument("--list", metavar="SEARCH", nargs="?", const="", help="show the stations, optionally only matching ones")
    args = parser.parse_args(argv)

    if args.list is not None or not args.series:
        words = (args.list or "").casefold().split()
        for s in stations():
            text = f"{s['number']} {s['shortname']} {s['water']['longname']}"
            if all(w in text.casefold() for w in words):
                kinds = " ".join(t["shortname"] for t in s["timeseries"])
                print(f"{s['number']:>10}  {s['shortname']:<28} {s['water']['longname']:<28} {kinds}")
        return
    try:
        known = stations()
        names = [resolve(spec, known) for spec in args.series]
        start, end = sources.span(args.start, args.end, TZ)
        table = read(names, start, end, args.resolution)
    except (ValueError, PegelonlineError) as error:
        parser.error(str(error))
    if args.out:
        with open(args.out, "w") as out:
            sources.write_csv(table, out, TZ)
    else:
        sources.write_csv(table, sys.stdout, TZ)


if __name__ == "__main__":
    main()
