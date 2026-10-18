"""Reader for the Umweltbundesamt's air quality data (UBA air data API v4).

Hourly PM10, PM2.5, NO2, O3, SO2 and CO from the measuring stations of the
German states and the UBA's own network, since 2016. No API key is needed;
the data are free to use with the source named ("Umweltbundesamt").

    https://luftdaten.umweltbundesamt.de/api/air-data/v4/meta/json?use=measure&date_from=...&date_to=...
        -> {"stations": {id: [...]}, "components": {...}, "scopes": {...},
            "limits": {...: [scope id, component id, station id, first, last]}, ...}
    https://luftdaten.umweltbundesamt.de/api/air-data/v4/measures/json?station=21&component=5&scope=2
            &date_from=2024-01-01&time_from=1&date_to=2024-12-31&time_to=24
        -> {"data": {"21": {"2024-01-01 00:00:00": [5, 2, 11, "2024-01-01 01:00:00", "0"], ...}}}

A component is measured as one or more scopes (averaging periods). A series
uses the first of SCOPES a station has for it: the hourly mean, else the
daily mean (some particulate matter samplers), else the 8-hour running mean
(carbon monoxide at most stations). The station list in the meta data holds
the stations still running; of those, a component is offered while its last
value is less than ACTIVE old.

A series is named "<station code>/<component>", e.g. "DEBB021/NO2" (Potsdam-
Zentrum, nitrogen dioxide); PM2.5 is "PM2", as the UBA writes it.

Values are labelled by the start of their interval. The API says its times
are in CET or CEST, but every day has 24 hours, across both clock changes:
they are CET (UTC+1) all year, as is usual for air quality data. A daily
mean is labelled with the start of its CET day.

Data of the current year are preliminary until June of the next. A year is
downloaded at once:

    meta data (stations, series)       1 day
    the current year                   1 hour
    last year, until July              1 day    (cached as <year>-preliminary.json)
    older years                        forever

Times are returned as UTC `datetime64[ms]`. Date arguments without a UTC
offset are read as Berlin time, and the CSV output is in Berlin time.

    python -m free_data.uba --list potsdam
    python -m free_data.uba DEBB021/NO2 DEBB021/PM10 --start 2025-01-01 --end 2025-01-31
"""

from __future__ import annotations

import argparse
import json
import sys
import time as clock
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from free_data import sources
from free_data.cache import DAY, HOUR, Cache, FetchError, Pool
from free_data.sources import Table, Time

BASE = "https://luftdaten.umweltbundesamt.de/api/air-data/v4"
SOURCE = "de-uba-luftdaten"  # the id in data/ and the cache folder
META_MAX_AGE = DAY
CURRENT_MAX_AGE = HOUR
PRELIMINARY_MAX_AGE = DAY
ACTIVE = 30 * DAY  # a series whose last value is older than this no longer reports
DOWNLOADS = 4
CET_MS = 3_600_000  # the API's times are UTC+1 all year
TZ = ZoneInfo("Europe/Berlin")
RESOLUTIONS = ("native", "day")
SCOPES = ("1SMW", "1TMW", "8SMW")  # hourly mean, daily mean, 8-hour running mean, in order of preference
DAILY = {"1TMW"}

# Component code -> English name; the API calls PM10 and PM2.5 both "Particulate matter".
COMPONENTS = {
    "PM10": "Particulate matter PM10",
    "PM2": "Particulate matter PM2.5",
    "NO2": "Nitrogen dioxide",
    "O3": "Ozone",
    "SO2": "Sulphur dioxide",
    "CO": "Carbon monoxide",
}


class UbaError(RuntimeError):
    pass


@dataclass(frozen=True)
class Series:
    """One component at one station, as the meta data describe it."""

    name: str  # "<station code>/<component>"
    station_id: str  # the API's number, "21"
    station: str  # the station code, "DEBB021"
    station_name: str
    city: str
    network: str  # "BB" for Brandenburg's network, "UB" for the UBA's own
    network_name: str
    setting: str  # urban, suburban, rural
    type: str  # background, traffic, industry
    lat: float | None
    lon: float | None
    component: str  # "NO2"
    component_id: str
    unit: str  # as the API writes it, "µg/m³"
    scope: str  # "1SMW"
    scope_id: str
    scope_name: str  # "One hour average"
    first: int  # epoch ms, UTC: the first and the last value
    last: int

    @property
    def reporting(self) -> bool:
        return self.last >= _now() - ACTIVE * 1000


def meta(cache: Path | None = None) -> dict:
    """The API's meta data for measurements: stations, components, scopes and each series' first and last value."""
    today = datetime.fromtimestamp(_now() / 1000, TZ).date()
    span = f"date_from={today - timedelta(days=ACTIVE / DAY):%Y-%m-%d}&date_to={today:%Y-%m-%d}"
    url = f"{BASE}/meta/json?use=measure&lang=en&{span}"
    return json.loads(Cache(SOURCE, cache).fetch(url, "meta.json", META_MAX_AGE))


def catalog(cache: Path | None = None) -> list[Series]:
    """Every series of the running stations, one scope per station and component (see SCOPES)."""
    found = meta(cache)
    components = {key: row for key, row in found["components"].items() if row[1] in COMPONENTS}
    scopes = {key: row for key, row in found["scopes"].items() if row[1] in SCOPES}
    best: dict[tuple[str, str], Series] = {}
    for limit in found["limits"].values():
        if len(limit) != 5:  # the span of a scope and component over all stations
            continue
        scope_id, component_id, station_id, first, last = limit
        station = found["stations"].get(station_id)
        if station is None or component_id not in components or scope_id not in scopes:
            continue
        component, scope = components[component_id], scopes[scope_id]
        series = Series(
            name=f"{station[1]}/{component[1]}", station_id=station_id, station=station[1],
            station_name=station[2], city=station[3], network=station[12], network_name=station[13],
            setting=station[15], type=station[16], lat=_float(station[8]), lon=_float(station[7]),
            component=component[1], component_id=component_id, unit=component[3],
            scope=scope[1], scope_id=scope_id, scope_name=scope[5], first=_epoch(first), last=_epoch(last),
        )
        known = best.get((station_id, component_id))
        if known is None or SCOPES.index(series.scope) < SCOPES.index(known.scope):
            best[station_id, component_id] = series
    return sorted(best.values(), key=lambda s: (s.station, list(COMPONENTS).index(s.component)))


def read(
    series: list[str] | tuple[str, ...],
    start: Time = None,
    end: Time = None,
    resolution: str = "native",
    cache: Path | None = None,
) -> Table:
    """The series ("<station code>/<component>") between `start` and `end` (both inclusive).

    "native" is every value as the UBA has it (hourly, or daily for a daily
    mean); "day" is the mean over each Berlin day of the values there are.
    """
    if resolution not in RESOLUTIONS:
        raise ValueError(f"resolution must be one of {RESOLUTIONS}, got {resolution!r}")
    lo, hi = sources.epoch_ms(start, TZ), sources.epoch_ms(end, TZ)
    if lo is not None and hi is not None and lo > hi:
        raise ValueError(f"start {start} is after end {end}")
    store = Cache(SOURCE, cache)
    known = {s.name.casefold(): s for s in catalog(cache)}
    wanted = [_find(known, name) for name in series]

    downloads = list(dict.fromkeys((s, year) for s in wanted for year in _years(s, lo, hi)))
    with Pool(DOWNLOADS) as pool:
        loaded = dict(zip(downloads, pool.map(lambda job: _year(store, *job), downloads), strict=True))

    columns = []
    for name, s in zip(series, wanted, strict=True):
        parts = [loaded[s, year] for year in _years(s, lo, hi)]
        time = np.concatenate([p[0] for p in parts] or [np.array([], np.int64)])
        values = np.concatenate([p[1] for p in parts] or [np.array([])])
        keep = np.ones(len(time), bool)
        if lo is not None:
            keep &= time >= lo
        if hi is not None:
            keep &= time <= hi
        columns.append(Table((name,), time[keep], values[keep].reshape(-1, 1)))
    combined = sources.combine(columns)
    time, values = combined.time.astype(np.int64), combined.values
    if resolution == "day":
        time, values = sources.aggregate(time, values, "day", TZ)
    return Table(tuple(series), time.astype("datetime64[ms]"), values)


def _find(known: dict[str, Series], name: str) -> Series:
    station, slash, component = name.partition("/")
    if not (station and slash and component):
        raise ValueError(f"a series is named '<station code>/<component>', e.g. 'DEBB021/NO2', not {name!r}")
    if name.casefold() not in known:
        raise UbaError(f"no series {component!r} at station {station!r}; see --list")
    return known[name.casefold()]


def _years(s: Series, lo: int | None, hi: int | None) -> range:
    """The CET years with values of `s` between `lo` and `hi`."""
    first = _cet_year(max(s.first, lo) if lo is not None else s.first)
    last = _cet_year(min(s.last, hi) if hi is not None else s.last)
    return range(first, last + 1)


def _year(store: Cache, s: Series, year: int) -> tuple[np.ndarray, np.ndarray]:
    """One series' values in one CET year: epoch ms (UTC) and values, NaN for gaps."""
    current = _cet_year(_now())
    if year >= current:
        suffix, max_age = "", CURRENT_MAX_AGE
    elif year == current - 1 and _cet_date(_now()).month < 7:
        suffix, max_age = "-preliminary", PRELIMINARY_MAX_AGE  # a new key once final, fetched afresh
    else:
        suffix, max_age = "", None
    url = (f"{BASE}/measures/json?station={s.station_id}&component={s.component_id}&scope={s.scope_id}"
           f"&date_from={year}-01-01&time_from=1&date_to={year}-12-31&time_to=24")
    try:
        body = json.loads(store.fetch(url, f"{s.station}/{s.component}/{s.scope}/{year}{suffix}.json", max_age))
    except FetchError as error:
        if error.status in (400, 404, 409):  # what the API answers for a station or component it doesn't know
            raise UbaError(f"the UBA has no {s.component} at {s.station} in {year}") from error
        raise
    rows = (body.get("data") or {}).get(s.station_id, {})
    daily = s.scope in DAILY
    stamps = sorted(rows)
    time = np.array([_epoch(f"{stamp[:10]} 00:00:00" if daily else stamp) for stamp in stamps], np.int64)
    values = np.array([np.nan if rows[stamp][2] is None else rows[stamp][2] for stamp in stamps], np.float64)
    return time, values


def _epoch(stamp: str) -> int:
    """"2024-01-01 00:00:00" in CET -> epoch ms, UTC."""
    return int(np.datetime64(stamp.replace(" ", "T"), "ms").astype(np.int64)) - CET_MS


def _cet_date(ms: int) -> date:
    return np.datetime64(ms + CET_MS, "ms").astype("datetime64[D]").item()


def _cet_year(ms: int) -> int:
    return _cet_date(ms).year


def _float(text: str | None) -> float | None:
    return float(text) if text not in (None, "") else None


def _now() -> int:
    return int(clock.time() * 1000)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download UBA air quality measurements to CSV.")
    parser.add_argument("series", nargs="*", help="<station code>/<component>, e.g. DEBB021/NO2; see --list")
    parser.add_argument("--resolution", default="native", choices=RESOLUTIONS)
    parser.add_argument("--start", help=f"ISO date or time, Berlin time unless an offset is given "
                        f"(default: {sources.CLI_DAYS} days before the end)")
    parser.add_argument("--end", help="ISO date or time, inclusive; a bare date means the whole day (default: now)")
    parser.add_argument("--out", help="CSV file (default: stdout)")
    parser.add_argument("--list", metavar="SEARCH", nargs="?", const="",
                        help="show the stations and their components, optionally only matching ones")
    args = parser.parse_args(argv)

    if args.list is not None or not args.series:
        words = (args.list or "").casefold().split()
        offered: dict[str, list[Series]] = {}
        for s in catalog():
            if s.reporting:
                offered.setdefault(s.station, []).append(s)
        for station, rows in offered.items():
            first = rows[0]
            text = f"{station} {first.station_name} {first.city} {first.network_name} {first.type}"
            if all(w in text.casefold() for w in words):
                print(f"{station}  {first.station_name:<40} {first.network_name:<24} {first.setting:<9} "
                      f"{first.type:<11} {' '.join(r.component for r in rows)}")
        return
    try:
        start, end = sources.span(args.start, args.end, TZ)
        table = read(args.series, start, end, args.resolution)
    except (ValueError, UbaError) as error:
        parser.error(str(error))
    if args.out:
        with open(args.out, "w") as out:
            sources.write_csv(table, out, TZ)
    else:
        sources.write_csv(table, sys.stdout, TZ)


if __name__ == "__main__":
    main()
