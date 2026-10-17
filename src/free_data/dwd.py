"""Reader for the DWD's hourly weather observations (Deutscher Wetterdienst, CDC open data).

Hourly temperature, humidity, precipitation, wind, sunshine and air pressure
from the DWD's German stations. No API key is needed; the licence is CC BY 4.0
("Deutscher Wetterdienst").

The data are zipped text files, one per station and parameter group:

    https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/{group}/
        recent/{CODE}_Stundenwerte_Beschreibung_Stationen.txt   the stations
        recent/stundenwerte_{CODE}_{station}_akt.zip            about the last 500 days, updated daily
        historical/stundenwerte_{CODE}_{station}_{from}_{to}_hist.zip   quality-checked, to the end of last year

Inside each zip, produkt_*.txt has ";"-separated columns, MESS_DATUM as
YYYYMMDDHH in UTC and -999 for missing values. Where the historical and the
recent file overlap, the quality-checked historical values are used.

A series is named "<station id>/<variable>", e.g. "00044/TT_TU" (air
temperature at Großenkneten); see VARIABLES. Only stations still reporting
(with a recent file) are offered; their history goes back as far as the DWD
has it.

"day" is the mean over each Berlin day (sums for precipitation and sunshine,
the vector mean for wind direction), not the DWD's own climate day.

Times are returned as UTC `datetime64[ms]`. Date arguments without a UTC
offset are read as Berlin time, and the CSV output is in Berlin time.

Downloads go through the free-data cache (cache.py), source "de-dwd-opendata":

    station and file lists                 1 day
    recent files                           6 hours
    historical files                       forever (a new version has a new name)

    python -m free_data.dwd --list potsdam
    python -m free_data.dwd 03987/TT_TU 03987/R1 --start 2025-07-01 --end 2025-07-31 --resolution day
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import time as clock
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from free_data import sources
from free_data.cache import DAY, HOUR, Cache, FetchError
from free_data.sources import Table, Time

BASE = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly"
SOURCE = "de-dwd-opendata"  # the id in data/ and the cache folder
LIST_MAX_AGE = DAY
RECENT_MAX_AGE = 6 * HOUR
RECENT_SPAN = 600 * DAY  # the recent files never reach further back than this
DOWNLOADS = 4
TZ = ZoneInfo("Europe/Berlin")
RESOLUTIONS = ("hour", "day")


@dataclass(frozen=True)
class Variable:
    group: str  # folder on the server
    code: str  # in the file names
    name: str
    unit: str
    daily: str  # "mean", "sum" or "direction"


VARIABLES = {
    "TT_TU": Variable("air_temperature", "TU", "Air temperature", "°C", "mean"),
    "RF_TU": Variable("air_temperature", "TU", "Relative humidity", "%", "mean"),
    "R1": Variable("precipitation", "RR", "Precipitation", "mm", "sum"),
    "F": Variable("wind", "FF", "Wind speed", "m/s", "mean"),
    "D": Variable("wind", "FF", "Wind direction", "°", "direction"),
    "SD_SO": Variable("sun", "SD", "Sunshine", "min", "sum"),
    "P": Variable("pressure", "P0", "Air pressure at sea level", "hPa", "mean"),
}


class DwdError(RuntimeError):
    pass


def stations(group: str, cache: Path | None = None) -> list[dict]:
    """The stations of one parameter group, from the DWD's fixed-width station list."""
    code = next(v.code for v in VARIABLES.values() if v.group == group)
    url = f"{BASE}/{group}/recent/{code}_Stundenwerte_Beschreibung_Stationen.txt"
    text = Cache(SOURCE, cache).fetch(url, f"{group}/stations.txt", LIST_MAX_AGE).decode("latin-1")
    lines = text.splitlines()
    rule = next(i for i, line in enumerate(lines) if line.startswith("---"))
    found = []
    for line in lines[rule + 1:]:
        # The columns don't line up with the header: six numbers, then name,
        # state and (not always) access, separated by runs of spaces.
        cells = line.split(maxsplit=6)
        if len(cells) < 7:
            continue
        rest = re.split(r"\s{2,}", cells[6].strip())
        found.append({"id": cells[0], "from": cells[1], "to": cells[2], "height": float(cells[3]),
                      "lat": float(cells[4]), "lon": float(cells[5]), "name": rest[0],
                      "state": rest[1] if len(rest) > 1 else ""})
    return found


def read(
    series: list[str] | tuple[str, ...],
    start: Time = None,
    end: Time = None,
    resolution: str = "hour",
    cache: Path | None = None,
) -> Table:
    """The series ("<station id>/<variable>") between `start` and `end` (both inclusive)."""
    if resolution not in RESOLUTIONS:
        raise ValueError(f"resolution must be one of {RESOLUTIONS}, got {resolution!r}")
    lo, hi = sources.epoch_ms(start, TZ), sources.epoch_ms(end, TZ)
    if lo is not None and hi is not None and lo > hi:
        raise ValueError(f"start {start} is after end {end}")
    store = Cache(SOURCE, cache)

    wanted: dict[tuple[str, str], list[str]] = {}  # (station, group) -> variables, one download each
    for name in series:
        station, variable = _split(name)
        wanted.setdefault((station, VARIABLES[variable].group), []).append(variable)
    with ThreadPoolExecutor(DOWNLOADS) as pool:
        loaded = dict(zip(wanted, pool.map(lambda key: _load(store, *key, wanted[key], lo, hi), wanted), strict=True))

    columns = []
    for name in series:
        station, variable = _split(name)
        time, values = loaded[station, VARIABLES[variable].group][variable]
        keep = np.ones(len(time), bool)
        if lo is not None:
            keep &= time >= lo
        if hi is not None:
            keep &= time <= hi
        time, values = time[keep], values[keep]
        if resolution == "day":
            time, values = _daily(time, values, VARIABLES[variable].daily)
        columns.append(Table((name,), time, values.reshape(-1, 1)))
    combined = sources.combine(columns)
    return Table(tuple(series), combined.time.astype("datetime64[ms]"), combined.values)


def _split(name: str) -> tuple[str, str]:
    station, slash, variable = name.partition("/")
    if not (station and slash and variable in VARIABLES):
        raise ValueError(f"a series is named '<station id>/<variable>' with a variable from {list(VARIABLES)}, "
                         f"not {name!r}")
    return station.zfill(5), variable


def _daily(time: np.ndarray, values: np.ndarray, how: str) -> tuple[np.ndarray, np.ndarray]:
    if how != "direction":
        days, result = sources.aggregate(time, values.reshape(-1, 1), "day", TZ, how)
        return days, result[:, 0]
    angle = np.radians(values)
    days, parts = sources.aggregate(time, np.column_stack([np.sin(angle), np.cos(angle)]), "day", TZ)
    return days, np.degrees(np.arctan2(parts[:, 0], parts[:, 1])) % 360


def _load(
    store: Cache, station: str, group: str, variables: list[str], lo: int | None, hi: int | None
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """The variables of one station and group, historical and recent files joined."""
    code = VARIABLES[variables[0]].code
    files = []  # historical first: where they overlap the recent file, they win
    hist_end = None
    for name, first, last in _historical(store, group, code, station):
        hist_end = max(hist_end or last, last)
        if (hi is None or first <= hi) and (lo is None or last + DAY * 1000 > lo):
            files.append((f"{group}/historical/{name}", f"{BASE}/{group}/historical/{name}", None))
    if (hi is None or hist_end is None or hi > hist_end) and (hi is None or hi > _now() - RECENT_SPAN * 1000):
        name = f"stundenwerte_{code}_{station}_akt.zip"
        files.append((f"{group}/recent/{station}.zip", f"{BASE}/{group}/recent/{name}", RECENT_MAX_AGE))

    parts = []
    for key, url, max_age in files:
        try:
            parts.append(_parse(store.fetch(url, key, max_age), variables))
        except FetchError as error:
            if error.status != 404:
                raise
            if max_age is None:  # a historical file the listing named: worth saying
                raise DwdError(f"{url} is gone; the historical file list may be stale") from error
            # no recent file: the station has stopped reporting this group
    result = {}
    for variable in variables:
        time = np.concatenate([p[0] for p in parts] or [np.array([], np.int64)])
        values = np.concatenate([p[1][variable] for p in parts] or [np.array([])])
        time, first = np.unique(time, return_index=True)  # the first file listed wins
        result[variable] = (time, values[first])
    return result


def _historical(store: Cache, group: str, code: str, station: str) -> list[tuple[str, int, int]]:
    """The station's historical files: name, first and last day (epoch ms, UTC)."""
    listing = store.fetch(f"{BASE}/{group}/historical/", f"{group}/historical.html", LIST_MAX_AGE).decode("latin-1")
    pattern = re.compile(rf"stundenwerte_{code}_{station}_(\d{{8}})_(\d{{8}})_hist\.zip")
    found = {m.group(0): (m.group(1), m.group(2)) for m in pattern.finditer(listing)}
    return sorted((name, _day_ms(a), _day_ms(b)) for name, (a, b) in found.items())


def _parse(body: bytes, variables: list[str]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with zipfile.ZipFile(io.BytesIO(body)) as archive:
        member = next((n for n in archive.namelist() if n.startswith("produkt")), None)
        if member is None:
            raise DwdError("no produkt_*.txt in the DWD zip file")
        text = archive.read(member).decode("latin-1")
    lines = text.splitlines()
    header = [cell.strip() for cell in lines[0].split(";")]
    at = {name: header.index(name) for name in ("MESS_DATUM", *variables) if name in header}
    missing = [v for v in variables if v not in at]
    if missing:
        raise DwdError(f"the DWD file has no column {missing}")
    rows = [line.split(";") for line in lines[1:] if line.strip()]
    stamps = [row[at["MESS_DATUM"]].strip()[:10] for row in rows]  # YYYYMMDDHH, UTC
    time = (np.array([f"{s[:4]}-{s[4:6]}-{s[6:8]}T{s[8:10]}" for s in stamps], "datetime64[ms]")
            .astype(np.int64))
    columns = {}
    for variable in variables:
        values = np.array([float(row[at[variable]]) for row in rows])
        values[values <= -999] = np.nan
        columns[variable] = values
    return time, columns


def _day_ms(yyyymmdd: str) -> int:
    return int(np.datetime64(f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:]}", "ms").astype(np.int64))


def _now() -> int:
    return int(clock.time() * 1000)


def reporting(group: str, cache: Path | None = None) -> list[dict]:
    """The stations of one parameter group that still report: those with a recent file.

    The station list's own end dates lag behind for some groups (sunshine).
    """
    store = Cache(SOURCE, cache)
    code = next(v.code for v in VARIABLES.values() if v.group == group)
    listing = store.fetch(f"{BASE}/{group}/recent/", f"{group}/recent.html", LIST_MAX_AGE).decode("latin-1")
    ids = set(re.findall(rf"stundenwerte_{code}_(\d{{5}})_akt\.zip", listing))
    return [row for row in stations(group, cache) if row["id"] in ids]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download DWD hourly weather observations to CSV.")
    parser.add_argument("series", nargs="*", help="<station id>/<variable>, e.g. 03987/TT_TU; see --list")
    parser.add_argument("--resolution", default="hour", choices=RESOLUTIONS)
    parser.add_argument("--start", help=f"ISO date or time, Berlin time unless an offset is given "
                        f"(default: {sources.CLI_DAYS} days before the end)")
    parser.add_argument("--end", help="ISO date or time, inclusive; a bare date means the whole day (default: now)")
    parser.add_argument("--out", help="CSV file (default: stdout)")
    parser.add_argument("--list", metavar="SEARCH", nargs="?", const="",
                        help="show the reporting stations and their variables, optionally only matching ones")
    args = parser.parse_args(argv)

    if args.list is not None or not args.series:
        words = (args.list or "").casefold().split()
        offered: dict[str, tuple[dict, list[str]]] = {}
        for group in dict.fromkeys(v.group for v in VARIABLES.values()):
            for row in reporting(group):
                entry = offered.setdefault(row["id"], (row, []))
                entry[1].extend(k for k, v in VARIABLES.items() if v.group == group)
        for station, (row, variables) in sorted(offered.items()):
            if all(w in f"{station} {row['name']} {row['state']}".casefold() for w in words):
                print(f"{station}  {row['name']:<36} {row['state']:<24} {' '.join(variables)}")
        return
    try:
        start, end = sources.span(args.start, args.end, TZ)
        table = read(args.series, start, end, args.resolution)
    except (ValueError, DwdError) as error:
        parser.error(str(error))
    if args.out:
        with open(args.out, "w") as out:
            sources.write_csv(table, out, TZ)
    else:
        sources.write_csv(table, sys.stdout, TZ)


if __name__ == "__main__":
    main()
