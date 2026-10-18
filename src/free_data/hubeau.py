"""Reader for Hub'eau hydrometry, the river gauges of France (Schapi, via eaufrance).

About 4,000 stations in service measure the water level (H), the flow (Q)
or both. No API key is needed; the licence is Etalab Licence Ouverte 2.0.

    https://hubeau.eaufrance.fr/api/v2/hydrometrie/referentiel/stations?en_service=true&size=500&page=N
    https://hubeau.eaufrance.fr/api/v2/hydrometrie/observations_tr?code_entite={station}&grandeur_hydro=H|Q
        -> the last 30 days as measured, every few minutes, not yet validated
    https://hubeau.eaufrance.fr/api/v2/hydrometrie/obs_elab?code_entite={station}&grandeur_hydro_elab=QmnJ
        -> daily mean flows, the station's whole history, mostly validated

A series is named "<station code>/H" or "<station code>/Q", e.g.
"F700000103/Q" (the Seine at Paris-Austerlitz). Hub'eau gives levels in mm
and flows in l/s; they are returned in cm and m³/s, the units PEGELONLINE
uses, so French and German gauges can share an axis.

"native" and "hour" cover only the last 30 days. "day" has the daily mean
flows of the whole history, continued with daily means of the real-time data
after the last published day; levels have no history. Hub'eau's days are
labelled at Paris midnight.

Times are returned as UTC `datetime64[ms]`. Date arguments without a UTC
offset are read as Paris time, and the CSV output is in Paris time.

Downloads go through the free-data cache (cache.py), source
"fr-hubeau-hydrometrie":

    station list                   1 day
    real-time data                15 min
    daily flows, this decade       1 day
    daily flows, older decades    30 days  (validation revises old values)

    python -m free_data.hubeau --list seine paris
    python -m free_data.hubeau F700000103/Q --start 2016-05-01 --end 2016-06-30 --resolution day
"""

from __future__ import annotations

import argparse
import json
import sys
import time as clock
import zlib
from datetime import UTC, date, datetime
from pathlib import Path
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

import numpy as np

from free_data import sources
from free_data.cache import DAY, MINUTE, Cache, Pool
from free_data.sources import Table, Time

BASE = "https://hubeau.eaufrance.fr/api/v2/hydrometrie"
SOURCE = "fr-hubeau-hydrometrie"  # the id in data/ and the cache folder
STATIONS_MAX_AGE = DAY
REALTIME_MAX_AGE = 15 * MINUTE
DAILY_MAX_AGE = DAY
OLD_DAILY_MAX_AGE = 30 * DAY
REALTIME_SPAN = 31 * DAY  # how far back the real-time data goes, with a day to spare
FIRST_YEAR = 1900
PAGE = 500  # stations per request; larger pages often time out
DOWNLOADS = 4
TZ = ZoneInfo("Europe/Paris")
RESOLUTIONS = ("native", "hour", "day")
STATION_FIELDS = ("code_station", "libelle_station", "type_station", "libelle_cours_eau", "libelle_commune",
                  "code_departement", "code_region", "date_ouverture_station", "longitude_station",
                  "latitude_station")

# Series -> (name, unit, divisor from Hub'eau's unit)
SERIES = {
    "H": ("Water level", "cm", 10),  # mm
    "Q": ("Flow", "m³/s", 1000),  # l/s
}
# Station type -> the series it has. FICTIVE (computed) stations are left out.
TYPES = {"STD": ("H", "Q"), "DEB": ("Q",), "H": ("H",), "LIMNI": ("H",), "LIMNIMERE": ("H",), "LIMNIFILLE": ("H",)}


def stations(cache: Path | None = None) -> list[dict]:
    """Every station in service, as Hub'eau describes it."""
    store = Cache(SOURCE, cache)
    found: dict[str, dict] = {}  # by code: pages cached at different times may overlap
    page = 1
    while True:
        query = {"format": "json", "en_service": "true", "size": PAGE, "page": page, "fields": ",".join(STATION_FIELDS)}
        body = json.loads(store.fetch(f"{BASE}/referentiel/stations?{urlencode(query)}",
                                      f"stations/{_fields_key()}-{page}.json", STATIONS_MAX_AGE))
        for station in body["data"]:
            found.setdefault(station["code_station"], station)
        if not body.get("next") or not body["data"]:
            return list(found.values())
        page += 1


def _fields_key() -> str:
    """Names the cached station pages after the page size and fields asked for, so a change refetches them."""
    return f"{PAGE}-{zlib.crc32(','.join(STATION_FIELDS).encode()):08x}"


def read(
    series: list[str] | tuple[str, ...],
    start: Time = None,
    end: Time = None,
    resolution: str = "day",
    cache: Path | None = None,
) -> Table:
    """The series ("<station code>/H|Q") between `start` and `end` (both inclusive)."""
    if resolution not in RESOLUTIONS:
        raise ValueError(f"resolution must be one of {RESOLUTIONS}, got {resolution!r}")
    lo, hi = sources.epoch_ms(start, TZ), sources.epoch_ms(end, TZ)
    if lo is not None and hi is not None and lo > hi:
        raise ValueError(f"start {start} is after end {end}")
    store = Cache(SOURCE, cache)
    now = _now()
    recent = hi is None or hi >= now - REALTIME_SPAN * 1000

    columns = []
    for name in series:
        station, kind = _split(name)
        time, values = _realtime(store, station, kind) if recent else (np.array([], np.int64), np.array([]))
        time, values = _within(time, values, lo, hi)
        if resolution != "native":
            time, values = sources.aggregate(time, values.reshape(-1, 1), resolution, TZ)
            values = values[:, 0]
        if resolution == "day" and kind == "Q":
            time, values = _with_history(store, station, time, values, lo, hi, now)
        columns.append(Table((name,), time, values.reshape(-1, 1)))
    combined = sources.combine(columns)
    return Table(tuple(series), combined.time.astype("datetime64[ms]"), combined.values)


def _split(name: str) -> tuple[str, str]:
    station, slash, kind = name.partition("/")
    if not (station and slash and kind in SERIES):
        raise ValueError(f"a series is named '<station code>/H' or '<station code>/Q', not {name!r}")
    return station, kind


def _within(time: np.ndarray, values: np.ndarray, lo: int | None, hi: int | None) -> tuple[np.ndarray, np.ndarray]:
    keep = np.ones(len(time), bool)
    if lo is not None:
        keep &= time >= lo
    if hi is not None:
        keep &= time <= hi
    return time[keep], values[keep]


def _realtime(store: Cache, station: str, kind: str) -> tuple[np.ndarray, np.ndarray]:
    query = {"code_entite": station, "grandeur_hydro": kind, "size": 20000, "fields": "date_obs,resultat_obs"}
    body = json.loads(store.fetch(f"{BASE}/observations_tr?{urlencode(query)}", f"{station}/{kind}.json",
                                  REALTIME_MAX_AGE))
    rows = body["data"]
    time = np.array([_utc_ms(row["date_obs"]) for row in rows], np.int64)
    values = np.array([np.nan if row["resultat_obs"] is None else row["resultat_obs"] for row in rows])
    values /= SERIES[kind][2]
    order = np.argsort(time, kind="stable")  # Hub'eau sends the newest first
    return time[order], values[order]


def _with_history(
    store: Cache, station: str, time: np.ndarray, values: np.ndarray, lo: int | None, hi: int | None, now: int
) -> tuple[np.ndarray, np.ndarray]:
    """Daily mean flows from the history, and the real-time daily means after its last day."""
    first = _year(lo) if lo is not None else FIRST_YEAR
    last = _year(min(hi, now) if hi is not None else now)
    decades = range(max(first, FIRST_YEAR) // 10 * 10, last // 10 * 10 + 1, 10)
    with Pool(DOWNLOADS) as pool:
        parts = list(pool.map(lambda decade: _daily(store, station, decade, now), decades))
    days = np.concatenate([p[0] for p in parts] or [np.array([], np.int64)])
    flows = np.concatenate([p[1] for p in parts] or [np.array([])])
    days, flows = _within(days, flows, lo, hi)
    if len(days):
        after = time > days[-1]
        days, flows = np.concatenate([days, time[after]]), np.concatenate([flows, values[after]])
        return days, flows
    return time, values


def _daily(store: Cache, station: str, decade: int, now: int) -> tuple[np.ndarray, np.ndarray]:
    query = {"code_entite": station, "grandeur_hydro_elab": "QmnJ", "size": 20000,
             "date_debut_obs_elab": f"{decade}-01-01", "date_fin_obs_elab": f"{decade + 9}-12-31",
             "fields": "date_obs_elab,resultat_obs_elab"}
    max_age = DAILY_MAX_AGE if decade + 10 > _year(now) else OLD_DAILY_MAX_AGE
    body = json.loads(store.fetch(f"{BASE}/obs_elab?{urlencode(query)}", f"{station}/QmnJ/{decade}.json", max_age))
    rows = sorted(body["data"], key=lambda row: row["date_obs_elab"])
    midnights = (datetime.combine(date.fromisoformat(row["date_obs_elab"]), datetime.min.time()) for row in rows)
    days = np.array([sources.epoch_ms(midnight, TZ) for midnight in midnights], np.int64)
    flows = np.array([np.nan if row["resultat_obs_elab"] is None else row["resultat_obs_elab"] for row in rows])
    return days, flows / SERIES["Q"][2]


def _utc_ms(stamp: str) -> int:
    return int(datetime.fromisoformat(stamp.replace("Z", "+00:00")).astimezone(UTC).timestamp() * 1000)


def _year(ms: int) -> int:
    return datetime.fromtimestamp(ms / 1000, TZ).year


def _now() -> int:
    return int(clock.time() * 1000)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download Hub'eau river gauge series to CSV.")
    parser.add_argument("series", nargs="*", help="<station code>/H or /Q, e.g. F700000103/Q; see --list")
    parser.add_argument("--resolution", default="day", choices=RESOLUTIONS)
    parser.add_argument("--start", help=f"ISO date or time, Paris time unless an offset is given "
                        f"(default: {sources.CLI_DAYS} days before the end)")
    parser.add_argument("--end", help="ISO date or time, inclusive; a bare date means the whole day (default: now)")
    parser.add_argument("--out", help="CSV file (default: stdout)")
    parser.add_argument("--list", metavar="SEARCH", nargs="?", const="",
                        help="show the stations, optionally only matching ones")
    args = parser.parse_args(argv)

    if args.list is not None or not args.series:
        words = (args.list or "").casefold().split()
        for s in stations():
            text = f"{s['code_station']} {s['libelle_station']} {s['libelle_cours_eau'] or ''}"
            if all(w in text.casefold() for w in words):
                kinds = " ".join(TYPES.get(s["type_station"], ()))
                print(f"{s['code_station']}  {kinds:<4} {s['libelle_station']}")
        return
    try:
        start, end = sources.span(args.start, args.end, TZ)
        table = read(args.series, start, end, args.resolution)
    except ValueError as error:
        parser.error(str(error))
    if args.out:
        with open(args.out, "w") as out:
            sources.write_csv(table, out, TZ)
    else:
        sources.write_csv(table, sys.stdout, TZ)


if __name__ == "__main__":
    main()
