"""Reader for RTE éCO2mix, France's national electricity mix, from ODRÉ.

ODRÉ (Open Data Réseaux Énergies, odre.opendatasoft.com) publishes éCO2mix as
two Opendatasoft datasets. No API key is needed; the licence is Etalab
Licence Ouverte 2.0.

    eco2mix-national-cons-def   consolidated and final data, from 2012 up to
                                the end of the last consolidated month
    eco2mix-national-tr         real-time data since then, with empty rows
                                up to the end of tomorrow

Both have one row per quarter hour, but the consolidated history fills most
fields only on the hour and half hour. Values are average power in MW (so an
hourly mean in MW equals the energy in MWh of that hour), the CO2 intensity
is g/kWh. Negative exchanges are exports, negative storage is charging.

    https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/{dataset}/exports/csv
        ?where=date_heure >= '...' and date_heure < '...'&timezone=UTC&order_by=date_heure

Times are returned as UTC `datetime64[ms]`. Date arguments without a UTC
offset are read as Paris time, and the CSV output is in Paris time.

Downloads go through the free-data cache (cache.py), source "fr-rte-eco2mix",
one file per calendar year of the history plus one for all real-time data:

    real-time data                    15 min
    history of this and last year      1 day   (consolidated values get final)
    older history                     forever

    python -m free_data.eco2mix consommation nucleaire --start 2024-01-01 --end 2024-12-31 --out fr.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import time as clock
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

import numpy as np

from free_data import sources
from free_data.cache import DAY, MINUTE, Cache
from free_data.sources import Table, Time

BASE = "https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets"
HISTORY = "eco2mix-national-cons-def"
REALTIME = "eco2mix-national-tr"
SOURCE = "fr-rte-eco2mix"  # the id in data/ and the cache folder
FIRST_YEAR = 2012
REALTIME_MAX_AGE = 15 * MINUTE
RECENT_MAX_AGE = DAY
REALTIME_SPAN = 180 * DAY  # the real-time data never reaches further back than this
DOWNLOADS = 4  # years fetched in parallel
TZ = ZoneInfo("Europe/Paris")
RESOLUTIONS = ("quarterhour", "hour", "day")

# Field -> (group, label, unit, description). Fields marked "real time only"
# are missing from the consolidated history.
FIELDS: dict[str, tuple[str, str, str, str]] = {
    "consommation": ("Consumption", "Total", "MW", "Electricity consumption of mainland France."),
    "prevision_j1": ("Consumption / Forecast", "Day-ahead", "MW", "Consumption forecast made the day before."),
    "prevision_j": ("Consumption / Forecast", "Same day", "MW", "Consumption forecast made the same day."),
    "nucleaire": ("Generation", "Nuclear", "MW", ""),
    "eolien": ("Generation", "Wind", "MW", "Onshore and offshore wind."),
    "eolien_terrestre": ("Generation / Wind", "Onshore", "MW", "Real time only (about the last two months)."),
    "eolien_offshore": ("Generation / Wind", "Offshore", "MW", "Real time only (about the last two months)."),
    "solaire": ("Generation", "Solar", "MW", ""),
    "hydraulique": ("Generation", "Hydro", "MW", "All hydro generation."),
    "hydraulique_fil_eau_eclusee": ("Generation / Hydro", "Run-of-river and pondage", "MW", ""),
    "hydraulique_lacs": ("Generation / Hydro", "Lakes", "MW", ""),
    "hydraulique_step_turbinage": ("Generation / Hydro", "Pumped storage (turbining)", "MW", ""),
    "bioenergies": ("Generation", "Bioenergy", "MW", "All bioenergy generation."),
    "bioenergies_dechets": ("Generation / Bioenergy", "Waste", "MW", ""),
    "bioenergies_biomasse": ("Generation / Bioenergy", "Biomass", "MW", ""),
    "bioenergies_biogaz": ("Generation / Bioenergy", "Biogas", "MW", ""),
    "gaz": ("Generation", "Gas", "MW", "All gas-fired generation."),
    "gaz_ccg": ("Generation / Gas", "Combined cycle", "MW", ""),
    "gaz_tac": ("Generation / Gas", "Combustion turbines", "MW", ""),
    "gaz_cogen": ("Generation / Gas", "Cogeneration", "MW", ""),
    "gaz_autres": ("Generation / Gas", "Other", "MW", ""),
    "charbon": ("Generation", "Coal", "MW", ""),
    "fioul": ("Generation", "Oil", "MW", "All oil-fired generation."),
    "fioul_tac": ("Generation / Oil", "Combustion turbines", "MW", ""),
    "fioul_cogen": ("Generation / Oil", "Cogeneration", "MW", ""),
    "fioul_autres": ("Generation / Oil", "Other", "MW", ""),
    "pompage": ("Storage", "Pumping", "MW", "Power drawn by pumped-storage plants (negative)."),
    "stockage_batterie": ("Storage", "Battery charging", "MW", "Real time only (about the last two months); negative."),
    "destockage_batterie": ("Storage", "Battery discharging", "MW", "Real time only (about the last two months)."),
    "ech_physiques": ("Exchanges", "Physical balance", "MW",
                      "Net physical flow on all interconnections; negative is export."),
    "ech_comm_angleterre": ("Exchanges / Commercial", "Great Britain", "MW",
                            "Net commercial exchange; negative is export."),
    "ech_comm_espagne": ("Exchanges / Commercial", "Spain", "MW", "Net commercial exchange; negative is export."),
    "ech_comm_italie": ("Exchanges / Commercial", "Italy", "MW", "Net commercial exchange; negative is export."),
    "ech_comm_suisse": ("Exchanges / Commercial", "Switzerland", "MW", "Net commercial exchange; negative is export."),
    "ech_comm_allemagne_belgique": ("Exchanges / Commercial", "Germany and Belgium", "MW",
                                    "Net commercial exchange; negative is export."),
    "taux_co2": ("Emissions", "CO2 intensity", "g/kWh", "Estimated CO2 emitted per kWh generated in France."),
}


class Rows:
    """One downloaded file: times (epoch ms, UTC) and a column per field."""

    def __init__(self, time: np.ndarray, columns: dict[str, np.ndarray]) -> None:
        self.time = time
        self.columns = columns

    def take(self, keep: np.ndarray, fields: list[str]) -> np.ndarray:
        """The rows where `keep` is true, one column per field (NaN where the file lacks it)."""
        empty = np.full(len(self.time), np.nan)
        return np.column_stack([self.columns.get(f, empty)[keep] for f in fields]).reshape(-1, len(fields))


def read(
    fields: list[str] | tuple[str, ...],
    start: Time = None,
    end: Time = None,
    resolution: str = "hour",
    cache: Path | None = None,
) -> Table:
    """The fields between `start` and `end` (both inclusive), one column each.

    "hour" and "day" are means over each Paris hour or day of the values
    there are. At "quarterhour" rows without any value are dropped, so the
    history comes out half-hourly; otherwise gaps stay NaN and only empty
    rows at either end are dropped.
    """
    fields = list(fields)
    unknown = [f for f in fields if f not in FIELDS]
    if unknown:
        raise ValueError(f"unknown fields {unknown}, see FIELDS")
    if resolution not in RESOLUTIONS:
        raise ValueError(f"resolution must be one of {RESOLUTIONS}, got {resolution!r}")
    lo, hi = sources.epoch_ms(start, TZ), sources.epoch_ms(end, TZ)
    if lo is not None and hi is not None and lo > hi:
        raise ValueError(f"start {start} is after end {end}")
    store = Cache(SOURCE, cache)
    now = _now()

    # The real-time data takes over where the history ends.
    realtime = None
    if hi is None or hi > now - REALTIME_SPAN * 1000:
        realtime = _parse(store.fetch(_url(REALTIME), f"{REALTIME}.csv", REALTIME_MAX_AGE))
    boundary = int(realtime.time[0]) if realtime is not None and len(realtime.time) else None

    first = _year(lo) if lo is not None else FIRST_YEAR
    last = _year(min(hi, now) if hi is not None else now)
    years = [y for y in range(max(first, FIRST_YEAR), last + 1) if boundary is None or _year_start(y) < boundary]
    if boundary is not None and lo is not None and lo >= boundary:
        years = []  # all of it is real-time data
    with ThreadPoolExecutor(DOWNLOADS) as pool:
        history = list(pool.map(lambda y: _parse(_history(store, y)), years))

    parts = [(rows, rows.time < boundary if boundary is not None else np.ones(len(rows.time), bool))
             for rows in history]
    if realtime is not None:
        parts.append((realtime, np.ones(len(realtime.time), bool)))
    time = np.concatenate([rows.time[keep] for rows, keep in parts] or [np.array([], np.int64)])
    values = np.concatenate([rows.take(keep, fields) for rows, keep in parts] or [np.empty((0, len(fields)))])

    keep = np.ones(len(time), bool)
    if lo is not None:
        keep &= time >= lo
    if hi is not None:
        keep &= time <= hi
    time, values = time[keep], values[keep]
    order = np.argsort(time, kind="stable")
    time, values = time[order], values[order]

    if resolution == "quarterhour":
        filled = ~np.isnan(values).all(axis=1)
        time, values = time[filled], values[filled]
    else:
        time, values = sources.aggregate(time, values, resolution, TZ)
        filled = np.flatnonzero(~np.isnan(values).all(axis=1))
        span = slice(filled[0], filled[-1] + 1) if len(filled) else slice(0, 0)
        time, values = time[span], values[span]
    return Table(tuple(fields), time.astype("datetime64[ms]"), values)


def _history(store: Cache, year: int) -> bytes:
    """One Paris calendar year of the consolidated history."""
    max_age = None if year < _year(_now()) - 1 else RECENT_MAX_AGE
    where = (f"date_heure >= '{_iso(_year_start(year))}' and date_heure < '{_iso(_year_start(year + 1))}'")
    return store.fetch(_url(HISTORY, where), f"{HISTORY}/{year}.csv", max_age)


def _url(dataset: str, where: str = "") -> str:
    query = {"order_by": "date_heure", "timezone": "UTC", **({"where": where} if where else {})}
    return f"{BASE}/{dataset}/exports/csv?{urlencode(query)}"


def _parse(body: bytes) -> Rows:
    """An ODRÉ CSV export: ";"-separated, times as "...+00:00", some numbers quoted as text."""
    reader = csv.reader(io.StringIO(body.decode("utf-8-sig")), delimiter=";")
    header = next(reader, [])
    rows = list(reader)
    at = {name: i for i, name in enumerate(header)}
    stamps = [row[at["date_heure"]] for row in rows]
    if any(not s.endswith("+00:00") for s in stamps):
        raise ValueError("expected UTC times in the éCO2mix export")
    time = np.array([s[:19] for s in stamps], dtype="datetime64[ms]").astype(np.int64)
    columns = {f: np.array([_number(row[at[f]]) for row in rows]) for f in FIELDS if f in at}
    return Rows(time, columns)


def _number(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        return np.nan  # empty, or a marker like "ND"


def _now() -> int:
    return int(clock.time() * 1000)


def _year(ms: int) -> int:
    return datetime.fromtimestamp(ms / 1000, TZ).year


def _year_start(year: int) -> int:
    return sources.epoch_ms(datetime(year, 1, 1), TZ)


def _iso(ms: int) -> str:
    return str(np.datetime64(ms, "ms").astype("datetime64[s]")) + "Z"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download RTE éCO2mix series to CSV.")
    parser.add_argument("fields", nargs="*", help="field names, see --list")
    parser.add_argument("--resolution", default="hour", choices=RESOLUTIONS)
    parser.add_argument("--start", help=f"ISO date or time, Paris time unless an offset is given "
                        f"(default: {sources.CLI_DAYS} days before the end)")
    parser.add_argument("--end", help="ISO date or time, inclusive; a bare date means the whole day (default: now)")
    parser.add_argument("--out", help="CSV file (default: stdout)")
    parser.add_argument("--list", action="store_true", help="show the known fields")
    args = parser.parse_args(argv)

    if args.list or not args.fields:
        for name, (group, label, unit, _) in FIELDS.items():
            print(f"{name:<28} {group} / {label} [{unit}]")
        return
    try:
        start, end = sources.span(args.start, args.end, TZ)
        table = read(args.fields, start, end, args.resolution)
    except ValueError as error:
        parser.error(str(error))
    if args.out:
        with open(args.out, "w") as out:
            sources.write_csv(table, out, TZ)
    else:
        sources.write_csv(table, sys.stdout, TZ)


if __name__ == "__main__":
    main()
