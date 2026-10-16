"""Reader for SMARD (smard.de), the Bundesnetzagentur's electricity market data.

SMARD serves its charts from static JSON files, one file per series, region,
resolution and chunk. No API key is needed and the data is free to use
(CC BY 4.0, "Bundesnetzagentur | SMARD.de").

    https://www.smard.de/app/chart_data/{filter}/{region}/index_{resolution}.json
        -> {"timestamps": [chunk start, epoch ms, ...]}
    https://www.smard.de/app/chart_data/{filter}/{region}/{filter}_{region}_{resolution}_{chunk}.json
        -> {"meta_data": {...}, "series": [[epoch ms, value | null], ...]}

A chunk is one week for "quarterhour" and "hour" and one year for the coarser
resolutions. Energy values are MWh per interval (so a quarter-hour value is a
quarter of the hourly one); prices are EUR/MWh.

Times are returned as UTC `datetime64[ms]`. Date arguments without a UTC
offset are read as Berlin time, and the CSV output is in Berlin time.

Downloads go through the free-data cache (cache.py), source "de-smard".
How long a file stays fresh depends on how much it can still change:

    index of chunks            1 hour   (a new chunk appears every week)
    newest chunk              15 min    (still being filled in)
    chunks of the last 90 days 1 day    (provisional values get consolidated)
    older chunks               forever

    python smard.py 410 --start 2024-01-01 --end 2024-12-31 --out load.csv

On the command line the end date is inclusive (a bare date means the whole
day) and the start defaults to 30 days before the end, since the full
history of one hourly series is several hundred files.
"""

from __future__ import annotations

import argparse
import sys
import time as clock
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

import sources
from cache import DAY, HOUR, MINUTE, Cache, FetchError
from sources import Table, Time, combine

BASE = "https://www.smard.de/app/chart_data"
SOURCE = "de-smard"  # the id in data/ and the cache folder
INDEX_MAX_AGE = HOUR
LATEST_MAX_AGE = 15 * MINUTE
RECENT = 90 * DAY
RECENT_MAX_AGE = DAY
TZ = ZoneInfo("Europe/Berlin")
RESOLUTIONS = ("quarterhour", "hour", "day", "week", "month", "year")
REGIONS = ("DE", "AT", "LU", "DE-LU", "DE-AT-LU", "50Hertz", "Amprion", "TenneT", "TransnetBW")

# Series checked against the live endpoints. Prices are published for the
# bidding zone "DE-LU", everything else for "DE" (or a sub-region).
FILTERS: dict[int, str] = {
    # Realised generation (MWh)
    1223: "generation_lignite",
    1224: "generation_nuclear",
    1225: "generation_wind_offshore",
    1226: "generation_hydro",
    1227: "generation_other_conventional",
    1228: "generation_other_renewable",
    4066: "generation_biomass",
    4067: "generation_wind_onshore",
    4068: "generation_solar",
    4069: "generation_hard_coal",
    4070: "generation_pumped_storage",
    4071: "generation_natural_gas",
    # Realised consumption (MWh)
    410: "consumption_total",
    4359: "consumption_residual_load",
    4387: "consumption_pumped_storage",
    # Forecasts (MWh)
    411: "forecast_consumption_total",
    4362: "forecast_residual_load",
    122: "forecast_generation_total",
    123: "forecast_generation_wind_onshore",
    125: "forecast_generation_solar",
    3791: "forecast_generation_wind_offshore",
    715: "forecast_generation_other",
    5097: "forecast_generation_wind_and_solar",
    # Day-ahead prices (EUR/MWh), region "DE-LU"
    4169: "price_de_lu",
    5078: "price_neighbours_de_lu",
    254: "price_france",
}

class SmardError(RuntimeError):
    pass


def read(
    filter: int,
    region: str = "DE",
    resolution: str = "hour",
    start: Time = None,
    end: Time = None,
    cache: Path | None = None,
) -> Table:
    """One SMARD series between `start` and `end` (both inclusive), as a one-column table.

    Gaps inside the series stay NaN. Leading and trailing NaN are dropped:
    SMARD pads the first chunk before the series begins and the newest chunk
    up to the end of the week.
    """
    if resolution not in RESOLUTIONS:
        raise ValueError(f"resolution must be one of {RESOLUTIONS}, got {resolution!r}")
    store = Cache(SOURCE, cache)
    lo, hi = sources.epoch_ms(start, TZ), sources.epoch_ms(end, TZ)
    if lo is not None and hi is not None and lo > hi:
        raise ValueError(f"start {start} is after end {end}")

    chunks = _chunks(store, filter, region, resolution)
    if not chunks:
        raise SmardError(f"no data for filter {filter} in region {region!r} at {resolution!r}")
    wanted = [
        c for i, c in enumerate(chunks)
        if (hi is None or c <= hi) and (lo is None or i + 1 == len(chunks) or chunks[i + 1] > lo)
    ]
    rows: dict[int, float] = {}  # later chunks win on overlapping times
    for chunk in wanted:
        for t, v in _chunk(store, filter, region, resolution, chunk, latest=chunk == chunks[-1]):
            rows[t] = np.nan if v is None else v

    time = np.array(sorted(rows), dtype=np.int64)
    values = np.array([rows[t] for t in time], dtype=np.float64)
    keep = np.ones(len(time), dtype=bool)
    if lo is not None:
        keep &= time >= lo
    if hi is not None:
        keep &= time <= hi
    time, values = time[keep], values[keep]

    valid = np.flatnonzero(~np.isnan(values))
    span = slice(valid[0], valid[-1] + 1) if len(valid) else slice(0, 0)
    name = FILTERS.get(filter, str(filter))
    return Table((name,), time[span].astype("datetime64[ms]"), values[span].reshape(-1, 1))


def read_frame(
    filters: list[int] | tuple[int, ...],
    region: str = "DE",
    resolution: str = "hour",
    start: Time = None,
    end: Time = None,
    cache: Path | None = None,
) -> Table:
    """Several series of one region side by side, one column each."""
    return combine([read(f, region, resolution, start, end, cache) for f in filters])


def _chunks(store: Cache, filter: int, region: str, resolution: str) -> list[int]:
    key = f"{filter}/{region}/{resolution}/index_{resolution}.json"
    data = _get(store, f"{BASE}/{filter}/{region}/index_{resolution}.json", key, INDEX_MAX_AGE)
    return sorted(data["timestamps"])


def _chunk(
    store: Cache, filter: int, region: str, resolution: str, chunk: int, latest: bool
) -> list[tuple[int, float | None]]:
    name = f"{filter}_{region}_{resolution}_{chunk}.json"
    if latest:
        max_age: float | None = LATEST_MAX_AGE
    elif clock.time() - chunk / 1000 < RECENT:
        max_age = RECENT_MAX_AGE
    else:
        max_age = None
    data = _get(store, f"{BASE}/{filter}/{region}/{name}", f"{filter}/{region}/{resolution}/{name}", max_age)
    return [(int(t), v) for t, v in data["series"]]


def _get(store: Cache, url: str, key: str, max_age: float | None) -> dict:
    try:
        return store.fetch_json(url, key, max_age)
    except FetchError as error:
        if error.status == 404:
            raise SmardError(f"not found: {url} (unknown filter/region/resolution?)") from error
        raise


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download SMARD series to CSV.")
    parser.add_argument("filters", nargs="*", type=int, help="filter ids, see --list")
    parser.add_argument("--region", default="DE", choices=REGIONS)
    parser.add_argument("--resolution", default="hour", choices=RESOLUTIONS)
    parser.add_argument("--start", help=f"ISO date or time, Berlin time unless an offset is given "
                        f"(default: {sources.CLI_DAYS} days before the end)")
    parser.add_argument("--end", help="ISO date or time, inclusive; a bare date means the whole day (default: now)")
    parser.add_argument("--out", help="CSV file (default: stdout)")
    parser.add_argument("--list", action="store_true", help="show the known filter ids")
    args = parser.parse_args(argv)

    if args.list or not args.filters:
        for key, name in FILTERS.items():
            print(f"{key:>5}  {name}")
        return
    try:
        start, end = sources.span(args.start, args.end, TZ)
        table = read_frame(args.filters, args.region, args.resolution, start, end)
    except (ValueError, SmardError) as error:
        parser.error(str(error))
    if args.out:
        with open(args.out, "w") as out:
            sources.write_csv(table, out, TZ)
    else:
        sources.write_csv(table, sys.stdout, TZ)


if __name__ == "__main__":
    main()
