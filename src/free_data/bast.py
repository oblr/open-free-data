"""Reader for the BASt's automatic traffic counting stations (Bundesanstalt für Straßenwesen).

Vehicles per hour at the permanent counting stations (Dauerzählstellen) on
motorways (A) and federal roads (B), about 2,100 of them, from 2003 to the
last published year. No API key is needed.

The BASt publishes one zip per year and road class, each holding a single
";"-separated text file of 1 to 2 GB with a row per station and hour:

    https://www.bast.de/videos/{year}_{A|B}_S.zip        the hourly counts, about 260 MB zipped
    .../verkehrszaehlung/Stundenwerte.html               the page that lists them
    .../verkehrszaehlung/Daten/{year}_1/Jawe{year}.csv   the stations of a year: name, road, coordinates

A row has the station (Zst), the date (YYMMDD), the hour (01 is 00:00 to
01:00 Berlin time), and per direction the count of all vehicles (KFZ) and
of the truck group (Lkw), each followed by a check mark: "-" correct, "u"
correct but unusual (an event), "s" and "k" estimated for a missing or
faulty count, "a" missing, "d" faulty, "z" the hour of a clock change.
Missing and faulty counts are NaN, as is -1 (not counted by the device);
estimates are kept. Every day has 24 rows: on the day the clocks go forward
the hour that doesn't exist is a "z" row with 0, on the day they go back
the two hours from 02:00 share one "z" row. Both are dropped, so the
autumn day has a gap of two hours (00:00 and 01:00 UTC).

The rows are fixed-width (each column padded with spaces), so a block of
them is read as a byte matrix; a block that isn't falls back to splitting.

A series is named "<station>/<variable>", e.g. "9014/KFZ": KFZ is all
vehicles in both directions, KFZ_R1 and KFZ_R2 one direction each, and LKW
the truck group in both directions. Which vehicles count as trucks depends
on the device's setting (heavy vehicles over 3.5 t, or truck-like), so LKW
compares poorly between stations. The stations offered are those of the
latest station list; their history goes back as far as the yearly files.

The yearly files are too big to keep. The first read of a year downloads
its file, reduces it to one small file per station and deletes it; reading
one year of one road class takes a minute or two. Downloads go through the
free-data cache (cache.py), source "de-bast-traffic-counts":

    the list of years and the station list      1 day
    <year>/<station>.npz                         forever (a year doesn't change once published)
    <year>/<A|B>.json                            the stations found in a year's file, kept forever

Times are returned as UTC `datetime64[ms]`. Date arguments without a UTC
offset are read as Berlin time, and the CSV output is in Berlin time.

    python -m free_data.bast --list köln
    python -m free_data.bast 5113/KFZ 5113/LKW --start 2024-06-01 --end 2024-06-30
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import threading
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from free_data import sources
from free_data.cache import DAY, Cache
from free_data.sources import Table, Time

SITE = "https://www.bast.de"
PAGES = f"{SITE}/DE/Themen/Digitales/HF_1/Massnahmen/verkehrszaehlung"
SOURCE = "de-bast-traffic-counts"  # the id in data/ and the cache folder
LIST_MAX_AGE = DAY
TZ = ZoneInfo("Europe/Berlin")
CLASSES = ("A", "B")  # motorways, federal roads
BLOCK = 64 << 20  # bytes of text parsed at a time
HOUR_MS = 3_600_000

# The variables of a series, from the columns kept per station: (columns summed, name, description).
COLUMNS = ("KFZ_R1", "KFZ_R2", "LKW_R1", "LKW_R2")
VARIABLES = {
    "KFZ": ((0, 1), "All vehicles", "all vehicles, both directions"),
    "KFZ_R1": ((0,), "All vehicles, direction 1", "all vehicles in direction 1"),
    "KFZ_R2": ((1,), "All vehicles, direction 2", "all vehicles in direction 2"),
    "LKW": ((2, 3), "Trucks", "the truck group, both directions"),
}
MISSING = frozenset(b"ad")  # check marks of a count that isn't there or is wrong
CLOCK_CHANGE = ord("z")


class BastError(RuntimeError):
    pass


@dataclass(frozen=True)
class Station:
    id: str  # the BASt's number, four digits: "5113", "0223"
    name: str  # "Köln-Nord (O)"
    state: str  # two letters, "NW"
    road_class: str  # "A" or "B"
    road: str  # "A 1", "B 51"
    lat: float | None
    lon: float | None
    direction1: str  # where direction 1 leads, "Dortmund"
    direction2: str


def years(cache: Path | None = None) -> dict[int, tuple[str, ...]]:
    """The published years and the road classes each has a file for."""
    page = Cache(SOURCE, cache).fetch(f"{PAGES}/Stundenwerte.html", "years.html", LIST_MAX_AGE).decode("latin-1")
    found: dict[int, set[str]] = {}
    for year, road_class in re.findall(r"videos/(\d{4})_([AB])_S\.zip", page):
        found.setdefault(int(year), set()).add(road_class)
    if not found:
        raise BastError(f"no yearly files listed on {PAGES}/Stundenwerte.html")
    return {year: tuple(sorted(found[year])) for year in sorted(found)}


def stations(cache: Path | None = None) -> list[Station]:
    """The stations of the latest station list (Jahresauswertung) that are on a motorway or federal road."""
    store = Cache(SOURCE, cache)
    page = store.fetch(f"{PAGES}/Aktuell/zaehl_aktuell_node.html", "stations.html", LIST_MAX_AGE)
    match = re.search(rb'(Daten/\d{4}_1/Jawe\d{4}\.csv)\?view=renderTcDataExportCSV"', page)
    if match is None:
        raise BastError(f"no station list linked from {PAGES}/Aktuell/zaehl_aktuell_node.html")
    url = f"{PAGES}/{match.group(1).decode()}?view=renderTcDataExportCSV"
    return parse_stations(store.fetch(url, "stations.csv", LIST_MAX_AGE).decode("latin-1"))


def parse_stations(text: str) -> list[Station]:
    found = []
    for row in csv.DictReader(io.StringIO(text), delimiter=";"):
        road_class = (row.get("Str_Kl") or "").strip()
        if road_class not in CLASSES or not (row.get("DZ_Nr") or "").strip():
            continue
        found.append(Station(
            id=row["DZ_Nr"].strip().zfill(4),
            name=row["DZ_Name"].strip(),
            state=row["Land_Code"].strip(),
            road_class=road_class,
            road=f"{road_class} {row['Str_Nr'].strip()}{(row.get('Str_Zus') or '').strip()}",
            lat=_number(row.get("Koor_WGS84_N")),
            lon=_number(row.get("Koor_WGS84_E")),
            direction1=(row.get("Fernziel_Ri1") or "").strip(),
            direction2=(row.get("Fernziel_Ri2") or "").strip(),
        ))
    return found


def _number(text: str | None) -> float | None:
    """A coordinate with a decimal comma, "53,50754848"."""
    text = (text or "").strip().replace(",", ".")
    try:
        return float(text) if text else None
    except ValueError:
        return None


def read(series: list[str] | tuple[str, ...], start: Time = None, end: Time = None,
         cache: Path | None = None) -> Table:
    """The series ("<station>/<variable>") between `start` and `end` (both inclusive)."""
    lo, hi = sources.epoch_ms(start, TZ), sources.epoch_ms(end, TZ)
    if lo is not None and hi is not None and lo > hi:
        raise ValueError(f"start {start} is after end {end}")
    store = Cache(SOURCE, cache)
    wanted = [_split(name) for name in series]
    classes = {s.id: s.road_class for s in stations(cache)}
    unknown = sorted({station for station, _ in wanted if station not in classes})
    if unknown:
        raise ValueError(f"no BASt counting station {', '.join(unknown)}; see --list")
    published = years(cache)
    first = _year(lo) if lo is not None else min(published)
    last = _year(hi) if hi is not None else max(published)

    loaded: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for station in dict.fromkeys(station for station, _ in wanted):
        parts = [_station_year(store, station, year, classes[station])
                 for year in range(first, last + 1) if classes[station] in published.get(year, ())]
        time = np.concatenate([p[0] for p in parts] or [np.array([], np.int64)])
        values = np.concatenate([p[1] for p in parts] or [np.empty((0, len(COLUMNS)), np.float32)])
        loaded[station] = time, values

    columns = []
    for name, (station, variable) in zip(series, wanted, strict=True):
        time, values = loaded[station]
        keep = np.ones(len(time), bool)
        if lo is not None:
            keep &= time >= lo
        if hi is not None:
            keep &= time <= hi
        summed = values[keep][:, list(VARIABLES[variable][0])].astype(np.float64).sum(axis=1)  # NaN if one is
        columns.append(Table((name,), time[keep].astype("datetime64[ms]"), summed.reshape(-1, 1)))
    combined = sources.combine(columns)
    return Table(tuple(series), combined.time.astype("datetime64[ms]"), combined.values)


def _split(name: str) -> tuple[str, str]:
    station, slash, variable = name.partition("/")
    variable = variable.upper()
    if not (station.strip() and slash and variable in VARIABLES):
        raise ValueError(f"a series is named '<station>/<variable>' with a variable from {list(VARIABLES)}, "
                         f"not {name!r}")
    return station.strip().zfill(4), variable


def _year(ms: int) -> int:
    return datetime.fromtimestamp(ms / 1000, TZ).year


_locks: dict[tuple[int, str], threading.Lock] = {}
_locks_lock = threading.Lock()


def _station_year(store: Cache, station: str, year: int, road_class: str) -> tuple[np.ndarray, np.ndarray]:
    """One station's hours in one year, reducing that year's file first if it hasn't been."""
    marker = f"{year}/{road_class}.json"
    if not store.cached(marker):
        with _locks_lock:
            lock = _locks.setdefault((year, road_class), threading.Lock())
        with lock:  # two reads of the same year wait for one reduction
            if not store.cached(marker):
                reduce(store, year, road_class)
    key = f"{year}/{station}.npz"
    if not store.cached(key):  # the station didn't count that year
        return np.array([], np.int64), np.empty((0, len(COLUMNS)), np.float32)
    with np.load(store.path(key)) as saved:
        return saved["time"], saved["values"]


def reduce(store: Cache, year: int, road_class: str) -> list[str]:
    """Downloads one yearly file, keeps a file per station, deletes the download; the stations found."""
    zipped = f"{year}/{road_class}.zip"
    body = store.fetch(f"{SITE}/videos/{year}_{road_class}_S.zip", zipped, None)
    try:
        found = parse_year(body, year)
    finally:
        store.path(zipped).unlink(missing_ok=True)
    for station, (time, values) in found.items():
        buffer = io.BytesIO()
        np.savez_compressed(buffer, time=time, values=values)
        store.write(f"{year}/{station}.npz", buffer.getvalue())
    store.write(f"{year}/{road_class}.json", json.dumps(sorted(found)).encode())
    return sorted(found)


def parse_year(body: bytes, year: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """A yearly zip as {station: (UTC epoch ms, values with one column per COLUMNS)}, sorted by time."""
    with zipfile.ZipFile(io.BytesIO(body)) as archive:
        members = [n for n in archive.namelist() if n.lower().endswith(".txt")]
        if len(members) != 1:
            raise BastError(f"expected one .txt file in the BASt zip of {year}, found {members}")
        with archive.open(members[0]) as text:
            header = text.readline().decode("latin-1").strip().split(";")
            layout = Layout(header)
            hours = _hours(year)
            parts: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
            rest = b""
            while chunk := text.read(BLOCK):
                block = rest + chunk
                cut = block.rfind(b"\n") + 1
                rest = block[cut:]
                _collect(layout.rows(block[:cut]), hours, year, parts)
            if rest.strip():
                _collect(layout.rows(rest + b"\n"), hours, year, parts)
    found = {}
    for station, pieces in parts.items():
        time = np.concatenate([p[0] for p in pieces])
        values = np.concatenate([p[1] for p in pieces])
        time, first = np.unique(time, return_index=True)
        found[station] = time, values[first]
    return found


def _hours(year: int) -> np.ndarray:
    """UTC epoch ms of each hour of `year` in Berlin, by [day of the year, hour - 1]; the earlier of two."""
    days = (np.datetime64(f"{year + 1}-01-01") - np.datetime64(f"{year}-01-01")).astype(int)
    table = np.zeros((days, 24), np.int64)
    first = datetime(year, 1, 1)
    for day in range(days):
        midnight = datetime.fromordinal(first.toordinal() + day)
        for hour in range(24):
            table[day, hour] = sources.epoch_ms(midnight.replace(hour=hour, tzinfo=TZ), TZ)
    return table


def _collect(rows: dict[str, np.ndarray], hours: np.ndarray, year: int,
             parts: dict[str, list[tuple[np.ndarray, np.ndarray]]]) -> None:
    """Adds a block's rows to `parts`, per station, without the clock-change rows."""
    keep = ~rows["clock"] & (rows["year"] == year % 100) & (rows["hour"] >= 1) & (rows["hour"] <= 24)
    if not keep.any():
        return
    station, day, hour, values = rows["station"][keep], rows["day"][keep], rows["hour"][keep], rows["values"][keep]
    time = hours[day, hour - 1]
    order = np.argsort(station, kind="stable")
    station, time, values = station[order], time[order], values[order]
    ids, starts = np.unique(station, return_index=True)
    for sid, a, b in zip(ids, starts, [*starts[1:], len(station)], strict=True):
        parts.setdefault(f"{sid:04d}", []).append((time[a:b], values[a:b]))


class Layout:
    """Where the columns of a yearly file are, from its header and, for the fast path, its first row."""

    def __init__(self, header: list[str]) -> None:
        names = [name.strip().upper() for name in header]
        try:
            self.at = {key: names.index(key) for key in ("ZST", "DATUM", "STUNDE", *COLUMNS)}
            self.marks = {key: names.index(f"K_{key}") for key in COLUMNS}
        except ValueError as error:
            raise BastError(f"the BASt file lacks a column: {error}") from error
        self.width = 0  # bytes per row, once known; 0 while unknown or rows differ
        self.spans: list[tuple[int, int]] = []  # start and end of each cell in a row

    def rows(self, block: bytes) -> dict[str, np.ndarray]:
        """The block's rows as arrays: station, year, day of the year, hour, clock change, values."""
        if not self.width:
            line = block[: block.find(b"\n") + 1]
            cuts = [-1] + [i for i, b in enumerate(line) if b == ord(";")] + [len(line.rstrip(b"\r\n"))]
            self.width = len(line)
            self.spans = [(a + 1, b) for a, b in zip(cuts, cuts[1:], strict=False)]
        if len(block) % self.width == 0 and len(self.spans) > max(*self.at.values(), *self.marks.values()):
            matrix = np.frombuffer(block, np.uint8).reshape(-1, self.width)
            numbers = [self.spans[i][1] - 1 for i in self.at.values()]  # the last byte of each number
            if (all((matrix[:, b] == ord(";")).all() for _, b in self.spans[:-1])
                    and (matrix[:, -1] == ord("\n")).all()
                    and all(_digits(matrix[:, b]).all() for b in numbers)):
                return self._fixed(matrix)
        return self._split(block)

    def _cell(self, matrix: np.ndarray, column: int) -> np.ndarray:
        a, b = self.spans[column]
        return matrix[:, a:b]

    def _fixed(self, matrix: np.ndarray) -> dict[str, np.ndarray]:
        def ints(key: str) -> np.ndarray:
            return _integers(self._cell(matrix, self.at[key]))
        return self._finish(ints("ZST"), ints("DATUM"), ints("STUNDE"),
                            np.column_stack([ints(key) for key in COLUMNS]).astype(np.float32),
                            np.column_stack([self._cell(matrix, self.marks[key])[:, 0] for key in COLUMNS]))

    def _split(self, block: bytes) -> dict[str, np.ndarray]:
        cells = [line.split(b";") for line in block.splitlines() if line.strip()]
        def ints(key: str, blank: int = 0) -> np.ndarray:
            return np.array([_int(c[self.at[key]], blank) for c in cells], np.int64).reshape(-1)
        values = np.column_stack([ints(key, -1) for key in COLUMNS]).astype(np.float32)
        marks = np.array([[(c[self.marks[key]].strip() or b"-")[0] for key in COLUMNS] for c in cells], np.uint8)
        return self._finish(ints("ZST"), ints("DATUM"), ints("STUNDE"), values.reshape(-1, len(COLUMNS)),
                            marks.reshape(-1, len(COLUMNS)))

    @staticmethod
    def _finish(station: np.ndarray, date: np.ndarray, hour: np.ndarray, values: np.ndarray,
                marks: np.ndarray) -> dict[str, np.ndarray]:
        values[np.isin(marks, list(MISSING)) | (values < 0)] = np.nan
        yy, mm, dd = date // 10000, date // 100 % 100, date % 100
        valid = (mm >= 1) & (mm <= 12) & (dd >= 1) & (dd <= 31) & (station > 0)
        leap = ((2000 + yy) % 4 == 0) & (mm > 2)  # 2000 to 2099
        day = MONTH_STARTS[np.clip(mm - 1, 0, 11)] + dd - 1 + leap
        return {
            "station": np.where(valid, station, 0),
            "year": np.where(valid, yy, -1),
            "day": np.where(valid, day, 0),
            "hour": np.where(valid, hour, 0),
            "clock": (marks == CLOCK_CHANGE).any(axis=1),
            "values": values,
        }


MONTH_STARTS = np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30])  # day of the year a month starts on


def _digits(cells: np.ndarray) -> np.ndarray:
    return (cells >= ord("0")) & (cells <= ord("9"))


def _integers(cells: np.ndarray) -> np.ndarray:
    """Right-aligned integers in a byte matrix, one per row, padded with spaces; "-1" reads -1."""
    digits = np.where(_digits(cells), cells - ord("0"), 0).astype(np.float64)
    number = (digits @ 10.0 ** np.arange(cells.shape[1] - 1, -1, -1)).astype(np.int64)
    return np.where((cells == ord("-")).any(axis=1), -number, number)


def _int(cell: bytes, blank: int) -> int:
    """A cell of a row that isn't fixed-width; `blank` if it isn't a number."""
    try:
        return int(cell.strip())
    except ValueError:
        return blank


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download BASt hourly traffic counts to CSV.")
    parser.add_argument("series", nargs="*", help="<station>/<variable>, e.g. 5113/KFZ; see --list")
    parser.add_argument("--start", help="ISO date or time, Berlin time unless an offset is given "
                        f"(default: {sources.CLI_DAYS} days before the end)")
    parser.add_argument("--end", help="ISO date or time, inclusive; a bare date means the whole day "
                        "(default: the end of the last published year)")
    parser.add_argument("--out", help="CSV file (default: stdout)")
    parser.add_argument("--list", metavar="SEARCH", nargs="?", const="",
                        help="show the stations, optionally only matching ones")
    args = parser.parse_args(argv)

    if args.list is not None or not args.series:
        words = (args.list or "").casefold().split()
        for s in stations():
            text = f"{s.id} {s.name} {s.road} {s.state} {s.direction1} {s.direction2}".casefold()
            if all(w in text for w in words):
                print(f"{s.id:>5}  {s.name:<36} {s.road:<8} {s.state}  {s.direction1} / {s.direction2}")
        return
    try:
        end = args.end or f"{max(years())}-12-31"
        start, end = sources.span(args.start, end, TZ)
        table = read(args.series, start, end)
    except (ValueError, BastError) as error:
        parser.error(str(error))
    if args.out:
        with open(args.out, "w") as out:
            sources.write_csv(table, out, TZ)
    else:
        sources.write_csv(table, sys.stdout, TZ)


if __name__ == "__main__":
    main()
