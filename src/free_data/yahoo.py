"""Reader for share and index prices from Yahoo Finance's chart endpoint.

Daily and hourly bars of shares, indices and funds on most exchanges:
Xetra as "SAP.DE", Euronext Paris as "TTE.PA", indices as "^GDAXI".

This is no open source (see its catalog entry, data/us-yahoo-finance.json):
the endpoint is the one Yahoo's own pages use, not a supported API, Yahoo's
terms allow personal use only, and how many requests it allows isn't
documented. cache.HOST_RATES keeps it to one request every two seconds.

    https://query1.finance.yahoo.com/v8/finance/chart/SAP.DE?period1=<unix>&period2=<unix>&interval=1d
        -> {"chart": {"result": [{"meta": {"currency": "EUR", "exchangeTimezoneName": "Europe/Berlin", ...},
                                  "timestamp": [...],
                                  "indicators": {"quote": [{"open": [...], "high": [...], "low": [...],
                                                            "close": [...], "volume": [...]}],
                                                 "adjclose": [{"adjclose": [...]}]}}],
                      "error": null}}

A series is named "<symbol>/<field>", e.g. "SAP.DE/close", with the fields
of FIELDS. "close" is adjusted for splits, "adjclose" also for dividends.
Both change in hindsight: after a split every earlier price is rescaled,
after a dividend every earlier adjusted close.

Daily bars are trading days, labelled with midnight in the exchange's
timezone; weekends and holidays have no row. Hourly bars are labelled with
the start of the hour and only exist for the last HOURLY_DAYS days.

Files are kept small, a year of daily or a month of hourly bars: Yahoo
often cuts off longer answers. What is downloaded, and for how long it is
kept:

    daily bars, one calendar year per file      the current year 6 hours, earlier
                                                years a week (splits and
                                                dividends rescale them)
    hourly bars, one calendar month per file    the current month 1 hour, earlier
                                                months a week
    a symbol's description (not in SYMBOLS)     a week

Months and years are those of Berlin, close enough for every European
exchange.

SYMBOLS (yahoo_symbols.json) lists what the browser offers: the large German
and French shares and their indices. FREE_DATA_YAHOO_SYMBOLS (comma
separated) adds more, and read() takes any symbol Yahoo knows.

Times are returned as UTC `datetime64[ms]`. Date arguments without a UTC
offset are read as Berlin time, and the CSV output is in Berlin time.

    python -m free_data.yahoo --list dax
    python -m free_data.yahoo SAP.DE/close TTE.PA/close --start 2024-01-01 --end 2024-12-31
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time as clock
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import quote
from zoneinfo import ZoneInfo

import numpy as np

from free_data import sources
from free_data.cache import DAY, HOUR, Cache, FetchError, Pool
from free_data.sources import Table, Time

HOST = "query1.finance.yahoo.com"
BASE = f"https://{HOST}/v8/finance/chart"
SOURCE = "us-yahoo-finance"  # the id in data/ and the cache folder
CURRENT_MAX_AGE = 6 * HOUR
PAST_MAX_AGE = 7 * DAY
HOURLY_MAX_AGE = HOUR
META_MAX_AGE = 7 * DAY
HOURLY_DAYS = 729  # Yahoo keeps hourly bars for 730 days and refuses a span reaching further back
DOWNLOADS = 2
TZ = ZoneInfo("Europe/Berlin")
INTERVALS = ("1d", "1h")
FIELDS = {  # field -> label; prices are in the symbol's currency (index points for an index)
    "close": "Close",
    "adjclose": "Adjusted close",
    "open": "Open",
    "high": "High",
    "low": "Low",
    "volume": "Volume",
}
SYMBOLS_FILE = Path(__file__).with_name("yahoo_symbols.json")
EXTRA = "FREE_DATA_YAHOO_SYMBOLS"


class YahooError(RuntimeError):
    pass


@dataclass(frozen=True)
class Symbol:
    symbol: str  # "SAP.DE"
    name: str  # "SAP SE"
    kind: str  # "share", "index" or "fund"
    exchange: str  # "Xetra"
    currency: str  # "EUR"
    timezone: str  # the exchange's, "Europe/Berlin"
    country: str = ""  # ISO 3166-1 alpha-2; empty for an index across countries
    since: str = ""  # the first trading day Yahoo has, "1998-04-09"


def symbols(cache: Path | None = None) -> list[Symbol]:
    """The symbols on offer: SYMBOLS, then those named in $FREE_DATA_YAHOO_SYMBOLS."""
    listed = [Symbol(**row) for row in json.loads(SYMBOLS_FILE.read_text())]
    known = {s.symbol for s in listed}
    extra = [s.strip() for s in os.environ.get(EXTRA, "").split(",") if s.strip()]
    return listed + [describe(s, cache) for s in dict.fromkeys(extra) if s not in known]


def describe(symbol: str, cache: Path | None = None) -> Symbol:
    """What Yahoo says about a symbol: name, exchange, currency, timezone."""
    url = f"{BASE}/{quote(symbol)}?range=5d&interval=1d"
    meta = _result(_fetch(Cache(SOURCE, cache), url, f"{symbol}/meta.json", META_MAX_AGE, symbol))["meta"]
    zone = meta.get("exchangeTimezoneName") or "UTC"
    first = meta.get("firstTradeDate")
    kinds = {"EQUITY": "share", "INDEX": "index", "ETF": "fund", "MUTUALFUND": "fund"}
    return Symbol(
        symbol=meta.get("symbol", symbol), name=(meta.get("longName") or meta.get("shortName") or symbol).strip(),
        kind=kinds.get(meta.get("instrumentType", ""), "share"), exchange=meta.get("fullExchangeName", ""),
        currency=meta.get("currency") or "", timezone=zone,
        since=str(datetime.fromtimestamp(first, ZoneInfo(zone)).date()) if first is not None else "",
    )


def read(
    series: list[str] | tuple[str, ...],
    start: Time = None,
    end: Time = None,
    interval: str = "1d",
    cache: Path | None = None,
) -> Table:
    """The series ("<symbol>/<field>") between `start` and `end` (both inclusive).

    Without a start, daily bars begin a year before the end; hourly bars
    can't go back further than HOURLY_DAYS days.
    """
    if interval not in INTERVALS:
        raise ValueError(f"interval must be one of {INTERVALS}, got {interval!r}")
    now = _now()
    hi = sources.epoch_ms(end, TZ)
    hi = now if hi is None else min(hi, now)
    lo = sources.epoch_ms(start, TZ)
    if lo is None:
        lo = hi - 365 * 86_400_000
    if lo > hi:
        raise ValueError(f"start {start} is after end {end}")
    if interval == "1h":
        lo = max(lo, now - HOURLY_DAYS * 86_400_000)
    wanted = [_parse(name) for name in series]
    store = Cache(SOURCE, cache)
    names = list(dict.fromkeys(symbol for symbol, _ in wanted))
    jobs = [(symbol, period) for symbol in names for period in _periods(lo, hi, interval)]
    with Pool(DOWNLOADS) as pool:
        bars = dict(zip(jobs, pool.map(lambda job: _bars(store, *job, now=now), jobs), strict=True))
    columns = []
    for name, (symbol, field) in zip(series, wanted, strict=True):
        parts = [bars[job] for job in jobs if job[0] == symbol]
        time = np.concatenate([p[0] for p in parts] or [np.array([], np.int64)])
        values = np.concatenate([p[1][field] for p in parts] or [np.array([])])
        keep = (time >= lo) & (time <= hi)
        columns.append(Table((name,), time[keep].astype("datetime64[ms]"), values[keep].reshape(-1, 1)))
    combined = sources.combine(columns)
    return Table(tuple(series), combined.time, combined.values)


def _parse(name: str) -> tuple[str, str]:
    symbol, slash, field = name.rpartition("/")
    if not (symbol and slash and field):
        raise ValueError(f"a series is named '<symbol>/<field>', e.g. 'SAP.DE/close', not {name!r}")
    if field not in FIELDS:
        raise ValueError(f"{name}: the field is one of {', '.join(FIELDS)}")
    return symbol, field


@dataclass(frozen=True)
class Period:
    """What one file holds: daily bars of a year (month 0) or hourly bars of a month."""

    interval: str
    year: int
    month: int = 0

    @property
    def key(self) -> str:
        return f"{self.interval}/{self.year}" + (f"-{self.month:02d}" if self.month else "")

    def bounds(self) -> tuple[int, int]:
        """Its start and the next one's, epoch seconds, in Berlin."""
        if not self.month:
            return (int(datetime(self.year, 1, 1, tzinfo=TZ).timestamp()),
                    int(datetime(self.year + 1, 1, 1, tzinfo=TZ).timestamp()))
        after = (self.year + self.month // 12, self.month % 12 + 1)
        return (int(datetime(self.year, self.month, 1, tzinfo=TZ).timestamp()),
                int(datetime(*after, 1, tzinfo=TZ).timestamp()))

    def holds(self, ms: int) -> bool:
        when = datetime.fromtimestamp(ms / 1000, TZ)
        return when.year == self.year and (not self.month or when.month == self.month)


def _periods(lo: int, hi: int, interval: str) -> list[Period]:
    """The years (daily bars) or months (hourly bars) from `lo` to `hi`."""
    first, last = (datetime.fromtimestamp(ms / 1000, TZ) for ms in (lo, hi))
    if interval == "1d":
        return [Period("1d", year) for year in range(first.year, last.year + 1)]
    months = range(first.year * 12 + first.month - 1, last.year * 12 + last.month)
    return [Period("1h", m // 12, m % 12 + 1) for m in months]


def _bars(store: Cache, symbol: str, period: Period, now: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """One symbol's bars in one period: epoch ms and the values of every field."""
    first, last = period.bounds()
    # a few days either side, so a bar labelled in another timezone than Berlin isn't lost at the edge
    first, last = first - 3 * 86_400, last + 3 * 86_400
    if period.interval == "1h":
        first = max(first, now // 1000 - HOURLY_DAYS * 86_400)  # Yahoo refuses anything older
    if not period.holds(now):
        max_age = PAST_MAX_AGE
    else:
        max_age = CURRENT_MAX_AGE if period.interval == "1d" else HOURLY_MAX_AGE
    url = f"{BASE}/{quote(symbol)}?period1={first}&period2={last}&interval={period.interval}"
    result = _result(_fetch(store, url, f"{symbol}/{period.key}.json", max_age, symbol))
    time, values = _columns(result, daily=period.interval == "1d")
    keep = np.array([period.holds(int(t)) for t in time], bool)  # the margins belong to the neighbours' files
    return time[keep], {f: v[keep] for f, v in values.items()}


def _fetch(store: Cache, url: str, key: str, max_age: float, symbol: str) -> dict:
    try:
        body = store.fetch(url, key, max_age)
    except FetchError as error:
        if error.status in (400, 404, 422):
            raise YahooError(f"Yahoo has no data for {symbol!r} (HTTP {error.status}): {error}") from error
        raise
    try:
        return json.loads(body)
    except ValueError as error:
        raise YahooError(f"Yahoo's answer for {symbol!r} isn't JSON; the unofficial API may have changed") from error


def _result(body: dict) -> dict:
    chart = body.get("chart") if isinstance(body, dict) else None
    if not isinstance(chart, dict):
        raise YahooError("Yahoo's answer has no 'chart'; the unofficial API may have changed")
    if chart.get("error"):
        error = chart["error"]
        raise YahooError(f"Yahoo: {error.get('description') or error.get('code') or error}")
    results = chart.get("result") or []
    if not results:
        raise YahooError("Yahoo answered without a result")
    return results[0]


def _columns(result: dict, daily: bool) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """The bars of one chart result: epoch ms, sorted and unique, and the values of every field, NaN for null.

    Daily bars are labelled with midnight in the exchange's timezone. When
    two bars land on one label (Yahoo adds today's bar while it is still
    trading), the later one wins.
    """
    stamps = result.get("timestamp") or []
    indicators = result.get("indicators") or {}
    quote_ = (indicators.get("quote") or [{}])[0]
    adjusted = (indicators.get("adjclose") or [{}])[0].get("adjclose")
    raw = {f: quote_.get(f) for f in FIELDS if f != "adjclose"}
    raw["adjclose"] = adjusted if adjusted is not None else raw["close"]  # hourly bars have no adjusted close
    if daily:
        zone = ZoneInfo(result.get("meta", {}).get("exchangeTimezoneName") or "UTC")
        time = np.array([_midnight(s, zone) for s in stamps], np.int64)
    else:
        time = np.array(stamps, np.int64) * 1000
    values = {}
    for field, column in raw.items():
        column = column if column is not None else [None] * len(stamps)
        if len(column) != len(stamps):
            raise YahooError(f"Yahoo's {field} has {len(column)} values for {len(stamps)} times")
        values[field] = np.array([np.nan if v is None else v for v in column], np.float64)
    order = np.argsort(time, kind="stable")
    time = time[order]
    last = np.append(time[1:] != time[:-1], True) if len(time) else np.array([], bool)
    return time[last], {f: v[order][last] for f, v in values.items()}


def _midnight(stamp: int, zone: ZoneInfo) -> int:
    day = datetime.fromtimestamp(stamp, zone).date()
    return int(datetime(day.year, day.month, day.day, tzinfo=zone).timestamp() * 1000)


def _now() -> int:
    return int(clock.time() * 1000)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Download share and index prices from Yahoo Finance to CSV. For personal use only: "
                    "Yahoo's terms don't allow passing the data on.")
    parser.add_argument("series", nargs="*", help="<symbol>/<field>, e.g. SAP.DE/close; see --list")
    parser.add_argument("--interval", default="1d", choices=INTERVALS,
                        help=f"daily bars, or hourly ones for the last {HOURLY_DAYS} days")
    parser.add_argument("--start", help="ISO date or time, Berlin time unless an offset is given "
                        "(default: a year before the end)")
    parser.add_argument("--end", help="ISO date or time, inclusive; a bare date means the whole day (default: now)")
    parser.add_argument("--out", help="CSV file (default: stdout)")
    parser.add_argument("--list", metavar="SEARCH", nargs="?", const="",
                        help="show the symbols on offer, optionally only matching ones")
    args = parser.parse_args(argv)

    if args.list is not None or not args.series:
        words = (args.list or "").casefold().split()
        for s in symbols():
            text = f"{s.symbol} {s.name} {s.kind} {s.exchange} {s.country}".casefold()
            if all(w in text for w in words):
                print(f"{s.symbol:<11} {s.name[:44]:<44} {s.kind:<6} {s.exchange:<15} {s.currency:<4} since {s.since}")
        print(f"fields: {', '.join(FIELDS)}", file=sys.stderr)
        return
    try:
        start, end = sources.span(args.start, args.end, TZ, days=365)
        table = read(args.series, start, end, args.interval)
    except (ValueError, YahooError) as error:
        parser.error(str(error))
    notes = [sources.notice(SOURCE)]
    if args.out:
        print(notes[0], file=sys.stderr)
        with open(args.out, "w") as out:
            sources.write_csv(table, out, TZ, notes)
    else:
        sources.write_csv(table, sys.stdout, TZ, notes)


if __name__ == "__main__":
    main()
