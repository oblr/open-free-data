"""Share and index prices from Yahoo Finance, read through yahoo.py; for personal use only."""

from __future__ import annotations

from datetime import datetime

from free_data import yahoo
from free_data.sources import Dataset, Option, Place, Plugin, Table

KINDS = {"share": "Shares", "index": "Indices", "fund": "Funds"}


class YahooFinance(Plugin):
    id = yahoo.SOURCE
    title = "Yahoo Finance"
    timezone = "Europe/Berlin"
    options = (
        Option("interval", "Interval", yahoo.INTERVALS, "1d",
               help=f"Daily bars on trading days, or hourly ones for the last {yahoo.HOURLY_DAYS} days."),
    )

    def datasets(self) -> list[Dataset]:
        """One per symbol and field, grouped by kind, exchange and name."""
        rows = []
        for s in yahoo.symbols():
            index = s.kind == "index"
            for field, label in yahoo.FIELDS.items():
                volume = field == "volume"
                unit = "count" if volume else "pt" if index else s.currency
                what = ("volume as Yahoo reports it" if index else "shares traded") if volume else (
                    f"{label.lower()} in {'index points' if index else s.currency}")
                rows.append(Dataset(
                    id=f"{s.symbol}/{field}",
                    label=label,
                    group=f"{KINDS[s.kind]} / {s.exchange} / {s.name}",
                    unit=unit,
                    description=(f"{s.name} ({s.symbol}, {s.exchange}): {what} per trading day or hour. "
                                 + ("" if index or volume else
                                    "Adjusted for splits and dividends. " if field == "adjclose" else
                                    "Adjusted for splits. ")
                                 + "Yahoo Finance, for personal use only."),
                    link=f"https://finance.yahoo.com/quote/{s.symbol}",
                    variable=("finance.trading_volume" if volume else
                              "finance.index_level" if index else "finance.share_price"),
                    measure="count" if volume else "index" if index else "price",
                    place=Place(country=s.country, name=s.exchange),
                    resolution="1d",
                    since=s.since,
                    quality="raw",
                ))
        return rows

    def read(
        self, datasets: list[str], start: datetime | None, end: datetime | None, options: dict[str, str]
    ) -> Table:
        return yahoo.read(datasets, start, end, options["interval"])


PLUGIN = YahooFinance()
