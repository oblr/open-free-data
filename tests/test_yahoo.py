"""Yahoo Finance against chart answers put into the cache; "now" is 2025-01-10 12:00 UTC."""

from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime

import numpy as np
import pytest

from conftest import put
from free_data import cache, sources, yahoo
from free_data.cache import FetchError
from free_data.plugins.us_yahoo_finance import PLUGIN
from free_data.yahoo import YahooError

NOW = datetime(2025, 1, 10, 12, tzinfo=UTC)


def unix(text: str) -> int:
    return int(datetime.fromisoformat(text).timestamp())


def ms(text: str) -> int:
    return unix(text) * 1000


def chart(stamps: list[str], close: list, *, zone: str = "Europe/Berlin", adjclose: list | None = None,
          volume: list | None = None) -> dict:
    n = len(stamps)
    indicators = {"quote": [{"open": close, "high": close, "low": close, "close": close,
                             "volume": volume if volume is not None else [100] * n}]}
    if adjclose is not None:
        indicators["adjclose"] = [{"adjclose": adjclose}]
    return {"chart": {"result": [{"meta": {"symbol": "SAP.DE", "currency": "EUR", "exchangeTimezoneName": zone},
                                  "timestamp": [unix(s) for s in stamps], "indicators": indicators}],
                      "error": None}}


@pytest.fixture
def now(monkeypatch):
    monkeypatch.setattr(yahoo, "_now", lambda: int(NOW.timestamp() * 1000))


@pytest.fixture
def daily(cache_root, now):
    """SAP's daily bars around new year: Yahoo labels them with the opening, 08:00 UTC."""
    put(cache_root, yahoo.SOURCE, "SAP.DE/1d/2024.json", chart(
        ["2024-12-27T08:00Z", "2024-12-30T08:00Z", "2025-01-02T08:00Z"],  # the last is a margin of 2025's file
        [239.5, 236.3, 999.0], adjclose=[230.0, 227.0, 999.0]))
    put(cache_root, yahoo.SOURCE, "SAP.DE/1d/2025.json", chart(
        ["2025-01-02T08:00Z", "2025-01-03T08:00Z", "2025-01-06T08:00Z", "2025-01-06T15:30Z"],
        [238.55, None, 239.0, 239.25], adjclose=[229.0, None, 230.0, 230.2], volume=[1, 2, 3, 4]))


def test_daily_bars_are_labelled_with_local_midnight(daily):
    table = yahoo.read(["SAP.DE/close", "SAP.DE/adjclose", "SAP.DE/volume"], "2024-12-27", "2025-01-06")
    assert table.columns == ("SAP.DE/close", "SAP.DE/adjclose", "SAP.DE/volume")
    assert table.time.astype(np.int64).tolist() == [
        ms(f"{d}T00:00+01:00") for d in ("2024-12-27", "2024-12-30", "2025-01-02", "2025-01-03", "2025-01-06")]
    # each year's margins are left to the other year's file; null is NaN; of two bars on one day the later wins
    np.testing.assert_array_equal(table.values[:, 0], [239.5, 236.3, 238.55, np.nan, 239.25])
    np.testing.assert_array_equal(table.values[:, 1], [230.0, 227.0, 229.0, np.nan, 230.2])
    np.testing.assert_array_equal(table.values[2:, 2], [1, 2, 4])


def test_the_span_is_cut_and_defaults_to_a_year(daily):
    table = yahoo.read(["SAP.DE/close"], "2024-12-30", "2025-01-02")
    assert table.values[:, 0].tolist() == [236.3, 238.55]
    assert len(yahoo.read(["SAP.DE/close"]).time) == 5  # a year back from now: both files


def test_the_current_year_is_fetched_again_and_older_years_kept(daily, cache_root, monkeypatch):
    urls = []

    def download(url):
        urls.append(url)
        return json.dumps(chart(["2025-01-02T08:00Z"], [1.0])).encode()

    monkeypatch.setattr(cache, "_download", download)
    for key in ("SAP.DE/1d/2024.json", "SAP.DE/1d/2025.json"):
        path = cache_root / yahoo.SOURCE / key
        then = time.time() - 2 * 86_400  # two days old (the cache's clock is the real one)
        os.utime(path, (then, then))
    yahoo.read(["SAP.DE/close"], "2024-12-27", "2025-01-06")
    assert len(urls) == 1 and "period1=" in urls[0] and "interval=1d" in urls[0]
    assert unix("2025-01-01T00:00+01:00") - 3 * 86_400 == int(urls[0].split("period1=")[1].split("&")[0])


def test_hourly_bars_by_month(cache_root, now):
    put(cache_root, yahoo.SOURCE, "SAP.DE/1h/2025-01.json", chart(
        ["2024-12-31T12:00Z", "2025-01-09T08:00Z", "2025-01-09T09:00Z"], [1.0, 2.0, 3.0]))
    table = yahoo.read(["SAP.DE/close", "SAP.DE/adjclose"], "2025-01-09", "2025-01-09T23:59", interval="1h")
    assert table.time.astype(np.int64).tolist() == [ms("2025-01-09T08:00Z"), ms("2025-01-09T09:00Z")]
    assert table.values.tolist() == [[2.0, 2.0], [3.0, 3.0]]  # hourly bars have no adjusted close: the close


def test_hourly_bars_only_for_the_last_two_years(cache_root, now, monkeypatch):
    urls = []
    monkeypatch.setattr(cache, "_download", lambda url: urls.append(url) or json.dumps(chart([], [])).encode())
    yahoo.read(["SAP.DE/close"], "2020-01-01", "2023-01-20", interval="1h")
    first = int(urls[0].split("period1=")[1].split("&")[0])
    assert first == NOW.timestamp() - yahoo.HOURLY_DAYS * 86_400  # nothing older is asked for
    assert len(urls) == 1  # just January 2023


def test_errors_name_the_symbol(cache_root, now, monkeypatch):
    put(cache_root, yahoo.SOURCE, "NOPE.DE/1d/2025.json",
        {"chart": {"result": None, "error": {"code": "Not Found", "description": "No data found"}}})
    with pytest.raises(YahooError, match="No data found"):
        yahoo.read(["NOPE.DE/close"], "2025-01-02", "2025-01-03")

    def missing(url):
        raise FetchError(url, 404, f"HTTP 404 for {url}")

    monkeypatch.setattr(cache, "_download", missing)
    with pytest.raises(YahooError, match="no data for 'XX.DE'"):
        yahoo.read(["XX.DE/close"], "2025-01-02", "2025-01-03")
    put(cache_root, yahoo.SOURCE, "ODD.DE/1d/2025.json", {"spark": {}})
    with pytest.raises(YahooError, match="unofficial API may have changed"):
        yahoo.read(["ODD.DE/close"], "2025-01-02", "2025-01-03")


@pytest.mark.parametrize(("name", "message"), [("SAP.DE", "<symbol>/<field>"), ("SAP.DE/bid", "the field is one of")])
def test_bad_names(now, name, message):
    with pytest.raises(ValueError, match=message):
        yahoo.read([name])


def test_the_plugin_offers_every_listed_symbol_and_field():
    datasets = {d.id: d for d in PLUGIN.datasets()}
    assert len(datasets) == len(yahoo.symbols()) * len(yahoo.FIELDS)
    sap, dax, volume = datasets["SAP.DE/close"], datasets["^GDAXI/close"], datasets["SAP.DE/volume"]
    assert (sap.unit, sap.variable, sap.measure, sap.place.country) == ("EUR", "finance.share_price", "price", "DE")
    assert (dax.unit, dax.variable, dax.group) == ("pt", "finance.index_level", "Indices / Xetra / DAX")
    assert (volume.unit, volume.measure) == ("count", "count")
    assert datasets["^STOXX50E/close"].place.country == ""  # across countries
    assert sap.since == "1998-04-09" and sap.group == "Shares / Xetra / SAP SE"


def test_more_symbols_from_the_environment(cache_root, monkeypatch):
    put(cache_root, yahoo.SOURCE, "NESN.SW/meta.json", {"chart": {"result": [{"meta": {
        "symbol": "NESN.SW", "longName": "Nestlé S.A.", "instrumentType": "EQUITY", "fullExchangeName": "Swiss",
        "currency": "CHF", "exchangeTimezoneName": "Europe/Zurich", "firstTradeDate": 946886400}}], "error": None}})
    monkeypatch.setenv(yahoo.EXTRA, "NESN.SW, SAP.DE")  # one already listed
    extra = [s for s in yahoo.symbols() if s.symbol == "NESN.SW"]
    assert [(s.name, s.currency, s.timezone, s.since) for s in extra] == [
        ("Nestlé S.A.", "CHF", "Europe/Zurich", "2000-01-03")]
    assert sum(s.symbol == "SAP.DE" for s in yahoo.symbols()) == 1


def test_cli_writes_the_constraints_into_the_csv(daily, capsys):
    yahoo.main(["SAP.DE/close", "--start", "2024-12-30", "--end", "2025-01-02"])
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == f"# {sources.notice(yahoo.SOURCE)}"
    assert lines[0].startswith("# us-yahoo-finance: Its terms allow private use only")
    assert lines[1:] == ["timestamp,SAP.DE/close", "2024-12-30T00:00:00+01:00,236.3",
                         "2025-01-02T00:00:00+01:00,238.55"]
