from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from datetime import datetime, timedelta

import pytest

import browser
from browser import RequestError


def test_moment_reads_bare_dates_in_the_source_zone():
    start = browser.moment("2024-01-01", "Europe/Berlin", end=False)
    end = browser.moment("2024-01-01", "Europe/Berlin", end=True)
    assert start.utcoffset() == timedelta(hours=1)
    assert end - start == timedelta(days=1, microseconds=-1)  # a bare end date is the whole day


def test_moment_keeps_an_explicit_offset_and_rejects_junk():
    assert browser.moment("2024-01-01T05:00+00:00", "Europe/Berlin", end=True) == datetime.fromisoformat(
        "2024-01-01T05:00+00:00"
    )
    assert browser.moment("", "UTC", end=False) is None
    with pytest.raises(RequestError, match="not a date"):
        browser.moment("yesterday", "UTC", end=False)


@pytest.mark.parametrize(
    ("query", "message"),
    [
        ({}, "pick at least one"),
        ({"d": ["de-smard:nope"]}, "no dataset"),
        ({"d": ["zz-none:1"]}, "no plugin"),
        ({"d": ["de-smard:410"], "de-smard.region": ["Mars"]}, "region must be one of"),
        ({"d": ["de-smard:410"], "start": ["2024-01-02"], "end": ["2024-01-01"]}, "after the end"),
    ],
)
def test_bad_requests(query, message):
    with pytest.raises(RequestError, match=message):
        browser.series(query)


def test_series_and_csv(smard_files):
    columns, table, on_grid = browser.series({"d": ["de-smard:410"], "end": ["2024-01-01T03:00"]})
    assert [c["key"] for c in columns] == ["de-smard:410"]
    assert columns[0]["unit"] == "MWh"
    assert table.values[:, 0].tolist() == [1.0, 2.0]
    assert on_grid.all()
    lines = browser.to_csv(columns, table).decode().splitlines()
    assert lines[0].startswith("timestamp_utc,de-smard:410 ")
    assert lines[1] == "2024-01-01T01:00:00+00:00,1.0"


@pytest.fixture
def server():
    httpd = browser.ThreadingHTTPServer(("127.0.0.1", 0), browser.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def get(url: str) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        with error:
            return error.code, error.read()


def test_http_api(server, smard_files):
    status, body = get(f"{server}/")
    assert status == 200 and b"<html" in body.lower()
    status, body = get(f"{server}/api/sources")
    assert status == 200 and any(s["id"] == "de-smard" and s["options"] for s in json.loads(body))
    status, body = get(f"{server}/api/sources/de-smard/datasets")
    assert status == 200 and "410" in [d["id"] for d in json.loads(body)]
    status, body = get(f"{server}/api/series?d=de-smard:410&end=2024-01-01T03:00")
    data = json.loads(body)
    assert status == 200 and data["values"] == [[1.0, 2.0]] and data["on_grid"] == ["11"]
    status, body = get(f"{server}/api/series?d=de-smard:410&format=csv&end=2024-01-01T03:00")
    assert status == 200 and body.startswith(b"timestamp_utc,")
    assert get(f"{server}/api/series")[0] == 400
    assert get(f"{server}/nope")[0] == 404
