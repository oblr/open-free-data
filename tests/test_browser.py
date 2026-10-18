from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from datetime import datetime, timedelta

import numpy as np
import pytest

from free_data import browser, fingerprints
from free_data.browser import RequestError


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


def test_series_in_another_unit_and_step(smard_files):
    query = {"d": ["de-smard:410"], "end": ["2024-01-01T03:00"], "unit.de-smard:410": ["MW"]}
    columns, table, _ = browser.series(query)
    assert (columns[0]["unit"], columns[0]["native_unit"]) == ("MW", "MWh")
    assert table.values[:, 0].tolist() == [1.0, 2.0]  # hourly: an MWh per hour is an MW
    columns, table, _ = browser.series({**query, "unit.de-smard:410": ["GWh"], "step": ["day"]})
    assert columns[0]["unit"] == "GWh" and table.values[:, 0].tolist() == [0.003]
    assert browser.to_csv(columns, table).decode().splitlines()[0].endswith("[GWh]")


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ({"unit.de-smard:410": ["°C"]}, "can't convert MWh to °C"),
        ({"unit.de-smard:4068": ["MW"]}, "not picked"),
        ({"step": ["week"]}, "step must be one of"),
    ],
)
def test_bad_transformations(extra, message):
    with pytest.raises(RequestError, match=message):
        browser.series({"d": ["de-smard:410"], **extra})

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
    assert status == 200 and b'<script type="module" src="dist/main.js">' in body
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


def test_compiled_page_is_served_and_nothing_else(server):
    status, body = get(f"{server}/dist/main.js")
    assert status == 200 and b'from "./api.js"' in body
    assert get(f"{server}/dist/plugins/de-smard.js")[0] == 200
    assert get(f"{server}/dist/../browser.py")[0] == 404  # urllib keeps the dots; the server must not follow them
    assert get(f"{server}/dist/%2E%2E/browser.py")[0] == 404
    assert get(f"{server}/dist/missing.js")[0] == 404


def test_every_source_module_is_compiled():
    """web/dist is committed so the browser works without Node; it must match web/src."""
    src = {p.relative_to(browser.WEB / "src").with_suffix(".js") for p in (browser.WEB / "src").rglob("*.ts")}
    dist = {p.relative_to(browser.DIST) for p in browser.DIST.rglob("*.js")}
    assert src == dist


def test_vocabulary_and_related_endpoints(server, smard_files, pegelonline_stations, hubeau_files, dwd_files,
                                          uba_files):
    status, body = get(f"{server}/api/vocabulary")
    terms = json.loads(body)
    assert status == 200 and terms["variables"]["electricity.generation.solar"]["standard"] == "entsoe:B16"
    assert terms["regions"]["DE4"]["name"] == "Brandenburg" and terms["countries"]["FR"]["name"] == "France"
    assert terms["units"]["°"]["spellings"] == ["Grad", "deg", "degree"]
    assert terms["dimensions"]["energy"] == {"label": "Energy", "base": "MWh", "rate": "power"}
    assert terms["measures"]["level"]["units"] == ["m", "m+NN", "m+NHN"]
    status, body = get(f"{server}/api/related?d=de-smard:4068")
    found = {r["key"]: r for r in json.loads(body)}
    assert status == 200
    assert found["de-smard:125"]["relation"] == "forecast"
    assert found["fr-rte-eco2mix:solaire"]["reason"] == "Solar in Mainland France"
    assert found["de-dwd-opendata:03987/SD_SO"]["relation"] == "driver"
    assert get(f"{server}/api/related?d=de-smard:nope")[0] == 400
    assert get(f"{server}/api/related")[0] == 400


def test_fingerprints_endpoint_reads_the_store(server):
    window = fingerprints.Window("2025-01-01", "2025-12-31")
    fingerprints.Store().save(fingerprints.Fingerprint(
        "de-smard:410", window, {"coverage": {"version": 1, "value": 0.5}}))
    status, body = get(f"{server}/api/fingerprints?d=de-smard:410&d=de-smard:4068")
    found = json.loads(body)
    assert status == 200 and list(found) == ["de-smard:410"]  # 4068 has none: left out, not computed
    assert found["de-smard:410"]["window"] == {"start": "2025-01-01", "end": "2025-12-31", "step": "hour"}
    assert found["de-smard:410"]["features"]["coverage"]["value"] == 0.5
    assert get(f"{server}/api/fingerprints")[0] == 400
    assert get(f"{server}/api/fingerprints?d=..:x")[0] == 400  # not under the store


def test_all_fingerprints_and_similar_endpoints(server):
    window = fingerprints.Window("2025-01-01", "2025-12-31")
    hours = np.arange(24)
    for key, peak in [("de-smard:410", 12), ("de-smard:4068", 14)]:
        fingerprints.Store().save(fingerprints.Fingerprint(key, window, {
            "profile.day": {"version": 1, "value": np.cos(2 * np.pi * (hours - peak) / 24).tolist()},
            "cycle.strength": {"version": 1, "value": {"day": 0.5}}}))
    status, body = get(f"{server}/api/fingerprints/all")
    assert status == 200 and {f["key"] for f in json.loads(body)} == {"de-smard:410", "de-smard:4068"}
    status, body = get(f"{server}/api/similar?d=de-smard:410")
    found = json.loads(body)
    assert status == 200 and found[0]["key"] == "de-smard:4068" and found[0]["day"]["lag"] == 2
    assert get(f"{server}/api/similar?d=de-smard:1")[0] == 400  # no fingerprint
    assert get(f"{server}/api/similar")[0] == 400
