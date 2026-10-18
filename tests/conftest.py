"""Shared fixtures: a temporary cache and no network.

Every test runs against its own cache folder ($FREE_DATA_CACHE) and fails if
anything tries to download; tests that want a download patch
cache._download themselves. Requests aren't throttled.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from free_data import cache

HOUR_MS = 3_600_000
T0 = 1_704_067_200_000  # 2024-01-01T00:00Z


# Two PEGELONLINE stations, as its station list describes them.
STATIONS = [
    {"uuid": "u-koeln", "number": "2730010", "shortname": "KÖLN", "km": 688.0, "agency": "STANDORT KÖLN",
     "water": {"shortname": "RHEIN", "longname": "RHEIN"},
     "timeseries": [
         {"shortname": "W", "longname": "WASSERSTAND ROHDATEN", "unit": "cm", "equidistance": 15,
          "gaugeZero": {"unit": "m. ü. NHN", "value": 35.038, "validFrom": "2019-11-01"}},
         {"shortname": "Q", "longname": "ABFLUSS_ROHDATEN", "unit": "m³/s", "equidistance": 15},
     ]},
    {"uuid": "u-bonn", "number": "2710080", "shortname": "BONN", "km": 654.8, "agency": "STANDORT BONN",
     "water": {"shortname": "RHEIN", "longname": "RHEIN"},
     "timeseries": [{"shortname": "XYZ", "longname": "SICHTWEITE", "unit": "m", "equidistance": 60}]},
]


class Unthrottled(cache.Throttle):
    def wait(self, url: str) -> float:
        return 0.0

    def slow_down(self, url: str, retry_after: float | None) -> None:
        pass


@pytest.fixture(autouse=True)
def cache_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "cache"
    monkeypatch.setenv("FREE_DATA_CACHE", str(root))
    monkeypatch.delenv("FREE_DATA_YAHOO_SYMBOLS", raising=False)  # more symbols would be looked up online

    def offline(url: str) -> bytes:
        raise AssertionError(f"unexpected download: {url}")

    monkeypatch.setattr(cache, "_download", offline)
    monkeypatch.setattr(cache, "RETRY_PAUSES", (0.0, 0.0))
    monkeypatch.setattr(cache, "THROTTLE", Unthrottled())  # tests that time it make their own
    return root


def put(root: Path, source: str, key: str, body: object) -> Path:
    """A cached file, fresh as of now."""
    path = root / source / key
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body))
    return path


@pytest.fixture
def smard_files(cache_root: Path) -> dict[str, list[float | None]]:
    """Two hourly chunks of filter 410 in region DE, overlapping at T0+3h.

    chunk A (T0):    T0 pad, 1, 2, 3
    chunk B (T0+3h): 30 (overrides A), gap, 5, trailing pad
    """
    a, b = T0, T0 + 3 * HOUR_MS
    chunks = {
        a: [[a + i * HOUR_MS, v] for i, v in enumerate([None, 1.0, 2.0, 3.0])],
        b: [[b + i * HOUR_MS, v] for i, v in enumerate([30.0, None, 5.0, None])],
    }
    folder = "410/DE/hour"
    put(cache_root, "de-smard", f"{folder}/index_hour.json", {"timestamps": [b, a]})
    for chunk, series in chunks.items():
        put(cache_root, "de-smard", f"{folder}/410_DE_hour_{chunk}.json", {"meta_data": {}, "series": series})
    return chunks


@pytest.fixture
def pegelonline_stations(cache_root: Path) -> list[dict]:
    put(cache_root, "de-pegelonline", "stations.json", STATIONS)
    return STATIONS


# ---- Hub'eau: "now" is 2024-01-10 12:00 Paris ----

def paris_ms(text: str) -> int:
    from free_data.sources import epoch_ms
    return epoch_ms(text, "Europe/Paris")


def utc_stamp(ms: int) -> str:
    import numpy as np
    return str(np.datetime64(ms, "ms").astype("datetime64[s]")) + "Z"


HUBEAU_STATIONS = [
    {"code_station": "F700000103", "libelle_station": "La Seine à Paris - Austerlitz", "type_station": "STD",
     "libelle_cours_eau": "La Seine", "libelle_commune": "PARIS", "code_departement": "75", "code_region": "11",
     "date_ouverture_station": "2006-01-26T00:00:00Z", "longitude_station": 2.3655, "latitude_station": 48.8447},
    {"code_station": "D0000001", "libelle_station": "Débit seul", "type_station": "DEB",
     "libelle_cours_eau": None, "libelle_commune": None, "code_departement": "59", "date_ouverture_station": None},
    {"code_station": "C0000001", "libelle_station": "Calculée", "type_station": "FICTIVE",
     "libelle_cours_eau": "La Seine", "libelle_commune": "PARIS", "code_departement": "75",
     "date_ouverture_station": None},
]


@pytest.fixture
def hubeau_files(cache_root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Two pages of stations; real-time H and Q every 30 min from 2024-01-08 (newest first);
    daily mean flows 2023-12-30 to 2024-01-08."""
    from free_data import hubeau
    monkeypatch.setattr(hubeau, "_now", lambda: paris_ms("2024-01-10T12:00"))
    source = hubeau.SOURCE
    pages = f"stations/{hubeau._fields_key()}"
    put(cache_root, source, f"{pages}-1.json", {"data": HUBEAU_STATIONS[:2], "next": "page 2"})
    # overlapping, as pages cached at different times can
    put(cache_root, source, f"{pages}-2.json", {"data": HUBEAU_STATIONS[1:], "next": None})
    start = paris_ms("2024-01-08T00:00")
    times = [start + i * 1_800_000 for i in range(96)]  # two days
    for kind, base in (("Q", 100_000.0), ("H", 1_000.0)):
        rows = [{"date_obs": utc_stamp(t), "resultat_obs": base + i} for i, t in enumerate(times)]
        put(cache_root, source, f"F700000103/{kind}.json", {"data": rows[::-1]})
    days = [f"2023-12-{d}" for d in (30, 31)] + [f"2024-01-0{d}" for d in range(1, 9)]
    put(cache_root, source, "F700000103/QmnJ/2020.json",
        {"data": [{"date_obs_elab": d, "resultat_obs_elab": 50_000.0 + i} for i, d in enumerate(days)]})
    return cache_root / source


# ---- DWD: "now" is 2024-01-10 12:00 UTC ----

DWD_GROUPS = {"air_temperature": "TU", "precipitation": "RR", "wind": "FF", "sun": "SD", "pressure": "P0"}
DWD_COLUMNS = {"TU": "QN_9;TT_TU;RF_TU", "RR": "QN_8;  R1;RS_IND;WRTR", "FF": "QN_3;   F;   D",
               "SD": "QN_7;SD_SO", "P0": "QN_8;   P;  P0"}


def dwd_zip(code: str, rows: list[tuple[str, list[float]]]) -> bytes:
    """A DWD zip: some metadata and produkt_*.txt with `rows` of (YYYYMMDDHH, values)."""
    import io
    import zipfile
    lines = [f"STATIONS_ID;MESS_DATUM;{DWD_COLUMNS[code]};eor"]
    lines += [f"      3987;{stamp};    3;" + ";".join(f"{v:7.1f}" for v in values) + ";eor" for stamp, values in rows]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("Metadaten_Geographie_03987.txt", "x")
        archive.writestr(f"produkt_{code.lower()}_stunde_03987.txt", "\r\n".join(lines).encode("latin-1"))
    return buffer.getvalue()


@pytest.fixture
def dwd_files(cache_root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Potsdam (03987) reports in every group; Aachen (00003) only in its station list.

    Air temperature: a historical file to 2024-01-01 00:00 UTC and a recent
    one from that hour, which overlap there. Wind: recent only."""
    from free_data import dwd
    monkeypatch.setattr(dwd, "_now", lambda: 1_704_888_000_000)
    folder = cache_root / dwd.SOURCE
    header = ("Stations_id von_datum bis_datum Stationshoehe geoBreite geoLaenge Stationsname Bundesland Abgabe\r\n"
              "----------- --------- --------- ------------- --------- --------- "
              "---------------- ---------- ------\r\n")
    for group, code in DWD_GROUPS.items():
        (folder / group / "recent").mkdir(parents=True)
        (folder / group / "historical").mkdir(parents=True)
        (folder / group / "stations.txt").write_bytes((header +
            "00003 19500401 20110331            202     50.7827    6.0941 Aachen                     "
            "Nordrhein-Westfalen        \r\n"
            "03987 18930101 20240109             81     52.3813   13.0622 Potsdam Telegrafenberg     "
            "Brandenburg                Frei\r\n").encode("latin-1"))
        (folder / group / "recent.html").write_text(f'<a href="stundenwerte_{code}_03987_akt.zip">x</a>')
        (folder / group / "historical.html").write_text(
            '<a href="stundenwerte_TU_03987_18930101_20240101_hist.zip">x</a>' if code == "TU" else "")
    hist = [("2023123122", [1.0, 80.0]), ("2023123123", [-999.0, 81.0]), ("2024010100", [3.0, 82.0])]
    recent = [("2024010100", [99.0, 99.0])] + [(f"20240101{h:02d}", [10.0 + h, 90.0]) for h in range(1, 24)]
    (folder / "air_temperature/historical/stundenwerte_TU_03987_18930101_20240101_hist.zip").write_bytes(
        dwd_zip("TU", hist))
    (folder / "air_temperature/recent/03987.zip").write_bytes(dwd_zip("TU", recent))
    (folder / "wind/recent/03987.zip").write_bytes(dwd_zip("FF", [
        ("2024010200", [2.0, 350.0]), ("2024010201", [4.0, 10.0])]))
    (folder / "precipitation/recent/03987.zip").write_bytes(dwd_zip("RR", [
        ("2024010200", [0.5, 1, -999]), ("2024010201", [1.5, 1, -999]), ("2024010202", [-999.0, 0, -999])]))
    return folder


# ---- UBA: "now" is 2024-01-10 12:00 Berlin ----

def uba_station(sid: str, code: str, name: str, city: str, lon: str, lat: str, network: tuple[str, str, str],
                setting: str, kind: str) -> list:
    """A station as the UBA's meta data list it."""
    return [sid, code, name, city, "", "2009-04-15", None, lon, lat, network[0], "1", "1", network[1], network[2],
            f"{setting} area", setting, kind, "", "", ""]


UBA_META = {
    "stations": {
        "21": uba_station("21", "DEBB021", "Potsdam-Zentrum", "Potsdam", "13.0602", "52.4014",
                          ("4", "BB", "Brandenburg"), "urban", "background"),
        "300": uba_station("300", "DEUB001", "Westerland", "Sylt", "8.3092", "54.9256",
                           ("17", "UB", "UBA"), "rural", "background"),
    },
    "components": {
        "1": ["1", "PM10", "PM₁₀", "µg/m³", "Particulate matter"],
        "2": ["2", "CO", "CO", "mg/m³", "Carbon monoxide"],
        "4": ["4", "SO2", "SO₂", "µg/m³", "Sulphur dioxide"],
        "5": ["5", "NO2", "NO₂", "µg/m³", "Nitrogen dioxide"],
        "7": ["7", "PM10BAP", "BaP", "ng/m³", "Benzo(a)pyrene in particulate matter"],
    },
    "scopes": {
        "1": ["1", "1TMW", "day", "86400", "0", "Daily average"],
        "2": ["2", "1SMW", "hour", "3600", "0", "One hour average"],
        "3": ["3", "1SMW_MAX", "hour", "3600", "1", "Maximum one hour average"],
        "4": ["4", "8SMW", "hour", "28800", "0", "Eight hour average"],
    },
    "limits": {
        "15": ["2016-01-01 00:00:00", "2024-01-10 10:00:00"],  # a scope and component over all stations
        "2521": ["2", "5", "21", "2016-01-01 00:00:00", "2024-01-10 10:00:00"],
        "3521": ["3", "5", "21", "2016-01-01 00:00:00", "2024-01-09 00:00:00"],
        "1121": ["1", "1", "21", "2016-01-01 00:00:00", "2024-01-09 00:00:00"],
        "2121": ["2", "1", "21", "2018-03-01 00:00:00", "2024-01-10 10:00:00"],
        "11300": ["1", "1", "300", "2016-01-01 00:00:00", "2024-01-09 00:00:00"],
        "42300": ["4", "2", "300", "2016-01-01 00:00:00", "2024-01-10 10:00:00"],
        "24300": ["2", "4", "300", "2016-01-01 00:00:00", "2020-06-30 23:00:00"],  # no longer measured
        "17300": ["1", "7", "300", "2016-01-01 00:00:00", "2024-01-09 00:00:00"],  # not a component offered
        "257": ["2", "5", "7", "2016-01-01 00:00:00", "2021-01-17 11:00:00"],  # a closed station
    },
    "networks": {"4": ["4", "BB", "Brandenburg"], "17": ["17", "UB", "UBA"]},
}


def uba_rows(station: str, component: int, scope: int, stamps: list[str], values: list) -> dict:
    """A measures answer: start -> [component, scope, value, end, index]."""
    return {"data": {station: {s: [component, scope, v, s, "0"] for s, v in zip(stamps, values, strict=True)}}}


@pytest.fixture
def uba_files(cache_root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Potsdam's hourly NO2 at the turn of 2023 (last year still preliminary), Westerland's daily PM10."""
    from free_data import uba
    from free_data.sources import epoch_ms
    monkeypatch.setattr(uba, "_now", lambda: epoch_ms("2024-01-10T12:00", "Europe/Berlin"))
    put(cache_root, uba.SOURCE, "meta.json", UBA_META)
    put(cache_root, uba.SOURCE, "DEBB021/NO2/1SMW/2023-preliminary.json",
        uba_rows("21", 5, 2, ["2023-12-31 22:00:00", "2023-12-31 23:00:00"], [20, 21]))
    put(cache_root, uba.SOURCE, "DEBB021/NO2/1SMW/2024.json",
        uba_rows("21", 5, 2, [f"2024-01-01 0{h}:00:00" for h in range(4)], [30, None, 32, 33]))
    put(cache_root, uba.SOURCE, "DEUB001/PM10/1TMW/2024.json",  # daily means come labelled at 11:00
        uba_rows("300", 1, 1, ["2024-01-01 11:00:00", "2024-01-02 11:00:00"], [18, 6]))
    return cache_root / uba.SOURCE


# ---- BASt: the 2024 motorway file, and a station list ----

BAST_COUNTS = ("KFZ_R1", "KFZ_R2", "Lkw_R1", "Lkw_R2", "PLZ_R1", "Pkw_R1", "Lfw_R1", "Mot_R1", "PmA_R1", "Bus_R1",
               "LoA_R1", "Lzg_R1", "Sat_R1", "Son_R1", "PLZ_R2", "Pkw_R2", "Lfw_R2", "Mot_R2", "PmA_R2", "Bus_R2",
               "LoA_R2", "Lzg_R2", "Sat_R2", "Son_R2")
BAST_HEADER = "TKNR;Zst;Land;Strklas;Strnum;Datum;Wotag;Fahrtzw;Stunde;" + ";".join(
    f"{c};K_{c}" for c in BAST_COUNTS)


def bast_row(station: int, date: str, hour: int, counts: list[tuple[int, str]]) -> str:
    """A row of a yearly file: the first four counts (KFZ and Lkw per direction) given, the rest -1."""
    cells = list(counts) + [(-1, "x")] * (len(BAST_COUNTS) - len(counts))
    return (f"5107;{station:4d};05;A;   1;{date}; 1;w;{hour:02d};"
            + ";".join(f"{v:5d};{mark}" for v, mark in cells))


def bast_zip(rows: list[str], name: str = "2024_A_S.txt") -> bytes:
    import io
    import zipfile
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, "\r\n".join([BAST_HEADER, *rows, ""]).encode("latin-1"))
    return buffer.getvalue()


BAST_STATIONS = (
    "TK_Nr;DZ_Nr;DZ_Name;Land_Nr;Land_Code;Str_Kl;Str_Nr;Str_Zus;Erf_Art;Fernziel_Ri1;Nahziel_Ri1;Hi_Ri1;"
    "Fernziel_Ri2;Nahziel_Ri2;Hi_Ri2;Koor_WGS84_N;Koor_WGS84_E\r\n"
    "5107;5113;Köln-Nord (O);5;NW;A;1;;8+1;Dortmund;AK Köln-Nord;N;Koblenz;AS Köln-Niehl;S;51,0123;6,9456\r\n"
    "6507;0223;Eppelborn;10;SL;B;10;;5+1;Saarbrücken;Eppelborn;S;Lebach;Lebach;N;49,4;6,97\r\n"
    "7000;7001;Stadtstraße;8;BW;L;12;;2;;;;;;;48,1;9,1\r\n"
    "    \r\n"
)


@pytest.fixture
def bast_files(cache_root: Path) -> Path:
    """Years 2023 (motorways only) and 2024 published; Köln-Nord on the A 1, Eppelborn on the B 10."""
    from free_data import bast
    folder = cache_root / bast.SOURCE
    folder.mkdir(parents=True)
    (folder / "years.html").write_text(
        '<a href="https://www.bast.de/videos/2023_A_S.zip">A</a> <a href="https://www.bast.de/videos/2024_A_S.zip">A</a>'
        ' <a href="https://www.bast.de/videos/2024_B_S.zip">B</a>')
    (folder / "stations.html").write_text(
        '<a href="DE/Themen/Digitales/HF_1/Massnahmen/verkehrszaehlung/Daten/2024_1/Jawe2024.csv'
        '?view=renderTcDataExportCSV">CSV</a>')
    (folder / "stations.csv").write_bytes(BAST_STATIONS.encode("latin-1"))
    return folder
