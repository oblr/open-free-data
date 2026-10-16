"""A local web page to search and plot the free data sources.

    python browser.py            # then open http://127.0.0.1:8050
    python browser.py --port 9000

Lists every source in data/; the ones with a plugin in plugins/ can be
searched and plotted, also side by side across sources. Standard library
only; data goes through each plugin's reader and so through the cache.

API (all GET):
    /api/sources                    sources, their options and whether they have a plugin
    /api/sources/<id>/datasets      the datasets of one plugged-in source
    /api/series?d=<source>:<dataset>&d=...&<source>.<option>=<value>&start=<date>&end=<date>[&format=csv]
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from dataclasses import asdict, replace
from datetime import datetime, time, timezone
from functools import cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse
from zoneinfo import ZoneInfo

import numpy as np

import sources
from cache import FetchError

PAGE = Path(__file__).with_name("browser.html")


class RequestError(ValueError):
    pass


def plugin(source: str) -> sources.Plugin:
    found = sources.discover().get(source)
    if found is None:
        raise RequestError(f"no plugin for source {source!r}")
    return found


@cache
def datasets(source: str) -> dict[str, sources.Dataset]:
    """The plugin's datasets; those without a link of their own get the source's."""
    fallback = next((row["link"] for row in sources.catalog() if row["id"] == source), "")
    return {d.id: d if d.link else replace(d, link=fallback) for d in plugin(source).datasets()}


def source_list() -> list[dict]:
    rows = sources.catalog()
    for row in rows:
        if row["plugin"]:
            p = plugin(row["id"])
            row["timezone"] = p.timezone
            row["options"] = [asdict(o) for o in p.options]
    return rows


def moment(text: str | None, zone: str, end: bool) -> datetime | None:
    """A date or date-time from the page; without an offset it is in the source's zone."""
    if not text:
        return None
    try:
        value = datetime.fromisoformat(text)
    except ValueError as error:
        raise RequestError(f"not a date: {text!r}") from error
    if len(text) == 10 and end:
        value = datetime.combine(value.date(), time.max)  # a bare end date means the whole day
    return value if value.tzinfo else value.replace(tzinfo=ZoneInfo(zone))


def series(query: dict[str, list[str]]) -> tuple[list[dict], sources.Table, np.ndarray]:
    """The columns, the merged table and which of its times each column's source has.

    Sources on different grids leave NaN in each other's columns. Those are
    not gaps in the data, so the page joins across them instead.
    """
    wanted: dict[str, list[str]] = {}  # source -> datasets, in the order asked for
    for item in query.get("d", []):
        source, _, dataset = item.partition(":")
        if dataset not in datasets(source):
            raise RequestError(f"{source} has no dataset {dataset!r}")
        wanted.setdefault(source, []).append(dataset)
    if not wanted:
        raise RequestError("pick at least one dataset")

    tables = []
    for source, ids in wanted.items():
        p = plugin(source)
        given = {k.split(".", 1)[1]: v[0] for k, v in query.items() if k.startswith(f"{source}.")}
        try:
            options = p.resolve(given)
        except ValueError as error:
            raise RequestError(str(error)) from error
        start = moment(query.get("start", [None])[0], p.timezone, end=False)
        end = moment(query.get("end", [None])[0], p.timezone, end=True)
        tables.append((source, p.read(ids, start, end, options)))

    table = sources.merge(tables)
    on_grid = np.column_stack([np.isin(table.time, t.time) for _, t in tables for _ in t.columns])
    columns = []
    for key in table.columns:
        source, _, dataset = key.partition(":")
        d = datasets(source)[dataset]
        columns.append({"key": key, "source": source, "dataset": dataset,
                        "label": d.label, "group": d.group, "unit": d.unit})
    return columns, table, on_grid


def to_csv(columns: list[dict], table: sources.Table) -> bytes:
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["timestamp_utc", *(f"{c['key']} {c['group']} {c['label']} [{c['unit']}]" for c in columns)])
    for t, row in zip(table.time.astype(np.int64), table.values, strict=True):
        stamp = datetime.fromtimestamp(t / 1000, timezone.utc).isoformat()
        writer.writerow([stamp, *("" if np.isnan(v) else repr(float(v)) for v in row)])
    return out.getvalue().encode()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 (http.server's naming)
        url = urlparse(self.path)
        query = parse_qs(url.query)
        parts = [unquote(p) for p in url.path.strip("/").split("/")]
        try:
            if url.path == "/":
                self._send(HTTPStatus.OK, "text/html; charset=utf-8", PAGE.read_bytes())
            elif parts == ["api", "sources"]:
                self._json(HTTPStatus.OK, source_list())
            elif len(parts) == 4 and parts[:2] == ["api", "sources"] and parts[3] == "datasets":
                self._json(HTTPStatus.OK, [asdict(d) for d in datasets(parts[2]).values()])
            elif parts == ["api", "series"]:
                columns, table, on_grid = series(query)
                if query.get("format", ["json"])[0] == "csv":
                    self._send(HTTPStatus.OK, "text/csv; charset=utf-8", to_csv(columns, table),
                               {"Content-Disposition": 'attachment; filename="free-data.csv"'})
                else:
                    self._json(HTTPStatus.OK, {
                        "columns": columns,
                        "time": (table.time.astype(np.int64) // 1000).tolist(),  # epoch seconds
                        "values": [[None if np.isnan(v) else float(v) for v in column]
                                   for column in table.values.T],
                        # "1" where the column's source has this time, "0" where it only comes from the merge
                        "on_grid": ["".join("1" if x else "0" for x in column) for column in on_grid.T],
                    })
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        except RequestError as error:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
        except FetchError as error:
            self._json(HTTPStatus.BAD_GATEWAY, {"error": f"source not reachable: {error}"})
        except Exception as error:  # a plugin's own error: show it rather than drop the connection
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"{type(error).__name__}: {error}"})

    def _json(self, status: HTTPStatus, body: object) -> None:
        self._send(status, "application/json", json.dumps(body).encode())

    def _send(self, status: HTTPStatus, kind: str, body: bytes, headers: dict[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Search and plot the free data sources in the browser.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args()
    found = sources.discover()  # import errors in a plugin show up now, not on the first request
    print(f"plugins: {', '.join(found) or 'none'}")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"free-data browser on http://{args.host}:{args.port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
