"""A local web page to search and plot the free data sources.

    python -m free_data.browser            # then open http://127.0.0.1:8050
    python -m free_data.browser --port 9000

Lists every source in data/; the ones with a plugin in plugins/ can be
searched and plotted, also side by side across sources. Standard library
only; data goes through each plugin's reader and so through the cache.

The page is web/index.html with the TypeScript in web/src, compiled to
web/dist (cd web && npm install && npm run build). The wheel ships index.html
and dist/ as free_data/web; from a source checkout they are read from web/.
It is served as:
    /                               web/index.html
    /dist/<file>.js                 web/dist/<file>.js, the compiled modules

API (all GET):
    /api/sources                    sources, their options and whether they have a plugin
    /api/sources/<id>/datasets      the datasets of one plugged-in source
    /api/series?d=<source>:<dataset>&d=...&<source>.<option>=<value>&start=<date>&end=<date>[&format=csv]
                [&unit.<source>:<dataset>=<unit>][&step=hour|day|month]   (see transforms.py)
    /api/vocabulary                 the terms of vocabulary.py and units.py
    /api/related?d=<source>:<dataset>&d=...   series related to these (see relations.py)
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from dataclasses import asdict, replace
from datetime import UTC, datetime
from functools import cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np

from free_data import relations, sources, transforms, units, vocabulary
from free_data.cache import FetchError

WEB = Path(__file__).with_name("web")  # installed from a wheel
if not WEB.is_dir():
    WEB = Path(__file__).parents[2] / "web"  # a source checkout or editable install
PAGE = WEB / "index.html"
DIST = WEB / "dist"
STATIC_TYPES = {".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8"}


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


def all_entries() -> list[relations.Entry]:
    """Every dataset of every plugged-in source whose list can be loaded."""
    found = []
    for source in sources.discover():
        try:
            found += [relations.Entry(source, d) for d in datasets(source).values()]
        except Exception:  # noqa: BLE001 (a source that can't list its datasets just has no relations)
            continue
    return found


def related(query: dict[str, list[str]]) -> list[dict]:
    entries = all_entries()
    by_key = {e.key: e for e in entries}
    targets = []
    for key in query.get("d", []):
        if key not in by_key:
            raise RequestError(f"no dataset {key!r}")
        targets.append(by_key[key])
    if not targets:
        raise RequestError("pick at least one dataset")
    return [asdict(r) for r in relations.related(targets, entries)]


@cache
def vocabulary_terms() -> dict:
    return {
        "variables": {k: asdict(v) for k, v in vocabulary.VARIABLES.items()},
        "measures": {k: {**asdict(v), "units": list(v.units)} for k, v in vocabulary.MEASURES.items()},
        "units": {k: asdict(v) for k, v in units.UNITS.items()},
        "dimensions": {k: asdict(v) for k, v in units.DIMENSIONS.items()},
        "kinds": list(vocabulary.KINDS),
        "regions": {k: asdict(v) for k, v in vocabulary.REGIONS.items()},
        "countries": {k: asdict(v) for k, v in vocabulary.COUNTRIES.items()},
    }


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
    try:
        return sources.moment(text, zone, end)
    except ValueError as error:
        raise RequestError(str(error)) from error


def series(query: dict[str, list[str]]) -> tuple[list[dict], sources.Table, np.ndarray]:
    """The columns, the merged table and which of its times each column's source has.

    Sources on different grids leave NaN in each other's columns. Those are
    not gaps in the data, so the page joins across them instead.

    `unit.<source>:<dataset>` converts a dataset to another unit and `step`
    aggregates every dataset, each in its source's timezone (transforms.py).
    """
    wanted: dict[str, list[str]] = {}  # source -> datasets, in the order asked for
    for item in query.get("d", []):
        source, _, dataset = item.partition(":")
        if dataset not in datasets(source):
            raise RequestError(f"{source} has no dataset {dataset!r}")
        wanted.setdefault(source, []).append(dataset)
    if not wanted:
        raise RequestError("pick at least one dataset")
    targets = {k.removeprefix("unit."): v[0] for k, v in query.items() if k.startswith("unit.") and v[0]}
    for key in targets:
        source, _, dataset = key.partition(":")
        if dataset not in wanted.get(source, []):
            raise RequestError(f"a unit for {key!r}, which is not picked")
    step = query.get("step", [""])[0]

    tables = []
    unit_of: dict[str, str] = {}  # column -> its unit after the transformation
    for source, ids in wanted.items():
        p = plugin(source)
        given = {k.split(".", 1)[1]: v[0] for k, v in query.items() if k.startswith(f"{source}.")}
        try:
            options = p.resolve(given)
        except ValueError as error:
            raise RequestError(str(error)) from error
        start = moment(query.get("start", [None])[0], p.timezone, end=False)
        end = moment(query.get("end", [None])[0], p.timezone, end=True)
        if start and end and start > end:
            raise RequestError("the start date is after the end date")
        transform = transforms.Transform(
            {k.partition(":")[2]: v for k, v in targets.items() if k.startswith(f"{source}:")}, step)
        try:
            transforms.check(transform, datasets(source))  # before the download
        except ValueError as error:
            raise RequestError(str(error)) from error
        table, column_units = transforms.apply(p.read(ids, start, end, options), datasets(source), transform,
                                               p.timezone)
        tables.append((source, table))
        unit_of.update({f"{source}:{c}": u for c, u in zip(table.columns, column_units, strict=True)})

    table = sources.merge(tables)
    on_grid = np.column_stack([np.isin(table.time, t.time) for _, t in tables for _ in t.columns])
    columns = []
    for key in table.columns:
        source, _, dataset = key.partition(":")
        d = datasets(source)[dataset]
        columns.append({"key": key, "source": source, "dataset": dataset,
                        "label": d.label, "group": d.group, "unit": unit_of[key], "native_unit": d.unit})
    return columns, table, on_grid


def to_csv(columns: list[dict], table: sources.Table) -> bytes:
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["timestamp_utc", *(f"{c['key']} {c['group']} {c['label']} [{c['unit']}]" for c in columns)])
    for t, row in zip(table.time.astype(np.int64), table.values, strict=True):
        stamp = datetime.fromtimestamp(t / 1000, UTC).isoformat()
        writer.writerow([stamp, *("" if np.isnan(v) else repr(float(v)) for v in row)])
    return out.getvalue().encode()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 (http.server's naming)
        url = urlparse(self.path)
        query = parse_qs(url.query)
        parts = [unquote(p) for p in url.path.strip("/").split("/")]
        try:
            if url.path == "/":
                if not (DIST / "main.js").exists():
                    raise RequestError("the page is not built: cd web && npm install && npm run build")
                self._send(HTTPStatus.OK, "text/html; charset=utf-8", PAGE.read_bytes())
            elif parts[0] == "dist" and len(parts) > 1:
                self._static(parts[1:])
            elif parts == ["api", "sources"]:
                self._json(HTTPStatus.OK, source_list())
            elif len(parts) == 4 and parts[:2] == ["api", "sources"] and parts[3] == "datasets":
                self._json(HTTPStatus.OK, [asdict(d) for d in datasets(parts[2]).values()])
            elif parts == ["api", "vocabulary"]:
                self._json(HTTPStatus.OK, vocabulary_terms())
            elif parts == ["api", "related"]:
                self._json(HTTPStatus.OK, related(query))
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

    def _static(self, parts: list[str]) -> None:
        """A compiled file from web/dist; nothing outside it, nothing but .js and .css."""
        target = DIST.joinpath(*parts).resolve()
        kind = STATIC_TYPES.get(target.suffix)
        if not (kind and target.is_relative_to(DIST.resolve()) and target.is_file()):
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        self._send(HTTPStatus.OK, kind, target.read_bytes(), {"Cache-Control": "no-cache"})

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
