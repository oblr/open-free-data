"""Local file cache for the free data sources.

Every reader stores what it downloads under one folder per source, named
after the source's id in data/ (e.g. "de-smard"):

    free-data/cache/<source>/<key>

The root is free-data/cache (git-ignored); $FREE_DATA_CACHE overrides it.

A file is fetched again once it is older than the `max_age` the reader asks
for: None keeps it forever (data that no longer changes), 0 always fetches.
If a fetch fails and a cached copy exists, the copy is used with a warning,
so readers keep working offline.

    python cache.py                 # size and file count per source
    python cache.py clear de-smard  # delete one source's files
    python cache.py clear --older-than 30d
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import threading
import time
import urllib.error
import urllib.request
import warnings
from pathlib import Path

DEFAULT_ROOT = Path(__file__).with_name("cache")
USER_AGENT = "free-data/0.1"

MINUTE = 60.0
HOUR = 60 * MINUTE
DAY = 24 * HOUR


class FetchError(RuntimeError):
    def __init__(self, url: str, status: int | None, message: str) -> None:
        super().__init__(message)
        self.url = url
        self.status = status  # HTTP status, None when the server wasn't reached


def root() -> Path:
    return Path(os.environ.get("FREE_DATA_CACHE") or DEFAULT_ROOT)


class Cache:
    """The cached files of one source."""

    def __init__(self, source: str, base: Path | None = None) -> None:
        self.source = source
        self.dir = (base or root()) / source

    def path(self, key: str) -> Path:
        target = (self.dir / key).resolve()
        if not target.is_relative_to(self.dir.resolve()):
            raise ValueError(f"cache key leaves the source folder: {key!r}")
        return target

    def fetch(self, url: str, key: str, max_age: float | None = None) -> bytes:
        """The body of `url`, from the cache while it is fresh enough."""
        target = self.path(key)
        if target.exists() and (max_age is None or time.time() - target.stat().st_mtime < max_age):
            return target.read_bytes()
        try:
            body = _download(url)
        except FetchError as error:
            if error.status is None and target.exists():
                warnings.warn(f"{url} not reachable, using the cached copy: {error}", stacklevel=2)
                return target.read_bytes()
            raise
        self._write(target, body)
        return body

    def fetch_json(self, url: str, key: str, max_age: float | None = None) -> dict:
        return json.loads(self.fetch(url, key, max_age))

    def _write(self, target: Path, body: bytes) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        # One temporary file per writer, renamed into place: readers only
        # ever see complete files, and parallel downloads don't collide.
        part = target.with_name(f"{target.name}.{os.getpid()}.{threading.get_ident()}.part")
        part.write_bytes(body)
        part.replace(target)


def _download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        raise FetchError(url, error.code, f"HTTP {error.code} for {url}") from error
    except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
        raise FetchError(url, None, f"{url}: {getattr(error, 'reason', error)}") from error


def _files(folder: Path) -> list[Path]:
    return [p for p in folder.rglob("*") if p.is_file() and not p.name.endswith(".part")]


def _age(text: str) -> float:
    units = {"m": MINUTE, "h": HOUR, "d": DAY}
    if text[-1:] not in units:
        raise argparse.ArgumentTypeError("use e.g. 30m, 12h or 7d")
    return float(text[:-1]) * units[text[-1]]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Inspect or clear the free-data cache.")
    sub = parser.add_subparsers(dest="command")
    clear = sub.add_parser("clear", help="delete cached files")
    clear.add_argument("sources", nargs="*", help="source ids (default: all)")
    clear.add_argument("--older-than", type=_age, help="only files older than this, e.g. 7d")
    args = parser.parse_args(argv)

    base = root()
    sources = sorted(p.name for p in base.iterdir() if p.is_dir()) if base.exists() else []
    if args.command == "clear":
        for source in args.sources or sources:
            folder = base / source
            if not folder.is_dir():
                print(f"{source}: nothing cached")
                continue
            if args.older_than is None:
                count = len(_files(folder))
                shutil.rmtree(folder)
            else:
                cutoff = time.time() - args.older_than
                old = [p for p in _files(folder) if p.stat().st_mtime < cutoff]
                for p in old:
                    p.unlink()
                count = len(old)
            print(f"{source}: deleted {count} files")
        return

    print(f"cache: {base}")
    if not sources:
        print("  (empty)")
    for source in sources:
        files = _files(base / source)
        size = sum(p.stat().st_size for p in files)
        newest = max((p.stat().st_mtime for p in files), default=None)
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(newest)) if newest else "-"
        print(f"  {source:<28} {len(files):>6} files {size / 1e6:>9.1f} MB   newest {when}")


if __name__ == "__main__":
    main()
