"""Local file cache for the free data sources.

Every reader stores what it downloads under one folder per source, named
after the source's id in data/ (e.g. "de-smard"):

    <root>/<source>/<key>

The root is $FREE_DATA_CACHE, else $XDG_CACHE_HOME/free-data, else
~/.cache/free-data.

A file is fetched again once it is older than the `max_age` the reader asks
for: None keeps it forever (data that no longer changes), 0 always fetches.
If the source can't be reached, or answers with a server error (5xx) or
"too many requests" (429), the download is tried again twice, after 2 and
8 seconds. If it still fails and a cached copy exists, the copy is used with a
warning, so readers keep working offline and through outages.

Requests are throttled per host, before they are sent: at most RATE per
second to any one host (HOST_RATES overrides it per host, FREE_DATA_RATE
sets the default), so a long ingest stays well under what a provider
tolerates. A 429 or 503 halves that host's rate for the rest of the process
and, with a Retry-After header, holds every request to it until then.

How cached files are reused can be changed for a block of work (ingest.py
does this):

    with cache.refreshing("never"):   # any cached file is used, however old
        plugin.read(...)
    "stale" (the default) refetches what is older than its max_age,
    "always" refetches everything, and "offline" never downloads: a file
    that isn't cached raises NotCached.

The setting follows the work into reader threads started through Pool
(a ThreadPoolExecutor that carries the caller's context), as does
`tally()`, which counts what was downloaded and what came from the cache.

Everything is logged to "free_data.cache": every download at INFO (key,
size, seconds), each file served from the cache at DEBUG, and at WARNING a
retry, a host slowing us down, and a cached copy used after a failure.

    python -m free_data.cache                 # size and file count per source
    python -m free_data.cache clear de-smard  # delete one source's files
    python -m free_data.cache clear --older-than 30d
"""

from __future__ import annotations

import argparse
import contextvars
import email.utils
import json
import logging
import os
import shutil
import threading
import time
import urllib.error
import urllib.request
import warnings
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse

USER_AGENT = "free-data/0.1"
RETRY_PAUSES = (2.0, 8.0)  # seconds before each new try after a temporary failure
MAX_RETRY_AFTER = 60.0  # the longest a retry waits for a Retry-After; the host stays held longer
MAX_HOLD = 15 * 60.0  # the longest a Retry-After holds a host

RATE = float(os.environ.get("FREE_DATA_RATE") or 2.0)  # requests per second to one host; 0 = no limit
HOST_RATES: dict[str, float] = {}  # per host, over RATE
MIN_RATE = 0.1  # what repeated 429s can bring a host down to
REFRESH = ("offline", "never", "stale", "always")
LONG_WAIT = 5.0  # a throttle wait longer than this is logged

log = logging.getLogger(__name__)

MINUTE = 60.0
HOUR = 60 * MINUTE
DAY = 24 * HOUR


class FetchError(RuntimeError):
    def __init__(self, url: str, status: int | None, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.url = url
        self.status = status  # HTTP status, None when the server wasn't reached
        self.retry_after = retry_after  # seconds the server asked to wait, if it said

    @property
    def transient(self) -> bool:
        """Whether the source may well answer later: unreachable, overloaded or failing."""
        return self.status is None or self.status == 429 or self.status >= 500


def root() -> Path:
    if os.environ.get("FREE_DATA_CACHE"):
        return Path(os.environ["FREE_DATA_CACHE"])
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "free-data"


class NotCached(FetchError):
    """A file that isn't in the cache, asked for under refreshing("offline")."""

    def __init__(self, url: str, path: Path) -> None:
        super().__init__(url, None, f"not cached: {path}")

    @property
    def transient(self) -> bool:
        return False


class Throttle:
    """Spaces the requests to each host: every request books the next free slot, then waits for it."""

    def __init__(self, rate: float | None = None, host_rates: dict[str, float] | None = None) -> None:
        self.rate = RATE if rate is None else rate
        self.rates = dict(HOST_RATES if host_rates is None else host_rates)
        self._next: dict[str, float] = {}
        self._lock = threading.Lock()

    def rate_for(self, host: str) -> float:
        return self.rates.get(host, self.rate)

    def wait(self, url: str) -> float:
        """Sleeps until this request's slot; returns how long it waited."""
        host = urlparse(url).netloc
        with self._lock:
            rate = self.rate_for(host)
            now = time.monotonic()
            slot = max(now, self._next.get(host, now))
            self._next[host] = slot + (1 / rate if rate > 0 else 0)
        if slot - now > LONG_WAIT:
            log.info("%s: waiting %.0f s before the next request", host, slot - now)
        if slot > now:
            time.sleep(slot - now)
        return slot - now

    def slow_down(self, url: str, retry_after: float | None) -> None:
        """After a 429 or 503: half the rate, and nothing to the host before Retry-After."""
        host = urlparse(url).netloc
        with self._lock:
            rate = self.rate_for(host)
            self.rates[host] = max(MIN_RATE, rate / 2) if rate > 0 else 1.0
            if retry_after:
                hold = time.monotonic() + min(retry_after, MAX_HOLD)
                self._next[host] = max(self._next.get(host, 0.0), hold)
        log.warning("%s asks us to slow down: now %.2g requests per second%s", host, self.rates[host],
                    f", nothing for {min(retry_after, MAX_HOLD):.0f} s" if retry_after else "")


THROTTLE = Throttle()

_refresh: contextvars.ContextVar[str] = contextvars.ContextVar("refresh", default="stale")
_tally: contextvars.ContextVar[Counter | None] = contextvars.ContextVar("tally", default=None)


@contextmanager
def refreshing(mode: str) -> Iterator[None]:
    """Within the block, cached files are used "never" refetched, refetched when "stale", or "always"."""
    if mode not in REFRESH:
        raise ValueError(f"refresh must be one of {REFRESH}, got {mode!r}")
    token = _refresh.set(mode)
    try:
        yield
    finally:
        _refresh.reset(token)


@contextmanager
def tally() -> Iterator[Counter]:
    """Counts, within the block, the files "downloaded" and those "cached" (served from the cache)."""
    counts: Counter = Counter()
    token = _tally.set(counts)
    try:
        yield counts
    finally:
        _tally.reset(token)


def _count(what: str) -> None:
    counts = _tally.get()
    if counts is not None:
        counts[what] += 1


class Pool(ThreadPoolExecutor):
    """A ThreadPoolExecutor whose tasks run in the context of the code that submits them,
    so refreshing() and tally() reach the readers' parallel downloads."""

    def submit(self, fn, /, *args, **kwargs):
        return super().submit(contextvars.copy_context().run, fn, *args, **kwargs)


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
        if target.exists() and self._fresh(target, max_age):
            log.debug("%s: cached %s", self.source, key)
            _count("cached")
            return target.read_bytes()
        if _refresh.get() == "offline":
            raise NotCached(url, target)
        try:
            body, seconds = _download_with_retries(url)
        except FetchError as error:
            if error.transient and target.exists():
                warnings.warn(f"{url} failed, using the cached copy: {error}", stacklevel=2)
                _count("cached")
                return target.read_bytes()
            raise
        self._write(target, body)
        _count("downloaded")
        log.info("%s: downloaded %s (%s, %.1f s)", self.source, key, _size(len(body)), seconds)
        return body

    def cached(self, key: str) -> bool:
        return self.path(key).exists()

    @staticmethod
    def _fresh(target: Path, max_age: float | None) -> bool:
        mode = _refresh.get()
        if mode == "always":
            return False
        if mode in ("never", "offline") or max_age is None:
            return True
        return time.time() - target.stat().st_mtime < max_age

    def fetch_json(self, url: str, key: str, max_age: float | None = None) -> dict:
        return json.loads(self.fetch(url, key, max_age))

    def _write(self, target: Path, body: bytes) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        # One temporary file per writer, renamed into place: readers only
        # ever see complete files, and parallel downloads don't collide.
        part = target.with_name(f"{target.name}.{os.getpid()}.{threading.get_ident()}.part")
        part.write_bytes(body)
        part.replace(target)


def _download_with_retries(url: str) -> tuple[bytes, float]:
    """`_download`, throttled, and tried again after a pause while the failure looks temporary;
    with the seconds the last try took, not counting the waits."""
    for pause in (*RETRY_PAUSES, None):
        THROTTLE.wait(url)
        began = time.monotonic()
        try:
            return _download(url), time.monotonic() - began
        except FetchError as error:
            if error.status in (429, 503):
                THROTTLE.slow_down(url, error.retry_after)
            if pause is None or not error.transient:
                raise
            wait = max(pause, min(error.retry_after or 0.0, MAX_RETRY_AFTER))
            log.warning("%s; trying again in %.0f s", error, wait)
            time.sleep(wait)
    raise AssertionError("unreachable")


def _download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        wait = _retry_after(error.headers.get("Retry-After") if error.headers else None)
        raise FetchError(url, error.code, f"HTTP {error.code} for {url}", wait) from error
    except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
        raise FetchError(url, None, f"{url}: {getattr(error, 'reason', error)}") from error


def _size(n: int) -> str:
    for unit in ("B", "kB", "MB"):
        if n < 1000 or unit == "MB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1000
    return ""


def _retry_after(value: str | None) -> float | None:
    """A Retry-After header, in seconds from now: either seconds or an HTTP date."""
    if not value:
        return None
    if value.strip().isdigit():
        return float(value)
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, when.timestamp() - time.time())


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
