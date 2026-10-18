"""Background downloads: series read ahead of time, so they are in the cache when needed.

    python -m free_data.ingest de-smard 410 4169 --start 2025-01-01 --end 2025-12-31
    python -m free_data.ingest de-uba-luftdaten --match potsdam --start 2025-01-01 --end 2025-12-31
    python -m free_data.ingest de-dwd-opendata --all --start 2025-01-01 --end 2025-12-31 --rate 1
    python -m free_data.ingest de-smard 410 --start 2025-01-01 --end 2025-12-31 --force

A job is a source, its datasets (all, when none are named), a period and
the source's options. It is read through the source's plugin like any
other read, so whatever the reader downloads lands in the cache; the values
themselves are dropped. Jobs run in the background, one thread per source,
so a slow source doesn't hold up the others; the datasets of a job are read
CHUNK at a time.

Nothing that is already cached is downloaded again: jobs run under
cache.refreshing("never"), which uses every cached file however old, so a
second run of the same job downloads nothing. `refresh="stale"` refetches
what is older than its reader's max_age (what the browser does), and
`refresh="always"` (--force) everything.

Requests go through the cache's throttle (cache.py): at most cache.RATE
per second to any host, slower after a 429 or 503.

Progress goes to the logger "free_data.ingest" (and every download to
"free_data.cache"): at INFO when a job starts, after each chunk with what
is left, and when it ends; at WARNING every dataset that fails, as it
fails. A chunk that fails is read again one dataset at a time, so the error
names the dataset. The command line shows INFO; -v adds every cached file,
-q leaves only warnings and the summary.
"""

from __future__ import annotations

import argparse
import logging
import queue
import sys
import threading
import time
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field

from free_data import cache, sources

CHUNK = 25  # datasets per plugin.read

log = logging.getLogger(__name__)


@dataclass
class Job:
    source: str
    datasets: tuple[str, ...]  # empty: every dataset of the source
    start: str  # ISO date or date-time, in the source's timezone unless it has an offset
    end: str
    options: dict[str, str] = field(default_factory=dict)
    refresh: str = "never"  # one of cache.REFRESH
    state: str = "queued"  # queued, running, done, failed, stopped
    total: int = 0  # datasets to read, once known
    read: int = 0  # datasets read so far
    failed: int = 0  # datasets that couldn't be read
    downloaded: int = 0  # files fetched
    cached: int = 0  # files that were already there
    errors: list[str] = field(default_factory=list)  # "<dataset>: <error>" per failure; the job goes on
    seconds: float = 0.0

    @property
    def finished(self) -> bool:
        return self.state in ("done", "failed", "stopped")

    def same(self, other: Job) -> bool:
        return (self.source, self.datasets, self.start, self.end, self.options, self.refresh) == (
            other.source, other.datasets, other.start, other.end, other.options, other.refresh)


class Ingester:
    """Runs jobs in the background, one worker thread per source."""

    def __init__(self, refresh: str = "never", chunk: int = CHUNK) -> None:
        if refresh not in cache.REFRESH:
            raise ValueError(f"refresh must be one of {cache.REFRESH}, got {refresh!r}")
        self.refresh = refresh
        self.chunk = chunk
        self._jobs: list[Job] = []
        self._queues: dict[str, queue.Queue[Job | None]] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._lock = threading.Lock()
        self._stopping = threading.Event()

    def submit(
        self,
        source: str,
        datasets: Iterable[str] = (),
        *,
        start: str,
        end: str,
        options: dict[str, str] | None = None,
        refresh: str | None = None,
    ) -> Job:
        """Queues a job and returns it; the same job, still queued or running, is returned instead."""
        plugin = sources.discover().get(source)
        if plugin is None:
            raise ValueError(f"no plugin for source {source!r}")
        sources.moment(start, plugin.timezone, end=False)  # bad dates fail here, not in the thread
        sources.moment(end, plugin.timezone, end=True)
        job = Job(source, tuple(dict.fromkeys(datasets)), start, end, plugin.resolve(options or {}),
                  refresh or self.refresh)
        if job.refresh not in cache.REFRESH:
            raise ValueError(f"refresh must be one of {cache.REFRESH}, got {job.refresh!r}")
        with self._lock:
            for other in self._jobs:
                if not other.finished and other.same(job):
                    return other
            self._jobs.append(job)
            if source not in self._threads:
                self._queues[source] = queue.Queue()
                thread = threading.Thread(target=self._work, args=(source,), name=f"ingest-{source}", daemon=True)
                self._threads[source] = thread
                thread.start()
            self._queues[source].put(job)
        return job

    def jobs(self) -> list[Job]:
        with self._lock:
            return list(self._jobs)

    def wait(self, timeout: float | None = None) -> bool:
        """Blocks until every job submitted so far has finished; False if `timeout` ran out first."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while not all(job.finished for job in self.jobs()):
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(0.05)
        return True

    def stop(self) -> None:
        """Lets each worker finish its current chunk, then stops; jobs not done become "stopped"."""
        self._stopping.set()
        with self._lock:
            for q in self._queues.values():
                q.put(None)
            threads = list(self._threads.values())
        for thread in threads:
            thread.join()
        for job in self.jobs():
            if not job.finished:
                job.state = "stopped"

    # ---- the worker ----

    def _work(self, source: str) -> None:
        q = self._queues[source]
        while not self._stopping.is_set():
            job = q.get()
            if job is None:
                return
            self._run(job)

    def _run(self, job: Job) -> None:
        began = time.monotonic()
        job.state = "running"
        plugin = sources.discover()[job.source]
        start = sources.moment(job.start, plugin.timezone, end=False)
        end = sources.moment(job.end, plugin.timezone, end=True)
        try:
            with cache.refreshing(job.refresh), cache.tally() as counts:
                if not job.datasets:
                    log.info("%s: listing its datasets", job.source)
                ids = list(job.datasets) or [d.id for d in plugin.datasets()]
                job.total = len(ids)
                log.info("%s: %d datasets, %s to %s, refresh %s", job.source, job.total, job.start, job.end,
                         job.refresh)
                for i in range(0, len(ids), self.chunk):
                    if self._stopping.is_set():
                        job.state = "stopped"
                        break
                    self._read(job, plugin, ids[i:i + self.chunk], start, end)
                    job.downloaded, job.cached = counts["downloaded"], counts["cached"]
                    log.info("%s: %s", job.source, _progress(job, time.monotonic() - began))
                job.downloaded, job.cached = counts["downloaded"], counts["cached"]
        except Exception as error:  # noqa: BLE001 (e.g. the dataset list can't be fetched)
            job.errors.append(str(error))
            job.state = "failed"
            log.error("%s: %s", job.source, error)
        if job.state == "running":
            job.state = "failed" if job.total and job.failed == job.total else "done"
        job.seconds = time.monotonic() - began
        level = logging.INFO if job.state == "done" and not job.failed else logging.WARNING
        log.log(level, "%s: %s", job.source, summary(job))

    def _read(self, job: Job, plugin: sources.Plugin, part: list[str], start, end) -> None:
        """Reads a chunk; if it fails, each of its datasets on its own, so the error names the dataset."""
        try:
            plugin.read(part, start, end, job.options)
        except Exception as error:  # noqa: BLE001 (one bad chunk doesn't end the job)
            if len(part) == 1:
                job.failed += 1
                job.errors.append(f"{part[0]}: {error}")
                log.warning("%s: %s failed: %s", job.source, part[0], error)
                return
            else:
                log.info("%s: a chunk of %d failed (%s); reading them one by one", job.source, len(part), error)
                for dataset in part:
                    if self._stopping.is_set():
                        return
                    self._read(job, plugin, [dataset], start, end)
                return
        job.read += len(part)


def _progress(job: Job, seconds: float) -> str:
    """ "[ 50/3412   1%] 37 downloaded, 13 cached, 0 failed, 12 s, about 13 min left" """
    done = job.read + job.failed
    width = len(str(job.total))
    text = (f"[{done:>{width}}/{job.total} {100 * done / max(job.total, 1):3.0f}%] {job.downloaded} downloaded, "
            f"{job.cached} cached, {job.failed} failed, {_duration(seconds)}")
    if 0 < done < job.total:
        text += f", about {_duration(seconds / done * (job.total - done))} left"
    return text


def summary(job: Job) -> str:
    return (f"{job.state}, {job.read}/{job.total} datasets read, {job.failed} failed, in {_duration(job.seconds)}; "
            f"{job.downloaded} files downloaded, {job.cached} already cached")


def _duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 90 * 60:
        return f"{seconds / 60:.0f} min"
    return f"{seconds // 3600:.0f} h {seconds % 3600 / 60:.0f} min"


def _fold(text: str) -> str:
    """Lower case without accents: "Débit" -> "debit"."""
    return "".join(c for c in unicodedata.normalize("NFKD", text.casefold()) if not unicodedata.combining(c))


def _matching(source: str, words: list[str]) -> list[str]:
    """The datasets whose id, label, group or place contain every word (case and accents aside)."""
    found = []
    for d in sources.discover()[source].datasets():
        place = d.place
        text = _fold(" ".join([d.id, d.label, d.group, d.variable,
                               place.name if place else "", place.water if place else ""]))
        if all(_fold(w) in text for w in words):
            found.append(d.id)
    return found


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download series into the cache ahead of time.")
    parser.add_argument("source", help="a source id with a plugin, e.g. de-smard")
    parser.add_argument("datasets", nargs="*", help="dataset ids (see the source's --list)")
    parser.add_argument("--all", action="store_true", help="every dataset of the source")
    parser.add_argument("--match", nargs="+", metavar="WORD", help="the datasets whose names contain these words")
    parser.add_argument("--start", required=True, help="ISO date, in the source's timezone")
    parser.add_argument("--end", required=True, help="ISO date, inclusive")
    parser.add_argument("--option", action="append", default=[], metavar="NAME=VALUE",
                        help="a source option, e.g. resolution=hour")
    parser.add_argument("--refresh", choices=cache.REFRESH, default="never",
                        help="never: keep every cached file (default); stale: refetch old ones; always: refetch all")
    parser.add_argument("--force", action="store_true", help="the same as --refresh always")
    parser.add_argument("--rate", type=float, help=f"requests per second to one host (default {cache.RATE:g})")
    parser.add_argument("-v", "--verbose", action="store_true", help="also every file taken from the cache")
    parser.add_argument("-q", "--quiet", action="store_true", help="only warnings and the summary")
    args = parser.parse_args(argv)

    level = logging.DEBUG if args.verbose else logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S", level=logging.WARNING)
    logging.getLogger("free_data").setLevel(level)
    logging.captureWarnings(True)  # the readers' warnings (e.g. a cached copy used after a failure) in the same log

    if args.rate is not None:
        cache.THROTTLE.rate = args.rate
    datasets = list(args.datasets)
    if args.match:
        log.info("%s: finding the datasets that match %s", args.source, " ".join(args.match))
        datasets += _matching(args.source, args.match)
        if not datasets:
            parser.error(f"no dataset of {args.source} matches {' '.join(args.match)!r}")
    if not datasets and not args.all:
        parser.error("name datasets, or use --match or --all")
    options = dict(o.split("=", 1) for o in args.option)

    ingester = Ingester(refresh="always" if args.force else args.refresh)
    try:
        job = ingester.submit(args.source, datasets, start=args.start, end=args.end, options=options)
    except ValueError as error:
        parser.error(str(error))
    try:
        ingester.wait()
    except KeyboardInterrupt:
        log.warning("stopping after the current dataset ...")
        ingester.stop()
    print(f"{job.source}: {summary(job)}")
    for error in job.errors:
        print(f"  {error}", file=sys.stderr)
    if job.state != "done" or job.failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
