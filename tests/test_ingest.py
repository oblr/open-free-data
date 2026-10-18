"""Background ingest: throttled, and nothing cached is downloaded twice."""

from __future__ import annotations

import threading
import time

import pytest

from free_data import cache, ingest
from free_data.cache import Cache, FetchError, Throttle
from free_data.sources import Dataset, Plugin, Table


class Server:
    """Serves every URL and records when; `fail` maps a call number to an error, `broken` a URL."""

    def __init__(self) -> None:
        self.urls: list[str] = []
        self.times: list[float] = []
        self.fail: dict[int, FetchError] = {}
        self.broken: dict[str, FetchError] = {}
        self.lock = threading.Lock()

    def __call__(self, url: str) -> bytes:
        with self.lock:
            self.urls.append(url)
            self.times.append(time.monotonic())
            error = self.fail.get(len(self.urls)) or self.broken.get(url)
        if error:
            raise error
        return b"{}"


@pytest.fixture
def server(monkeypatch) -> Server:
    s = Server()
    monkeypatch.setattr(cache, "_download", s)
    return s


class Fake(Plugin):
    """Three datasets; reading one fetches its own file, each in a thread of a cache.Pool."""

    id = "xx-fake"
    title = "Fake"
    timezone = "UTC"

    def datasets(self):
        Cache(self.id).fetch("https://fake.example/list", "list.json", max_age=0)
        return [Dataset(i, f"Station {i}", group="Ströme") for i in ("a", "b", "c")]

    def read(self, datasets, start, end, options):
        store = Cache(self.id)
        with cache.Pool(4) as pool:
            list(pool.map(lambda d: store.fetch(f"https://fake.example/{d}", f"{d}.json", max_age=0), datasets))
        return Table((), None, None)


@pytest.fixture
def fake(monkeypatch):
    plugin = Fake()
    monkeypatch.setattr(ingest.sources, "discover", lambda: {plugin.id: plugin})
    return plugin


def run(**kwargs) -> ingest.Job:
    ingester = ingest.Ingester(chunk=2, **{k: v for k, v in kwargs.items() if k == "refresh"})
    job = ingester.submit("xx-fake", kwargs.get("datasets", ()), start="2025-01-01", end="2025-01-31")
    assert ingester.wait(timeout=5)
    return job


def test_a_job_reads_every_dataset_into_the_cache(server, fake, cache_root):
    job = run()
    assert job.state == "done" and (job.total, job.read) == (3, 3)
    assert job.downloaded == 4 and job.cached == 0  # the list and three series
    assert sorted(p.name for p in (cache_root / "xx-fake").iterdir()) == ["a.json", "b.json", "c.json", "list.json"]


def test_nothing_cached_is_downloaded_again_unless_forced(server, fake):
    run()
    again = run()  # every reader asked for max_age=0, but ingest keeps cached files
    assert (again.downloaded, again.cached) == (0, 4) and len(server.urls) == 4
    forced = run(refresh="always")
    assert forced.downloaded == 4 and len(server.urls) == 8


def test_only_what_is_missing_is_downloaded(server, fake):
    run(datasets=["a"])
    job = run(datasets=["a", "b"])
    assert (job.downloaded, job.cached) == (1, 1)
    assert server.urls[-1] == "https://fake.example/b"


def test_the_same_job_is_queued_once(server, fake):
    ingester = ingest.Ingester()
    first = ingester.submit("xx-fake", ["a"], start="2025-01-01", end="2025-01-02")
    assert ingester.submit("xx-fake", ["a"], start="2025-01-01", end="2025-01-02") is first or first.finished
    assert ingester.wait(timeout=5)


def test_a_failing_dataset_is_named_and_the_rest_still_read(server, fake, caplog):
    server.broken = {"https://fake.example/a": FetchError("u", 404, "HTTP 404")}
    with caplog.at_level("INFO", logger="free_data"):
        job = run()
    assert job.state == "done" and (job.read, job.failed) == (2, 1)
    assert job.errors == ["a: HTTP 404"]  # its chunk (a, b) was read again one by one
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert warnings[0] == "xx-fake: a failed: HTTP 404"
    assert "1 failed" in warnings[-1]  # the summary


def test_progress_is_logged(server, fake, caplog):
    with caplog.at_level("DEBUG", logger="free_data"):
        run()
        run()
    lines = [r.getMessage() for r in caplog.records]
    assert "xx-fake: 3 datasets, 2025-01-01 to 2025-01-31, refresh never" in lines
    assert any(line.startswith("xx-fake: [2/3  67%] 3 downloaded, 0 cached, 0 failed") and "left" in line
               for line in lines)
    assert any(line.startswith("xx-fake: downloaded a.json (2 B, ") for line in lines)
    assert "xx-fake: cached a.json" in lines  # the second run
    assert lines[-1].startswith("xx-fake: done, 3/3 datasets read, 0 failed")


def test_slowing_down_is_a_warning(caplog):
    with caplog.at_level("WARNING", logger="free_data"):
        Throttle(rate=4).slow_down("https://busy.example/a", retry_after=30)
    assert caplog.messages == ["busy.example asks us to slow down: now 2 requests per second, nothing for 30 s"]


def test_bad_input_fails_before_the_thread(fake):
    with pytest.raises(ValueError, match="no plugin"):
        ingest.Ingester().submit("nope", start="2025-01-01", end="2025-01-02")
    with pytest.raises(ValueError, match="not a date"):
        ingest.Ingester().submit("xx-fake", start="soon", end="2025-01-02")


def test_constrained_sources_only_when_allowed(fake, server, monkeypatch):
    monkeypatch.setattr(ingest.sources, "constraints", lambda source: ["personal-use", "unofficial-api"])
    with pytest.raises(ValueError, match="personal-use.*--allow-constrained"):
        ingest.Ingester().submit("xx-fake", ["a"], start="2025-01-01", end="2025-01-02")
    ingester = ingest.Ingester()
    job = ingester.submit("xx-fake", ["a"], start="2025-01-01", end="2025-01-02", allow_constrained=True)
    assert ingester.wait(timeout=5) and job.state == "done"


def test_requests_to_one_host_are_spaced(server, monkeypatch):
    monkeypatch.setattr(cache, "THROTTLE", Throttle(rate=20))  # 50 ms apart
    store = Cache("x")
    with cache.Pool(4) as pool:
        list(pool.map(lambda i: store.fetch(f"https://one.example/{i}", f"{i}"), range(5)))
    store.fetch("https://other.example/x", "other")  # another host doesn't wait for this one
    gaps = [b - a for a, b in zip(server.times[:5], server.times[1:5], strict=False)]
    assert min(gaps) >= 0.045
    assert server.times[5] - server.times[4] < 0.045


def test_a_429_halves_the_rate_and_holds_the_host(monkeypatch):
    throttle = Throttle(rate=4, host_rates={})
    throttle.slow_down("https://busy.example/a", retry_after=0.2)
    assert throttle.rate_for("busy.example") == 2
    assert throttle.wait("https://busy.example/b") == pytest.approx(0.2, abs=0.05)
    assert throttle.rate_for("calm.example") == 4


def test_retry_after_is_waited_for(server, monkeypatch):
    monkeypatch.setattr(cache, "THROTTLE", Throttle(rate=0))
    server.fail = {1: FetchError("u", 429, "HTTP 429", retry_after=0.1)}
    began = time.monotonic()
    assert Cache("x").fetch("https://busy.example/a", "a") == b"{}"
    assert time.monotonic() - began >= 0.1 and len(server.urls) == 2


@pytest.mark.parametrize("header, seconds", [("3", 3.0), (None, None), ("soon", None)])
def test_retry_after_header(header, seconds):
    assert cache._retry_after(header) == seconds


def test_matching_folds_case_and_accents(fake, server):
    assert ingest._matching("xx-fake", ["strome", "station"]) == ["a", "b", "c"]
    assert ingest._matching("xx-fake", ["STATION", "b"]) == ["b"]
