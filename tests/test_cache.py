from __future__ import annotations

import http.client
import os
import time

import pytest

from free_data import cache
from free_data.cache import Cache, FetchError

download = cache._download  # the real one; conftest replaces it


@pytest.fixture
def downloads(monkeypatch):
    """Serves `body` for every URL and counts the calls; set `error` to fail instead."""

    class Server:
        body = b"new"
        error: FetchError | None = None
        calls = 0

        def __call__(self, url: str) -> bytes:
            self.calls += 1
            if self.error:
                raise self.error
            return self.body

    server = Server()
    monkeypatch.setattr(cache, "_download", server)
    return server


def age(path, seconds: float) -> None:
    then = time.time() - seconds
    os.utime(path, (then, then))


def test_root_follows_the_environment(cache_root):
    assert cache.root() == cache_root
    assert Cache("x").dir == cache_root / "x"


def test_fetch_stores_and_reuses(downloads):
    store = Cache("x")
    assert store.fetch("u", "a/b.json") == b"new"
    downloads.body = b"newer"
    assert store.fetch("u", "a/b.json") == b"new"  # max_age None: kept forever
    assert downloads.calls == 1
    assert not list(store.dir.rglob("*.part"))


def test_stale_files_are_fetched_again(downloads):
    store = Cache("x")
    store.fetch("u", "k", max_age=60)
    age(store.path("k"), 120)
    downloads.body = b"newer"
    assert store.fetch("u", "k", max_age=60) == b"newer"
    assert store.fetch("u", "k", max_age=0) == b"newer"
    assert downloads.calls == 3


def test_unreachable_source_falls_back_to_the_cached_copy(downloads):
    store = Cache("x")
    store.fetch("u", "k")
    downloads.error = FetchError("u", None, "no route")
    with pytest.warns(UserWarning, match="cached copy"):
        assert store.fetch("u", "k", max_age=0) == b"new"


@pytest.mark.parametrize("status", [429, 500, 503])
def test_server_trouble_falls_back_to_the_cached_copy(downloads, status):
    store = Cache("x")
    store.fetch("u", "k")
    downloads.error = FetchError("u", status, f"HTTP {status}")
    with pytest.warns(UserWarning, match="cached copy"):
        assert store.fetch("u", "k", max_age=0) == b"new"


@pytest.mark.parametrize("status", [400, 403, 404])
def test_client_errors_are_raised_even_with_a_cached_copy(downloads, status):
    store = Cache("x")
    store.fetch("u", "k")
    downloads.error = FetchError("u", status, f"HTTP {status}")
    with pytest.raises(FetchError):
        store.fetch("u", "k", max_age=0)


def test_server_trouble_without_a_cached_copy_is_raised(downloads):
    downloads.error = FetchError("u", 503, "HTTP 503")
    with pytest.raises(FetchError):
        Cache("x").fetch("u", "k")


def test_keys_stay_inside_the_source_folder():
    with pytest.raises(ValueError, match="leaves"):
        Cache("x").path("../y/k")


def test_cli_lists_and_clears(downloads, capsys):
    store = Cache("x")
    store.fetch("u", "old")
    store.fetch("u", "new")
    age(store.path("old"), 3 * cache.DAY)
    cache.main(["clear", "--older-than", "2d"])
    assert "x: deleted 1 files" in capsys.readouterr().out
    cache.main([])
    assert "1 files" in capsys.readouterr().out
    cache.main(["clear", "x", "missing"])
    out = capsys.readouterr().out
    assert "x: deleted 1 files" in out and "missing: nothing cached" in out


def test_temporary_failures_are_tried_again(downloads, monkeypatch):
    failures = [FetchError("u", 503, "HTTP 503"), FetchError("u", None, "timeout")]
    real = downloads.__class__.__call__

    def flaky(self, url):
        if failures:
            self.calls += 1
            raise failures.pop(0)
        return real(self, url)

    monkeypatch.setattr(downloads.__class__, "__call__", flaky)
    assert Cache("x").fetch("u", "k") == b"new"
    assert downloads.calls == 3


def test_a_body_cut_off_is_a_temporary_failure(monkeypatch):
    class CutOff:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            raise http.client.IncompleteRead(b"{\"chart", 100)

    monkeypatch.setattr(cache, "_download", download)
    monkeypatch.setattr(cache.urllib.request, "urlopen", lambda request, timeout: CutOff())
    with pytest.raises(FetchError, match="cut off after 7 bytes") as error:
        cache._download("https://example.org/x")
    assert error.value.transient


def test_client_errors_are_not_tried_again(downloads):
    downloads.error = FetchError("u", 404, "HTTP 404")
    with pytest.raises(FetchError):
        Cache("x").fetch("u", "k")
    assert downloads.calls == 1


def test_retries_give_up_after_three_tries(downloads):
    downloads.error = FetchError("u", 503, "HTTP 503")
    with pytest.raises(FetchError):
        Cache("x").fetch("u", "k")
    assert downloads.calls == 3


def test_offline_uses_any_cached_file_and_never_downloads(downloads):
    store = Cache("x")
    store.fetch("u", "k")
    age(store.path("k"), 3600)
    with cache.refreshing("offline"):
        assert store.fetch("u", "k", max_age=0) == b"new"
        with pytest.raises(cache.NotCached, match="not cached"):
            store.fetch("u", "missing")
    assert downloads.calls == 1
