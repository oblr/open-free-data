# free-data

Free public time series from German and French sources: a catalog of
sources, plugins that read them into one table format, a local file cache,
and a browser page to search, plot and export them.

## Setup

    python -m venv .venv
    .venv/bin/pip install -r requirements.txt pytest

Run everything from the project folder; it isn't an installable package.

Python 3.11 or newer; the only dependency is numpy.

## Use

    python browser.py                     # then open http://127.0.0.1:8050
    python smard.py --list                # SMARD series ids
    python smard.py 410 4169 --start 2024-01-01 --end 2024-12-31 --out load.csv
    python cache.py                       # what is cached, per source
    python cache.py clear de-smard --older-than 30d

The browser lists every source in `data/`; those with a plugin can be
plotted, also several sources side by side, and downloaded as CSV. It binds
to 127.0.0.1 and has no authentication.

## Layout

| File | What it does |
| --- | --- |
| `data/<id>.json` | One source: id, link, description. |
| `sources.py` | The plugin interface (`Plugin`, `Dataset`, `Option`, `Table`), discovery and the catalog. |
| `plugins/` | One module per readable source, each with a module-level `PLUGIN`. |
| `cache.py` | Downloads, cached under `cache/<source>/` (git-ignored). |
| `smard.py` | Reader for SMARD, the Bundesnetzagentur's electricity market data. |
| `browser.py`, `browser.html` | The local web page and its JSON/CSV API. |

A `Table` has UTC `datetime64[ms]` times, strictly increasing, and float64
values with NaN for gaps. Dates without a UTC offset are read in the
source's own timezone.

## Adding a source

1. Add `data/<country>-<name>.json` with `data-source-id`, `link` and `description`.
2. Add `plugins/<country>_<name>.py` defining a `Plugin` subclass whose `id`
   matches the JSON file, and `PLUGIN = YourPlugin()`. See the docstring of
   `sources.py` and `plugins/de_smard.py`.
3. Download through `cache.Cache(<id>)`, choosing a `max_age` per file by how
   much it can still change.

## Cache

Files live in `cache/` next to the code; set `FREE_DATA_CACHE` to put them
elsewhere. A file is fetched again once it is older than its reader's
`max_age`. If the source can't be reached, the cached copy is used with a
warning.

## Tests

    .venv/bin/python -m pytest

The tests run offline against a temporary cache; any download attempt fails them.
