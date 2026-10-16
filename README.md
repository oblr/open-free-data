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
    python eco2mix.py --list              # RTE éCO2mix fields
    python eco2mix.py consommation taux_co2 --resolution day --start 2024-01-01
    python pegelonline.py --list rhein    # gauging stations, by name or water
    python pegelonline.py KÖLN/W KÖLN/Q --resolution native
    python hubeau.py --list seine paris   # French river gauges
    python hubeau.py F700000103/Q --start 2016-05-01 --end 2016-06-30
    python dwd.py --list potsdam          # DWD weather stations
    python dwd.py 03987/TT_TU 03987/R1 --resolution day --start 2025-07-01
    python cache.py                       # what is cached, per source
    python cache.py clear de-smard --older-than 30d

The browser lists every source in `data/`; those with a plugin can be
plotted, also several sources side by side (one axis per unit), and
downloaded as CSV. A source's first list of datasets can take up to a
minute (Hub'eau's station list); it is cached for a day. The browser binds
to 127.0.0.1 and has no authentication.

## Layout

| File | What it does |
| --- | --- |
| `data/<id>.json` | One source: id, link, description. |
| `sources.py` | The plugin interface (`Plugin`, `Dataset`, `Option`, `Table`), discovery and the catalog. |
| `plugins/` | One module per readable source, each with a module-level `PLUGIN`. |
| `cache.py` | Downloads, cached under `cache/<source>/` (git-ignored). |
| `smard.py` | Reader for SMARD, the Bundesnetzagentur's electricity market data. |
| `eco2mix.py` | Reader for RTE éCO2mix, France's electricity mix (via ODRÉ). |
| `pegelonline.py` | Reader for PEGELONLINE, the gauges of the German federal waterways (last 31 days only). |
| `hubeau.py` | Reader for Hub'eau, the river gauges of France (daily flows for decades, the rest 30 days). |
| `dwd.py` | Reader for the DWD's hourly weather observations at German stations. |
| `browser.py`, `browser.html` | The local web page and its JSON/CSV API. |

The readers' command lines take inclusive dates in the source's timezone
(a bare end date means the whole day) and default to the last 30 days.

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
`max_age`. If the source can't be reached or answers with a server error,
the download is tried twice more, then the cached copy is used with a warning.

## Tests

    .venv/bin/python -m pytest

The tests run offline against a temporary cache; any download attempt fails them.
