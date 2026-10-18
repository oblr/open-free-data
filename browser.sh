#!/bin/sh
# Start the free-data browser on http://127.0.0.1:8050.
#
#     ./browser.sh                  # sets up .venv on the first run
#     ./browser.sh --port 9000      # arguments go to the browser (--host, --port)
#
# The page needs web/dist, which isn't committed: it is built here when
# missing (needs Node: cd web && npm install && npm run build).
set -eu
cd "$(dirname "$0")"

if [ ! -x .venv/bin/free-data ]; then
    echo "setting up .venv ..."
    [ -d .venv ] || python3 -m venv .venv
    .venv/bin/pip install --quiet -e .
fi

if [ ! -f web/dist/main.js ]; then
    echo "building the page in web/dist ..."
    (cd web && npm install --silent && npm run --silent build)
fi

exec .venv/bin/free-data "$@"
