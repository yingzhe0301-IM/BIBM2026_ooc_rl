#!/bin/bash
# Rebuild every table and figure of the paper from the public sources.
#   prep.py           downloads the three cohorts, writes out/processed, draws Figs 2-3
#   protocol.py       runs the three-stage protocol, writes out/results, draws Fig 4
#   stress_test.py    runs the simulations, writes out/results/stress_*.csv, draws Fig 5
#   fig1_overview.py  draws Fig 1
# dataset_search.py queries live services and is not rerun here (see its docstring).
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-python3}"
export SOURCE_DATE_EPOCH=0  # fixed PDF timestamps, so unchanged figures stay byte-identical

"$PY" prep.py
"$PY" protocol.py
"$PY" stress_test.py
"$PY" fig1_overview.py
