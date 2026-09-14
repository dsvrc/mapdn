#!/usr/bin/env bash
#  The paired sweep: every arm through the identical launcher, severity from
#  outside the method (P-10.1).  Run from the repository root.
#
#      bash mapdn_ns/scripts/sweep.sh matd3 1.0 "0 1 2" paper
#      bash mapdn_ns/scripts/sweep.sh mappo 1.0 "0 1 2" paper
#
#  $1 algorithm   (matd3 | mappo | maddpg | ...)
#  $2 sigma*      the operating point from mapdn_ns/RESULTS.md
#  $3 seeds       a quoted list
#  $4 alias       a tag for this campaign
#
#  Runs are sequential; put `&` after a line and cap concurrency yourself if
#  you have the cores (each run is one CPU-bound pandapower process, ~5 h on
#  a laptop for 400 episodes).
set -euo pipefail
ALG=${1:-matd3}; SIGMA=${2:-1.0}; SEEDS=${3:-"0 1 2"}; ALIAS=${4:-paper}
PY=${PYTHON:-python}

for s in $SEEDS; do
  # 1. no NS -- the stock task, B0
  $PY mapdn_ns/run.py train --alg "$ALG" --arm blind     --sigma 0        --alias "$ALIAS" --seed "$s"
  # 2. NS -- the existing algorithm, blind
  $PY mapdn_ns/run.py train --alg "$ALG" --arm blind     --sigma "$SIGMA" --alias "$ALIAS" --seed "$s"
  # 3. NS -- the same algorithm with PACT
  $PY mapdn_ns/run.py train --alg "$ALG" --arm pact      --sigma "$SIGMA" --alias "$ALIAS" --seed "$s"
  # the ablation arms at the operating point
  $PY mapdn_ns/run.py train --alg "$ALG" --arm oracle    --sigma "$SIGMA" --alias "$ALIAS" --seed "$s"
  $PY mapdn_ns/run.py train --alg "$ALG" --arm pactoff   --sigma "$SIGMA" --alias "$ALIAS" --seed "$s"
  $PY mapdn_ns/run.py train --alg "$ALG" --arm intercept --sigma "$SIGMA" --alias "$ALIAS" --seed "$s"
  # the (B) control: same driver, same scale, no neighbours
  $PY mapdn_ns/run.py train --alg "$ALG" --arm blind     --sigma "$SIGMA" --alias "$ALIAS" --seed "$s" --direct
  $PY mapdn_ns/run.py train --alg "$ALG" --arm pact      --sigma "$SIGMA" --alias "$ALIAS" --seed "$s" --direct
  $PY mapdn_ns/run.py train --alg "$ALG" --arm intercept --sigma "$SIGMA" --alias "$ALIAS" --seed "$s" --direct
done
