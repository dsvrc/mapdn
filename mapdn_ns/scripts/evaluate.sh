#!/usr/bin/env bash
#  Evaluate trained arms the way MAPDN does (test.py --test-mode batch, 10
#  episodes, no data noise), through the same launcher.
#
#      bash mapdn_ns/scripts/evaluate.sh matd3 1.0 "0 1 2" paper
set -euo pipefail
ALG=${1:-matd3}; SIGMA=${2:-1.0}; SEEDS=${3:-"0 1 2"}; ALIAS=${4:-paper}
PY=${PYTHON:-python}
for s in $SEEDS; do
  $PY mapdn_ns/run.py test --alg "$ALG" --arm blind --sigma 0        --alias "$ALIAS" --seed "$s" --test-mode batch
  for arm in blind pact oracle pactoff intercept; do
    $PY mapdn_ns/run.py test --alg "$ALG" --arm "$arm" --sigma "$SIGMA" --alias "$ALIAS" --seed "$s" --test-mode batch
  done
  for arm in blind pact intercept; do
    $PY mapdn_ns/run.py test --alg "$ALG" --arm "$arm" --sigma "$SIGMA" --alias "$ALIAS" --seed "$s" --test-mode batch --direct
  done
done

# the sigma = 0 policy under the disturbance, zero-shot: how much of the
# blind-at-sigma* loss is the disturbance itself, and how much is what the
# learner did about it
for s in $SEEDS; do
  $PY mapdn_ns/run.py test --alg "$ALG" --arm blind --sigma "$SIGMA" --alias "$ALIAS" --seed "$s" --test-mode batch \
      --load-alias "$ALIAS-blind-s0-seed$s"
done
