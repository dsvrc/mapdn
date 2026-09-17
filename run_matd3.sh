#!/bin/bash

python mapdn_ns/run.py train \
    --alg matd3 \
    --arm blind \
    --sigma 0 \
    --alias paper \
    --seed 2 \
    --scenario case141_3min_final \
    > matd3_sig0_2.log 2>&1

if [ $? -eq 0 ]; then
    python mapdn_ns/run.py train \
        --alg matd3 \
        --arm blind \
        --sigma 0 \
        --alias paper \
        --seed 3 \
        --scenario case141_3min_final \
        > matd3_sig0_3.log 2>&1
else
    echo "Seed 3 failed. Seed 4 will not start."
fi
