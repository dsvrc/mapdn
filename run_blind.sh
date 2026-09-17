#!/bin/bash

python mapdn_ns/run.py train \
    --alg mappo \
    --arm blind \
    --sigma 3 \
    --alias paper \
    --seed 2 \
    --scenario case141_3min_final \
    > mappo_sig3_2.log 2>&1

if [ $? -eq 0 ]; then
    python mapdn_ns/run.py train \
        --alg mappo \
        --arm blind \
        --sigma 3 \
        --alias paper \
        --seed 3 \
        --scenario case141_3min_final \
        > mappo_sig3_3.log 2>&1
else
    echo "Seed 3 failed. Seed 4 will not start."
fi
