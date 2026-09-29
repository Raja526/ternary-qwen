#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6

RAW_DIR="results/opt010_raw"

mkdir -p "$RAW_DIR"

MODES=(
    avx2
    grain16
    grain64
    grain256
)

echo "============================================================"
echo " OPT010 — THREAD-SCHEDULING BENCHMARK"
echo "============================================================"
echo
echo "Threads          : 6"
echo "Warmups/process  : 2"
echo "Timed/process    : 5"
echo "Processes/mode   : 3"
echo

for mode in "${MODES[@]}"; do
    echo
    echo "============================================================"
    echo " MODE: ${mode}"
    echo "============================================================"

    for run in 1 2 3; do
        echo
        echo "--- ${mode} / process ${run} ---"

        /usr/bin/time \
            -v \
            -o "${RAW_DIR}/${mode}_run${run}.time.txt" \
            python scripts/opt010_benchmark_runtime.py \
                --mode "$mode" \
            2>&1 \
            | tee \
                "${RAW_DIR}/${mode}_run${run}.log"
    done
done

echo
echo "============================================================"
echo " OPT010 RAW BENCHMARK COMPLETE"
echo "============================================================"
