#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6

OUTDIR="results/opt009_block4_block16_raw"

mkdir -p "$OUTDIR"

echo "========================================================================"
echo "OPT009 — OFFICIAL BLOCK-4 / BLOCK-16 BENCHMARK"
echo "========================================================================"
echo
echo "Frozen protocol:"
echo "  CPU threads          : 6"
echo "  Inter-op threads     : 6"
echo "  Warmups/process      : 2"
echo "  Timed runs/process   : 5"
echo "  Processes/condition  : 3"
echo "  Input tokens         : 22"
echo
echo "Existing OPT008/block-8 measurements are NOT modified."
echo

run_condition() {
    local mode="$1"
    local label="$2"

    for run in 1 2 3; do
        json_log="$OUTDIR/${label}_run${run}.log"
        time_log="$OUTDIR/${label}_run${run}.time.txt"

        echo "------------------------------------------------------------------------"
        echo "${label}: official process ${run}/3"
        echo "------------------------------------------------------------------------"

        /usr/bin/time \
            -v \
            -o "$time_log" \
            python scripts/opt009_benchmark_runtime.py \
                --mode "$mode" \
            | tee "$json_log"

        echo
    done
}

run_condition \
    "blocked4_avx2" \
    "block4"

run_condition \
    "blocked16_avx2" \
    "block16"

echo "========================================================================"
echo "OPT009 BLOCK-4 / BLOCK-16 COLLECTION COMPLETE"
echo "========================================================================"
