#!/usr/bin/env bash
set -euo pipefail

RAW_DIR="results/opt009_raw"
RUNTIME="scripts/opt009_benchmark_runtime.py"

echo "============================================================"
echo " OPT009 Stage 4 — Resume Candidate Benchmark"
echo "============================================================"

echo
echo "[1/3] Verifying preserved OPT008 baseline..."

for run in 1 2 3
do
    test -s "${RAW_DIR}/avx2_run${run}.json"
    test -s "${RAW_DIR}/avx2_run${run}.time.txt"
done

echo "PASS: all three baseline processes preserved."

echo
echo "[2/3] Running three official OPT009 candidate processes..."

run_candidate () {
    local run="$1"

    local log_file="${RAW_DIR}/blocked_avx2_run${run}.log"
    local json_file="${RAW_DIR}/blocked_avx2_run${run}.json"
    local time_file="${RAW_DIR}/blocked_avx2_run${run}.time.txt"

    echo
    echo "------------------------------------------------------------"
    echo "OPT009 blocked_avx2 | Process ${run}/3"
    echo "------------------------------------------------------------"

    /usr/bin/time -v \
        -o "$time_file" \
        env \
        PYTHONPATH="$PWD" \
        OMP_NUM_THREADS=6 \
        MKL_NUM_THREADS=6 \
        python "$RUNTIME" \
            --mode blocked_avx2 \
        2>&1 \
        | tee "$log_file"

    python - "$log_file" "$json_file" <<'PY'
import json
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
json_path = Path(sys.argv[2])

text = log_path.read_text()

marker = "RESULT_JSON"

if marker not in text:
    raise SystemExit(
        f"ERROR: RESULT_JSON missing in {log_path}"
    )

tail = text.split(marker, 1)[1]

start = tail.find("{")

if start < 0:
    raise SystemExit(
        f"ERROR: JSON missing after RESULT_JSON in {log_path}"
    )

obj, _ = json.JSONDecoder().raw_decode(
    tail[start:]
)

json_path.write_text(
    json.dumps(
        obj,
        indent=2,
        sort_keys=True,
    )
    + "\n"
)

print(f"PASS: extracted {json_path}")
PY
}

for run in 1 2 3
do
    run_candidate "$run"
done

echo
echo "[3/3] Candidate collection complete."

for run in 1 2 3
do
    test -s "${RAW_DIR}/blocked_avx2_run${run}.json"
    test -s "${RAW_DIR}/blocked_avx2_run${run}.time.txt"
done

echo
echo "PASS: 3 baseline + 3 candidate processes available."
echo "Ready for final OPT009 aggregation."
