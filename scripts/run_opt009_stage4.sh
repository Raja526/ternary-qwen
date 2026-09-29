#!/usr/bin/env bash
set -euo pipefail

echo "============================================================"
echo " OPT009 Stage 4 — Controlled CPU Benchmark"
echo "============================================================"

EXPECTED_BRANCH="opt009-cache-blocking"
EXPECTED_PARENT="d3a870820d26e18bf0a40f5d21b3bc48c909e874"

RUNTIME="scripts/opt009_benchmark_runtime.py"
RAW_DIR="results/opt009_raw"
FINAL_REPORT="results/OPT009_block8_cpu_benchmark.txt"

echo
echo "[1/5] Verifying experiment state..."

if [[ "$(git branch --show-current)" != "$EXPECTED_BRANCH" ]]; then
    echo "ERROR: expected branch $EXPECTED_BRANCH"
    exit 1
fi

if [[ "$(git rev-parse HEAD)" != "$EXPECTED_PARENT" ]]; then
    echo "ERROR: HEAD changed before OPT009 freeze."
    exit 1
fi

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "ERROR: .venv is not active."
    exit 1
fi

test -f "$RUNTIME"
test -f results/OPT009_operator_equivalence.txt
test -f results/OPT009_full_model_equivalence.txt

grep -q \
"PASS: OPT009 output-row blocking is bit-exact with frozen OPT008." \
results/OPT009_operator_equivalence.txt

grep -q \
"PASS: OPT009 full-model behavior is bit-exact with frozen OPT008." \
results/OPT009_full_model_equivalence.txt

python -m py_compile "$RUNTIME"

HELP="$(
    PYTHONPATH="$PWD" \
    python "$RUNTIME" --help
)"

grep -q "{avx2,blocked_avx2}" <<< "$HELP"

grep -q \
"load_blocked_avx2_runtime_modules" \
"$RUNTIME"

echo "PASS: branch, correctness gates, and runtime verified."

echo
echo "[2/5] Preparing raw benchmark directory..."

rm -rf "$RAW_DIR"
mkdir -p "$RAW_DIR"

run_one () {
    local mode="$1"
    local run="$2"

    local log_file="${RAW_DIR}/${mode}_run${run}.log"
    local json_file="${RAW_DIR}/${mode}_run${run}.json"
    local time_file="${RAW_DIR}/${mode}_run${run}.time.txt"

    echo
    echo "------------------------------------------------------------"
    echo "Mode: ${mode} | Process: ${run}/3"
    echo "------------------------------------------------------------"

    /usr/bin/time -v \
        -o "$time_file" \
        env \
        PYTHONPATH="$PWD" \
        OMP_NUM_THREADS=6 \
        MKL_NUM_THREADS=6 \
        python "$RUNTIME" \
            --mode "$mode" \
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
        f"ERROR: JSON object missing after RESULT_JSON "
        f"in {log_path}"
    )

decoder = json.JSONDecoder()

obj, _ = decoder.raw_decode(
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

echo
echo "[3/5] Running frozen OPT008 baseline..."

for run in 1 2 3
do
    run_one avx2 "$run"
done

echo
echo "[4/5] Running OPT009 block-8 candidate..."

for run in 1 2 3
do
    run_one blocked_avx2 "$run"
done

echo
echo "[5/5] Aggregating controlled results..."

python - <<'PY' | tee "$FINAL_REPORT"
import json
import re
import statistics
from pathlib import Path

RAW = Path("results/opt009_raw")


def load_condition(mode):
    rows = []

    for run in range(1, 4):
        json_path = RAW / f"{mode}_run{run}.json"
        time_path = RAW / f"{mode}_run{run}.time.txt"

        data = json.loads(
            json_path.read_text()
        )

        time_text = time_path.read_text()

        match = re.search(
            r"Maximum resident set size "
            r"\(kbytes\):\s*(\d+)",
            time_text,
        )

        if match is None:
            raise RuntimeError(
                f"GNU max RSS missing: {time_path}"
            )

        rows.append(
            {
                "run": run,
                "data": data,
                "gnu_rss_kb": int(match.group(1)),
            }
        )

    return rows


baseline = load_condition("avx2")
candidate = load_condition("blocked_avx2")

all_rows = baseline + candidate


def require_constant(field, expected=None):
    values = [
        row["data"][field]
        for row in all_rows
    ]

    first = values[0]

    if any(value != first for value in values):
        raise RuntimeError(
            f"Protocol mismatch for {field}: {values}"
        )

    if expected is not None and first != expected:
        raise RuntimeError(
            f"{field}: expected {expected!r}, "
            f"observed {first!r}"
        )

    return first


threads = require_constant(
    "torch_threads",
    6,
)

interop = require_constant(
    "torch_interop_threads",
    6,
)

warmups = require_constant(
    "warmup_runs",
    2,
)

timed_runs = require_constant(
    "timed_runs",
    5,
)

tokens = require_constant(
    "input_tokens",
    22,
)

matrices = require_constant(
    "selected_matrices",
    3,
)

parameters = require_constant(
    "selected_parameters",
    8912896,
)

fp16_bytes = require_constant(
    "selected_fp16_bytes",
    17825792,
)

packed_bytes = require_constant(
    "selected_packed_bytes",
    2367488,
)


def collect_latencies(rows):
    values = []

    for row in rows:
        current = row["data"]["latencies_ms"]

        if len(current) != 5:
            raise RuntimeError(
                f"Run {row['run']} contains "
                f"{len(current)} timings; expected 5."
            )

        values.extend(
            float(value)
            for value in current
        )

    return values


base_lat = collect_latencies(baseline)
cand_lat = collect_latencies(candidate)

if len(base_lat) != 15:
    raise RuntimeError(
        f"Baseline contains {len(base_lat)} timings; "
        "expected 15."
    )

if len(cand_lat) != 15:
    raise RuntimeError(
        f"Candidate contains {len(cand_lat)} timings; "
        "expected 15."
    )


def stats(values):
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "std": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
    }


bs = stats(base_lat)
cs = stats(cand_lat)

change_pct = (
    (cs["mean"] - bs["mean"])
    / bs["mean"]
    * 100.0
)


def rss_mean(rows, key):
    return statistics.mean(
        float(
            row["data"]["process_rss_mib"][key]
        )
        for row in rows
    )


base_ready = rss_mean(
    baseline,
    "ready",
)

cand_ready = rss_mean(
    candidate,
    "ready",
)

base_warm = rss_mean(
    baseline,
    "after_warmup",
)

cand_warm = rss_mean(
    candidate,
    "after_warmup",
)

base_timed = rss_mean(
    baseline,
    "timed_max",
)

cand_timed = rss_mean(
    candidate,
    "timed_max",
)

base_gnu = statistics.mean(
    row["gnu_rss_kb"]
    for row in baseline
)

cand_gnu = statistics.mean(
    row["gnu_rss_kb"]
    for row in candidate
)


print("=" * 72)
print("OPT009 — OUTPUT-ROW BLOCKING CPU BENCHMARK")
print("=" * 72)

print()
print("Baseline  : OPT008 explicit AVX2 packed decode")
print("Candidate : OPT009 AVX2 + output-row block size 8")

print()
print("Controlled variable:")
print("  output rows per parallel job: 1 -> 8")

print()
print("Held constant:")
print("  packed representation")
print("  AVX2 packed-code decoding")
print("  group size 128")
print("  FP32 accumulation order within each output row")
print("  compiler flags -O2 -mavx2 -mno-fma")
print("  selected matrices")
print("  model and input")
print("  CPU thread count")
print("  benchmark protocol")

print()
print("PROTOCOL")
print("-" * 72)

print(f"CPU threads              : {threads}")
print(f"Inter-op threads         : {interop}")
print(f"Input tokens             : {tokens}")
print(f"Warmups/process          : {warmups}")
print(f"Timed forwards/process   : {timed_runs}")
print("Processes/condition      : 3")
print("Timed forwards/condition : 15")
print(f"Selected matrices        : {matrices}")
print(f"Selected parameters      : {parameters}")
print(f"FP16 equivalent bytes    : {fp16_bytes}")
print(f"Packed bytes             : {packed_bytes}")

print()
print("LATENCY")
print("-" * 72)

print(
    f"OPT008 mean   : {bs['mean']:.3f} ms"
)
print(
    f"OPT009 mean   : {cs['mean']:.3f} ms"
)
print(
    f"Change        : {change_pct:+.3f}%"
)

print()
print(
    f"OPT008 median : {bs['median']:.3f} ms"
)
print(
    f"OPT009 median : {cs['median']:.3f} ms"
)

print()
print(
    f"OPT008 std    : {bs['std']:.3f} ms"
)
print(
    f"OPT009 std    : {cs['std']:.3f} ms"
)

print()
print(
    "OPT008 min/max:",
    f"{bs['min']:.3f} / {bs['max']:.3f} ms",
)

print(
    "OPT009 min/max:",
    f"{cs['min']:.3f} / {cs['max']:.3f} ms",
)

print()
print("MEMORY")
print("-" * 72)

print(
    "Ready RSS mean:",
    f"{base_ready:.3f} -> {cand_ready:.3f} MiB",
    f"({cand_ready-base_ready:+.3f} MiB)",
)

print(
    "After-warmup RSS mean:",
    f"{base_warm:.3f} -> {cand_warm:.3f} MiB",
    f"({cand_warm-base_warm:+.3f} MiB)",
)

print(
    "Timed-max RSS mean:",
    f"{base_timed:.3f} -> {cand_timed:.3f} MiB",
    f"({cand_timed-base_timed:+.3f} MiB)",
)

print(
    "GNU max RSS mean:",
    f"{base_gnu:.3f} -> {cand_gnu:.3f} KB",
    f"({cand_gnu-base_gnu:+.3f} KB)",
)

print()
print("CORRECTNESS PREREQUISITES")
print("-" * 72)

print(
    "Operator equivalence : PASS "
    "(35,555 outputs, 0 mismatches)"
)

print(
    "Full-model logits    : PASS "
    "(5,463,040 logits, 0 mismatches)"
)

print(
    "Top-1 behavior       : PASS (22/22)"
)

print()
print("=" * 72)

if change_pct < 0:
    print(
        "OBSERVED RESULT: OPT009 block-8 reduced mean latency "
        "under this controlled benchmark."
    )
elif change_pct > 0:
    print(
        "OBSERVED RESULT: OPT009 block-8 increased mean latency "
        "under this controlled benchmark."
    )
else:
    print(
        "OBSERVED RESULT: mean latency was unchanged."
    )

print("=" * 72)
PY

echo
echo "============================================================"
echo " OPT009 STAGE 4 COMPLETE"
echo "============================================================"
echo
echo "Final report:"
echo "  $FINAL_REPORT"
echo
echo "No commit or push performed."
