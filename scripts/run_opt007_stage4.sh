#!/usr/bin/env bash
set -euo pipefail

DIR="results/opt007"
RAW="$DIR/perf_raw"
REPORT="results/OPT007_perf_analysis.txt"

EVENTS="cycles,instructions,branches,branch-misses"

echo "=========================================="
echo " OPT007 Stage 4 — Hardware Counters"
echo "=========================================="

[[ "$(git branch --show-current)" == "opt007-compiler-analysis" ]] || {
    echo "ERROR: Expected opt007-compiler-analysis branch."
    exit 1
}

[[ -n "${VIRTUAL_ENV:-}" ]] || {
    echo "ERROR: .venv is not active."
    exit 1
}

PARANOID="$(cat /proc/sys/kernel/perf_event_paranoid)"

echo "perf_event_paranoid=$PARANOID"

if (( PARANOID > 1 )); then
    echo "ERROR: perf hardware counters are still restricted."
    echo "Expected perf_event_paranoid <= 1."
    exit 1
fi

mkdir -p "$RAW"
rm -f "$RAW"/*.txt

echo
echo "[1/5] Checking perf..."

perf --version

perf stat \
    -e "$EVENTS" \
    -x, \
    -o "$RAW/sanity.txt" \
    true

if grep -qiE 'not supported|not counted|permission|error' "$RAW/sanity.txt"; then
    echo "ERROR: perf sanity test failed:"
    cat "$RAW/sanity.txt"
    exit 1
fi

echo "perf sanity check PASS"

echo
echo "[2/5] Creating controlled benchmark driver..."

cat > "$DIR/perf_driver.py" <<'PY'
import argparse
import os

os.environ["OMP_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"

import torch

torch.set_num_threads(6)
torch.set_num_interop_threads(6)

from packed_ternary.opt005_extension import load_opt005_extension
from packed_ternary.opt006_extension import load_opt006_extension


parser = argparse.ArgumentParser()
parser.add_argument("--mode", required=True, choices=["opt005", "opt006"])
parser.add_argument("--warmup", type=int, default=3)
parser.add_argument("--iterations", type=int, default=20)
args = parser.parse_args()

torch.manual_seed(20260928)

# Representative packed-ternary matrix workload.
# Shape corresponds to a substantial linear operation while keeping
# Stage 4 focused on the compiled packed operator itself.
batch = 22
in_features = 1024
out_features = 1024
group_size = 128

x = torch.randn(batch, in_features, dtype=torch.float16)

groups_per_row = in_features // group_size
packed_bytes_per_group = group_size // 4

packed = torch.randint(
    0,
    255,
    (out_features, groups_per_row, packed_bytes_per_group),
    dtype=torch.uint8,
)

# Prevent reserved ternary code 11 from appearing.
# Construct valid packed bytes from explicit 2-bit codes.
codes = torch.randint(
    0,
    3,
    (out_features, in_features),
    dtype=torch.uint8,
)

c = codes.reshape(out_features, -1, 4)

packed = (
    c[..., 0]
    | (c[..., 1] << 2)
    | (c[..., 2] << 4)
    | (c[..., 3] << 6)
).contiguous().reshape(-1)

scales = torch.rand(
    out_features,
    groups_per_row,
    dtype=torch.float16
).contiguous()

x = x.contiguous()

if args.mode == "opt005":
    ext = load_opt005_extension()
else:
    ext = load_opt006_extension()

fn = ext.packed_ternary_linear_cpu_parallel

for _ in range(args.warmup):
    y = fn(
        x,
        packed,
        scales,
        in_features,
        group_size,
    )

for _ in range(args.iterations):
    y = fn(
        x,
        packed,
        scales,
        in_features,
        group_size,
    )

# Force result consumption.
print(
    args.mode,
    tuple(y.shape),
    float(y.reshape(-1)[0]),
)
PY

echo
echo "[3/5] Running controlled perf measurements..."

for MODE in opt005 opt006; do
    echo
    echo "Condition: $MODE"

    for RUN in 1 2 3; do
        echo "  run $RUN/3"

        perf stat \
            -e "$EVENTS" \
            -x, \
            -o "$RAW/${MODE}_${RUN}.txt" \
            env PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
            python "$DIR/perf_driver.py" \
                --mode "$MODE" \
                --warmup 3 \
                --iterations 20 \
                > "$RAW/${MODE}_${RUN}_stdout.txt"
    done
done

echo
echo "[4/5] Aggregating hardware counters..."

python - <<'PY' | tee results/opt007/perf_summary.txt
from pathlib import Path
from statistics import mean, stdev

root = Path("results/opt007/perf_raw")

events = [
    "cycles",
    "instructions",
    "branches",
    "branch-misses",
]


def parse(path):
    values = {}

    for line in path.read_text(errors="ignore").splitlines():
        parts = line.split(",")

        if len(parts) < 3:
            continue

        raw = parts[0].strip()
        event = parts[2].strip()

        if event not in events:
            continue

        if (
            not raw
            or "<not" in raw.lower()
            or "not supported" in raw.lower()
        ):
            continue

        try:
            values[event] = float(raw)
        except ValueError:
            pass

    return values


all_results = {}

for mode in ["opt005", "opt006"]:
    runs = []

    for i in range(1, 4):
        path = root / f"{mode}_{i}.txt"
        values = parse(path)

        missing = [e for e in events if e not in values]
        if missing:
            raise SystemExit(
                f"ERROR: {path} missing counters: {missing}"
            )

        runs.append(values)

    all_results[mode] = runs


print("=" * 76)
print("OPT007 HARDWARE COUNTER SUMMARY")
print("=" * 76)

for event in events:
    a = [r[event] for r in all_results["opt005"]]
    b = [r[event] for r in all_results["opt006"]]

    ma = mean(a)
    mb = mean(b)

    delta = ((mb - ma) / ma) * 100.0

    print()
    print(event)
    print(f"  OPT005 mean : {ma:,.3f}")
    print(f"  OPT006 mean : {mb:,.3f}")
    print(f"  change      : {delta:+.3f}%")

    if len(a) > 1:
        print(f"  OPT005 std  : {stdev(a):,.3f}")
        print(f"  OPT006 std  : {stdev(b):,.3f}")


i5 = mean(r["instructions"] for r in all_results["opt005"])
i6 = mean(r["instructions"] for r in all_results["opt006"])

c5 = mean(r["cycles"] for r in all_results["opt005"])
c6 = mean(r["cycles"] for r in all_results["opt006"])

b5 = mean(r["branches"] for r in all_results["opt005"])
b6 = mean(r["branches"] for r in all_results["opt006"])

bm5 = mean(r["branch-misses"] for r in all_results["opt005"])
bm6 = mean(r["branch-misses"] for r in all_results["opt006"])

print()
print("=" * 76)
print("DERIVED METRICS")
print("=" * 76)

print(f"OPT005 IPC: {i5 / c5:.6f}")
print(f"OPT006 IPC: {i6 / c6:.6f}")

print(
    f"OPT005 branch-miss rate: "
    f"{(bm5 / b5) * 100:.6f}%"
)

print(
    f"OPT006 branch-miss rate: "
    f"{(bm6 / b6) * 100:.6f}%"
)

print()
print("NOTE:")
print(
    "perf measures the complete benchmark process, including runtime "
    "overhead, not only the isolated hot function."
)
PY

echo
echo "[5/5] Writing OPT007 report..."

{
    echo "OPT007 — HARDWARE COUNTER ANALYSIS"
    echo "=================================="
    echo
    echo "Comparison:"
    echo "OPT005 frozen comparison-mask packed ternary operator"
    echo "OPT006 frozen fixed lookup-table packed ternary operator"
    echo
    echo "Configuration:"
    echo "CPU execution"
    echo "PyTorch threads: 6"
    echo "Inter-op threads: 6"
    echo "Warmups per process: 3"
    echo "Measured iterations per process: 20"
    echo "Independent processes per condition: 3"
    echo
    echo "Counters:"
    echo "$EVENTS"
    echo
    cat "$DIR/perf_summary.txt"
    echo
    echo "INTERPRETATION LIMIT"
    echo "--------------------"
    echo "Hardware counters describe complete benchmark-process execution."
    echo "Together with Stage-3 function-level disassembly they can support"
    echo "a stronger execution-level comparison, but causal attribution must"
    echo "remain limited to what the measurements directly establish."
} > "$REPORT"

echo
echo "=========================================="
echo " OPT007 Stage 4 complete"
echo "=========================================="
echo
echo "Report:"
echo "  $REPORT"
echo
echo "Summary:"
cat "$DIR/perf_summary.txt"

echo
echo "IMPORTANT:"
echo "OPT007 has NOT been committed or pushed."
echo
echo "After saving the output, restore the original perf restriction with:"
echo "  sudo sysctl kernel.perf_event_paranoid=4"
