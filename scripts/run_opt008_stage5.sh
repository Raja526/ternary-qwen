#!/usr/bin/env bash
set -euo pipefail

echo "============================================================"
echo " OPT008 Stage 5 — Consolidation / Freeze Preparation"
echo "============================================================"

EXPECTED_BRANCH="opt008-avx2-ternary-compute"
EXPECTED_PARENT="cd2f52b"

# ------------------------------------------------------------
# 1. Verify repository state
# ------------------------------------------------------------

echo
echo "[1/6] Verifying repository state..."

CURRENT_BRANCH="$(git branch --show-current)"
CURRENT_HEAD="$(git rev-parse HEAD)"

echo "Branch : $CURRENT_BRANCH"
echo "HEAD   : $CURRENT_HEAD"

if [[ "$CURRENT_BRANCH" != "$EXPECTED_BRANCH" ]]; then
    echo "ERROR: Expected branch:"
    echo "  $EXPECTED_BRANCH"
    exit 1
fi

if [[ "$CURRENT_HEAD" != "$(git rev-parse "$EXPECTED_PARENT")" ]]; then
    echo "ERROR: HEAD moved away from frozen OPT007."
    git log -3 --oneline
    exit 1
fi

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "ERROR: .venv is not active."
    exit 1
fi

echo "PASS: repository state."

# ------------------------------------------------------------
# 2. Verify all OPT008 scientific gates
# ------------------------------------------------------------

echo
echo "[2/6] Verifying OPT008 gates..."

test -s results/OPT008_operator_equivalence.txt
test -s results/OPT008_avx2_machine_code.txt
test -s results/OPT008_full_model_equivalence.txt
test -s results/OPT008_avx2_cpu_benchmark.txt

grep -q \
"PASS: OPT008 operator is bit-exact with frozen OPT006." \
results/OPT008_operator_equivalence.txt

grep -q \
"Explicit AVX2 instructions are present" \
results/OPT008_avx2_machine_code.txt

grep -q \
"OPT008 full-model runtime is bit-exact with frozen OPT006." \
results/OPT008_full_model_equivalence.txt

grep -q \
"OPT008 mean latency change: -8.307%" \
results/OPT008_avx2_cpu_benchmark.txt

grep -q \
"OPT006 baseline mean ms: 2175.356" \
results/OPT008_avx2_cpu_benchmark.txt

grep -q \
"OPT008 AVX2 mean ms: 1994.648" \
results/OPT008_avx2_cpu_benchmark.txt

echo "PASS: operator equivalence."
echo "PASS: AVX2 machine-code verification."
echo "PASS: full-model equivalence."
echo "PASS: controlled performance benchmark."

# ------------------------------------------------------------
# 3. Cosmetic cleanup
# ------------------------------------------------------------

echo
echo "[3/6] Cleaning experiment labels..."

python - <<'PY'
from pathlib import Path

p = Path(
    "scripts/opt008_benchmark_runtime.py"
)

s = p.read_text()

# Cosmetic only. Do not alter benchmark behavior.
s = s.replace(
    "OPT006 BENCHMARK —",
    "OPT008 BENCHMARK —",
)

s = s.replace(
    "OPT006 {mode.upper()} BENCHMARK: PASS",
    "OPT008 {mode.upper()} BENCHMARK: PASS",
)

p.write_text(s)

print(
    "PASS: benchmark display labels normalized."
)
PY

# Ensure the cosmetic edit did not touch protocol constants.
PYTHONPATH="$PWD" python - <<'PY'
from pathlib import Path

s = Path(
    "scripts/opt008_benchmark_runtime.py"
).read_text()

required = [
    "WARMUP_RUNS = 2",
    "TIMED_RUNS = 5",
    "torch.set_num_threads(6)",
    "torch.set_num_interop_threads(6)",
    '"model.layers.3.self_attn.k_proj"',
    '"model.layers.4.linear_attn.out_proj"',
    '"model.layers.4.linear_attn.in_proj_qkv"',
]

for item in required:
    if item not in s:
        raise SystemExit(
            f"ERROR: benchmark invariant missing: {item}"
        )

print(
    "PASS: benchmark protocol invariants unchanged."
)
PY

# ------------------------------------------------------------
# 4. Create final OPT008 consolidation report
# ------------------------------------------------------------

echo
echo "[4/6] Writing final OPT008 report..."

cat > results/OPT008_final_report.txt <<'REPORT'
OPT008 — EXPLICIT AVX2 PACKED TERNARY CPU COMPUTE
================================================================================

Status
--------------------------------------------------------------------------------
OPT008 passed all correctness, implementation, full-model, and controlled
performance gates.

Parent frozen milestone:
    OPT007 commit cd2f52b

Branch:
    opt008-avx2-ternary-compute


Scientific question
--------------------------------------------------------------------------------
With packed representation, table-dispatch semantics, G128 grouping, FP32
accumulation semantics, six-thread parallel scheduling, selected matrices,
and benchmark protocol held constant, can explicit AVX2 packed-code
processing reduce CPU inference latency while preserving validated behavior?


Controlled change
--------------------------------------------------------------------------------
OPT006 baseline:
    fixed lookup-table source dispatch.

OPT008 candidate:
    explicit AVX2 processing of eight packed bytes at once, corresponding
    to 32 ternary codes.

Compiler:
    -O2 -mavx2 -mno-fma

FMA was deliberately disabled so that the experiment did not introduce
fused multiply-add rounding changes.

The first OPT008 implementation vectorizes packed-code decoding. Activation
conversion and FP32 contribution accumulation remain scalar and preserve
the established logical accumulation order.

Therefore OPT008 does NOT claim that the entire ternary dot product is
vectorized.


Unchanged conditions
--------------------------------------------------------------------------------
Packed 2-bit ternary representation:
    -1 = 00
     0 = 01
    +1 = 10
    11 = invalid/reserved

Group size:
    128

Selected matrices:
    model.layers.3.self_attn.k_proj
    model.layers.4.linear_attn.out_proj
    model.layers.4.linear_attn.in_proj_qkv

Selected parameters:
    8,912,896

Persistent packed bytes:
    2,367,488

FP16-equivalent bytes:
    17,825,792

Persistent compression:
    7.529412x

Parallel scheduling:
    at::parallel_for

CPU benchmark threads:
    6 intra-op
    6 inter-op

Model:
    Qwen/Qwen3.5-0.8B


Gate 1 — operator equivalence
--------------------------------------------------------------------------------
Reference:
    frozen OPT006 operator

Candidate:
    OPT008 explicit AVX2 operator

Tested shapes:
    (1, 128, 1)
    (3, 128, 7)
    (5, 256, 13)
    (7, 512, 31)
    (22, 1024, 64)
    (22, 1024, 1024)

Total outputs:
    24,240

Exact mismatches:
    0

Maximum absolute error:
    0.0

Result:
    PASS — bit-exact operator equivalence.


Gate 2 — compiled machine-code verification
--------------------------------------------------------------------------------
Compiled hot function:
    packed_ternary_linear_cpu_parallel

Function size:
    4,688 bytes

Static instructions:
    1,125

Observed AVX2 evidence:
    vpmovzxbd      : 1
    vpand          : 4
    vpsrld         : 3
    YMM references : 13
    vzeroupper     : 8

Result:
    PASS — explicit AVX2 instructions are present inside the actual compiled
    ternary hot function.

This establishes that the experiment is not merely compiled with -mavx2;
the inspected hot function itself contains AVX2 instructions.


Gate 3 — full-model equivalence
--------------------------------------------------------------------------------
Reference:
    frozen OPT006 TableDispatchPackedTernaryLinear

Candidate:
    OPT008 AVX2PackedTernaryLinear

Replaced matrices:
    3

Replaced parameters:
    8,912,896

Compared logits:
    5,463,040

Exact mismatches:
    0

Maximum absolute difference:
    0.0

Mean absolute difference:
    0.0

Bit-exact logits:
    True

Top-1 matches:
    22 / 22

Result:
    PASS — full-model behavior is bit-exact with frozen OPT006 under the
    established validation input.


Gate 4 — controlled CPU benchmark
--------------------------------------------------------------------------------
Protocol:
    CPU
    6 PyTorch threads
    6 PyTorch inter-op threads
    22 input tokens
    2 warmup forwards per process
    5 timed forwards per process
    3 independent processes per condition
    15 timed forwards per condition

OPT006 baseline mean:
    2175.356 ms

OPT008 AVX2 mean:
    1994.648 ms

Mean latency change:
    -8.307%

OPT006 median:
    2117.194 ms

OPT008 median:
    1981.365 ms

OPT006 standard deviation:
    228.954 ms

OPT008 standard deviation:
    37.023 ms

OPT006 min / max:
    2006.847 / 2861.198 ms

OPT008 min / max:
    1963.687 / 2103.123 ms


Memory observations
--------------------------------------------------------------------------------
Mean ready RSS:
    OPT006 2337.546 MiB
    OPT008 2343.263 MiB
    difference +5.717 MiB

Mean after-warmup RSS:
    OPT006 2369.039 MiB
    OPT008 2372.928 MiB
    difference +3.889 MiB

Mean timed-max RSS:
    OPT006 2371.818 MiB
    OPT008 2379.875 MiB
    difference +8.057 MiB

GNU maximum RSS:
    OPT006 3,859,530.667 KB
    OPT008 3,865,245.333 KB
    difference +5,714.667 KB

No memory advantage is claimed.


Defensible conclusion
--------------------------------------------------------------------------------
In the controlled six-thread CPU benchmark, OPT008's explicit AVX2
packed-code decoding preserved bit-exact behavior relative to frozen OPT006
and reduced observed mean full-model forward latency from 2175.356 ms to
1994.648 ms (-8.307%) across 15 timed forwards per condition.

The compiled hot function was independently verified to contain AVX2
instructions.

Timed-max process RSS was 8.057 MiB higher and GNU maximum RSS was
5714.667 KB higher on average, so no memory advantage is claimed.

The result is specific to Qwen3.5-0.8B, the three selected packed matrices,
a 22-token input, six-thread CPU execution, the tested Intel Core
i7-8850H system, and this benchmark protocol.


Claim boundaries
--------------------------------------------------------------------------------
OPT008 does NOT establish:

    - a general CPU inference speedup,
    - a whole-model compression ratio,
    - a general peak-memory reduction,
    - a GPU or CUDA speedup,
    - AVX-512 performance,
    - FMA performance,
    - full ternary dot-product vectorization,
    - performance on CPUs other than the tested system,
    - performance on workloads outside the tested protocol.

OPT008 establishes a controlled observed latency improvement for this
specific explicit-AVX2 packed-code decoding implementation relative to
frozen OPT006 under the documented test conditions.
REPORT

echo "PASS: final report written."

# ------------------------------------------------------------
# 5. Syntax / import validation
# ------------------------------------------------------------

echo
echo "[5/6] Running static validation..."

python -m py_compile \
    packed_ternary/opt008_extension.py \
    packed_ternary/avx2_linear.py \
    packed_ternary/avx2_runtime_loader.py \
    scripts/opt008_benchmark_runtime.py \
    scripts/opt008_aggregate_results.py

echo "PASS: Python syntax."

grep -q \
"_mm256_cvtepu8_epi32" \
packed_ternary/csrc/packed_ternary_cpu_avx2.cpp

grep -q \
"_mm256_and_si256" \
packed_ternary/csrc/packed_ternary_cpu_avx2.cpp

grep -q \
"_mm256_srli_epi32" \
packed_ternary/csrc/packed_ternary_cpu_avx2.cpp

grep -q \
'"-mavx2"' \
packed_ternary/opt008_extension.py

grep -q \
'"-mno-fma"' \
packed_ternary/opt008_extension.py

echo "PASS: AVX2 source/compiler contract."

# ------------------------------------------------------------
# 6. Show exact freeze candidate
# ------------------------------------------------------------

echo
echo "[6/6] Freeze-candidate inspection..."
echo

echo "---------------- git status ----------------"
git status --short

echo
echo "---------------- tracked diff ---------------"
git diff --stat

echo
echo "---------------- candidate files ------------"

for path in \
    packed_ternary/csrc/packed_ternary_cpu_avx2.cpp \
    packed_ternary/opt008_extension.py \
    packed_ternary/avx2_linear.py \
    packed_ternary/avx2_runtime_loader.py \
    scripts/run_opt008.sh \
    scripts/run_opt008_stage2.sh \
    scripts/run_opt008_stage2b.sh \
    scripts/run_opt008_stage3.sh \
    scripts/run_opt008_stage4.sh \
    scripts/run_opt008_stage5.sh \
    scripts/opt008_benchmark_runtime.py \
    scripts/opt008_aggregate_results.py \
    results/OPT008_precheck.txt \
    results/OPT008_experiment_contract.txt \
    results/OPT008_operator_equivalence.txt \
    results/OPT008_avx2_machine_code.txt \
    results/OPT008_full_model_equivalence.txt \
    results/OPT008_avx2_cpu_benchmark.txt \
    results/OPT008_final_report.txt
do
    if [[ -f "$path" ]]; then
        echo "FOUND   $path"
    else
        echo "MISSING $path"
    fi
done

echo
echo "============================================================"
echo " OPT008 STAGE 5 COMPLETE"
echo "============================================================"
echo
echo "No commit performed."
echo "No push performed."
echo
echo "Review the git status and candidate-file list before freezing."
