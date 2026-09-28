#!/usr/bin/env bash
set -euo pipefail

echo "============================================================"
echo " OPT008 — Freeze Milestone"
echo "============================================================"

EXPECTED_BRANCH="opt008-avx2-ternary-compute"
EXPECTED_PARENT="cd2f52b"

echo
echo "[1/6] Verifying freeze state..."

if [[ "$(git branch --show-current)" != "$EXPECTED_BRANCH" ]]; then
    echo "ERROR: wrong branch."
    exit 1
fi

if [[ "$(git rev-parse HEAD)" != "$(git rev-parse "$EXPECTED_PARENT")" ]]; then
    echo "ERROR: HEAD is not frozen OPT007 parent."
    git log -3 --oneline
    exit 1
fi

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "ERROR: .venv is not active."
    exit 1
fi

echo "PASS: branch and parent."

echo
echo "[2/6] Verifying final gates..."

grep -q -- \
"PASS: OPT008 operator is bit-exact with frozen OPT006." \
results/OPT008_operator_equivalence.txt

grep -q -- \
"Explicit AVX2 instructions are present" \
results/OPT008_avx2_machine_code.txt

grep -q -- \
"OPT008 full-model runtime is bit-exact with frozen OPT006." \
results/OPT008_full_model_equivalence.txt

grep -q -- \
"OPT008 mean latency change: -8.307%" \
results/OPT008_avx2_cpu_benchmark.txt

grep -q -- \
"2175.356 ms" \
results/OPT008_final_report.txt

grep -q -- \
"1994.648 ms" \
results/OPT008_final_report.txt

grep -q -- \
"-8.307%" \
results/OPT008_final_report.txt

echo "PASS: all final gates/reports."

echo
echo "[3/6] Static validation..."

python -m py_compile \
    packed_ternary/opt008_extension.py \
    packed_ternary/avx2_linear.py \
    packed_ternary/avx2_runtime_loader.py \
    scripts/opt008_benchmark_runtime.py \
    scripts/opt008_aggregate_results.py

grep -q -- \
"_mm256_cvtepu8_epi32" \
packed_ternary/csrc/packed_ternary_cpu_avx2.cpp

grep -q -- \
'"-mavx2"' \
packed_ternary/opt008_extension.py

grep -q -- \
'"-mno-fma"' \
packed_ternary/opt008_extension.py

echo "PASS: implementation contract."

echo
echo "[4/6] Staging exact OPT008 artifact set..."

git add -- \
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
    scripts/freeze_opt008.sh \
    scripts/opt008_benchmark_runtime.py \
    scripts/opt008_aggregate_results.py \
    results/OPT008_precheck.txt \
    results/OPT008_experiment_contract.txt \
    results/OPT008_operator_equivalence.txt \
    results/OPT008_avx2_machine_code.txt \
    results/OPT008_full_model_equivalence.txt \
    results/OPT008_avx2_cpu_benchmark.txt \
    results/OPT008_final_report.txt

echo
echo "Staged files:"
git diff --cached --name-status

echo
echo "[5/6] Guarding against accidental raw/generated files..."

FORBIDDEN="$(
    git diff --cached --name-only |
    grep -E \
    '^(results/opt007/|results/opt008/|results/opt008_raw/|results/opt008_validate_)' \
    || true
)"

if [[ -n "$FORBIDDEN" ]]; then
    echo "ERROR: generated/raw files accidentally staged:"
    echo "$FORBIDDEN"
    exit 1
fi

STAGED_COUNT="$(
    git diff --cached --name-only |
    wc -l
)"

echo "Staged file count: $STAGED_COUNT"

if [[ "$STAGED_COUNT" -ne 20 ]]; then
    echo "ERROR: expected exactly 20 staged files."
    echo
    git diff --cached --name-only
    exit 1
fi

echo "PASS: exact freeze set."

echo
echo "[6/6] Creating frozen OPT008 commit..."

git commit \
    -m "OPT008: add explicit AVX2 ternary CPU compute"

echo
echo "============================================================"
echo " OPT008 FROZEN LOCALLY"
echo "============================================================"

echo
echo "Commit:"
git log -1 --oneline

echo
echo "Full SHA:"
git rev-parse HEAD

echo
echo "Parent:"
git rev-parse HEAD^

echo
echo "Committed files:"
git show \
    --name-only \
    --format='' \
    HEAD

echo
echo "Remaining untracked files:"
git status --short

echo
echo "IMPORTANT:"
echo "  Commit created locally."
echo "  Push has NOT been performed."
