#!/usr/bin/env bash
set -euo pipefail

echo "============================================================"
echo " OPT009 STAGE 3 — FULL-MODEL CORRECTNESS GATE"
echo "============================================================"
echo
echo "Reference : frozen OPT008 explicit AVX2"
echo "Candidate : OPT009 block-8 output-row blocking"
echo

export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6

# ------------------------------------------------------------
# 1. Verify required reproducible artifacts.
# ------------------------------------------------------------

echo "[1/4] Verifying required artifacts..."

required_files=(
    "packed_ternary/csrc/packed_ternary_cpu_avx2_blocked.cpp"
    "packed_ternary/opt009_extension.py"
    "packed_ternary/blocked_avx2_linear.py"
    "packed_ternary/blocked_avx2_runtime_loader.py"
    "results/opt009_validate_full_model.py"
    "results/ENG003_multimatrix_checkpoint.pt"
)

for file in "${required_files[@]}"; do
    if [[ ! -f "$file" ]]; then
        echo "ERROR: required file missing: $file"
        exit 1
    fi
done

echo "PASS: required artifacts present."

# ------------------------------------------------------------
# 2. Verify final loader API.
# ------------------------------------------------------------

echo
echo "[2/4] Verifying final runtime-loader API..."

grep -q \
    '^def load_blocked_avx2_runtime_modules(' \
    packed_ternary/blocked_avx2_runtime_loader.py

grep -q \
    'BlockedAVX2PackedTernaryLinear' \
    packed_ternary/blocked_avx2_runtime_loader.py

echo "PASS: final blocked runtime-loader API verified."

# ------------------------------------------------------------
# 3. Syntax gate.
# ------------------------------------------------------------

echo
echo "[3/4] Running syntax gate..."

python -m py_compile \
    packed_ternary/opt009_extension.py \
    packed_ternary/blocked_avx2_linear.py \
    packed_ternary/blocked_avx2_runtime_loader.py \
    results/opt009_validate_full_model.py

echo "PASS: syntax gate."

# ------------------------------------------------------------
# 4. Full-model correctness gate.
# ------------------------------------------------------------

echo
echo "[4/4] Running full-model equivalence gate..."

python results/opt009_validate_full_model.py \
    | tee results/OPT009_full_model_equivalence.txt

grep -q \
    'Exact mismatches  : 0' \
    results/OPT009_full_model_equivalence.txt

grep -q \
    'Bit-exact logits  : True' \
    results/OPT009_full_model_equivalence.txt

grep -q \
    'Top-1 matches     : 22/22' \
    results/OPT009_full_model_equivalence.txt

grep -q \
    'PASS: OPT009 full-model behavior is bit-exact with frozen OPT008.' \
    results/OPT009_full_model_equivalence.txt

echo
echo "============================================================"
echo " OPT009 STAGE 3 COMPLETE — PASS"
echo "============================================================"
