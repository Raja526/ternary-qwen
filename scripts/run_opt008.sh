#!/usr/bin/env bash
set -euo pipefail

echo "=========================================="
echo " OPT008 — Explicit AVX2 Ternary Compute"
echo " Stage 1: Controlled Setup"
echo "=========================================="

BASE_COMMIT="cd2f52b"
BRANCH="opt008-avx2-ternary-compute"

# ---------------------------------------------------------
# 1. Environment
# ---------------------------------------------------------

echo
echo "[1/7] Checking environment..."

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "ERROR: Python virtual environment is not active."
    echo "Run:"
    echo "  source .venv/bin/activate"
    exit 1
fi

echo "Python : $(which python)"
python --version

# ---------------------------------------------------------
# 2. Repository safety
# ---------------------------------------------------------

echo
echo "[2/7] Checking frozen parent..."

CURRENT_HEAD="$(git rev-parse HEAD)"
EXPECTED_HEAD="$(git rev-parse "$BASE_COMMIT")"

if [[ "$CURRENT_HEAD" != "$EXPECTED_HEAD" ]]; then
    echo "ERROR: OPT008 must start from frozen OPT007."
    echo
    echo "Expected:"
    git show -s --oneline "$BASE_COMMIT"
    echo
    echo "Current:"
    git show -s --oneline HEAD
    exit 1
fi

echo "PASS:"
git show -s --oneline HEAD

# Allow the known local raw OPT007 directory and this runner only.
UNEXPECTED="$(
    git status --porcelain |
    grep -vE '^\?\? results/opt007/$' |
    grep -vE '^\?\? scripts/run_opt008\.sh$' || true
)"

if [[ -n "$UNEXPECTED" ]]; then
    echo
    echo "ERROR: Unexpected working-tree changes:"
    echo "$UNEXPECTED"
    echo
    echo "OPT008 will not modify the repository."
    exit 1
fi

# ---------------------------------------------------------
# 3. Create / verify branch
# ---------------------------------------------------------

echo
echo "[3/7] Preparing OPT008 branch..."

CURRENT_BRANCH="$(git branch --show-current)"

if [[ "$CURRENT_BRANCH" == "$BRANCH" ]]; then
    echo "Already on $BRANCH"

elif git show-ref --verify --quiet "refs/heads/$BRANCH"; then
    echo "ERROR: Branch already exists but is not checked out:"
    echo "  $BRANCH"
    echo
    echo "Inspect it before continuing."
    exit 1

else
    git switch -c "$BRANCH"
fi

echo "Branch: $(git branch --show-current)"
echo "Parent: $(git rev-parse --short HEAD)"

# ---------------------------------------------------------
# 4. Verify CPU ISA
# ---------------------------------------------------------

echo
echo "[4/7] Verifying AVX2 hardware support..."

FLAGS="$(lscpu | grep '^Flags:' || true)"

for flag in avx avx2; do
    if ! grep -qw "$flag" <<< "$FLAGS"; then
        echo "ERROR: Required CPU feature missing: $flag"
        exit 1
    fi
done

echo "PASS: AVX + AVX2 supported."

if grep -qw fma <<< "$FLAGS"; then
    echo "INFO: FMA supported, but OPT008 Stage 1 will NOT enable FMA."
fi

if grep -qw f16c <<< "$FLAGS"; then
    echo "INFO: F16C supported."
fi

# ---------------------------------------------------------
# 5. Verify compiler accepts AVX2
# ---------------------------------------------------------

echo
echo "[5/7] Verifying compiler AVX2 support..."

TMP_CPP="$(mktemp --suffix=.cpp)"
TMP_BIN="$(mktemp)"

cat > "$TMP_CPP" <<'CPP'
#include <immintrin.h>
#include <cstdio>

int main() {
    __m256 a = _mm256_set1_ps(2.0f);
    __m256 b = _mm256_add_ps(a, a);

    alignas(32) float out[8];
    _mm256_store_ps(out, b);

    std::printf("%.1f\n", out[0]);
    return 0;
}
CPP

g++ \
    -O2 \
    -mavx2 \
    -mno-fma \
    "$TMP_CPP" \
    -o "$TMP_BIN"

RESULT="$("$TMP_BIN")"

rm -f "$TMP_CPP" "$TMP_BIN"

if [[ "$RESULT" != "4.0" ]]; then
    echo "ERROR: AVX2 compiler/runtime sanity test failed."
    echo "Result: $RESULT"
    exit 1
fi

echo "PASS: GCC AVX2 intrinsic test."

# ---------------------------------------------------------
# 6. Capture frozen OPT006 kernel context
# ---------------------------------------------------------

echo
echo "[6/7] Capturing OPT006 kernel context..."

mkdir -p results/opt008

SOURCE="packed_ternary/csrc/packed_ternary_cpu_table_dispatch.cpp"
EXTENSION="packed_ternary/opt006_extension.py"

if [[ ! -f "$SOURCE" ]]; then
    echo "ERROR: Missing frozen OPT006 source:"
    echo "  $SOURCE"
    exit 1
fi

if [[ ! -f "$EXTENSION" ]]; then
    echo "ERROR: Missing frozen OPT006 extension:"
    echo "  $EXTENSION"
    exit 1
fi

{
    echo "OPT008 Stage 1 — CPU / Compiler Precheck"
    echo "========================================"
    echo
    echo "Frozen parent:"
    git show -s --format='%H %s' HEAD
    echo
    echo "CPU:"
    lscpu | grep -E \
        'Model name|Architecture|CPU\(s\)|Thread|Core|Socket'
    echo
    echo "Relevant ISA:"
    echo "$FLAGS" | tr ' ' '\n' | \
        grep -E \
        '^(avx|avx2|f16c|fma|sse|sse2|ssse3|sse4_1|sse4_2)$' | \
        sort -u
    echo
    echo "Compiler:"
    g++ --version | head -1
    echo
    echo "Frozen OPT006 flags:"
    grep -n -E \
        'extra_cflags|O2|march|avx|fma' \
        "$EXTENSION" || true
    echo
    echo "OPT008 proposed flags:"
    echo "  -O2"
    echo "  -mavx2"
    echo "  -mno-fma"
    echo
    echo "Scientific constraint:"
    echo "  No -march=native"
    echo "  No FMA"
    echo "  No cache blocking changes"
    echo "  No thread scheduling changes"
    echo "  No packed-format changes"
    echo "  No group-size changes"
} > results/OPT008_precheck.txt

# Save the exact relevant source section for the next stage.
sed -n '120,310p' "$SOURCE" \
    > results/opt008/opt006_kernel_context.txt

echo "PASS: kernel context captured."

# ---------------------------------------------------------
# 7. Record experiment contract
# ---------------------------------------------------------

echo
echo "[7/7] Writing OPT008 experiment contract..."

cat > results/OPT008_experiment_contract.txt <<'EOF'
OPT008 — Explicit AVX2 Ternary Compute
======================================

Frozen baseline
---------------
OPT007 commit:
  cd2f52b

Runtime baseline:
  OPT006 table-dispatch packed ternary kernel

Controlled scientific question
------------------------------
With the packed representation, table-dispatch semantics,
group size 128, FP32 accumulation semantics, six-thread
parallel scheduling, selected matrices, and benchmark
protocol held constant, can explicit AVX2 processing reduce
CPU inference latency while preserving validated behavior?

Permitted OPT008 change
-----------------------
Introduce an explicit AVX2 compute path.

Compiler target
---------------
-O2
-mavx2
-mno-fma

Explicitly prohibited in this experiment
----------------------------------------
-march=native
FMA
AVX-512
cache blocking changes
thread-count changes
parallel scheduling changes
packed-format changes
group-size changes
quantizer changes
selected-matrix changes
benchmark protocol changes

Correctness gates
-----------------
Gate 1:
  AVX2 operator vs frozen OPT006 operator.

Gate 2:
  Full-model AVX2 runtime vs frozen OPT006 runtime.

Gate 3:
  Top-1 behavior must remain preserved under the established
  validation protocol.

Performance gate
----------------
Only after correctness gates pass:

  OPT006 table-dispatch baseline
      vs
  OPT008 explicit AVX2 candidate

No performance claim is permitted before correctness passes.

Claim discipline
----------------
A successful result is specific to:
  Qwen3.5-0.8B
  selected packed matrices
  Intel Core i7-8850H
  tested input shape
  six-thread CPU execution
  frozen benchmark protocol

No general CPU speedup claim.
No GPU claim.
No memory advantage claim unless separately measured.
EOF

echo
echo "=========================================="
echo " OPT008 STAGE 1 COMPLETE"
echo "=========================================="
echo
echo "Branch:"
echo "  $(git branch --show-current)"
echo
echo "Parent:"
echo "  $(git rev-parse HEAD)"
echo
echo "Artifacts:"
echo "  results/OPT008_precheck.txt"
echo "  results/OPT008_experiment_contract.txt"
echo "  results/opt008/opt006_kernel_context.txt"
echo
echo "IMPORTANT:"
echo "  No kernel has been modified."
echo "  No commit has been created."
echo "  No push has occurred."
echo
echo "Next:"
echo "  Build the explicit AVX2 kernel from the captured"
echo "  frozen OPT006 implementation."
