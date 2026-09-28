#!/usr/bin/env bash
set -euo pipefail

echo "=========================================="
echo " OPT008 Stage 2B — AVX2 Machine Code Gate"
echo "=========================================="

EXPECTED_BRANCH="opt008-avx2-ternary-compute"
EXPECTED_PARENT="cd2f52b"

echo
echo "[1/4] Checking experiment state..."

if [[ "$(git branch --show-current)" != "$EXPECTED_BRANCH" ]]; then
    echo "ERROR: Expected branch $EXPECTED_BRANCH"
    exit 1
fi

if [[ "$(git rev-parse HEAD)" != "$(git rev-parse "$EXPECTED_PARENT")" ]]; then
    echo "ERROR: HEAD moved away from frozen OPT007 parent."
    git log -1 --oneline
    exit 1
fi

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "ERROR: .venv is not active."
    exit 1
fi

if [[ ! -s results/OPT008_operator_equivalence.txt ]]; then
    echo "ERROR: Gate-1 report is missing."
    exit 1
fi

grep -q \
"PASS: OPT008 operator is bit-exact with frozen OPT006." \
results/OPT008_operator_equivalence.txt || {
    echo "ERROR: Gate 1 has not passed."
    exit 1
}

echo "PASS: Gate 1 confirmed."

echo
echo "[2/4] Loading compiled OPT008 extension..."

mkdir -p results/opt008

SO_PATH="$(
PYTHONPATH="$PWD" python - <<'PY'
from packed_ternary.opt008_extension import (
    load_opt008_extension,
)

ext = load_opt008_extension(
    verbose=False,
)

print(ext.__file__)
PY
)"

if [[ ! -f "$SO_PATH" ]]; then
    echo "ERROR: compiled extension not found:"
    echo "  $SO_PATH"
    exit 1
fi

echo "Extension:"
echo "  $SO_PATH"

echo
echo "[3/4] Extracting hot function assembly..."

SYMBOL="$(
nm -C "$SO_PATH" |
grep ' T packed_ternary_linear_cpu_parallel(' |
head -1 || true
)"

if [[ -z "$SYMBOL" ]]; then
    echo "ERROR: hot function symbol not found."
    exit 1
fi

echo "Symbol:"
echo "  $SYMBOL"

START_HEX="$(awk '{print $1}' <<< "$SYMBOL")"

NEXT_SYMBOL="$(
nm -n -C "$SO_PATH" |
python -c '
import sys

start = sys.argv[1].lower()
found = False

for line in sys.stdin:
    parts = line.strip().split(None, 2)

    if len(parts) < 2:
        continue

    addr = parts[0].lower()
    symbol_type = parts[1]

    if addr == start:
        found = True
        continue

    if (
        found
        and symbol_type in {"T", "t"}
        and all(
            c in "0123456789abcdef"
            for c in addr
        )
    ):
        print(addr)
        break
' "$START_HEX"
)"

if [[ -z "$NEXT_SYMBOL" ]]; then
    echo "ERROR: Could not determine function end."
    exit 1
fi

START_DEC=$((16#$START_HEX))
END_DEC=$((16#$NEXT_SYMBOL))

if (( END_DEC <= START_DEC )); then
    echo "ERROR: Invalid function boundaries."
    exit 1
fi

FUNCTION_SIZE=$((END_DEC - START_DEC))

printf \
    "Hot function: 0x%s -> 0x%s (%d bytes)\n" \
    "$START_HEX" \
    "$NEXT_SYMBOL" \
    "$FUNCTION_SIZE"

objdump \
    -d \
    -M intel \
    --start-address="$START_DEC" \
    --stop-address="$END_DEC" \
    "$SO_PATH" \
    > results/opt008/opt008_hot_function.asm

echo
echo "[4/4] Verifying AVX2 inside hot function..."

ASM="results/opt008/opt008_hot_function.asm"

# Expected instructions from our explicit intrinsic path.
VPMOVZXBD="$(
    grep -Eic '\bvpmovzxbd\b' "$ASM" || true
)"

VPAND="$(
    grep -Eic '\bvpand\b' "$ASM" || true
)"

VPSRLD="$(
    grep -Eic '\bvpsrld\b' "$ASM" || true
)"

YMM="$(
    grep -Eic '\bymm[0-9]+\b' "$ASM" || true
)"

VZEROUPPER="$(
    grep -Eic '\bvzeroupper\b' "$ASM" || true
)"

TOTAL_INSTRUCTIONS="$(
    awk '
        /^[[:space:]]*[0-9a-f]+:/ {
            count++
        }
        END {
            print count+0
        }
    ' "$ASM"
)"

echo "Function size       : $FUNCTION_SIZE bytes"
echo "Instructions        : $TOTAL_INSTRUCTIONS"
echo "vpmovzxbd           : $VPMOVZXBD"
echo "vpand               : $VPAND"
echo "vpsrld              : $VPSRLD"
echo "YMM references      : $YMM"
echo "vzeroupper          : $VZEROUPPER"

if [[ "$VPMOVZXBD" -eq 0 ]]; then
    echo "ERROR: vpmovzxbd absent from hot function."
    exit 1
fi

if [[ "$VPAND" -eq 0 ]]; then
    echo "ERROR: vpand absent from hot function."
    exit 1
fi

if [[ "$VPSRLD" -eq 0 ]]; then
    echo "ERROR: vpsrld absent from hot function."
    exit 1
fi

if [[ "$YMM" -eq 0 ]]; then
    echo "ERROR: no YMM register use in hot function."
    exit 1
fi

{
    echo "OPT008 Stage 2B — AVX2 Machine-Code Verification"
    echo "================================================"
    echo
    echo "Extension:"
    echo "  $SO_PATH"
    echo
    echo "Symbol:"
    echo "  $SYMBOL"
    echo
    echo "Function boundaries:"
    printf \
        "  0x%s -> 0x%s\n" \
        "$START_HEX" \
        "$NEXT_SYMBOL"
    echo
    echo "Function size:"
    echo "  $FUNCTION_SIZE bytes"
    echo
    echo "Instruction count:"
    echo "  $TOTAL_INSTRUCTIONS"
    echo
    echo "AVX2 evidence inside packed_ternary_linear_cpu_parallel:"
    echo "  vpmovzxbd      : $VPMOVZXBD"
    echo "  vpand          : $VPAND"
    echo "  vpsrld         : $VPSRLD"
    echo "  YMM references : $YMM"
    echo "  vzeroupper     : $VZEROUPPER"
    echo
    echo "Interpretation:"
    echo "  Explicit AVX2 instructions are present inside"
    echo "  the compiled ternary hot function."
    echo
    echo "Correctness:"
    echo "  Gate 1 previously established bit-exact"
    echo "  operator equivalence with frozen OPT006."
    echo
    echo "Claim boundary:"
    echo "  This proves execution code contains AVX2."
    echo "  It does not establish a performance advantage."
} > results/OPT008_avx2_machine_code.txt

echo
echo "Selected AVX2 instructions:"
grep -Ei \
'vpmovzxbd|vpand|vpsrld' \
"$ASM" | head -30 || true

echo
echo "=========================================="
echo " OPT008 STAGE 2B COMPLETE"
echo "=========================================="
echo
echo "PASS:"
echo "  AVX2 instructions verified inside"
echo "  packed_ternary_linear_cpu_parallel."
echo
echo "Reports:"
echo "  results/OPT008_avx2_machine_code.txt"
echo "  results/opt008/opt008_hot_function.asm"
echo
echo "No benchmark, commit, or push performed."
