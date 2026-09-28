#!/usr/bin/env bash
set -euo pipefail

DIR="results/opt007"
REPORT="results/OPT007_function_level_analysis.txt"
START="0xdd10"

echo "=========================================="
echo " OPT007 Stage 3 — Function-Level Assembly"
echo "=========================================="

[[ "$(git branch --show-current)" == "opt007-compiler-analysis" ]] || {
    echo "ERROR: Wrong branch."
    exit 1
}

OPT005_SO="$(grep '^OPT005_SO=' "$DIR/extension_paths.txt" | tail -1 | cut -d= -f2-)"
OPT006_SO="$(grep '^OPT006_SO=' "$DIR/extension_paths.txt" | tail -1 | cut -d= -f2-)"

[[ -f "$OPT005_SO" ]] || { echo "ERROR: OPT005 binary missing"; exit 1; }
[[ -f "$OPT006_SO" ]] || { echo "ERROR: OPT006 binary missing"; exit 1; }

echo
echo "[1/5] Determining hot-function boundaries..."

python - "$OPT005_SO" "$OPT006_SO" <<'PY' > results/opt007/function_boundaries.txt
import subprocess
import sys

for label, so in zip(("OPT005", "OPT006"), sys.argv[1:]):
    output = subprocess.check_output(
        ["nm", "-n", "-C", "--defined-only", so],
        text=True,
        errors="ignore",
    )

    symbols = []
    for line in output.splitlines():
        parts = line.split(maxsplit=2)
        if len(parts) < 3:
            continue
        try:
            address = int(parts[0], 16)
        except ValueError:
            continue
        symbols.append((address, parts[2]))

    target = None

    for i, (address, name) in enumerate(symbols):
        if (
            name.startswith("packed_ternary_linear_cpu_parallel(")
            and "[clone .cold]" not in name
        ):
            next_addr = None
            for j in range(i + 1, len(symbols)):
                if symbols[j][0] > address:
                    next_addr = symbols[j][0]
                    break
            target = (address, next_addr, name)
            break

    if target is None:
        raise SystemExit(f"ERROR: hot function not found for {label}")

    start, end, name = target
    if end is None:
        raise SystemExit(f"ERROR: unable to determine end for {label}")

    print(f"{label}_START=0x{start:x}")
    print(f"{label}_END=0x{end:x}")
    print(f"{label}_SIZE={end-start}")
    print(f"{label}_NAME={name}")
PY

cat "$DIR/function_boundaries.txt"

OPT005_START="$(grep '^OPT005_START=' "$DIR/function_boundaries.txt" | cut -d= -f2)"
OPT005_END="$(grep '^OPT005_END=' "$DIR/function_boundaries.txt" | cut -d= -f2)"
OPT006_START="$(grep '^OPT006_START=' "$DIR/function_boundaries.txt" | cut -d= -f2)"
OPT006_END="$(grep '^OPT006_END=' "$DIR/function_boundaries.txt" | cut -d= -f2)"

echo
echo "[2/5] Disassembling only the target functions..."

objdump -d -C -Mintel \
    --start-address="$OPT005_START" \
    --stop-address="$OPT005_END" \
    "$OPT005_SO" > "$DIR/opt005_hot_function.asm"

objdump -d -C -Mintel \
    --start-address="$OPT006_START" \
    --stop-address="$OPT006_END" \
    "$OPT006_SO" > "$DIR/opt006_hot_function.asm"

echo
echo "[3/5] Analysing hot-function instructions..."

python - <<'PY' | tee results/opt007/function_instruction_comparison.txt
from pathlib import Path
from collections import Counter
import re

paths = {
    "OPT005_COMPARISON_MASK":
        Path("results/opt007/opt005_hot_function.asm"),
    "OPT006_TABLE_DISPATCH":
        Path("results/opt007/opt006_hot_function.asm"),
}

def parse(path):
    instructions = []

    for line in path.read_text(errors="ignore").splitlines():
        m = re.match(
            r'^\s*[0-9a-f]+:\s+(?:[0-9a-f]{2}\s+)+\s*'
            r'([a-z][a-z0-9.]*)\s*(.*)$',
            line.lower()
        )
        if m:
            instructions.append((m.group(1), m.group(2)))

    return instructions

for label, path in paths.items():
    inst = parse(path)
    c = Counter(m for m, _ in inst)

    conditional = sum(
        n for m, n in c.items()
        if m.startswith("j") and m not in {"jmp", "jmpq"}
    )

    setcc = sum(n for m, n in c.items() if m.startswith("set"))
    cmov = sum(n for m, n in c.items() if m.startswith("cmov"))

    scalar_fp = sum(
        c[x] for x in [
            "addss", "subss", "mulss", "divss",
            "movss", "maxss", "minss"
        ]
    )

    vector_fp = sum(
        n for m, n in c.items()
        if (
            m.startswith("v")
            or m.endswith("ps")
            or m.endswith("pd")
        )
    )

    print("=" * 72)
    print(label)
    print("=" * 72)
    print(f"total instructions       : {len(inst)}")
    print(f"conditional jumps        : {conditional}")
    print(f"setcc                    : {setcc}")
    print(f"cmov                     : {cmov}")
    print(f"mulss                    : {c['mulss']}")
    print(f"addss                    : {c['addss']}")
    print(f"movss                    : {c['movss']}")
    print(f"scalar FP selected       : {scalar_fp}")
    print(f"vector-like FP mnemonics : {vector_fp}")
    print()

    print("Most common instructions:")
    for mnemonic, count in c.most_common(30):
        print(f"  {mnemonic:12s} {count}")

    print()

    print("Instructions involving SET/CMOV/FP/LUT-style indexed memory:")
    for mnemonic, operands in inst:
        interesting = (
            mnemonic.startswith("set")
            or mnemonic.startswith("cmov")
            or mnemonic in {
                "mulss", "addss", "subss",
                "movss", "lea"
            }
        )

        if interesting:
            print(f"  {mnemonic:10s} {operands}")

    print()
PY

echo
echo "[4/5] Producing normalized assembly diff..."

sed -E \
    's/^[[:space:]]*[0-9a-f]+:[[:space:]]+([0-9a-f]{2}[[:space:]]+)+//' \
    "$DIR/opt005_hot_function.asm" \
    > "$DIR/opt005_hot_normalized.asm"

sed -E \
    's/^[[:space:]]*[0-9a-f]+:[[:space:]]+([0-9a-f]{2}[[:space:]]+)+//' \
    "$DIR/opt006_hot_function.asm" \
    > "$DIR/opt006_hot_normalized.asm"

diff -u \
    "$DIR/opt005_hot_normalized.asm" \
    "$DIR/opt006_hot_normalized.asm" \
    > "$DIR/opt005_vs_opt006_hot.diff" || true

echo
echo "[5/5] Writing report..."

{
    echo "OPT007 — FUNCTION-LEVEL COMPILER ANALYSIS"
    echo "========================================="
    echo
    echo "Purpose:"
    echo "Compare compiler-generated machine code for the frozen OPT005"
    echo "comparison-mask formulation and frozen OPT006 lookup-table formulation."
    echo
    echo "No kernel source or compiler flag was changed in OPT007."
    echo
    echo "FUNCTION BOUNDARIES"
    echo "-------------------"
    cat "$DIR/function_boundaries.txt"
    echo
    echo "INSTRUCTION COMPARISON"
    echo "----------------------"
    cat "$DIR/function_instruction_comparison.txt"
    echo
    echo "CAUTION"
    echo "-------"
    echo "Static assembly differences establish compiler lowering differences."
    echo "They do not by themselves establish which difference caused the"
    echo "previously measured runtime latency change."
    echo "Hardware-counter evidence would be required for stronger causal analysis."
} > "$REPORT"

echo
echo "=========================================="
echo " OPT007 Stage 3 complete"
echo "=========================================="
echo
echo "Report:"
echo "  $REPORT"
echo
echo "Assembly:"
echo "  $DIR/opt005_hot_function.asm"
echo "  $DIR/opt006_hot_function.asm"
echo
echo "Diff:"
echo "  $DIR/opt005_vs_opt006_hot.diff"
echo
echo "Key comparison:"
cat "$DIR/function_instruction_comparison.txt" \
    | grep -E \
    '^(OPT|total instructions|conditional jumps|setcc|cmov|mulss|addss|movss|scalar FP|vector-like)'
echo
echo "NO COMMIT OR PUSH PERFORMED."
