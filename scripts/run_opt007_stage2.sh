#!/usr/bin/env bash
set -euo pipefail

DIR="results/opt007"
REPORT="results/OPT007_hot_loop_analysis.txt"

echo "=========================================="
echo " OPT007 Stage 2 — Hot Kernel Analysis"
echo "=========================================="

if [[ "$(git branch --show-current)" != "opt007-compiler-analysis" ]]; then
    echo "ERROR: Expected branch opt007-compiler-analysis"
    exit 1
fi

if [[ ! -f "$DIR/extension_paths.txt" ]]; then
    echo "ERROR: Stage 1 results not found."
    exit 1
fi

OPT005_SO="$(grep '^OPT005_SO=' "$DIR/extension_paths.txt" | tail -1 | cut -d= -f2-)"
OPT006_SO="$(grep '^OPT006_SO=' "$DIR/extension_paths.txt" | tail -1 | cut -d= -f2-)"

[[ -f "$OPT005_SO" ]] || { echo "ERROR: OPT005 .so missing"; exit 1; }
[[ -f "$OPT006_SO" ]] || { echo "ERROR: OPT006 .so missing"; exit 1; }

echo
echo "[1/4] Finding packed ternary symbols..."

nm -C --defined-only "$OPT005_SO" \
    | grep -Ei 'packed|ternary|forward|cpu' \
    > "$DIR/opt005_defined_candidates.txt" || true

nm -C --defined-only "$OPT006_SO" \
    | grep -Ei 'packed|ternary|forward|cpu' \
    > "$DIR/opt006_defined_candidates.txt" || true

echo
echo "OPT005 candidates:"
head -30 "$DIR/opt005_defined_candidates.txt"

echo
echo "OPT006 candidates:"
head -30 "$DIR/opt006_defined_candidates.txt"

echo
echo "[2/4] Locating source-related assembly..."

grep -n -Ei \
    'packed_ternary|branch_reduced|table_dispatch|parallel_for' \
    "$DIR/opt005_disassembly.txt" \
    > "$DIR/opt005_hot_locations.txt" || true

grep -n -Ei \
    'packed_ternary|branch_reduced|table_dispatch|parallel_for' \
    "$DIR/opt006_disassembly.txt" \
    > "$DIR/opt006_hot_locations.txt" || true

echo
echo "[3/4] Building mnemonic summaries..."

python - <<'PY' > results/opt007/hot_analysis_summary.txt
from pathlib import Path
from collections import Counter
import re

for name in ["opt005", "opt006"]:
    path = Path(f"results/opt007/{name}_disassembly.txt")
    text = path.read_text(errors="ignore")

    mnemonics = Counter()

    for line in text.splitlines():
        m = re.search(
            r'^\s*[0-9a-f]+:\s+(?:[0-9a-f]{2}\s+)+\s*([a-z][a-z0-9.]*)',
            line.lower()
        )
        if m:
            mnemonics[m.group(1)] += 1

    print("=" * 70)
    print(name.upper())
    print("=" * 70)

    groups = {
        "jumps": lambda x: x.startswith("j"),
        "setcc": lambda x: x.startswith("set"),
        "cmov": lambda x: x.startswith("cmov"),
        "scalar_fp": lambda x: x in {
            "addss", "subss", "mulss", "divss",
            "movss", "maxss", "minss"
        },
        "xmm_ops": lambda x: x.endswith("ss") or x.endswith("ps"),
        "ymm_avx": lambda x: x.startswith("v"),
    }

    for group, fn in groups.items():
        total = sum(v for k, v in mnemonics.items() if fn(k))
        print(f"{group:20s}: {total}")

    print()
    print("Most common instructions:")
    for mnemonic, count in mnemonics.most_common(25):
        print(f"{mnemonic:12s} {count}")
    print()
PY

echo
echo "[4/4] Writing report..."

{
    echo "OPT007 — HOT KERNEL / COMPILER ANALYSIS"
    echo "======================================="
    echo
    echo "Frozen comparison:"
    echo "OPT005 = comparison-mask source formulation"
    echo "OPT006 = fixed lookup-table source formulation"
    echo
    echo "No runtime kernel was modified during OPT007."
    echo
    echo "WHOLE-EXTENSION SUMMARY"
    echo "-----------------------"
    cat "$DIR/instruction_summary.txt"
    echo
    echo "MNEMONIC SUMMARY"
    echo "----------------"
    cat "$DIR/hot_analysis_summary.txt"
    echo
    echo "INTERPRETATION LIMIT"
    echo "--------------------"
    echo "Whole-extension instruction totals include binding/runtime/template code."
    echo "They cannot independently explain the measured latency difference."
    echo "Function-level hot-loop disassembly must support any causal explanation."
} > "$REPORT"

echo
echo "=========================================="
echo " Stage 2 complete"
echo "=========================================="
echo
echo "Report: $REPORT"
echo
echo "Relevant candidate symbols:"
echo
echo "--- OPT005 ---"
cat "$DIR/opt005_defined_candidates.txt" | head -20
echo
echo "--- OPT006 ---"
cat "$DIR/opt006_defined_candidates.txt" | head -20
echo
echo "NO COMMIT OR PUSH PERFORMED."
