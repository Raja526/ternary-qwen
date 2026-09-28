#!/usr/bin/env bash
set -euo pipefail

BASE_COMMIT="b3371f9"
BRANCH="opt007-compiler-analysis"
RESULT="results/OPT007_compiler_analysis.txt"

echo "=========================================="
echo " OPT007 — Compiler / Assembly Analysis"
echo "=========================================="

# --------------------------------------------------
# 1. Environment checks
# --------------------------------------------------
echo
echo "[1/7] Checking environment..."

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "ERROR: Python virtual environment is not active."
    echo "Run: source .venv/bin/activate"
    exit 1
fi

echo "VENV: $VIRTUAL_ENV"
echo "Python: $(which python)"
python --version

# --------------------------------------------------
# 2. Git safety checks
# --------------------------------------------------
echo
echo "[2/7] Checking Git state..."

DIRTY_FILES="$(git status --porcelain | grep -v '?? scripts/run_opt007.sh' || true)"

if [[ -n "$DIRTY_FILES" ]]; then
    echo "ERROR: Working tree contains unexpected changes."
    echo "$DIRTY_FILES"
    exit 1
fi

echo "Working tree clean except for OPT007 runner."

CURRENT_COMMIT="$(git rev-parse HEAD)"
EXPECTED_COMMIT="$(git rev-parse "$BASE_COMMIT")"

if [[ "$CURRENT_COMMIT" != "$EXPECTED_COMMIT" ]]; then
    echo "ERROR: Expected frozen OPT006 base:"
    echo "  $EXPECTED_COMMIT"
    echo "Current:"
    echo "  $CURRENT_COMMIT"
    exit 1
fi

if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
    git checkout "$BRANCH"
else
    git checkout -b "$BRANCH"
fi

echo "Branch: $(git branch --show-current)"
echo "Base:   $(git rev-parse --short HEAD)"

# --------------------------------------------------
# 3. Collect system/compiler information
# --------------------------------------------------
echo
echo "[3/7] Collecting CPU/compiler information..."

mkdir -p results/opt007

{
    echo "OPT007 — COMPILER / ASSEMBLY ANALYSIS"
    echo "====================================="
    echo
    echo "Base commit: $BASE_COMMIT"
    echo "Branch: $BRANCH"
    echo
    echo "SYSTEM"
    echo "------"
    uname -a
    echo
    lscpu | grep -E \
        "Architecture|Model name|CPU\\(s\\)|Thread|Core|Socket|Flags" || true
    echo
    echo "COMPILER"
    echo "--------"
    g++ --version | head -1
    objdump --version | head -1
    echo
    echo "PYTORCH"
    echo "-------"
    python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CXX11 ABI:", torch.compiled_with_cxx11_abi())
print("Threads:", torch.get_num_threads())
print("Interop threads:", torch.get_num_interop_threads())
PY
} > "$RESULT"

# --------------------------------------------------
# 4. Build/load frozen OPT005 and OPT006 extensions
# --------------------------------------------------
echo
echo "[4/7] Loading frozen OPT005/OPT006 extensions..."

python - <<'PY' | tee results/opt007/extension_paths.txt
from packed_ternary.opt005_extension import load_opt005_extension
from packed_ternary.opt006_extension import load_opt006_extension

a = load_opt005_extension()
b = load_opt006_extension()

print("OPT005_SO=" + a.__file__)
print("OPT006_SO=" + b.__file__)
PY

OPT005_SO="$(grep '^OPT005_SO=' results/opt007/extension_paths.txt | tail -1 | cut -d= -f2-)"
OPT006_SO="$(grep '^OPT006_SO=' results/opt007/extension_paths.txt | tail -1 | cut -d= -f2-)"

if [[ ! -f "$OPT005_SO" ]]; then
    echo "ERROR: OPT005 extension not found: $OPT005_SO"
    exit 1
fi

if [[ ! -f "$OPT006_SO" ]]; then
    echo "ERROR: OPT006 extension not found: $OPT006_SO"
    exit 1
fi

echo "OPT005: $OPT005_SO"
echo "OPT006: $OPT006_SO"

# --------------------------------------------------
# 5. Symbols and disassembly
# --------------------------------------------------
echo
echo "[5/7] Extracting symbols and assembly..."

nm -C "$OPT005_SO" > results/opt007/opt005_symbols.txt
nm -C "$OPT006_SO" > results/opt007/opt006_symbols.txt

objdump -d -C "$OPT005_SO" > results/opt007/opt005_disassembly.txt
objdump -d -C "$OPT006_SO" > results/opt007/opt006_disassembly.txt

grep -Ei \
    "packed|ternary|cpu" \
    results/opt007/opt005_symbols.txt \
    > results/opt007/opt005_relevant_symbols.txt || true

grep -Ei \
    "packed|ternary|cpu" \
    results/opt007/opt006_symbols.txt \
    > results/opt007/opt006_relevant_symbols.txt || true

# --------------------------------------------------
# 6. Static instruction summary
# --------------------------------------------------
echo
echo "[6/7] Creating static instruction summary..."

python - <<'PY' | tee results/opt007/instruction_summary.txt
from pathlib import Path
import re

files = {
    "OPT005_BRANCH_REDUCED":
        Path("results/opt007/opt005_disassembly.txt"),
    "OPT006_TABLE_DISPATCH":
        Path("results/opt007/opt006_disassembly.txt"),
}

patterns = {
    "conditional_jumps": r"\b(j[a-z]{1,3})\b",
    "setcc": r"\bset[a-z]+\b",
    "cmov": r"\bcmov[a-z]+\b",
    "mulss": r"\bmulss\b",
    "addss": r"\baddss\b",
    "movss": r"\bmovss\b",
    "mulsd": r"\bmulsd\b",
    "addsd": r"\baddsd\b",
    "xmm_mentions": r"\bxmm\d+\b",
    "ymm_mentions": r"\bymm\d+\b",
    "zmm_mentions": r"\bzmm\d+\b",
}

for name, path in files.items():
    text = path.read_text(errors="ignore").lower()

    print("=" * 70)
    print(name)
    print("=" * 70)

    for label, pattern in patterns.items():
        count = len(re.findall(pattern, text))
        print(f"{label:24s}: {count}")

    print()
PY

{
    echo
    echo "EXTENSIONS"
    echo "----------"
    echo "OPT005: $OPT005_SO"
    echo "OPT006: $OPT006_SO"
    echo

    echo "STATIC INSTRUCTION SUMMARY"
    echo "--------------------------"
    cat results/opt007/instruction_summary.txt

    echo
    echo "IMPORTANT INTERPRETATION RULE"
    echo "-----------------------------"
    echo "These counts cover extension disassembly and are descriptive."
    echo "They are not by themselves proof of hot-loop behavior."
    echo "Hot-function inspection is required before causal claims."
    echo
    echo "OPT007 does not modify either frozen runtime kernel."
} >> "$RESULT"

# --------------------------------------------------
# 7. Finish
# --------------------------------------------------
echo
echo "[7/7] Final checks..."

git diff --check

echo
echo "=========================================="
echo " OPT007 STATIC ANALYSIS COMPLETE"
echo "=========================================="
echo
echo "Report:"
echo "  $RESULT"
echo
echo "Artifacts:"
echo "  results/opt007/extension_paths.txt"
echo "  results/opt007/opt005_relevant_symbols.txt"
echo "  results/opt007/opt006_relevant_symbols.txt"
echo "  results/opt007/instruction_summary.txt"
echo
echo "Git status:"
git status --short
echo
echo "NO COMMIT OR PUSH WAS PERFORMED."
echo "Review the OPT007 results before freezing."
