#!/usr/bin/env bash
set -euo pipefail

echo "=========================================="
echo " OPT007 — Freeze Experiment"
echo "=========================================="

EXPECTED_BRANCH="opt007-compiler-analysis"
EXPECTED_PARENT="b3371f9"

# ---------------------------------------------------------
# 1. Repository / branch checks
# ---------------------------------------------------------

echo
echo "[1/7] Checking repository state..."

CURRENT_BRANCH="$(git branch --show-current)"

if [[ "$CURRENT_BRANCH" != "$EXPECTED_BRANCH" ]]; then
    echo "ERROR: Expected branch:"
    echo "  $EXPECTED_BRANCH"
    echo
    echo "Current branch:"
    echo "  $CURRENT_BRANCH"
    exit 1
fi

CURRENT_HEAD="$(git rev-parse HEAD)"
EXPECTED_HEAD="$(git rev-parse "$EXPECTED_PARENT")"

if [[ "$CURRENT_HEAD" != "$EXPECTED_HEAD" ]]; then
    echo "ERROR: OPT007 HEAD is no longer based directly on frozen OPT006."
    echo
    echo "Expected HEAD:"
    echo "  $EXPECTED_HEAD"
    echo
    echo "Current HEAD:"
    echo "  $CURRENT_HEAD"
    exit 1
fi

echo "Branch : $CURRENT_BRANCH"
echo "Base   : $(git rev-parse --short HEAD)"
echo "PASS"

# ---------------------------------------------------------
# 2. Verify final reports
# ---------------------------------------------------------

echo
echo "[2/7] Checking required OPT007 reports..."

REQUIRED_FILES=(
    "results/OPT007_compiler_analysis.txt"
    "results/OPT007_hot_loop_analysis.txt"
    "results/OPT007_function_level_analysis.txt"
    "results/OPT007_perf_analysis.txt"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [[ ! -s "$file" ]]; then
        echo "ERROR: Required report missing or empty:"
        echo "  $file"
        exit 1
    fi

    echo "PASS: $file"
done

# ---------------------------------------------------------
# 3. Verify experiment scripts
# ---------------------------------------------------------

echo
echo "[3/7] Checking OPT007 scripts..."

REQUIRED_SCRIPTS=(
    "scripts/run_opt007.sh"
    "scripts/run_opt007_stage2.sh"
    "scripts/run_opt007_stage3.sh"
    "scripts/run_opt007_stage4.sh"
)

for file in "${REQUIRED_SCRIPTS[@]}"; do
    if [[ ! -s "$file" ]]; then
        echo "ERROR: Required script missing or empty:"
        echo "  $file"
        exit 1
    fi

    echo "PASS: $file"
done

# ---------------------------------------------------------
# 4. Verify final perf evidence
# ---------------------------------------------------------

echo
echo "[4/7] Checking Stage 4 evidence..."

REPORT="results/OPT007_perf_analysis.txt"

grep -q "cycles" "$REPORT" || {
    echo "ERROR: cycles result missing."
    exit 1
}

grep -q "instructions" "$REPORT" || {
    echo "ERROR: instructions result missing."
    exit 1
}

grep -q "branches" "$REPORT" || {
    echo "ERROR: branches result missing."
    exit 1
}

grep -q "branch-misses" "$REPORT" || {
    echo "ERROR: branch-misses result missing."
    exit 1
}

grep -q -- "-44.313%" "$REPORT" || {
    echo "ERROR: Expected cycle result not found."
    exit 1
}

grep -q -- "-5.896%" "$REPORT" || {
    echo "ERROR: Expected instruction result not found."
    exit 1
}

grep -q -- "-27.210%" "$REPORT" || {
    echo "ERROR: Expected branch result not found."
    exit 1
}

grep -q -- "-89.949%" "$REPORT" || {
    echo "ERROR: Expected branch-miss result not found."
    exit 1
}

echo "PASS: Stage 4 hardware-counter results verified."

# ---------------------------------------------------------
# 5. Git whitespace check
# ---------------------------------------------------------

echo
echo "[5/7] Running git checks..."

git diff --check

echo "PASS: git diff --check"

# ---------------------------------------------------------
# 6. Stage ONLY intended scientific artifacts
# ---------------------------------------------------------

echo
echo "[6/7] Staging OPT007 artifacts..."

git reset

git add \
    scripts/run_opt007.sh \
    scripts/run_opt007_stage2.sh \
    scripts/run_opt007_stage3.sh \
    scripts/run_opt007_stage4.sh \
    scripts/freeze_opt007.sh \
    results/OPT007_compiler_analysis.txt \
    results/OPT007_hot_loop_analysis.txt \
    results/OPT007_function_level_analysis.txt \
    results/OPT007_perf_analysis.txt

echo
echo "Files staged for OPT007:"
echo "------------------------------------------"
git diff --cached --name-status
echo "------------------------------------------"

# Protect against accidentally staging raw perf/disassembly files.

if git diff --cached --name-only | \
    grep -E '^results/opt007/' >/dev/null; then

    echo "ERROR: Raw results/opt007 artifacts were staged."
    echo "Aborting before commit."
    git reset
    exit 1
fi

STAGED_COUNT="$(git diff --cached --name-only | wc -l)"

if [[ "$STAGED_COUNT" -ne 9 ]]; then
    echo
    echo "ERROR: Expected exactly 9 staged files."
    echo "Found: $STAGED_COUNT"
    echo
    echo "Nothing committed."
    git reset
    exit 1
fi

echo
echo "PASS: exactly 9 intended files staged."

# ---------------------------------------------------------
# 7. Commit and push
# ---------------------------------------------------------

echo
echo "[7/7] Freezing OPT007..."

git commit -m \
"OPT007: analyze compiler lowering and hardware counters"

git push -u origin "$EXPECTED_BRANCH"

echo
echo "=========================================="
echo " OPT007 FROZEN SUCCESSFULLY"
echo "=========================================="
echo
echo "Branch:"
echo "  $(git branch --show-current)"
echo
echo "Commit:"
git log -1 --oneline
echo
echo "Parent:"
git log -1 --format='%P'
echo
echo "Remote:"
git status -sb
echo
echo "OPT007 is now frozen."
