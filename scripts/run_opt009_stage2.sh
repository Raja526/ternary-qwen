#!/usr/bin/env bash
set -euo pipefail

echo "============================================================"
echo " OPT009 Stage 2 — Output-Row Blocking"
echo "============================================================"

EXPECTED_BRANCH="opt009-cache-blocking"
EXPECTED_PARENT="d3a870820d26e18bf0a40f5d21b3bc48c909e874"

BLOCK_ROWS=8

echo
echo "[1/5] Checking experiment state..."

if [[ "$(git branch --show-current)" != "$EXPECTED_BRANCH" ]]; then
    echo "ERROR: expected branch $EXPECTED_BRANCH"
    exit 1
fi

if [[ "$(git rev-parse HEAD)" != "$EXPECTED_PARENT" ]]; then
    echo "ERROR: OPT009 must start from frozen OPT008."
    exit 1
fi

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "ERROR: .venv is not active."
    exit 1
fi

echo "Parent     : $EXPECTED_PARENT"
echo "Block rows : $BLOCK_ROWS"
echo "PASS"

echo
echo "[2/5] Creating blocked kernel from frozen OPT008..."

cp \
    packed_ternary/csrc/packed_ternary_cpu_avx2.cpp \
    packed_ternary/csrc/packed_ternary_cpu_avx2_blocked.cpp

python - <<'PY'
from pathlib import Path

p = Path(
    "packed_ternary/csrc/"
    "packed_ternary_cpu_avx2_blocked.cpp"
)

s = p.read_text()

old = """    const int64_t total_jobs =
        batch_size * out_features;
"""

new = """    // OPT009 controlled variable:
    // group contiguous output rows into one parallel job.
    //
    // The computation of each individual output row remains
    // identical to frozen OPT008.
    constexpr int64_t output_row_block = 8;

    const int64_t blocks_per_batch =
        (
            out_features
            + output_row_block
            - 1
        )
        / output_row_block;

    const int64_t total_jobs =
        batch_size * blocks_per_batch;
"""

if old not in s:
    raise SystemExit(
        "ERROR: total_jobs pattern not found."
    )

s = s.replace(
    old,
    new,
    1,
)

old = """                const int64_t b =
                    job / out_features;

                const int64_t out =
                    job % out_features;

                const at::Half* input_row =
                    input_ptr
                    + b * in_features;

                const uint8_t* packed_row =
                    packed_ptr
                    + out * bytes_per_row;

                float output_acc = 0.0f;
"""

new = """                const int64_t b =
                    job / blocks_per_batch;

                const int64_t block_index =
                    job % blocks_per_batch;

                const int64_t out_begin =
                    block_index
                    * output_row_block;

                const int64_t out_end =
                    std::min(
                        out_begin
                            + output_row_block,
                        out_features
                    );

                const at::Half* input_row =
                    input_ptr
                    + b * in_features;

                for (
                    int64_t out = out_begin;
                    out < out_end;
                    ++out
                ) {

                    const uint8_t* packed_row =
                        packed_ptr
                        + out * bytes_per_row;

                    float output_acc = 0.0f;
"""

if old not in s:
    raise SystemExit(
        "ERROR: job mapping pattern not found."
    )

s = s.replace(
    old,
    new,
    1,
)

old = """                output_ptr[
                    b * out_features
                    + out
                ] = output_acc;
"""

new = """                    output_ptr[
                        b * out_features
                        + out
                    ] = output_acc;
                }
"""

if old not in s:
    raise SystemExit(
        "ERROR: output-store pattern not found."
    )

s = s.replace(
    old,
    new,
    1,
)

# std::min requires algorithm.
if "#include <algorithm>" not in s:
    s = s.replace(
        "#include <torch/extension.h>",
        "#include <torch/extension.h>\\n"
        "#include <algorithm>",
        1,
    )

p.write_text(s)

print(
    "PASS: blocked kernel generated."
)
PY

echo
echo "[3/5] Creating OPT009 extension..."

cp \
    packed_ternary/opt008_extension.py \
    packed_ternary/opt009_extension.py

python - <<'PY'
from pathlib import Path

p = Path(
    "packed_ternary/opt009_extension.py"
)

s = p.read_text()

s = s.replace(
    "opt008_packed_ternary_cpu_avx2",
    "opt009_packed_ternary_cpu_avx2_blocked",
)

s = s.replace(
    "packed_ternary_cpu_avx2.cpp",
    "packed_ternary_cpu_avx2_blocked.cpp",
)

s = s.replace(
    "load_opt008_extension",
    "load_opt009_extension",
)

p.write_text(s)

print(
    "PASS: OPT009 extension generated."
)
PY

grep -q \
'"-O2"' \
packed_ternary/opt009_extension.py

grep -q \
'"-mavx2"' \
packed_ternary/opt009_extension.py

grep -q \
'"-mno-fma"' \
packed_ternary/opt009_extension.py

echo "PASS: compiler contract unchanged."

echo
echo "[4/5] Verifying controlled source change..."

grep -n \
"output_row_block = 8" \
packed_ternary/csrc/packed_ternary_cpu_avx2_blocked.cpp

grep -n \
"blocks_per_batch" \
packed_ternary/csrc/packed_ternary_cpu_avx2_blocked.cpp

echo
echo "Compiler flags:"
grep -n -E \
'O2|mavx2|mno-fma|march|mtune' \
packed_ternary/opt009_extension.py

echo
echo "PASS: blocking structure present."

echo
echo "[5/5] Creating and running Gate 1..."

cat > results/opt009_validate_operator.py <<'PY'
import torch

from packed_ternary.opt008_extension import (
    load_opt008_extension,
)

from packed_ternary.opt009_extension import (
    load_opt009_extension,
)


GROUP_SIZE = 128
SEED = 20260928


def pack_codes(codes):
    assert codes.dtype == torch.uint8
    assert codes.numel() % 4 == 0

    c = codes.reshape(-1, 4)

    packed = (
        c[:, 0]
        | (c[:, 1] << 2)
        | (c[:, 2] << 4)
        | (c[:, 3] << 6)
    )

    return packed.contiguous()


def run_case(
    reference,
    candidate,
    batch,
    in_features,
    out_features,
):
    assert in_features % GROUP_SIZE == 0

    torch.manual_seed(
        SEED
        + batch
        + in_features
        + out_features
    )

    x = torch.randn(
        batch,
        in_features,
        dtype=torch.float16,
    )

    # Valid ternary encodings only:
    # 0 -> -1
    # 1 ->  0
    # 2 -> +1
    codes = torch.randint(
        0,
        3,
        (
            out_features
            * in_features,
        ),
        dtype=torch.uint8,
    )

    packed = pack_codes(codes)

    groups_per_row = (
        in_features
        // GROUP_SIZE
    )

    scales = torch.rand(
        out_features
        * groups_per_row,
        dtype=torch.float16,
    )

    ref = (
        reference
        .packed_ternary_linear_cpu_parallel(
            x,
            packed,
            scales,
            out_features,
            GROUP_SIZE,
        )
    )

    cand = (
        candidate
        .packed_ternary_linear_cpu_parallel(
            x,
            packed,
            scales,
            out_features,
            GROUP_SIZE,
        )
    )

    assert ref.shape == cand.shape
    assert ref.dtype == cand.dtype

    mismatches = int(
        (ref != cand)
        .sum()
        .item()
    )

    max_abs = float(
        (
            cand.float()
            - ref.float()
        )
        .abs()
        .max()
        .item()
    )

    return (
        int(ref.numel()),
        mismatches,
        max_abs,
    )


def main():
    torch.set_num_threads(6)

    reference = (
        load_opt008_extension(
            verbose=False
        )
    )

    candidate = (
        load_opt009_extension(
            verbose=False
        )
    )

    cases = [
        (1, 128, 1),
        (3, 128, 7),

        # Boundary around block size 8.
        (3, 128, 8),
        (3, 128, 9),

        (5, 256, 13),

        # Multiple complete row blocks.
        (7, 512, 31),

        # Realistic shapes.
        (22, 1024, 64),
        (22, 1024, 512),
        (22, 2048, 1024),
    ]

    total_outputs = 0
    total_mismatches = 0
    global_max = 0.0

    print(
        "=" * 72
    )
    print(
        "OPT009 GATE 1 — OPERATOR EQUIVALENCE"
    )
    print(
        "=" * 72
    )

    for case in cases:
        (
            outputs,
            mismatches,
            max_abs,
        ) = run_case(
            reference,
            candidate,
            *case,
        )

        total_outputs += outputs
        total_mismatches += mismatches
        global_max = max(
            global_max,
            max_abs,
        )

        print()
        print(
            "shape:",
            case,
        )
        print(
            "outputs       :",
            outputs,
        )
        print(
            "mismatches    :",
            mismatches,
        )
        print(
            "max abs error :",
            max_abs,
        )

    print()
    print(
        "=" * 72
    )
    print(
        "AGGREGATE"
    )
    print(
        "=" * 72
    )

    print(
        "total outputs    :",
        total_outputs,
    )
    print(
        "total mismatches :",
        total_mismatches,
    )
    print(
        "max abs error    :",
        global_max,
    )

    if total_mismatches != 0:
        raise RuntimeError(
            "FAIL: OPT009 is not bit-exact "
            "with frozen OPT008."
        )

    if global_max != 0.0:
        raise RuntimeError(
            "FAIL: non-zero operator error."
        )

    print()
    print(
        "PASS: OPT009 output-row blocking "
        "is bit-exact with frozen OPT008."
    )


if __name__ == "__main__":
    main()
PY

PYTHONPATH="$PWD" \
python results/opt009_validate_operator.py \
| tee results/OPT009_operator_equivalence.txt

grep -q \
"PASS: OPT009 output-row blocking is bit-exact with frozen OPT008." \
results/OPT009_operator_equivalence.txt

echo
echo "============================================================"
echo " OPT009 STAGE 2 COMPLETE"
echo "============================================================"
echo
echo "Controlled variable:"
echo "  output rows per parallel job: 1 -> 8"
echo
echo "Unchanged:"
echo "  AVX2 packed decode"
echo "  G128"
echo "  FP32 accumulation order per output row"
echo "  -O2 -mavx2 -mno-fma"
echo "  six-thread execution"
echo
echo "Gate 1: PASS"
echo
echo "No benchmark, commit, or push performed."
