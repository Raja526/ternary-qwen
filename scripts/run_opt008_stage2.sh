#!/usr/bin/env bash
set -euo pipefail

echo "=========================================="
echo " OPT008 Stage 2 — AVX2 Decode Kernel"
echo "=========================================="

EXPECTED_BRANCH="opt008-avx2-ternary-compute"
EXPECTED_PARENT="cd2f52b"

BASE_SOURCE="packed_ternary/csrc/packed_ternary_cpu_table_dispatch.cpp"
NEW_SOURCE="packed_ternary/csrc/packed_ternary_cpu_avx2.cpp"

BASE_EXTENSION="packed_ternary/opt006_extension.py"
NEW_EXTENSION="packed_ternary/opt008_extension.py"

echo
echo "[1/6] Checking experiment state..."

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

grep -qw avx2 < <(lscpu | grep '^Flags:' | tr ' ' '\n') || {
    echo "ERROR: AVX2 unavailable."
    exit 1
}

echo "PASS: controlled OPT008 state."

echo
echo "[2/6] Creating AVX2 candidate source..."

cp "$BASE_SOURCE" "$NEW_SOURCE"

python - <<'PY'
from pathlib import Path

p = Path("packed_ternary/csrc/packed_ternary_cpu_avx2.cpp")
s = p.read_text()

# -------------------------------------------------------
# Add AVX2 intrinsic header.
# -------------------------------------------------------

if "#include <immintrin.h>" not in s:
    lines = s.splitlines()

    insert_at = 0
    for i, line in enumerate(lines):
        if line.startswith("#include"):
            insert_at = i + 1

    lines.insert(insert_at, "#include <immintrin.h>")
    s = "\n".join(lines) + "\n"

# -------------------------------------------------------
# Replace only the packed-byte inner loop.
#
# OPT006:
#   one packed byte -> four codes
#
# OPT008:
#   eight packed bytes -> 32 codes using AVX2
#
# Accumulation remains scalar and in original logical
# x0,x1,... order to preserve FP32 reduction semantics.
# -------------------------------------------------------

start_marker = """                    for (int64_t byte_index = 0;
                         byte_index < group_bytes;
                         ++byte_index) {"""

end_marker = """                    const int64_t scale_index ="""

start = s.find(start_marker)

if start == -1:
    raise SystemExit(
        "ERROR: OPT006 inner byte loop start not found."
    )

end = s.find(end_marker, start)

if end == -1:
    raise SystemExit(
        "ERROR: scale_index boundary not found."
    )

replacement = r'''                    /*
                     * OPT008 controlled change:
                     *
                     * Decode eight packed bytes at once.
                     *
                     * Eight bytes contain 32 ternary codes.
                     * AVX2 expands these into 32 uint8 code
                     * values.
                     *
                     * IMPORTANT:
                     * Activation conversion and FP32
                     * accumulation remain scalar and occur
                     * in the original logical input order.
                     *
                     * This avoids changing reduction order
                     * in the first SIMD experiment.
                     */

                    alignas(32) uint8_t decoded_codes[32];

                    int64_t byte_index = 0;

                    for (;
                         byte_index + 8 <= group_bytes;
                         byte_index += 8) {

                        uint64_t packed8 = 0;

                        std::memcpy(
                            &packed8,
                            packed_row
                                + group_byte_start
                                + byte_index,
                            sizeof(uint64_t)
                        );

                        /*
                         * Put the eight packed bytes into the
                         * low 64 bits.
                         */
                        const __m128i packed128 =
                            _mm_cvtsi64_si128(
                                static_cast<long long>(
                                    packed8
                                )
                            );

                        /*
                         * Expand 8 uint8 values to 8 uint32
                         * lanes.
                         */
                        const __m256i packed32 =
                            _mm256_cvtepu8_epi32(
                                packed128
                            );

                        const __m256i mask =
                            _mm256_set1_epi32(0x3);

                        const __m256i c0 =
                            _mm256_and_si256(
                                packed32,
                                mask
                            );

                        const __m256i c1 =
                            _mm256_and_si256(
                                _mm256_srli_epi32(
                                    packed32,
                                    2
                                ),
                                mask
                            );

                        const __m256i c2 =
                            _mm256_and_si256(
                                _mm256_srli_epi32(
                                    packed32,
                                    4
                                ),
                                mask
                            );

                        const __m256i c3 =
                            _mm256_and_si256(
                                _mm256_srli_epi32(
                                    packed32,
                                    6
                                ),
                                mask
                            );

                        alignas(32) uint32_t t0[8];
                        alignas(32) uint32_t t1[8];
                        alignas(32) uint32_t t2[8];
                        alignas(32) uint32_t t3[8];

                        _mm256_store_si256(
                            reinterpret_cast<__m256i*>(t0),
                            c0
                        );

                        _mm256_store_si256(
                            reinterpret_cast<__m256i*>(t1),
                            c1
                        );

                        _mm256_store_si256(
                            reinterpret_cast<__m256i*>(t2),
                            c2
                        );

                        _mm256_store_si256(
                            reinterpret_cast<__m256i*>(t3),
                            c3
                        );

                        for (int lane = 0;
                             lane < 8;
                             ++lane) {

                            const uint8_t code0 =
                                static_cast<uint8_t>(
                                    t0[lane]
                                );

                            const uint8_t code1 =
                                static_cast<uint8_t>(
                                    t1[lane]
                                );

                            const uint8_t code2 =
                                static_cast<uint8_t>(
                                    t2[lane]
                                );

                            const uint8_t code3 =
                                static_cast<uint8_t>(
                                    t3[lane]
                                );

                            TORCH_CHECK(
                                code0 != 0x3
                                && code1 != 0x3
                                && code2 != 0x3
                                && code3 != 0x3,
                                "reserved ternary code "
                                "11 encountered"
                            );

                            const int64_t input_base =
                                group_input_start
                                + (
                                    byte_index
                                    + lane
                                ) * 4;

                            const float x0 =
                                static_cast<float>(
                                    input_row[
                                        input_base
                                    ]
                                );

                            const float x1 =
                                static_cast<float>(
                                    input_row[
                                        input_base + 1
                                    ]
                                );

                            const float x2 =
                                static_cast<float>(
                                    input_row[
                                        input_base + 2
                                    ]
                                );

                            const float x3 =
                                static_cast<float>(
                                    input_row[
                                        input_base + 3
                                    ]
                                );

                            positive_sum +=
                                x0
                                * positive_lut[
                                    code0
                                ];

                            negative_sum +=
                                x0
                                * negative_lut[
                                    code0
                                ];

                            positive_sum +=
                                x1
                                * positive_lut[
                                    code1
                                ];

                            negative_sum +=
                                x1
                                * negative_lut[
                                    code1
                                ];

                            positive_sum +=
                                x2
                                * positive_lut[
                                    code2
                                ];

                            negative_sum +=
                                x2
                                * negative_lut[
                                    code2
                                ];

                            positive_sum +=
                                x3
                                * positive_lut[
                                    code3
                                ];

                            negative_sum +=
                                x3
                                * negative_lut[
                                    code3
                                ];
                        }
                    }

                    /*
                     * Scalar tail.
                     *
                     * G128 currently gives exactly 32 bytes
                     * per group, so the frozen experiment
                     * should normally have no tail.
                     *
                     * Keeping this path makes the operator
                     * structurally safe for compatible group
                     * sizes not divisible by eight.
                     */
                    for (;
                         byte_index < group_bytes;
                         ++byte_index) {

                        const uint8_t packed =
                            packed_row[
                                group_byte_start
                                + byte_index
                            ];

                        const uint8_t code0 =
                            packed & 0x3;

                        const uint8_t code1 =
                            (packed >> 2) & 0x3;

                        const uint8_t code2 =
                            (packed >> 4) & 0x3;

                        const uint8_t code3 =
                            (packed >> 6) & 0x3;

                        TORCH_CHECK(
                            code0 != 0x3
                            && code1 != 0x3
                            && code2 != 0x3
                            && code3 != 0x3,
                            "reserved ternary code "
                            "11 encountered"
                        );

                        const int64_t input_base =
                            group_input_start
                            + byte_index * 4;

                        const float x0 =
                            static_cast<float>(
                                input_row[
                                    input_base
                                ]
                            );

                        const float x1 =
                            static_cast<float>(
                                input_row[
                                    input_base + 1
                                ]
                            );

                        const float x2 =
                            static_cast<float>(
                                input_row[
                                    input_base + 2
                                ]
                            );

                        const float x3 =
                            static_cast<float>(
                                input_row[
                                    input_base + 3
                                ]
                            );

                        positive_sum +=
                            x0 * positive_lut[code0];

                        negative_sum +=
                            x0 * negative_lut[code0];

                        positive_sum +=
                            x1 * positive_lut[code1];

                        negative_sum +=
                            x1 * negative_lut[code1];

                        positive_sum +=
                            x2 * positive_lut[code2];

                        negative_sum +=
                            x2 * negative_lut[code2];

                        positive_sum +=
                            x3 * positive_lut[code3];

                        negative_sum +=
                            x3 * negative_lut[code3];
                    }

'''

s = s[:start] + replacement + s[end:]

# memcpy is used explicitly.
if "#include <cstring>" not in s:
    lines = s.splitlines()

    insert_at = 0
    for i, line in enumerate(lines):
        if line.startswith("#include"):
            insert_at = i + 1

    lines.insert(insert_at, "#include <cstring>")
    s = "\n".join(lines) + "\n"

p.write_text(s)

print("PASS: created explicit AVX2 decode candidate.")
PY

echo
echo "[3/6] Creating OPT008 extension loader..."

cp "$BASE_EXTENSION" "$NEW_EXTENSION"

python - <<'PY'
from pathlib import Path

p = Path("packed_ternary/opt008_extension.py")
s = p.read_text()

s = s.replace(
    "packed_ternary_cpu_table_dispatch.cpp",
    "packed_ternary_cpu_avx2.cpp",
)

# Replace the frozen OPT006 extension name wherever present.
s = s.replace(
    "opt006_packed_ternary_cpu_table_dispatch",
    "opt008_packed_ternary_cpu_avx2",
)

s = s.replace(
    "load_opt006_extension",
    "load_opt008_extension",
)

# Add only the explicit ISA flags.
needle = '"-O2",'

if needle not in s:
    raise SystemExit(
        "ERROR: Could not locate OPT006 -O2 flag."
    )

s = s.replace(
    needle,
    '"-O2",\n'
    '            "-mavx2",\n'
    '            "-mno-fma",',
    1,
)

p.write_text(s)

print("PASS: created OPT008 extension loader.")
PY

echo
echo "[4/6] Verifying controlled source differences..."

echo
echo "Compiler flags:"
grep -n -A6 "extra_cflags" "$NEW_EXTENSION"

echo
echo "AVX2 intrinsics:"
grep -n -E \
'_mm256_|_mm_cvtsi64_si128' \
"$NEW_SOURCE" | head -40

if grep -q -- '-march=native' "$NEW_EXTENSION"; then
    echo "ERROR: -march=native detected."
    exit 1
fi

if grep -q -- '-mfma' "$NEW_EXTENSION"; then
    echo "ERROR: FMA enabling flag detected."
    exit 1
fi

grep -q -- '-mavx2' "$NEW_EXTENSION" || {
    echo "ERROR: -mavx2 missing."
    exit 1
}

grep -q -- '-mno-fma' "$NEW_EXTENSION" || {
    echo "ERROR: -mno-fma missing."
    exit 1
}

echo "PASS: controlled AVX2 build configuration."

echo
echo "[5/6] Building OPT006 + OPT008 and running Gate 1..."

cat > results/opt008_validate_operator.py <<'PY'
import torch

from packed_ternary.opt006_extension import (
    load_opt006_extension,
)

from packed_ternary.opt008_extension import (
    load_opt008_extension,
)


torch.manual_seed(20260928)
torch.set_num_threads(6)
torch.set_num_interop_threads(6)

opt006 = load_opt006_extension()
opt008 = load_opt008_extension()

GROUP_SIZE = 128


def pack_codes(codes):
    assert codes.ndim == 2
    assert codes.shape[1] % 4 == 0

    c = codes.reshape(
        codes.shape[0],
        -1,
        4,
    ).to(torch.uint8)

    packed = (
        c[..., 0]
        | (c[..., 1] << 2)
        | (c[..., 2] << 4)
        | (c[..., 3] << 6)
    )

    return packed.contiguous().reshape(-1)


shapes = [
    (1, 128, 1),
    (3, 128, 7),
    (5, 256, 13),
    (7, 512, 31),
    (22, 1024, 64),
    (22, 1024, 1024),
]

total_outputs = 0
total_mismatches = 0
global_max_abs = 0.0

print("=" * 72)
print("OPT008 GATE 1 — OPERATOR EQUIVALENCE")
print("=" * 72)

for batch, in_features, out_features in shapes:

    assert in_features % GROUP_SIZE == 0

    groups_per_row = (
        in_features // GROUP_SIZE
    )

    x = torch.randn(
        batch,
        in_features,
        dtype=torch.float16,
    ).contiguous()

    codes = torch.randint(
        0,
        3,
        (
            out_features,
            in_features,
        ),
        dtype=torch.uint8,
    )

    packed = pack_codes(codes)

    scales = torch.rand(
        out_features,
        groups_per_row,
        dtype=torch.float16,
    ).contiguous()

    with torch.no_grad():

        y006 = (
            opt006
            .packed_ternary_linear_cpu_parallel(
                x,
                packed,
                scales,
                out_features,
                GROUP_SIZE,
            )
        )

        y008 = (
            opt008
            .packed_ternary_linear_cpu_parallel(
                x,
                packed,
                scales,
                out_features,
                GROUP_SIZE,
            )
        )

    if y006.shape != y008.shape:
        raise RuntimeError(
            f"Shape mismatch: "
            f"{y006.shape} vs {y008.shape}"
        )

    exact = torch.eq(y006, y008)

    mismatches = (
        (~exact)
        .sum()
        .item()
    )

    diff = (
        y006.float()
        - y008.float()
    ).abs()

    max_abs = (
        diff.max().item()
        if diff.numel()
        else 0.0
    )

    outputs = y006.numel()

    total_outputs += outputs
    total_mismatches += mismatches
    global_max_abs = max(
        global_max_abs,
        max_abs,
    )

    print()
    print(
        "shape:",
        (
            batch,
            in_features,
            out_features,
        ),
    )
    print("outputs       :", outputs)
    print("mismatches    :", mismatches)
    print("max abs error :", max_abs)

print()
print("=" * 72)
print("AGGREGATE")
print("=" * 72)
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
    global_max_abs,
)

if total_mismatches != 0:
    raise SystemExit(
        "FAIL: OPT008 is not bit-exact "
        "with frozen OPT006."
    )

if global_max_abs != 0.0:
    raise SystemExit(
        "FAIL: non-zero numerical "
        "difference detected."
    )

print()
print(
    "PASS: OPT008 operator is bit-exact "
    "with frozen OPT006."
)
PY

PYTHONPATH="$PWD" \
python results/opt008_validate_operator.py \
| tee results/OPT008_operator_equivalence.txt

echo
echo "[6/6] Inspecting generated AVX2 machine code..."

SO_PATH="$(
PYTHONPATH="$PWD" python - <<'PY'
from packed_ternary.opt008_extension import load_opt008_extension

ext = load_opt008_extension()
print(ext.__file__)
PY
)"

echo "Extension:"
echo "  $SO_PATH"

objdump -d -M intel "$SO_PATH" \
    > results/opt008/opt008_full_disassembly.asm

AVX_COUNT="$(
    grep -E '\bv(pand|psrld|pmovzxbd|movdqu|movdqa|movq)\b' \
    results/opt008/opt008_full_disassembly.asm \
    | wc -l
)"

echo "AVX/AVX2 relevant instruction matches: $AVX_COUNT"

if [[ "$AVX_COUNT" -eq 0 ]]; then
    echo "ERROR: No expected AVX/AVX2 instructions found."
    exit 1
fi

{
    echo "OPT008 Stage 2"
    echo "=============="
    echo
    echo "Controlled change:"
    echo "  Eight packed bytes decoded using explicit AVX2."
    echo
    echo "Arithmetic:"
    echo "  FP32 positive/negative accumulation remains"
    echo "  scalar and in original logical input order."
    echo
    echo "Compiler:"
    echo "  -O2 -mavx2 -mno-fma"
    echo
    echo "Correctness:"
    tail -20 results/OPT008_operator_equivalence.txt
    echo
    echo "Machine-code evidence:"
    echo "  relevant AVX/AVX2 matches: $AVX_COUNT"
} > results/OPT008_stage2.txt

echo
echo "=========================================="
echo " OPT008 STAGE 2 COMPLETE"
echo "=========================================="
echo
echo "Gate 1:"
echo "  PASS — operator bit-exact vs OPT006"
echo
echo "Report:"
echo "  results/OPT008_operator_equivalence.txt"
echo "  results/OPT008_stage2.txt"
echo
echo "IMPORTANT:"
echo "  No benchmark has been run."
echo "  No full-model claim has been made."
echo "  No commit or push has occurred."
