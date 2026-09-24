"""
ENG011-A synthetic validation.

Objective
---------
Verify that StreamingDirectTernaryLinear produces bit-exact outputs relative
to the frozen ENG010 DirectTernaryLinear when both consume the same packed
ternary weight.

The only intended change is:

    ENG010: decode the full matrix to temporary int8 codes
    ENG011: decode only the current output-row chunk

Therefore exact output equality is required.
"""

import torch

from packed_ternary.packed_weight import quantize_to_packed
from packed_ternary.direct_linear import DirectTernaryLinear
from packed_ternary.streaming_direct_linear import (
    StreamingDirectTernaryLinear,
)


SEED = 20260924
GROUP_SIZE = 128
THRESHOLD_MULTIPLIER = 0.5


def run_case(
    test_id,
    out_features,
    in_features,
    input_shape,
    rows_per_chunk,
):
    torch.manual_seed(
        SEED + test_id
    )

    weight = torch.randn(
        out_features,
        in_features,
        dtype=torch.float16,
    )

    packed = quantize_to_packed(
        weight,
        group_size=GROUP_SIZE,
        threshold_multiplier=THRESHOLD_MULTIPLIER,
    )

    reference = DirectTernaryLinear(
        packed_weight=packed,
        rows_per_chunk=rows_per_chunk,
    )

    streaming = StreamingDirectTernaryLinear(
        packed_weight=packed,
        rows_per_chunk=rows_per_chunk,
    )

    input_tensor = torch.randn(
        *input_shape,
        dtype=torch.float16,
    )

    with torch.inference_mode():

        reference_output = reference(
            input_tensor
        )

        streaming_output = streaming(
            input_tensor
        )

    if (
        reference_output.shape
        != streaming_output.shape
    ):
        raise AssertionError(
            "Output shape mismatch"
        )

    if (
        reference_output.dtype
        != streaming_output.dtype
    ):
        raise AssertionError(
            "Output dtype mismatch"
        )

    diff = (
        reference_output.float()
        - streaming_output.float()
    ).abs()

    mismatch_count = int(
        (
            reference_output
            != streaming_output
        )
        .sum()
        .item()
    )

    total_outputs = (
        reference_output.numel()
    )

    max_abs_diff = float(
        diff.max().item()
    )

    mean_abs_diff = float(
        diff.mean().item()
    )

    expected_full_decode_bytes = (
        packed.original_numel
    )

    expected_chunk_decode_bytes = (
        min(
            rows_per_chunk,
            out_features,
        )
        * in_features
    )

    if (
        streaming.full_decoded_code_bytes
        != expected_full_decode_bytes
    ):
        raise AssertionError(
            "Full decoded-code byte accounting mismatch"
        )

    if (
        streaming.maximum_chunk_decoded_code_bytes
        != expected_chunk_decode_bytes
    ):
        raise AssertionError(
            "Chunk decoded-code byte accounting mismatch"
        )

    if mismatch_count != 0:
        raise AssertionError(
            f"ENG011 output mismatch: {mismatch_count}"
        )

    print(
        f"TEST {test_id}"
    )

    print(
        f"  Weight shape              : "
        f"({out_features}, {in_features})"
    )

    print(
        f"  Input shape               : "
        f"{tuple(input_tensor.shape)}"
    )

    print(
        f"  Rows per chunk            : "
        f"{rows_per_chunk}"
    )

    print(
        f"  Outputs                   : "
        f"{total_outputs}"
    )

    print(
        f"  Mismatches                : "
        f"{mismatch_count}"
    )

    print(
        f"  Max abs diff              : "
        f"{max_abs_diff}"
    )

    print(
        f"  Mean abs diff             : "
        f"{mean_abs_diff}"
    )

    print(
        f"  ENG010 full decode bytes  : "
        f"{expected_full_decode_bytes}"
    )

    print(
        f"  ENG011 max chunk bytes    : "
        f"{expected_chunk_decode_bytes}"
    )

    print(
        f"  Decode working-set ratio  : "
        f"{expected_full_decode_bytes / expected_chunk_decode_bytes:.6f}x"
    )

    print()

    return (
        total_outputs,
        mismatch_count,
        max_abs_diff,
        expected_full_decode_bytes,
        expected_chunk_decode_bytes,
    )


def main():

    print(
        "ENG011-A STREAMING DIRECT TERNARY SYNTHETIC VALIDATION"
    )

    print(
        "====================================================="
    )

    print()

    cases = [
        (1, 8, 128, (1, 128), 1),
        (2, 16, 256, (3, 256), 4),
        (3, 32, 512, (2, 512), 8),
        (4, 64, 1024, (4, 1024), 16),
        (5, 13, 256, (2, 3, 256), 5),
        (6, 17, 512, (2, 5, 512), 7),
        (7, 65, 1024, (2, 1024), 64),
    ]

    total_outputs = 0
    total_mismatches = 0
    global_max_abs_diff = 0.0

    for case in cases:

        (
            outputs,
            mismatches,
            max_abs_diff,
            _,
            _,
        ) = run_case(
            *case
        )

        total_outputs += outputs
        total_mismatches += mismatches

        global_max_abs_diff = max(
            global_max_abs_diff,
            max_abs_diff,
        )

    print(
        "SUMMARY"
    )

    print(
        "-------"
    )

    print(
        f"Tests                     : "
        f"{len(cases)}"
    )

    print(
        f"Total outputs             : "
        f"{total_outputs}"
    )

    print(
        f"Total mismatches          : "
        f"{total_mismatches}"
    )

    print(
        f"Global max abs diff       : "
        f"{global_max_abs_diff}"
    )

    print(
        "Bit-exact equivalence     : "
        f"{total_mismatches == 0}"
    )

    if total_mismatches != 0:
        raise AssertionError(
            "ENG011-A failed bit-exact equivalence"
        )

    print()

    print(
        "ENG011-A STREAMING SYNTHETIC VALIDATION: PASS"
    )


if __name__ == "__main__":
    main()
