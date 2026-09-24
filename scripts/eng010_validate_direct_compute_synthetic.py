"""
ENG010 — Synthetic validation of direct ternary computation.

Compares:

    A. frozen packed representation
       -> reconstruct FP16 weight
       -> torch.nn.functional.linear

against:

    B. frozen packed representation
       -> DirectTernaryLinear
       -> no reconstructed FP16 weight tensor

This experiment measures numerical differences caused by changing the
floating-point operation ordering.

It does NOT claim speedup or memory reduction.
"""

import torch
import torch.nn.functional as F

from packed_ternary.direct_linear import (
    DirectTernaryLinear,
)

from packed_ternary.packed_weight import (
    quantize_to_packed,
    reconstruct_packed,
)


SEED = 20260924

GROUP_SIZE = 128

TESTS = [
    # out_features, in_features, batch_shape, rows_per_chunk
    (4, 128, (1,), 1),
    (8, 256, (3,), 2),
    (16, 512, (2,), 4),
    (32, 1024, (4,), 8),
    (7, 256, (2, 3), 3),
    (13, 512, (2, 5), 4),
]


def main():
    torch.manual_seed(SEED)

    print("=" * 100)
    print("ENG010 — SYNTHETIC DIRECT TERNARY COMPUTE VALIDATION")
    print("=" * 100)

    total_outputs = 0
    total_mismatches = 0

    global_max_abs = 0.0
    global_sum_abs = 0.0

    exact_tests = 0

    for test_index, (
        out_features,
        in_features,
        batch_shape,
        rows_per_chunk,
    ) in enumerate(
        TESTS,
        start=1,
    ):
        print()
        print("-" * 100)
        print(f"TEST {test_index}")
        print("-" * 100)

        weight = torch.randn(
            out_features,
            in_features,
            dtype=torch.float16,
        )

        packed = quantize_to_packed(
            weight,
            group_size=GROUP_SIZE,
        )

        assert (
            packed.padded_numel
            == packed.original_numel
        )

        reconstructed = (
            reconstruct_packed(
                packed
            )
        )

        x = torch.randn(
            *batch_shape,
            in_features,
            dtype=torch.float16,
        )

        reference = F.linear(
            x,
            reconstructed,
            bias=None,
        )

        direct_layer = (
            DirectTernaryLinear(
                packed_weight=packed,
                bias=None,
                rows_per_chunk=(
                    rows_per_chunk
                ),
            )
        )

        direct = direct_layer(x)

        assert (
            direct.shape
            == reference.shape
        )

        difference = (
            direct.float()
            - reference.float()
        ).abs()

        mismatch_count = int(
            (
                direct
                != reference
            )
            .sum()
            .item()
        )

        output_count = (
            reference.numel()
        )

        max_abs = float(
            difference.max().item()
        )

        mean_abs = float(
            difference.mean().item()
        )

        sum_abs = float(
            difference.sum().item()
        )

        exact = (
            mismatch_count == 0
        )

        if exact:
            exact_tests += 1

        total_outputs += (
            output_count
        )

        total_mismatches += (
            mismatch_count
        )

        global_sum_abs += (
            sum_abs
        )

        global_max_abs = max(
            global_max_abs,
            max_abs,
        )

        print(
            "Weight shape           :",
            tuple(weight.shape),
        )

        print(
            "Input shape            :",
            tuple(x.shape),
        )

        print(
            "Output shape           :",
            tuple(reference.shape),
        )

        print(
            "Groups per row         :",
            in_features
            // GROUP_SIZE,
        )

        print(
            "Rows per chunk         :",
            rows_per_chunk,
        )

        print(
            "Reference dtype        :",
            reference.dtype,
        )

        print(
            "Direct dtype           :",
            direct.dtype,
        )

        print(
            "Outputs compared       :",
            output_count,
        )

        print(
            "Mismatches             :",
            mismatch_count,
        )

        print(
            "Mismatch percent       :",
            (
                100.0
                * mismatch_count
                / output_count
            ),
        )

        print(
            "Max abs difference     :",
            max_abs,
        )

        print(
            "Mean abs difference    :",
            mean_abs,
        )

        print(
            "Bit-exact              :",
            exact,
        )

    print()
    print("=" * 100)
    print("ENG010 SYNTHETIC SUMMARY")
    print("=" * 100)

    global_mean_abs = (
        global_sum_abs
        / total_outputs
    )

    print(
        "Tests                   :",
        len(TESTS),
    )

    print(
        "Bit-exact tests         :",
        exact_tests,
    )

    print(
        "Outputs compared        :",
        total_outputs,
    )

    print(
        "Total mismatches        :",
        total_mismatches,
    )

    print(
        "Mismatch percent        :",
        (
            100.0
            * total_mismatches
            / total_outputs
        ),
    )

    print(
        "Global max abs diff     :",
        global_max_abs,
    )

    print(
        "Global mean abs diff    :",
        global_mean_abs,
    )

    print()
    print(
        "NOTE: Nonzero differences are not automatically failures."
    )

    print(
        "ENG010 is measuring the numerical effect of replacing "
        "FP16 reconstructed-weight F.linear with direct ternary "
        "accumulation."
    )

    print()
    print(
        "ENG010 SYNTHETIC DIRECT COMPUTE VALIDATION: COMPLETE"
    )


if __name__ == "__main__":
    main()
