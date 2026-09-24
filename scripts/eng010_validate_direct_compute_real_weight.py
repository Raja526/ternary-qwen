"""
ENG010 — Real-weight validation of direct ternary computation.

Target:
    model.layers.3.self_attn.k_proj

Comparison:
    A. packed weight -> reconstructed FP16 -> F.linear
    B. packed weight -> DirectTernaryLinear

The direct path does not construct a reconstructed FP16 weight matrix or
FP16 weight-row chunk.

This experiment measures numerical deviation only.
It does not claim speedup or peak-memory reduction.
"""

import torch
import torch.nn.functional as F

from packed_ternary.checkpoint import (
    load_packed_checkpoint,
)

from packed_ternary.direct_linear import (
    DirectTernaryLinear,
)

from packed_ternary.packed_weight import (
    reconstruct_packed,
)


CHECKPOINT_PATH = (
    "results/ENG003_multimatrix_checkpoint.pt"
)

TARGET = (
    "model.layers.3.self_attn.k_proj"
)

SEED = 20260924

ROWS_PER_CHUNK_VALUES = [
    1,
    8,
    32,
    64,
    128,
    512,
]

INPUT_SHAPES = [
    (4, 1024),
    (2, 7, 1024),
]


def main():
    torch.manual_seed(SEED)

    print("=" * 100)
    print(
        "ENG010 — REAL K_PROJ DIRECT TERNARY COMPUTE VALIDATION"
    )
    print("=" * 100)

    metadata, weights = (
        load_packed_checkpoint(
            CHECKPOINT_PATH
        )
    )

    packed = weights[TARGET]

    print(
        "Checkpoint model        :",
        metadata["model_name"],
    )
    print(
        "Target                  :",
        TARGET,
    )
    print(
        "Weight shape            :",
        packed.original_shape,
    )
    print(
        "Parameters              :",
        packed.original_numel,
    )
    print(
        "Group size              :",
        packed.group_size,
    )
    print(
        "Packed bytes            :",
        packed.packed_storage_bytes,
    )
    print(
        "FP16 equivalent bytes   :",
        packed.original_fp16_bytes,
    )
    print(
        "Persistent compression  :",
        f"{packed.compression_factor:.6f}x",
    )

    assert (
        packed.original_shape
        == (512, 1024)
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

    total_outputs = 0
    total_mismatches = 0

    global_max_abs = 0.0
    global_sum_abs = 0.0

    configuration_count = 0

    for input_shape in INPUT_SHAPES:
        x = torch.randn(
            *input_shape,
            dtype=torch.float16,
        )

        reference = F.linear(
            x,
            reconstructed,
            bias=None,
        )

        print()
        print("-" * 100)
        print(
            "INPUT SHAPE:",
            input_shape,
        )
        print("-" * 100)

        baseline_direct = None

        for rows_per_chunk in (
            ROWS_PER_CHUNK_VALUES
        ):
            layer = DirectTernaryLinear(
                packed_weight=packed,
                bias=None,
                rows_per_chunk=(
                    rows_per_chunk
                ),
            )

            direct = layer(x)

            assert (
                direct.shape
                == reference.shape
            )

            assert (
                direct.dtype
                == reference.dtype
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

            if baseline_direct is None:
                baseline_direct = (
                    direct
                    .detach()
                    .clone()
                )

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

            else:
                chunk_mismatch = int(
                    (
                        direct
                        != baseline_direct
                    )
                    .sum()
                    .item()
                )

                assert (
                    chunk_mismatch
                    == 0
                )

            configuration_count += 1

            print()
            print(
                "Rows per chunk         :",
                rows_per_chunk,
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
                "Direct dtype           :",
                direct.dtype,
            )

    global_mean_abs = (
        global_sum_abs
        / total_outputs
    )

    print()
    print("=" * 100)
    print("ENG010 REAL-WEIGHT SUMMARY")
    print("=" * 100)

    print(
        "Configurations          :",
        configuration_count,
    )
    print(
        "Unique input sets       :",
        len(INPUT_SHAPES),
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
        "Chunk-size invariance   : PASS"
    )

    print()
    print(
        "ENG010 REAL K_PROJ DIRECT COMPUTE VALIDATION: COMPLETE"
    )


if __name__ == "__main__":
    main()
