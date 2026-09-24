"""
ENG011-B real-weight streaming validation.

Compare the frozen ENG010 DirectTernaryLinear against ENG011
StreamingDirectTernaryLinear on the real ENG003 k_proj packed weight.

Only the decoding scope changes:

    ENG010 -> full-matrix int8 decode
    ENG011 -> current-row-chunk int8 decode

Bit-exact output equality is required.
"""

import torch

from packed_ternary.checkpoint import (
    load_packed_checkpoint,
)

from packed_ternary.direct_linear import (
    DirectTernaryLinear,
)

from packed_ternary.streaming_direct_linear import (
    StreamingDirectTernaryLinear,
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


def main():

    torch.manual_seed(SEED)

    print(
        "ENG011-B REAL-WEIGHT STREAMING VALIDATION"
    )
    print(
        "========================================="
    )
    print()

    (
        metadata,
        weights,
    ) = load_packed_checkpoint(
        CHECKPOINT_PATH
    )

    if TARGET not in weights:
        raise KeyError(
            f"Target not found in checkpoint: {TARGET}"
        )

    packed = weights[TARGET]

    out_features = int(
        packed.original_shape[0]
    )

    in_features = int(
        packed.original_shape[1]
    )

    print(f"Checkpoint       : {CHECKPOINT_PATH}")
    print(f"Target           : {TARGET}")
    print(f"Shape            : {packed.original_shape}")
    print(f"Parameters       : {packed.original_numel}")
    print(f"Group size       : {packed.group_size}")
    print(
        f"Packed bytes     : "
        f"{packed.packed_storage_bytes}"
    )
    print(
        f"FP16 bytes       : "
        f"{packed.original_fp16_bytes}"
    )
    print(
        f"Compression      : "
        f"{packed.compression_factor:.6f}x"
    )
    print()

    input_sets = [
        torch.randn(
            4,
            in_features,
            dtype=torch.float16,
        ),
        torch.randn(
            2,
            7,
            in_features,
            dtype=torch.float16,
        ),
    ]

    total_outputs = 0
    total_mismatches = 0
    global_max_abs_diff = 0.0

    for input_index, input_tensor in enumerate(
        input_sets,
        start=1,
    ):

        print(
            f"INPUT SET {input_index}"
        )
        print(
            f"  Shape: {tuple(input_tensor.shape)}"
        )
        print()

        for rows_per_chunk in ROWS_PER_CHUNK_VALUES:

            reference = DirectTernaryLinear(
                packed_weight=packed,
                rows_per_chunk=rows_per_chunk,
            )

            streaming = StreamingDirectTernaryLinear(
                packed_weight=packed,
                rows_per_chunk=rows_per_chunk,
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

            mismatch_count = int(
                (
                    reference_output
                    != streaming_output
                )
                .sum()
                .item()
            )

            diff = (
                reference_output.float()
                - streaming_output.float()
            ).abs()

            max_abs_diff = float(
                diff.max().item()
            )

            mean_abs_diff = float(
                diff.mean().item()
            )

            output_count = (
                reference_output.numel()
            )

            full_decode_bytes = (
                streaming.full_decoded_code_bytes
            )

            max_chunk_decode_bytes = (
                streaming.maximum_chunk_decoded_code_bytes
            )

            decode_ratio = (
                full_decode_bytes
                / max_chunk_decode_bytes
            )

            print(
                f"  rows_per_chunk={rows_per_chunk}"
            )

            print(
                f"    Outputs                  : "
                f"{output_count}"
            )

            print(
                f"    Mismatches               : "
                f"{mismatch_count}"
            )

            print(
                f"    Max abs diff             : "
                f"{max_abs_diff}"
            )

            print(
                f"    Mean abs diff            : "
                f"{mean_abs_diff}"
            )

            print(
                f"    ENG010 decode bytes      : "
                f"{full_decode_bytes}"
            )

            print(
                f"    ENG011 max chunk bytes   : "
                f"{max_chunk_decode_bytes}"
            )

            print(
                f"    Decode working-set ratio : "
                f"{decode_ratio:.6f}x"
            )

            if mismatch_count != 0:
                raise AssertionError(
                    "ENG011 real-weight output mismatch"
                )

            total_outputs += (
                output_count
            )

            total_mismatches += (
                mismatch_count
            )

            global_max_abs_diff = max(
                global_max_abs_diff,
                max_abs_diff,
            )

        print()

    print("SUMMARY")
    print("-------")
    print(
        f"Configurations          : "
        f"{len(input_sets) * len(ROWS_PER_CHUNK_VALUES)}"
    )
    print(
        f"Total outputs           : "
        f"{total_outputs}"
    )
    print(
        f"Total mismatches        : "
        f"{total_mismatches}"
    )
    print(
        f"Global max abs diff     : "
        f"{global_max_abs_diff}"
    )
    print(
        f"Bit-exact equivalence   : "
        f"{total_mismatches == 0}"
    )

    if total_mismatches != 0:
        raise AssertionError(
            "ENG011-B failed"
        )

    print()
    print(
        "ENG011-B REAL-WEIGHT STREAMING VALIDATION: PASS"
    )


if __name__ == "__main__":
    main()
