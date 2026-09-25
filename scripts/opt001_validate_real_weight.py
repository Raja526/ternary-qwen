"""
OPT001 real-weight correctness validation.

Compare the frozen ENG011 StreamingDirectTernaryLinear against
the compiled OPT001 packed ternary CPU operator on the real
ENG003 k_proj packed weight.

Controlled change:

    ENG011:
        packed codes
        -> chunk int8 decode
        -> positive/negative masks
        -> PyTorch reductions

    OPT001:
        packed codes
        -> direct C++ 2-bit interpretation
        -> direct accumulation

No reconstructed FP16 weight is created by OPT001.
"""

import torch

from packed_ternary.checkpoint import (
    load_packed_checkpoint,
)

from packed_ternary.streaming_direct_linear import (
    StreamingDirectTernaryLinear,
)

from packed_ternary.opt001_extension import (
    load_opt001_extension,
)


CHECKPOINT_PATH = (
    "results/ENG003_multimatrix_checkpoint.pt"
)

TARGET = (
    "model.layers.3.self_attn.k_proj"
)

SEED = 20260924
ROWS_PER_CHUNK = 64


def main():

    torch.manual_seed(SEED)

    print(
        "OPT001 REAL-WEIGHT CORRECTNESS VALIDATION"
    )
    print(
        "=" * 72
    )
    print()

    metadata, weights = (
        load_packed_checkpoint(
            CHECKPOINT_PATH
        )
    )

    if TARGET not in weights:
        raise KeyError(
            f"Target not found: {TARGET}"
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
        f"Packed storage   : "
        f"{packed.packed_storage_bytes} bytes"
    )
    print(
        f"FP16 equivalent  : "
        f"{packed.original_fp16_bytes} bytes"
    )
    print(
        f"Compression      : "
        f"{packed.compression_factor:.6f}x"
    )
    print()

    # Two input layouts:
    #
    # 1. already flattened
    # 2. transformer-like [batch, sequence, hidden]
    #
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

    reference = (
        StreamingDirectTernaryLinear(
            packed_weight=packed,
            rows_per_chunk=ROWS_PER_CHUNK,
        )
    )

    ext = load_opt001_extension(
        verbose=False
    )

    total_values = 0
    total_mismatches = 0
    global_max_abs_diff = 0.0
    weighted_abs_diff_sum = 0.0

    for index, input_tensor in enumerate(
        input_sets,
        start=1,
    ):

        print(f"INPUT SET {index}")
        print(
            f"  Shape                  : "
            f"{tuple(input_tensor.shape)}"
        )

        original_shape = tuple(
            input_tensor.shape
        )

        flat_input = (
            input_tensor
            .reshape(
                -1,
                in_features,
            )
            .contiguous()
        )

        with torch.inference_mode():

            reference_output = reference(
                input_tensor
            )

            opt001_flat = (
                ext.packed_ternary_linear_cpu(
                    flat_input,
                    packed.packed_codes,
                    packed.scales.reshape(-1),
                    out_features,
                    packed.group_size,
                )
            )

        opt001_output = (
            opt001_flat.reshape(
                *original_shape[:-1],
                out_features,
            )
        )

        if (
            reference_output.shape
            != opt001_output.shape
        ):
            raise AssertionError(
                "Output shape mismatch"
            )

        if (
            reference_output.dtype
            != opt001_output.dtype
        ):
            raise AssertionError(
                "Output dtype mismatch"
            )

        diff = (
            reference_output.float()
            - opt001_output.float()
        ).abs()

        mismatch_count = int(
            (
                reference_output
                != opt001_output
            )
            .sum()
            .item()
        )

        value_count = int(
            reference_output.numel()
        )

        max_abs_diff = float(
            diff.max().item()
        )

        mean_abs_diff = float(
            diff.mean().item()
        )

        close = torch.allclose(
            reference_output,
            opt001_output,
            atol=2e-2,
            rtol=2e-2,
        )

        print(
            f"  Compared values        : "
            f"{value_count}"
        )
        print(
            f"  Exact mismatches       : "
            f"{mismatch_count}"
        )
        print(
            f"  Max absolute diff      : "
            f"{max_abs_diff}"
        )
        print(
            f"  Mean absolute diff     : "
            f"{mean_abs_diff}"
        )
        print(
            f"  Numerically close      : "
            f"{close}"
        )
        print()

        if not close:
            raise RuntimeError(
                f"OPT001 real-weight input set "
                f"{index} FAILED"
            )

        total_values += value_count
        total_mismatches += mismatch_count

        global_max_abs_diff = max(
            global_max_abs_diff,
            max_abs_diff,
        )

        weighted_abs_diff_sum += (
            mean_abs_diff
            * value_count
        )

    weighted_mean_abs_diff = (
        weighted_abs_diff_sum
        / total_values
    )

    print("=" * 72)
    print("AGGREGATE")
    print("=" * 72)

    print(
        "Compared values          :",
        total_values,
    )

    print(
        "Exact mismatches         :",
        total_mismatches,
    )

    print(
        "Exact matches            :",
        total_values - total_mismatches,
    )

    print(
        "Mismatch percentage      :",
        (
            100.0
            * total_mismatches
            / total_values
        ),
    )

    print(
        "Global max absolute diff :",
        global_max_abs_diff,
    )

    print(
        "Weighted mean abs diff   :",
        weighted_mean_abs_diff,
    )

    print()
    print(
        "OPT001 REAL-WEIGHT CORRECTNESS: PASS"
    )


if __name__ == "__main__":
    main()
