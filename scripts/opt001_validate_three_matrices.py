"""
OPT001 final three-matrix correctness validation.

Validates the compiled CPU packed-ternary operator against the
frozen ENG011 StreamingDirectTernaryLinear implementation using
all three weights from the frozen ENG003 packed checkpoint.

OPT001 controlled change:
    packed 2-bit codes are interpreted directly inside compiled C++
    rather than decoded into temporary int8 tensors and masks.

This is a correctness validation, not a performance benchmark.
"""

import torch

from packed_ternary.checkpoint import load_packed_checkpoint
from packed_ternary.streaming_direct_linear import (
    StreamingDirectTernaryLinear,
)
from packed_ternary.opt001_extension import (
    load_opt001_extension,
)


CHECKPOINT_PATH = (
    "results/ENG003_multimatrix_checkpoint.pt"
)

TARGETS = [
    "model.layers.3.self_attn.k_proj",
    "model.layers.4.linear_attn.out_proj",
    "model.layers.4.linear_attn.in_proj_qkv",
]

SEED = 20260925
ROWS_PER_CHUNK = 64

ATOL = 2e-2
RTOL = 2e-2


def main():

    torch.manual_seed(SEED)

    print("=" * 78)
    print("OPT001 FINAL THREE-MATRIX CORRECTNESS VALIDATION")
    print("=" * 78)

    metadata, weights = load_packed_checkpoint(
        CHECKPOINT_PATH
    )

    ext = load_opt001_extension(
        verbose=False
    )

    aggregate_values = 0
    aggregate_mismatches = 0
    aggregate_abs_diff_sum = 0.0
    aggregate_max_abs_diff = 0.0

    total_parameters = 0
    total_packed_bytes = 0
    total_fp16_bytes = 0

    for target_index, target in enumerate(
        TARGETS,
        start=1,
    ):

        if target not in weights:
            raise KeyError(
                f"Target not found: {target}"
            )

        packed = weights[target]

        out_features = int(
            packed.original_shape[0]
        )

        in_features = int(
            packed.original_shape[1]
        )

        print()
        print("-" * 78)
        print(
            f"MATRIX {target_index}: {target}"
        )
        print("-" * 78)

        print(
            "Shape                 :",
            packed.original_shape,
        )
        print(
            "Parameters            :",
            packed.original_numel,
        )
        print(
            "Group size            :",
            packed.group_size,
        )
        print(
            "Packed bytes          :",
            packed.packed_storage_bytes,
        )
        print(
            "FP16 equivalent bytes :",
            packed.original_fp16_bytes,
        )
        print(
            "Compression           :",
            f"{packed.compression_factor:.6f}x",
        )

        #
        # Small deterministic input.
        #
        # We intentionally keep this a correctness test,
        # not a benchmark.
        #
        input_tensor = torch.randn(
            3,
            in_features,
            dtype=torch.float16,
        )

        reference = (
            StreamingDirectTernaryLinear(
                packed_weight=packed,
                rows_per_chunk=ROWS_PER_CHUNK,
            )
        )

        with torch.inference_mode():

            reference_output = reference(
                input_tensor
            )

            opt001_output = (
                ext.packed_ternary_linear_cpu(
                    input_tensor.contiguous(),
                    packed.packed_codes,
                    packed.scales.reshape(-1),
                    out_features,
                    packed.group_size,
                )
            )

        if (
            reference_output.shape
            != opt001_output.shape
        ):
            raise AssertionError(
                f"Shape mismatch for {target}"
            )

        if (
            reference_output.dtype
            != opt001_output.dtype
        ):
            raise AssertionError(
                f"Dtype mismatch for {target}"
            )

        diff = (
            reference_output.float()
            - opt001_output.float()
        ).abs()

        values = int(
            reference_output.numel()
        )

        mismatches = int(
            (
                reference_output
                != opt001_output
            )
            .sum()
            .item()
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
            atol=ATOL,
            rtol=RTOL,
        )

        print()
        print(
            "Compared values        :",
            values,
        )
        print(
            "Exact mismatches       :",
            mismatches,
        )
        print(
            "Mismatch percentage    :",
            (
                100.0
                * mismatches
                / values
            ),
        )
        print(
            "Max absolute diff      :",
            max_abs_diff,
        )
        print(
            "Mean absolute diff     :",
            mean_abs_diff,
        )
        print(
            "Numerically close      :",
            close,
        )

        if not close:
            raise RuntimeError(
                f"OPT001 correctness FAILED: {target}"
            )

        aggregate_values += values
        aggregate_mismatches += mismatches

        aggregate_abs_diff_sum += (
            mean_abs_diff
            * values
        )

        aggregate_max_abs_diff = max(
            aggregate_max_abs_diff,
            max_abs_diff,
        )

        total_parameters += (
            packed.original_numel
        )

        total_packed_bytes += (
            packed.packed_storage_bytes
        )

        total_fp16_bytes += (
            packed.original_fp16_bytes
        )

    aggregate_mean_abs_diff = (
        aggregate_abs_diff_sum
        / aggregate_values
    )

    compression = (
        total_fp16_bytes
        / total_packed_bytes
    )

    print()
    print("=" * 78)
    print("AGGREGATE")
    print("=" * 78)

    print(
        "Selected matrices       :",
        len(TARGETS),
    )
    print(
        "Selected parameters     :",
        total_parameters,
    )
    print(
        "Packed bytes            :",
        total_packed_bytes,
    )
    print(
        "FP16 equivalent bytes   :",
        total_fp16_bytes,
    )
    print(
        "Persistent compression  :",
        f"{compression:.6f}x",
    )

    print()
    print(
        "Compared values         :",
        aggregate_values,
    )
    print(
        "Exact mismatches        :",
        aggregate_mismatches,
    )
    print(
        "Exact matches           :",
        (
            aggregate_values
            - aggregate_mismatches
        ),
    )
    print(
        "Mismatch percentage     :",
        (
            100.0
            * aggregate_mismatches
            / aggregate_values
        ),
    )
    print(
        "Global max abs diff     :",
        aggregate_max_abs_diff,
    )
    print(
        "Weighted mean abs diff  :",
        aggregate_mean_abs_diff,
    )

    print()
    print(
        "Decoded int8 tensors    : NOT CREATED BY OPT001"
    )
    print(
        "Positive/negative masks : NOT CREATED BY OPT001"
    )
    print(
        "Reconstructed FP16 W    : NOT CREATED BY OPT001"
    )

    print()
    print("=" * 78)
    print(
        "OPT001 FINAL THREE-MATRIX CORRECTNESS: PASS"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
