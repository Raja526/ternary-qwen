"""
OPT002 compiled wrapper validation.

Compare:
    frozen ENG011 StreamingDirectTernaryLinear
vs
    OPT002 CompiledPackedTernaryLinear

using the real ENG003 k_proj packed weight.

This validates the nn.Module integration layer only.
"""

import torch

from packed_ternary.checkpoint import (
    load_packed_checkpoint,
)

from packed_ternary.streaming_direct_linear import (
    StreamingDirectTernaryLinear,
)

from packed_ternary.compiled_linear import (
    CompiledPackedTernaryLinear,
)


CHECKPOINT_PATH = (
    "results/ENG003_multimatrix_checkpoint.pt"
)

TARGET = (
    "model.layers.3.self_attn.k_proj"
)

SEED = 20260925
ROWS_PER_CHUNK = 64

ATOL = 2e-2
RTOL = 2e-2


def main():

    torch.manual_seed(SEED)

    print("=" * 72)
    print("OPT002 COMPILED WRAPPER VALIDATION")
    print("=" * 72)

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

    in_features = int(
        packed.original_shape[1]
    )

    reference = (
        StreamingDirectTernaryLinear(
            packed_weight=packed,
            rows_per_chunk=ROWS_PER_CHUNK,
        )
    )

    compiled = (
        CompiledPackedTernaryLinear(
            packed_weight=packed,
        )
    )

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

    total_values = 0
    total_mismatches = 0
    abs_diff_sum = 0.0
    global_max_diff = 0.0

    for index, x in enumerate(
        input_sets,
        start=1,
    ):

        with torch.inference_mode():

            reference_output = (
                reference(x)
            )

            compiled_output = (
                compiled(x)
            )

        if (
            reference_output.shape
            != compiled_output.shape
        ):
            raise AssertionError(
                "Output shape mismatch"
            )

        if (
            reference_output.dtype
            != compiled_output.dtype
        ):
            raise AssertionError(
                "Output dtype mismatch"
            )

        diff = (
            reference_output.float()
            - compiled_output.float()
        ).abs()

        values = int(
            reference_output.numel()
        )

        mismatches = int(
            (
                reference_output
                != compiled_output
            )
            .sum()
            .item()
        )

        max_diff = float(
            diff.max().item()
        )

        mean_diff = float(
            diff.mean().item()
        )

        close = torch.allclose(
            reference_output,
            compiled_output,
            atol=ATOL,
            rtol=RTOL,
        )

        print()
        print(
            f"INPUT SET {index}"
        )
        print(
            "Shape             :",
            tuple(x.shape),
        )
        print(
            "Compared values   :",
            values,
        )
        print(
            "Exact mismatches  :",
            mismatches,
        )
        print(
            "Max abs diff      :",
            max_diff,
        )
        print(
            "Mean abs diff     :",
            mean_diff,
        )
        print(
            "Numerically close :",
            close,
        )

        if not close:
            raise RuntimeError(
                f"OPT002 wrapper validation "
                f"failed on input set {index}"
            )

        total_values += values
        total_mismatches += mismatches

        abs_diff_sum += (
            mean_diff
            * values
        )

        global_max_diff = max(
            global_max_diff,
            max_diff,
        )

    print()
    print("=" * 72)
    print("AGGREGATE")
    print("=" * 72)

    print(
        "Compared values        :",
        total_values,
    )

    print(
        "Exact mismatches       :",
        total_mismatches,
    )

    print(
        "Mismatch percentage    :",
        (
            100.0
            * total_mismatches
            / total_values
        ),
    )

    print(
        "Global max abs diff    :",
        global_max_diff,
    )

    print(
        "Weighted mean abs diff :",
        (
            abs_diff_sum
            / total_values
        ),
    )

    print(
        "Persistent compression :",
        f"{compiled.persistent_compression_factor:.6f}x",
    )

    print()
    print(
        "OPT002 COMPILED WRAPPER VALIDATION: PASS"
    )


if __name__ == "__main__":
    main()
