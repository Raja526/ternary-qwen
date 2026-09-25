import torch

from packed_ternary.opt001_extension import (
    load_opt001_extension,
)
from packed_ternary.packing import (
    pack_ternary_codes,
)


torch.manual_seed(1234)

GROUP_SIZE = 128
IN_FEATURES = 128
OUT_FEATURES = 4
BATCH_SIZE = 3


def main():

    print("=" * 72)
    print("OPT001 SYNTHETIC CORRECTNESS VALIDATION")
    print("=" * 72)

    # ---------------------------------------------------------
    # 1. Deterministic FP16 input
    # ---------------------------------------------------------

    x = torch.randn(
        BATCH_SIZE,
        IN_FEATURES,
        dtype=torch.float16,
    )

    # ---------------------------------------------------------
    # 2. Deterministic ternary codes
    # ---------------------------------------------------------

    codes = torch.randint(
        low=-1,
        high=2,
        size=(
            OUT_FEATURES,
            IN_FEATURES,
        ),
        dtype=torch.int8,
    )

    # ---------------------------------------------------------
    # 3. Deterministic FP16 scales
    # ---------------------------------------------------------

    scales = torch.rand(
        OUT_FEATURES,
        IN_FEATURES // GROUP_SIZE,
        dtype=torch.float16,
    )

    # ---------------------------------------------------------
    # 4. Pack codes using frozen ENG codec
    # ---------------------------------------------------------

    packed_codes, original_numel = (
        pack_ternary_codes(
            codes.reshape(-1)
        )
    )

    assert (
        original_numel
        == OUT_FEATURES * IN_FEATURES
    )

    # ---------------------------------------------------------
    # 5. ENG011-style PyTorch reference
    # ---------------------------------------------------------

    grouped_input = x.reshape(
        BATCH_SIZE,
        IN_FEATURES // GROUP_SIZE,
        GROUP_SIZE,
    )

    grouped_codes = codes.reshape(
        OUT_FEATURES,
        IN_FEATURES // GROUP_SIZE,
        GROUP_SIZE,
    )

    positive_mask = (
        grouped_codes == 1
    )

    negative_mask = (
        grouped_codes == -1
    )

    positive_sum = (
        grouped_input[
            :,
            None,
            :,
            :,
        ]
        .masked_fill(
            ~positive_mask[
                None,
                :,
                :,
                :,
            ],
            0,
        )
        .sum(dim=-1)
    )

    negative_sum = (
        grouped_input[
            :,
            None,
            :,
            :,
        ]
        .masked_fill(
            ~negative_mask[
                None,
                :,
                :,
                :,
            ],
            0,
        )
        .sum(dim=-1)
    )

    reference = (
        (
            positive_sum
            - negative_sum
        )
        * scales[None, :, :]
    ).sum(dim=-1)

    # ---------------------------------------------------------
    # 6. OPT001 compiled operator
    # ---------------------------------------------------------

    ext = load_opt001_extension(
        verbose=False
    )

    opt001 = (
        ext.packed_ternary_linear_cpu(
            x,
            packed_codes,
            scales.reshape(-1),
            OUT_FEATURES,
            GROUP_SIZE,
        )
    )

    # ---------------------------------------------------------
    # 7. Compare
    # ---------------------------------------------------------

    difference = (
        opt001.float()
        - reference.float()
    ).abs()

    mismatch_count = int(
        (
            opt001 != reference
        )
        .sum()
        .item()
    )

    total_values = (
        reference.numel()
    )

    max_abs_diff = float(
        difference.max().item()
    )

    mean_abs_diff = float(
        difference.mean().item()
    )

    print()
    print("Configuration")
    print("-" * 72)
    print(
        "Input shape:",
        tuple(x.shape),
    )
    print(
        "Output features:",
        OUT_FEATURES,
    )
    print(
        "Group size:",
        GROUP_SIZE,
    )
    print(
        "Packed bytes:",
        packed_codes.numel(),
    )

    print()
    print("Structural checks")
    print("-" * 72)
    print(
        "Original ternary values:",
        codes.numel(),
    )
    print(
        "Expected packed bytes:",
        codes.numel() // 4,
    )
    print(
        "Actual packed bytes:",
        packed_codes.numel(),
    )

    print()
    print("Numerical comparison")
    print("-" * 72)
    print(
        "Reference dtype:",
        reference.dtype,
    )
    print(
        "OPT001 dtype:",
        opt001.dtype,
    )
    print(
        "Compared values:",
        total_values,
    )
    print(
        "Exact mismatches:",
        mismatch_count,
    )
    print(
        "Exact matches:",
        total_values - mismatch_count,
    )
    print(
        "Max absolute difference:",
        max_abs_diff,
    )
    print(
        "Mean absolute difference:",
        mean_abs_diff,
    )

    # ---------------------------------------------------------
    # 8. Correctness acceptance
    # ---------------------------------------------------------

    close = torch.allclose(
        opt001,
        reference,
        atol=2e-2,
        rtol=2e-2,
    )

    print()
    print(
        "Numerically close:",
        close,
    )

    if not close:
        raise RuntimeError(
            "OPT001 synthetic correctness validation FAILED"
        )

    print()
    print(
        "OPT001 SYNTHETIC CORRECTNESS: PASS"
    )


if __name__ == "__main__":
    main()
