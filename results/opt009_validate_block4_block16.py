import torch

from packed_ternary.opt008_extension import (
    load_opt008_extension,
)

from packed_ternary.opt009_block4_extension import (
    load_opt009_extension as load_block4_extension,
)

from packed_ternary.opt009_block16_extension import (
    load_opt009_extension as load_block16_extension,
)


GROUP_SIZE = 128
SEED = 20260929


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
    block4,
    block16,
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

    out4 = (
        block4
        .packed_ternary_linear_cpu_parallel(
            x,
            packed,
            scales,
            out_features,
            GROUP_SIZE,
        )
    )

    out16 = (
        block16
        .packed_ternary_linear_cpu_parallel(
            x,
            packed,
            scales,
            out_features,
            GROUP_SIZE,
        )
    )

    assert ref.shape == out4.shape
    assert ref.shape == out16.shape

    assert ref.dtype == out4.dtype
    assert ref.dtype == out16.dtype

    mismatch4 = int(
        (ref != out4)
        .sum()
        .item()
    )

    mismatch16 = int(
        (ref != out16)
        .sum()
        .item()
    )

    max_abs4 = float(
        (
            out4.float()
            - ref.float()
        )
        .abs()
        .max()
        .item()
    )

    max_abs16 = float(
        (
            out16.float()
            - ref.float()
        )
        .abs()
        .max()
        .item()
    )

    return (
        int(ref.numel()),
        mismatch4,
        max_abs4,
        mismatch16,
        max_abs16,
    )


def main():
    torch.set_num_threads(6)

    print("=" * 72)
    print(
        "OPT009 — BLOCK 4 / BLOCK 16 "
        "OPERATOR EQUIVALENCE"
    )
    print("=" * 72)

    print()
    print("Loading frozen OPT008 extension...")

    reference = load_opt008_extension(
        verbose=False
    )

    print("Loading OPT009 block-4 extension...")

    block4 = load_block4_extension(
        verbose=False
    )

    print("Loading OPT009 block-16 extension...")

    block16 = load_block16_extension(
        verbose=False
    )

    print("PASS: all extensions loaded.")

    # Boundaries around block sizes 4 and 16,
    # plus realistic selected-matrix dimensions.
    cases = [
        (1, 128, 1),

        (3, 128, 3),
        (3, 128, 4),
        (3, 128, 5),

        (3, 128, 7),
        (3, 128, 8),
        (3, 128, 9),

        (3, 128, 15),
        (3, 128, 16),
        (3, 128, 17),

        (5, 256, 31),
        (5, 256, 32),
        (5, 256, 33),

        (22, 1024, 64),
        (22, 1024, 512),
        (22, 2048, 1024),
    ]

    total_outputs = 0

    total_mismatch4 = 0
    total_mismatch16 = 0

    global_max4 = 0.0
    global_max16 = 0.0

    print()
    print(
        f"{'shape':>22} | "
        f"{'outputs':>8} | "
        f"{'B4 mism':>8} | "
        f"{'B16 mism':>9}"
    )

    print("-" * 72)

    for (
        batch,
        in_features,
        out_features,
    ) in cases:

        (
            outputs,
            mismatch4,
            max_abs4,
            mismatch16,
            max_abs16,
        ) = run_case(
            reference,
            block4,
            block16,
            batch,
            in_features,
            out_features,
        )

        total_outputs += outputs

        total_mismatch4 += mismatch4
        total_mismatch16 += mismatch16

        global_max4 = max(
            global_max4,
            max_abs4,
        )

        global_max16 = max(
            global_max16,
            max_abs16,
        )

        shape = (
            f"({batch},"
            f"{in_features},"
            f"{out_features})"
        )

        print(
            f"{shape:>22} | "
            f"{outputs:8d} | "
            f"{mismatch4:8d} | "
            f"{mismatch16:9d}"
        )

    print()
    print("SUMMARY")
    print("-" * 72)

    print(
        f"Compared outputs        : "
        f"{total_outputs}"
    )

    print(
        f"Block-4 mismatches      : "
        f"{total_mismatch4}"
    )

    print(
        f"Block-4 max abs diff    : "
        f"{global_max4}"
    )

    print(
        f"Block-16 mismatches     : "
        f"{total_mismatch16}"
    )

    print(
        f"Block-16 max abs diff   : "
        f"{global_max16}"
    )

    if total_mismatch4 != 0:
        raise SystemExit(
            "FAIL: block-4 is not bit-exact "
            "with frozen OPT008."
        )

    if total_mismatch16 != 0:
        raise SystemExit(
            "FAIL: block-16 is not bit-exact "
            "with frozen OPT008."
        )

    print()
    print(
        "PASS: block-4 and block-16 are "
        "bit-exact with frozen OPT008."
    )


if __name__ == "__main__":
    main()
