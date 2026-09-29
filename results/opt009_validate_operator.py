import torch

from packed_ternary.opt008_extension import (
    load_opt008_extension,
)

from packed_ternary.opt009_extension import (
    load_opt009_extension,
)


GROUP_SIZE = 128
SEED = 20260928


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
    candidate,
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

    cand = (
        candidate
        .packed_ternary_linear_cpu_parallel(
            x,
            packed,
            scales,
            out_features,
            GROUP_SIZE,
        )
    )

    assert ref.shape == cand.shape
    assert ref.dtype == cand.dtype

    mismatches = int(
        (ref != cand)
        .sum()
        .item()
    )

    max_abs = float(
        (
            cand.float()
            - ref.float()
        )
        .abs()
        .max()
        .item()
    )

    return (
        int(ref.numel()),
        mismatches,
        max_abs,
    )


def main():
    torch.set_num_threads(6)

    reference = (
        load_opt008_extension(
            verbose=False
        )
    )

    candidate = (
        load_opt009_extension(
            verbose=False
        )
    )

    cases = [
        (1, 128, 1),
        (3, 128, 7),

        # Boundary around block size 8.
        (3, 128, 8),
        (3, 128, 9),

        (5, 256, 13),

        # Multiple complete row blocks.
        (7, 512, 31),

        # Realistic shapes.
        (22, 1024, 64),
        (22, 1024, 512),
        (22, 2048, 1024),
    ]

    total_outputs = 0
    total_mismatches = 0
    global_max = 0.0

    print(
        "=" * 72
    )
    print(
        "OPT009 GATE 1 — OPERATOR EQUIVALENCE"
    )
    print(
        "=" * 72
    )

    for case in cases:
        (
            outputs,
            mismatches,
            max_abs,
        ) = run_case(
            reference,
            candidate,
            *case,
        )

        total_outputs += outputs
        total_mismatches += mismatches
        global_max = max(
            global_max,
            max_abs,
        )

        print()
        print(
            "shape:",
            case,
        )
        print(
            "outputs       :",
            outputs,
        )
        print(
            "mismatches    :",
            mismatches,
        )
        print(
            "max abs error :",
            max_abs,
        )

    print()
    print(
        "=" * 72
    )
    print(
        "AGGREGATE"
    )
    print(
        "=" * 72
    )

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
        global_max,
    )

    if total_mismatches != 0:
        raise RuntimeError(
            "FAIL: OPT009 is not bit-exact "
            "with frozen OPT008."
        )

    if global_max != 0.0:
        raise RuntimeError(
            "FAIL: non-zero operator error."
        )

    print()
    print(
        "PASS: OPT009 output-row blocking "
        "is bit-exact with frozen OPT008."
    )


if __name__ == "__main__":
    main()
