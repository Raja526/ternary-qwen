import torch

from packed_ternary.opt008_extension import (
    load_opt008_extension,
)
from packed_ternary.opt010_grain16_extension import (
    load_opt008_extension as load_grain16_extension,
)
from packed_ternary.opt010_grain64_extension import (
    load_opt008_extension as load_grain64_extension,
)
from packed_ternary.opt010_grain256_extension import (
    load_opt008_extension as load_grain256_extension,
)


GROUP_SIZE = 128
SEED = 20260929
CPU_THREADS = 6


CASES = [
    # Small scheduling boundaries.
    (1, 128, 1),
    (3, 128, 15),
    (3, 128, 16),
    (3, 128, 17),

    (3, 128, 63),
    (3, 128, 64),
    (3, 128, 65),

    (3, 128, 255),
    (3, 128, 256),
    (3, 128, 257),

    # Larger / realistic dimensions.
    (5, 256, 31),
    (7, 512, 64),
    (22, 1024, 64),
    (22, 1024, 512),
    (22, 2048, 1024),
]


def pack_codes(codes):
    flat = (
        codes
        .reshape(-1)
        .to(torch.uint8)
    )

    assert flat.numel() % 4 == 0

    groups = flat.reshape(-1, 4)

    packed = (
        groups[:, 0]
        | (groups[:, 1] << 2)
        | (groups[:, 2] << 4)
        | (groups[:, 3] << 6)
    )

    return (
        packed
        .contiguous()
        .reshape(-1)
    )


def run_case(
    reference,
    candidates,
    batch,
    in_features,
    out_features,
):
    assert in_features % GROUP_SIZE == 0

    groups_per_row = (
        in_features // GROUP_SIZE
    )

    x = torch.randn(
        batch,
        in_features,
        dtype=torch.float16,
    )

    # Encoding:
    # 0 = -1
    # 1 =  0
    # 2 = +1
    codes = torch.randint(
        low=0,
        high=3,
        size=(
            out_features,
            in_features,
        ),
        dtype=torch.uint8,
    )

    packed = pack_codes(
        codes
    )

    scales = torch.rand(
        out_features * groups_per_row,
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

    results = {}

    for name, extension in candidates.items():
        candidate = (
            extension
            .packed_ternary_linear_cpu_parallel(
                x,
                packed,
                scales,
                out_features,
                GROUP_SIZE,
            )
        )

        if candidate.shape != ref.shape:
            raise RuntimeError(
                f"{name}: shape mismatch "
                f"{candidate.shape} != {ref.shape}"
            )

        mismatch = int(
            torch.count_nonzero(
                candidate != ref
            ).item()
        )

        max_abs = float(
            (
                candidate.float()
                - ref.float()
            )
            .abs()
            .max()
            .item()
        )

        results[name] = (
            mismatch,
            max_abs,
            candidate.numel(),
        )

    return results


def main():
    torch.manual_seed(SEED)

    torch.set_num_threads(
        CPU_THREADS
    )

    torch.set_num_interop_threads(
        CPU_THREADS
    )

    print("=" * 72)
    print("OPT010 — THREAD-SCHEDULING OPERATOR EQUIVALENCE")
    print("=" * 72)
    print()
    print("Reference : OPT008 grain-size 1")
    print("Candidates: grain 16 / 64 / 256")
    print(f"CPU threads: {CPU_THREADS}")
    print(f"Group size : {GROUP_SIZE}")
    print()

    reference = (
        load_opt008_extension(
            verbose=False
        )
    )

    candidates = {
        "grain16": (
            load_grain16_extension(
                verbose=False
            )
        ),
        "grain64": (
            load_grain64_extension(
                verbose=False
            )
        ),
        "grain256": (
            load_grain256_extension(
                verbose=False
            )
        ),
    }

    totals = {
        name: {
            "outputs": 0,
            "mismatches": 0,
            "max_abs": 0.0,
        }
        for name in candidates
    }

    for (
        batch,
        in_features,
        out_features,
    ) in CASES:

        results = run_case(
            reference,
            candidates,
            batch,
            in_features,
            out_features,
        )

        print(
            f"shape=({batch}, "
            f"{in_features}, "
            f"{out_features})"
        )

        for name in candidates:
            mismatch, max_abs, count = (
                results[name]
            )

            totals[name][
                "outputs"
            ] += count

            totals[name][
                "mismatches"
            ] += mismatch

            totals[name][
                "max_abs"
            ] = max(
                totals[name]["max_abs"],
                max_abs,
            )

            print(
                f"  {name:8s} "
                f"outputs={count:6d} "
                f"mismatches={mismatch:4d} "
                f"max_abs={max_abs}"
            )

    print()
    print("=" * 72)
    print("AGGREGATE")
    print("=" * 72)

    all_pass = True

    for name in candidates:
        stats = totals[name]

        passed = (
            stats["mismatches"] == 0
            and stats["max_abs"] == 0.0
        )

        all_pass &= passed

        print(
            f"{name:8s}: "
            f"outputs={stats['outputs']}, "
            f"mismatches={stats['mismatches']}, "
            f"max_abs={stats['max_abs']}, "
            f"bit_exact={passed}"
        )

    if not all_pass:
        raise RuntimeError(
            "FAIL: one or more OPT010 grain-size "
            "variants differ from frozen OPT008."
        )

    print()
    print(
        "PASS: grain-16, grain-64, and grain-256 "
        "are bit-exact with frozen OPT008."
    )


if __name__ == "__main__":
    main()
