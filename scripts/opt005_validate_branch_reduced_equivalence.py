import torch

from packed_ternary.opt004_extension import (
    load_opt004_extension,
)
from packed_ternary.opt005_extension import (
    load_opt005_extension,
)


SEED = 20260925
GROUP_SIZE = 128
THREADS = 6

TESTS = [
    (1, 128, 4),
    (3, 128, 17),
    (4, 1024, 512),
    (3, 2048, 1024),
    (2, 1024, 6144),
]


def make_valid_packed_codes(num_codes):
    assert num_codes % 4 == 0

    # Valid frozen encoding:
    # -1 -> 00
    #  0 -> 01
    # +1 -> 10
    codes = torch.randint(
        0,
        3,
        (num_codes,),
        dtype=torch.uint8,
    )

    codes = codes.reshape(-1, 4)

    packed = (
        codes[:, 0]
        | (codes[:, 1] << 2)
        | (codes[:, 2] << 4)
        | (codes[:, 3] << 6)
    )

    return packed.contiguous()


def main():
    torch.manual_seed(SEED)
    torch.set_num_threads(THREADS)
    torch.set_num_interop_threads(THREADS)

    opt004 = load_opt004_extension(
        verbose=False,
    )

    opt005 = load_opt005_extension(
        verbose=False,
    )

    total_outputs = 0
    total_mismatches = 0
    global_max_diff = 0.0

    print(
        "OPT005 — BRANCH-REDUCED DECODE EQUIVALENCE"
    )
    print()
    print(f"Seed: {SEED}")
    print(f"Threads: {THREADS}")
    print(f"Group size: {GROUP_SIZE}")
    print()

    for test_id, (
        batch_size,
        in_features,
        out_features,
    ) in enumerate(TESTS, start=1):

        assert in_features % GROUP_SIZE == 0

        groups_per_row = (
            in_features // GROUP_SIZE
        )

        input_tensor = torch.randn(
            batch_size,
            in_features,
            dtype=torch.float16,
        )

        packed_codes = make_valid_packed_codes(
            out_features * in_features
        )

        scales = (
            torch.rand(
                out_features,
                groups_per_row,
                dtype=torch.float16,
            )
            + 0.01
        ).contiguous()

        out004 = (
            opt004
            .packed_ternary_linear_cpu_parallel(
                input_tensor,
                packed_codes,
                scales,
                out_features,
                GROUP_SIZE,
            )
        )

        out005 = (
            opt005
            .packed_ternary_linear_cpu_parallel(
                input_tensor,
                packed_codes,
                scales,
                out_features,
                GROUP_SIZE,
            )
        )

        exact = torch.equal(
            out004,
            out005,
        )

        diff = (
            out004.float()
            - out005.float()
        ).abs()

        mismatches = (
            out004 != out005
        ).sum().item()

        max_diff = diff.max().item()

        total_outputs += out004.numel()
        total_mismatches += mismatches
        global_max_diff = max(
            global_max_diff,
            max_diff,
        )

        print(
            f"Test {test_id}: "
            f"B={batch_size}, "
            f"K={in_features}, "
            f"N={out_features}"
        )
        print(
            f"  outputs: {out004.numel()}"
        )
        print(
            f"  exact: {exact}"
        )
        print(
            f"  mismatches: {mismatches}"
        )
        print(
            f"  max_abs_diff: {max_diff}"
        )

        if not exact:
            raise RuntimeError(
                f"Test {test_id} is not bit-exact."
            )

    print()
    print("SUMMARY")
    print(f"Tests: {len(TESTS)}")
    print(
        f"Compared outputs: {total_outputs}"
    )
    print(
        f"Exact mismatches: "
        f"{total_mismatches}"
    )
    print(
        f"Global max abs diff: "
        f"{global_max_diff}"
    )

    assert total_mismatches == 0
    assert global_max_diff == 0.0

    print()
    print(
        "OPT004 vs OPT005 "
        "BIT-EXACT EQUIVALENCE: PASS"
    )


if __name__ == "__main__":
    main()
