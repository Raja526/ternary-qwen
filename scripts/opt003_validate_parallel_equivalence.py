"""
OPT003 — Parallel operator equivalence validation.

Reference:
    OPT001 scalar compiled CPU operator.

Candidate:
    OPT003 parallel compiled CPU operator.

Controlled difference:
    sequential (batch, output-row) execution
        ->
    parallel (batch, output-row) execution.

Expected:
    bit-exact FP16 output.
"""

import torch

from packed_ternary.packing import (
    pack_ternary_codes,
)

from packed_ternary.opt001_extension import (
    load_opt001_extension,
)

from packed_ternary.opt003_extension import (
    load_opt003_extension,
)


SEED = 20260925
GROUP_SIZE = 128


TESTS = (
    # batch, in_features, out_features
    (1, 128, 4),
    (3, 128, 17),
    (4, 1024, 512),
    (3, 2048, 1024),
    (2, 1024, 6144),
)


def section(title):
    print()
    print("=" * 90)
    print(title)
    print("=" * 90)


def main():
    torch.manual_seed(SEED)

    torch.set_num_threads(6)

    section(
        "OPT003 — OPT001 VS PARALLEL OPERATOR EQUIVALENCE"
    )

    print(
        "PyTorch CPU threads      :",
        torch.get_num_threads(),
    )

    scalar_ext = load_opt001_extension(
        verbose=False
    )

    parallel_ext = load_opt003_extension(
        verbose=False
    )

    total_outputs = 0
    total_mismatches = 0
    global_max_abs = 0.0

    for test_index, (
        batch,
        in_features,
        out_features,
    ) in enumerate(
        TESTS,
        start=1,
    ):
        num_codes = (
            out_features
            * in_features
        )

        codes = torch.randint(
            low=-1,
            high=2,
            size=(num_codes,),
            dtype=torch.int8,
        )

        packed_codes, original_numel = (
            pack_ternary_codes(
                codes
            )
        )

        assert (
            original_numel
            == num_codes
        )

        groups_per_row = (
            in_features
            // GROUP_SIZE
        )

        scales = (
            torch.rand(
                out_features
                * groups_per_row,
                dtype=torch.float32,
            )
            .mul_(0.2)
            .add_(0.001)
            .to(torch.float16)
        )

        x = torch.randn(
            batch,
            in_features,
            dtype=torch.float16,
        )

        reference = (
            scalar_ext
            .packed_ternary_linear_cpu(
                x,
                packed_codes,
                scales,
                out_features,
                GROUP_SIZE,
            )
        )

        candidate = (
            parallel_ext
            .packed_ternary_linear_cpu_parallel(
                x,
                packed_codes,
                scales,
                out_features,
                GROUP_SIZE,
            )
        )

        assert (
            reference.shape
            == candidate.shape
        )

        assert (
            reference.dtype
            == candidate.dtype
        )

        compared = int(
            reference.numel()
        )

        mismatches = int(
            (
                reference
                != candidate
            )
            .sum()
            .item()
        )

        max_abs = float(
            (
                reference.float()
                - candidate.float()
            )
            .abs()
            .max()
            .item()
        )

        bit_exact = torch.equal(
            reference,
            candidate,
        )

        print()
        print(
            f"Test {test_index}"
        )
        print(
            "Input shape             :",
            tuple(x.shape),
        )
        print(
            "Matrix shape            :",
            (
                out_features,
                in_features,
            ),
        )
        print(
            "Compared outputs        :",
            compared,
        )
        print(
            "Exact mismatches        :",
            mismatches,
        )
        print(
            "Max abs difference      :",
            max_abs,
        )
        print(
            "Bit-exact               :",
            bit_exact,
        )

        assert (
            mismatches == 0
        )

        assert (
            max_abs == 0.0
        )

        assert bit_exact

        total_outputs += (
            compared
        )

        total_mismatches += (
            mismatches
        )

        global_max_abs = max(
            global_max_abs,
            max_abs,
        )

    section(
        "OPT003 PARALLEL EQUIVALENCE SUMMARY"
    )

    print(
        "Tests                   :",
        len(TESTS),
    )
    print(
        "Compared outputs        :",
        total_outputs,
    )
    print(
        "Exact mismatches        :",
        total_mismatches,
    )
    print(
        "Global max abs diff     :",
        global_max_abs,
    )
    print(
        "Arithmetic              :",
        "unchanged",
    )
    print(
        "Controlled difference   :",
        "sequential jobs -> parallel jobs",
    )

    print()
    print(
        "OPT003 PARALLEL EQUIVALENCE: PASS"
    )


if __name__ == "__main__":
    main()
