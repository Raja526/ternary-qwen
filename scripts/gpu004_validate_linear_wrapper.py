import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from packed_ternary.checkpoint import (
    load_packed_checkpoint,
)

from packed_ternary.opt006_extension import (
    load_opt006_extension,
)

from packed_ternary.gpu004_linear import (
    CUDAPackedTernaryLinear,
)


CHECKPOINT_PATH = (
    "results/ENG003_multimatrix_checkpoint.pt"
)

TARGET = (
    "model.layers.3.self_attn.k_proj"
)

SEED = 20261012


def main():
    torch.manual_seed(SEED)

    print(
        "===== GPU004 LINEAR WRAPPER VALIDATION ====="
    )

    metadata, weights = (
        load_packed_checkpoint(
            CHECKPOINT_PATH
        )
    )

    packed = weights[TARGET]

    in_features = int(
        packed.original_shape[1]
    )

    out_features = int(
        packed.original_shape[0]
    )

    print("Target:", TARGET)
    print("Shape:", packed.original_shape)
    print("Group size:", packed.group_size)

    # Transformer-like input verifies flatten + reshape behavior.
    input_cpu = torch.randn(
        2,
        7,
        in_features,
        dtype=torch.float16,
    )

    cpu_ext = load_opt006_extension(
        verbose=False
    )

    flat_cpu = (
        input_cpu
        .reshape(-1, in_features)
        .contiguous()
    )

    reference = (
        cpu_ext
        .packed_ternary_linear_cpu_parallel(
            flat_cpu,
            packed.packed_codes.contiguous(),
            packed.scales.reshape(-1).contiguous(),
            out_features,
            packed.group_size,
        )
        .reshape(
            2,
            7,
            out_features,
        )
    )

    layer = (
        CUDAPackedTernaryLinear(
            packed_weight=packed,
        )
        .cuda()
        .eval()
    )

    input_gpu = input_cpu.cuda()

    with torch.inference_mode():
        candidate = (
            layer(input_gpu)
            .cpu()
        )

    torch.cuda.synchronize()

    print(
        "Candidate shape:",
        tuple(candidate.shape),
    )

    print(
        "Candidate dtype:",
        candidate.dtype,
    )

    mismatches = int(
        (
            reference
            != candidate
        )
        .sum()
        .item()
    )

    diff = (
        candidate.float()
        - reference.float()
    ).abs()

    max_abs = float(
        diff.max().item()
    )

    mean_abs = float(
        diff.double().mean().item()
    )

    compared = int(
        reference.numel()
    )

    print("Compared outputs:", compared)
    print("Exact mismatches:", mismatches)
    print(
        "Max abs difference:",
        f"{max_abs:.10f}",
    )
    print(
        "Mean abs difference:",
        f"{mean_abs:.10f}",
    )

    if (
        mismatches == 0
        and torch.equal(
            reference,
            candidate,
        )
    ):
        print("Bit-exact vs OPT006: YES")
        print(
            "GPU004 LINEAR WRAPPER: PASS"
        )
    else:
        print("Bit-exact vs OPT006: NO")
        print(
            "GPU004 LINEAR WRAPPER: NEEDS ANALYSIS"
        )


if __name__ == "__main__":
    main()
