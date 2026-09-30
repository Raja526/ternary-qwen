import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from packed_ternary.gpu003_extension import (
    load_gpu003_extension,
)
from packed_ternary.opt006_extension import (
    load_opt006_extension,
)
from packed_ternary.packed_weight import (
    quantize_to_packed,
)


GROUP_SIZE = 128
SEED = 20260930


def run_case(
    cpu_ext,
    gpu_ext,
    batch_size,
    in_features,
    out_features,
):
    torch.manual_seed(SEED + batch_size + in_features + out_features)

    weight = torch.randn(
        out_features,
        in_features,
        dtype=torch.float16,
    )

    x = torch.randn(
        batch_size,
        in_features,
        dtype=torch.float16,
    )

    packed = quantize_to_packed(
        weight,
        group_size=GROUP_SIZE,
    )

    packed_codes_cpu = (
        packed.packed_codes
        .contiguous()
        .cpu()
    )

    scales_cpu = (
        packed.scales
        .reshape(-1)
        .contiguous()
        .cpu()
    )

    x_cpu = x.contiguous().cpu()

    reference = cpu_ext.packed_ternary_linear_cpu_parallel(
        x_cpu,
        packed_codes_cpu,
        scales_cpu,
        out_features,
        GROUP_SIZE,
    )

    x_gpu = x_cpu.cuda()

    packed_codes_gpu = packed_codes_cpu.cuda()

    scales_gpu = scales_cpu.cuda()

    candidate = gpu_ext.packed_ternary_linear(
        x_gpu,
        packed_codes_gpu,
        scales_gpu,
        out_features,
        GROUP_SIZE,
    )

    torch.cuda.synchronize()

    candidate_cpu = candidate.cpu()

    exact_mismatches = int(
        torch.count_nonzero(
            reference != candidate_cpu
        ).item()
    )

    abs_diff = (
        reference.float()
        - candidate_cpu.float()
    ).abs()

    max_abs_diff = float(
        abs_diff.max().item()
    )

    mean_abs_diff = float(
        abs_diff.mean().item()
    )

    total = reference.numel()

    print(
        f"case batch={batch_size} "
        f"in={in_features} "
        f"out={out_features}"
    )

    print(
        f"  outputs: {total}"
    )

    print(
        f"  exact mismatches: "
        f"{exact_mismatches}"
    )

    print(
        f"  max abs diff: "
        f"{max_abs_diff:.10f}"
    )

    print(
        f"  mean abs diff: "
        f"{mean_abs_diff:.10f}"
    )

    return {
        "outputs": total,
        "mismatches": exact_mismatches,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
    }


def main():
    print("===== GPU003 OPERATOR VALIDATION =====")
    print("PyTorch:", torch.__version__)
    print("PyTorch CUDA:", torch.version.cuda)
    print(
        "CUDA_HOME:",
        os.environ.get("CUDA_HOME"),
    )
    print(
        "TORCH_CUDA_ARCH_LIST:",
        os.environ.get(
            "TORCH_CUDA_ARCH_LIST"
        ),
    )

    assert torch.cuda.is_available()

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

    print(
        "Capability:",
        torch.cuda.get_device_capability(0),
    )

    print()

    print("Loading OPT006 CPU reference...")
    cpu_ext = load_opt006_extension(
        verbose=False
    )

    print("Loading GPU003 CUDA candidate...")
    gpu_ext = load_gpu003_extension(
        verbose=False
    )

    print()

    cases = [
        (1, 128, 8),
        (2, 256, 16),
        (3, 512, 32),
        (2, 1024, 64),
        (4, 2048, 32),
    ]

    total_outputs = 0
    total_mismatches = 0
    global_max_abs_diff = 0.0

    for case in cases:
        result = run_case(
            cpu_ext,
            gpu_ext,
            *case,
        )

        total_outputs += result["outputs"]

        total_mismatches += (
            result["mismatches"]
        )

        global_max_abs_diff = max(
            global_max_abs_diff,
            result["max_abs_diff"],
        )

        print()

    print("===== SUMMARY =====")
    print(
        "Cases:",
        len(cases),
    )
    print(
        "Total outputs:",
        total_outputs,
    )
    print(
        "Total exact mismatches:",
        total_mismatches,
    )
    print(
        "Global max abs diff:",
        f"{global_max_abs_diff:.10f}",
    )

    if total_mismatches == 0:
        print(
            "Bit-exact vs OPT006: YES"
        )
        print(
            "GPU003 OPERATOR CORRECTNESS: PASS"
        )
    else:
        print(
            "Bit-exact vs OPT006: NO"
        )
        print(
            "GPU003 OPERATOR CORRECTNESS: "
            "NEEDS ANALYSIS"
        )


if __name__ == "__main__":
    main()
