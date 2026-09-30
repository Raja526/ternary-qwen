import os
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
from packed_ternary.gpu003_extension import (
    load_gpu003_extension,
)


CHECKPOINT_PATH = (
    "results/ENG003_multimatrix_checkpoint.pt"
)

TARGET = (
    "model.layers.3.self_attn.k_proj"
)

SEED = 20260930


def main():
    torch.manual_seed(SEED)

    print("===== GPU003 REAL-WEIGHT VALIDATION =====")
    print("PyTorch:", torch.__version__)
    print("PyTorch CUDA:", torch.version.cuda)
    print("CUDA_HOME:", os.environ.get("CUDA_HOME"))
    print(
        "TORCH_CUDA_ARCH_LIST:",
        os.environ.get("TORCH_CUDA_ARCH_LIST"),
    )
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "Capability:",
        torch.cuda.get_device_capability(0),
    )
    print()

    metadata, weights = load_packed_checkpoint(
        CHECKPOINT_PATH
    )

    if TARGET not in weights:
        raise KeyError(
            f"Target not found: {TARGET}"
        )

    packed = weights[TARGET]

    out_features = int(
        packed.original_shape[0]
    )
    in_features = int(
        packed.original_shape[1]
    )
    group_size = int(
        packed.group_size
    )

    print("Checkpoint       :", CHECKPOINT_PATH)
    print("Target           :", TARGET)
    print("Shape            :", packed.original_shape)
    print("Parameters       :", packed.original_numel)
    print("Group size       :", group_size)
    print(
        "Packed storage   :",
        packed.packed_storage_bytes,
        "bytes",
    )
    print(
        "FP16 equivalent  :",
        packed.original_fp16_bytes,
        "bytes",
    )
    print(
        "Compression      :",
        f"{packed.compression_factor:.6f}x",
    )
    print()

    cpu_ext = load_opt006_extension(
        verbose=False
    )
    gpu_ext = load_gpu003_extension(
        verbose=False
    )

    # Multiple flattened input shapes.
    #
    # The CUDA operator itself accepts [N, in_features].
    # These N values represent different batch/token counts.
    input_sets = [
        torch.randn(
            1,
            in_features,
            dtype=torch.float16,
        ),
        torch.randn(
            4,
            in_features,
            dtype=torch.float16,
        ),
        torch.randn(
            14,
            in_features,
            dtype=torch.float16,
        ),
    ]

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

    packed_codes_gpu = (
        packed_codes_cpu.cuda()
    )
    scales_gpu = (
        scales_cpu.cuda()
    )

    total_outputs = 0
    total_mismatches = 0
    global_max_abs_diff = 0.0
    weighted_abs_diff_sum = 0.0

    for index, input_cpu in enumerate(
        input_sets,
        start=1,
    ):
        print(f"INPUT SET {index}")
        print(
            "  Shape                  :",
            tuple(input_cpu.shape),
        )

        reference = (
            cpu_ext
            .packed_ternary_linear_cpu_parallel(
                input_cpu,
                packed_codes_cpu,
                scales_cpu,
                out_features,
                group_size,
            )
        )

        candidate_gpu = (
            gpu_ext
            .packed_ternary_linear(
                input_cpu.cuda(),
                packed_codes_gpu,
                scales_gpu,
                out_features,
                group_size,
            )
        )

        torch.cuda.synchronize()

        candidate = candidate_gpu.cpu()

        diff = (
            reference.float()
            - candidate.float()
        ).abs()

        mismatches = int(
            torch.count_nonzero(
                reference != candidate
            ).item()
        )

        max_abs_diff = float(
            diff.max().item()
        )

        mean_abs_diff = float(
            diff.mean().item()
        )

        outputs = reference.numel()

        print(
            "  Outputs                :",
            outputs,
        )
        print(
            "  Exact mismatches       :",
            mismatches,
        )
        print(
            "  Max absolute difference:",
            f"{max_abs_diff:.10f}",
        )
        print(
            "  Mean absolute difference:",
            f"{mean_abs_diff:.10f}",
        )
        print()

        total_outputs += outputs
        total_mismatches += mismatches

        global_max_abs_diff = max(
            global_max_abs_diff,
            max_abs_diff,
        )

        weighted_abs_diff_sum += (
            mean_abs_diff * outputs
        )

    global_mean_abs_diff = (
        weighted_abs_diff_sum
        / total_outputs
    )

    print("===== SUMMARY =====")
    print(
        "Input sets:",
        len(input_sets),
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
    print(
        "Global mean abs diff:",
        f"{global_mean_abs_diff:.10f}",
    )

    if total_mismatches == 0:
        print("Bit-exact vs OPT006: YES")
        print(
            "GPU003 REAL-WEIGHT CORRECTNESS: PASS"
        )
    else:
        print("Bit-exact vs OPT006: NO")
        print(
            "GPU003 REAL-WEIGHT CORRECTNESS: "
            "NEEDS ANALYSIS"
        )


if __name__ == "__main__":
    main()
