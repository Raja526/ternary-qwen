import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from transformers import (
    AutoModelForCausalLM,
)

from packed_ternary.gpu004_linear import (
    CUDAPackedTernaryLinear,
)

from packed_ternary.gpu004_runtime_loader import (
    load_gpu004_runtime_modules,
)


MODEL_NAME = "Qwen/Qwen3.5-0.8B"

CHECKPOINT_PATH = Path(
    "results/ENG003_multimatrix_checkpoint.pt"
)

TARGETS = (
    "model.layers.3.self_attn.k_proj",
    "model.layers.4.linear_attn.out_proj",
    "model.layers.4.linear_attn.in_proj_qkv",
)


def get_module(model, name):
    modules = dict(
        model.named_modules()
    )

    return modules[name]


def main():

    print(
        "===== GPU004 RUNTIME INTEGRATION VALIDATION ====="
    )

    print(
        "Model:",
        MODEL_NAME,
    )

    print(
        "Checkpoint:",
        CHECKPOINT_PATH,
    )

    print(
        "Selected matrices:",
        len(TARGETS),
    )

    # ----------------------------------------------------------
    # Load model on CPU.
    #
    # This gate validates replacement structure first.
    # No model forward pass is performed here.
    # ----------------------------------------------------------

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            MODEL_NAME,
            dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
    )

    model.eval()

    print()
    print("Before replacement:")

    for name in TARGETS:
        module = get_module(
            model,
            name,
        )

        print(
            name,
            type(module).__name__,
            module.weight.device,
            module.weight.dtype,
        )

        assert isinstance(
            module,
            torch.nn.Linear,
        )

    # ----------------------------------------------------------
    # Atomic GPU004 replacement.
    # ----------------------------------------------------------

    result = (
        load_gpu004_runtime_modules(
            model=model,
            checkpoint_path=CHECKPOINT_PATH,
            selected_module_names=TARGETS,
            expected_model_name=MODEL_NAME,
        )
    )

    print()
    print("Load result:")
    print(
        "Loaded modules:",
        result.loaded_module_names,
    )
    print(
        "Replaced matrices:",
        result.replaced_matrix_count,
    )
    print(
        "Replaced parameters:",
        result.replaced_parameter_count,
    )
    print(
        "Persistent packed bytes:",
        result.persistent_packed_bytes,
    )
    print(
        "FP16 equivalent bytes:",
        result.fp16_equivalent_bytes,
    )

    compression = (
        result.fp16_equivalent_bytes
        / result.persistent_packed_bytes
    )

    print(
        "Selected-weight compression:",
        f"{compression:.6f}x",
    )

    assert (
        result.loaded_module_names
        == TARGETS
    )

    assert (
        result.replaced_matrix_count
        == 3
    )

    assert (
        result.replaced_parameter_count
        == 8_912_896
    )

    # ----------------------------------------------------------
    # Validate installed modules.
    # ----------------------------------------------------------

    print()
    print("After replacement:")

    for name in TARGETS:
        module = get_module(
            model,
            name,
        )

        print(
            name,
            type(module).__name__,
            "codes=",
            tuple(module.packed_codes.shape),
            module.packed_codes.dtype,
            module.packed_codes.device,
            "scales=",
            tuple(module.scales.shape),
            module.scales.dtype,
            module.scales.device,
        )

        assert isinstance(
            module,
            CUDAPackedTernaryLinear,
        )

        assert (
            module.packed_codes.dtype
            == torch.uint8
        )

        assert (
            module.scales.dtype
            == torch.float16
        )

        # Model is intentionally still CPU-resident
        # during this structural gate.
        assert (
            module.packed_codes.device.type
            == "cpu"
        )

        assert (
            module.scales.device.type
            == "cpu"
        )

    # ----------------------------------------------------------
    # Now verify normal nn.Module device migration.
    #
    # Move only the three replacement modules to CUDA rather than
    # the entire model. No forward pass is executed.
    # ----------------------------------------------------------

    print()
    print("CUDA migration:")

    for name in TARGETS:
        module = get_module(
            model,
            name,
        )

        module.cuda()

        print(
            name,
            "codes=",
            module.packed_codes.device,
            "scales=",
            module.scales.device,
        )

        assert (
            module.packed_codes.device.type
            == "cuda"
        )

        assert (
            module.scales.device.type
            == "cuda"
        )

    torch.cuda.synchronize()

    print()
    print(
        "GPU004 RUNTIME INTEGRATION: PASS"
    )


if __name__ == "__main__":
    main()
