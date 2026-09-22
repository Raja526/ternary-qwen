import gc

import torch
from transformers import Qwen3_5ForCausalLM

from ternary.quantizer import ternarize_modules


# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_ID = "Qwen/Qwen3.5-0.8B"

DEVICE = "cuda"
DTYPE = torch.float16

MIN_WEIGHT_PARAMS = 100_000


# Exact MIXED_MEDIUM configuration from Experiment 011
TARGET_BLOCKS = {
    "mlp": [6, 12],
    "linear_attn": [4, 8],
    "self_attn": [3, 7],
}


# Expected Experiment 011 optimized-scale statistics
EXPECTED_PARAMETERS = 57_671_680

EXPECTED_MEAN_ERROR = 0.472396
EXPECTED_MAX_ERROR = 0.502314
EXPECTED_SCALE_RATIO = 1.351844


# =============================================================================
# HELPERS
# =============================================================================

def get_block_index(name):

    parts = name.split(".")

    try:
        position = parts.index("layers")
        return int(parts[position + 1])

    except (ValueError, IndexError):
        return None


def get_component(name, module):

    if not isinstance(
        module,
        torch.nn.Linear,
    ):
        return None

    if (
        module.weight.numel()
        < MIN_WEIGHT_PARAMS
    ):
        return None

    lower_name = name.lower()

    if ".mlp." in lower_name:
        return "mlp"

    if ".linear_attn." in lower_name:
        return "linear_attn"

    if ".self_attn." in lower_name:
        return "self_attn"

    return None


def is_target_matrix(name, module):

    component = get_component(
        name,
        module,
    )

    if component is None:
        return False

    if component not in TARGET_BLOCKS:
        return False

    block = get_block_index(name)

    if block is None:
        return False

    return (
        block
        in TARGET_BLOCKS[component]
    )


# =============================================================================
# MODEL
# =============================================================================

def load_model():

    gc.collect()
    torch.cuda.empty_cache()

    print(
        "Loading fresh FP16 model..."
    )

    model = (
        Qwen3_5ForCausalLM
        .from_pretrained(
            MODEL_ID,
            torch_dtype=DTYPE,
            device_map=DEVICE,
        )
    )

    model.eval()

    return model


# =============================================================================
# DISCOVER EXACT EXP011 TARGETS
# =============================================================================

def discover_modules(model):

    modules = []

    for name, module in (
        model.named_modules()
    ):

        if not is_target_matrix(
            name,
            module,
        ):
            continue

        modules.append(
            (
                name,
                module,
                get_component(
                    name,
                    module,
                ),
                get_block_index(name),
            )
        )

    return modules


# =============================================================================
# COMPARISON
# =============================================================================

def compare(
    label,
    actual,
    expected,
    tolerance,
):

    difference = abs(
        actual - expected
    )

    passed = (
        difference <= tolerance
    )

    print(
        f"{label:<20} "
        f"actual={actual:.9f}  "
        f"expected={expected:.9f}  "
        f"diff={difference:.9f}  "
        f"{'PASS' if passed else 'FAIL'}"
    )

    return passed


# =============================================================================
# MAIN
# =============================================================================

def main():

    print(
        "\n"
        "========================================"
    )

    print(
        "FROZEN QUANTIZER REPRODUCIBILITY TEST"
    )

    print(
        "========================================"
    )

    print(
        "\nReference: Experiment 011"
    )

    print(
        "Configuration: MIXED_MEDIUM"
    )

    print(
        "Method: optimized"
    )

    model = load_model()

    modules = discover_modules(
        model
    )

    print(
        f"\nDiscovered matrices : "
        f"{len(modules)}"
    )

    print(
        "\nTarget matrices:"
    )

    for (
        name,
        module,
        component,
        block,
    ) in modules:

        print(
            f"  {component:<12} "
            f"block={block:<2} "
            f"params="
            f"{module.weight.numel():>10,}  "
            f"{name}"
        )

    print(
        "\nApplying shared frozen quantizer..."
    )

    result = ternarize_modules(
        modules,
        method="optimized",
    )

    total_model_parameters = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    coverage = (
        100.0
        * result["parameters"]
        / total_model_parameters
    )

    print(
        "\n"
        "========================================"
    )

    print(
        "RESULT"
    )

    print(
        "========================================"
    )

    print(
        f"Matrices          : "
        f"{result['linear_layers']}"
    )

    print(
        f"Parameters        : "
        f"{result['parameters']:,}"
    )

    print(
        f"Coverage          : "
        f"{coverage:.4f}%"
    )

    print(
        f"Mean error        : "
        f"{result['mean_error']:.9f}"
    )

    print(
        f"Max error         : "
        f"{result['max_error']:.9f}"
    )

    print(
        f"Mean scale ratio  : "
        f"{result['mean_scale_ratio']:.9f}"
    )

    total_codes = (
        result["negative"]
        + result["zero"]
        + result["positive"]
    )

    print(
        f"-1 codes          : "
        f"{result['negative']:,} "
        f"({100 * result['negative'] / total_codes:.2f}%)"
    )

    print(
        f"0 codes           : "
        f"{result['zero']:,} "
        f"({100 * result['zero'] / total_codes:.2f}%)"
    )

    print(
        f"+1 codes          : "
        f"{result['positive']:,} "
        f"({100 * result['positive'] / total_codes:.2f}%)"
    )

    print(
        "\n"
        "========================================"
    )

    print(
        "REPRODUCIBILITY CHECK"
    )

    print(
        "========================================"
    )

    parameter_pass = (
        result["parameters"]
        == EXPECTED_PARAMETERS
    )

    print(
        f"Parameters           "
        f"actual={result['parameters']:,}  "
        f"expected={EXPECTED_PARAMETERS:,}  "
        f"{'PASS' if parameter_pass else 'FAIL'}"
    )

    mean_pass = compare(
        "Mean error",
        result["mean_error"],
        EXPECTED_MEAN_ERROR,
        tolerance=0.000001,
    )

    max_pass = compare(
        "Max error",
        result["max_error"],
        EXPECTED_MAX_ERROR,
        tolerance=0.000001,
    )

    scale_pass = compare(
        "Mean scale ratio",
        result["mean_scale_ratio"],
        EXPECTED_SCALE_RATIO,
        tolerance=0.000001,
    )

    all_pass = (
        parameter_pass
        and mean_pass
        and max_pass
        and scale_pass
    )

    print(
        "\n"
        "========================================"
    )

    if all_pass:

        print(
            "FINAL STATUS: PASS"
        )

        print(
            "Shared quantizer reproduces "
            "Experiment 011."
        )

    else:

        print(
            "FINAL STATUS: FAIL"
        )

        print(
            "Do NOT freeze the quantizer yet."
        )

    print(
        "========================================"
    )


if __name__ == "__main__":
    main()
