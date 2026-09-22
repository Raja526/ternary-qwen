import gc
import time

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, Qwen3_5ForCausalLM


# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_ID = "Qwen/Qwen3.5-0.8B"

DEVICE = "cuda"
DTYPE = torch.float16

GROUP_SIZE = 128
MIN_WEIGHT_PARAMS = 100_000

MAX_NEW_TOKENS = 40

PROMPTS = [
    "Explain in simple terms why the sky is blue.",
    "What is 17 multiplied by 23?",
    "If a train travels 60 km per hour for 2.5 hours, how far does it travel?",
]


# =============================================================================
# EXPERIMENT 008 DESIGN
# =============================================================================
#
# Experiment 007 showed:
#
# - Individual MLP blocks were generally tolerant.
# - Linear Attention block 0 was catastrophically sensitive.
# - Other sampled Linear Attention blocks were much more tolerant.
# - Individual Self Attention blocks were generally tolerant.
#
# Experiment 008 asks:
#
# Can individually tolerant blocks remain functional when combined?
#
# IMPORTANT:
# Each case starts from a fresh FP16 model.
#
# We first test WITHIN-COMPONENT combinations.
# Then we test a few CROSS-COMPONENT combinations.
#
# We deliberately avoid Linear Attention block 0.
# =============================================================================


TEST_CASES = [

    # -------------------------------------------------------------------------
    # MLP combinations
    # -------------------------------------------------------------------------

    {
        "name": "MLP_2",
        "blocks": {
            "mlp": [6, 12],
        },
    },

    {
        "name": "MLP_3",
        "blocks": {
            "mlp": [6, 12, 18],
        },
    },

    {
        "name": "MLP_4",
        "blocks": {
            "mlp": [6, 12, 18, 23],
        },
    },


    # -------------------------------------------------------------------------
    # Linear Attention combinations
    #
    # Block 0 intentionally excluded.
    # -------------------------------------------------------------------------

    {
        "name": "LINEAR_ATTN_2",
        "blocks": {
            "linear_attn": [4, 8],
        },
    },

    {
        "name": "LINEAR_ATTN_3",
        "blocks": {
            "linear_attn": [4, 8, 12],
        },
    },

    {
        "name": "LINEAR_ATTN_4",
        "blocks": {
            "linear_attn": [4, 8, 12, 16],
        },
    },

    {
        "name": "LINEAR_ATTN_5",
        "blocks": {
            "linear_attn": [4, 8, 12, 16, 22],
        },
    },


    # -------------------------------------------------------------------------
    # Self Attention combinations
    # -------------------------------------------------------------------------

    {
        "name": "SELF_ATTN_2",
        "blocks": {
            "self_attn": [3, 7],
        },
    },

    {
        "name": "SELF_ATTN_3",
        "blocks": {
            "self_attn": [3, 7, 11],
        },
    },

    {
        "name": "SELF_ATTN_4",
        "blocks": {
            "self_attn": [3, 7, 11, 15],
        },
    },


    # -------------------------------------------------------------------------
    # Cross-component combinations
    #
    # Start conservatively.
    # -------------------------------------------------------------------------

    {
        "name": "MIXED_SMALL",
        "blocks": {
            "mlp": [6],
            "linear_attn": [4],
            "self_attn": [3],
        },
    },

    {
        "name": "MIXED_MEDIUM",
        "blocks": {
            "mlp": [6, 12],
            "linear_attn": [4, 8],
            "self_attn": [3, 7],
        },
    },

    {
        "name": "MIXED_LARGE",
        "blocks": {
            "mlp": [6, 12, 18],
            "linear_attn": [4, 8, 12],
            "self_attn": [3, 7, 11],
        },
    },
]


# =============================================================================
# GPU MEMORY
# =============================================================================

def print_gpu_memory(label):

    allocated = (
        torch.cuda.memory_allocated()
        / (1024 ** 2)
    )

    reserved = (
        torch.cuda.memory_reserved()
        / (1024 ** 2)
    )

    free_bytes, total_bytes = (
        torch.cuda.mem_get_info()
    )

    free_mb = (
        free_bytes
        / (1024 ** 2)
    )

    total_mb = (
        total_bytes
        / (1024 ** 2)
    )

    print(
        f"\nGPU MEMORY — {label}"
    )

    print(
        f"Allocated : {allocated:.2f} MB"
    )

    print(
        f"Reserved  : {reserved:.2f} MB"
    )

    print(
        f"Free      : {free_mb:.2f} MB"
    )

    print(
        f"Total     : {total_mb:.2f} MB"
    )


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model():

    gc.collect()
    torch.cuda.empty_cache()

    print_gpu_memory(
        "before model load"
    )

    print(
        "\nLoading fresh FP16 model..."
    )

    start = time.perf_counter()

    model = (
        Qwen3_5ForCausalLM
        .from_pretrained(
            MODEL_ID,
            torch_dtype=DTYPE,
            device_map=DEVICE,
        )
    )

    model.eval()

    torch.cuda.synchronize()

    elapsed = (
        time.perf_counter()
        - start
    )

    print(
        f"Model loaded in "
        f"{elapsed:.2f} sec"
    )

    print_gpu_memory(
        "after model load"
    )

    return model


# =============================================================================
# MODEL CLEANUP
# =============================================================================

def release_model(model):

    print(
        "\nReleasing model..."
    )

    model.to("cpu")

    del model

    gc.collect()
    gc.collect()

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.synchronize()

    print_gpu_memory(
        "after model cleanup"
    )


# =============================================================================
# ARCHITECTURE HELPERS
# =============================================================================

def get_block_index(name):

    parts = name.split(".")

    try:

        position = (
            parts.index(
                "layers"
            )
        )

        return int(
            parts[
                position + 1
            ]
        )

    except (
        ValueError,
        IndexError,
    ):

        return None


def get_component(
    name,
    module,
):

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

    lower_name = (
        name.lower()
    )

    if ".mlp." in lower_name:
        return "mlp"

    if ".linear_attn." in lower_name:
        return "linear_attn"

    if ".self_attn." in lower_name:
        return "self_attn"

    return None


def discover_component_blocks(
    model,
):

    components = {
        "mlp": set(),
        "linear_attn": set(),
        "self_attn": set(),
    }

    for name, module in (
        model.named_modules()
    ):

        component = (
            get_component(
                name,
                module,
            )
        )

        if component is None:
            continue

        block = (
            get_block_index(
                name
            )
        )

        if block is not None:

            components[
                component
            ].add(
                block
            )

    return {
        component: sorted(blocks)
        for component, blocks
        in components.items()
    }


# =============================================================================
# VALIDATE TEST CASES
# =============================================================================

def validate_test_cases(
    component_blocks,
):

    print(
        "\nValidating Experiment 008 "
        "test cases..."
    )

    for case in TEST_CASES:

        case_name = (
            case["name"]
        )

        for (
            component,
            requested_blocks,
        ) in (
            case["blocks"].items()
        ):

            if (
                component
                not in component_blocks
            ):

                raise RuntimeError(
                    f"Unknown component "
                    f"'{component}' "
                    f"in test case "
                    f"'{case_name}'."
                )

            available = set(
                component_blocks[
                    component
                ]
            )

            for block in (
                requested_blocks
            ):

                if (
                    block
                    not in available
                ):

                    raise RuntimeError(
                        f"Block {block} "
                        f"is not available "
                        f"for component "
                        f"'{component}' "
                        f"in test case "
                        f"'{case_name}'. "
                        f"Available: "
                        f"{sorted(available)}"
                    )

    print(
        "All requested test cases "
        "are valid."
    )


# =============================================================================
# GROUP-WISE TERNARY QUANTIZATION
# =============================================================================

@torch.no_grad()
def groupwise_ternarize_weight(
    weight,
    group_size=GROUP_SIZE,
):

    original_shape = (
        weight.shape
    )

    flat = (
        weight
        .detach()
        .float()
        .reshape(-1)
    )

    num_values = (
        flat.numel()
    )

    padding = (
        group_size
        - (
            num_values
            % group_size
        )
    ) % group_size

    if padding > 0:

        padded = F.pad(
            flat,
            (0, padding),
            mode="constant",
            value=0.0,
        )

    else:

        padded = flat

    groups = (
        padded.reshape(
            -1,
            group_size,
        )
    )

    alpha = (
        groups
        .abs()
        .mean(
            dim=1,
            keepdim=True,
        )
    )

    threshold = (
        0.5 * alpha
    )

    ternary = (
        torch.zeros_like(
            groups
        )
    )

    ternary[
        groups > threshold
    ] = 1.0

    ternary[
        groups < -threshold
    ] = -1.0

    reconstructed = (
        ternary
        * alpha
    )

    reconstructed = (
        reconstructed
        .reshape(-1)[
            :num_values
        ]
        .reshape(
            original_shape
        )
        .to(
            dtype=weight.dtype
        )
    )

    del flat
    del padded
    del groups
    del alpha
    del threshold
    del ternary

    return reconstructed


# =============================================================================
# GENERATION
# =============================================================================

@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt,
):

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value
        in inputs.items()
    }

    torch.cuda.synchronize()

    start = (
        time.perf_counter()
    )

    output = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
    )

    torch.cuda.synchronize()

    elapsed = (
        time.perf_counter()
        - start
    )

    input_tokens = (
        inputs[
            "input_ids"
        ].shape[1]
    )

    generated_tokens = (
        output.shape[1]
        - input_tokens
    )

    text = tokenizer.decode(
        output[0],
        skip_special_tokens=True,
    )

    speed = (
        generated_tokens
        / elapsed
        if elapsed > 0
        else 0.0
    )

    del output

    for value in (
        inputs.values()
    ):
        del value

    del inputs

    return {
        "text": text,
        "tokens": generated_tokens,
        "time": elapsed,
        "speed": speed,
    }


# =============================================================================
# TERNARIZE A GUIDED COMBINATION
# =============================================================================

@torch.no_grad()
def ternarize_combination(
    model,
    block_map,
):

    affected_linear_layers = 0
    affected_parameters = 0

    relative_errors = []

    affected_by_component = {
        "mlp": {
            "linear_layers": 0,
            "parameters": 0,
            "errors": [],
        },

        "linear_attn": {
            "linear_layers": 0,
            "parameters": 0,
            "errors": [],
        },

        "self_attn": {
            "linear_layers": 0,
            "parameters": 0,
            "errors": [],
        },
    }

    affected_names = []

    selected_sets = {
        component: set(blocks)
        for component, blocks
        in block_map.items()
    }

    for name, module in (
        model.named_modules()
    ):

        component = (
            get_component(
                name,
                module,
            )
        )

        if component is None:
            continue

        if (
            component
            not in selected_sets
        ):
            continue

        block = (
            get_block_index(
                name
            )
        )

        if (
            block
            not in selected_sets[
                component
            ]
        ):
            continue

        weight = (
            module.weight
        )

        original_fp32 = (
            weight
            .detach()
            .float()
        )

        ternary_weight = (
            groupwise_ternarize_weight(
                weight,
                GROUP_SIZE,
            )
        )

        reconstructed_fp32 = (
            ternary_weight.float()
        )

        error = (
            original_fp32
            - reconstructed_fp32
        )

        original_norm = (
            torch.linalg.vector_norm(
                original_fp32
            ).item()
        )

        error_norm = (
            torch.linalg.vector_norm(
                error
            ).item()
        )

        relative_error = (
            error_norm
            / original_norm
            if original_norm > 0
            else 0.0
        )

        weight.copy_(
            ternary_weight
        )

        parameter_count = (
            weight.numel()
        )

        affected_linear_layers += 1

        affected_parameters += (
            parameter_count
        )

        relative_errors.append(
            relative_error
        )

        affected_by_component[
            component
        ][
            "linear_layers"
        ] += 1

        affected_by_component[
            component
        ][
            "parameters"
        ] += parameter_count

        affected_by_component[
            component
        ][
            "errors"
        ].append(
            relative_error
        )

        affected_names.append(
            (
                component,
                block,
                name,
                parameter_count,
                relative_error,
            )
        )

        del original_fp32
        del reconstructed_fp32
        del ternary_weight
        del error

    if relative_errors:

        mean_error = (
            sum(relative_errors)
            / len(relative_errors)
        )

        max_error = max(
            relative_errors
        )

    else:

        mean_error = 0.0
        max_error = 0.0

    component_summary = {}

    for (
        component,
        data,
    ) in (
        affected_by_component.items()
    ):

        errors = (
            data["errors"]
        )

        if errors:

            component_mean = (
                sum(errors)
                / len(errors)
            )

            component_max = (
                max(errors)
            )

        else:

            component_mean = 0.0
            component_max = 0.0

        component_summary[
            component
        ] = {
            "linear_layers":
                data[
                    "linear_layers"
                ],

            "parameters":
                data[
                    "parameters"
                ],

            "mean_error":
                component_mean,

            "max_error":
                component_max,
        }

    return {
        "linear_layers":
            affected_linear_layers,

        "parameters":
            affected_parameters,

        "mean_error":
            mean_error,

        "max_error":
            max_error,

        "component_summary":
            component_summary,

        "affected_names":
            affected_names,
    }


# =============================================================================
# HEADER
# =============================================================================

print(
    "=" * 110
)

print(
    "EXPERIMENT 008 — "
    "Qwen3.5-0.8B "
    "Sensitivity-Guided Block Combinations"
)

print(
    "=" * 110
)

print(
    "\nConfiguration"
)

print(
    "-" * 110
)

print(
    f"Model              : "
    f"{MODEL_ID}"
)

print(
    f"Device             : "
    f"{DEVICE}"
)

print(
    f"Data type          : "
    f"{DTYPE}"
)

print(
    f"Group size         : "
    f"{GROUP_SIZE}"
)

print(
    f"Maximum new tokens : "
    f"{MAX_NEW_TOKENS}"
)

print(
    f"Number test cases  : "
    f"{len(TEST_CASES)}"
)


# =============================================================================
# ENVIRONMENT
# =============================================================================

print(
    "\nEnvironment"
)

print(
    "-" * 110
)

print(
    f"PyTorch            : "
    f"{torch.__version__}"
)

print(
    f"CUDA runtime       : "
    f"{torch.version.cuda}"
)

print(
    f"CUDA available     : "
    f"{torch.cuda.is_available()}"
)

if not torch.cuda.is_available():

    raise RuntimeError(
        "CUDA is required."
    )

print(
    f"GPU                : "
    f"{torch.cuda.get_device_name(0)}"
)

print(
    f"GPU capability     : "
    f"{torch.cuda.get_device_capability(0)}"
)

total_vram = (
    torch.cuda
    .get_device_properties(0)
    .total_memory
    / (1024 ** 3)
)

print(
    f"Total VRAM         : "
    f"{total_vram:.2f} GB"
)


# =============================================================================
# TOKENIZER
# =============================================================================

print(
    "\nLoading tokenizer..."
)

tokenizer = (
    AutoTokenizer
    .from_pretrained(
        MODEL_ID
    )
)


# =============================================================================
# DISCOVER ARCHITECTURE
# =============================================================================

discovery_model = (
    load_model()
)

TOTAL_PARAMS = sum(
    parameter.numel()
    for parameter
    in discovery_model.parameters()
)

component_blocks = (
    discover_component_blocks(
        discovery_model
    )
)

print(
    "\nDiscovered component blocks"
)

print(
    "-" * 110
)

for component, blocks in (
    component_blocks.items()
):

    print(
        f"{component:<15}: "
        f"{blocks}"
    )

print(
    f"\nTotal parameters : "
    f"{TOTAL_PARAMS:,}"
)

validate_test_cases(
    component_blocks
)


# =============================================================================
# FP16 REFERENCE
# =============================================================================

print(
    "\n" + "=" * 110
)

print(
    "FP16 REFERENCE"
)

print(
    "=" * 110
)

for index, prompt in enumerate(
    PROMPTS,
    start=1,
):

    result = generate(
        discovery_model,
        tokenizer,
        prompt,
    )

    print(
        f"\nPROMPT {index}"
    )

    print(
        "-" * 110
    )

    print(
        result["text"]
    )

    print(
        f"\nGenerated tokens : "
        f"{result['tokens']}"
    )

    print(
        f"Generation time  : "
        f"{result['time']:.3f} sec"
    )

    print(
        f"Tokens/sec       : "
        f"{result['speed']:.2f}"
    )

    del result


release_model(
    discovery_model
)

del discovery_model

gc.collect()
torch.cuda.empty_cache()


# =============================================================================
# RUN GUIDED COMBINATIONS
# =============================================================================

all_results = []


for case_index, case in enumerate(
    TEST_CASES,
    start=1,
):

    case_name = (
        case["name"]
    )

    block_map = (
        case["blocks"]
    )

    print(
        "\n\n" + "#" * 110
    )

    print(
        f"TEST CASE "
        f"{case_index}/"
        f"{len(TEST_CASES)} — "
        f"{case_name}"
    )

    print(
        "#" * 110
    )

    print(
        "\nSelected blocks:"
    )

    for (
        component,
        blocks,
    ) in (
        block_map.items()
    ):

        print(
            f"  "
            f"{component:<15}: "
            f"{blocks}"
        )


    # =========================================================================
    # Fresh FP16 model for each combination.
    # =========================================================================

    model = load_model()

    stats = (
        ternarize_combination(
            model,
            block_map,
        )
    )

    if (
        stats["linear_layers"]
        == 0
    ):

        raise RuntimeError(
            f"No layers were ternarized "
            f"for test case "
            f"{case_name}."
        )

    coverage = (
        stats["parameters"]
        / TOTAL_PARAMS
        * 100
    )


    # =========================================================================
    # OVERALL STATS
    # =========================================================================

    print(
        "\nOVERALL TERNARY STATS"
    )

    print(
        "-" * 110
    )

    print(
        f"Affected Linear layers : "
        f"{stats['linear_layers']}"
    )

    print(
        f"Affected parameters    : "
        f"{stats['parameters']:,}"
    )

    print(
        f"Model coverage         : "
        f"{coverage:.2f}%"
    )

    print(
        f"Mean relative error    : "
        f"{stats['mean_error']:.6f}"
    )

    print(
        f"Max relative error     : "
        f"{stats['max_error']:.6f}"
    )


    # =========================================================================
    # COMPONENT BREAKDOWN
    # =========================================================================

    print(
        "\nCOMPONENT BREAKDOWN"
    )

    print(
        "-" * 110
    )

    for component in [
        "mlp",
        "linear_attn",
        "self_attn",
    ]:

        data = (
            stats[
                "component_summary"
            ][
                component
            ]
        )

        if (
            data["linear_layers"]
            == 0
        ):
            continue

        component_coverage = (
            data["parameters"]
            / TOTAL_PARAMS
            * 100
        )

        print(
            f"{component:<15}"
            f" | Linear layers "
            f"{data['linear_layers']:>3}"
            f" | Parameters "
            f"{data['parameters']:>12,}"
            f" | Coverage "
            f"{component_coverage:>6.2f}%"
            f" | Mean error "
            f"{data['mean_error']:.4f}"
            f" | Max error "
            f"{data['max_error']:.4f}"
        )


    # =========================================================================
    # EXACT MATRICES
    # =========================================================================

    print(
        "\nTERNARIZED MATRICES"
    )

    print(
        "-" * 110
    )

    for (
        component,
        block,
        layer_name,
        parameter_count,
        relative_error,
    ) in (
        stats[
            "affected_names"
        ]
    ):

        print(
            f"{component:<15}"
            f" block "
            f"{block:>2}"
            f" | "
            f"{layer_name}"
        )

        print(
            f"    Parameters     : "
            f"{parameter_count:,}"
        )

        print(
            f"    Relative error : "
            f"{relative_error:.6f}"
        )


    # =========================================================================
    # GENERATION
    # =========================================================================

    print(
        "\nGENERATION"
    )

    print(
        "-" * 110
    )

    total_time = 0.0
    total_tokens = 0

    for (
        prompt_index,
        prompt,
    ) in enumerate(
        PROMPTS,
        start=1,
    ):

        result = generate(
            model,
            tokenizer,
            prompt,
        )

        print(
            f"\nPROMPT "
            f"{prompt_index}"
        )

        print(
            result["text"]
        )

        print(
            f"\nGenerated tokens : "
            f"{result['tokens']}"
        )

        print(
            f"Generation time  : "
            f"{result['time']:.3f} sec"
        )

        print(
            f"Tokens/sec       : "
            f"{result['speed']:.2f}"
        )

        total_time += (
            result["time"]
        )

        total_tokens += (
            result["tokens"]
        )

        del result


    throughput = (
        total_tokens
        / total_time
        if total_time > 0
        else 0.0
    )

    print(
        f"\nAggregate throughput : "
        f"{throughput:.2f} "
        f"tokens/sec"
    )


    # =========================================================================
    # STORE SUMMARY
    # =========================================================================

    all_results.append(
        {
            "name":
                case_name,

            "block_map":
                {
                    component:
                        list(blocks)

                    for (
                        component,
                        blocks,
                    ) in (
                        block_map.items()
                    )
                },

            "linear_layers":
                stats[
                    "linear_layers"
                ],

            "parameters":
                stats[
                    "parameters"
                ],

            "coverage":
                coverage,

            "mean_error":
                stats[
                    "mean_error"
                ],

            "max_error":
                stats[
                    "max_error"
                ],

            "throughput":
                throughput,
        }
    )


    del stats


    # =========================================================================
    # CLEANUP
    # =========================================================================

    release_model(
        model
    )

    del model

    gc.collect()
    gc.collect()

    torch.cuda.empty_cache()

    print_gpu_memory(
        "ready for next combination"
    )


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print(
    "\n\n" + "=" * 125
)

print(
    "EXPERIMENT 008 — "
    "SENSITIVITY-GUIDED "
    "COMBINATION SUMMARY"
)

print(
    "=" * 125
)

print(
    f"{'Case':<24}"
    f"{'Linear':>8}"
    f"{'Parameters':>18}"
    f"{'Coverage':>12}"
    f"{'Mean Err':>12}"
    f"{'Max Err':>12}"
    f"{'Tok/s':>10}"
)

print(
    "-" * 125
)

for result in all_results:

    print(
        f"{result['name']:<24}"
        f"{result['linear_layers']:>8}"
        f"{result['parameters']:>18,}"
        f"{result['coverage']:>11.2f}%"
        f"{result['mean_error']:>12.4f}"
        f"{result['max_error']:>12.4f}"
        f"{result['throughput']:>10.2f}"
    )


# =============================================================================
# BLOCK MAP SUMMARY
# =============================================================================

print(
    "\n" + "=" * 125
)

print(
    "TESTED BLOCK COMBINATIONS"
)

print(
    "=" * 125
)

for result in all_results:

    print(
        f"\n{result['name']}"
    )

    for (
        component,
        blocks,
    ) in (
        result[
            "block_map"
        ].items()
    ):

        print(
            f"  "
            f"{component:<15}: "
            f"{blocks}"
        )


# =============================================================================
# IMPORTANT INTERPRETATION
# =============================================================================

print(
    "\n" + "=" * 110
)

print(
    "IMPORTANT INTERPRETATION"
)

print(
    "=" * 110
)

print(
    "Experiment 008 uses the same "
    "ternary quantization method as "
    "Experiments 004-007."
)

print()

print(
    f"Group size       : "
    f"{GROUP_SIZE}"
)

print(
    "Scale            : "
    "mean(abs(group))"
)

print(
    "Threshold        : "
    "0.5 * scale"
)

print(
    "Codes            : "
    "{-1, 0, +1}"
)

print(
    "Runtime storage  : "
    "reconstructed FP16"
)

print()

print(
    "Every combination starts from "
    "a fresh original FP16 model."
)

print()

print(
    "Linear Attention block 0 is "
    "intentionally excluded from "
    "the guided combinations because "
    "Experiment 007 showed catastrophic "
    "generation degradation when that "
    "block alone was ternarized."
)

print()

print(
    "The primary question is whether "
    "individually tolerated blocks "
    "remain functional when ternarized "
    "together."
)

print()

print(
    "Generation throughput is only "
    "a runtime reference."
)

print(
    "It is NOT packed ternary "
    "inference throughput."
)

print()

print(
    "Prompt quality remains a "
    "preliminary functional diagnostic "
    "rather than a formal benchmark."
)

print(
    "\n" + "=" * 110
)

print(
    "EXPERIMENT 008 COMPLETE"
)

print(
    "=" * 110
)
