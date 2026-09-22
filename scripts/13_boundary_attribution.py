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
THRESHOLD_MULTIPLIER = 0.50

MIN_WEIGHT_PARAMS = 100_000
MAX_NEW_TOKENS = 40


PROMPTS = [
    "Explain in simple terms why the sky is blue.",
    "What is 17 multiplied by 23?",
    "If a train travels 60 km per hour for 2.5 hours, how far does it travel?",
]


# =============================================================================
# BASE CONFIGURATION
# =============================================================================
#
# MEDIUM from Experiment 012:
#
# MLP          [6, 12]
# Linear Attn  [4, 8]
# Self Attn    [3, 7]
#
# 20 Linear matrices
# 57,671,680 parameters
# 7.67% coverage
#
# This configuration remained functional on the three diagnostic prompts
# with optimized reconstruction scaling.
#
# =============================================================================

BASE_BLOCKS = {
    "mlp": [6, 12],
    "linear_attn": [4, 8],
    "self_attn": [3, 7],
}


# =============================================================================
# BOUNDARY ATTRIBUTION CASES
# =============================================================================

TEST_CASES = [

    # -------------------------------------------------------------------------
    # BASELINE CONTROL
    # -------------------------------------------------------------------------

    {
        "name": "MEDIUM_BASE",
        "blocks": {
            "mlp": [6, 12],
            "linear_attn": [4, 8],
            "self_attn": [3, 7],
        },
    },


    # -------------------------------------------------------------------------
    # SINGLE ADDITIONS
    # -------------------------------------------------------------------------

    {
        "name": "ADD_MLP18",
        "blocks": {
            "mlp": [6, 12, 18],
            "linear_attn": [4, 8],
            "self_attn": [3, 7],
        },
    },

    {
        "name": "ADD_LINEAR12",
        "blocks": {
            "mlp": [6, 12],
            "linear_attn": [4, 8, 12],
            "self_attn": [3, 7],
        },
    },

    {
        "name": "ADD_SELF11",
        "blocks": {
            "mlp": [6, 12],
            "linear_attn": [4, 8],
            "self_attn": [3, 7, 11],
        },
    },


    # -------------------------------------------------------------------------
    # PAIRWISE ADDITIONS
    # -------------------------------------------------------------------------

    {
        "name": "ADD_MLP18_LINEAR12",
        "blocks": {
            "mlp": [6, 12, 18],
            "linear_attn": [4, 8, 12],
            "self_attn": [3, 7],
        },
    },

    {
        "name": "ADD_MLP18_SELF11",
        "blocks": {
            "mlp": [6, 12, 18],
            "linear_attn": [4, 8],
            "self_attn": [3, 7, 11],
        },
    },

    {
        "name": "ADD_LINEAR12_SELF11",
        "blocks": {
            "mlp": [6, 12],
            "linear_attn": [4, 8, 12],
            "self_attn": [3, 7, 11],
        },
    },


    # -------------------------------------------------------------------------
    # FULL LARGE CONFIGURATION — EXPERIMENT 012 REPRODUCTION
    # -------------------------------------------------------------------------

    {
        "name": "LARGE_FULL",
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

    print(f"\nGPU MEMORY — {label}")

    print(
        f"Allocated : {allocated:.2f} MB"
    )

    print(
        f"Reserved  : {reserved:.2f} MB"
    )

    print(
        f"Free      : "
        f"{free_bytes / (1024 ** 2):.2f} MB"
    )

    print(
        f"Total     : "
        f"{total_bytes / (1024 ** 2):.2f} MB"
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

        position = parts.index(
            "layers"
        )

        return int(
            parts[position + 1]
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

    lower = name.lower()

    if ".mlp." in lower:
        return "mlp"

    if ".linear_attn." in lower:
        return "linear_attn"

    if ".self_attn." in lower:
        return "self_attn"

    return None


def is_target_matrix(
    name,
    module,
    target_blocks,
):

    component = get_component(
        name,
        module,
    )

    if component is None:
        return False

    if component not in target_blocks:
        return False

    block = get_block_index(
        name
    )

    if block is None:
        return False

    return (
        block
        in target_blocks[
            component
        ]
    )


# =============================================================================
# DISCOVERY
# =============================================================================

def discover_target_matrices(
    model,
    target_blocks,
):

    targets = []

    for name, module in (
        model.named_modules()
    ):

        if not is_target_matrix(
            name,
            module,
            target_blocks,
        ):
            continue

        targets.append(
            {
                "name":
                    name,

                "component":
                    get_component(
                        name,
                        module,
                    ),

                "block":
                    get_block_index(
                        name
                    ),

                "parameters":
                    module.weight.numel(),
            }
        )

    return targets


# =============================================================================
# OPTIMIZED TERNARY QUANTIZER
# =============================================================================

@torch.no_grad()
def optimized_ternarize_weight(
    weight,
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
        GROUP_SIZE
        - (
            num_values
            % GROUP_SIZE
        )
    ) % GROUP_SIZE


    if padding > 0:

        padded = F.pad(
            flat,
            (0, padding),
            mode="constant",
            value=0.0,
        )

    else:

        padded = flat


    groups = padded.reshape(
        -1,
        GROUP_SIZE,
    )


    # =========================================================================
    # INITIAL SCALE
    # =========================================================================

    initial_alpha = (
        groups
        .abs()
        .mean(
            dim=1,
            keepdim=True,
        )
    )


    # =========================================================================
    # FIXED THRESHOLD
    # =========================================================================

    threshold = (
        THRESHOLD_MULTIPLIER
        * initial_alpha
    )


    codes = torch.zeros_like(
        groups
    )

    codes[
        groups > threshold
    ] = 1.0

    codes[
        groups < -threshold
    ] = -1.0


    # =========================================================================
    # OPTIMIZED NON-ZERO SCALE
    # =========================================================================

    nonzero_mask = (
        codes != 0
    )


    nonzero_count = (
        nonzero_mask
        .sum(
            dim=1,
            keepdim=True,
        )
    )


    nonzero_abs_sum = (
        groups
        .abs()
        .mul(
            nonzero_mask
        )
        .sum(
            dim=1,
            keepdim=True,
        )
    )


    optimized_alpha = (
        nonzero_abs_sum
        / nonzero_count.clamp(
            min=1
        )
    )


    reconstruction_alpha = (
        torch.where(
            nonzero_count > 0,
            optimized_alpha,
            initial_alpha,
        )
    )


    # =========================================================================
    # RECONSTRUCT
    # =========================================================================

    reconstructed = (
        codes
        * reconstruction_alpha
    )


    # =========================================================================
    # CODE STATISTICS
    # =========================================================================

    valid_codes = (
        codes
        .reshape(-1)[
            :num_values
        ]
    )


    negative = (
        valid_codes < 0
    ).sum().item()


    zero = (
        valid_codes == 0
    ).sum().item()


    positive = (
        valid_codes > 0
    ).sum().item()


    # =========================================================================
    # SCALE STATISTICS
    # =========================================================================

    initial_mean = (
        initial_alpha
        .mean()
        .item()
    )


    optimized_mean = (
        reconstruction_alpha
        .mean()
        .item()
    )


    scale_ratio = (
        optimized_mean
        / initial_mean
        if initial_mean > 0
        else 0.0
    )


    # =========================================================================
    # RESTORE SHAPE
    # =========================================================================

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


    return {
        "reconstructed":
            reconstructed,

        "negative":
            negative,

        "zero":
            zero,

        "positive":
            positive,

        "scale_ratio":
            scale_ratio,
    }


# =============================================================================
# APPLY QUANTIZATION
# =============================================================================

@torch.no_grad()
def quantize_configuration(
    model,
    target_blocks,
):

    matrix_stats = []

    total_negative = 0
    total_zero = 0
    total_positive = 0


    for name, module in (
        model.named_modules()
    ):

        if not is_target_matrix(
            name,
            module,
            target_blocks,
        ):
            continue


        weight = module.weight


        original = (
            weight
            .detach()
            .float()
        )


        result = (
            optimized_ternarize_weight(
                weight
            )
        )


        reconstructed = (
            result[
                "reconstructed"
            ]
        )


        reconstructed_fp32 = (
            reconstructed.float()
        )


        difference = (
            original
            - reconstructed_fp32
        )


        original_norm = (
            torch.linalg.vector_norm(
                original
            ).item()
        )


        error_norm = (
            torch.linalg.vector_norm(
                difference
            ).item()
        )


        relative_error = (
            error_norm
            / original_norm
            if original_norm > 0
            else 0.0
        )


        parameters = (
            weight.numel()
        )


        component = (
            get_component(
                name,
                module,
            )
        )


        block = (
            get_block_index(
                name
            )
        )


        weight.copy_(
            reconstructed
        )


        matrix_stats.append(
            {
                "name":
                    name,

                "component":
                    component,

                "block":
                    block,

                "parameters":
                    parameters,

                "relative_error":
                    relative_error,

                "scale_ratio":
                    result[
                        "scale_ratio"
                    ],
            }
        )


        total_negative += (
            result["negative"]
        )

        total_zero += (
            result["zero"]
        )

        total_positive += (
            result["positive"]
        )


        del original
        del reconstructed
        del reconstructed_fp32
        del difference
        del result


    if not matrix_stats:

        raise RuntimeError(
            "No matrices were selected."
        )


    total_parameters = sum(
        item["parameters"]
        for item
        in matrix_stats
    )


    mean_error = sum(
        item["relative_error"]
        * item["parameters"]
        for item
        in matrix_stats
    ) / total_parameters


    max_error = max(
        item["relative_error"]
        for item
        in matrix_stats
    )


    mean_scale_ratio = sum(
        item["scale_ratio"]
        * item["parameters"]
        for item
        in matrix_stats
    ) / total_parameters


    return {
        "matrix_stats":
            matrix_stats,

        "linear_layers":
            len(matrix_stats),

        "parameters":
            total_parameters,

        "mean_error":
            mean_error,

        "max_error":
            max_error,

        "scale_ratio":
            mean_scale_ratio,

        "negative":
            total_negative,

        "zero":
            total_zero,

        "positive":
            total_positive,
    }


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
        "text":
            text,

        "tokens":
            generated_tokens,

        "time":
            elapsed,

        "speed":
            speed,
    }


# =============================================================================
# HEADER
# =============================================================================

print(
    "=" * 125
)

print(
    "EXPERIMENT 013 — "
    "Qwen3.5-0.8B Boundary Attribution "
    "with Optimized Ternary Scale"
)

print(
    "=" * 125
)


print(
    "\nConfiguration"
)

print(
    "-" * 125
)


print(
    f"Model                  : "
    f"{MODEL_ID}"
)

print(
    f"Device                 : "
    f"{DEVICE}"
)

print(
    f"Data type              : "
    f"{DTYPE}"
)

print(
    f"Group size             : "
    f"{GROUP_SIZE}"
)

print(
    f"Threshold multiplier   : "
    f"{THRESHOLD_MULTIPLIER}"
)

print(
    "Reconstruction scale   : "
    "mean(abs(non-zero selected weights))"
)

print(
    f"Maximum new tokens     : "
    f"{MAX_NEW_TOKENS}"
)

print(
    f"Attribution cases      : "
    f"{len(TEST_CASES)}"
)


# =============================================================================
# ENVIRONMENT
# =============================================================================

print(
    "\nEnvironment"
)

print(
    "-" * 125
)


print(
    f"PyTorch                : "
    f"{torch.__version__}"
)

print(
    f"CUDA runtime           : "
    f"{torch.version.cuda}"
)

print(
    f"CUDA available         : "
    f"{torch.cuda.is_available()}"
)


if not torch.cuda.is_available():

    raise RuntimeError(
        "CUDA is required."
    )


print(
    f"GPU                    : "
    f"{torch.cuda.get_device_name(0)}"
)

print(
    f"GPU capability         : "
    f"{torch.cuda.get_device_capability(0)}"
)


total_vram = (
    torch.cuda
    .get_device_properties(0)
    .total_memory
    / (1024 ** 3)
)


print(
    f"Total VRAM             : "
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
# REFERENCE MODEL
# =============================================================================

reference_model = (
    load_model()
)


TOTAL_PARAMS = sum(
    parameter.numel()
    for parameter
    in reference_model.parameters()
)


print(
    f"\nTotal model parameters : "
    f"{TOTAL_PARAMS:,}"
)


# =============================================================================
# VALIDATE CASES
# =============================================================================

print(
    "\nBOUNDARY ATTRIBUTION CASE VALIDATION"
)

print(
    "-" * 125
)


case_metadata = {}


for case in TEST_CASES:

    targets = (
        discover_target_matrices(
            reference_model,
            case["blocks"],
        )
    )


    parameters = sum(
        item["parameters"]
        for item
        in targets
    )


    coverage = (
        parameters
        / TOTAL_PARAMS
        * 100
    )


    case_metadata[
        case["name"]
    ] = {
        "linear_layers":
            len(targets),

        "parameters":
            parameters,

        "coverage":
            coverage,
    }


    print(
        f"\n{case['name']}"
    )

    print(
        f"  Blocks       : "
        f"{case['blocks']}"
    )

    print(
        f"  Linear layers: "
        f"{len(targets)}"
    )

    print(
        f"  Parameters   : "
        f"{parameters:,}"
    )

    print(
        f"  Coverage     : "
        f"{coverage:.2f}%"
    )


# =============================================================================
# VALIDATE BASE AND LARGE
# =============================================================================

medium_metadata = (
    case_metadata[
        "MEDIUM_BASE"
    ]
)


if (
    medium_metadata["linear_layers"]
    != 20
    or medium_metadata["parameters"]
    != 57_671_680
):

    raise RuntimeError(
        "MEDIUM_BASE does not match "
        "Experiment 012 MEDIUM."
    )


large_metadata = (
    case_metadata[
        "LARGE_FULL"
    ]
)


if (
    large_metadata["linear_layers"]
    != 30
    or large_metadata["parameters"]
    != 86_507_520
):

    raise RuntimeError(
        "LARGE_FULL does not match "
        "Experiment 012 LARGE."
    )


# =============================================================================
# FP16 REFERENCE
# =============================================================================

print(
    "\n" + "=" * 125
)

print(
    "FP16 REFERENCE"
)

print(
    "=" * 125
)


for index, prompt in enumerate(
    PROMPTS,
    start=1,
):

    result = generate(
        reference_model,
        tokenizer,
        prompt,
    )


    print(
        f"\nPROMPT {index}"
    )

    print(
        "-" * 125
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
    reference_model
)

del reference_model

gc.collect()
torch.cuda.empty_cache()


# =============================================================================
# ATTRIBUTION TESTS
# =============================================================================

all_results = []


for test_index, case in enumerate(
    TEST_CASES,
    start=1,
):

    name = (
        case["name"]
    )

    target_blocks = (
        case["blocks"]
    )


    print(
        "\n\n" + "#" * 125
    )

    print(
        f"ATTRIBUTION TEST "
        f"{test_index}/"
        f"{len(TEST_CASES)}"
        f" — {name}"
    )

    print(
        "#" * 125
    )


    print(
        "\nSelected blocks:"
    )


    for component, blocks in (
        target_blocks.items()
    ):

        print(
            f"  {component:<15}: "
            f"{blocks}"
        )


    # =========================================================================
    # FRESH MODEL
    # =========================================================================

    model = load_model()


    # =========================================================================
    # QUANTIZE
    # =========================================================================

    stats = (
        quantize_configuration(
            model,
            target_blocks,
        )
    )


    coverage = (
        stats["parameters"]
        / TOTAL_PARAMS
        * 100
    )


    expected = (
        case_metadata[name]
    )


    if (
        stats["linear_layers"]
        != expected[
            "linear_layers"
        ]
    ):

        raise RuntimeError(
            f"{name}: Linear layer "
            "count changed unexpectedly."
        )


    if (
        stats["parameters"]
        != expected[
            "parameters"
        ]
    ):

        raise RuntimeError(
            f"{name}: parameter count "
            "changed unexpectedly."
        )


    total_codes = (
        stats["negative"]
        + stats["zero"]
        + stats["positive"]
    )


    negative_pct = (
        stats["negative"]
        / total_codes
        * 100
    )


    zero_pct = (
        stats["zero"]
        / total_codes
        * 100
    )


    positive_pct = (
        stats["positive"]
        / total_codes
        * 100
    )


    print(
        "\nTERNARY STATS"
    )

    print(
        "-" * 125
    )


    print(
        f"Case                  : "
        f"{name}"
    )

    print(
        f"Affected Linear layers: "
        f"{stats['linear_layers']}"
    )

    print(
        f"Affected parameters   : "
        f"{stats['parameters']:,}"
    )

    print(
        f"Model coverage        : "
        f"{coverage:.2f}%"
    )

    print(
        f"Mean relative error   : "
        f"{stats['mean_error']:.6f}"
    )

    print(
        f"Max relative error    : "
        f"{stats['max_error']:.6f}"
    )

    print(
        f"Mean scale ratio      : "
        f"{stats['scale_ratio']:.6f}"
    )

    print(
        f"-1 codes              : "
        f"{stats['negative']:,} "
        f"({negative_pct:.2f}%)"
    )

    print(
        f" 0 codes              : "
        f"{stats['zero']:,} "
        f"({zero_pct:.2f}%)"
    )

    print(
        f"+1 codes              : "
        f"{stats['positive']:,} "
        f"({positive_pct:.2f}%)"
    )


    # =========================================================================
    # TOP MATRIX ERRORS
    # =========================================================================

    print(
        "\nTOP 10 MATRIX ERRORS"
    )

    print(
        "-" * 125
    )


    worst = sorted(
        stats["matrix_stats"],
        key=lambda item:
            item["relative_error"],
        reverse=True,
    )[:10]


    for item in worst:

        print(
            f"{item['component']:<15}"
            f" block "
            f"{item['block']:>2}"
            f" | "
            f"{item['name']:<55}"
            f" | Err "
            f"{item['relative_error']:.6f}"
            f" | Scale "
            f"{item['scale_ratio']:.4f}"
        )


    # =========================================================================
    # GENERATION
    # =========================================================================

    print(
        "\nGENERATION"
    )

    print(
        "-" * 125
    )


    total_tokens = 0
    total_time = 0.0


    for prompt_index, prompt in enumerate(
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


        total_tokens += (
            result["tokens"]
        )

        total_time += (
            result["time"]
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
    # STORE RESULT
    # =========================================================================

    all_results.append(
        {
            "name":
                name,

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

            "scale_ratio":
                stats[
                    "scale_ratio"
                ],

            "zero_pct":
                zero_pct,

            "throughput":
                throughput,
        }
    )


    del worst
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
        "ready for next attribution test"
    )


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print(
    "\n\n" + "=" * 155
)

print(
    "EXPERIMENT 013 — "
    "BOUNDARY ATTRIBUTION SUMMARY"
)

print(
    "=" * 155
)


print(
    f"{'Case':<24}"
    f"{'Linear':>8}"
    f"{'Parameters':>16}"
    f"{'Coverage':>11}"
    f"{'Mean Err':>12}"
    f"{'Max Err':>12}"
    f"{'Scale':>10}"
    f"{'Zero %':>10}"
    f"{'Tok/s':>10}"
)


print(
    "-" * 155
)


for result in all_results:

    print(
        f"{result['name']:<24}"
        f"{result['linear_layers']:>8}"
        f"{result['parameters']:>16,}"
        f"{result['coverage']:>10.2f}%"
        f"{result['mean_error']:>12.6f}"
        f"{result['max_error']:>12.6f}"
        f"{result['scale_ratio']:>10.4f}"
        f"{result['zero_pct']:>10.2f}"
        f"{result['throughput']:>10.2f}"
    )


# =============================================================================
# COVERAGE INCREMENTS RELATIVE TO MEDIUM
# =============================================================================

base_parameters = (
    medium_metadata[
        "parameters"
    ]
)


base_coverage = (
    medium_metadata[
        "coverage"
    ]
)


print(
    "\nCOVERAGE INCREASE RELATIVE "
    "TO MEDIUM_BASE"
)

print(
    "-" * 125
)


for result in all_results:

    added_parameters = (
        result["parameters"]
        - base_parameters
    )


    added_coverage = (
        result["coverage"]
        - base_coverage
    )


    print(
        f"{result['name']:<24}"
        f" +{added_parameters:>12,} params"
        f" | +{added_coverage:>6.2f}% "
        f"model coverage"
    )


# =============================================================================
# IMPORTANT INTERPRETATION
# =============================================================================

print(
    "\n" + "=" * 125
)

print(
    "IMPORTANT INTERPRETATION"
)

print(
    "=" * 125
)


print(
    "Experiment 013 keeps the optimized "
    "ternary quantizer completely fixed."
)

print()

print(
    f"Group size       : "
    f"{GROUP_SIZE}"
)

print(
    f"Threshold        : "
    f"{THRESHOLD_MULTIPLIER} "
    "* initial_alpha"
)

print(
    "Initial scale    : "
    "mean(abs(group))"
)

print(
    "Recon scale      : "
    "mean(abs(non-zero selected weights))"
)

print(
    "Codes            : "
    "{-1, 0, +1}"
)

print()

print(
    "MEDIUM_BASE reproduces the "
    "7.67% Experiment 012 configuration."
)

print(
    "LARGE_FULL reproduces the "
    "11.50% Experiment 012 configuration."
)

print()

print(
    "Single-addition cases test whether "
    "MLP block 18, Linear Attention "
    "block 12, or Self Attention block "
    "11 independently causes degradation."
)

print()

print(
    "Pairwise cases test whether degradation "
    "emerges from interactions between "
    "two added components."
)

print()

print(
    "LARGE_FULL tests the combined "
    "three-way interaction."
)

print()

print(
    "Mean reconstruction error alone "
    "must not be interpreted as a "
    "functional quality score."
)

print()

print(
    "Throughput remains only a runtime "
    "reference and is NOT packed ternary "
    "inference speed."
)

print()

print(
    "The three prompts remain preliminary "
    "functional diagnostics, not a formal "
    "quality benchmark."
)


print(
    "\n" + "=" * 125
)

print(
    "EXPERIMENT 013 COMPLETE"
)

print(
    "=" * 125
)
