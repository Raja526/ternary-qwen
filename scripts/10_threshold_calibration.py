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

THRESHOLD_MULTIPLIERS = [
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
]

PROMPTS = [
    "Explain in simple terms why the sky is blue.",
    "What is 17 multiplied by 23?",
    "If a train travels 60 km per hour for 2.5 hours, how far does it travel?",
]


# =============================================================================
# CALIBRATION TARGET
# =============================================================================
#
# This is MIXED_MEDIUM from Experiment 008.
#
# At threshold = 0.5 * alpha:
#
#   MLP          : [6, 12]
#   Linear Attn  : [4, 8]
#   Self Attn    : [3, 7]
#
#   Affected Linear layers : 20
#   Affected parameters    : 57,671,680
#   Coverage               : ~7.67%
#
# Experiment 008 showed clear functional degradation for this configuration.
#
# Experiment 010 changes ONLY the threshold multiplier.
#
# =============================================================================

TARGET_BLOCKS = {
    "mlp": [6, 12],
    "linear_attn": [4, 8],
    "self_attn": [3, 7],
}


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

    print(f"\nGPU MEMORY — {label}")

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

    lower_name = name.lower()

    if ".mlp." in lower_name:
        return "mlp"

    if ".linear_attn." in lower_name:
        return "linear_attn"

    if ".self_attn." in lower_name:
        return "self_attn"

    return None


def is_target_matrix(
    name,
    module,
):

    component = get_component(
        name,
        module,
    )

    if component is None:
        return False

    if component not in TARGET_BLOCKS:
        return False

    block = get_block_index(
        name
    )

    if block is None:
        return False

    return (
        block
        in TARGET_BLOCKS[
            component
        ]
    )


# =============================================================================
# DISCOVER TARGET MATRICES
# =============================================================================

def discover_target_matrices(
    model,
):

    targets = []

    for name, module in (
        model.named_modules()
    ):

        if not is_target_matrix(
            name,
            module,
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
# GROUP-WISE TERNARY
# =============================================================================

@torch.no_grad()
def groupwise_ternarize_weight(
    weight,
    threshold_multiplier,
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
        threshold_multiplier
        * alpha
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

    # -------------------------------------------------------------------------
    # Ternary-code statistics
    # -------------------------------------------------------------------------

    valid_ternary = (
        ternary
        .reshape(-1)[
            :num_values
        ]
    )

    negative_count = (
        valid_ternary < 0
    ).sum().item()

    zero_count = (
        valid_ternary == 0
    ).sum().item()

    positive_count = (
        valid_ternary > 0
    ).sum().item()

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
    del valid_ternary

    return (
        reconstructed,
        negative_count,
        zero_count,
        positive_count,
    )


# =============================================================================
# APPLY THRESHOLD TO TARGET CONFIGURATION
# =============================================================================

@torch.no_grad()
def ternarize_target_configuration(
    model,
    threshold_multiplier,
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
        ):
            continue

        weight = module.weight

        original_fp32 = (
            weight
            .detach()
            .float()
        )

        (
            reconstructed,
            negative_count,
            zero_count,
            positive_count,
        ) = groupwise_ternarize_weight(
            weight,
            threshold_multiplier,
            GROUP_SIZE,
        )

        reconstructed_fp32 = (
            reconstructed.float()
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

        parameter_count = (
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
                    parameter_count,

                "relative_error":
                    relative_error,
            }
        )

        total_negative += (
            negative_count
        )

        total_zero += (
            zero_count
        )

        total_positive += (
            positive_count
        )

        del original_fp32
        del reconstructed_fp32
        del reconstructed
        del error

    if not matrix_stats:

        raise RuntimeError(
            "No target matrices "
            "were ternarized."
        )

    total_parameters = sum(
        item["parameters"]
        for item
        in matrix_stats
    )

    weighted_error = sum(
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

    return {
        "matrix_stats":
            matrix_stats,

        "linear_layers":
            len(matrix_stats),

        "parameters":
            total_parameters,

        "mean_error":
            weighted_error,

        "max_error":
            max_error,

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
        generated_tokens / elapsed
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
    "=" * 120
)

print(
    "EXPERIMENT 010 — "
    "Qwen3.5-0.8B "
    "Ternary Threshold Calibration"
)

print(
    "=" * 120
)

print(
    "\nConfiguration"
)

print(
    "-" * 120
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
    "Scale                  : "
    "mean(abs(group))"
)

print(
    f"Threshold multipliers  : "
    f"{THRESHOLD_MULTIPLIERS}"
)

print(
    f"Maximum new tokens     : "
    f"{MAX_NEW_TOKENS}"
)

print(
    "\nTarget configuration:"
)

for component, blocks in (
    TARGET_BLOCKS.items()
):

    print(
        f"  {component:<15}: "
        f"{blocks}"
    )


# =============================================================================
# ENVIRONMENT
# =============================================================================

print(
    "\nEnvironment"
)

print(
    "-" * 120
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
# REFERENCE / DISCOVERY MODEL
# =============================================================================

reference_model = (
    load_model()
)

TOTAL_PARAMS = sum(
    parameter.numel()
    for parameter
    in reference_model.parameters()
)

target_matrices = (
    discover_target_matrices(
        reference_model
    )
)

TARGET_PARAMS = sum(
    item["parameters"]
    for item
    in target_matrices
)

TARGET_COVERAGE = (
    TARGET_PARAMS
    / TOTAL_PARAMS
    * 100
)


print(
    "\nTARGET MATRICES"
)

print(
    "-" * 120
)

for item in target_matrices:

    print(
        f"{item['component']:<15}"
        f" block "
        f"{item['block']:>2}"
        f" | "
        f"{item['name']:<55}"
        f" | "
        f"{item['parameters']:>12,}"
    )


print(
    f"\nAffected Linear layers : "
    f"{len(target_matrices)}"
)

print(
    f"Affected parameters    : "
    f"{TARGET_PARAMS:,}"
)

print(
    f"Model coverage         : "
    f"{TARGET_COVERAGE:.2f}%"
)


if len(target_matrices) != 20:

    raise RuntimeError(
        "Expected 20 target Linear "
        f"matrices, found "
        f"{len(target_matrices)}."
    )


if TARGET_PARAMS != 57_671_680:

    raise RuntimeError(
        "Unexpected target parameter "
        f"count: {TARGET_PARAMS:,}. "
        "Expected 57,671,680."
    )


# =============================================================================
# FP16 REFERENCE
# =============================================================================

print(
    "\n" + "=" * 120
)

print(
    "FP16 REFERENCE"
)

print(
    "=" * 120
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
        "-" * 120
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
# THRESHOLD SWEEP
# =============================================================================

all_results = []


for test_index, multiplier in enumerate(
    THRESHOLD_MULTIPLIERS,
    start=1,
):

    print(
        "\n\n" + "#" * 120
    )

    print(
        f"THRESHOLD TEST "
        f"{test_index}/"
        f"{len(THRESHOLD_MULTIPLIERS)}"
        f" — {multiplier:.2f} × alpha"
    )

    print(
        "#" * 120
    )


    # =========================================================================
    # FRESH MODEL
    # =========================================================================

    model = load_model()


    # =========================================================================
    # APPLY TERNARY QUANTIZATION
    # =========================================================================

    stats = (
        ternarize_target_configuration(
            model,
            multiplier,
        )
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
        "-" * 120
    )

    print(
        f"Threshold multiplier  : "
        f"{multiplier:.2f}"
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
        f"{TARGET_COVERAGE:.2f}%"
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
    # MATRIX ERROR DETAILS
    # =========================================================================

    print(
        "\nMATRIX ERRORS"
    )

    print(
        "-" * 120
    )

    sorted_matrix_stats = sorted(
        stats["matrix_stats"],
        key=lambda item:
            item["relative_error"],
        reverse=True,
    )

    for item in sorted_matrix_stats:

        print(
            f"{item['component']:<15}"
            f" block "
            f"{item['block']:>2}"
            f" | "
            f"{item['name']:<55}"
            f" | "
            f"{item['relative_error']:.6f}"
        )


    # =========================================================================
    # GENERATION
    # =========================================================================

    print(
        "\nGENERATION"
    )

    print(
        "-" * 120
    )

    total_tokens = 0
    total_time = 0.0

    outputs = []

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

        outputs.append(
            result["text"]
        )

        del result


    throughput = (
        total_tokens / total_time
        if total_time > 0
        else 0.0
    )

    print(
        f"\nAggregate throughput : "
        f"{throughput:.2f} "
        f"tokens/sec"
    )


    # =========================================================================
    # SAVE RESULT
    # =========================================================================

    all_results.append(
        {
            "multiplier":
                multiplier,

            "mean_error":
                stats["mean_error"],

            "max_error":
                stats["max_error"],

            "negative_pct":
                negative_pct,

            "zero_pct":
                zero_pct,

            "positive_pct":
                positive_pct,

            "throughput":
                throughput,
        }
    )


    del outputs
    del sorted_matrix_stats
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
        "ready for next threshold"
    )


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print(
    "\n\n" + "=" * 130
)

print(
    "EXPERIMENT 010 — "
    "TERNARY THRESHOLD "
    "CALIBRATION SUMMARY"
)

print(
    "=" * 130
)

print(
    f"{'Threshold':>10}"
    f"{'Mean Err':>12}"
    f"{'Max Err':>12}"
    f"{'-1 %':>10}"
    f"{'0 %':>10}"
    f"{'+1 %':>10}"
    f"{'Tok/s':>12}"
)

print(
    "-" * 130
)

for result in all_results:

    print(
        f"{result['multiplier']:>10.2f}"
        f"{result['mean_error']:>12.6f}"
        f"{result['max_error']:>12.6f}"
        f"{result['negative_pct']:>10.2f}"
        f"{result['zero_pct']:>10.2f}"
        f"{result['positive_pct']:>10.2f}"
        f"{result['throughput']:>12.2f}"
    )


# =============================================================================
# NUMERICAL BEST THRESHOLD
# =============================================================================

best_error_result = min(
    all_results,
    key=lambda item:
        item["mean_error"],
)

print(
    "\nLowest mean reconstruction "
    "error in this sweep:"
)

print(
    f"Threshold multiplier : "
    f"{best_error_result['multiplier']:.2f}"
)

print(
    f"Mean relative error  : "
    f"{best_error_result['mean_error']:.6f}"
)

print(
    "\nIMPORTANT:"
)

print(
    "The numerically lowest-error "
    "threshold must NOT automatically "
    "be considered the best functional "
    "threshold."
)

print(
    "Generation quality must be "
    "analyzed separately."
)


# =============================================================================
# INTERPRETATION
# =============================================================================

print(
    "\n" + "=" * 120
)

print(
    "IMPORTANT INTERPRETATION"
)

print(
    "=" * 120
)

print(
    "Experiment 010 changes only the "
    "ternary threshold multiplier."
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
    "multiplier * scale"
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
    "Target configuration is the "
    "MIXED_MEDIUM configuration from "
    "Experiment 008."
)

print()

print(
    "The threshold changes both the "
    "reconstruction error and the "
    "distribution of {-1, 0, +1} codes."
)

print()

print(
    "Throughput remains only a runtime "
    "reference and is NOT packed ternary "
    "inference speed."
)

print()

print(
    "The three generation prompts are "
    "preliminary functional diagnostics, "
    "not a formal quality benchmark."
)

print(
    "\n" + "=" * 120
)

print(
    "EXPERIMENT 010 COMPLETE"
)

print(
    "=" * 120
)
