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
# SAME MIXED_MEDIUM CONFIGURATION USED IN EXPERIMENTS 008 AND 010
# =============================================================================

TARGET_BLOCKS = {
    "mlp": [6, 12],
    "linear_attn": [4, 8],
    "self_attn": [3, 7],
}


# =============================================================================
# METHODS
# =============================================================================

METHODS = [
    "baseline",
    "optimized",
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

    print(
        f"\nGPU MEMORY — {label}"
    )

    print(
        f"Allocated : "
        f"{allocated:.2f} MB"
    )

    print(
        f"Reserved  : "
        f"{reserved:.2f} MB"
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
# CLEANUP
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
    method,
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
    # STEP 1 — ORIGINAL SCALE
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
    # STEP 2 — FIXED THRESHOLD DECISION
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
    # STEP 3 — RECONSTRUCTION SCALE
    # =========================================================================

    if method == "baseline":

        reconstruction_alpha = (
            initial_alpha
        )

    elif method == "optimized":

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

    else:

        raise ValueError(
            f"Unknown method: "
            f"{method}"
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

    negative_count = (
        valid_codes < 0
    ).sum().item()

    zero_count = (
        valid_codes == 0
    ).sum().item()

    positive_count = (
        valid_codes > 0
    ).sum().item()


    # =========================================================================
    # SCALE STATISTICS
    # =========================================================================

    mean_initial_alpha = (
        initial_alpha
        .mean()
        .item()
    )

    mean_reconstruction_alpha = (
        reconstruction_alpha
        .mean()
        .item()
    )

    scale_ratio = (
        mean_reconstruction_alpha
        / mean_initial_alpha
        if mean_initial_alpha > 0
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
            negative_count,

        "zero":
            zero_count,

        "positive":
            positive_count,

        "mean_initial_alpha":
            mean_initial_alpha,

        "mean_reconstruction_alpha":
            mean_reconstruction_alpha,

        "scale_ratio":
            scale_ratio,
    }


# =============================================================================
# APPLY TERNARY QUANTIZATION
# =============================================================================

@torch.no_grad()
def ternarize_target_configuration(
    model,
    method,
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

        result = (
            groupwise_ternarize_weight(
                weight,
                method,
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

                "initial_alpha":
                    result[
                        "mean_initial_alpha"
                    ],

                "reconstruction_alpha":
                    result[
                        "mean_reconstruction_alpha"
                    ],

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

        del original_fp32
        del reconstructed_fp32
        del reconstructed
        del error
        del result


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

    weighted_scale_ratio = sum(
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
            weighted_error,

        "max_error":
            max_error,

        "negative":
            total_negative,

        "zero":
            total_zero,

        "positive":
            total_positive,

        "mean_scale_ratio":
            weighted_scale_ratio,
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

    start = time.perf_counter()

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
    "=" * 120
)

print(
    "EXPERIMENT 011 — "
    "Qwen3.5-0.8B "
    "Optimized Ternary Scale Calibration"
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
    f"Threshold multiplier   : "
    f"{THRESHOLD_MULTIPLIER}"
)

print(
    f"Methods                : "
    f"{METHODS}"
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
# DISCOVERY / FP16 REFERENCE
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
# METHOD COMPARISON
# =============================================================================

all_results = []


for test_index, method in enumerate(
    METHODS,
    start=1,
):

    print(
        "\n\n" + "#" * 120
    )

    print(
        f"METHOD TEST "
        f"{test_index}/"
        f"{len(METHODS)}"
        f" — {method.upper()}"
    )

    print(
        "#" * 120
    )


    # =========================================================================
    # FRESH MODEL
    # =========================================================================

    model = load_model()


    # =========================================================================
    # APPLY QUANTIZATION
    # =========================================================================

    stats = (
        ternarize_target_configuration(
            model,
            method,
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
        f"Method                : "
        f"{method}"
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
        f"Mean scale ratio      : "
        f"{stats['mean_scale_ratio']:.6f}"
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
    # MATRIX DETAILS
    # =========================================================================

    print(
        "\nMATRIX DETAILS"
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
            f" | Err "
            f"{item['relative_error']:.6f}"
            f" | Scale ratio "
            f"{item['scale_ratio']:.4f}"
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
    # STORE RESULT
    # =========================================================================

    all_results.append(
        {
            "method":
                method,

            "mean_error":
                stats["mean_error"],

            "max_error":
                stats["max_error"],

            "scale_ratio":
                stats[
                    "mean_scale_ratio"
                ],

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
        "ready for next method"
    )


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print(
    "\n\n" + "=" * 135
)

print(
    "EXPERIMENT 011 — "
    "OPTIMIZED TERNARY SCALE "
    "CALIBRATION SUMMARY"
)

print(
    "=" * 135
)

print(
    f"{'Method':<15}"
    f"{'Mean Err':>12}"
    f"{'Max Err':>12}"
    f"{'Scale Ratio':>14}"
    f"{'-1 %':>10}"
    f"{'0 %':>10}"
    f"{'+1 %':>10}"
    f"{'Tok/s':>12}"
)

print(
    "-" * 135
)

for result in all_results:

    print(
        f"{result['method']:<15}"
        f"{result['mean_error']:>12.6f}"
        f"{result['max_error']:>12.6f}"
        f"{result['scale_ratio']:>14.6f}"
        f"{result['negative_pct']:>10.2f}"
        f"{result['zero_pct']:>10.2f}"
        f"{result['positive_pct']:>10.2f}"
        f"{result['throughput']:>12.2f}"
    )


baseline_result = next(
    result
    for result
    in all_results
    if result["method"]
    == "baseline"
)

optimized_result = next(
    result
    for result
    in all_results
    if result["method"]
    == "optimized"
)


error_change = (
    (
        optimized_result[
            "mean_error"
        ]
        - baseline_result[
            "mean_error"
        ]
    )
    / baseline_result[
        "mean_error"
    ]
    * 100
)


print(
    "\nMean reconstruction "
    "error change:"
)

print(
    f"Baseline  : "
    f"{baseline_result['mean_error']:.6f}"
)

print(
    f"Optimized : "
    f"{optimized_result['mean_error']:.6f}"
)

print(
    f"Change    : "
    f"{error_change:+.2f}%"
)


print(
    "\nIMPORTANT:"
)

print(
    "The optimized scale uses exactly "
    "the same ternary {-1,0,+1} "
    "assignments as the baseline."
)

print(
    "Only the reconstruction magnitude "
    "of the non-zero ternary weights "
    "is changed."
)

print(
    "Therefore the {-1,0,+1} percentages "
    "should be identical between the "
    "two methods."
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
    "Experiment 011 changes only the "
    "reconstruction scale."
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
    "Codes            : "
    "{-1, 0, +1}"
)

print(
    "Baseline scale   : "
    "mean(abs(all group weights))"
)

print(
    "Optimized scale  : "
    "mean(abs(non-zero selected weights))"
)

print(
    "Runtime storage  : "
    "reconstructed FP16"
)

print()

print(
    "Target configuration is the "
    "MIXED_MEDIUM configuration used "
    "in Experiments 008 and 010."
)

print()

print(
    "A lower numerical reconstruction "
    "error does not automatically mean "
    "better functional generation."
)

print()

print(
    "Throughput remains a runtime "
    "reference and must NOT be treated "
    "as packed ternary inference speed."
)

print()

print(
    "The three prompts remain preliminary "
    "functional diagnostics rather than "
    "a formal quality benchmark."
)

print(
    "\n" + "=" * 120
)

print(
    "EXPERIMENT 011 COMPLETE"
)

print(
    "=" * 120
)
