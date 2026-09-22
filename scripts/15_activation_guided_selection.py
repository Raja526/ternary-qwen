import gc
import time
from collections import defaultdict

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
MAX_NEW_TOKENS = 60


# =============================================================================
# CALIBRATION SET
# =============================================================================
#
# Used ONLY to calculate activation-aware sensitivity.
#
# These prompts are intentionally different from the evaluation prompts.
# =============================================================================

CALIBRATION_PROMPTS = [
    "Describe what causes ocean tides.",
    "What is 14 multiplied by 27?",
    "A car travels 72 km per hour for 3 hours. How far does it travel?",
    "Explain why getting enough sleep is important.",
    "What is photosynthesis?",
    "What is 144 divided by 12?",
]


# =============================================================================
# EVALUATION SET
# =============================================================================
#
# These prompts are NOT used to construct the activation ranking.
# =============================================================================

EVALUATION_PROMPTS = [
    "Explain in simple terms why we see lightning before we hear thunder.",
    "What is 19 multiplied by 26?",
    "A cyclist travels 18 km per hour for 2.5 hours. How far does the cyclist travel?",
    "Explain in simple terms why drinking enough water is important.",
    "What is 225 divided by 15?",
]


# =============================================================================
# CANDIDATE POOL
# =============================================================================
#
# Exact LARGE boundary from Experiments 012–014.
#
# 30 Linear matrices
# 86,507,520 parameters
# 11.50% total model coverage
# =============================================================================

TARGET_BLOCKS = {
    "mlp": [6, 12, 18],
    "linear_attn": [4, 8, 12],
    "self_attn": [3, 7, 11],
}


# =============================================================================
# TARGET COVERAGE BUDGETS
# =============================================================================
#
# These are target fractions of TOTAL model parameters.
#
# Selection is matrix-granular, so exact coverage may differ slightly.
# =============================================================================

TARGET_COVERAGES = [
    7.67,
    9.00,
    10.50,
    11.50,
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
# MODEL MANAGEMENT
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


def is_candidate(
    name,
    module,
):

    component = get_component(
        name,
        module,
    )

    if component is None:
        return False

    block = get_block_index(
        name
    )

    if block is None:
        return False

    return (
        block
        in TARGET_BLOCKS.get(
            component,
            []
        )
    )


# =============================================================================
# OPTIMIZED TERNARY QUANTIZER
# =============================================================================
#
# IDENTICAL QUANTIZER TO EXPERIMENTS 011–014.
#
# Assignment:
#   initial_alpha = mean(abs(group))
#   threshold     = 0.5 * initial_alpha
#
# Reconstruction:
#   alpha_opt = mean(abs(non-zero selected weights))
#
# =============================================================================

@torch.no_grad()
def optimized_ternary_weight(
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


    initial_alpha = (
        groups
        .abs()
        .mean(
            dim=1,
            keepdim=True,
        )
    )


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


    reconstructed = (
        codes
        * reconstruction_alpha
    )


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
        "weight":
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
# DISCOVER MATRICES
# =============================================================================

def discover_candidates(model):

    candidates = {}

    for name, module in (
        model.named_modules()
    ):

        if not is_candidate(
            name,
            module,
        ):
            continue

        candidates[name] = {
            "module":
                module,

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

    return candidates


# =============================================================================
# PREPARE TERNARY WEIGHTS + WEIGHT ERROR
# =============================================================================

@torch.no_grad()
def prepare_candidates(
    candidates,
):

    print(
        "\nPreparing optimized ternary "
        "candidate weights..."
    )


    for name, info in (
        candidates.items()
    ):

        module = (
            info["module"]
        )


        original = (
            module.weight
            .detach()
            .float()
        )


        result = (
            optimized_ternary_weight(
                module.weight
            )
        )


        ternary = (
            result["weight"]
        )


        difference = (
            original
            - ternary.float()
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


        info[
            "weight_error"
        ] = relative_error

        info[
            "ternary_weight"
        ] = ternary

        info[
            "scale_ratio"
        ] = result[
            "scale_ratio"
        ]

        info[
            "negative"
        ] = result[
            "negative"
        ]

        info[
            "zero"
        ] = result[
            "zero"
        ]

        info[
            "positive"
        ] = result[
            "positive"
        ]


        del original
        del difference
        del result


    print(
        "Candidate preparation complete."
    )


# =============================================================================
# ACTIVATION ACCUMULATORS
# =============================================================================

activation_stats = defaultdict(
    lambda: {
        "original_sq": 0.0,
        "error_sq": 0.0,
        "samples": 0,
        "calls": 0,
    }
)


# =============================================================================
# ACTIVATION HOOK
# =============================================================================

def make_activation_hook(
    name,
    info,
):

    @torch.no_grad()
    def hook(
        module,
        inputs,
        output,
    ):

        if not inputs:
            return

        x = inputs[0]

        if x is None:
            return


        x_2d = (
            x
            .detach()
            .reshape(
                -1,
                x.shape[-1],
            )
            .float()
        )


        original_weight = (
            module.weight
            .detach()
            .float()
        )


        ternary_weight = (
            info[
                "ternary_weight"
            ]
            .float()
        )


        bias = None

        if module.bias is not None:

            bias = (
                module.bias
                .detach()
                .float()
            )


        y_original = F.linear(
            x_2d,
            original_weight,
            bias,
        )


        y_ternary = F.linear(
            x_2d,
            ternary_weight,
            bias,
        )


        delta = (
            y_original
            - y_ternary
        )


        stats = (
            activation_stats[
                name
            ]
        )


        stats[
            "original_sq"
        ] += (
            y_original
            .pow(2)
            .sum()
            .item()
        )


        stats[
            "error_sq"
        ] += (
            delta
            .pow(2)
            .sum()
            .item()
        )


        stats[
            "samples"
        ] += (
            x_2d.shape[0]
        )


        stats[
            "calls"
        ] += 1


        del x_2d
        del original_weight
        del ternary_weight
        del y_original
        del y_ternary
        del delta


    return hook


# =============================================================================
# CALIBRATION
# =============================================================================

@torch.no_grad()
def run_calibration(
    model,
    tokenizer,
):

    print(
        "\nRunning independent "
        "calibration set..."
    )


    for index, prompt in enumerate(
        CALIBRATION_PROMPTS,
        start=1,
    ):

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
        )


        inputs = {
            key: value.to(
                DEVICE
            )
            for key, value
            in inputs.items()
        }


        print(
            f"\nCalibration "
            f"{index}/"
            f"{len(CALIBRATION_PROMPTS)}"
        )

        print(
            f"  Prompt      : "
            f"{prompt}"
        )

        print(
            f"  Input tokens: "
            f"{inputs['input_ids'].shape[1]}"
        )


        outputs = model(
            **inputs,
            use_cache=False,
        )


        del outputs

        for value in (
            inputs.values()
        ):
            del value

        del inputs


# =============================================================================
# FINALIZE ACTIVATION SCORES
# =============================================================================

def finalize_activation_scores(
    candidates,
):

    results = []


    for name, info in (
        candidates.items()
    ):

        stats = (
            activation_stats[
                name
            ]
        )


        if (
            stats["original_sq"]
            > 0
        ):

            activation_error = (
                stats["error_sq"]
                / stats["original_sq"]
            ) ** 0.5

        else:

            activation_error = (
                float("inf")
            )


        total_codes = (
            info["negative"]
            + info["zero"]
            + info["positive"]
        )


        zero_pct = (
            info["zero"]
            / total_codes
            * 100
        )


        results.append(
            {
                "name":
                    name,

                "component":
                    info[
                        "component"
                    ],

                "block":
                    info[
                        "block"
                    ],

                "parameters":
                    info[
                        "parameters"
                    ],

                "weight_error":
                    info[
                        "weight_error"
                    ],

                "activation_error":
                    activation_error,

                "scale_ratio":
                    info[
                        "scale_ratio"
                    ],

                "zero_pct":
                    zero_pct,

                "calls":
                    stats[
                        "calls"
                    ],
            }
        )


    return results


# =============================================================================
# PARAMETER-BUDGET SELECTION
# =============================================================================

def build_activation_guided_selection(
    ranked_candidates,
    target_parameters,
):

    selected = []

    selected_parameters = 0


    for item in ranked_candidates:

        if (
            selected_parameters
            >= target_parameters
        ):
            break


        selected.append(
            item
        )


        selected_parameters += (
            item[
                "parameters"
            ]
        )


    return selected


# =============================================================================
# APPLY SELECTED MATRICES
# =============================================================================

@torch.no_grad()
def apply_selection(
    model,
    selected_names,
):

    affected_parameters = 0

    affected_matrices = 0

    total_negative = 0
    total_zero = 0
    total_positive = 0

    weight_errors = []

    scale_ratios = []


    for name, module in (
        model.named_modules()
    ):

        if (
            name
            not in selected_names
        ):
            continue


        result = (
            optimized_ternary_weight(
                module.weight
            )
        )


        original = (
            module.weight
            .detach()
            .float()
        )


        ternary_fp32 = (
            result[
                "weight"
            ]
            .float()
        )


        difference = (
            original
            - ternary_fp32
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
            module.weight.numel()
        )


        module.weight.copy_(
            result["weight"]
        )


        affected_parameters += (
            parameters
        )

        affected_matrices += 1


        total_negative += (
            result["negative"]
        )

        total_zero += (
            result["zero"]
        )

        total_positive += (
            result["positive"]
        )


        weight_errors.append(
            (
                relative_error,
                parameters,
            )
        )


        scale_ratios.append(
            (
                result[
                    "scale_ratio"
                ],
                parameters,
            )
        )


        del original
        del ternary_fp32
        del difference
        del result


    mean_weight_error = (
        sum(
            error * params
            for error, params
            in weight_errors
        )
        / affected_parameters
    )


    mean_scale_ratio = (
        sum(
            ratio * params
            for ratio, params
            in scale_ratios
        )
        / affected_parameters
    )


    max_weight_error = max(
        error
        for error, _
        in weight_errors
    )


    return {
        "matrices":
            affected_matrices,

        "parameters":
            affected_parameters,

        "mean_weight_error":
            mean_weight_error,

        "max_weight_error":
            max_weight_error,

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
        key: value.to(
            DEVICE
        )
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
    "=" * 145
)

print(
    "EXPERIMENT 015 — "
    "Qwen3.5-0.8B Activation-Guided "
    "Matrix-Level Ternary Selection"
)

print(
    "=" * 145
)


print(
    "\nConfiguration"
)

print(
    "-" * 145
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
    f"Calibration prompts    : "
    f"{len(CALIBRATION_PROMPTS)}"
)

print(
    f"Evaluation prompts     : "
    f"{len(EVALUATION_PROMPTS)}"
)

print(
    f"Maximum new tokens     : "
    f"{MAX_NEW_TOKENS}"
)


# =============================================================================
# ENVIRONMENT
# =============================================================================

print(
    "\nEnvironment"
)

print(
    "-" * 145
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
# PHASE 1 — ACTIVATION-AWARE RANKING
# =============================================================================

print(
    "\n\n" + "=" * 145
)

print(
    "PHASE 1 — BUILD ACTIVATION-AWARE "
    "MATRIX RANKING"
)

print(
    "=" * 145
)


ranking_model = (
    load_model()
)


TOTAL_PARAMS = sum(
    parameter.numel()
    for parameter
    in ranking_model.parameters()
)


print(
    f"\nTotal model parameters : "
    f"{TOTAL_PARAMS:,}"
)


candidates = (
    discover_candidates(
        ranking_model
    )
)


candidate_parameters = sum(
    item["parameters"]
    for item
    in candidates.values()
)


candidate_coverage = (
    candidate_parameters
    / TOTAL_PARAMS
    * 100
)


print(
    "\nCANDIDATE POOL"
)

print(
    "-" * 145
)


print(
    f"Candidate matrices     : "
    f"{len(candidates)}"
)

print(
    f"Candidate parameters   : "
    f"{candidate_parameters:,}"
)

print(
    f"Candidate coverage     : "
    f"{candidate_coverage:.2f}%"
)


if len(candidates) != 30:

    raise RuntimeError(
        "Expected 30 candidate matrices."
    )


if (
    candidate_parameters
    != 86_507_520
):

    raise RuntimeError(
        "Candidate pool does not match "
        "Experiment 014."
    )


prepare_candidates(
    candidates
)


# =============================================================================
# REGISTER HOOKS
# =============================================================================

handles = []


for name, info in (
    candidates.items()
):

    handle = (
        info[
            "module"
        ]
        .register_forward_hook(
            make_activation_hook(
                name,
                info,
            )
        )
    )

    handles.append(
        handle
    )


print(
    f"\nRegistered "
    f"{len(handles)} "
    f"activation hooks."
)


# =============================================================================
# RUN CALIBRATION
# =============================================================================

run_calibration(
    ranking_model,
    tokenizer,
)


# =============================================================================
# REMOVE HOOKS
# =============================================================================

for handle in handles:

    handle.remove()


handles.clear()


# =============================================================================
# FINALIZE RANKING
# =============================================================================

activation_results = (
    finalize_activation_scores(
        candidates
    )
)


# Lower activation error = preferred
ranked_candidates = sorted(
    activation_results,
    key=lambda item: (
        item[
            "activation_error"
        ],
        item[
            "weight_error"
        ],
        item[
            "name"
        ],
    ),
)


print(
    "\n\n" + "=" * 160
)

print(
    "ACTIVATION-GUIDED MATRIX RANKING "
    "(LOWEST DISTORTION FIRST)"
)

print(
    "=" * 160
)


print(
    f"{'Rank':>4} "
    f"{'Component':<15}"
    f"{'Block':>7} "
    f"{'Parameters':>12} "
    f"{'ActErr':>11} "
    f"{'WeightErr':>11} "
    f"{'Scale':>9} "
    f"{'Zero%':>8} "
    f"Name"
)


print(
    "-" * 160
)


for rank, item in enumerate(
    ranked_candidates,
    start=1,
):

    print(
        f"{rank:>4} "
        f"{item['component']:<15}"
        f"{item['block']:>7} "
        f"{item['parameters']:>12,} "
        f"{item['activation_error']:>11.6f} "
        f"{item['weight_error']:>11.6f} "
        f"{item['scale_ratio']:>9.4f} "
        f"{item['zero_pct']:>8.2f} "
        f"{item['name']}"
    )


# =============================================================================
# BUILD BUDGETS
# =============================================================================

selection_cases = []


print(
    "\n\n" + "=" * 145
)

print(
    "ACTIVATION-GUIDED PARAMETER BUDGETS"
)

print(
    "=" * 145
)


for target_coverage in (
    TARGET_COVERAGES
):

    target_parameters = int(
        TOTAL_PARAMS
        * target_coverage
        / 100
    )


    selected = (
        build_activation_guided_selection(
            ranked_candidates,
            target_parameters,
        )
    )


    selected_parameters = sum(
        item["parameters"]
        for item
        in selected
    )


    actual_coverage = (
        selected_parameters
        / TOTAL_PARAMS
        * 100
    )


    case_name = (
        "ACT_GUIDED_"
        + str(
            target_coverage
        ).replace(
            ".",
            "_",
        )
    )


    selection_cases.append(
        {
            "name":
                case_name,

            "target_coverage":
                target_coverage,

            "actual_coverage":
                actual_coverage,

            "parameters":
                selected_parameters,

            "selected":
                selected,
        }
    )


    print(
        f"\n{case_name}"
    )

    print(
        f"  Target coverage : "
        f"{target_coverage:.2f}%"
    )

    print(
        f"  Actual coverage : "
        f"{actual_coverage:.2f}%"
    )

    print(
        f"  Matrices        : "
        f"{len(selected)}"
    )

    print(
        f"  Parameters      : "
        f"{selected_parameters:,}"
    )


    print(
        "  Selected:"
    )


    for item in selected:

        print(
            f"    "
            f"{item['component']:<15}"
            f" block "
            f"{item['block']:>2}"
            f" | ActErr "
            f"{item['activation_error']:.6f}"
            f" | "
            f"{item['name']}"
        )


# =============================================================================
# DELETE TEMPORARY TERNARY WEIGHTS
# =============================================================================

for info in (
    candidates.values()
):

    if "ternary_weight" in info:

        del info[
            "ternary_weight"
        ]


release_model(
    ranking_model
)

del ranking_model
del candidates

gc.collect()
torch.cuda.empty_cache()


# =============================================================================
# PHASE 2 — FP16 REFERENCE ON UNSEEN EVALUATION SET
# =============================================================================

print(
    "\n\n" + "=" * 145
)

print(
    "PHASE 2 — FP16 REFERENCE "
    "ON UNSEEN EVALUATION SET"
)

print(
    "=" * 145
)


reference_model = (
    load_model()
)


for index, prompt in enumerate(
    EVALUATION_PROMPTS,
    start=1,
):

    result = generate(
        reference_model,
        tokenizer,
        prompt,
    )


    print(
        f"\nEVALUATION PROMPT "
        f"{index}"
    )

    print(
        "-" * 145
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
# PHASE 3 — TEST ACTIVATION-GUIDED CONFIGURATIONS
# =============================================================================

all_results = []


for case_index, case in enumerate(
    selection_cases,
    start=1,
):

    print(
        "\n\n" + "#" * 145
    )

    print(
        f"ACTIVATION-GUIDED TEST "
        f"{case_index}/"
        f"{len(selection_cases)}"
        f" — {case['name']}"
    )

    print(
        "#" * 145
    )


    selected_names = {
        item["name"]
        for item
        in case["selected"]
    }


    model = load_model()


    stats = apply_selection(
        model,
        selected_names,
    )


    actual_coverage = (
        stats["parameters"]
        / TOTAL_PARAMS
        * 100
    )


    if (
        stats["parameters"]
        != case["parameters"]
    ):

        raise RuntimeError(
            f"{case['name']}: "
            "applied parameter count "
            "does not match selection."
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
        "\nTERNARY CONFIGURATION"
    )

    print(
        "-" * 145
    )


    print(
        f"Case                  : "
        f"{case['name']}"
    )

    print(
        f"Target coverage       : "
        f"{case['target_coverage']:.2f}%"
    )

    print(
        f"Actual coverage       : "
        f"{actual_coverage:.2f}%"
    )

    print(
        f"Affected matrices     : "
        f"{stats['matrices']}"
    )

    print(
        f"Affected parameters   : "
        f"{stats['parameters']:,}"
    )

    print(
        f"Mean weight error     : "
        f"{stats['mean_weight_error']:.6f}"
    )

    print(
        f"Max weight error      : "
        f"{stats['max_weight_error']:.6f}"
    )

    print(
        f"Mean scale ratio      : "
        f"{stats['scale_ratio']:.6f}"
    )

    print(
        f"-1 codes              : "
        f"{negative_pct:.2f}%"
    )

    print(
        f" 0 codes              : "
        f"{zero_pct:.2f}%"
    )

    print(
        f"+1 codes              : "
        f"{positive_pct:.2f}%"
    )


    print(
        "\nSELECTED MATRICES"
    )

    print(
        "-" * 145
    )


    for rank, item in enumerate(
        case["selected"],
        start=1,
    ):

        print(
            f"{rank:>2}. "
            f"{item['component']:<15}"
            f" block "
            f"{item['block']:>2}"
            f" | ActErr "
            f"{item['activation_error']:.6f}"
            f" | "
            f"{item['name']}"
        )


    # =========================================================================
    # UNSEEN EVALUATION
    # =========================================================================

    print(
        "\nUNSEEN EVALUATION"
    )

    print(
        "-" * 145
    )


    total_tokens = 0
    total_time = 0.0


    for prompt_index, prompt in enumerate(
        EVALUATION_PROMPTS,
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


    all_results.append(
        {
            "name":
                case["name"],

            "target_coverage":
                case[
                    "target_coverage"
                ],

            "actual_coverage":
                actual_coverage,

            "matrices":
                stats["matrices"],

            "parameters":
                stats["parameters"],

            "mean_weight_error":
                stats[
                    "mean_weight_error"
                ],

            "max_weight_error":
                stats[
                    "max_weight_error"
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


    release_model(
        model
    )

    del model
    del stats

    gc.collect()
    torch.cuda.empty_cache()


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print(
    "\n\n" + "=" * 165
)

print(
    "EXPERIMENT 015 — "
    "ACTIVATION-GUIDED SELECTION SUMMARY"
)

print(
    "=" * 165
)


print(
    f"{'Case':<24}"
    f"{'Target':>10}"
    f"{'Actual':>10}"
    f"{'Matrices':>10}"
    f"{'Parameters':>16}"
    f"{'MeanErr':>12}"
    f"{'MaxErr':>12}"
    f"{'Scale':>10}"
    f"{'Zero%':>10}"
    f"{'Tok/s':>10}"
)


print(
    "-" * 165
)


for result in all_results:

    print(
        f"{result['name']:<24}"
        f"{result['target_coverage']:>9.2f}%"
        f"{result['actual_coverage']:>9.2f}%"
        f"{result['matrices']:>10}"
        f"{result['parameters']:>16,}"
        f"{result['mean_weight_error']:>12.6f}"
        f"{result['max_weight_error']:>12.6f}"
        f"{result['scale_ratio']:>10.4f}"
        f"{result['zero_pct']:>10.2f}"
        f"{result['throughput']:>10.2f}"
    )


# =============================================================================
# SCIENTIFIC NOTES
# =============================================================================

print(
    "\n" + "=" * 145
)

print(
    "SCIENTIFIC INTERPRETATION NOTES"
)

print(
    "=" * 145
)


print(
    "1. Matrix ranking was built using "
    "calibration prompts that are separate "
    "from the evaluation prompts."
)

print()

print(
    "2. Lower activation distortion was "
    "used as the matrix-selection criterion."
)

print()

print(
    "3. The optimized ternary quantizer "
    "itself was unchanged."
)

print()

print(
    "4. Coverage is controlled by total "
    "selected parameters, but exact coverage "
    "can differ from the requested budget "
    "because matrices are indivisible."
)

print()

print(
    "5. Generation quality must be analyzed "
    "from the unseen evaluation outputs."
)

print()

print(
    "6. Throughput is NOT packed ternary "
    "speed because selected weights are "
    "reconstructed into FP16 tensors."
)

print()

print(
    "7. A successful result would show that "
    "activation-guided matrix selection can "
    "maintain useful generation at coverage "
    "where block-level selection degraded."
)


print(
    "\n" + "=" * 145
)

print(
    "EXPERIMENT 015 COMPLETE"
)

print(
    "=" * 145
)
