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

# Keep calibration deterministic and small for the first activation experiment.
CALIBRATION_PROMPTS = [
    "Explain in simple terms why the sky is blue.",
    "What is 17 multiplied by 23?",
    "If a train travels 60 km per hour for 2.5 hours, how far does it travel?",
    "Write a short paragraph explaining why exercise is important.",
]

# Complete LARGE boundary from Experiments 012/013.
TARGET_BLOCKS = {
    "mlp": [6, 12, 18],
    "linear_attn": [4, 8, 12],
    "self_attn": [3, 7, 11],
}

# These are the three additions tested in Experiment 013.
BOUNDARY_ADDITIONS = {
    ("mlp", 18),
    ("linear_attn", 12),
    ("self_attn", 11),
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
# MODEL
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
# OPTIMIZED TERNARY QUANTIZER — IDENTICAL TO EXPERIMENTS 011–013
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


    # -------------------------------------------------------------------------
    # Initial scale
    # -------------------------------------------------------------------------

    initial_alpha = (
        groups
        .abs()
        .mean(
            dim=1,
            keepdim=True,
        )
    )


    # -------------------------------------------------------------------------
    # Fixed threshold
    # -------------------------------------------------------------------------

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


    # -------------------------------------------------------------------------
    # Optimized non-zero reconstruction scale
    # -------------------------------------------------------------------------

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
# DISCOVER TARGET MATRICES
# =============================================================================

def discover_targets(model):

    targets = {}

    for name, module in (
        model.named_modules()
    ):

        if not is_target_matrix(
            name,
            module,
        ):
            continue

        component = get_component(
            name,
            module,
        )

        block = get_block_index(
            name
        )

        targets[name] = {
            "module":
                module,

            "component":
                component,

            "block":
                block,

            "parameters":
                module.weight.numel(),

            "boundary_addition":
                (
                    component,
                    block,
                )
                in BOUNDARY_ADDITIONS,
        }

    return targets


# =============================================================================
# WEIGHT-SPACE METRICS
# =============================================================================

@torch.no_grad()
def prepare_ternary_candidates(
    targets,
):

    for name, info in (
        targets.items()
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

        ternary_fp32 = (
            ternary.float()
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
        del ternary_fp32
        del difference
        del result


# =============================================================================
# ACTIVATION METRIC ACCUMULATORS
# =============================================================================
#
# We accumulate squared norms rather than storing all activations.
#
# For each layer:
#
#   original_sq = sum ||Y||^2
#   error_sq    = sum ||Y - Y_t||^2
#
# Final metric:
#
#   sqrt(error_sq / original_sq)
#
# This is preferable to averaging per-token ratios because it produces one
# global relative L2 distortion across the calibration set.
#
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
# HOOK FACTORY
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


        # ---------------------------------------------------------------------
        # Flatten all token/batch dimensions.
        # Keep the hidden/input dimension.
        # ---------------------------------------------------------------------

        x_2d = (
            x
            .detach()
            .reshape(
                -1,
                x.shape[-1],
            )
        )


        # ---------------------------------------------------------------------
        # Use FP32 for metric calculation.
        # ---------------------------------------------------------------------

        x_fp32 = (
            x_2d.float()
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


        # ---------------------------------------------------------------------
        # Original and ternary outputs
        # ---------------------------------------------------------------------

        y_original = F.linear(
            x_fp32,
            original_weight,
            bias,
        )


        y_ternary = F.linear(
            x_fp32,
            ternary_weight,
            bias,
        )


        delta = (
            y_original
            - y_ternary
        )


        original_sq = (
            y_original
            .pow(2)
            .sum()
            .item()
        )


        error_sq = (
            delta
            .pow(2)
            .sum()
            .item()
        )


        stats = (
            activation_stats[name]
        )


        stats[
            "original_sq"
        ] += original_sq


        stats[
            "error_sq"
        ] += error_sq


        stats[
            "samples"
        ] += x_2d.shape[0]


        stats[
            "calls"
        ] += 1


        del x_2d
        del x_fp32
        del original_weight
        del ternary_weight
        del y_original
        del y_ternary
        del delta


    return hook


# =============================================================================
# RUN CALIBRATION
# =============================================================================

@torch.no_grad()
def run_calibration(
    model,
    tokenizer,
):

    print(
        "\nRunning calibration prompts..."
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
            key: value.to(DEVICE)
            for key, value
            in inputs.items()
        }


        print(
            f"\nCalibration prompt "
            f"{index}/"
            f"{len(CALIBRATION_PROMPTS)}"
        )

        print(
            f"  {prompt}"
        )

        print(
            f"  Input tokens: "
            f"{inputs['input_ids'].shape[1]}"
        )


        # Important:
        # We only need the normal forward pass.
        # No generation is needed for activation calibration.

        _ = model(
            **inputs,
            use_cache=False,
        )


        del _

        for value in (
            inputs.values()
        ):
            del value

        del inputs


# =============================================================================
# FINALIZE ACTIVATION METRICS
# =============================================================================

def finalize_metrics(
    targets,
):

    results = []


    for name, info in (
        targets.items()
    ):

        stats = (
            activation_stats[
                name
            ]
        )


        original_sq = (
            stats[
                "original_sq"
            ]
        )


        error_sq = (
            stats[
                "error_sq"
            ]
        )


        if original_sq > 0:

            activation_error = (
                error_sq
                / original_sq
            ) ** 0.5

        else:

            activation_error = (
                float("nan")
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

                "boundary_addition":
                    info[
                        "boundary_addition"
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

                "samples":
                    stats[
                        "samples"
                    ],

                "calls":
                    stats[
                        "calls"
                    ],
            }
        )


    return results


# =============================================================================
# CORRELATION HELPER
# =============================================================================

def pearson_correlation(
    xs,
    ys,
):

    x = torch.tensor(
        xs,
        dtype=torch.float64,
    )

    y = torch.tensor(
        ys,
        dtype=torch.float64,
    )


    x = (
        x
        - x.mean()
    )

    y = (
        y
        - y.mean()
    )


    denominator = (
        torch.sqrt(
            (x ** 2).sum()
            * (y ** 2).sum()
        )
    )


    if denominator.item() == 0:

        return float("nan")


    return (
        (
            x * y
        ).sum()
        / denominator
    ).item()


# =============================================================================
# HEADER
# =============================================================================

print(
    "=" * 135
)

print(
    "EXPERIMENT 014 — "
    "Qwen3.5-0.8B Activation-Aware "
    "Ternary Sensitivity"
)

print(
    "=" * 135
)


print(
    "\nConfiguration"
)

print(
    "-" * 135
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
    "\nTarget blocks:"
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
    "-" * 135
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
# TOKENIZER + MODEL
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


model = load_model()


TOTAL_PARAMS = sum(
    parameter.numel()
    for parameter
    in model.parameters()
)


print(
    f"\nTotal model parameters : "
    f"{TOTAL_PARAMS:,}"
)


# =============================================================================
# DISCOVER TARGETS
# =============================================================================

targets = (
    discover_targets(
        model
    )
)


TARGET_PARAMS = sum(
    info["parameters"]
    for info
    in targets.values()
)


TARGET_COVERAGE = (
    TARGET_PARAMS
    / TOTAL_PARAMS
    * 100
)


print(
    "\nTARGET VALIDATION"
)

print(
    "-" * 135
)


print(
    f"Target Linear matrices : "
    f"{len(targets)}"
)

print(
    f"Target parameters      : "
    f"{TARGET_PARAMS:,}"
)

print(
    f"Target model coverage  : "
    f"{TARGET_COVERAGE:.2f}%"
)


if len(targets) != 30:

    raise RuntimeError(
        "Expected 30 Linear matrices "
        f"for LARGE boundary, found "
        f"{len(targets)}."
    )


if TARGET_PARAMS != 86_507_520:

    raise RuntimeError(
        "Expected 86,507,520 target "
        f"parameters, found "
        f"{TARGET_PARAMS:,}."
    )


print(
    "\nBoundary additions:"
)


for name, info in (
    targets.items()
):

    if info[
        "boundary_addition"
    ]:

        print(
            f"  {info['component']:<15}"
            f" block "
            f"{info['block']:>2}"
            f" | "
            f"{name}"
        )


# =============================================================================
# PREPARE TERNARY CANDIDATES
# =============================================================================

print(
    "\nPreparing optimized ternary "
    "candidate weights..."
)


prepare_ternary_candidates(
    targets
)


print(
    "Ternary candidates prepared."
)


# =============================================================================
# REGISTER HOOKS
# =============================================================================

print(
    "\nRegistering activation hooks..."
)


handles = []


for name, info in (
    targets.items()
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
    f"Registered "
    f"{len(handles)} hooks."
)


# =============================================================================
# CALIBRATION
# =============================================================================

run_calibration(
    model,
    tokenizer,
)


# =============================================================================
# REMOVE HOOKS
# =============================================================================

for handle in handles:

    handle.remove()


handles.clear()


# =============================================================================
# FINALIZE
# =============================================================================

results = (
    finalize_metrics(
        targets
    )
)


# =============================================================================
# TABLE 1 — ACTIVATION DISTORTION RANKING
# =============================================================================

ranked_activation = sorted(
    results,
    key=lambda item:
        item[
            "activation_error"
        ],
    reverse=True,
)


print(
    "\n\n" + "=" * 155
)

print(
    "ACTIVATION-AWARE SENSITIVITY RANKING "
    "(HIGHER = MORE SENSITIVE)"
)

print(
    "=" * 155
)


print(
    f"{'Rank':>4} "
    f"{'Component':<15}"
    f"{'Block':>7} "
    f"{'Boundary':>9} "
    f"{'Parameters':>12} "
    f"{'WeightErr':>11} "
    f"{'ActErr':>11} "
    f"{'Scale':>9} "
    f"{'Zero%':>8} "
    f"{'Calls':>7} "
    f"Name"
)


print(
    "-" * 155
)


for rank, item in enumerate(
    ranked_activation,
    start=1,
):

    boundary = (
        "YES"
        if item[
            "boundary_addition"
        ]
        else "NO"
    )


    print(
        f"{rank:>4} "
        f"{item['component']:<15}"
        f"{item['block']:>7} "
        f"{boundary:>9} "
        f"{item['parameters']:>12,} "
        f"{item['weight_error']:>11.6f} "
        f"{item['activation_error']:>11.6f} "
        f"{item['scale_ratio']:>9.4f} "
        f"{item['zero_pct']:>8.2f} "
        f"{item['calls']:>7} "
        f"{item['name']}"
    )


# =============================================================================
# TABLE 2 — WEIGHT ERROR RANKING
# =============================================================================

ranked_weight = sorted(
    results,
    key=lambda item:
        item[
            "weight_error"
        ],
    reverse=True,
)


print(
    "\n\n" + "=" * 155
)

print(
    "WEIGHT-SPACE RECONSTRUCTION ERROR "
    "RANKING"
)

print(
    "=" * 155
)


print(
    f"{'Rank':>4} "
    f"{'Component':<15}"
    f"{'Block':>7} "
    f"{'Boundary':>9} "
    f"{'WeightErr':>11} "
    f"{'ActErr':>11} "
    f"Name"
)


print(
    "-" * 155
)


for rank, item in enumerate(
    ranked_weight,
    start=1,
):

    boundary = (
        "YES"
        if item[
            "boundary_addition"
        ]
        else "NO"
    )


    print(
        f"{rank:>4} "
        f"{item['component']:<15}"
        f"{item['block']:>7} "
        f"{boundary:>9} "
        f"{item['weight_error']:>11.6f} "
        f"{item['activation_error']:>11.6f} "
        f"{item['name']}"
    )


# =============================================================================
# BOUNDARY ADDITION SUMMARY
# =============================================================================

print(
    "\n\n" + "=" * 135
)

print(
    "EXPERIMENT 013 BOUNDARY ADDITIONS — "
    "ACTIVATION METRICS"
)

print(
    "=" * 135
)


boundary_results = [
    item
    for item in results
    if item[
        "boundary_addition"
    ]
]


boundary_results = sorted(
    boundary_results,
    key=lambda item: (
        item["component"],
        item["block"],
        item["name"],
    )
)


for item in boundary_results:

    print(
        f"{item['component']:<15}"
        f" block "
        f"{item['block']:>2}"
        f" | "
        f"{item['name']:<60}"
        f" | WeightErr "
        f"{item['weight_error']:.6f}"
        f" | ActErr "
        f"{item['activation_error']:.6f}"
    )


# =============================================================================
# COMPONENT SUMMARY
# =============================================================================

print(
    "\n\n" + "=" * 135
)

print(
    "COMPONENT ACTIVATION-DISTORTION SUMMARY"
)

print(
    "=" * 135
)


for component in [
    "mlp",
    "linear_attn",
    "self_attn",
]:

    component_results = [
        item
        for item in results
        if item[
            "component"
        ] == component
    ]


    if not component_results:
        continue


    mean_weight = sum(
        item["weight_error"]
        for item
        in component_results
    ) / len(
        component_results
    )


    mean_activation = sum(
        item["activation_error"]
        for item
        in component_results
    ) / len(
        component_results
    )


    max_activation_item = max(
        component_results,
        key=lambda item:
            item[
                "activation_error"
            ],
    )


    print(
        f"\n{component}"
    )

    print(
        f"  Matrices             : "
        f"{len(component_results)}"
    )

    print(
        f"  Mean weight error    : "
        f"{mean_weight:.6f}"
    )

    print(
        f"  Mean activation error: "
        f"{mean_activation:.6f}"
    )

    print(
        f"  Highest activation   : "
        f"{max_activation_item['activation_error']:.6f}"
    )

    print(
        f"  Highest matrix       : "
        f"{max_activation_item['name']}"
    )


# =============================================================================
# CORRELATION
# =============================================================================

weight_errors = [
    item["weight_error"]
    for item in results
]


activation_errors = [
    item["activation_error"]
    for item in results
]


correlation = (
    pearson_correlation(
        weight_errors,
        activation_errors,
    )
)


print(
    "\n\n" + "=" * 135
)

print(
    "WEIGHT ERROR vs ACTIVATION ERROR"
)

print(
    "=" * 135
)


print(
    f"Pearson correlation : "
    f"{correlation:.6f}"
)


print(
    "\nInterpret carefully:"
)

print(
    "A high correlation would mean the two "
    "metrics often rank matrices similarly."
)

print(
    "A low correlation would show that "
    "activation-aware sensitivity captures "
    "information not visible from weight "
    "reconstruction error alone."
)


# =============================================================================
# TOP 10 COMPARISON
# =============================================================================

print(
    "\n\n" + "=" * 135
)

print(
    "TOP 10 ACTIVATION-SENSITIVE MATRICES"
)

print(
    "=" * 135
)


for rank, item in enumerate(
    ranked_activation[:10],
    start=1,
):

    print(
        f"{rank:>2}. "
        f"{item['component']:<15}"
        f" block "
        f"{item['block']:>2}"
        f" | ActErr "
        f"{item['activation_error']:.6f}"
        f" | WeightErr "
        f"{item['weight_error']:.6f}"
        f" | "
        f"{item['name']}"
    )


# =============================================================================
# CLEANUP STORED TERNARY WEIGHTS
# =============================================================================

print(
    "\nCleaning temporary ternary "
    "candidate tensors..."
)


for info in targets.values():

    if "ternary_weight" in info:

        del info[
            "ternary_weight"
        ]


gc.collect()
torch.cuda.empty_cache()


# =============================================================================
# IMPORTANT INTERPRETATION
# =============================================================================

print(
    "\n\n" + "=" * 135
)

print(
    "IMPORTANT INTERPRETATION"
)

print(
    "=" * 135
)


print(
    "Experiment 014 does NOT modify "
    "the model weights during calibration."
)

print()

print(
    "The FP16 model supplies the real "
    "input activations seen by each "
    "target Linear matrix."
)

print()

print(
    "For those same activations, the "
    "experiment compares the output of "
    "the original FP16 weight against "
    "its optimized ternary candidate."
)

print()

print(
    "Activation distortion:"
)

print(
    "  ||Y_fp16 - Y_ternary||_2"
)

print(
    "  --------------------------"
)

print(
    "       ||Y_fp16||_2"
)

print()

print(
    "This metric measures local output "
    "distortion under representative "
    "activations."
)

print()

print(
    "It is NOT yet a proof that the "
    "activation metric predicts final "
    "generation quality."
)

print()

print(
    "The calibration set contains only "
    f"{len(CALIBRATION_PROMPTS)} prompts."
)

print()

print(
    "Experiment 014 should therefore "
    "be interpreted as a sensitivity-"
    "metric study, not a formal model "
    "quality benchmark."
)


# =============================================================================
# MODEL CLEANUP
# =============================================================================

release_model(
    model
)

del model

gc.collect()
torch.cuda.empty_cache()


print(
    "\n" + "=" * 135
)

print(
    "EXPERIMENT 014 COMPLETE"
)

print(
    "=" * 135
)
