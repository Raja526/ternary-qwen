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

# Shorter generation because many variants are tested.
MAX_NEW_TOKENS = 40

PROMPTS = [
    "Explain in simple terms why the sky is blue.",
    "What is 17 multiplied by 23?",
    "If a train travels 60 km per hour for 2.5 hours, how far does it travel?",
]

# Progressive number of transformer blocks.
PROGRESSIVE_TESTS = {
    "mlp": [1, 2, 4, 8, 12, 24],
    "linear_attn": [1, 2, 4, 8, 12, 18],
    "self_attn": [1, 2, 4, 6],
}


# =============================================================================
# CUDA MEMORY INFO
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
        free_bytes / (1024 ** 2)
    )

    total_mb = (
        total_bytes / (1024 ** 2)
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

    # Clean unused cached blocks before loading.
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

    # Explicitly move parameters away from GPU.
    model.to("cpu")

    del model

    # Collect Python objects/reference cycles.
    gc.collect()
    gc.collect()

    # Release unused PyTorch CUDA cache.
    torch.cuda.empty_cache()

    # Collect CUDA IPC allocations if any.
    torch.cuda.ipc_collect()

    torch.cuda.synchronize()

    print_gpu_memory(
        "after model cleanup"
    )


# =============================================================================
# LAYER IDENTIFICATION
# =============================================================================

def get_block_index(name):

    parts = name.split(".")

    try:

        layer_position = (
            parts.index("layers")
        )

        return int(
            parts[
                layer_position + 1
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

    name_lower = (
        name.lower()
    )

    if ".mlp." in name_lower:
        return "mlp"

    if ".linear_attn." in name_lower:
        return "linear_attn"

    if ".self_attn." in name_lower:
        return "self_attn"

    return None


def discover_component_blocks(
    model,
):

    result = {
        "mlp": set(),
        "linear_attn": set(),
        "self_attn": set(),
    }

    for name, module in (
        model.named_modules()
    ):

        component = get_component(
            name,
            module,
        )

        if component is None:
            continue

        block = get_block_index(
            name
        )

        if block is not None:

            result[
                component
            ].add(
                block
            )

    return {
        key: sorted(value)
        for key, value
        in result.items()
    }


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

    # Delete temporary CUDA tensors before returning.
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

    # Explicitly move only tensors to GPU.
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

    # Decode before deleting output tensor.
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

    # Explicit cleanup of per-generation CUDA tensors.
    del output

    for value in inputs.values():
        del value

    del inputs

    return {
        "text": text,
        "tokens": generated_tokens,
        "time": elapsed,
        "speed": speed,
    }


# =============================================================================
# TERNARIZE SELECTED BLOCKS
# =============================================================================

@torch.no_grad()
def ternarize_selected_blocks(
    model,
    target_component,
    selected_blocks,
):

    selected_blocks = set(
        selected_blocks
    )

    affected_linear_layers = 0
    affected_parameters = 0

    relative_errors = []

    for name, module in (
        model.named_modules()
    ):

        component = (
            get_component(
                name,
                module,
            )
        )

        if (
            component
            != target_component
        ):
            continue

        block = (
            get_block_index(
                name
            )
        )

        if (
            block
            not in selected_blocks
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

        affected_linear_layers += 1

        affected_parameters += (
            weight.numel()
        )

        relative_errors.append(
            relative_error
        )

        # Important:
        # release temporary FP32 tensors immediately.
        del original_fp32
        del reconstructed_fp32
        del ternary_weight
        del error

    if relative_errors:

        mean_error = (
            sum(relative_errors)
            / len(relative_errors)
        )

        max_error = (
            max(relative_errors)
        )

    else:

        mean_error = 0.0
        max_error = 0.0

    return {
        "linear_layers":
            affected_linear_layers,

        "parameters":
            affected_parameters,

        "mean_error":
            mean_error,

        "max_error":
            max_error,
    }


# =============================================================================
# HEADER
# =============================================================================

print(
    "=" * 100
)

print(
    "EXPERIMENT 006 — "
    "Qwen3.5-0.8B "
    "Progressive Layer Sensitivity"
)

print(
    "=" * 100
)

print(
    "\nConfiguration"
)

print(
    "-" * 100
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


# =============================================================================
# ENVIRONMENT
# =============================================================================

print(
    "\nEnvironment"
)

print(
    "-" * 100
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
    torch.cuda.get_device_properties(
        0
    ).total_memory
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
    "-" * 100
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


# =============================================================================
# FP16 REFERENCE
# =============================================================================

print(
    "\n" + "=" * 100
)

print(
    "FP16 REFERENCE"
)

print(
    "=" * 100
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
        "-" * 100
    )

    print(
        result["text"]
    )

    print(
        f"\nTokens/sec : "
        f"{result['speed']:.2f}"
    )

    del result


# Explicitly release baseline model.

release_model(
    discovery_model
)

del discovery_model

gc.collect()

torch.cuda.empty_cache()


# =============================================================================
# PROGRESSIVE EXPERIMENTS
# =============================================================================

all_results = []


for component, requested_counts in (
    PROGRESSIVE_TESTS.items()
):

    available_blocks = (
        component_blocks[
            component
        ]
    )

    print(
        "\n\n"
    )

    print(
        "#" * 100
    )

    print(
        f"PROGRESSIVE COMPONENT — "
        f"{component.upper()}"
    )

    print(
        "#" * 100
    )

    print(
        f"Available blocks: "
        f"{available_blocks}"
    )

    for requested_count in (
        requested_counts
    ):

        actual_count = min(
            requested_count,
            len(
                available_blocks
            ),
        )

        selected_blocks = (
            available_blocks[
                :actual_count
            ]
        )

        print(
            "\n" + "=" * 100
        )

        print(
            f"{component.upper()} — "
            f"FIRST {actual_count} "
            f"BLOCK(S)"
        )

        print(
            "=" * 100
        )

        print(
            f"Selected blocks : "
            f"{selected_blocks}"
        )

        # =============================================================
        # Fresh FP16 model for every test.
        # =============================================================

        model = load_model()

        stats = (
            ternarize_selected_blocks(
                model,
                component,
                selected_blocks,
            )
        )

        coverage = (
            stats["parameters"]
            / TOTAL_PARAMS
            * 100
        )

        print(
            f"\nAffected Linear layers : "
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


        # =============================================================
        # GENERATION
        # =============================================================

        print(
            "\nGENERATION"
        )

        print(
            "-" * 100
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

            # Result contains only CPU/Python values now,
            # but delete it anyway between prompts.
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

        # Save only lightweight Python metadata.
        all_results.append(
            {
                "component":
                    component,

                "blocks":
                    actual_count,

                "selected":
                    list(
                        selected_blocks
                    ),

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

        # =============================================================
        # CRITICAL CLEANUP BETWEEN TESTS
        # =============================================================

        release_model(
            model
        )

        del model

        gc.collect()
        gc.collect()

        torch.cuda.empty_cache()

        print_gpu_memory(
            "ready for next experiment"
        )


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print(
    "\n\n"
)

print(
    "=" * 120
)

print(
    "EXPERIMENT 006 — "
    "PROGRESSIVE SENSITIVITY SUMMARY"
)

print(
    "=" * 120
)

print(
    f"{'Component':<16}"
    f"{'Blocks':>8}"
    f"{'Linear':>9}"
    f"{'Parameters':>18}"
    f"{'Coverage':>12}"
    f"{'Mean Err':>12}"
    f"{'Max Err':>12}"
    f"{'Tok/s':>10}"
)

print(
    "-" * 120
)

for result in all_results:

    print(
        f"{result['component']:<16}"
        f"{result['blocks']:>8}"
        f"{result['linear_layers']:>9}"
        f"{result['parameters']:>18,}"
        f"{result['coverage']:>11.2f}%"
        f"{result['mean_error']:>12.4f}"
        f"{result['max_error']:>12.4f}"
        f"{result['throughput']:>10.2f}"
    )


# =============================================================================
# SELECTED BLOCK MAP
# =============================================================================

print(
    "\n" + "=" * 120
)

print(
    "SELECTED BLOCK MAP"
)

print(
    "=" * 120
)

for result in all_results:

    print(
        f"{result['component']:<15} "
        f"{result['blocks']:>2} "
        f"blocks -> "
        f"{result['selected']}"
    )


# =============================================================================
# IMPORTANT INTERPRETATION
# =============================================================================

print(
    "\n" + "=" * 100
)

print(
    "IMPORTANT INTERPRETATION"
)

print(
    "=" * 100
)

print(
    "Every progressive test starts from a "
    "fresh original FP16 model."
)

print()

print(
    "The quantization method is unchanged "
    "from Experiment 005."
)

print(
    f"Group-wise ternary quantization uses "
    f"group size {GROUP_SIZE}."
)

print()

print(
    "Only the number of transformer blocks "
    "being ternarized changes."
)

print()

print(
    "This experiment measures progressive "
    "architectural sensitivity."
)

print()

print(
    "It does NOT measure packed ternary "
    "inference speed."
)

print()

print(
    "The first available blocks of each "
    "component are tested progressively."
)

print()

print(
    "Early-vs-middle-vs-late location "
    "sensitivity should be tested separately "
    "after this experiment."
)

print(
    "\n" + "=" * 100
)

print(
    "EXPERIMENT 006 COMPLETE"
)

print(
    "=" * 100
)
