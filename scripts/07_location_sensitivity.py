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

# -------------------------------------------------------------------------
# Experiment 007:
# Ternarize ONE transformer block at a time.
#
# These locations sample early, middle, and late regions.
# -------------------------------------------------------------------------

LOCATION_TESTS = {
    "mlp": [
        0,
        6,
        12,
        18,
        23,
    ],

    "linear_attn": [
        0,
        4,
        8,
        12,
        16,
        22,
    ],

    "self_attn": [
        3,
        7,
        11,
        15,
        19,
        23,
    ],
}


# =============================================================================
# CUDA MEMORY
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
# TERNARIZE ONE BLOCK
# =============================================================================

@torch.no_grad()
def ternarize_single_block(
    model,
    target_component,
    target_block,
):

    affected_linear_layers = 0
    affected_parameters = 0

    relative_errors = []

    affected_names = []

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
            != target_block
        ):
            continue

        weight = module.weight

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

        affected_names.append(
            (
                name,
                weight.numel(),
                relative_error,
            )
        )

        del original_fp32
        del ternary_weight
        del reconstructed_fp32
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

    return {
        "linear_layers":
            affected_linear_layers,

        "parameters":
            affected_parameters,

        "mean_error":
            mean_error,

        "max_error":
            max_error,

        "affected_names":
            affected_names,
    }


# =============================================================================
# HEADER
# =============================================================================

print(
    "=" * 100
)

print(
    "EXPERIMENT 007 — "
    "Qwen3.5-0.8B "
    "Individual Block Location Sensitivity"
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
# ARCHITECTURE DISCOVERY + FP16 REFERENCE
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
# VALIDATE TEST LOCATIONS
# =============================================================================

print(
    "\nValidating Experiment 007 block locations..."
)

for component, blocks in (
    LOCATION_TESTS.items()
):

    available = set(
        component_blocks[
            component
        ]
    )

    for block in blocks:

        if block not in available:

            raise RuntimeError(
                f"Block {block} is not available "
                f"for component '{component}'. "
                f"Available blocks: "
                f"{sorted(available)}"
            )

print(
    "All requested block locations are valid."
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
# LOCATION SENSITIVITY TESTS
# =============================================================================

all_results = []


for component, blocks in (
    LOCATION_TESTS.items()
):

    print(
        "\n\n" + "#" * 100
    )

    print(
        f"COMPONENT — "
        f"{component.upper()}"
    )

    print(
        "#" * 100
    )

    for block in blocks:

        print(
            "\n" + "=" * 100
        )

        print(
            f"{component.upper()} — "
            f"BLOCK {block}"
        )

        print(
            "=" * 100
        )

        # -----------------------------------------------------------------
        # Every location test starts from fresh FP16 weights.
        # -----------------------------------------------------------------

        model = load_model()

        stats = (
            ternarize_single_block(
                model,
                component,
                block,
            )
        )

        if (
            stats["linear_layers"]
            == 0
        ):

            raise RuntimeError(
                f"No candidate Linear layers "
                f"were found for "
                f"{component} block {block}."
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


        # -----------------------------------------------------------------
        # Print exact matrices changed.
        # -----------------------------------------------------------------

        print(
            "\nTERNARIZED MATRICES"
        )

        print(
            "-" * 100
        )

        for (
            layer_name,
            parameter_count,
            relative_error,
        ) in stats[
            "affected_names"
        ]:

            print(
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


        # -----------------------------------------------------------------
        # Generation
        # -----------------------------------------------------------------

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


        # -----------------------------------------------------------------
        # Store lightweight summary only.
        # -----------------------------------------------------------------

        all_results.append(
            {
                "component":
                    component,

                "block":
                    block,

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


        # -----------------------------------------------------------------
        # Critical GPU cleanup.
        # -----------------------------------------------------------------

        release_model(
            model
        )

        del model

        gc.collect()
        gc.collect()

        torch.cuda.empty_cache()

        print_gpu_memory(
            "ready for next location test"
        )


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print(
    "\n\n" + "=" * 120
)

print(
    "EXPERIMENT 007 — "
    "INDIVIDUAL BLOCK LOCATION "
    "SENSITIVITY SUMMARY"
)

print(
    "=" * 120
)

print(
    f"{'Component':<16}"
    f"{'Block':>8}"
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
        f"{result['block']:>8}"
        f"{result['linear_layers']:>9}"
        f"{result['parameters']:>18,}"
        f"{result['coverage']:>11.2f}%"
        f"{result['mean_error']:>12.4f}"
        f"{result['max_error']:>12.4f}"
        f"{result['throughput']:>10.2f}"
    )


# =============================================================================
# COMPONENT-WISE SUMMARY
# =============================================================================

print(
    "\n" + "=" * 120
)

print(
    "COMPONENT-WISE LOCATION MAP"
)

print(
    "=" * 120
)

for component in (
    LOCATION_TESTS.keys()
):

    print(
        f"\n{component.upper()}"
    )

    print(
        "-" * 120
    )

    component_results = [
        result
        for result in all_results
        if (
            result["component"]
            == component
        )
    ]

    for result in (
        component_results
    ):

        print(
            f"Block "
            f"{result['block']:>2}"
            f" | Coverage "
            f"{result['coverage']:>5.2f}%"
            f" | Mean error "
            f"{result['mean_error']:.4f}"
            f" | Max error "
            f"{result['max_error']:.4f}"
            f" | "
            f"{result['throughput']:.2f} tok/s"
        )


# =============================================================================
# INTERPRETATION NOTES
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
    "Each test ternarizes exactly ONE "
    "transformer block of ONE component."
)

print()

print(
    "Every test begins from a fresh "
    "original FP16 model."
)

print()

print(
    "The ternary method is unchanged:"
)

print(
    f"  Group size       : "
    f"{GROUP_SIZE}"
)

print(
    "  Scale            : "
    "mean(abs(group))"
)

print(
    "  Threshold        : "
    "0.5 * scale"
)

print(
    "  Codes            : "
    "{-1, 0, +1}"
)

print(
    "  Runtime storage  : "
    "reconstructed FP16"
)

print()

print(
    "Therefore this experiment isolates "
    "BLOCK LOCATION as the primary "
    "experimental variable."
)

print()

print(
    "Generation throughput is not packed "
    "ternary inference throughput."
)

print()

print(
    "Prompt quality remains a preliminary "
    "functional diagnostic rather than a "
    "formal benchmark score."
)

print(
    "\n" + "=" * 100
)

print(
    "EXPERIMENT 007 COMPLETE"
)

print(
    "=" * 100
)
