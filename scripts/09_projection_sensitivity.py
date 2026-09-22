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
# EXPERIMENT 009
# =============================================================================
#
# Goal:
#
# Determine which projection matrices inside MLP, Linear Attention,
# and Self Attention are individually sensitive to ternarization.
#
# IMPORTANT:
#
# - Same ternary method as Experiments 004-008.
# - Exactly ONE projection type in ONE block is ternarized per test.
# - Every test starts from a fresh FP16 model.
#
# Representative blocks:
#
# MLP             : block 6
# Linear Attention: block 4
# Self Attention  : block 7
#
# Additional self-attention sensitivity probe:
#
# Self Attention  : block 15
#
# =============================================================================


TEST_CASES = [

    # -------------------------------------------------------------------------
    # MLP — BLOCK 6
    # -------------------------------------------------------------------------

    {
        "name": "MLP_B6_GATE",
        "component": "mlp",
        "block": 6,
        "projection": "gate_proj",
    },

    {
        "name": "MLP_B6_UP",
        "component": "mlp",
        "block": 6,
        "projection": "up_proj",
    },

    {
        "name": "MLP_B6_DOWN",
        "component": "mlp",
        "block": 6,
        "projection": "down_proj",
    },


    # -------------------------------------------------------------------------
    # LINEAR ATTENTION — BLOCK 4
    # -------------------------------------------------------------------------

    {
        "name": "LINEAR_B4_OUT",
        "component": "linear_attn",
        "block": 4,
        "projection": "out_proj",
    },

    {
        "name": "LINEAR_B4_QKV",
        "component": "linear_attn",
        "block": 4,
        "projection": "in_proj_qkv",
    },

    {
        "name": "LINEAR_B4_Z",
        "component": "linear_attn",
        "block": 4,
        "projection": "in_proj_z",
    },


    # -------------------------------------------------------------------------
    # SELF ATTENTION — BLOCK 7
    # -------------------------------------------------------------------------

    {
        "name": "SELF_B7_Q",
        "component": "self_attn",
        "block": 7,
        "projection": "q_proj",
    },

    {
        "name": "SELF_B7_K",
        "component": "self_attn",
        "block": 7,
        "projection": "k_proj",
    },

    {
        "name": "SELF_B7_V",
        "component": "self_attn",
        "block": 7,
        "projection": "v_proj",
    },

    {
        "name": "SELF_B7_O",
        "component": "self_attn",
        "block": 7,
        "projection": "o_proj",
    },


    # -------------------------------------------------------------------------
    # SELF ATTENTION — BLOCK 15
    #
    # Full block 15 showed noticeably weaker behavior in Experiment 007.
    # -------------------------------------------------------------------------

    {
        "name": "SELF_B15_Q",
        "component": "self_attn",
        "block": 15,
        "projection": "q_proj",
    },

    {
        "name": "SELF_B15_K",
        "component": "self_attn",
        "block": 15,
        "projection": "k_proj",
    },

    {
        "name": "SELF_B15_V",
        "component": "self_attn",
        "block": 15,
        "projection": "v_proj",
    },

    {
        "name": "SELF_B15_O",
        "component": "self_attn",
        "block": 15,
        "projection": "o_proj",
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


def get_projection_name(name):

    return name.split(".")[-1]


# =============================================================================
# DISCOVER MATRICES
# =============================================================================

def discover_projection_matrices(
    model,
):

    discovered = []

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

        if block is None:
            continue

        projection = (
            get_projection_name(
                name
            )
        )

        discovered.append(
            {
                "component":
                    component,

                "block":
                    block,

                "projection":
                    projection,

                "name":
                    name,

                "parameters":
                    module.weight.numel(),
            }
        )

    return discovered


# =============================================================================
# VALIDATION
# =============================================================================

def validate_test_cases(
    discovered,
):

    print(
        "\nValidating Experiment 009 "
        "test cases..."
    )

    discovered_keys = {
        (
            item["component"],
            item["block"],
            item["projection"],
        )
        for item in discovered
    }

    for case in TEST_CASES:

        key = (
            case["component"],
            case["block"],
            case["projection"],
        )

        if key not in discovered_keys:

            raise RuntimeError(
                "Requested projection "
                "was not found: "
                f"{key}"
            )

    print(
        "All projection-level "
        "test cases are valid."
    )


# =============================================================================
# GROUP-WISE TERNARY
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
# TERNARIZE EXACTLY ONE PROJECTION
# =============================================================================

@torch.no_grad()
def ternarize_projection(
    model,
    target_component,
    target_block,
    target_projection,
):

    matches = []

    for name, module in (
        model.named_modules()
    ):

        component = get_component(
            name,
            module,
        )

        if component != target_component:
            continue

        block = get_block_index(
            name
        )

        if block != target_block:
            continue

        projection = (
            get_projection_name(
                name
            )
        )

        if projection != target_projection:
            continue

        matches.append(
            (name, module)
        )

    if len(matches) != 1:

        raise RuntimeError(
            "Expected exactly one "
            "matching matrix for "
            f"{target_component}, "
            f"block {target_block}, "
            f"{target_projection}; "
            f"found {len(matches)}."
        )

    name, module = matches[0]

    weight = module.weight

    original_fp32 = (
        weight
        .detach()
        .float()
    )

    reconstructed = (
        groupwise_ternarize_weight(
            weight,
            GROUP_SIZE,
        )
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
        error_norm / original_norm
        if original_norm > 0
        else 0.0
    )

    parameter_count = (
        weight.numel()
    )

    weight.copy_(
        reconstructed
    )

    del original_fp32
    del reconstructed_fp32
    del reconstructed
    del error

    return {
        "name":
            name,

        "component":
            target_component,

        "block":
            target_block,

        "projection":
            target_projection,

        "parameters":
            parameter_count,

        "relative_error":
            relative_error,
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
    "=" * 110
)

print(
    "EXPERIMENT 009 — "
    "Qwen3.5-0.8B "
    "Projection-Level Ternary Sensitivity"
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
# DISCOVERY MODEL
# =============================================================================

discovery_model = (
    load_model()
)

TOTAL_PARAMS = sum(
    parameter.numel()
    for parameter
    in discovery_model.parameters()
)

discovered = (
    discover_projection_matrices(
        discovery_model
    )
)

print(
    "\nDiscovered projection matrices"
)

print(
    "-" * 110
)

for item in discovered:

    if (
        (
            item["component"],
            item["block"],
        )
        in {
            ("mlp", 6),
            ("linear_attn", 4),
            ("self_attn", 7),
            ("self_attn", 15),
        }
    ):

        print(
            f"{item['component']:<15}"
            f" block "
            f"{item['block']:>2}"
            f" | "
            f"{item['projection']:<15}"
            f" | "
            f"{item['parameters']:>12,}"
        )

print(
    f"\nTotal parameters : "
    f"{TOTAL_PARAMS:,}"
)

validate_test_cases(
    discovered
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
# RUN TESTS
# =============================================================================

all_results = []


for test_index, case in enumerate(
    TEST_CASES,
    start=1,
):

    print(
        "\n\n" + "#" * 110
    )

    print(
        f"TEST CASE "
        f"{test_index}/"
        f"{len(TEST_CASES)} — "
        f"{case['name']}"
    )

    print(
        "#" * 110
    )

    print(
        f"\nComponent  : "
        f"{case['component']}"
    )

    print(
        f"Block      : "
        f"{case['block']}"
    )

    print(
        f"Projection : "
        f"{case['projection']}"
    )


    # =========================================================================
    # FRESH MODEL
    # =========================================================================

    model = load_model()


    # =========================================================================
    # TERNARIZE EXACT PROJECTION
    # =========================================================================

    stats = (
        ternarize_projection(
            model,
            case["component"],
            case["block"],
            case["projection"],
        )
    )

    coverage = (
        stats["parameters"]
        / TOTAL_PARAMS
        * 100
    )


    print(
        "\nTERNARY STATS"
    )

    print(
        "-" * 110
    )

    print(
        f"Matrix             : "
        f"{stats['name']}"
    )

    print(
        f"Parameters         : "
        f"{stats['parameters']:,}"
    )

    print(
        f"Model coverage     : "
        f"{coverage:.4f}%"
    )

    print(
        f"Relative error     : "
        f"{stats['relative_error']:.6f}"
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
            "name":
                case["name"],

            "component":
                case["component"],

            "block":
                case["block"],

            "projection":
                case["projection"],

            "parameters":
                stats["parameters"],

            "coverage":
                coverage,

            "relative_error":
                stats[
                    "relative_error"
                ],

            "throughput":
                throughput,
        }
    )


    del outputs
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
        "ready for next projection"
    )


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print(
    "\n\n" + "=" * 125
)

print(
    "EXPERIMENT 009 — "
    "PROJECTION-LEVEL "
    "SENSITIVITY SUMMARY"
)

print(
    "=" * 125
)

print(
    f"{'Case':<20}"
    f"{'Component':<15}"
    f"{'Block':>7}"
    f"{'Projection':<16}"
    f"{'Parameters':>14}"
    f"{'Coverage':>11}"
    f"{'Rel Err':>11}"
    f"{'Tok/s':>10}"
)

print(
    "-" * 125
)

for result in all_results:

    print(
        f"{result['name']:<20}"
        f"{result['component']:<15}"
        f"{result['block']:>7}"
        f"{result['projection']:<16}"
        f"{result['parameters']:>14,}"
        f"{result['coverage']:>10.4f}%"
        f"{result['relative_error']:>11.4f}"
        f"{result['throughput']:>10.2f}"
    )


# =============================================================================
# COMPONENT GROUPED SUMMARY
# =============================================================================

print(
    "\n" + "=" * 125
)

print(
    "GROUPED BY COMPONENT / BLOCK"
)

print(
    "=" * 125
)

groups = {}

for result in all_results:

    key = (
        result["component"],
        result["block"],
    )

    groups.setdefault(
        key,
        []
    ).append(
        result
    )


for (
    component,
    block,
), results in groups.items():

    print(
        f"\n{component} — "
        f"block {block}"
    )

    print(
        "-" * 80
    )

    for result in results:

        print(
            f"{result['projection']:<16}"
            f" | Parameters "
            f"{result['parameters']:>10,}"
            f" | Coverage "
            f"{result['coverage']:>7.4f}%"
            f" | Error "
            f"{result['relative_error']:.6f}"
            f" | Tok/s "
            f"{result['throughput']:.2f}"
        )


# =============================================================================
# INTERPRETATION
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
    "Experiment 009 changes only "
    "one projection matrix at a time."
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
    "The purpose is NOT to rank "
    "projections only by numerical "
    "reconstruction error."
)

print()

print(
    "The main question is whether "
    "different projection types cause "
    "different functional degradation "
    "under the same ternary method."
)

print()

print(
    "Throughput is only a runtime "
    "reference and must not be treated "
    "as packed ternary inference speed."
)

print()

print(
    "The three prompts remain preliminary "
    "functional diagnostics rather than "
    "a formal quality benchmark."
)

print(
    "\n" + "=" * 110
)

print(
    "EXPERIMENT 009 COMPLETE"
)

print(
    "=" * 110
)
