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
MAX_NEW_TOKENS = 50

PROMPTS = [
    "Explain in simple terms why the sky is blue.",
    "What is 17 multiplied by 23?",
    "If a train travels 60 km per hour for 2.5 hours, how far does it travel?",
    "Write a short paragraph explaining why exercise is important.",
]

COMPONENTS = [
    "mlp",
    "linear_attn",
    "self_attn",
]


# =============================================================================
# COMPONENT CLASSIFICATION
# =============================================================================

def get_component(name, module):
    """
    Classify a Linear layer into one of the components being tested.

    mlp:
        model.layers.X.mlp.*

    linear_attn:
        model.layers.X.linear_attn.*
        Qwen3.5 DeltaNet / linear-attention blocks

    self_attn:
        model.layers.X.self_attn.*
        Full-attention blocks
    """

    if not isinstance(module, torch.nn.Linear):
        return None

    if module.weight.numel() < MIN_WEIGHT_PARAMS:
        return None

    name_lower = name.lower()

    if ".mlp." in name_lower:
        return "mlp"

    if ".linear_attn." in name_lower:
        return "linear_attn"

    if ".self_attn." in name_lower:
        return "self_attn"

    return None


# =============================================================================
# GROUP-WISE TERNARIZATION
# =============================================================================

@torch.no_grad()
def groupwise_ternarize_weight(weight, group_size=128):

    original_shape = weight.shape

    flat = weight.detach().float().reshape(-1)

    num_values = flat.numel()

    padding = (
        group_size - (num_values % group_size)
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

    groups = padded.reshape(
        -1,
        group_size,
    )

    alpha = groups.abs().mean(
        dim=1,
        keepdim=True,
    )

    threshold = 0.5 * alpha

    ternary = torch.zeros_like(groups)

    ternary[
        groups > threshold
    ] = 1.0

    ternary[
        groups < -threshold
    ] = -1.0

    reconstructed = (
        ternary * alpha
    )

    reconstructed = (
        reconstructed
        .reshape(-1)[:num_values]
        .reshape(original_shape)
    )

    ternary_real = (
        ternary
        .reshape(-1)[:num_values]
    )

    negative = (
        ternary_real == -1
    ).sum().item()

    zero = (
        ternary_real == 0
    ).sum().item()

    positive = (
        ternary_real == 1
    ).sum().item()

    return {
        "weight": reconstructed.to(
            dtype=weight.dtype
        ),
        "negative": negative,
        "zero": zero,
        "positive": positive,
        "groups": alpha.numel(),
        "alpha_min": alpha.min().item(),
        "alpha_max": alpha.max().item(),
        "alpha_mean": alpha.mean().item(),
    }


# =============================================================================
# GENERATION
# =============================================================================

@torch.no_grad()
def generate(model, tokenizer, prompt):

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(DEVICE)

    torch.cuda.synchronize()

    start = time.perf_counter()

    output = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
    )

    torch.cuda.synchronize()

    elapsed = (
        time.perf_counter() - start
    )

    input_tokens = (
        inputs["input_ids"].shape[1]
    )

    generated_tokens = (
        output.shape[1] - input_tokens
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

    return {
        "text": text,
        "generated_tokens": generated_tokens,
        "time": elapsed,
        "speed": speed,
    }


# =============================================================================
# LOAD MODEL
# =============================================================================

def load_original_model():

    print("\nLoading fresh original FP16 model...")

    start = time.perf_counter()

    model = Qwen3_5ForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        device_map=DEVICE,
    )

    model.eval()

    torch.cuda.synchronize()

    elapsed = (
        time.perf_counter() - start
    )

    print(
        f"Model loaded in {elapsed:.2f} sec"
    )

    return model


# =============================================================================
# RELEASE MODEL
# =============================================================================

def release_model(model):

    del model

    gc.collect()

    torch.cuda.empty_cache()

    torch.cuda.synchronize()


# =============================================================================
# TERNARIZE ONE COMPONENT
# =============================================================================

@torch.no_grad()
def ternarize_component(
    model,
    target_component,
    total_model_params,
):

    layer_stats = []

    candidate_layers = 0
    candidate_params = 0
    total_groups = 0

    total_negative = 0
    total_zero = 0
    total_positive = 0

    print(
        f"\nTernarizing component: "
        f"{target_component}"
    )

    for name, module in model.named_modules():

        component = get_component(
            name,
            module,
        )

        if component != target_component:
            continue

        weight = module.weight

        params = weight.numel()

        original_fp32 = (
            weight.detach().float()
        )

        result = groupwise_ternarize_weight(
            weight,
            GROUP_SIZE,
        )

        ternary_weight = result["weight"]

        reconstructed_fp32 = (
            ternary_weight.float()
        )

        error = (
            original_fp32
            - reconstructed_fp32
        )

        mae = (
            error.abs()
            .mean()
            .item()
        )

        rmse = torch.sqrt(
            torch.mean(
                error ** 2
            )
        ).item()

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

        weight.copy_(
            ternary_weight
        )

        candidate_layers += 1
        candidate_params += params

        total_groups += result["groups"]

        total_negative += result["negative"]
        total_zero += result["zero"]
        total_positive += result["positive"]

        layer_stats.append(
            {
                "name": name,
                "params": params,
                "groups": result["groups"],
                "relative_error": relative_error,
                "mae": mae,
                "rmse": rmse,
            }
        )

        del original_fp32
        del reconstructed_fp32
        del ternary_weight
        del error
        del result

    ternary_total = (
        total_negative
        + total_zero
        + total_positive
    )

    coverage = (
        candidate_params
        / total_model_params
        * 100
    )

    negative_pct = (
        total_negative / ternary_total * 100
        if ternary_total > 0
        else 0
    )

    zero_pct = (
        total_zero / ternary_total * 100
        if ternary_total > 0
        else 0
    )

    positive_pct = (
        total_positive / ternary_total * 100
        if ternary_total > 0
        else 0
    )

    if layer_stats:

        mean_relative_error = sum(
            x["relative_error"]
            for x in layer_stats
        ) / len(layer_stats)

        max_relative_error = max(
            x["relative_error"]
            for x in layer_stats
        )

    else:

        mean_relative_error = 0
        max_relative_error = 0

    return {
        "layers": candidate_layers,
        "params": candidate_params,
        "coverage": coverage,
        "groups": total_groups,
        "negative": total_negative,
        "zero": total_zero,
        "positive": total_positive,
        "negative_pct": negative_pct,
        "zero_pct": zero_pct,
        "positive_pct": positive_pct,
        "mean_relative_error": mean_relative_error,
        "max_relative_error": max_relative_error,
        "layer_stats": layer_stats,
    }


# =============================================================================
# HEADER
# =============================================================================

print("=" * 100)
print(
    "EXPERIMENT 005 — Qwen3.5-0.8B "
    "Component Sensitivity"
)
print("=" * 100)

print("\nConfiguration")
print("-" * 100)

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
    f"Minimum parameters : "
    f"{MIN_WEIGHT_PARAMS:,}"
)


# =============================================================================
# ENVIRONMENT
# =============================================================================

print("\nEnvironment")
print("-" * 100)

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


# =============================================================================
# TOKENIZER
# =============================================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID
)


# =============================================================================
# GET TOTAL PARAMETER COUNT
# =============================================================================

model = load_original_model()

TOTAL_MODEL_PARAMS = sum(
    p.numel()
    for p in model.parameters()
)

print(
    f"Total parameters   : "
    f"{TOTAL_MODEL_PARAMS:,}"
)


# =============================================================================
# FP16 REFERENCE GENERATION
# =============================================================================

print("\n" + "=" * 100)
print("FP16 REFERENCE GENERATION")
print("=" * 100)

fp16_results = []

for index, prompt in enumerate(
    PROMPTS,
    start=1,
):

    result = generate(
        model,
        tokenizer,
        prompt,
    )

    fp16_results.append(result)

    print(
        f"\nPROMPT {index}"
    )

    print("-" * 100)

    print(
        result["text"]
    )

    print(
        f"\nTokens/sec : "
        f"{result['speed']:.2f}"
    )


# Release baseline model before component experiments.

release_model(model)


# =============================================================================
# COMPONENT EXPERIMENTS
# =============================================================================

experiment_results = {}


for component in COMPONENTS:

    print("\n\n")
    print("#" * 100)

    print(
        f"COMPONENT TEST — "
        f"{component.upper()}"
    )

    print("#" * 100)

    # -------------------------------------------------------------
    # IMPORTANT:
    # Fresh original model for every component.
    # -------------------------------------------------------------

    model = load_original_model()

    stats = ternarize_component(
        model,
        component,
        TOTAL_MODEL_PARAMS,
    )

    print("\n" + "=" * 100)

    print(
        f"{component.upper()} TERNARY STATISTICS"
    )

    print("=" * 100)

    print(
        f"Candidate layers       : "
        f"{stats['layers']}"
    )

    print(
        f"Candidate parameters   : "
        f"{stats['params']:,}"
    )

    print(
        f"Model coverage         : "
        f"{stats['coverage']:.2f}%"
    )

    print(
        f"Scale groups           : "
        f"{stats['groups']:,}"
    )

    print()

    print(
        f"-1 weights             : "
        f"{stats['negative']:,} "
        f"({stats['negative_pct']:.2f}%)"
    )

    print(
        f" 0 weights             : "
        f"{stats['zero']:,} "
        f"({stats['zero_pct']:.2f}%)"
    )

    print(
        f"+1 weights             : "
        f"{stats['positive']:,} "
        f"({stats['positive_pct']:.2f}%)"
    )

    print()

    print(
        f"Mean relative error    : "
        f"{stats['mean_relative_error']:.6f}"
    )

    print(
        f"Max relative error     : "
        f"{stats['max_relative_error']:.6f}"
    )


    # -------------------------------------------------------------
    # Worst layers within component.
    # -------------------------------------------------------------

    print("\nTOP 5 ERROR LAYERS")
    print("-" * 100)

    sorted_layers = sorted(
        stats["layer_stats"],
        key=lambda x: x["relative_error"],
        reverse=True,
    )

    for rank, layer in enumerate(
        sorted_layers[:5],
        start=1,
    ):

        print(
            f"{rank}. "
            f"{layer['name']}"
        )

        print(
            f"   Parameters     : "
            f"{layer['params']:,}"
        )

        print(
            f"   Relative error : "
            f"{layer['relative_error']:.6f}"
        )

        print(
            f"   MAE            : "
            f"{layer['mae']:.8f}"
        )

        print(
            f"   RMSE           : "
            f"{layer['rmse']:.8f}"
        )


    # -------------------------------------------------------------
    # Generation test.
    # -------------------------------------------------------------

    print("\n" + "=" * 100)

    print(
        f"{component.upper()}-ONLY "
        f"TERNARY GENERATION"
    )

    print("=" * 100)

    generation_results = []

    torch.cuda.reset_peak_memory_stats()

    for index, prompt in enumerate(
        PROMPTS,
        start=1,
    ):

        result = generate(
            model,
            tokenizer,
            prompt,
        )

        generation_results.append(
            result
        )

        print(
            f"\nPROMPT {index}"
        )

        print("-" * 100)

        print(
            result["text"]
        )

        print(
            f"\nGenerated tokens : "
            f"{result['generated_tokens']}"
        )

        print(
            f"Generation time  : "
            f"{result['time']:.3f} sec"
        )

        print(
            f"Tokens/sec       : "
            f"{result['speed']:.2f}"
        )


    total_time = sum(
        x["time"]
        for x in generation_results
    )

    total_tokens = sum(
        x["generated_tokens"]
        for x in generation_results
    )

    throughput = (
        total_tokens / total_time
        if total_time > 0
        else 0
    )

    peak_memory = (
        torch.cuda.max_memory_allocated()
        / (1024 ** 2)
    )

    experiment_results[component] = {
        "stats": stats,
        "throughput": throughput,
        "peak_memory": peak_memory,
        "generation": generation_results,
    }

    print(
        f"\nAggregate throughput : "
        f"{throughput:.2f} tokens/sec"
    )

    print(
        f"Peak GPU memory      : "
        f"{peak_memory:.2f} MB"
    )

    # -------------------------------------------------------------
    # Destroy this modified model before loading next one.
    # -------------------------------------------------------------

    release_model(model)


# =============================================================================
# FINAL COMPARISON
# =============================================================================

print("\n\n")
print("=" * 100)
print("EXPERIMENT 005 — COMPONENT COMPARISON")
print("=" * 100)

print(
    f"{'Component':<20}"
    f"{'Layers':>10}"
    f"{'Parameters':>18}"
    f"{'Coverage':>12}"
    f"{'Mean Err':>12}"
    f"{'Max Err':>12}"
    f"{'Tok/s':>12}"
)

print("-" * 96)

for component in COMPONENTS:

    result = experiment_results[
        component
    ]

    stats = result["stats"]

    print(
        f"{component:<20}"
        f"{stats['layers']:>10}"
        f"{stats['params']:>18,}"
        f"{stats['coverage']:>11.2f}%"
        f"{stats['mean_relative_error']:>12.4f}"
        f"{stats['max_relative_error']:>12.4f}"
        f"{result['throughput']:>12.2f}"
    )


# =============================================================================
# THEORETICAL COMPONENT STORAGE
# =============================================================================

print("\n" + "=" * 100)
print("THEORETICAL COMPONENT STORAGE")
print("=" * 100)

original_fp16_bytes = (
    TOTAL_MODEL_PARAMS * 2
)

for component in COMPONENTS:

    stats = (
        experiment_results[
            component
        ]["stats"]
    )

    ternary_params = stats["params"]

    groups = stats["groups"]

    ternary_bytes = (
        ternary_params * 2 / 8
    )

    scale_bytes = (
        groups * 2
    )

    remaining_fp16_bytes = (
        (
            TOTAL_MODEL_PARAMS
            - ternary_params
        )
        * 2
    )

    total_bytes = (
        ternary_bytes
        + scale_bytes
        + remaining_fp16_bytes
    )

    compression = (
        original_fp16_bytes
        / total_bytes
    )

    reduction = (
        1
        - total_bytes
        / original_fp16_bytes
    ) * 100

    print(
        f"\n{component}"
    )

    print(
        f"  Hybrid size       : "
        f"{total_bytes / (1024 ** 3):.3f} GiB"
    )

    print(
        f"  Compression       : "
        f"{compression:.2f}x"
    )

    print(
        f"  Storage reduction : "
        f"{reduction:.2f}%"
    )


# =============================================================================
# IMPORTANT NOTE
# =============================================================================

print("\n" + "=" * 100)
print("IMPORTANT INTERPRETATION")
print("=" * 100)

print(
    "Each component experiment started from a fresh "
    "original FP16 Qwen3.5-0.8B model."
)

print()

print(
    "Only one component was ternarized at a time."
)

print()

print(
    "Therefore differences in generation behavior can "
    "be used to identify which architectural components "
    "are most sensitive to the current group-wise "
    "ternarization method."
)

print()

print(
    "Weights are still reconstructed into FP16 tensors. "
    "This experiment evaluates numerical sensitivity, "
    "not packed ternary inference performance."
)

print("\n" + "=" * 100)
print("EXPERIMENT 005 COMPLETE")
print("=" * 100)
