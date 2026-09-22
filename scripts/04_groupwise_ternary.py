import gc
import time

import torch
from transformers import AutoTokenizer, Qwen3_5ForCausalLM


# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_ID = "Qwen/Qwen3.5-0.8B"

DEVICE = "cuda"
DTYPE = torch.float16

MIN_WEIGHT_PARAMS = 100_000

GROUP_SIZE = 128
MAX_NEW_TOKENS = 50

PROMPTS = [
    "Explain in simple terms why the sky is blue.",
    "What is 17 multiplied by 23?",
    "If a train travels 60 km per hour for 2.5 hours, how far does it travel?",
    "Write a short paragraph explaining why exercise is important.",
]


# =============================================================================
# CANDIDATE SELECTION
# =============================================================================

def is_ternary_candidate(name, module):

    if not isinstance(module, torch.nn.Linear):
        return False

    if "lm_head" in name.lower():
        return False

    if module.weight.numel() < MIN_WEIGHT_PARAMS:
        return False

    return True


# =============================================================================
# GROUP-WISE TERNARIZATION
# =============================================================================

@torch.no_grad()
def groupwise_ternarize_weight(weight, group_size=128):
    """
    Group-wise symmetric ternarization.

    The flattened weight tensor is divided into groups of GROUP_SIZE.

    For each group:

        alpha = mean(abs(group))
        threshold = 0.5 * alpha

        T = -1 if w < -threshold
             0 if |w| <= threshold
            +1 if w > threshold

        reconstructed = alpha * T

    Experiment 004 changes only the scale granularity compared with
    Experiment 003.
    """

    original_shape = weight.shape

    # Use FP32 for quantization calculations.
    flat = weight.detach().float().reshape(-1)

    num_values = flat.numel()

    padding = (
        group_size - (num_values % group_size)
    ) % group_size

    if padding > 0:

        padded = torch.nn.functional.pad(
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

    # -----------------------------------------------------------------
    # Compute one scale per group.
    # -----------------------------------------------------------------

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

    # -----------------------------------------------------------------
    # Remove padding.
    # -----------------------------------------------------------------

    reconstructed = (
        reconstructed.reshape(-1)[:num_values]
    )

    reconstructed = reconstructed.reshape(
        original_shape
    )

    # Count only real weights, not padded values.
    ternary_flat = (
        ternary.reshape(-1)[:num_values]
    )

    negative = (
        ternary_flat == -1
    ).sum().item()

    zero = (
        ternary_flat == 0
    ).sum().item()

    positive = (
        ternary_flat == 1
    ).sum().item()

    # Scale statistics.
    alpha_flat = alpha.reshape(-1)

    alpha_min = alpha_flat.min().item()
    alpha_max = alpha_flat.max().item()
    alpha_mean = alpha_flat.mean().item()

    reconstructed = reconstructed.to(
        dtype=weight.dtype
    )

    return {
        "weight": reconstructed,
        "negative": negative,
        "zero": zero,
        "positive": positive,
        "alpha_min": alpha_min,
        "alpha_max": alpha_max,
        "alpha_mean": alpha_mean,
        "groups": alpha.numel(),
    }


# =============================================================================
# GENERATION
# =============================================================================

@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=MAX_NEW_TOKENS,
):

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(DEVICE)

    torch.cuda.synchronize()

    start = time.perf_counter()

    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
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
        "input_tokens": input_tokens,
        "generated_tokens": generated_tokens,
        "time": elapsed,
        "speed": speed,
    }


# =============================================================================
# HEADER
# =============================================================================

print("=" * 100)
print("EXPERIMENT 004 — Qwen3.5-0.8B Group-wise Ternary")
print("=" * 100)

print("\nConfiguration")
print("-" * 100)

print(f"Model           : {MODEL_ID}")
print(f"Device          : {DEVICE}")
print(f"Data type       : {DTYPE}")
print(f"Group size      : {GROUP_SIZE}")
print(f"Minimum weights : {MIN_WEIGHT_PARAMS:,}")


# =============================================================================
# ENVIRONMENT
# =============================================================================

print("\nEnvironment")
print("-" * 100)

print(f"PyTorch         : {torch.__version__}")
print(f"CUDA runtime    : {torch.version.cuda}")
print(f"CUDA available  : {torch.cuda.is_available()}")

if not torch.cuda.is_available():

    raise RuntimeError(
        "CUDA is required for this experiment."
    )

print(
    f"GPU             : "
    f"{torch.cuda.get_device_name(0)}"
)

print(
    f"GPU capability  : "
    f"{torch.cuda.get_device_capability(0)}"
)


# =============================================================================
# LOAD MODEL
# =============================================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID
)

print("Loading FP16 model...")

load_start = time.perf_counter()

model = Qwen3_5ForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    device_map=DEVICE,
)

model.eval()

torch.cuda.synchronize()

load_time = (
    time.perf_counter() - load_start
)

total_params = sum(
    p.numel()
    for p in model.parameters()
)

print(
    f"\nParameters      : "
    f"{total_params:,}"
)

print(
    f"Load time       : "
    f"{load_time:.2f} sec"
)


# =============================================================================
# FP16 BASELINE
# =============================================================================

print("\n" + "=" * 100)
print("FP16 BASELINE GENERATION")
print("=" * 100)

baseline_results = []

for index, prompt in enumerate(
    PROMPTS,
    start=1,
):

    print(
        f"\nPROMPT {index}"
    )

    print("-" * 100)

    print(prompt)

    result = generate(
        model,
        tokenizer,
        prompt,
    )

    baseline_results.append(result)

    print("\nOUTPUT:")
    print(result["text"])

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


# =============================================================================
# GROUP-WISE TERNARIZATION
# =============================================================================

print("\n" + "=" * 100)
print("GROUP-WISE TERNARIZATION")
print("=" * 100)

candidate_layers = 0
candidate_params = 0

total_groups = 0

total_negative = 0
total_zero = 0
total_positive = 0

layer_statistics = []


with torch.no_grad():

    for name, module in model.named_modules():

        if not is_ternary_candidate(
            name,
            module,
        ):
            continue

        weight = module.weight

        params = weight.numel()

        # -------------------------------------------------------------
        # Keep temporary original FP32 values for error calculation.
        # -------------------------------------------------------------

        original_fp32 = (
            weight.detach().float()
        )

        # -------------------------------------------------------------
        # Group-wise ternarization.
        # -------------------------------------------------------------

        result = groupwise_ternarize_weight(
            weight,
            GROUP_SIZE,
        )

        ternary_weight = result["weight"]

        reconstructed_fp32 = (
            ternary_weight.float()
        )

        # -------------------------------------------------------------
        # Quantization error.
        # -------------------------------------------------------------

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

        # -------------------------------------------------------------
        # Replace original FP16 weights.
        # -------------------------------------------------------------

        weight.copy_(
            ternary_weight
        )

        # -------------------------------------------------------------
        # Global statistics.
        # -------------------------------------------------------------

        candidate_layers += 1
        candidate_params += params

        total_groups += result["groups"]

        total_negative += result["negative"]
        total_zero += result["zero"]
        total_positive += result["positive"]

        layer_statistics.append(
            {
                "name": name,
                "params": params,
                "groups": result["groups"],
                "alpha_min": result["alpha_min"],
                "alpha_max": result["alpha_max"],
                "alpha_mean": result["alpha_mean"],
                "mae": mae,
                "rmse": rmse,
                "relative_error": relative_error,
            }
        )

        del original_fp32
        del reconstructed_fp32
        del ternary_weight
        del error
        del result


gc.collect()
torch.cuda.empty_cache()


# =============================================================================
# STATISTICS
# =============================================================================

print("\n" + "=" * 100)
print("GROUP-WISE TERNARY STATISTICS")
print("=" * 100)

coverage = (
    candidate_params
    / total_params
    * 100
)

ternary_total = (
    total_negative
    + total_zero
    + total_positive
)

negative_pct = (
    total_negative
    / ternary_total
    * 100
)

zero_pct = (
    total_zero
    / ternary_total
    * 100
)

positive_pct = (
    total_positive
    / ternary_total
    * 100
)

print(
    f"Group size             : "
    f"{GROUP_SIZE}"
)

print(
    f"Candidate layers       : "
    f"{candidate_layers}"
)

print(
    f"Candidate parameters   : "
    f"{candidate_params:,}"
)

print(
    f"Ternary coverage       : "
    f"{coverage:.2f}%"
)

print(
    f"Total scale groups     : "
    f"{total_groups:,}"
)

print()

print(
    f"-1 weights             : "
    f"{total_negative:,} "
    f"({negative_pct:.2f}%)"
)

print(
    f" 0 weights             : "
    f"{total_zero:,} "
    f"({zero_pct:.2f}%)"
)

print(
    f"+1 weights             : "
    f"{total_positive:,} "
    f"({positive_pct:.2f}%)"
)


# =============================================================================
# SCALE OVERHEAD
# =============================================================================

#
# Assume one FP16 scale for every group.
#

scale_bytes = (
    total_groups * 2
)

ternary_bytes = (
    candidate_params * 2 / 8
)

remaining_params = (
    total_params - candidate_params
)

remaining_fp16_bytes = (
    remaining_params * 2
)

hybrid_bytes = (
    ternary_bytes
    + scale_bytes
    + remaining_fp16_bytes
)

fp16_bytes = (
    total_params * 2
)

fp16_gib = (
    fp16_bytes / (1024 ** 3)
)

hybrid_gib = (
    hybrid_bytes / (1024 ** 3)
)

compression_ratio = (
    fp16_gib / hybrid_gib
)

reduction = (
    1 - hybrid_gib / fp16_gib
) * 100


print("\n" + "=" * 100)
print("THEORETICAL PACKED STORAGE")
print("=" * 100)

print(
    f"Ternary weight bytes  : "
    f"{ternary_bytes / (1024 ** 2):.2f} MiB"
)

print(
    f"FP16 scale bytes      : "
    f"{scale_bytes / (1024 ** 2):.2f} MiB"
)

print(
    f"Remaining FP16 bytes  : "
    f"{remaining_fp16_bytes / (1024 ** 2):.2f} MiB"
)

print()

print(
    f"Original FP16         : "
    f"{fp16_gib:.3f} GiB"
)

print(
    f"Hybrid + scales       : "
    f"{hybrid_gib:.3f} GiB"
)

print(
    f"Compression ratio     : "
    f"{compression_ratio:.2f}x"
)

print(
    f"Storage reduction     : "
    f"{reduction:.2f}%"
)


# =============================================================================
# ERROR ANALYSIS
# =============================================================================

print("\n" + "=" * 100)
print("TOP 10 LAYERS BY RELATIVE QUANTIZATION ERROR")
print("=" * 100)

layer_statistics.sort(
    key=lambda x: x["relative_error"],
    reverse=True,
)

for index, stats in enumerate(
    layer_statistics[:10],
    start=1,
):

    print(
        f"\n{index}. {stats['name']}"
    )

    print(
        f"   Parameters     : "
        f"{stats['params']:,}"
    )

    print(
        f"   Groups         : "
        f"{stats['groups']:,}"
    )

    print(
        f"   Alpha min      : "
        f"{stats['alpha_min']:.8f}"
    )

    print(
        f"   Alpha max      : "
        f"{stats['alpha_max']:.8f}"
    )

    print(
        f"   Alpha mean     : "
        f"{stats['alpha_mean']:.8f}"
    )

    print(
        f"   Relative error : "
        f"{stats['relative_error']:.6f}"
    )

    print(
        f"   MAE            : "
        f"{stats['mae']:.8f}"
    )

    print(
        f"   RMSE           : "
        f"{stats['rmse']:.8f}"
    )


# =============================================================================
# GROUP-WISE TERNARY GENERATION
# =============================================================================

print("\n" + "=" * 100)
print("GROUP-WISE TERNARY GENERATION")
print("=" * 100)

ternary_results = []

torch.cuda.reset_peak_memory_stats()

for index, prompt in enumerate(
    PROMPTS,
    start=1,
):

    print(
        f"\nPROMPT {index}"
    )

    print("-" * 100)

    print(prompt)

    result = generate(
        model,
        tokenizer,
        prompt,
    )

    ternary_results.append(result)

    print("\nOUTPUT:")
    print(result["text"])

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


# =============================================================================
# PERFORMANCE
# =============================================================================

print("\n" + "=" * 100)
print("FP16 vs GROUP-WISE TERNARY PERFORMANCE")
print("=" * 100)

baseline_time = sum(
    x["time"]
    for x in baseline_results
)

baseline_tokens = sum(
    x["generated_tokens"]
    for x in baseline_results
)

ternary_time = sum(
    x["time"]
    for x in ternary_results
)

ternary_tokens = sum(
    x["generated_tokens"]
    for x in ternary_results
)

baseline_speed = (
    baseline_tokens / baseline_time
)

ternary_speed = (
    ternary_tokens / ternary_time
)

print(
    f"FP16 throughput       : "
    f"{baseline_speed:.2f} tokens/sec"
)

print(
    f"Group-wise throughput : "
    f"{ternary_speed:.2f} tokens/sec"
)

print(
    f"Throughput change     : "
    f"{((ternary_speed / baseline_speed) - 1) * 100:+.2f}%"
)


# =============================================================================
# GPU MEMORY
# =============================================================================

peak_memory = (
    torch.cuda.max_memory_allocated()
    / (1024 ** 2)
)

print(
    f"\nPeak GPU memory       : "
    f"{peak_memory:.2f} MB"
)


# =============================================================================
# FINAL NOTE
# =============================================================================

print("\n" + "=" * 100)
print("IMPORTANT INTERPRETATION")
print("=" * 100)

print(
    "Experiment 004 uses group-wise ternary scaling "
    f"with group size {GROUP_SIZE}."
)

print()

print(
    "Each group receives its own alpha scale instead "
    "of using one alpha for the entire matrix."
)

print()

print(
    "The ternary weights are still reconstructed into "
    "FP16 tensors for normal PyTorch inference."
)

print()

print(
    "Therefore generation quality can be compared with "
    "Experiment 003, but inference speed is NOT yet "
    "packed ternary inference speed."
)

print("\n" + "=" * 100)
print("EXPERIMENT 004 COMPLETE")
print("=" * 100)
