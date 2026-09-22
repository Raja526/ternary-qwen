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

# Same conservative selection policy established in Experiment 002.
MIN_WEIGHT_PARAMS = 100_000

MAX_NEW_TOKENS = 50

PROMPTS = [
    "Explain in simple terms why the sky is blue.",
    "What is 17 multiplied by 23?",
    "If a train travels 60 km per hour for 2.5 hours, how far does it travel?",
    "Write a short paragraph explaining why exercise is important.",
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def is_ternary_candidate(name, module):
    """
    Decide whether a module should be ternarized in Ternary-V0.

    V0 policy:
    - Only torch.nn.Linear layers
    - Exclude LM head
    - Exclude small matrices (<100K weights)
    """

    if not isinstance(module, torch.nn.Linear):
        return False

    if "lm_head" in name.lower():
        return False

    if module.weight.numel() < MIN_WEIGHT_PARAMS:
        return False

    return True


@torch.no_grad()
def ternarize_weight(weight):
    """
    Naive per-tensor symmetric ternarization.

    alpha = mean(abs(W))

    threshold = 0.5 * alpha

    T =
        -1  if W < -threshold
         0  if |W| <= threshold
        +1  if W > threshold

    Reconstructed weight:

        W_ternary = alpha * T

    Important:
    The resulting tensor is converted back to the original dtype
    (FP16 in this experiment).

    Therefore this experiment tests numerical ternarization only.
    It does NOT provide actual 2-bit storage yet.
    """

    # Perform statistics in FP32 for numerical stability.
    w = weight.detach().float()

    alpha = w.abs().mean()

    if alpha.item() == 0:
        reconstructed = weight.detach().clone()

        return (
            reconstructed,
            0.0,
            0,
            weight.numel(),
            0,
        )

    threshold = 0.5 * alpha

    ternary = torch.zeros_like(w)

    ternary[w > threshold] = 1.0
    ternary[w < -threshold] = -1.0

    reconstructed = ternary * alpha

    negative = (ternary == -1).sum().item()
    zero = (ternary == 0).sum().item()
    positive = (ternary == 1).sum().item()

    reconstructed = reconstructed.to(
        dtype=weight.dtype
    )

    return (
        reconstructed,
        alpha.item(),
        negative,
        zero,
        positive,
    )


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=MAX_NEW_TOKENS,
):
    """
    Generate deterministic output and measure generation speed.
    """

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

    elapsed = time.perf_counter() - start

    input_tokens = inputs["input_ids"].shape[1]

    generated_tokens = (
        output.shape[1] - input_tokens
    )

    text = tokenizer.decode(
        output[0],
        skip_special_tokens=True,
    )

    if elapsed > 0:
        tokens_per_sec = generated_tokens / elapsed
    else:
        tokens_per_sec = 0.0

    return {
        "text": text,
        "input_tokens": input_tokens,
        "generated_tokens": generated_tokens,
        "time": elapsed,
        "speed": tokens_per_sec,
    }


# =============================================================================
# EXPERIMENT HEADER
# =============================================================================

print("=" * 100)
print("EXPERIMENT 003 — Qwen3.5-0.8B Ternary-V0")
print("=" * 100)


# =============================================================================
# ENVIRONMENT
# =============================================================================

print("\nEnvironment")
print("-" * 100)

print(f"PyTorch        : {torch.__version__}")
print(f"CUDA runtime   : {torch.version.cuda}")
print(f"CUDA available : {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is required for this experiment."
    )

print(
    f"GPU            : "
    f"{torch.cuda.get_device_name(0)}"
)

print(
    f"GPU capability : "
    f"{torch.cuda.get_device_capability(0)}"
)

total_vram = (
    torch.cuda.get_device_properties(0).total_memory
    / (1024 ** 3)
)

print(
    f"Total VRAM     : "
    f"{total_vram:.2f} GB"
)


# =============================================================================
# LOAD TOKENIZER
# =============================================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID
)


# =============================================================================
# LOAD MODEL
# =============================================================================

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
    parameter.numel()
    for parameter in model.parameters()
)

parameter_bytes = sum(
    parameter.numel() * parameter.element_size()
    for parameter in model.parameters()
)

parameter_memory_gb = (
    parameter_bytes / (1024 ** 3)
)

print()
print(f"Model            : {MODEL_ID}")
print(f"Device           : {DEVICE}")
print(f"Data type        : {DTYPE}")

print(
    f"Parameters       : "
    f"{total_params:,}"
)

print(
    f"Parameter memory : "
    f"{parameter_memory_gb:.3f} GB"
)

print(
    f"Load time        : "
    f"{load_time:.2f} sec"
)


# =============================================================================
# FP16 BASELINE GENERATION
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
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
    )

    baseline_results.append(result)

    print("\nOUTPUT:")

    print(
        result["text"]
    )

    print(
        f"\nInput tokens     : "
        f"{result['input_tokens']}"
    )

    print(
        f"Generated tokens : "
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
# IDENTIFY CANDIDATES
# =============================================================================

print("\n" + "=" * 100)
print("TERNARIZING SELECTED WEIGHTS")
print("=" * 100)

candidate_layers = 0
candidate_params = 0

total_negative = 0
total_zero = 0
total_positive = 0

alpha_values = []

layer_statistics = []


# =============================================================================
# TERNARIZE WEIGHTS
# =============================================================================

#
# IMPORTANT:
#
# The entire modification stage runs under torch.no_grad().
#
# This prevents:
#
# RuntimeError:
# "a leaf Variable that requires grad is being used
#  in an in-place operation."
#

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
        # Save original weight temporarily in FP32.
        # -------------------------------------------------------------

        original_fp32 = (
            weight.detach()
            .float()
        )

        # -------------------------------------------------------------
        # Ternarize.
        # -------------------------------------------------------------

        (
            ternary_weight,
            alpha,
            negative,
            zero,
            positive,
        ) = ternarize_weight(weight)

        # -------------------------------------------------------------
        # Quantization error.
        # -------------------------------------------------------------

        reconstructed_fp32 = (
            ternary_weight.float()
        )

        error = (
            original_fp32
            - reconstructed_fp32
        )

        mae = (
            error
            .abs()
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

        if original_norm > 0:
            relative_error = (
                error_norm
                / original_norm
            )
        else:
            relative_error = 0.0

        # -------------------------------------------------------------
        # Replace model weight.
        # -------------------------------------------------------------

        weight.copy_(
            ternary_weight
        )

        # -------------------------------------------------------------
        # Accumulate statistics.
        # -------------------------------------------------------------

        candidate_layers += 1

        candidate_params += params

        total_negative += negative
        total_zero += zero
        total_positive += positive

        alpha_values.append(alpha)

        layer_statistics.append(
            {
                "name": name,
                "params": params,
                "alpha": alpha,
                "negative": negative,
                "zero": zero,
                "positive": positive,
                "mae": mae,
                "rmse": rmse,
                "relative_error": relative_error,
            }
        )

        # -------------------------------------------------------------
        # Free temporary tensors.
        # -------------------------------------------------------------

        del original_fp32
        del reconstructed_fp32
        del ternary_weight
        del error


gc.collect()

torch.cuda.empty_cache()


# =============================================================================
# TERNARY STATISTICS
# =============================================================================

print("\n" + "=" * 100)
print("TERNARY STATISTICS")
print("=" * 100)

ternary_total = (
    total_negative
    + total_zero
    + total_positive
)

coverage = (
    candidate_params
    / total_params
    * 100
)

print(
    f"Total parameters       : "
    f"{total_params:,}"
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


if ternary_total > 0:

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

else:

    negative_pct = 0
    zero_pct = 0
    positive_pct = 0


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


if alpha_values:

    print()

    print(
        f"Alpha minimum          : "
        f"{min(alpha_values):.8f}"
    )

    print(
        f"Alpha maximum          : "
        f"{max(alpha_values):.8f}"
    )

    print(
        f"Alpha average          : "
        f"{sum(alpha_values) / len(alpha_values):.8f}"
    )


# =============================================================================
# THEORETICAL STORAGE
# =============================================================================

print("\n" + "=" * 100)
print("THEORETICAL STORAGE")
print("=" * 100)

fp16_bytes = (
    total_params * 2
)

ternary_candidate_bytes = (
    candidate_params * 2 / 8
)

remaining_params = (
    total_params
    - candidate_params
)

remaining_fp16_bytes = (
    remaining_params * 2
)

hybrid_bytes = (
    ternary_candidate_bytes
    + remaining_fp16_bytes
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
    1
    - hybrid_gib / fp16_gib
) * 100

print(
    f"FP16 raw storage       : "
    f"{fp16_gib:.3f} GiB"
)

print(
    f"Hybrid theoretical     : "
    f"{hybrid_gib:.3f} GiB"
)

print(
    f"Compression ratio      : "
    f"{compression_ratio:.2f}x"
)

print(
    f"Storage reduction      : "
    f"{reduction:.2f}%"
)

print()
print(
    "NOTE: The model is still physically stored as FP16 "
    "in this experiment."
)


# =============================================================================
# LAYERS WITH HIGHEST ERROR
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
        f"   Alpha          : "
        f"{stats['alpha']:.8f}"
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
# TERNARY GENERATION
# =============================================================================

print("\n" + "=" * 100)
print("TERNARY-V0 GENERATION")
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
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
    )

    ternary_results.append(result)

    print("\nOUTPUT:")

    print(
        result["text"]
    )

    print(
        f"\nInput tokens     : "
        f"{result['input_tokens']}"
    )

    print(
        f"Generated tokens : "
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
# PERFORMANCE COMPARISON
# =============================================================================

print("\n" + "=" * 100)
print("FP16 vs TERNARY-V0 PERFORMANCE")
print("=" * 100)


baseline_total_time = sum(
    result["time"]
    for result in baseline_results
)

baseline_total_tokens = sum(
    result["generated_tokens"]
    for result in baseline_results
)


ternary_total_time = sum(
    result["time"]
    for result in ternary_results
)

ternary_total_tokens = sum(
    result["generated_tokens"]
    for result in ternary_results
)


if baseline_total_time > 0:

    baseline_speed = (
        baseline_total_tokens
        / baseline_total_time
    )

else:

    baseline_speed = 0


if ternary_total_time > 0:

    ternary_speed = (
        ternary_total_tokens
        / ternary_total_time
    )

else:

    ternary_speed = 0


print(
    f"FP16 generated tokens : "
    f"{baseline_total_tokens}"
)

print(
    f"FP16 total time       : "
    f"{baseline_total_time:.3f} sec"
)

print(
    f"FP16 throughput       : "
    f"{baseline_speed:.2f} tokens/sec"
)

print()

print(
    f"Ternary tokens        : "
    f"{ternary_total_tokens}"
)

print(
    f"Ternary total time    : "
    f"{ternary_total_time:.3f} sec"
)

print(
    f"Ternary throughput    : "
    f"{ternary_speed:.2f} tokens/sec"
)


if baseline_speed > 0:

    speed_change = (
        (
            ternary_speed
            - baseline_speed
        )
        / baseline_speed
        * 100
    )

else:

    speed_change = 0


print()

print(
    f"Throughput change     : "
    f"{speed_change:+.2f}%"
)


# =============================================================================
# GPU MEMORY
# =============================================================================

peak_memory_mb = (
    torch.cuda.max_memory_allocated()
    / (1024 ** 2)
)

current_allocated_mb = (
    torch.cuda.memory_allocated()
    / (1024 ** 2)
)

current_reserved_mb = (
    torch.cuda.memory_reserved()
    / (1024 ** 2)
)


print("\n" + "=" * 100)
print("GPU MEMORY")
print("=" * 100)

print(
    f"Peak allocated        : "
    f"{peak_memory_mb:.2f} MB"
)

print(
    f"Current allocated     : "
    f"{current_allocated_mb:.2f} MB"
)

print(
    f"Current reserved      : "
    f"{current_reserved_mb:.2f} MB"
)


# =============================================================================
# IMPORTANT INTERPRETATION
# =============================================================================

print("\n" + "=" * 100)
print("IMPORTANT INTERPRETATION")
print("=" * 100)

print(
    "The selected model weights now contain only "
    "{-alpha, 0, +alpha} values."
)

print()

print(
    "However, those values are still physically stored "
    "inside FP16 tensors."
)

print()

print(
    "Therefore Experiment 003 measures the effect of "
    "TERNARY NUMERICAL CONSTRAINTS on model quality."
)

print()

print(
    "It does NOT yet measure true packed 2-bit model size "
    "or ternary-kernel inference speed."
)

print()

print(
    "Actual packing and optimized ternary inference will "
    "be evaluated in later experiments."
)


print("\n" + "=" * 100)
print("EXPERIMENT 003 COMPLETE")
print("=" * 100)
