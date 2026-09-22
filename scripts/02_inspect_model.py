import torch
from collections import defaultdict
from transformers import Qwen3_5ForCausalLM


MODEL_ID = "Qwen/Qwen3.5-0.8B"


def format_params(n):
    if n >= 1_000_000:
        return f"{n / 1_000_000:.3f} M"
    if n >= 1_000:
        return f"{n / 1_000:.3f} K"
    return str(n)


print("=" * 100)
print("EXPERIMENT 002 — Qwen3.5-0.8B Architecture & Parameter Inspection")
print("=" * 100)

print("\nLoading model...")

model = Qwen3_5ForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="cpu",
)

model.eval()

total_params = sum(p.numel() for p in model.parameters())

print(f"\nTotal parameters: {total_params:,}")
print(f"Total parameters: {total_params / 1e6:.3f} M")


# ----------------------------------------------------------------------
# 1. CONFIGURATION
# ----------------------------------------------------------------------

print("\n" + "=" * 100)
print("MODEL CONFIGURATION")
print("=" * 100)

config = model.config

important_config = [
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "intermediate_size",
    "vocab_size",
    "head_dim",
    "layer_types",
]

for key in important_config:
    if hasattr(config, key):
        print(f"{key:25}: {getattr(config, key)}")


# ----------------------------------------------------------------------
# 2. MODULE TYPE SUMMARY
# ----------------------------------------------------------------------

print("\n" + "=" * 100)
print("MODULE TYPE SUMMARY")
print("=" * 100)

module_counts = defaultdict(int)

for _, module in model.named_modules():
    module_counts[module.__class__.__name__] += 1

for module_type, count in sorted(
    module_counts.items(),
    key=lambda x: (-x[1], x[0])
):
    print(f"{module_type:50} {count}")


# ----------------------------------------------------------------------
# 3. ALL PARAMETER TENSORS
# ----------------------------------------------------------------------

print("\n" + "=" * 100)
print("PARAMETER TENSORS")
print("=" * 100)

print(
    f"{'NAME':70} "
    f"{'SHAPE':25} "
    f"{'PARAMETERS':>15} "
    f"{'DTYPE':>15}"
)

print("-" * 130)

for name, parameter in model.named_parameters():

    shape = str(tuple(parameter.shape))

    print(
        f"{name:70} "
        f"{shape:25} "
        f"{parameter.numel():15,} "
        f"{str(parameter.dtype):>15}"
    )


# ----------------------------------------------------------------------
# 4. PARAMETER GROUPING
# ----------------------------------------------------------------------

groups = defaultdict(int)


def classify_parameter(name):

    lower = name.lower()

    if "embed_tokens" in lower:
        return "Token Embeddings"

    if "lm_head" in lower:
        return "LM Head"

    if "norm" in lower:
        return "Normalization"

    if "mlp" in lower:
        return "MLP"

    if "linear_attn" in lower:
        return "Linear Attention / DeltaNet"

    if "self_attn" in lower:
        return "Full Attention"

    return "Other"


for name, parameter in model.named_parameters():

    group = classify_parameter(name)

    groups[group] += parameter.numel()


print("\n" + "=" * 100)
print("PARAMETER GROUP SUMMARY")
print("=" * 100)

for group, count in sorted(
    groups.items(),
    key=lambda x: -x[1]
):

    percentage = count / total_params * 100

    print(
        f"{group:35} "
        f"{count:15,} "
        f"{format_params(count):>12} "
        f"{percentage:8.2f}%"
    )


# ----------------------------------------------------------------------
# 5. LINEAR LAYER ANALYSIS
# ----------------------------------------------------------------------

print("\n" + "=" * 100)
print("LINEAR LAYER ANALYSIS")
print("=" * 100)

linear_params = 0
linear_layers = 0

for name, module in model.named_modules():

    if isinstance(module, torch.nn.Linear):

        params = sum(
            p.numel()
            for p in module.parameters(recurse=False)
        )

        linear_params += params
        linear_layers += 1

        print(
            f"{name:75} "
            f"{str(module.in_features) + ' -> ' + str(module.out_features):20} "
            f"{params:15,}"
        )


linear_percentage = (
    linear_params / total_params * 100
)


print("\n" + "-" * 100)

print(f"Linear layers      : {linear_layers}")
print(f"Linear parameters  : {linear_params:,}")
print(f"Linear coverage    : {linear_percentage:.2f}%")


# ----------------------------------------------------------------------
# 6. INITIAL TERNARY CANDIDATES
# ----------------------------------------------------------------------

print("\n" + "=" * 100)
print("INITIAL TERNARY CANDIDATE ANALYSIS")
print("=" * 100)

candidate_params = 0
candidate_layers = 0

for name, module in model.named_modules():

    if not isinstance(module, torch.nn.Linear):
        continue

    # Initial conservative policy:
    # large internal Linear layers only.
    #
    # Exclude LM head initially because output projection
    # may be more sensitive.

    if "lm_head" in name.lower():
        continue

    params = module.weight.numel()

    # Ignore very small matrices initially.
    if params < 100_000:
        continue

    candidate_params += params
    candidate_layers += 1

    print(
        f"CANDIDATE | "
        f"{name:70} "
        f"{tuple(module.weight.shape)!s:25} "
        f"{params:15,}"
    )


candidate_percentage = (
    candidate_params / total_params * 100
)


print("\n" + "=" * 100)
print("TERNARY COVERAGE ESTIMATE")
print("=" * 100)

print(f"Total parameters        : {total_params:,}")
print(f"Candidate layers        : {candidate_layers}")
print(f"Candidate parameters    : {candidate_params:,}")
print(f"Candidate coverage      : {candidate_percentage:.2f}%")


# ----------------------------------------------------------------------
# 7. THEORETICAL STORAGE
# ----------------------------------------------------------------------

fp32_gb = total_params * 4 / (1024 ** 3)
fp16_gb = total_params * 2 / (1024 ** 3)

candidate_ternary_bytes = (
    candidate_params * 2 / 8
)

remaining_params = (
    total_params - candidate_params
)

remaining_fp16_bytes = (
    remaining_params * 2
)

hybrid_bytes = (
    candidate_ternary_bytes +
    remaining_fp16_bytes
)

hybrid_gb = hybrid_bytes / (1024 ** 3)


print("\n" + "=" * 100)
print("THEORETICAL STORAGE ESTIMATE")
print("=" * 100)

print(f"FP32 all parameters             : {fp32_gb:.3f} GB")
print(f"FP16 all parameters             : {fp16_gb:.3f} GB")

print(
    f"Hybrid: candidates @2-bit + "
    f"remaining FP16 : {hybrid_gb:.3f} GB"
)

print()
print("NOTE:")
print(
    "The hybrid number is theoretical raw parameter storage only."
)
print(
    "It excludes scales, metadata, packing overhead, runtime buffers,"
)
print(
    "activations, caches, and other inference memory."
)

print("\nExperiment 002 inspection complete.")
