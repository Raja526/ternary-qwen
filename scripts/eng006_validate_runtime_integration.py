"""
ENG006 — Real Qwen packed-runtime integration validation.

Objective
---------
Validate that the reusable packed runtime loader can:

1. Load the frozen ENG003 multi-matrix checkpoint.
2. Select exactly ONE packed matrix:
       model.layers.3.self_attn.k_proj
3. Replace the corresponding nn.Linear with PackedTernaryLinear.
4. Leave selected runtime storage as packed codes + scales.
5. Produce exactly the same full-model logits as an ordinary nn.Linear
   containing the reconstructed FP16 ternary weight.

Controlled variable
-------------------
Path A:
    ENG003 packed k_proj
        -> reconstruct_packed()
        -> ordinary nn.Linear

Path B:
    Same ENG003 packed k_proj
        -> load_packed_runtime_modules()
        -> PackedTernaryLinear

The quantized representation is therefore identical between both paths.
Only the runtime representation/integration mechanism differs.

This experiment does NOT claim:
    - inference speedup,
    - peak RAM/VRAM reduction,
    - fused 2-bit CUDA execution,
    - whole-model compression.
"""

import gc
from pathlib import Path

import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from packed_ternary.checkpoint import (
    load_packed_checkpoint,
)

from packed_ternary.packed_linear import (
    PackedTernaryLinear,
)

from packed_ternary.packed_weight import (
    reconstruct_packed,
)

from packed_ternary.runtime_loader import (
    load_packed_runtime_modules,
)


MODEL_NAME = "Qwen/Qwen3.5-0.8B"

CHECKPOINT_PATH = Path(
    "results/ENG003_multimatrix_checkpoint.pt"
)

TARGET = (
    "model.layers.3.self_attn.k_proj"
)

EXPECTED_CHECKPOINT_TARGETS = {
    "model.layers.3.self_attn.k_proj",
    "model.layers.4.linear_attn.out_proj",
    "model.layers.4.linear_attn.in_proj_qkv",
}

REAL_ROWS_PER_CHUNK = 64

PROMPT = (
    "A box contains 12 red balls and 8 blue balls. "
    "How many balls are in the box?"
)

SEED = 20260924


def section(title):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def load_model():

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            MODEL_NAME,
            dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
    )

    model.eval()

    return model


def get_target(model):

    return (
        model
        .model
        .layers[3]
        .self_attn
        .k_proj
    )


def main():

    torch.manual_seed(
        SEED
    )

    section(
        "ENG006 — REAL QWEN PACKED RUNTIME INTEGRATION"
    )

    print(
        "Model                   :",
        MODEL_NAME,
    )

    print(
        "Checkpoint              :",
        CHECKPOINT_PATH,
    )

    print(
        "Selected target         :",
        TARGET,
    )

    print(
        "Rows/chunk              :",
        REAL_ROWS_PER_CHUNK,
    )

    # ------------------------------------------------------------------
    # Load frozen ENG003 checkpoint.
    # ------------------------------------------------------------------

    section(
        "STEP 1 — LOAD AND VERIFY ENG003 CHECKPOINT"
    )

    metadata, packed_weights = (
        load_packed_checkpoint(
            CHECKPOINT_PATH
        )
    )

    print(
        "Checkpoint model        :",
        metadata["model_name"],
    )

    print(
        "Checkpoint group size   :",
        metadata["group_size"],
    )

    print(
        "Threshold multiplier    :",
        metadata[
            "threshold_multiplier"
        ],
    )

    print(
        "Checkpoint matrices     :",
        len(packed_weights),
    )

    for name in packed_weights:
        print(
            "  -",
            name,
        )

    assert (
        metadata["model_name"]
        == MODEL_NAME
    )

    assert (
        set(packed_weights.keys())
        == EXPECTED_CHECKPOINT_TARGETS
    )

    assert TARGET in packed_weights

    packed = packed_weights[
        TARGET
    ]

    print()
    print(
        "Selected parameters     :",
        packed.original_numel,
    )

    print(
        "Packed code bytes       :",
        packed.code_bytes,
    )

    print(
        "Scale bytes             :",
        packed.scale_bytes,
    )

    print(
        "Persistent packed bytes :",
        packed.packed_storage_bytes,
    )

    print(
        "FP16 equivalent bytes   :",
        packed.original_fp16_bytes,
    )

    print(
        "Compression factor      :",
        f"{packed.compression_factor:.6f}x",
    )

    # ------------------------------------------------------------------
    # Tokenization: identical to ENG005.
    # ------------------------------------------------------------------

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            MODEL_NAME
        )
    )

    encoded = tokenizer(
        PROMPT,
        return_tensors="pt",
        add_special_tokens=True,
    )

    input_ids = (
        encoded["input_ids"]
        .cpu()
    )

    attention_mask = (
        encoded.get(
            "attention_mask"
        )
    )

    if attention_mask is not None:
        attention_mask = (
            attention_mask.cpu()
        )

    print(
        "Input tokens            :",
        input_ids.shape[1],
    )

    # ------------------------------------------------------------------
    # PATH A
    #
    # Same ENG003 packed matrix, reconstructed into an ordinary Linear.
    # ------------------------------------------------------------------

    section(
        "STEP 2 — PATH A: RECONSTRUCTED FP16 ORDINARY LINEAR"
    )

    reference_model = (
        load_model()
    )

    reference_target = (
        get_target(
            reference_model
        )
    )

    assert isinstance(
        reference_target,
        torch.nn.Linear,
    )

    print(
        "Original target type    :",
        type(
            reference_target
        ).__name__,
    )

    print(
        "Original weight shape   :",
        tuple(
            reference_target
            .weight
            .shape
        ),
    )

    print(
        "Original weight dtype   :",
        reference_target
        .weight
        .dtype,
    )

    reconstructed_weight = (
        reconstruct_packed(
            packed
        )
    )

    assert (
        tuple(
            reconstructed_weight.shape
        )
        == tuple(
            reference_target
            .weight
            .shape
        )
    )

    assert (
        reconstructed_weight.dtype
        == reference_target
        .weight
        .dtype
    )

    with torch.no_grad():

        reference_target.weight.copy_(
            reconstructed_weight
        )

    with torch.inference_mode():

        reference_logits = (
            reference_model(
                input_ids=input_ids,
                attention_mask=(
                    attention_mask
                ),
                use_cache=False,
            )
            .logits
            .detach()
            .cpu()
            .clone()
        )

    print(
        "Reference logits shape  :",
        tuple(
            reference_logits.shape
        ),
    )

    print(
        "Reference logits dtype  :",
        reference_logits.dtype,
    )

    del reference_target
    del reference_model
    del reconstructed_weight

    gc.collect()

    # ------------------------------------------------------------------
    # PATH B
    #
    # Fresh model + reusable ENG006 runtime loader.
    # ------------------------------------------------------------------

    section(
        "STEP 3 — PATH B: REUSABLE PACKED RUNTIME LOADER"
    )

    runtime_model = (
        load_model()
    )

    target_before = (
        get_target(
            runtime_model
        )
    )

    assert isinstance(
        target_before,
        torch.nn.Linear,
    )

    original_target_object = (
        target_before
    )

    print(
        "Before loader type      :",
        type(
            target_before
        ).__name__,
    )

    load_result = (
        load_packed_runtime_modules(
            model=runtime_model,
            checkpoint_path=(
                CHECKPOINT_PATH
            ),
            selected_module_names=[
                TARGET,
            ],
            expected_model_name=(
                MODEL_NAME
            ),
            rows_per_chunk=(
                REAL_ROWS_PER_CHUNK
            ),
        )
    )

    target_after = (
        get_target(
            runtime_model
        )
    )

    print(
        "After loader type       :",
        type(
            target_after
        ).__name__,
    )

    print(
        "Object replaced         :",
        target_after
        is not original_target_object,
    )

    print(
        "Loaded module names     :",
        load_result
        .loaded_module_names,
    )

    print(
        "Replaced matrices       :",
        load_result
        .replaced_matrix_count,
    )

    print(
        "Replaced parameters     :",
        load_result
        .replaced_parameter_count,
    )

    print(
        "Loader packed bytes     :",
        load_result
        .persistent_packed_bytes,
    )

    print(
        "Loader FP16 bytes       :",
        load_result
        .fp16_equivalent_bytes,
    )

    assert isinstance(
        target_after,
        PackedTernaryLinear,
    )

    assert (
        target_after
        is not original_target_object
    )

    assert (
        load_result
        .loaded_module_names
        == (TARGET,)
    )

    assert (
        load_result
        .replaced_matrix_count
        == 1
    )

    assert (
        load_result
        .replaced_parameter_count
        == packed.original_numel
    )

    # ------------------------------------------------------------------
    # Inspect persistent representation.
    # ------------------------------------------------------------------

    parameters = dict(
        target_after.named_parameters()
    )

    buffers = dict(
        target_after.named_buffers()
    )

    print()
    print(
        "Full weight Parameter   :",
        "weight"
        in parameters,
    )

    print(
        "Packed codes buffer     :",
        "packed_codes"
        in buffers,
    )

    print(
        "Scales buffer           :",
        "scales"
        in buffers,
    )

    print(
        "Persistent weight bytes :",
        target_after
        .persistent_weight_bytes,
    )

    print(
        "FP16 equivalent bytes   :",
        target_after
        .fp16_equivalent_weight_bytes,
    )

    print(
        "Persistent compression  :",
        f"{target_after.persistent_compression_factor:.6f}x",
    )

    assert (
        "weight"
        not in parameters
    )

    assert (
        "packed_codes"
        in buffers
    )

    assert (
        "scales"
        in buffers
    )

    assert (
        target_after
        .persistent_weight_bytes
        == packed
        .packed_storage_bytes
    )

    assert (
        target_after
        .fp16_equivalent_weight_bytes
        == packed
        .original_fp16_bytes
    )

    assert (
        load_result
        .persistent_packed_bytes
        == packed
        .packed_storage_bytes
    )

    assert (
        load_result
        .fp16_equivalent_bytes
        == packed
        .original_fp16_bytes
    )

    # ------------------------------------------------------------------
    # Confirm the other ENG003 matrices were NOT replaced.
    #
    # This is critical for ENG006:
    # checkpoint contains 3 matrices, but selected subset contains only 1.
    # ------------------------------------------------------------------

    untargeted_out_proj = (
        runtime_model
        .model
        .layers[4]
        .linear_attn
        .out_proj
    )

    untargeted_qkv = (
        runtime_model
        .model
        .layers[4]
        .linear_attn
        .in_proj_qkv
    )

    print()
    print(
        "Untargeted out_proj type:",
        type(
            untargeted_out_proj
        ).__name__,
    )

    print(
        "Untargeted qkv type     :",
        type(
            untargeted_qkv
        ).__name__,
    )

    assert isinstance(
        untargeted_out_proj,
        torch.nn.Linear,
    )

    assert isinstance(
        untargeted_qkv,
        torch.nn.Linear,
    )

    # ------------------------------------------------------------------
    # Full-model forward.
    # ------------------------------------------------------------------

    section(
        "STEP 4 — FULL-MODEL LOGIT COMPARISON"
    )

    with torch.inference_mode():

        runtime_logits = (
            runtime_model(
                input_ids=input_ids,
                attention_mask=(
                    attention_mask
                ),
                use_cache=False,
            )
            .logits
            .detach()
            .cpu()
        )

    difference = (
        runtime_logits.float()
        - reference_logits.float()
    ).abs()

    mismatches = int(
        (
            runtime_logits
            != reference_logits
        )
        .sum()
        .item()
    )

    max_difference = float(
        difference.max().item()
    )

    mean_difference = float(
        difference.mean().item()
    )

    print(
        "Reference logits shape  :",
        tuple(
            reference_logits.shape
        ),
    )

    print(
        "Runtime logits shape    :",
        tuple(
            runtime_logits.shape
        ),
    )

    print(
        "Compared logits         :",
        reference_logits.numel(),
    )

    print(
        "Logit mismatches        :",
        mismatches,
    )

    print(
        "Max abs difference      :",
        max_difference,
    )

    print(
        "Mean abs difference     :",
        mean_difference,
    )

    print(
        "Dtype match             :",
        runtime_logits.dtype
        == reference_logits.dtype,
    )

    assert (
        runtime_logits.shape
        == reference_logits.shape
    )

    assert (
        runtime_logits.dtype
        == reference_logits.dtype
    )

    assert mismatches == 0
    assert max_difference == 0.0

    section(
        "ENG006 FINAL SUMMARY"
    )

    print(
        "Real target             :",
        TARGET,
    )

    print(
        "Checkpoint matrices     :",
        len(packed_weights),
    )

    print(
        "Runtime matrices loaded :",
        load_result
        .replaced_matrix_count,
    )

    print(
        "Runtime parameters      :",
        load_result
        .replaced_parameter_count,
    )

    print(
        "Persistent packed bytes :",
        load_result
        .persistent_packed_bytes,
    )

    print(
        "FP16 equivalent bytes   :",
        load_result
        .fp16_equivalent_bytes,
    )

    compression = (
        load_result
        .fp16_equivalent_bytes
        / load_result
        .persistent_packed_bytes
    )

    print(
        "Compression factor      :",
        f"{compression:.6f}x",
    )

    print(
        "Persistent full weight  :",
        "weight"
        in dict(
            target_after
            .named_parameters()
        ),
    )

    print(
        "Compared logits         :",
        reference_logits.numel(),
    )

    print(
        "Logit mismatches        :",
        mismatches,
    )

    print(
        "Max abs difference      :",
        max_difference,
    )

    print()
    print("Scope:")
    print(
        "  The ENG003 checkpoint contains three packed matrices."
    )
    print(
        "  ENG006 selected and runtime-loaded exactly one matrix."
    )
    print(
        "  The selected matrix persists as packed codes + scales."
    )
    print(
        "  Temporary row chunks are reconstructed during forward execution."
    )
    print(
        "  No inference speedup is claimed."
    )
    print(
        "  No peak RAM/VRAM reduction is claimed."
    )
    print(
        "  No fused 2-bit CUDA kernel is claimed."
    )
    print(
        "  No whole-model compression is claimed."
    )

    section(
        "ENG006 REAL QWEN PACKED RUNTIME INTEGRATION: PASS"
    )


if __name__ == "__main__":
    main()
