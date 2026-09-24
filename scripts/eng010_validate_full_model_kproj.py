"""
ENG010 — Full-model validation of direct ternary computation.

Controlled comparison
---------------------

Path A:
    frozen ENG003 packed k_proj
        -> reconstruct_packed()
        -> ordinary FP16 nn.Linear

Path B:
    same frozen ENG003 packed k_proj
        -> DirectTernaryLinear
        -> direct ternary arithmetic

The packed representation is identical between both paths.

Only the execution method differs.

Important
---------
DirectTernaryLinear does NOT reconstruct an FP16 weight matrix or FP16
weight-row chunk.

It still decodes packed 2-bit codes into temporary int8 tensors, so this
is NOT a fused 2-bit kernel.

This experiment measures numerical propagation through the complete Qwen
model. It does not claim inference speedup or peak-memory reduction.
"""

import gc
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from packed_ternary.checkpoint import (
    load_packed_checkpoint,
)

from packed_ternary.direct_linear import (
    DirectTernaryLinear,
)

from packed_ternary.packed_weight import (
    reconstruct_packed,
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

ROWS_PER_CHUNK = 64

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


def set_target(model, module):
    model.model.layers[3].self_attn.k_proj = module


def calculate_top1_margin(logits):
    """
    Return top-1 minus top-2 logit margin for each token position.

    Input:
        [batch, sequence, vocabulary]

    Output:
        [batch, sequence]
    """

    top2 = torch.topk(
        logits.float(),
        k=2,
        dim=-1,
    ).values

    return (
        top2[..., 0]
        - top2[..., 1]
    )


def main():
    torch.manual_seed(SEED)

    section(
        "ENG010 — FULL-MODEL DIRECT TERNARY K_PROJ VALIDATION"
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
        ROWS_PER_CHUNK,
    )

    # ------------------------------------------------------------------
    # STEP 1
    # Frozen ENG003 checkpoint.
    # ------------------------------------------------------------------

    section(
        "STEP 1 — LOAD AND VERIFY ENG003 CHECKPOINT"
    )

    metadata, packed_weights = (
        load_packed_checkpoint(
            CHECKPOINT_PATH
        )
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

    packed = packed_weights[TARGET]

    assert (
        packed.original_shape
        == (512, 1024)
    )

    assert (
        packed.padded_numel
        == packed.original_numel
    )

    assert (
        packed.original_shape[1]
        % packed.group_size
        == 0
    )

    print(
        "Checkpoint model        :",
        metadata["model_name"],
    )

    print(
        "Group size              :",
        packed.group_size,
    )

    print(
        "Weight shape            :",
        packed.original_shape,
    )

    print(
        "Parameters              :",
        packed.original_numel,
    )

    print(
        "Packed bytes            :",
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
    # Tokenization.
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
    # Packed k_proj reconstructed into ordinary FP16 Linear.
    # ------------------------------------------------------------------

    section(
        "STEP 2 — PATH A: RECONSTRUCTED FP16 K_PROJ"
    )

    reference_model = load_model()

    reference_target = (
        get_target(
            reference_model
        )
    )

    assert isinstance(
        reference_target,
        torch.nn.Linear,
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
            reference_target.weight.shape
        )
    )

    assert (
        reconstructed_weight.dtype
        == reference_target.weight.dtype
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
    # Same packed k_proj executed through DirectTernaryLinear.
    # ------------------------------------------------------------------

    section(
        "STEP 3 — PATH B: DIRECT TERNARY K_PROJ"
    )

    direct_model = load_model()

    target_before = (
        get_target(
            direct_model
        )
    )

    assert isinstance(
        target_before,
        torch.nn.Linear,
    )

    original_target_object = (
        target_before
    )

    bias = (
        target_before.bias
    )

    direct_target = (
        DirectTernaryLinear(
            packed_weight=packed,
            bias=bias,
            rows_per_chunk=(
                ROWS_PER_CHUNK
            ),
        )
    )

    direct_target = direct_target.to(
        device=target_before.weight.device
    )

    set_target(
        direct_model,
        direct_target,
    )

    target_after = (
        get_target(
            direct_model
        )
    )

    assert isinstance(
        target_after,
        DirectTernaryLinear,
    )

    assert (
        target_after
        is not original_target_object
    )

    parameters = dict(
        target_after.named_parameters()
    )

    buffers = dict(
        target_after.named_buffers()
    )

    print(
        "Before type             :",
        type(
            original_target_object
        ).__name__,
    )

    print(
        "After type              :",
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
        == packed.packed_storage_bytes
    )

    assert (
        target_after
        .fp16_equivalent_weight_bytes
        == packed.original_fp16_bytes
    )

    # Other ENG003 targets must remain ordinary Linear.
    untargeted_out_proj = (
        direct_model
        .model
        .layers[4]
        .linear_attn
        .out_proj
    )

    untargeted_qkv = (
        direct_model
        .model
        .layers[4]
        .linear_attn
        .in_proj_qkv
    )

    assert isinstance(
        untargeted_out_proj,
        torch.nn.Linear,
    )

    assert isinstance(
        untargeted_qkv,
        torch.nn.Linear,
    )

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

    # ------------------------------------------------------------------
    # Full-model forward.
    # ------------------------------------------------------------------

    section(
        "STEP 4 — FULL-MODEL LOGIT COMPARISON"
    )

    with torch.inference_mode():
        direct_logits = (
            direct_model(
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

    assert (
        direct_logits.shape
        == reference_logits.shape
    )

    assert (
        direct_logits.dtype
        == reference_logits.dtype
    )

    reference_float = (
        reference_logits.float()
    )

    direct_float = (
        direct_logits.float()
    )

    signed_difference = (
        direct_float
        - reference_float
    )

    absolute_difference = (
        signed_difference.abs()
    )

    compared_logits = (
        reference_logits.numel()
    )

    mismatches = int(
        (
            direct_logits
            != reference_logits
        )
        .sum()
        .item()
    )

    mismatch_percent = (
        100.0
        * mismatches
        / compared_logits
    )

    max_abs_difference = float(
        absolute_difference.max().item()
    )

    mean_abs_difference = float(
        absolute_difference.mean().item()
    )

    rmse = float(
        torch.sqrt(
            torch.mean(
                signed_difference
                * signed_difference
            )
        ).item()
    )

    # Cosine similarity is evaluated in FP64 because this reduction
    # spans more than five million logits. FP32 accumulation can produce
    # a small numerical overshoot above the mathematical upper bound 1.0.
    reference_cosine = (
        reference_logits
        .double()
        .reshape(
            1,
            -1,
        )
    )

    direct_cosine = (
        direct_logits
        .double()
        .reshape(
            1,
            -1,
        )
    )

    cosine_similarity = float(
        F.cosine_similarity(
            reference_cosine,
            direct_cosine,
            dim=1,
        )
        .item()
    )

    # Guard only against tiny floating-point boundary overshoot.
    cosine_similarity = max(
        -1.0,
        min(
            1.0,
            cosine_similarity,
        ),
    )

    # ------------------------------------------------------------------
    # Behavioral comparison: top-1 token per prompt position.
    # ------------------------------------------------------------------

    reference_top1 = (
        reference_logits.argmax(
            dim=-1
        )
    )

    direct_top1 = (
        direct_logits.argmax(
            dim=-1
        )
    )

    top1_match = (
        reference_top1
        == direct_top1
    )

    top1_total = (
        top1_match.numel()
    )

    top1_matches = int(
        top1_match.sum().item()
    )

    top1_agreement = (
        100.0
        * top1_matches
        / top1_total
    )

    differing_positions = (
        torch.nonzero(
            ~top1_match,
            as_tuple=False,
        )
        .cpu()
        .tolist()
    )

    # ------------------------------------------------------------------
    # Top-1 margins.
    # ------------------------------------------------------------------

    reference_margin = (
        calculate_top1_margin(
            reference_logits
        )
    )

    direct_margin = (
        calculate_top1_margin(
            direct_logits
        )
    )

    margin_difference = (
        direct_margin
        - reference_margin
    ).abs()

    reference_min_margin = float(
        reference_margin.min().item()
    )

    direct_min_margin = float(
        direct_margin.min().item()
    )

    max_margin_difference = float(
        margin_difference.max().item()
    )

    mean_margin_difference = float(
        margin_difference.mean().item()
    )

    print(
        "Reference logits shape  :",
        tuple(
            reference_logits.shape
        ),
    )

    print(
        "Direct logits shape     :",
        tuple(
            direct_logits.shape
        ),
    )

    print(
        "Compared logits         :",
        compared_logits,
    )

    print(
        "Logit mismatches        :",
        mismatches,
    )

    print(
        "Mismatch percent        :",
        mismatch_percent,
    )

    print(
        "Max abs difference      :",
        max_abs_difference,
    )

    print(
        "Mean abs difference     :",
        mean_abs_difference,
    )

    print(
        "RMSE                    :",
        rmse,
    )

    print(
        "Cosine similarity       :",
        cosine_similarity,
    )

    print(
        "Dtype match             :",
        direct_logits.dtype
        == reference_logits.dtype,
    )

    print()
    print(
        "Prompt positions        :",
        top1_total,
    )

    print(
        "Top-1 matches           :",
        top1_matches,
    )

    print(
        "Top-1 agreement         :",
        f"{top1_agreement:.6f}%",
    )

    print(
        "Differing positions     :",
        differing_positions,
    )

    print()
    print(
        "Reference min margin    :",
        reference_min_margin,
    )

    print(
        "Direct min margin       :",
        direct_min_margin,
    )

    print(
        "Max margin difference   :",
        max_margin_difference,
    )

    print(
        "Mean margin difference  :",
        mean_margin_difference,
    )

    # ------------------------------------------------------------------
    # Final summary.
    # ------------------------------------------------------------------

    section(
        "ENG010 FULL-MODEL K_PROJ SUMMARY"
    )

    print(
        "Selected matrices       :",
        1,
    )

    print(
        "Selected parameters     :",
        packed.original_numel,
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
        "Persistent compression  :",
        f"{packed.compression_factor:.6f}x",
    )

    print(
        "Persistent FP16 weight  :",
        False,
    )

    print(
        "Temporary FP16 weight   :",
        False,
    )

    print(
        "Compared logits         :",
        compared_logits,
    )

    print(
        "Logit mismatch percent  :",
        mismatch_percent,
    )

    print(
        "Max abs logit diff      :",
        max_abs_difference,
    )

    print(
        "Mean abs logit diff     :",
        mean_abs_difference,
    )

    print(
        "Logit RMSE              :",
        rmse,
    )

    print(
        "Logit cosine similarity :",
        cosine_similarity,
    )

    print(
        "Top-1 agreement         :",
        f"{top1_agreement:.6f}%",
    )

    print(
        "Differing top-1 pos.    :",
        len(
            differing_positions
        ),
    )

    print()
    print(
        "Scope:"
    )

    print(
        "  - One frozen ENG003 matrix only."
    )

    print(
        "  - Direct path does not reconstruct FP16 weight tensors."
    )

    print(
        "  - Packed codes are still decoded to temporary int8 tensors."
    )

    print(
        "  - No inference-speedup claim."
    )

    print(
        "  - No peak RAM/VRAM reduction claim."
    )

    print(
        "  - No fused 2-bit kernel claim."
    )

    print()
    print(
        "ENG010 FULL-MODEL DIRECT K_PROJ VALIDATION: COMPLETE"
    )


if __name__ == "__main__":
    main()
