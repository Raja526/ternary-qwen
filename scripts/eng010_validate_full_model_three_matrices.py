"""
ENG010 — Full-model three-matrix direct ternary validation.

Controlled comparison
---------------------

Path A:
    frozen ENG003 packed checkpoint
        -> reconstruct all three selected weights to FP16
        -> ordinary nn.Linear modules

Path B:
    same frozen ENG003 packed checkpoint
        -> DirectTernaryLinear for all three selected modules
        -> direct ternary arithmetic

Selected matrices:
    model.layers.3.self_attn.k_proj
    model.layers.4.linear_attn.out_proj
    model.layers.4.linear_attn.in_proj_qkv

Only the execution method differs between Path A and Path B.

DirectTernaryLinear does not reconstruct FP16 weight matrices or FP16
weight-row chunks. It still decodes packed 2-bit codes into temporary
int8 tensors and uses temporary arithmetic tensors.

This experiment does NOT claim:
    - inference speedup,
    - peak RAM/VRAM reduction,
    - fused 2-bit execution,
    - whole-model compression.
"""

import gc
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

from packed_ternary.direct_runtime_loader import (
    load_direct_runtime_modules,
)

from packed_ternary.packed_weight import (
    reconstruct_packed,
)


MODEL_NAME = "Qwen/Qwen3.5-0.8B"

CHECKPOINT_PATH = Path(
    "results/ENG003_multimatrix_checkpoint.pt"
)

TARGETS = (
    "model.layers.3.self_attn.k_proj",
    "model.layers.4.linear_attn.out_proj",
    "model.layers.4.linear_attn.in_proj_qkv",
)

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


def get_module(
    model,
    module_name,
):
    modules = dict(
        model.named_modules()
    )

    if module_name not in modules:
        raise KeyError(
            f"Module not found: {module_name}"
        )

    return modules[module_name]


def calculate_top1_margin(
    logits,
):
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
        "ENG010 — FULL-MODEL THREE-MATRIX DIRECT TERNARY VALIDATION"
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
        "Rows/chunk              :",
        ROWS_PER_CHUNK,
    )

    # ------------------------------------------------------------------
    # STEP 1 — Frozen checkpoint.
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
        == set(TARGETS)
    )

    total_parameters = 0
    total_packed_bytes = 0
    total_fp16_bytes = 0

    for name in TARGETS:
        packed = packed_weights[
            name
        ]

        assert (
            packed.padded_numel
            == packed.original_numel
        )

        assert (
            packed.original_shape[1]
            % packed.group_size
            == 0
        )

        total_parameters += (
            packed.original_numel
        )

        total_packed_bytes += (
            packed.packed_storage_bytes
        )

        total_fp16_bytes += (
            packed.original_fp16_bytes
        )

        print()
        print(
            "Target                  :",
            name,
        )

        print(
            "Shape                   :",
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
            "FP16 bytes              :",
            packed.original_fp16_bytes,
        )

        print(
            "Compression             :",
            f"{packed.compression_factor:.6f}x",
        )

    aggregate_compression = (
        total_fp16_bytes
        / total_packed_bytes
    )

    print()
    print(
        "Total parameters        :",
        total_parameters,
    )

    print(
        "Total packed bytes      :",
        total_packed_bytes,
    )

    print(
        "Total FP16 bytes        :",
        total_fp16_bytes,
    )

    print(
        "Aggregate compression   :",
        f"{aggregate_compression:.6f}x",
    )

    assert (
        total_parameters
        == 8_912_896
    )

    assert (
        total_packed_bytes
        == 2_367_488
    )

    assert (
        total_fp16_bytes
        == 17_825_792
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
    # STEP 2 — Path A.
    # ------------------------------------------------------------------

    section(
        "STEP 2 — PATH A: THREE RECONSTRUCTED FP16 LINEARS"
    )

    reference_model = (
        load_model()
    )

    for name in TARGETS:
        module = get_module(
            reference_model,
            name,
        )

        packed = packed_weights[
            name
        ]

        assert isinstance(
            module,
            torch.nn.Linear,
        )

        reconstructed = (
            reconstruct_packed(
                packed
            )
        )

        assert (
            tuple(
                reconstructed.shape
            )
            == tuple(
                module.weight.shape
            )
        )

        assert (
            reconstructed.dtype
            == module.weight.dtype
        )

        with torch.no_grad():
            module.weight.copy_(
                reconstructed
            )

        print(
            "Reconstructed           :",
            name,
        )

        del reconstructed

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

    del reference_model

    gc.collect()

    # ------------------------------------------------------------------
    # STEP 3 — Path B.
    # ------------------------------------------------------------------

    section(
        "STEP 3 — PATH B: THREE DIRECT TERNARY LINEARS"
    )

    direct_model = (
        load_model()
    )

    original_objects = {
        name: get_module(
            direct_model,
            name,
        )
        for name in TARGETS
    }

    for name in TARGETS:
        assert isinstance(
            original_objects[name],
            torch.nn.Linear,
        )

    load_result = (
        load_direct_runtime_modules(
            model=direct_model,
            checkpoint_path=(
                CHECKPOINT_PATH
            ),
            selected_module_names=(
                TARGETS
            ),
            expected_model_name=(
                MODEL_NAME
            ),
            rows_per_chunk=(
                ROWS_PER_CHUNK
            ),
        )
    )

    print(
        "Loaded names            :",
        load_result.loaded_module_names,
    )

    print(
        "Replaced matrices       :",
        load_result.replaced_matrix_count,
    )

    print(
        "Replaced parameters     :",
        load_result.replaced_parameter_count,
    )

    print(
        "Persistent packed bytes :",
        load_result.persistent_packed_bytes,
    )

    print(
        "FP16 equivalent bytes   :",
        load_result.fp16_equivalent_bytes,
    )

    assert (
        load_result.loaded_module_names
        == TARGETS
    )

    assert (
        load_result.replaced_matrix_count
        == 3
    )

    assert (
        load_result.replaced_parameter_count
        == total_parameters
    )

    assert (
        load_result.persistent_packed_bytes
        == total_packed_bytes
    )

    assert (
        load_result.fp16_equivalent_bytes
        == total_fp16_bytes
    )

    for name in TARGETS:
        module = get_module(
            direct_model,
            name,
        )

        assert isinstance(
            module,
            DirectTernaryLinear,
        )

        assert (
            module
            is not original_objects[name]
        )

        parameters = dict(
            module.named_parameters()
        )

        buffers = dict(
            module.named_buffers()
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

        print()
        print(
            "Direct target           :",
            name,
        )

        print(
            "Type                    :",
            type(module).__name__,
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
            "Persistent bytes        :",
            module.persistent_weight_bytes,
        )

        print(
            "FP16 equivalent bytes   :",
            module.fp16_equivalent_weight_bytes,
        )

    # ------------------------------------------------------------------
    # STEP 4 — Full-model forward.
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

    # FP64 reduction avoids the FP32 cosine overshoot observed in the
    # single-matrix ENG010 validation.
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

    cosine_similarity = max(
        -1.0,
        min(
            1.0,
            cosine_similarity,
        ),
    )

    # ------------------------------------------------------------------
    # Top-1 behavior.
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
    # Margin comparison.
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
        "ENG010 THREE-MATRIX FULL-MODEL SUMMARY"
    )

    print(
        "Selected matrices       :",
        3,
    )

    print(
        "Selected parameters     :",
        total_parameters,
    )

    print(
        "Persistent packed bytes :",
        total_packed_bytes,
    )

    print(
        "FP16 equivalent bytes   :",
        total_fp16_bytes,
    )

    print(
        "Persistent compression  :",
        f"{aggregate_compression:.6f}x",
    )

    print(
        "Persistent FP16 weights :",
        False,
    )

    print(
        "Temporary FP16 weights  :",
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
    print("Scope:")

    print(
        "  - Three frozen ENG003 matrices only."
    )

    print(
        "  - Direct path does not reconstruct FP16 weight tensors."
    )

    print(
        "  - Packed codes are still decoded to temporary int8 tensors."
    )

    print(
        "  - Temporary arithmetic tensors are still created."
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

    print(
        "  - No whole-model compression claim."
    )

    print()
    print(
        "ENG010 THREE-MATRIX DIRECT VALIDATION: COMPLETE"
    )


if __name__ == "__main__":
    main()
