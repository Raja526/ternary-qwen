"""
OPT002 — Full-model single-matrix compiled packed ternary validation.

Controlled comparison
---------------------

Path A:
    frozen ENG003 packed checkpoint
        -> ENG011 StreamingDirectTernaryLinear
        -> full Qwen forward

Path B:
    same frozen ENG003 packed checkpoint
        -> OPT001 compiled packed ternary CPU operator
        -> OPT002 CompiledPackedTernaryLinear
        -> full Qwen forward

Selected matrix:
    model.layers.3.self_attn.k_proj

Controlled variable:
    ternary compute implementation.

This experiment validates full-model numerical behavior only.

It does NOT claim:
    - inference speedup,
    - peak RAM/VRAM reduction,
    - CUDA acceleration,
    - whole-model compression.
"""

import gc
from pathlib import Path

import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from packed_ternary.streaming_runtime_loader import (
    load_streaming_runtime_modules,
)

from packed_ternary.compiled_runtime_loader import (
    load_compiled_runtime_modules,
)


MODEL_NAME = "Qwen/Qwen3.5-0.8B"

CHECKPOINT_PATH = Path(
    "results/ENG003_multimatrix_checkpoint.pt"
)

TARGET = (
    "model.layers.3.self_attn.k_proj"
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


def run_model(
    model,
    input_ids,
    attention_mask,
):
    kwargs = {
        "input_ids": input_ids,
    }

    if attention_mask is not None:
        kwargs["attention_mask"] = (
            attention_mask
        )

    with torch.inference_mode():
        output = model(
            **kwargs
        )

    return (
        output.logits
        .detach()
        .cpu()
        .clone()
    )


def main():
    torch.manual_seed(SEED)

    section(
        "OPT002 — FULL-MODEL SINGLE-MATRIX VALIDATION"
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
        "Target                  :",
        TARGET,
    )

    print(
        "Reference               :",
        "ENG011 StreamingDirectTernaryLinear",
    )

    print(
        "Candidate               :",
        "OPT002 CompiledPackedTernaryLinear",
    )

    # --------------------------------------------------------------
    # Tokenization — frozen ENG011 configuration.
    # --------------------------------------------------------------

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

    # --------------------------------------------------------------
    # PATH A — frozen ENG011.
    # --------------------------------------------------------------

    section(
        "PATH A — ENG011 STREAMING RUNTIME"
    )

    reference_model = (
        load_model()
    )

    reference_result = (
        load_streaming_runtime_modules(
            model=reference_model,
            checkpoint_path=CHECKPOINT_PATH,
            selected_module_names=(
                TARGET,
            ),
            expected_model_name=MODEL_NAME,
            rows_per_chunk=ROWS_PER_CHUNK,
        )
    )

    print(
        "Replaced matrices       :",
        reference_result.replaced_matrix_count,
    )

    print(
        "Replaced parameters     :",
        reference_result.replaced_parameter_count,
    )

    reference_logits = run_model(
        reference_model,
        input_ids,
        attention_mask,
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

    # --------------------------------------------------------------
    # PATH B — OPT002 compiled runtime.
    # --------------------------------------------------------------

    section(
        "PATH B — OPT002 COMPILED RUNTIME"
    )

    compiled_model = (
        load_model()
    )

    compiled_result = (
        load_compiled_runtime_modules(
            model=compiled_model,
            checkpoint_path=CHECKPOINT_PATH,
            selected_module_names=(
                TARGET,
            ),
            expected_model_name=MODEL_NAME,
        )
    )

    print(
        "Replaced matrices       :",
        compiled_result.replaced_matrix_count,
    )

    print(
        "Replaced parameters     :",
        compiled_result.replaced_parameter_count,
    )

    print(
        "Persistent packed bytes :",
        compiled_result.persistent_packed_bytes,
    )

    print(
        "FP16 equivalent bytes   :",
        compiled_result.fp16_equivalent_bytes,
    )

    print(
        "Compression factor      :",
        (
            compiled_result.fp16_equivalent_bytes
            / compiled_result.persistent_packed_bytes
        ),
    )

    compiled_logits = run_model(
        compiled_model,
        input_ids,
        attention_mask,
    )

    print(
        "Compiled logits shape   :",
        tuple(
            compiled_logits.shape
        ),
    )

    print(
        "Compiled logits dtype   :",
        compiled_logits.dtype,
    )

    del compiled_model

    gc.collect()

    # --------------------------------------------------------------
    # Comparison.
    # --------------------------------------------------------------

    section(
        "ENG011 VS OPT002 LOGIT COMPARISON"
    )

    assert (
        compiled_logits.shape
        == reference_logits.shape
    )

    assert (
        compiled_logits.dtype
        == reference_logits.dtype
    )

    compared_logits = int(
        reference_logits.numel()
    )

    mismatches = int(
        (
            compiled_logits
            != reference_logits
        )
        .sum()
        .item()
    )

    difference = (
        compiled_logits.float()
        - reference_logits.float()
    )

    absolute_difference = (
        difference.abs()
    )

    max_abs_difference = float(
        absolute_difference
        .max()
        .item()
    )

    mean_abs_difference = float(
        absolute_difference
        .mean()
        .item()
    )

    rmse = float(
        torch.sqrt(
            torch.mean(
                difference
                * difference
            )
        )
        .item()
    )

    reference_flat = (
        reference_logits
        .float()
        .reshape(-1)
    )

    compiled_flat = (
        compiled_logits
        .float()
        .reshape(-1)
    )

    cosine_similarity = float(
        torch.nn.functional
        .cosine_similarity(
            reference_flat.unsqueeze(0),
            compiled_flat.unsqueeze(0),
            dim=1,
        )
        .item()
    )

    reference_top1 = (
        reference_logits.argmax(
            dim=-1
        )
    )

    compiled_top1 = (
        compiled_logits.argmax(
            dim=-1
        )
    )

    top1_matches = int(
        (
            reference_top1
            == compiled_top1
        )
        .sum()
        .item()
    )

    top1_total = int(
        reference_top1.numel()
    )

    mismatch_percentage = (
        100.0
        * mismatches
        / compared_logits
    )

    print(
        "Compared logits         :",
        compared_logits,
    )

    print(
        "Exact mismatches        :",
        mismatches,
    )

    print(
        "Mismatch percentage     :",
        mismatch_percentage,
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
        "Top-1 matches           :",
        f"{top1_matches}/{top1_total}",
    )

    # --------------------------------------------------------------
    # Acceptance criteria.
    #
    # OPT001 arithmetic is not expected to be bit-exact with ENG011.
    # Therefore acceptance is based on bounded numerical difference
    # and prediction preservation, not exact FP16 equality.
    # --------------------------------------------------------------

    assert torch.isfinite(
        compiled_logits
    ).all()

    assert (
        max_abs_difference
        <= 0.1
    )

    assert (
        mean_abs_difference
        <= 0.01
    )

    assert (
        rmse
        <= 0.02
    )

    assert (
        cosine_similarity
        >= 0.999
    )

    assert (
        top1_matches
        == top1_total
    )

    section(
        "OPT002 SINGLE-MATRIX FULL-MODEL SUMMARY"
    )

    print(
        "Selected matrices       :",
        1,
    )

    print(
        "Selected parameters     :",
        compiled_result.replaced_parameter_count,
    )

    print(
        "Compared logits         :",
        compared_logits,
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
        "Top-1 agreement         :",
        f"{top1_matches}/{top1_total}",
    )

    print()

    print(
        "OPT002 SINGLE-MATRIX FULL-MODEL VALIDATION: PASS"
    )


if __name__ == "__main__":
    main()
