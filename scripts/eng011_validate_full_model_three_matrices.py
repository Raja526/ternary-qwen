"""
ENG011 — Full-model three-matrix streaming direct ternary validation.

Controlled comparison
---------------------

Path A:
    frozen ENG003 packed checkpoint
        -> ENG010 DirectTernaryLinear
        -> full-matrix int8 ternary-code decode
        -> direct ternary arithmetic

Path B:
    same frozen ENG003 packed checkpoint
        -> ENG011 StreamingDirectTernaryLinear
        -> row-chunk-local int8 ternary-code decode
        -> same direct ternary arithmetic

Selected matrices:
    model.layers.3.self_attn.k_proj
    model.layers.4.linear_attn.out_proj
    model.layers.4.linear_attn.in_proj_qkv

Controlled variable:
    temporary ternary-code decoding strategy.

Expected result:
    Path B must be bit-exact with Path A.

This experiment does NOT claim:
    - inference speedup,
    - measured peak RAM/VRAM reduction,
    - fused 2-bit execution,
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

from packed_ternary.direct_linear import (
    DirectTernaryLinear,
)

from packed_ternary.direct_runtime_loader import (
    load_direct_runtime_modules,
)

from packed_ternary.streaming_direct_linear import (
    StreamingDirectTernaryLinear,
)

from packed_ternary.streaming_runtime_loader import (
    load_streaming_runtime_modules,
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


def main():
    torch.manual_seed(SEED)

    section(
        "ENG011 — FULL-MODEL THREE-MATRIX STREAMING VALIDATION"
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

    total_full_decode_bytes = 0
    total_max_chunk_decode_bytes = 0

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

        assert (
            packed.original_shape[1]
            % 4
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

        out_features = (
            packed.original_shape[0]
        )

        in_features = (
            packed.original_shape[1]
        )

        full_decode_bytes = (
            packed.original_numel
        )

        max_chunk_decode_bytes = (
            min(
                ROWS_PER_CHUNK,
                out_features,
            )
            * in_features
        )

        total_full_decode_bytes += (
            full_decode_bytes
        )

        total_max_chunk_decode_bytes += (
            max_chunk_decode_bytes
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

        print(
            "ENG010 full decode bytes:",
            full_decode_bytes,
        )

        print(
            "ENG011 max chunk bytes  :",
            max_chunk_decode_bytes,
        )

        print(
            "Decode tensor ratio     :",
            f"{full_decode_bytes / max_chunk_decode_bytes:.6f}x",
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

    print(
        "Sum full decode bytes   :",
        total_full_decode_bytes,
    )

    print(
        "Sum max chunk bytes     :",
        total_max_chunk_decode_bytes,
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
    # STEP 2 — Path A: frozen ENG010 direct runtime.
    # ------------------------------------------------------------------

    section(
        "STEP 2 — PATH A: ENG010 DIRECT TERNARY RUNTIME"
    )

    direct_model = (
        load_model()
    )

    direct_result = (
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

    assert (
        direct_result.loaded_module_names
        == TARGETS
    )

    assert (
        direct_result.replaced_matrix_count
        == 3
    )

    assert (
        direct_result.replaced_parameter_count
        == total_parameters
    )

    assert (
        direct_result.persistent_packed_bytes
        == total_packed_bytes
    )

    assert (
        direct_result.fp16_equivalent_bytes
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
            "weight"
            not in dict(
                module.named_parameters()
            )
        )

        print(
            "ENG010 target           :",
            name,
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
            .clone()
        )

    print(
        "ENG010 logits shape     :",
        tuple(
            direct_logits.shape
        ),
    )

    print(
        "ENG010 logits dtype     :",
        direct_logits.dtype,
    )

    del direct_model

    gc.collect()

    # ------------------------------------------------------------------
    # STEP 3 — Path B: ENG011 streaming runtime.
    # ------------------------------------------------------------------

    section(
        "STEP 3 — PATH B: ENG011 STREAMING DIRECT RUNTIME"
    )

    streaming_model = (
        load_model()
    )

    streaming_result = (
        load_streaming_runtime_modules(
            model=streaming_model,
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

    assert (
        streaming_result.loaded_module_names
        == TARGETS
    )

    assert (
        streaming_result.replaced_matrix_count
        == 3
    )

    assert (
        streaming_result.replaced_parameter_count
        == total_parameters
    )

    assert (
        streaming_result.persistent_packed_bytes
        == total_packed_bytes
    )

    assert (
        streaming_result.fp16_equivalent_bytes
        == total_fp16_bytes
    )

    for name in TARGETS:
        module = get_module(
            streaming_model,
            name,
        )

        packed = packed_weights[
            name
        ]

        assert isinstance(
            module,
            StreamingDirectTernaryLinear,
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

        expected_full_decode = (
            packed.original_numel
        )

        expected_chunk_decode = (
            min(
                ROWS_PER_CHUNK,
                packed.original_shape[0],
            )
            * packed.original_shape[1]
        )

        assert (
            module.full_decoded_code_bytes
            == expected_full_decode
        )

        assert (
            module.maximum_chunk_decoded_code_bytes
            == expected_chunk_decode
        )

        print()
        print(
            "ENG011 target           :",
            name,
        )

        print(
            "Type                    :",
            type(module).__name__,
        )

        print(
            "Persistent bytes        :",
            module.persistent_weight_bytes,
        )

        print(
            "FP16 equivalent bytes   :",
            module.fp16_equivalent_weight_bytes,
        )

        print(
            "ENG010 full decode bytes:",
            module.full_decoded_code_bytes,
        )

        print(
            "ENG011 max chunk bytes  :",
            module.maximum_chunk_decoded_code_bytes,
        )

        print(
            "Decode tensor ratio     :",
            f"{module.full_decoded_code_bytes / module.maximum_chunk_decoded_code_bytes:.6f}x",
        )

    with torch.inference_mode():
        streaming_logits = (
            streaming_model(
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
        "ENG011 logits shape     :",
        tuple(
            streaming_logits.shape
        ),
    )

    print(
        "ENG011 logits dtype     :",
        streaming_logits.dtype,
    )

    del streaming_model

    gc.collect()

    # ------------------------------------------------------------------
    # STEP 4 — Strict full-model equivalence.
    # ------------------------------------------------------------------

    section(
        "STEP 4 — ENG010 VS ENG011 LOGIT COMPARISON"
    )

    assert (
        streaming_logits.shape
        == direct_logits.shape
    )

    assert (
        streaming_logits.dtype
        == direct_logits.dtype
    )

    compared_logits = (
        direct_logits.numel()
    )

    mismatches = int(
        (
            streaming_logits
            != direct_logits
        )
        .sum()
        .item()
    )

    difference = (
        streaming_logits.float()
        - direct_logits.float()
    )

    absolute_difference = (
        difference.abs()
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
                difference
                * difference
            )
        ).item()
    )

    bit_exact = torch.equal(
        direct_logits,
        streaming_logits,
    )

    direct_top1 = (
        direct_logits.argmax(
            dim=-1
        )
    )

    streaming_top1 = (
        streaming_logits.argmax(
            dim=-1
        )
    )

    top1_matches = int(
        (
            direct_top1
            == streaming_top1
        )
        .sum()
        .item()
    )

    top1_total = (
        direct_top1.numel()
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
        "Bit-exact logits        :",
        bit_exact,
    )

    print(
        "Top-1 matches           :",
        f"{top1_matches}/{top1_total}",
    )

    # ------------------------------------------------------------------
    # Strict acceptance criteria.
    # ------------------------------------------------------------------

    assert (
        mismatches
        == 0
    )

    assert (
        max_abs_difference
        == 0.0
    )

    assert (
        mean_abs_difference
        == 0.0
    )

    assert (
        rmse
        == 0.0
    )

    assert bit_exact

    assert (
        top1_matches
        == top1_total
    )

    # ------------------------------------------------------------------
    # Final summary.
    # ------------------------------------------------------------------

    section(
        "ENG011 THREE-MATRIX FULL-MODEL SUMMARY"
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
        "Rows/chunk              :",
        ROWS_PER_CHUNK,
    )

    print(
        "ENG010 decode strategy  :",
        "full selected matrix",
    )

    print(
        "ENG011 decode strategy  :",
        "current output-row chunk only",
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
        "Max abs difference      :",
        max_abs_difference,
    )

    print(
        "Bit-exact logits        :",
        bit_exact,
    )

    print()
    print(
        "NOTE: decode byte counts above are tensor-size accounting."
    )

    print(
        "They are NOT measured process RAM/VRAM results."
    )

    print()
    print(
        "ENG011 THREE-MATRIX FULL-MODEL VALIDATION: PASS"
    )


if __name__ == "__main__":
    main()
