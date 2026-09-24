"""
ENG011-E — Fixed multi-prompt streaming equivalence validation.

Objective
---------
Compare the same three frozen ENG003 ternary matrices using:

Path A:
    ENG010 DirectTernaryLinear
    with full selected-matrix int8 code decoding

Path B:
    ENG011 StreamingDirectTernaryLinear
    with current output-row-chunk int8 code decoding

Everything else is held fixed:
    - model
    - checkpoint
    - selected matrices
    - packed representation
    - scales
    - rows_per_chunk
    - prompts
    - direct ternary arithmetic

Expected result:
    ENG011 must be bit-exact with ENG010 for every prompt.

This experiment does NOT claim:
    - general behavioral equivalence,
    - inference speedup,
    - measured peak RAM/VRAM reduction,
    - fused 2-bit execution,
    - whole-model compression.
"""

import gc
import math
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

SEED = 20260924


# Frozen ENG010 fixed prompt suite.
PROMPTS = (
    "A box contains 12 red balls and 8 blue balls. How many balls are in the box?",
    "What is 17 multiplied by 6?",
    "Explain in one sentence why plants need sunlight.",
    "Complete the sequence: 2, 4, 8, 16,",
    "The capital of France is",
    "If a train travels 60 miles in one hour, how far does it travel in three hours?",
    "Write a short greeting to a friend.",
    "Which is larger: 3/4 or 2/3?",
    "Water freezes at what temperature in degrees Celsius?",
    "A student has 25 pencils and gives 7 away. How many pencils remain?",
)


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


def prepare_direct_model():
    model = load_model()

    result = (
        load_direct_runtime_modules(
            model=model,
            checkpoint_path=CHECKPOINT_PATH,
            selected_module_names=TARGETS,
            expected_model_name=MODEL_NAME,
            rows_per_chunk=ROWS_PER_CHUNK,
        )
    )

    assert (
        result.loaded_module_names
        == TARGETS
    )

    assert (
        result.replaced_matrix_count
        == 3
    )

    for name in TARGETS:
        assert isinstance(
            get_module(
                model,
                name,
            ),
            DirectTernaryLinear,
        )

    return model, result


def prepare_streaming_model():
    model = load_model()

    result = (
        load_streaming_runtime_modules(
            model=model,
            checkpoint_path=CHECKPOINT_PATH,
            selected_module_names=TARGETS,
            expected_model_name=MODEL_NAME,
            rows_per_chunk=ROWS_PER_CHUNK,
        )
    )

    assert (
        result.loaded_module_names
        == TARGETS
    )

    assert (
        result.replaced_matrix_count
        == 3
    )

    for name in TARGETS:
        assert isinstance(
            get_module(
                model,
                name,
            ),
            StreamingDirectTernaryLinear,
        )

    return model, result


def forward_prompt(
    model,
    tokenizer,
    prompt,
):
    encoded = tokenizer(
        prompt,
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

    with torch.inference_mode():
        logits = (
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            .logits
            .detach()
            .cpu()
            .clone()
        )

    return (
        logits,
        int(
            input_ids.shape[1]
        ),
    )


def main():
    torch.manual_seed(SEED)

    section(
        "ENG011-E — FIXED MULTI-PROMPT STREAMING EQUIVALENCE VALIDATION"
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
        "Selected matrices       :",
        len(TARGETS),
    )

    print(
        "Rows/chunk              :",
        ROWS_PER_CHUNK,
    )

    print(
        "Prompt count            :",
        len(PROMPTS),
    )

    # --------------------------------------------------------------
    # Frozen checkpoint verification.
    # --------------------------------------------------------------

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

    total_parameters = sum(
        packed_weights[name]
        .original_numel
        for name in TARGETS
    )

    total_packed_bytes = sum(
        packed_weights[name]
        .packed_storage_bytes
        for name in TARGETS
    )

    total_fp16_bytes = sum(
        packed_weights[name]
        .original_fp16_bytes
        for name in TARGETS
    )

    compression = (
        total_fp16_bytes
        / total_packed_bytes
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
        f"{compression:.6f}x",
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            MODEL_NAME
        )
    )

    # --------------------------------------------------------------
    # Path A — frozen ENG010.
    # --------------------------------------------------------------

    section(
        "PATH A — ENG010 DIRECT TERNARY RUNTIME"
    )

    direct_model, direct_result = (
        prepare_direct_model()
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

    direct_results = []

    for index, prompt in enumerate(
        PROMPTS,
        start=1,
    ):
        logits, token_count = (
            forward_prompt(
                direct_model,
                tokenizer,
                prompt,
            )
        )

        direct_results.append(
            (
                logits,
                token_count,
            )
        )

        print(
            f"ENG010 prompt {index:02d}: "
            f"{token_count} tokens"
        )

    del direct_model

    gc.collect()

    # --------------------------------------------------------------
    # Path B — ENG011 streaming.
    # --------------------------------------------------------------

    section(
        "PATH B — ENG011 STREAMING DIRECT RUNTIME"
    )

    streaming_model, streaming_result = (
        prepare_streaming_model()
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

    streaming_results = []

    for index, prompt in enumerate(
        PROMPTS,
        start=1,
    ):
        logits, token_count = (
            forward_prompt(
                streaming_model,
                tokenizer,
                prompt,
            )
        )

        streaming_results.append(
            (
                logits,
                token_count,
            )
        )

        print(
            f"ENG011 prompt {index:02d}: "
            f"{token_count} tokens"
        )

    del streaming_model

    gc.collect()

    # --------------------------------------------------------------
    # Exact equivalence metrics.
    # --------------------------------------------------------------

    section(
        "PER-PROMPT EXACT EQUIVALENCE"
    )

    aggregate_logits = 0
    aggregate_mismatches = 0
    aggregate_abs_sum = 0.0
    aggregate_squared_sum = 0.0
    global_max_abs = 0.0

    aggregate_positions = 0
    aggregate_top1_matches = 0

    all_prompts_bit_exact = True

    for index, prompt in enumerate(
        PROMPTS,
        start=1,
    ):
        direct_logits, direct_tokens = (
            direct_results[
                index - 1
            ]
        )

        streaming_logits, streaming_tokens = (
            streaming_results[
                index - 1
            ]
        )

        assert (
            direct_tokens
            == streaming_tokens
        )

        assert (
            direct_logits.shape
            == streaming_logits.shape
        )

        assert (
            direct_logits.dtype
            == streaming_logits.dtype
        )

        compared = (
            direct_logits.numel()
        )

        mismatches = int(
            (
                direct_logits
                != streaming_logits
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

        max_abs = float(
            absolute_difference
            .max()
            .item()
        )

        abs_sum = float(
            absolute_difference
            .double()
            .sum()
            .item()
        )

        squared_sum = float(
            (
                difference.double()
                * difference.double()
            )
            .sum()
            .item()
        )

        mean_abs = (
            abs_sum
            / compared
        )

        rmse = math.sqrt(
            squared_sum
            / compared
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

        top1_match = (
            direct_top1
            == streaming_top1
        )

        positions = (
            top1_match.numel()
        )

        top1_matches = int(
            top1_match
            .sum()
            .item()
        )

        # Strict per-prompt acceptance.
        assert (
            mismatches
            == 0
        )

        assert (
            max_abs
            == 0.0
        )

        assert (
            mean_abs
            == 0.0
        )

        assert (
            rmse
            == 0.0
        )

        assert bit_exact

        assert (
            top1_matches
            == positions
        )

        aggregate_logits += (
            compared
        )

        aggregate_mismatches += (
            mismatches
        )

        aggregate_abs_sum += (
            abs_sum
        )

        aggregate_squared_sum += (
            squared_sum
        )

        global_max_abs = max(
            global_max_abs,
            max_abs,
        )

        aggregate_positions += (
            positions
        )

        aggregate_top1_matches += (
            top1_matches
        )

        all_prompts_bit_exact = (
            all_prompts_bit_exact
            and bit_exact
        )

        print()
        print(
            f"Prompt {index:02d}"
        )

        print(
            "Text                    :",
            prompt,
        )

        print(
            "Tokens                  :",
            direct_tokens,
        )

        print(
            "Compared logits         :",
            compared,
        )

        print(
            "Logit mismatches        :",
            mismatches,
        )

        print(
            "Max abs difference      :",
            max_abs,
        )

        print(
            "Mean abs difference     :",
            mean_abs,
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
            f"{top1_matches}/{positions}",
        )

    # --------------------------------------------------------------
    # Aggregate exact-equivalence metrics.
    # --------------------------------------------------------------

    aggregate_mean_abs = (
        aggregate_abs_sum
        / aggregate_logits
    )

    aggregate_rmse = math.sqrt(
        aggregate_squared_sum
        / aggregate_logits
    )

    aggregate_top1_agreement = (
        100.0
        * aggregate_top1_matches
        / aggregate_positions
    )

    # Frozen suite sanity checks from ENG010.
    assert (
        aggregate_positions
        == 134
    )

    assert (
        aggregate_logits
        == 33_274_880
    )

    # Strict aggregate acceptance.
    assert (
        aggregate_mismatches
        == 0
    )

    assert (
        global_max_abs
        == 0.0
    )

    assert (
        aggregate_mean_abs
        == 0.0
    )

    assert (
        aggregate_rmse
        == 0.0
    )

    assert all_prompts_bit_exact

    assert (
        aggregate_top1_matches
        == aggregate_positions
    )

    section(
        "ENG011-E MULTI-PROMPT SUMMARY"
    )

    print(
        "Prompts                 :",
        len(PROMPTS),
    )

    print(
        "Selected matrices       :",
        len(TARGETS),
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
        f"{compression:.6f}x",
    )

    print(
        "Rows/chunk              :",
        ROWS_PER_CHUNK,
    )

    print()
    print(
        "Total logits compared   :",
        aggregate_logits,
    )

    print(
        "Logit mismatches        :",
        aggregate_mismatches,
    )

    print(
        "Global max abs diff     :",
        global_max_abs,
    )

    print(
        "Weighted mean abs diff  :",
        aggregate_mean_abs,
    )

    print(
        "Global RMSE             :",
        aggregate_rmse,
    )

    print(
        "All prompts bit-exact   :",
        all_prompts_bit_exact,
    )

    print()
    print(
        "Prompt positions        :",
        aggregate_positions,
    )

    print(
        "Top-1 matches           :",
        aggregate_top1_matches,
    )

    print(
        "Top-1 agreement         :",
        f"{aggregate_top1_agreement:.6f}%",
    )

    print()
    print(
        "Controlled difference   :",
        "full-matrix decode -> row-chunk decode",
    )

    print()
    print(
        "ENG011-E MULTI-PROMPT VALIDATION: PASS"
    )


if __name__ == "__main__":
    main()
