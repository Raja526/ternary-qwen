"""
OPT002 — Fixed 10-prompt full-model validation.

Reference:
    ENG011 StreamingDirectTernaryLinear

Candidate:
    OPT002 CompiledPackedTernaryLinear using the frozen OPT001
    compiled CPU operator.

Frozen:
    model
    checkpoint
    selected matrices
    group size
    quantization
    tokenizer
    10-prompt suite

Controlled variable:
    ternary compute implementation.

This experiment validates numerical behavior only.

It does NOT claim:
    - inference speedup,
    - peak RAM/VRAM reduction,
    - CUDA acceleration,
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

TARGETS = (
    "model.layers.3.self_attn.k_proj",
    "model.layers.4.linear_attn.out_proj",
    "model.layers.4.linear_attn.in_proj_qkv",
)

ROWS_PER_CHUNK = 64

SEED = 20260924

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


def run_prompts(
    model,
    tokenizer,
    label,
):
    results = []

    for index, prompt in enumerate(
        PROMPTS,
        start=1,
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

        logits = (
            output.logits
            .detach()
            .cpu()
            .clone()
        )

        token_count = int(
            input_ids.shape[1]
        )

        results.append(
            (
                logits,
                token_count,
            )
        )

        print(
            f"{label} prompt {index:02d}: "
            f"{token_count} tokens"
        )

    return results


def main():
    torch.manual_seed(SEED)

    section(
        "OPT002 — FIXED 10-PROMPT FULL-MODEL VALIDATION"
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
        "Prompts                 :",
        len(PROMPTS),
    )

    print(
        "Selected matrices       :",
        len(TARGETS),
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            MODEL_NAME
        )
    )

    # --------------------------------------------------------------
    # PATH A — frozen ENG011.
    # --------------------------------------------------------------

    section(
        "PATH A — ENG011 STREAMING RUNTIME"
    )

    reference_model = load_model()

    reference_load = (
        load_streaming_runtime_modules(
            model=reference_model,
            checkpoint_path=CHECKPOINT_PATH,
            selected_module_names=TARGETS,
            expected_model_name=MODEL_NAME,
            rows_per_chunk=ROWS_PER_CHUNK,
        )
    )

    assert (
        reference_load.replaced_matrix_count
        == 3
    )

    reference_results = run_prompts(
        reference_model,
        tokenizer,
        "ENG011",
    )

    del reference_model

    gc.collect()

    # --------------------------------------------------------------
    # PATH B — OPT002 compiled runtime.
    # --------------------------------------------------------------

    section(
        "PATH B — OPT002 COMPILED RUNTIME"
    )

    compiled_model = load_model()

    compiled_load = (
        load_compiled_runtime_modules(
            model=compiled_model,
            checkpoint_path=CHECKPOINT_PATH,
            selected_module_names=TARGETS,
            expected_model_name=MODEL_NAME,
        )
    )

    assert (
        compiled_load.replaced_matrix_count
        == 3
    )

    compiled_results = run_prompts(
        compiled_model,
        tokenizer,
        "OPT002",
    )

    total_parameters = (
        compiled_load.replaced_parameter_count
    )

    total_packed_bytes = (
        compiled_load.persistent_packed_bytes
    )

    total_fp16_bytes = (
        compiled_load.fp16_equivalent_bytes
    )

    compression = (
        total_fp16_bytes
        / total_packed_bytes
    )

    del compiled_model

    gc.collect()

    # --------------------------------------------------------------
    # Numerical comparison.
    # --------------------------------------------------------------

    section(
        "PER-PROMPT NUMERICAL COMPARISON"
    )

    aggregate_logits = 0
    aggregate_mismatches = 0

    aggregate_abs_sum = 0.0
    aggregate_squared_sum = 0.0

    global_max_abs = 0.0

    aggregate_positions = 0
    aggregate_top1_matches = 0

    prompt_cosines = []

    for index, prompt in enumerate(
        PROMPTS,
        start=1,
    ):

        reference_logits, reference_tokens = (
            reference_results[
                index - 1
            ]
        )

        compiled_logits, compiled_tokens = (
            compiled_results[
                index - 1
            ]
        )

        assert (
            reference_tokens
            == compiled_tokens
        )

        assert (
            reference_logits.shape
            == compiled_logits.shape
        )

        assert (
            reference_logits.dtype
            == compiled_logits.dtype
        )

        compared = int(
            reference_logits.numel()
        )

        mismatches = int(
            (
                reference_logits
                != compiled_logits
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

        # Float64 prevents the >1 cosine artifact seen when a very
        # large flattened vector is accumulated in float32.
        reference_flat = (
            reference_logits
            .double()
            .reshape(-1)
        )

        compiled_flat = (
            compiled_logits
            .double()
            .reshape(-1)
        )

        cosine = float(
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

        top1_match = (
            reference_top1
            == compiled_top1
        )

        positions = int(
            top1_match.numel()
        )

        top1_matches = int(
            top1_match
            .sum()
            .item()
        )

        # Per-prompt correctness-oriented acceptance.
        assert torch.isfinite(
            compiled_logits
        ).all()

        assert (
            max_abs
            <= 0.1
        )

        assert (
            mean_abs
            <= 0.01
        )

        assert (
            rmse
            <= 0.02
        )

        assert (
            cosine
            >= 0.999
        )

        assert (
            top1_matches
            == positions
        )

        aggregate_logits += compared

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

        prompt_cosines.append(
            cosine
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
            reference_tokens,
        )

        print(
            "Compared logits         :",
            compared,
        )

        print(
            "Exact mismatches        :",
            mismatches,
        )

        print(
            "Mismatch percentage     :",
            (
                100.0
                * mismatches
                / compared
            ),
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
            "Cosine similarity       :",
            cosine,
        )

        print(
            "Top-1 matches           :",
            f"{top1_matches}/{positions}",
        )

    # --------------------------------------------------------------
    # Aggregate metrics.
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

    mean_prompt_cosine = (
        sum(prompt_cosines)
        / len(prompt_cosines)
    )

    minimum_prompt_cosine = min(
        prompt_cosines
    )

    # Frozen suite sanity checks from ENG010/ENG011.
    assert (
        aggregate_positions
        == 134
    )

    assert (
        aggregate_logits
        == 33_274_880
    )

    # Aggregate acceptance.
    assert (
        global_max_abs
        <= 0.1
    )

    assert (
        aggregate_mean_abs
        <= 0.01
    )

    assert (
        aggregate_rmse
        <= 0.02
    )

    assert (
        minimum_prompt_cosine
        >= 0.999
    )

    assert (
        aggregate_top1_matches
        == aggregate_positions
    )

    section(
        "OPT002 MULTI-PROMPT SUMMARY"
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

    print()

    print(
        "Total logits compared   :",
        aggregate_logits,
    )

    print(
        "Exact mismatches        :",
        aggregate_mismatches,
    )

    print(
        "Mismatch percentage     :",
        (
            100.0
            * aggregate_mismatches
            / aggregate_logits
        ),
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
        "Mean prompt cosine      :",
        mean_prompt_cosine,
    )

    print(
        "Minimum prompt cosine   :",
        minimum_prompt_cosine,
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
        "ENG011 PyTorch streaming compute -> "
        "OPT001/OPT002 compiled packed CPU compute",
    )

    print()

    print(
        "OPT002 MULTI-PROMPT VALIDATION: PASS"
    )


if __name__ == "__main__":
    main()
