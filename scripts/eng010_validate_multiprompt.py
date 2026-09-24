"""
ENG010-D — Fixed multi-prompt validation.

Objective
---------
Compare the same three frozen ENG003 ternary matrices using:

Path A:
    reconstructed FP16 weights in ordinary nn.Linear modules

Path B:
    DirectTernaryLinear operating directly from ternary codes + scales

No quantizer, checkpoint, runtime operator, target matrix, or chunk-size
change is introduced.

This validation expands only the evaluation inputs from one prompt to a
small fixed prompt suite.

Reported metrics:
    - exact FP16 logit mismatch
    - max absolute logit difference
    - mean absolute logit difference
    - RMSE
    - cosine similarity
    - top-1 agreement

Aggregate metrics are computed across all logits and prompt positions.

This experiment does NOT claim:
    - general behavioral equivalence,
    - inference speedup,
    - peak RAM/VRAM reduction,
    - fused 2-bit execution,
    - whole-model compression.
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

SEED = 20260924


# Fixed before running the experiment.
#
# The suite intentionally contains short prompts from several simple
# language/reasoning styles. It is a numerical/runtime validation suite,
# not a benchmark and not a claim of broad task coverage.
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


def prepare_reference_model(
    packed_weights,
):
    model = load_model()

    for name in TARGETS:
        module = get_module(
            model,
            name,
        )

        assert isinstance(
            module,
            torch.nn.Linear,
        )

        reconstructed = (
            reconstruct_packed(
                packed_weights[name]
            )
        )

        assert (
            reconstructed.shape
            == module.weight.shape
        )

        assert (
            reconstructed.dtype
            == module.weight.dtype
        )

        with torch.no_grad():
            module.weight.copy_(
                reconstructed
            )

        del reconstructed

    return model


def prepare_direct_model():
    model = load_model()

    result = load_direct_runtime_modules(
        model=model,
        checkpoint_path=CHECKPOINT_PATH,
        selected_module_names=TARGETS,
        expected_model_name=MODEL_NAME,
        rows_per_chunk=ROWS_PER_CHUNK,
    )

    assert (
        result.loaded_module_names
        == TARGETS
    )

    assert (
        result.replaced_matrix_count
        == 3
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
        "ENG010-D — FIXED MULTI-PROMPT DIRECT TERNARY VALIDATION"
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
    # Path A.
    #
    # Run every prompt through one fixed reference model.
    # --------------------------------------------------------------

    section(
        "PATH A — RECONSTRUCTED FP16 REFERENCE"
    )

    reference_model = (
        prepare_reference_model(
            packed_weights
        )
    )

    reference_results = []

    for index, prompt in enumerate(
        PROMPTS,
        start=1,
    ):
        logits, token_count = (
            forward_prompt(
                reference_model,
                tokenizer,
                prompt,
            )
        )

        reference_results.append(
            (
                logits,
                token_count,
            )
        )

        print(
            f"Reference prompt {index:02d}: "
            f"{token_count} tokens"
        )

    del reference_model

    gc.collect()

    # --------------------------------------------------------------
    # Path B.
    # --------------------------------------------------------------

    section(
        "PATH B — DIRECT TERNARY RUNTIME"
    )

    direct_model, load_result = (
        prepare_direct_model()
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
            f"Direct prompt    {index:02d}: "
            f"{token_count} tokens"
        )

    del direct_model

    gc.collect()

    # --------------------------------------------------------------
    # Metrics.
    # --------------------------------------------------------------

    section(
        "PER-PROMPT RESULTS"
    )

    aggregate_logits = 0
    aggregate_mismatches = 0
    aggregate_abs_sum = 0.0
    aggregate_squared_sum = 0.0

    global_max_abs = 0.0

    aggregate_positions = 0
    aggregate_top1_matches = 0

    cosine_values = []

    differing_position_records = []

    for index, prompt in enumerate(
        PROMPTS,
        start=1,
    ):
        reference_logits, reference_tokens = (
            reference_results[
                index - 1
            ]
        )

        direct_logits, direct_tokens = (
            direct_results[
                index - 1
            ]
        )

        assert (
            reference_tokens
            == direct_tokens
        )

        assert (
            reference_logits.shape
            == direct_logits.shape
        )

        assert (
            reference_logits.dtype
            == direct_logits.dtype
        )

        reference_float = (
            reference_logits.float()
        )

        direct_float = (
            direct_logits.float()
        )

        difference = (
            direct_float
            - reference_float
        )

        absolute_difference = (
            difference.abs()
        )

        compared = (
            reference_logits.numel()
        )

        mismatches = int(
            (
                reference_logits
                != direct_logits
            )
            .sum()
            .item()
        )

        mismatch_percent = (
            100.0
            * mismatches
            / compared
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

        cosine = float(
            F.cosine_similarity(
                reference_cosine,
                direct_cosine,
                dim=1,
            )
            .item()
        )

        cosine = max(
            -1.0,
            min(
                1.0,
                cosine,
            ),
        )

        cosine_values.append(
            cosine
        )

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

        positions = (
            top1_match.numel()
        )

        top1_matches = int(
            top1_match
            .sum()
            .item()
        )

        top1_agreement = (
            100.0
            * top1_matches
            / positions
        )

        differing_positions = (
            torch.nonzero(
                ~top1_match,
                as_tuple=False,
            )
            .cpu()
            .tolist()
        )

        if differing_positions:
            differing_position_records.append(
                (
                    index,
                    differing_positions,
                )
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
            "Mismatch percent        :",
            mismatch_percent,
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
            "Top-1 agreement         :",
            f"{top1_agreement:.6f}%",
        )

        print(
            "Differing positions     :",
            differing_positions,
        )

    # --------------------------------------------------------------
    # Aggregate metrics.
    # --------------------------------------------------------------

    aggregate_mismatch_percent = (
        100.0
        * aggregate_mismatches
        / aggregate_logits
    )

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
        sum(
            cosine_values
        )
        / len(
            cosine_values
        )
    )

    min_prompt_cosine = min(
        cosine_values
    )

    section(
        "ENG010-D MULTI-PROMPT SUMMARY"
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
        "Exact FP16 mismatches   :",
        aggregate_mismatches,
    )

    print(
        "Mismatch percent        :",
        aggregate_mismatch_percent,
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
        min_prompt_cosine,
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

    print(
        "Differing top-1 pos.    :",
        aggregate_positions
        - aggregate_top1_matches,
    )

    print(
        "Prompts with differences:",
        len(
            differing_position_records
        ),
    )

    if differing_position_records:
        print(
            "Difference records      :",
            differing_position_records,
        )

    print()
    print(
        "Persistent FP16 weights :",
        False,
    )

    print(
        "Temporary FP16 weights  :",
        False,
    )

    print()
    print("Scope:")

    print(
        "  - Fixed 10-prompt numerical validation suite."
    )

    print(
        "  - Three frozen ENG003 matrices only."
    )

    print(
        "  - Same packed checkpoint used by both paths."
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
        "  - Top-1 agreement is descriptive for this fixed suite only."
    )

    print(
        "  - No general behavioral-equivalence claim."
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
        "ENG010-D MULTI-PROMPT VALIDATION: COMPLETE"
    )


if __name__ == "__main__":
    main()
