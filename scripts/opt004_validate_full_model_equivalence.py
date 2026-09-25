"""
OPT004 — Full-model unrolled-decode equivalence.

Reference:
    OPT004 UnrolledPackedTernaryLinear
    using frozen OPT001 scalar compiled operator.

Candidate:
    OPT003 ParallelPackedTernaryLinear
    using parallel compiled operator.

Controlled difference:
    looped four-code decode -> manually unrolled four-code decode.

Expected:
    bit-exact full-model logits.
"""

import gc
from pathlib import Path

import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from packed_ternary.parallel_runtime_loader import (
    load_parallel_runtime_modules,
)

from packed_ternary.unrolled_runtime_loader import (
    load_unrolled_runtime_modules,
)

from packed_ternary.parallel_runtime_loader import (
    load_parallel_runtime_modules,
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

PROMPT = (
    "A box contains 12 red balls and 8 blue balls. "
    "How many balls are in the box?"
)

SEED = 20260925
CPU_THREADS = 6


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
        logits = model(
            **kwargs
        ).logits

    return (
        logits
        .detach()
        .cpu()
        .clone()
    )


def main():
    torch.manual_seed(SEED)

    torch.set_num_threads(
        CPU_THREADS
    )

    section(
        "OPT004 — FULL-MODEL UNROLLED-DECODE EQUIVALENCE"
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
        "CPU threads             :",
        torch.get_num_threads(),
    )

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

    # ----------------------------------------------------------
    # Reference — frozen OPT002.
    # ----------------------------------------------------------

    section(
        "PATH A — OPT003 PARALLEL COMPILED RUNTIME"
    )

    reference_model = load_model()

    reference_load = (
        load_parallel_runtime_modules(
            model=reference_model,
            checkpoint_path=CHECKPOINT_PATH,
            selected_module_names=TARGETS,
            expected_model_name=MODEL_NAME,
        )
    )

    print(
        "Replaced matrices       :",
        reference_load.replaced_matrix_count,
    )
    print(
        "Replaced parameters     :",
        reference_load.replaced_parameter_count,
    )

    reference_logits = run_model(
        reference_model,
        input_ids,
        attention_mask,
    )

    print(
        "Reference logits shape  :",
        tuple(reference_logits.shape),
    )

    del reference_model
    gc.collect()

    # ----------------------------------------------------------
    # Candidate — OPT003 parallel.
    # ----------------------------------------------------------

    section(
        "PATH B — OPT004 UNROLLED PARALLEL RUNTIME"
    )

    candidate_model = load_model()

    candidate_load = (
        load_unrolled_runtime_modules(
            model=candidate_model,
            checkpoint_path=CHECKPOINT_PATH,
            selected_module_names=TARGETS,
            expected_model_name=MODEL_NAME,
        )
    )

    print(
        "Replaced matrices       :",
        candidate_load.replaced_matrix_count,
    )
    print(
        "Replaced parameters     :",
        candidate_load.replaced_parameter_count,
    )
    print(
        "Persistent packed bytes :",
        candidate_load.persistent_packed_bytes,
    )
    print(
        "FP16 equivalent bytes   :",
        candidate_load.fp16_equivalent_bytes,
    )

    candidate_logits = run_model(
        candidate_model,
        input_ids,
        attention_mask,
    )

    print(
        "Candidate logits shape  :",
        tuple(candidate_logits.shape),
    )

    del candidate_model
    gc.collect()

    # ----------------------------------------------------------
    # Strict equivalence.
    # ----------------------------------------------------------

    section(
        "OPT003 VS OPT004 LOGIT COMPARISON"
    )

    assert (
        reference_logits.shape
        == candidate_logits.shape
    )

    assert (
        reference_logits.dtype
        == candidate_logits.dtype
    )

    compared = int(
        reference_logits.numel()
    )

    mismatches = int(
        (
            reference_logits
            != candidate_logits
        )
        .sum()
        .item()
    )

    difference = (
        candidate_logits.float()
        - reference_logits.float()
    )

    max_abs = float(
        difference
        .abs()
        .max()
        .item()
    )

    mean_abs = float(
        difference
        .abs()
        .double()
        .mean()
        .item()
    )

    bit_exact = torch.equal(
        reference_logits,
        candidate_logits,
    )

    reference_top1 = (
        reference_logits.argmax(
            dim=-1
        )
    )

    candidate_top1 = (
        candidate_logits.argmax(
            dim=-1
        )
    )

    top1_matches = int(
        (
            reference_top1
            == candidate_top1
        )
        .sum()
        .item()
    )

    top1_total = int(
        reference_top1.numel()
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
        "Max abs difference      :",
        max_abs,
    )
    print(
        "Mean abs difference     :",
        mean_abs,
    )
    print(
        "Bit-exact logits        :",
        bit_exact,
    )
    print(
        "Top-1 matches           :",
        f"{top1_matches}/{top1_total}",
    )

    assert mismatches == 0
    assert max_abs == 0.0
    assert mean_abs == 0.0
    assert bit_exact

    assert (
        top1_matches
        == top1_total
    )

    section(
        "OPT004 FULL-MODEL EQUIVALENCE SUMMARY"
    )

    print(
        "Selected matrices       :",
        candidate_load.replaced_matrix_count,
    )
    print(
        "Selected parameters     :",
        candidate_load.replaced_parameter_count,
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
        "Bit-exact               :",
        bit_exact,
    )
    print(
        "Controlled difference   :",
        "looped four-code decode -> unrolled four-code decode",
    )

    print()
    print(
        "OPT004 FULL-MODEL UNROLLED-DECODE EQUIVALENCE: PASS"
    )


if __name__ == "__main__":
    main()
