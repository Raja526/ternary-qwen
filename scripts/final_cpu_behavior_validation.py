import gc
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from packed_ternary.table_dispatch_runtime_loader import (
    load_table_dispatch_runtime_modules,
)
from packed_ternary.avx2_runtime_loader import (
    load_avx2_runtime_modules,
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

CPU_THREADS = 6
SEED = 20260929

PROMPTS = (
    (
        "short_factual",
        "The capital of France is",
    ),
    (
        "arithmetic",
        "A shop has 17 boxes with 23 pencils in each box. "
        "How many pencils are there in total?",
    ),
    (
        "reasoning",
        "A train leaves a station at 9:15 AM and travels "
        "for 2 hours and 35 minutes. What time does it arrive?",
    ),
    (
        "instruction",
        "Explain in three short sentences why regular exercise "
        "is useful for maintaining good health.",
    ),
    (
        "long_context",
        "A warehouse receives shipments from three suppliers. "
        "Supplier A delivers 120 units on Monday and 85 units "
        "on Tuesday. Supplier B delivers 75 units on Monday, "
        "95 units on Wednesday, and 60 units on Thursday. "
        "Supplier C delivers 110 units on Tuesday and 90 units "
        "on Friday. The warehouse ships 140 units to customers "
        "during the same week. Based only on these numbers, "
        "calculate the total number of units received before "
        "subtracting the units shipped.",
    ),
)


def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model


def run_logits(model, encoded):
    kwargs = {
        "input_ids": encoded["input_ids"].cpu(),
    }

    if "attention_mask" in encoded:
        kwargs["attention_mask"] = (
            encoded["attention_mask"].cpu()
        )

    with torch.inference_mode():
        return (
            model(**kwargs)
            .logits
            .detach()
            .cpu()
            .clone()
        )


def main():
    torch.manual_seed(SEED)
    torch.set_num_threads(CPU_THREADS)
    torch.set_num_interop_threads(CPU_THREADS)

    print("=" * 88)
    print("FINAL CPU VALIDATION — GATE 3")
    print("MULTI-PROMPT BEHAVIORAL EQUIVALENCE")
    print("=" * 88)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    encoded_prompts = []

    for name, prompt in PROMPTS:
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True,
        )

        encoded_prompts.append(
            (name, prompt, encoded)
        )

    # --------------------------------------------------
    # Reference: frozen OPT006
    # --------------------------------------------------

    print("\nLoading frozen OPT006 reference...")

    reference_model = load_model()

    reference_load = (
        load_table_dispatch_runtime_modules(
            model=reference_model,
            checkpoint_path=CHECKPOINT_PATH,
            selected_module_names=TARGETS,
            expected_model_name=MODEL_NAME,
        )
    )

    assert reference_load.replaced_matrix_count == 3
    assert (
        reference_load.replaced_parameter_count
        == 8_912_896
    )
    assert (
        reference_load.persistent_packed_bytes
        == 2_367_488
    )
    assert (
        reference_load.fp16_equivalent_bytes
        == 17_825_792
    )

    reference_outputs = {}

    for name, _, encoded in encoded_prompts:
        reference_outputs[name] = run_logits(
            reference_model,
            encoded,
        )

    del reference_model
    gc.collect()

    # --------------------------------------------------
    # Candidate: retained OPT008
    # --------------------------------------------------

    print("Loading retained OPT008 candidate...")

    candidate_model = load_model()

    candidate_load = load_avx2_runtime_modules(
        model=candidate_model,
        checkpoint_path=CHECKPOINT_PATH,
        selected_module_names=TARGETS,
        expected_model_name=MODEL_NAME,
    )

    assert candidate_load.replaced_matrix_count == 3
    assert (
        candidate_load.replaced_parameter_count
        == 8_912_896
    )
    assert (
        candidate_load.persistent_packed_bytes
        == 2_367_488
    )
    assert (
        candidate_load.fp16_equivalent_bytes
        == 17_825_792
    )

    total_logits = 0
    total_mismatches = 0
    total_top1 = 0
    total_top1_matches = 0
    global_max_abs = 0.0

    for name, _, encoded in encoded_prompts:
        candidate = run_logits(
            candidate_model,
            encoded,
        )

        reference = reference_outputs[name]

        assert candidate.shape == reference.shape
        assert candidate.dtype == reference.dtype

        compared = reference.numel()

        mismatches = int(
            (reference != candidate)
            .sum()
            .item()
        )

        diff = (
            candidate.float()
            - reference.float()
        ).abs()

        max_abs = float(
            diff.max().item()
        )

        mean_abs = float(
            diff.double().mean().item()
        )

        bit_exact = torch.equal(
            reference,
            candidate,
        )

        ref_top1 = reference.argmax(dim=-1)
        cand_top1 = candidate.argmax(dim=-1)

        top1_matches = int(
            (ref_top1 == cand_top1)
            .sum()
            .item()
        )

        top1_total = ref_top1.numel()

        tokens = encoded[
            "input_ids"
        ].shape[1]

        print()
        print(f"Prompt                 : {name}")
        print(f"Input tokens           : {tokens}")
        print(
            f"Compared logits        : {compared}"
        )
        print(
            f"Exact mismatches       : {mismatches}"
        )
        print(
            f"Max abs difference     : {max_abs}"
        )
        print(
            f"Mean abs difference    : {mean_abs}"
        )
        print(
            f"Bit-exact              : {bit_exact}"
        )
        print(
            "Top-1 matches          : "
            f"{top1_matches}/{top1_total}"
        )

        if mismatches != 0:
            raise RuntimeError(
                f"{name}: logit mismatch"
            )

        if max_abs != 0.0:
            raise RuntimeError(
                f"{name}: non-zero max error"
            )

        if mean_abs != 0.0:
            raise RuntimeError(
                f"{name}: non-zero mean error"
            )

        if not bit_exact:
            raise RuntimeError(
                f"{name}: torch.equal failed"
            )

        if top1_matches != top1_total:
            raise RuntimeError(
                f"{name}: top-1 mismatch"
            )

        total_logits += compared
        total_mismatches += mismatches
        total_top1 += top1_total
        total_top1_matches += top1_matches
        global_max_abs = max(
            global_max_abs,
            max_abs,
        )

    del candidate_model
    gc.collect()

    print()
    print("=" * 88)
    print("AGGREGATE")
    print("=" * 88)

    print(
        "Prompts validated       :",
        len(PROMPTS),
    )
    print(
        "Total logits compared   :",
        total_logits,
    )
    print(
        "Total mismatches        :",
        total_mismatches,
    )
    print(
        "Global max abs error    :",
        global_max_abs,
    )
    print(
        "Top-1 agreement         :",
        f"{total_top1_matches}/{total_top1}",
    )

    assert total_mismatches == 0
    assert global_max_abs == 0.0
    assert total_top1_matches == total_top1

    print()
    print("=" * 88)
    print("FINAL CPU VALIDATION GATE 3 — PASS")
    print("=" * 88)

    print(
        "Retained OPT008 is bit-exact with frozen "
        "OPT006 across the fixed multi-prompt suite."
    )


if __name__ == "__main__":
    main()
