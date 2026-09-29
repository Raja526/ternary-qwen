import gc
from pathlib import Path

import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from packed_ternary.avx2_runtime_loader import (
    load_avx2_runtime_modules,
)

from packed_ternary.grain16_avx2_runtime_loader import (
    load_grain16_avx2_runtime_modules,
)

from packed_ternary.grain64_avx2_runtime_loader import (
    load_grain64_avx2_runtime_modules,
)

from packed_ternary.grain256_avx2_runtime_loader import (
    load_grain256_avx2_runtime_modules,
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

EXPECTED_MATRICES = 3
EXPECTED_PARAMETERS = 8_912_896
EXPECTED_TOKENS = 22
EXPECTED_LOGITS = 5_463_040


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
            torch_dtype=torch.float16,
        )
    )

    model.eval()
    model.cpu()

    return model


def run_model(
    model,
    input_ids,
    attention_mask,
):
    with torch.inference_mode():
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits

    return (
        logits
        .detach()
        .cpu()
        .clone()
    )


def print_load_stats(load_result):
    print(
        "Replaced matrices        :",
        load_result.replaced_matrix_count,
    )
    print(
        "Replaced parameters      :",
        load_result.replaced_parameter_count,
    )
    print(
        "Persistent packed bytes  :",
        load_result.persistent_packed_bytes,
    )
    print(
        "FP16 equivalent bytes    :",
        load_result.fp16_equivalent_bytes,
    )


def validate_structure(
    reference_load,
    candidate_load,
    candidate_name,
):
    if (
        reference_load.replaced_matrix_count
        != EXPECTED_MATRICES
    ):
        raise RuntimeError(
            "Frozen OPT008 reference replaced an "
            "unexpected number of matrices."
        )

    if (
        candidate_load.replaced_matrix_count
        != EXPECTED_MATRICES
    ):
        raise RuntimeError(
            f"{candidate_name}: unexpected "
            "replacement matrix count."
        )

    if (
        reference_load.replaced_parameter_count
        != EXPECTED_PARAMETERS
    ):
        raise RuntimeError(
            "Frozen OPT008 reference replaced an "
            "unexpected number of parameters."
        )

    if (
        candidate_load.replaced_parameter_count
        != EXPECTED_PARAMETERS
    ):
        raise RuntimeError(
            f"{candidate_name}: unexpected "
            "replacement parameter count."
        )

    if (
        reference_load.persistent_packed_bytes
        != candidate_load.persistent_packed_bytes
    ):
        raise RuntimeError(
            f"{candidate_name}: persistent packed "
            "bytes differ from frozen OPT008."
        )

    if (
        reference_load.fp16_equivalent_bytes
        != candidate_load.fp16_equivalent_bytes
    ):
        raise RuntimeError(
            f"{candidate_name}: FP16-equivalent "
            "bytes differ from frozen OPT008."
        )


def compare_logits(
    reference_logits,
    candidate_logits,
    candidate_name,
):
    section(
        f"OPT008 GRAIN-1 VS {candidate_name}"
    )

    if (
        reference_logits.shape
        != candidate_logits.shape
    ):
        raise RuntimeError(
            f"{candidate_name}: logits shape differs."
        )

    if (
        reference_logits.dtype
        != candidate_logits.dtype
    ):
        raise RuntimeError(
            f"{candidate_name}: logits dtype differs."
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
        "Compared logits          :",
        compared,
    )
    print(
        "Exact mismatches         :",
        mismatches,
    )
    print(
        "Max abs difference       :",
        max_abs,
    )
    print(
        "Mean abs difference      :",
        mean_abs,
    )
    print(
        "Bit-exact logits         :",
        bit_exact,
    )
    print(
        "Top-1 matches            :",
        f"{top1_matches}/{top1_total}",
    )

    if compared != EXPECTED_LOGITS:
        raise RuntimeError(
            f"{candidate_name}: expected "
            f"{EXPECTED_LOGITS} logits, "
            f"got {compared}."
        )

    if mismatches != 0:
        raise RuntimeError(
            f"{candidate_name}: full-model logits "
            "are not bit-exact with frozen OPT008."
        )

    if max_abs != 0.0:
        raise RuntimeError(
            f"{candidate_name}: non-zero maximum "
            "logit difference."
        )

    if mean_abs != 0.0:
        raise RuntimeError(
            f"{candidate_name}: non-zero mean "
            "logit difference."
        )

    if not bit_exact:
        raise RuntimeError(
            f"{candidate_name}: torch.equal "
            "returned False."
        )

    if top1_matches != top1_total:
        raise RuntimeError(
            f"{candidate_name}: top-1 predictions "
            "differ from frozen OPT008."
        )

    if top1_total != EXPECTED_TOKENS:
        raise RuntimeError(
            f"{candidate_name}: expected "
            f"{EXPECTED_TOKENS} top-1 positions, "
            f"got {top1_total}."
        )

    print()
    print(
        f"PASS: {candidate_name} is bit-exact "
        "with frozen OPT008."
    )


def run_candidate(
    name,
    loader,
    reference_load,
    reference_logits,
    input_ids,
    attention_mask,
):
    section(
        f"CANDIDATE — {name}"
    )

    model = load_model()

    load_result = loader(
        model=model,
        checkpoint_path=CHECKPOINT_PATH,
        selected_module_names=TARGETS,
        expected_model_name=MODEL_NAME,
    )

    print_load_stats(
        load_result
    )

    validate_structure(
        reference_load,
        load_result,
        name,
    )

    logits = run_model(
        model,
        input_ids,
        attention_mask,
    )

    print(
        "Candidate logits shape   :",
        tuple(logits.shape),
    )

    del model
    gc.collect()

    compare_logits(
        reference_logits,
        logits,
        name,
    )

    del logits
    gc.collect()


def main():
    torch.manual_seed(SEED)

    torch.set_num_threads(
        CPU_THREADS
    )

    torch.set_num_interop_threads(
        CPU_THREADS
    )

    section(
        "OPT010 — FULL-MODEL THREAD-SCHEDULING EQUIVALENCE"
    )

    print(
        "Reference                : OPT008 grain-size 1"
    )
    print(
        "Candidates               : grain 16 / 64 / 256"
    )
    print(
        "CPU threads              :",
        CPU_THREADS,
    )
    print(
        "Checkpoint               :",
        CHECKPOINT_PATH,
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
        "Input tokens             :",
        input_ids.shape[1],
    )

    if (
        input_ids.shape[1]
        != EXPECTED_TOKENS
    ):
        raise RuntimeError(
            "Unexpected token count; frozen "
            "full-model protocol changed."
        )

    # --------------------------------------------------
    # Reference — frozen OPT008 grain size 1.
    # --------------------------------------------------

    section(
        "REFERENCE — FROZEN OPT008 GRAIN-1"
    )

    reference_model = load_model()

    reference_load = (
        load_avx2_runtime_modules(
            model=reference_model,
            checkpoint_path=CHECKPOINT_PATH,
            selected_module_names=TARGETS,
            expected_model_name=MODEL_NAME,
        )
    )

    print_load_stats(
        reference_load
    )

    if (
        reference_load.replaced_matrix_count
        != EXPECTED_MATRICES
    ):
        raise RuntimeError(
            "Frozen OPT008 matrix count changed."
        )

    if (
        reference_load.replaced_parameter_count
        != EXPECTED_PARAMETERS
    ):
        raise RuntimeError(
            "Frozen OPT008 parameter count changed."
        )

    reference_logits = run_model(
        reference_model,
        input_ids,
        attention_mask,
    )

    print(
        "Reference logits shape   :",
        tuple(reference_logits.shape),
    )

    if (
        reference_logits.numel()
        != EXPECTED_LOGITS
    ):
        raise RuntimeError(
            "Frozen OPT008 logit count changed."
        )

    del reference_model
    gc.collect()

    # --------------------------------------------------
    # OPT010 candidates.
    # --------------------------------------------------

    candidates = (
        (
            "OPT010 GRAIN-16",
            load_grain16_avx2_runtime_modules,
        ),
        (
            "OPT010 GRAIN-64",
            load_grain64_avx2_runtime_modules,
        ),
        (
            "OPT010 GRAIN-256",
            load_grain256_avx2_runtime_modules,
        ),
    )

    for name, loader in candidates:
        run_candidate(
            name=name,
            loader=loader,
            reference_load=reference_load,
            reference_logits=reference_logits,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    section(
        "OPT010 FULL-MODEL CORRECTNESS GATE — PASS"
    )

    print(
        "All three OPT010 grain-size variants are "
        "bit-exact with frozen OPT008."
    )


if __name__ == "__main__":
    main()
