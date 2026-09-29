import gc

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import packed_ternary.avx2_runtime_loader as ref_loader
import packed_ternary.blocked4_avx2_runtime_loader as block4_loader
import packed_ternary.blocked16_avx2_runtime_loader as block16_loader


MODEL_ID = "Qwen/Qwen3.5-0.8B"

CHECKPOINT = (
    "results/"
    "ENG003_multimatrix_checkpoint.pt"
)

SELECTED_MODULE_NAMES = (
    "model.layers.3.self_attn.k_proj",
    "model.layers.4.linear_attn.out_proj",
    "model.layers.4.linear_attn.in_proj_qkv",
)

PROMPT = (
    "A box contains 12 red balls and 8 blue balls. "
    "How many balls are in the box?"
)

CPU_THREADS = 6
SEED = 20260925


def load_base_model():
    torch.manual_seed(SEED)

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
        )
    )

    model.eval()
    model.to("cpu")

    return model


def extract_stats(result):
    stats = {}

    if isinstance(result, dict):
        stats.update(result)

    elif isinstance(result, (tuple, list)):
        for item in result:
            if isinstance(item, dict):
                stats.update(item)

    return stats


def print_stats(label, stats):
    print()
    print(label)

    wanted = [
        "replaced_matrices",
        "replaced_parameters",
        "persistent_packed_bytes",
        "fp16_equivalent_bytes",
    ]

    for key in wanted:
        if key in stats:
            print(
                f"{key}:",
                stats[key],
            )


def run_path(
    path_name,
    input_ids,
):
    model = load_base_model()

    if path_name == "ref":
        loader_fn = (
            ref_loader
            .load_avx2_runtime_modules
        )

    elif path_name == "block4":
        loader_fn = (
            block4_loader
            .load_blocked4_avx2_runtime_modules
        )

    elif path_name == "block16":
        loader_fn = (
            block16_loader
            .load_blocked16_avx2_runtime_modules
        )

    else:
        raise RuntimeError(
            f"Unknown path: {path_name}"
        )

    loader_result = loader_fn(
        model=model,
        checkpoint_path=CHECKPOINT,
        selected_module_names=SELECTED_MODULE_NAMES,
        expected_model_name=MODEL_ID,
    )

    stats = extract_stats(
        loader_result
    )

    with torch.inference_mode():
        logits = (
            model(
                input_ids=input_ids,
            )
            .logits
            .detach()
            .cpu()
            .clone()
        )

    del model
    gc.collect()

    return logits, stats


def compare(
    label,
    ref_logits,
    cand_logits,
):
    if (
        ref_logits.shape
        != cand_logits.shape
    ):
        raise RuntimeError(
            f"FAIL: {label} logits "
            "shapes differ."
        )

    exact = (
        ref_logits
        == cand_logits
    )

    mismatches = int(
        (~exact)
        .sum()
        .item()
    )

    diff = (
        cand_logits.float()
        - ref_logits.float()
    ).abs()

    max_abs = float(
        diff.max().item()
    )

    mean_abs = float(
        diff.mean().item()
    )

    ref_top1 = (
        ref_logits.argmax(
            dim=-1
        )
    )

    cand_top1 = (
        cand_logits.argmax(
            dim=-1
        )
    )

    top1_matches = int(
        (
            ref_top1
            == cand_top1
        )
        .sum()
        .item()
    )

    top1_total = int(
        ref_top1.numel()
    )

    compared = int(
        ref_logits.numel()
    )

    bit_exact = (
        mismatches == 0
        and max_abs == 0.0
    )

    print()
    print("=" * 72)
    print(label)
    print("=" * 72)

    print(
        "Logits shape      :",
        tuple(ref_logits.shape),
    )

    print(
        "Compared logits   :",
        compared,
    )

    print(
        "Exact mismatches  :",
        mismatches,
    )

    print(
        "Max abs error     :",
        max_abs,
    )

    print(
        "Mean abs error    :",
        mean_abs,
    )

    print(
        "Bit-exact logits  :",
        bit_exact,
    )

    print(
        "Top-1 matches     :",
        f"{top1_matches}/{top1_total}",
    )

    if not bit_exact:
        raise RuntimeError(
            f"FAIL: {label} is not "
            "bit-exact with OPT008."
        )

    if (
        top1_matches
        != top1_total
    ):
        raise RuntimeError(
            f"FAIL: {label} top-1 "
            "behavior differs."
        )

    return {
        "compared": compared,
        "mismatches": mismatches,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "top1_matches": top1_matches,
        "top1_total": top1_total,
    }


def main():
    torch.set_num_threads(
        CPU_THREADS
    )

    try:
        torch.set_num_interop_threads(
            CPU_THREADS
        )
    except RuntimeError:
        pass

    torch.manual_seed(SEED)

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            MODEL_ID
        )
    )

    encoded = tokenizer(
        PROMPT,
        return_tensors="pt",
    )

    input_ids = encoded[
        "input_ids"
    ]

    print("=" * 72)
    print(
        "OPT009 — BLOCK 4 / BLOCK 16 "
        "FULL-MODEL EQUIVALENCE"
    )
    print("=" * 72)

    print()
    print(
        "Reference : OPT008 explicit AVX2 decode"
    )
    print(
        "Candidate A: OPT009 block-4"
    )
    print(
        "Candidate B: OPT009 block-16"
    )
    print(
        "Model     :",
        MODEL_ID,
    )
    print(
        "CPU threads:",
        CPU_THREADS,
    )
    print(
        "Input tokens:",
        int(input_ids.numel()),
    )

    # --------------------------------------
    # Frozen reference
    # --------------------------------------

    print()
    print(
        "Running PATH A — frozen OPT008..."
    )

    ref_logits, ref_stats = (
        run_path(
            "ref",
            input_ids,
        )
    )

    print_stats(
        "PATH A statistics:",
        ref_stats,
    )

    # --------------------------------------
    # Block 4
    # --------------------------------------

    print()
    print(
        "Running PATH B — OPT009 block-4..."
    )

    block4_logits, block4_stats = (
        run_path(
            "block4",
            input_ids,
        )
    )

    print_stats(
        "PATH B statistics:",
        block4_stats,
    )

    block4_result = compare(
        "BLOCK-4 vs OPT008",
        ref_logits,
        block4_logits,
    )

    del block4_logits
    gc.collect()

    # --------------------------------------
    # Block 16
    # --------------------------------------

    print()
    print(
        "Running PATH C — OPT009 block-16..."
    )

    block16_logits, block16_stats = (
        run_path(
            "block16",
            input_ids,
        )
    )

    print_stats(
        "PATH C statistics:",
        block16_stats,
    )

    block16_result = compare(
        "BLOCK-16 vs OPT008",
        ref_logits,
        block16_logits,
    )

    del block16_logits
    del ref_logits
    gc.collect()

    # --------------------------------------
    # Final gate
    # --------------------------------------

    print()
    print("=" * 72)
    print("FINAL CORRECTNESS GATE")
    print("=" * 72)

    print(
        "Block-4 : PASS — "
        f"{block4_result['compared']:,} logits, "
        f"{block4_result['mismatches']} mismatches, "
        f"top-1 "
        f"{block4_result['top1_matches']}/"
        f"{block4_result['top1_total']}"
    )

    print(
        "Block-16: PASS — "
        f"{block16_result['compared']:,} logits, "
        f"{block16_result['mismatches']} mismatches, "
        f"top-1 "
        f"{block16_result['top1_matches']}/"
        f"{block16_result['top1_total']}"
    )

    print()
    print(
        "PASS: block-4 and block-16 full-model "
        "behavior is bit-exact with frozen OPT008."
    )


if __name__ == "__main__":
    main()
