import gc
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from packed_ternary.table_dispatch_runtime_loader import (
    load_table_dispatch_runtime_modules,
)

from packed_ternary.gpu004_runtime_loader import (
    load_gpu004_runtime_modules,
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

SEED = 20261012
CPU_THREADS = 6


def section(title):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def load_model_cpu():
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
        kwargs["attention_mask"] = attention_mask

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
        "GPU004 — FULL-MODEL CUDA CORRECTNESS"
    )

    print("Model                   :", MODEL_NAME)
    print("Checkpoint              :", CHECKPOINT_PATH)
    print("Selected matrices       :", len(TARGETS))
    print("CUDA available          :", torch.cuda.is_available())
    print("GPU                     :", torch.cuda.get_device_name(0))
    print("Capability              :", torch.cuda.get_device_capability(0))

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

    input_ids_cpu = (
        encoded["input_ids"]
        .cpu()
    )

    attention_mask_cpu = (
        encoded.get(
            "attention_mask"
        )
    )

    if attention_mask_cpu is not None:
        attention_mask_cpu = (
            attention_mask_cpu.cpu()
        )

    print(
        "Input tokens            :",
        input_ids_cpu.shape[1],
    )

    # ==========================================================
    # PATH A
    # Frozen OPT006 CPU packed arithmetic.
    # ==========================================================

    section(
        "PATH A — FROZEN OPT006 CPU PACKED REFERENCE"
    )

    reference_model = (
        load_model_cpu()
    )

    reference_load = (
        load_table_dispatch_runtime_modules(
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

    reference_logits = (
        run_model(
            reference_model,
            input_ids_cpu,
            attention_mask_cpu,
        )
    )

    print(
        "Reference logits shape  :",
        tuple(reference_logits.shape),
    )

    print(
        "Reference logits dtype  :",
        reference_logits.dtype,
    )

    del reference_model

    gc.collect()

    # ==========================================================
    # PATH B
    # GPU004 CUDA packed runtime.
    # ==========================================================

    section(
        "PATH B — GPU004 CUDA PACKED RUNTIME"
    )

    torch.cuda.empty_cache()

    candidate_model = (
        load_model_cpu()
    )

    candidate_load = (
        load_gpu004_runtime_modules(
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

    # Move complete hybrid model to GPU.
    candidate_model = (
        candidate_model.cuda()
    )

    input_ids_gpu = (
        input_ids_cpu.cuda()
    )

    if attention_mask_cpu is None:
        attention_mask_gpu = None
    else:
        attention_mask_gpu = (
            attention_mask_cpu.cuda()
        )

    torch.cuda.synchronize()

    allocated_before = (
        torch.cuda.memory_allocated()
    )

    reserved_before = (
        torch.cuda.memory_reserved()
    )

    candidate_logits = (
        run_model(
            candidate_model,
            input_ids_gpu,
            attention_mask_gpu,
        )
    )

    torch.cuda.synchronize()

    allocated_after = (
        torch.cuda.memory_allocated()
    )

    reserved_after = (
        torch.cuda.memory_reserved()
    )

    print(
        "Candidate logits shape  :",
        tuple(candidate_logits.shape),
    )

    print(
        "Candidate logits dtype  :",
        candidate_logits.dtype,
    )

    # These are diagnostic only.
    # GPU004 makes no memory-advantage claim.
    print(
        "CUDA allocated before   :",
        allocated_before,
    )

    print(
        "CUDA allocated after    :",
        allocated_after,
    )

    print(
        "CUDA reserved before    :",
        reserved_before,
    )

    print(
        "CUDA reserved after     :",
        reserved_after,
    )

    del candidate_model

    gc.collect()
    torch.cuda.empty_cache()

    # ==========================================================
    # Comparison
    # ==========================================================

    section(
        "OPT006 CPU VS GPU004 CUDA LOGIT COMPARISON"
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

    exact_percent = (
        100.0
        * (compared - mismatches)
        / compared
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

    top1_agreement = (
        100.0
        * top1_matches
        / top1_total
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
        "Exact logit agreement   :",
        f"{exact_percent:.6f}%",
    )

    print(
        "Max abs difference      :",
        f"{max_abs:.10f}",
    )

    print(
        "Mean abs difference     :",
        f"{mean_abs:.10f}",
    )

    print(
        "Top-1 matches           :",
        f"{top1_matches}/{top1_total}",
    )

    print(
        "Top-1 agreement         :",
        f"{top1_agreement:.6f}%",
    )

    print()
    print(
        "Interpretation:"
    )

    print(
        "- GPU003 selected packed operator arithmetic was already "
        "validated bit-exact against OPT006."
    )

    print(
        "- This GPU004 comparison executes the surrounding model "
        "on different devices (CPU vs CUDA)."
    )

    print(
        "- Therefore full-model CPU-vs-CUDA logit bit-exactness "
        "is not required by this gate."
    )

    print(
        "- No GPU performance or memory-advantage claim is made."
    )

    if (
        torch.isfinite(
            candidate_logits
        ).all()
        and top1_matches
        == top1_total
    ):
        print()
        print(
            "GPU004 FULL-MODEL CUDA CORRECTNESS: PASS"
        )
    else:
        print()
        print(
            "GPU004 FULL-MODEL CUDA CORRECTNESS: NEEDS ANALYSIS"
        )


if __name__ == "__main__":
    main()
