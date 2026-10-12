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

PROMPTS = (
    "The capital of France is",
    "17 multiplied by 23 equals",
    "Explain why regular exercise is important for health.",
    "A train travels 60 miles per hour for 3 hours. How far does it travel?",
    "Write a short sentence about artificial intelligence.",
)

SEED = 20261012
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


def run_prompt(
    model,
    tokenizer,
    prompt,
    device,
):
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
    )

    input_ids = (
        encoded["input_ids"]
        .to(device)
    )

    attention_mask = (
        encoded.get(
            "attention_mask"
        )
    )

    if attention_mask is not None:
        attention_mask = (
            attention_mask.to(device)
        )

    kwargs = {
        "input_ids": input_ids,
    }

    if attention_mask is not None:
        kwargs["attention_mask"] = (
            attention_mask
        )

    with torch.inference_mode():
        logits = (
            model(**kwargs)
            .logits
            .detach()
            .cpu()
            .clone()
        )

    return logits


def main():

    torch.manual_seed(SEED)

    torch.set_num_threads(
        CPU_THREADS
    )

    section(
        "GPU004 — MULTI-PROMPT CUDA VALIDATION"
    )

    print("Model              :", MODEL_NAME)
    print("Prompts            :", len(PROMPTS))
    print("Selected matrices  :", len(TARGETS))
    print("GPU                :", torch.cuda.get_device_name(0))
    print("Capability         :", torch.cuda.get_device_capability(0))

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            MODEL_NAME
        )
    )

    # ==========================================================
    # PATH A — OPT006 CPU
    # ==========================================================

    section(
        "PATH A — FROZEN OPT006 CPU PACKED REFERENCE"
    )

    reference_model = (
        load_model()
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
        "Replaced matrices  :",
        reference_load.replaced_matrix_count,
    )

    reference_outputs = []

    for index, prompt in enumerate(
        PROMPTS,
        start=1,
    ):
        logits = run_prompt(
            reference_model,
            tokenizer,
            prompt,
            torch.device("cpu"),
        )

        reference_outputs.append(
            logits
        )

        print(
            f"Reference prompt {index}:",
            tuple(logits.shape),
        )

    del reference_model
    gc.collect()

    # ==========================================================
    # PATH B — GPU004 CUDA
    # ==========================================================

    section(
        "PATH B — GPU004 CUDA PACKED RUNTIME"
    )

    torch.cuda.empty_cache()

    candidate_model = (
        load_model()
    )

    candidate_load = (
        load_gpu004_runtime_modules(
            model=candidate_model,
            checkpoint_path=CHECKPOINT_PATH,
            selected_module_names=TARGETS,
            expected_model_name=MODEL_NAME,
        )
    )

    candidate_model = (
        candidate_model.cuda()
    )

    print(
        "Replaced matrices  :",
        candidate_load.replaced_matrix_count,
    )

    candidate_outputs = []

    for index, prompt in enumerate(
        PROMPTS,
        start=1,
    ):
        logits = run_prompt(
            candidate_model,
            tokenizer,
            prompt,
            torch.device("cuda"),
        )

        torch.cuda.synchronize()

        candidate_outputs.append(
            logits
        )

        print(
            f"Candidate prompt {index}:",
            tuple(logits.shape),
        )

    del candidate_model
    gc.collect()
    torch.cuda.empty_cache()

    # ==========================================================
    # Comparisons
    # ==========================================================

    section(
        "PER-PROMPT COMPARISON"
    )

    global_compared = 0
    global_mismatches = 0

    global_abs_sum = 0.0
    global_max_abs = 0.0

    global_top1_matches = 0
    global_top1_total = 0

    all_finite = True

    for index, (
        prompt,
        reference,
        candidate,
    ) in enumerate(
        zip(
            PROMPTS,
            reference_outputs,
            candidate_outputs,
        ),
        start=1,
    ):

        assert (
            reference.shape
            == candidate.shape
        )

        assert (
            reference.dtype
            == candidate.dtype
        )

        finite = bool(
            torch.isfinite(
                candidate
            ).all().item()
        )

        all_finite = (
            all_finite
            and finite
        )

        compared = int(
            reference.numel()
        )

        mismatches = int(
            (
                reference
                != candidate
            )
            .sum()
            .item()
        )

        abs_diff = (
            candidate.float()
            - reference.float()
        ).abs()

        max_abs = float(
            abs_diff.max().item()
        )

        mean_abs = float(
            abs_diff
            .double()
            .mean()
            .item()
        )

        reference_top1 = (
            reference.argmax(
                dim=-1
            )
        )

        candidate_top1 = (
            candidate.argmax(
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

        exact_percent = (
            100.0
            * (compared - mismatches)
            / compared
        )

        top1_percent = (
            100.0
            * top1_matches
            / top1_total
        )

        print()
        print(
            f"Prompt {index}: {prompt}"
        )

        print(
            "  Shape                :",
            tuple(reference.shape),
        )

        print(
            "  Candidate finite     :",
            finite,
        )

        print(
            "  Compared logits      :",
            compared,
        )

        print(
            "  Exact mismatches     :",
            mismatches,
        )

        print(
            "  Exact agreement      :",
            f"{exact_percent:.6f}%",
        )

        print(
            "  Max abs difference   :",
            f"{max_abs:.10f}",
        )

        print(
            "  Mean abs difference  :",
            f"{mean_abs:.10f}",
        )

        print(
            "  Top-1 agreement      :",
            f"{top1_matches}/{top1_total}",
            f"({top1_percent:.6f}%)",
        )

        global_compared += (
            compared
        )

        global_mismatches += (
            mismatches
        )

        global_abs_sum += (
            float(
                abs_diff
                .double()
                .sum()
                .item()
            )
        )

        global_max_abs = max(
            global_max_abs,
            max_abs,
        )

        global_top1_matches += (
            top1_matches
        )

        global_top1_total += (
            top1_total
        )

    # ==========================================================
    # Aggregate
    # ==========================================================

    section(
        "AGGREGATE"
    )

    global_exact = (
        100.0
        * (
            global_compared
            - global_mismatches
        )
        / global_compared
    )

    global_mean_abs = (
        global_abs_sum
        / global_compared
    )

    global_top1_percent = (
        100.0
        * global_top1_matches
        / global_top1_total
    )

    print(
        "Total prompts           :",
        len(PROMPTS),
    )

    print(
        "Total compared logits   :",
        global_compared,
    )

    print(
        "Total exact mismatches  :",
        global_mismatches,
    )

    print(
        "Global exact agreement  :",
        f"{global_exact:.6f}%",
    )

    print(
        "Global max abs diff     :",
        f"{global_max_abs:.10f}",
    )

    print(
        "Global mean abs diff    :",
        f"{global_mean_abs:.10f}",
    )

    print(
        "All candidate finite    :",
        all_finite,
    )

    print(
        "Total top-1 agreement   :",
        f"{global_top1_matches}/{global_top1_total}",
        f"({global_top1_percent:.6f}%)",
    )

    print()
    print(
        "No GPU latency, throughput, or memory-advantage "
        "claim is made by GPU004."
    )

    if (
        all_finite
        and global_top1_matches
        == global_top1_total
    ):
        print()
        print(
            "GPU004 MULTI-PROMPT VALIDATION: PASS"
        )
    else:
        print()
        print(
            "GPU004 MULTI-PROMPT VALIDATION: NEEDS ANALYSIS"
        )


if __name__ == "__main__":
    main()
