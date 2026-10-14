import gc
import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
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

SEED = 20261014

WARMUP_RUNS = 3
TIMED_RUNS = 10


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


def model_forward(
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
        return model(**kwargs).logits


def benchmark_model(
    mode,
    tokenizer,
):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = load_model()

    load_result = None

    if mode == "gpu004_packed":
        load_result = (
            load_gpu004_runtime_modules(
                model=model,
                checkpoint_path=CHECKPOINT_PATH,
                selected_module_names=TARGETS,
                expected_model_name=MODEL_NAME,
            )
        )

        if load_result.replaced_matrix_count != 3:
            raise RuntimeError(
                "Expected exactly three packed replacements."
            )

    model = model.cuda()

    encoded = tokenizer(
        PROMPT,
        return_tensors="pt",
        add_special_tokens=True,
    )

    input_ids = encoded["input_ids"].cuda()

    attention_mask = encoded.get(
        "attention_mask"
    )

    if attention_mask is not None:
        attention_mask = attention_mask.cuda()

    torch.cuda.synchronize()

    allocated_ready = int(
        torch.cuda.memory_allocated()
    )

    reserved_ready = int(
        torch.cuda.memory_reserved()
    )

    for _ in range(WARMUP_RUNS):
        logits = model_forward(
            model,
            input_ids,
            attention_mask,
        )

        torch.cuda.synchronize()

    warmup_peak_allocated = int(
        torch.cuda.max_memory_allocated()
    )

    torch.cuda.reset_peak_memory_stats()

    timings_ms = []

    final_logits = None

    for _ in range(TIMED_RUNS):
        torch.cuda.synchronize()

        start = time.perf_counter()

        final_logits = model_forward(
            model,
            input_ids,
            attention_mask,
        )

        torch.cuda.synchronize()

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000.0

        timings_ms.append(
            float(elapsed_ms)
        )

    timed_peak_allocated = int(
        torch.cuda.max_memory_allocated()
    )

    timed_peak_reserved = int(
        torch.cuda.max_memory_reserved()
    )

    finite = bool(
        torch.isfinite(
            final_logits
        ).all().item()
    )

    top1 = (
        final_logits
        .argmax(dim=-1)
        .detach()
        .cpu()
    )

    result = {
        "mode": mode,
        "input_tokens": int(
            input_ids.shape[1]
        ),
        "warmup_runs": WARMUP_RUNS,
        "timed_runs": TIMED_RUNS,
        "timings_ms": timings_ms,
        "mean_ms": float(
            statistics.mean(timings_ms)
        ),
        "median_ms": float(
            statistics.median(timings_ms)
        ),
        "stdev_ms": float(
            statistics.stdev(timings_ms)
            if len(timings_ms) > 1
            else 0.0
        ),
        "min_ms": float(min(timings_ms)),
        "max_ms": float(max(timings_ms)),
        "allocated_ready_bytes":
            allocated_ready,
        "reserved_ready_bytes":
            reserved_ready,
        "warmup_peak_allocated_bytes":
            warmup_peak_allocated,
        "timed_peak_allocated_bytes":
            timed_peak_allocated,
        "timed_peak_reserved_bytes":
            timed_peak_reserved,
        "finite": finite,
        "top1": top1.tolist(),
    }

    if load_result is not None:
        result[
            "replaced_matrix_count"
        ] = int(
            load_result.replaced_matrix_count
        )

        result[
            "replaced_parameter_count"
        ] = int(
            load_result.replaced_parameter_count
        )

        result[
            "persistent_packed_bytes"
        ] = int(
            load_result.persistent_packed_bytes
        )

        result[
            "fp16_equivalent_bytes"
        ] = int(
            load_result.fp16_equivalent_bytes
        )

    del final_logits
    del model

    gc.collect()

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    return result


def main():
    torch.manual_seed(SEED)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required."
        )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            MODEL_NAME
        )
    )

    print(
        "GPU005 — CUDA PERFORMANCE CHARACTERIZATION"
    )

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "Capability:",
        torch.cuda.get_device_capability(0)
    )

    print(
        "Warmups:",
        WARMUP_RUNS
    )

    print(
        "Timed runs:",
        TIMED_RUNS
    )

    # Baseline first.
    baseline = benchmark_model(
        "fp16_baseline",
        tokenizer,
    )

    # Release baseline before candidate.
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    candidate = benchmark_model(
        "gpu004_packed",
        tokenizer,
    )

    if not baseline["finite"]:
        raise RuntimeError(
            "Baseline produced non-finite logits."
        )

    if not candidate["finite"]:
        raise RuntimeError(
            "Packed candidate produced non-finite logits."
        )

    mean_change_percent = (
        (
            candidate["mean_ms"]
            - baseline["mean_ms"]
        )
        / baseline["mean_ms"]
        * 100.0
    )

    median_change_percent = (
        (
            candidate["median_ms"]
            - baseline["median_ms"]
        )
        / baseline["median_ms"]
        * 100.0
    )

    ready_memory_change = (
        candidate[
            "allocated_ready_bytes"
        ]
        - baseline[
            "allocated_ready_bytes"
        ]
    )

    peak_memory_change = (
        candidate[
            "timed_peak_allocated_bytes"
        ]
        - baseline[
            "timed_peak_allocated_bytes"
        ]
    )

    top1_matches = 0
    top1_total = 0

    baseline_top1 = baseline["top1"]
    candidate_top1 = candidate["top1"]

    if (
        len(baseline_top1)
        == len(candidate_top1)
    ):
        for a, b in zip(
            baseline_top1,
            candidate_top1,
        ):
            if len(a) != len(b):
                continue

            for x, y in zip(a, b):
                top1_total += 1

                if x == y:
                    top1_matches += 1

    result = {
        "experiment": "GPU005",
        "model": MODEL_NAME,
        "checkpoint": str(
            CHECKPOINT_PATH
        ),
        "targets": list(TARGETS),
        "prompt": PROMPT,
        "seed": SEED,
        "gpu": torch.cuda.get_device_name(0),
        "capability": list(
            torch.cuda.get_device_capability(0)
        ),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "baseline": baseline,
        "candidate": candidate,
        "mean_latency_change_percent":
            float(mean_change_percent),
        "median_latency_change_percent":
            float(median_change_percent),
        "ready_allocated_change_bytes":
            int(ready_memory_change),
        "timed_peak_allocated_change_bytes":
            int(peak_memory_change),
        "top1_matches":
            int(top1_matches),
        "top1_total":
            int(top1_total),
    }

    output_path = Path(
        "results/GPU005_raw.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            result,
            indent=2,
        )
        + "\n"
    )

    print()
    print(
        "Baseline mean ms :",
        baseline["mean_ms"]
    )

    print(
        "Candidate mean ms:",
        candidate["mean_ms"]
    )

    print(
        "Mean change %    :",
        mean_change_percent
    )

    print(
        "Median change %  :",
        median_change_percent
    )

    print(
        "Top-1 agreement  :",
        f"{top1_matches}/{top1_total}"
    )

    print(
        "Raw result       :",
        output_path
    )

    print()
    print("GPU005 BENCHMARK: PASS")


if __name__ == "__main__":
    main()
