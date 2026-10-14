import gc
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from packed_ternary.checkpoint import load_packed_checkpoint
from packed_ternary.gpu004_linear import CUDAPackedTernaryLinear
from packed_ternary.gpu004_runtime_loader import load_gpu004_runtime_modules


MODEL_NAME = "Qwen/Qwen3.5-0.8B"
CHECKPOINT_PATH = Path("results/ENG003_multimatrix_checkpoint.pt")
TARGETS = (
    "model.layers.3.self_attn.k_proj",
    "model.layers.4.linear_attn.out_proj",
    "model.layers.4.linear_attn.in_proj_qkv",
)
PROMPTS = (
    ("short", "What is 2 + 2?"),
    (
        "medium",
        "A box contains 12 red balls and 8 blue balls. "
        "How many balls are in the box?",
    ),
    (
        "long",
        "Explain carefully how a small language model could compare two "
        "candidate answers, identify the relevant facts in a word problem, "
        "and produce a concise final response while avoiding unsupported "
        "claims. Use a clear sequence of reasoning steps and distinguish "
        "observations from conclusions.",
    ),
)
SEED = 20261014
WARMUP_RUNS = 3
TIMED_RUNS = 10


def module_by_name(model, name):
    modules = dict(model.named_modules())
    if name not in modules:
        raise KeyError(f"Missing target module: {name}")
    return modules[name]


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_checkpoint_identity():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(CHECKPOINT_PATH)
    metadata, packed_weights = load_packed_checkpoint(CHECKPOINT_PATH)
    if metadata.get("model_name") != MODEL_NAME:
        raise RuntimeError("Checkpoint model identity mismatch")
    if set(packed_weights) != set(TARGETS):
        raise RuntimeError("Checkpoint selected target set mismatch")
    for name in TARGETS:
        packed = packed_weights[name]
        if packed.packed_codes.dtype != torch.uint8:
            raise RuntimeError(f"Packed-code dtype invalid for {name}")
        if packed.scales.dtype != torch.float16:
            raise RuntimeError(f"Scale dtype invalid for {name}")
        if packed.group_size <= 0:
            raise RuntimeError(f"Group size invalid for {name}")
        if packed.padded_numel != packed.original_numel:
            raise RuntimeError(f"Unexpected padding for {name}")
        codes = packed.packed_codes
        for shift in (0, 2, 4, 6):
            if bool((((codes >> shift) & 0x3) == 0x3).any().item()):
                raise RuntimeError(f"Reserved packed code found for {name}")
    return {
        "path": str(CHECKPOINT_PATH),
        "sha256": sha256_file(CHECKPOINT_PATH),
        "model_name": metadata["model_name"],
        "targets": list(TARGETS),
    }


def validate_retained_modules(model):
    modules = []
    for name in TARGETS:
        module = module_by_name(model, name)
        if not isinstance(module, CUDAPackedTernaryLinear):
            raise RuntimeError(f"Retained runtime type mismatch for {name}")
        if module.__class__.__module__.startswith("packed_ternary.gpu007"):
            raise RuntimeError(f"GPU007 runtime unexpectedly selected for {name}")
        if hasattr(module, "weight"):
            raise RuntimeError(f"Ordinary FP16 weight fallback found for {name}")
        if module.packed_codes.dtype != torch.uint8:
            raise RuntimeError(f"Packed codes are not uint8 for {name}")
        if module.scales.dtype != torch.float16:
            raise RuntimeError(f"Scales are not FP16 for {name}")
        if module.packed_codes.device.type != "cuda" or module.scales.device.type != "cuda":
            raise RuntimeError(f"Packed buffers are not CUDA-resident for {name}")
        modules.append(module)
    if len(modules) != 3:
        raise RuntimeError("Expected exactly three retained packed modules")
    return True


def forward(model, input_ids, attention_mask):
    kwargs = {"input_ids": input_ids}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    with torch.inference_mode():
        return model(**kwargs).logits


def benchmark_prompt(model, tokenizer, prompt_name, prompt_text):
    encoded = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded["input_ids"].cuda()
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.cuda()
    torch.cuda.synchronize()

    ready_allocated = int(torch.cuda.memory_allocated())
    ready_reserved = int(torch.cuda.memory_reserved())
    for _ in range(WARMUP_RUNS):
        logits = forward(model, input_ids, attention_mask)
        torch.cuda.synchronize()
    warmup_peak = int(torch.cuda.max_memory_allocated())
    torch.cuda.reset_peak_memory_stats()

    timings = []
    final_logits = None
    for _ in range(TIMED_RUNS):
        torch.cuda.synchronize()
        start = time.perf_counter()
        final_logits = forward(model, input_ids, attention_mask)
        torch.cuda.synchronize()
        timings.append(float((time.perf_counter() - start) * 1000.0))

    timed_peak = int(torch.cuda.max_memory_allocated())
    timed_peak_reserved = int(torch.cuda.max_memory_reserved())
    finite = bool(torch.isfinite(final_logits).all().item())
    top1 = final_logits.argmax(dim=-1).detach().cpu().tolist()

    result = {
        "prompt_name": prompt_name,
        "prompt": prompt_text,
        "input_tokens": int(input_ids.shape[1]),
        "warmup_runs": WARMUP_RUNS,
        "timed_runs": TIMED_RUNS,
        "timings_ms": timings,
        "mean_ms": float(statistics.mean(timings)),
        "median_ms": float(statistics.median(timings)),
        "stdev_ms": float(statistics.stdev(timings)),
        "min_ms": float(min(timings)),
        "max_ms": float(max(timings)),
        "allocated_ready_bytes": ready_allocated,
        "reserved_ready_bytes": ready_reserved,
        "warmup_peak_allocated_bytes": warmup_peak,
        "timed_peak_allocated_bytes": timed_peak,
        "timed_peak_reserved_bytes": timed_peak_reserved,
        "finite": finite,
        "top1": top1,
    }
    del final_logits, input_ids, attention_mask
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return result


def load_condition(mode, checkpoint_identity):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, low_cpu_mem_usage=True
    )
    model.eval()
    load_result = None
    if mode == "retained_gpu004_packed":
        load_result = load_gpu004_runtime_modules(
            model=model,
            checkpoint_path=CHECKPOINT_PATH,
            selected_module_names=TARGETS,
            expected_model_name=MODEL_NAME,
        )
        if load_result.replaced_matrix_count != 3:
            raise RuntimeError("Retained runtime did not replace exactly three matrices")
        if tuple(load_result.loaded_module_names) != TARGETS:
            raise RuntimeError("Retained runtime target order mismatch")
    elif mode != "fp16_baseline":
        raise ValueError(mode)

    model = model.cuda()
    if load_result is not None:
        validate_retained_modules(model)
        if load_result.model_name != checkpoint_identity["model_name"]:
            raise RuntimeError("Loaded runtime checkpoint identity mismatch")
    return model, load_result


def top1_agreement(left, right):
    matches = 0
    total = 0
    if len(left) != len(right):
        return 0, 0
    for left_row, right_row in zip(left, right):
        if len(left_row) != len(right_row):
            continue
        for left_value, right_value in zip(left_row, right_row):
            total += 1
            matches += int(left_value == right_value)
    return matches, total


def mib(value):
    return value / (1024.0 * 1024.0)


def main():
    torch.manual_seed(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    checkpoint_identity = validate_checkpoint_identity()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    baseline_model, _ = load_condition("fp16_baseline", checkpoint_identity)
    baseline = {}
    for prompt_name, prompt_text in PROMPTS:
        baseline[prompt_name] = benchmark_prompt(
            baseline_model, tokenizer, prompt_name, prompt_text
        )
        if not baseline[prompt_name]["finite"]:
            raise RuntimeError(f"Baseline logits are non-finite for {prompt_name}")
    del baseline_model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    packed_model, load_result = load_condition(
        "retained_gpu004_packed", checkpoint_identity
    )
    packed = {}
    for prompt_name, prompt_text in PROMPTS:
        packed[prompt_name] = benchmark_prompt(
            packed_model, tokenizer, prompt_name, prompt_text
        )
        if not packed[prompt_name]["finite"]:
            raise RuntimeError(f"Packed logits are non-finite for {prompt_name}")
    del packed_model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    per_prompt = {}
    for prompt_name, _ in PROMPTS:
        b = baseline[prompt_name]
        c = packed[prompt_name]
        matches, total = top1_agreement(b["top1"], c["top1"])
        per_prompt[prompt_name] = {
            "baseline": b,
            "retained_gpu004_packed": c,
            "mean_latency_change_percent": (c["mean_ms"] - b["mean_ms"]) / b["mean_ms"] * 100.0,
            "median_latency_change_percent": (c["median_ms"] - b["median_ms"]) / b["median_ms"] * 100.0,
            "ready_allocated_change_bytes": c["allocated_ready_bytes"] - b["allocated_ready_bytes"],
            "timed_peak_allocated_change_bytes": c["timed_peak_allocated_bytes"] - b["timed_peak_allocated_bytes"],
            "top1_matches": matches,
            "top1_total": total,
        }

    result = {
        "experiment": "GPU008",
        "model": MODEL_NAME,
        "checkpoint": checkpoint_identity,
        "targets": list(TARGETS),
        "prompts": [{"name": name, "text": text} for name, text in PROMPTS],
        "seed": SEED,
        "gpu": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "warmup_runs": WARMUP_RUNS,
        "timed_runs": TIMED_RUNS,
        "retained_runtime": "GPU004 wrapper with frozen GPU003 CUDA operator",
        "replaced_matrix_count": int(load_result.replaced_matrix_count),
        "replaced_parameter_count": int(load_result.replaced_parameter_count),
        "persistent_packed_bytes": int(load_result.persistent_packed_bytes),
        "fp16_equivalent_bytes": int(load_result.fp16_equivalent_bytes),
        "per_prompt": per_prompt,
        "validation": {
            "exact_target_set_and_order": tuple(load_result.loaded_module_names) == TARGETS,
            "exactly_three_replaced": load_result.replaced_matrix_count == 3,
            "checkpoint_model_identity": checkpoint_identity["model_name"] == MODEL_NAME,
            "checkpoint_target_identity": checkpoint_identity["targets"] == list(TARGETS),
            "packed_structure_validated": True,
            "retained_runtime_is_gpu003_gpu004": True,
            "no_gpu007_runtime": True,
            "no_fp16_weight_fallback": True,
            "all_baseline_outputs_finite": all(item["baseline"]["finite"] for item in per_prompt.values()),
            "all_packed_outputs_finite": all(item["retained_gpu004_packed"]["finite"] for item in per_prompt.values()),
            "warmups_exact": all(
                item["baseline"]["warmup_runs"] == WARMUP_RUNS
                and item["retained_gpu004_packed"]["warmup_runs"] == WARMUP_RUNS
                for item in per_prompt.values()
            ),
            "timed_repetitions_exact": all(
                item["baseline"]["timed_runs"] == TIMED_RUNS
                and item["retained_gpu004_packed"]["timed_runs"] == TIMED_RUNS
                for item in per_prompt.values()
            ),
        },
    }
    output_path = Path("results/GPU008_raw.json")
    output_path.write_text(json.dumps(result, indent=2) + "\n")

    print("GPU008 — FULL-MODEL RETAINED GPU004 VALIDATION")
    print("GPU:", result["gpu"])
    for name, item in per_prompt.items():
        print(
            name,
            "tokens", item["baseline"]["input_tokens"],
            "FP16", f'{item["baseline"]["mean_ms"]:.6f} ms',
            "packed", f'{item["retained_gpu004_packed"]["mean_ms"]:.6f} ms',
            f'{item["mean_latency_change_percent"]:+.3f}%',
            "top1", f'{item["top1_matches"]}/{item["top1_total"]}',
        )
    print("Raw result:", output_path)
    print("GPU008 BENCHMARK: PASS")


if __name__ == "__main__":
    main()
