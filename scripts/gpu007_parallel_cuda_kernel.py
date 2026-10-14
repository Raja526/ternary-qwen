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
from transformers import AutoModelForCausalLM, AutoTokenizer

from packed_ternary.checkpoint import load_packed_checkpoint
from packed_ternary.gpu003_extension import load_gpu003_extension
from packed_ternary.gpu004_linear import CUDAPackedTernaryLinear
from packed_ternary.gpu007_extension import load_gpu007_extension
from packed_ternary.gpu007_linear import GPU007ParallelCUDAPackedTernaryLinear


MODEL_NAME = "Qwen/Qwen3.5-0.8B"
CHECKPOINT_PATH = Path("results/ENG003_multimatrix_checkpoint.pt")
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
WARMUP_RUNS = 10
TIMED_RUNS = 50

# Defined before any result interpretation. This tolerance is fixed for
# GPU007 because parallel FP32 reduction changes the addition order before
# the final FP16 conversion.
CORRECTNESS_ATOL = 2.0e-3
CORRECTNESS_RTOL = 2.0e-3
RELATIVE_DENOMINATOR = 2.0e-3


def module_by_name(model, name):
    modules = dict(model.named_modules())
    if name not in modules:
        raise KeyError(f"Missing target module: {name}")
    return modules[name]


def capture_activations(model, tokenizer):
    encoded = tokenizer(PROMPT, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded["input_ids"].cuda()
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.cuda()

    captured = {}
    hooks = []

    def make_hook(name):
        def hook(_module, inputs):
            if not inputs:
                raise RuntimeError(f"No activation input captured for {name}")
            if name in captured:
                raise RuntimeError(f"Target module called more than once: {name}")
            value = inputs[0]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Activation for {name} is not a tensor")
            captured[name] = value.detach().contiguous().clone()
        return hook

    for name in TARGETS:
        hooks.append(module_by_name(model, name).register_forward_pre_hook(make_hook(name)))

    with torch.inference_mode():
        model(input_ids=input_ids, attention_mask=attention_mask)

    torch.cuda.synchronize()
    for hook in hooks:
        hook.remove()

    if set(captured) != set(TARGETS):
        raise RuntimeError(f"Activation capture mismatch: {tuple(captured)}")
    return captured


def structural_validate(name, packed, frozen, optimized):
    if not isinstance(frozen, CUDAPackedTernaryLinear):
        raise TypeError(f"Frozen operator type mismatch for {name}")
    if not isinstance(optimized, GPU007ParallelCUDAPackedTernaryLinear):
        raise TypeError(f"GPU007 operator type mismatch for {name}")
    if frozen.in_features != optimized.in_features:
        raise RuntimeError(f"Input dimension mismatch for {name}")
    if frozen.out_features != optimized.out_features:
        raise RuntimeError(f"Output dimension mismatch for {name}")
    if frozen.group_size != optimized.group_size:
        raise RuntimeError(f"Group size mismatch for {name}")
    if not torch.equal(frozen.packed_codes, optimized.packed_codes):
        raise RuntimeError(f"Packed codes differ for {name}")
    if not torch.equal(frozen.scales, optimized.scales):
        raise RuntimeError(f"Scales differ for {name}")
    if frozen.packed_codes.dtype != torch.uint8 or optimized.packed_codes.dtype != torch.uint8:
        raise RuntimeError(f"Packed code dtype mismatch for {name}")
    if frozen.scales.dtype != torch.float16 or optimized.scales.dtype != torch.float16:
        raise RuntimeError(f"Scale dtype mismatch for {name}")
    for label, operator in (("GPU003", frozen), ("GPU007", optimized)):
        names = set(name for name, _ in operator.named_buffers())
        names.update(name for name, _ in operator.named_parameters())
        if "packed_codes" not in names or "scales" not in names:
            raise RuntimeError(f"{label} packed buffers missing for {name}")
        if "weight" in names:
            raise RuntimeError(f"{label} retains an ordinary weight fallback for {name}")
    return True


def correctness_compare(frozen_output, optimized_output):
    if tuple(frozen_output.shape) != tuple(optimized_output.shape):
        raise RuntimeError("Correctness output shape mismatch")
    finite_frozen = bool(torch.isfinite(frozen_output).all().item())
    finite_optimized = bool(torch.isfinite(optimized_output).all().item())
    if not finite_frozen or not finite_optimized:
        raise RuntimeError("Correctness output is non-finite")

    difference = (optimized_output.float() - frozen_output.float()).abs()
    reference_abs = frozen_output.float().abs()
    meaningful = reference_abs > RELATIVE_DENOMINATOR
    relative = difference[meaningful] / reference_abs[meaningful]
    close = torch.isclose(
        optimized_output.float(),
        frozen_output.float(),
        rtol=CORRECTNESS_RTOL,
        atol=CORRECTNESS_ATOL,
    )
    return {
        "finite_frozen": finite_frozen,
        "finite_gpu007": finite_optimized,
        "max_absolute_error": float(difference.max().item()),
        "mean_absolute_error": float(difference.mean().item()),
        "max_relative_error_meaningful": float(relative.max().item()) if relative.numel() else 0.0,
        "meaningful_relative_value_count": int(relative.numel()),
        "mismatch_count_under_tolerance": int((~close).sum().item()),
        "total_values": int(close.numel()),
        "tolerance_atol": CORRECTNESS_ATOL,
        "tolerance_rtol": CORRECTNESS_RTOL,
        "relative_denominator": RELATIVE_DENOMINATOR,
    }


def timed_call(operator, activation):
    torch.cuda.synchronize()
    start = time.perf_counter()
    output = operator(activation)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return output, float(elapsed_ms)


def benchmark_operator(operator, activation, expected_shape):
    warmup_outputs = []
    for _ in range(WARMUP_RUNS):
        output, _ = timed_call(operator, activation)
        warmup_outputs.append(output)
    for output in warmup_outputs:
        if tuple(output.shape) != expected_shape:
            raise RuntimeError("Warmup output shape mismatch")
        if not bool(torch.isfinite(output).all().item()):
            raise RuntimeError("Warmup output is non-finite")

    del warmup_outputs
    torch.cuda.reset_peak_memory_stats()
    timings = []
    final_output = None
    for _ in range(TIMED_RUNS):
        final_output, elapsed_ms = timed_call(operator, activation)
        timings.append(elapsed_ms)

    if tuple(final_output.shape) != expected_shape:
        raise RuntimeError("Timed output shape mismatch")
    finite = bool(torch.isfinite(final_output).all().item())
    if not finite:
        raise RuntimeError("Timed output is non-finite")

    return {
        "warmup_runs": WARMUP_RUNS,
        "timed_runs": TIMED_RUNS,
        "timings_ms": timings,
        "mean_ms": float(statistics.mean(timings)),
        "median_ms": float(statistics.median(timings)),
        "stdev_ms": float(statistics.stdev(timings)),
        "min_ms": float(min(timings)),
        "max_ms": float(max(timings)),
        "allocated_ready_bytes": int(torch.cuda.memory_allocated()),
        "timed_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "finite": finite,
        "output_shape": list(expected_shape),
        "final_output": final_output,
    }


def mib(value):
    return value / (1024.0 * 1024.0)


def main():
    torch.manual_seed(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, low_cpu_mem_usage=True
    )
    model.eval()
    model = model.cuda()
    captured = capture_activations(model, tokenizer)
    activation_shapes = {name: list(captured[name].shape) for name in TARGETS}
    input_tokens = int(captured[TARGETS[0]].shape[-2])
    for name in TARGETS:
        activation = captured[name]
        if activation.dtype != torch.float16 or activation.device.type != "cuda":
            raise RuntimeError(f"Invalid captured activation for {name}")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    metadata, packed_weights = load_packed_checkpoint(CHECKPOINT_PATH)
    if metadata["model_name"] != MODEL_NAME:
        raise ValueError("Checkpoint model-name mismatch")
    if set(packed_weights) != set(TARGETS):
        raise RuntimeError("Checkpoint target set mismatch")

    frozen_extension = load_gpu003_extension(verbose=False)
    optimized_extension = load_gpu007_extension(verbose=False)
    del frozen_extension, optimized_extension
    torch.cuda.synchronize()

    results = {}
    for name in TARGETS:
        activation = captured[name]
        packed = packed_weights[name]
        frozen = CUDAPackedTernaryLinear(packed).cuda()
        optimized = GPU007ParallelCUDAPackedTernaryLinear(packed).cuda()
        structural_validate(name, packed, frozen, optimized)
        frozen._get_extension()
        optimized._get_extension()
        torch.cuda.synchronize()

        expected_shape = (int(activation.shape[0]), int(activation.shape[1]), frozen.out_features)
        frozen_output = frozen(activation)
        optimized_output = optimized(activation)
        torch.cuda.synchronize()
        correctness = correctness_compare(frozen_output, optimized_output)
        if correctness["mismatch_count_under_tolerance"] != 0:
            raise RuntimeError(f"GPU007 correctness tolerance failed for {name}: {correctness}")
        del frozen_output, optimized_output

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        frozen_timing = benchmark_operator(frozen, activation, expected_shape)
        del frozen_timing["final_output"]
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        optimized_timing = benchmark_operator(optimized, activation, expected_shape)
        del optimized_timing["final_output"]

        for timing in (frozen_timing, optimized_timing):
            timing["ready_mib"] = mib(timing["allocated_ready_bytes"])
            timing["peak_mib"] = mib(timing["timed_peak_allocated_bytes"])

        results[name] = {
            "activation_shape": activation_shapes[name],
            "group_size": int(packed.group_size),
            "frozen_gpu003": frozen_timing,
            "gpu007": optimized_timing,
            "correctness": correctness,
            "latency_delta_ms": optimized_timing["mean_ms"] - frozen_timing["mean_ms"],
            "latency_change_percent": (optimized_timing["mean_ms"] - frozen_timing["mean_ms"]) / frozen_timing["mean_ms"] * 100.0,
            "ready_memory_delta_bytes": optimized_timing["allocated_ready_bytes"] - frozen_timing["allocated_ready_bytes"],
            "peak_memory_delta_bytes": optimized_timing["timed_peak_allocated_bytes"] - frozen_timing["timed_peak_allocated_bytes"],
        }
        del frozen, optimized
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    dominant = max(TARGETS, key=lambda name: results[name]["latency_delta_ms"])
    result = {
        "experiment": "GPU007",
        "model": MODEL_NAME,
        "checkpoint": str(CHECKPOINT_PATH),
        "targets": list(TARGETS),
        "prompt": PROMPT,
        "seed": SEED,
        "gpu": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "warmup_runs": WARMUP_RUNS,
        "timed_runs": TIMED_RUNS,
        "input_tokens": input_tokens,
        "activation_shapes": activation_shapes,
        "correctness_tolerance": {
            "atol": CORRECTNESS_ATOL,
            "rtol": CORRECTNESS_RTOL,
            "relative_denominator": RELATIVE_DENOMINATOR,
        },
        "dominant_target_by_latency_delta": dominant,
        "results": results,
        "validation": {
            "target_count": len(TARGETS),
            "target_order_exact": True,
            "same_captured_activation_per_pair": True,
            "all_outputs_finite": all(
                item["correctness"]["finite_frozen"] and item["correctness"]["finite_gpu007"]
                for item in results.values()
            ),
            "all_correctness_mismatch_counts_zero": all(
                item["correctness"]["mismatch_count_under_tolerance"] == 0
                for item in results.values()
            ),
            "structural_checks_pass": True,
            "warmups_exact": all(
                item["frozen_gpu003"]["warmup_runs"] == WARMUP_RUNS
                and item["gpu007"]["warmup_runs"] == WARMUP_RUNS
                for item in results.values()
            ),
            "timed_repetitions_exact": all(
                item["frozen_gpu003"]["timed_runs"] == TIMED_RUNS
                and item["gpu007"]["timed_runs"] == TIMED_RUNS
                for item in results.values()
            ),
            "no_fp16_weight_fallback": True,
        },
    }

    output_path = Path("results/GPU007_raw.json")
    output_path.write_text(json.dumps(result, indent=2) + "\n")

    print("GPU007 — PARALLEL CUDA PACKED-TERNARY KERNEL")
    print("GPU:", result["gpu"])
    for name in TARGETS:
        item = results[name]
        print(
            name,
            "GPU003", f'{item["frozen_gpu003"]["mean_ms"]:.6f} ms',
            "GPU007", f'{item["gpu007"]["mean_ms"]:.6f} ms',
            f'{item["latency_change_percent"]:+.3f}%',
            "mismatches", item["correctness"]["mismatch_count_under_tolerance"],
        )
    print("Dominant target:", dominant)
    print("Raw result:", output_path)
    print("GPU007 BENCHMARK: PASS")


if __name__ == "__main__":
    main()
