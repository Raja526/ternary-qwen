import gc
import json
import math
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from packed_ternary.checkpoint import load_packed_checkpoint
from packed_ternary.gpu004_linear import CUDAPackedTernaryLinear


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


def timed_call(fn, activation):
    torch.cuda.synchronize()
    start = time.perf_counter()
    output = fn(activation)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return output, float(elapsed_ms)


def benchmark_impl(fn, activation, expected_shape):
    warmup_outputs = []
    for _ in range(WARMUP_RUNS):
        output, _ = timed_call(fn, activation)
        warmup_outputs.append(output)

    for output in warmup_outputs:
        if tuple(output.shape) != expected_shape:
            raise RuntimeError("Warmup output shape mismatch")
        if not bool(torch.isfinite(output).all().item()):
            raise RuntimeError("Warmup output contains non-finite values")

    torch.cuda.reset_peak_memory_stats()
    timings = []
    final_output = None
    for _ in range(TIMED_RUNS):
        final_output, elapsed_ms = timed_call(fn, activation)
        timings.append(elapsed_ms)

    if tuple(final_output.shape) != expected_shape:
        raise RuntimeError("Timed output shape mismatch")
    finite = bool(torch.isfinite(final_output).all().item())
    if not finite:
        raise RuntimeError("Timed output contains non-finite values")

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

    original = {}
    for name in TARGETS:
        module = module_by_name(model, name)
        if not isinstance(module, torch.nn.Linear):
            raise TypeError(f"{name} is not torch.nn.Linear")
        original[name] = {
            "shape": [int(module.out_features), int(module.in_features)],
            "weight": module.weight.detach().clone(),
            "bias": None if module.bias is None else module.bias.detach().clone(),
            "dtype": str(module.weight.dtype),
        }

    model = model.cuda()
    captured = capture_activations(model, tokenizer)
    input_tokens = int(captured[TARGETS[0]].shape[-2])
    activation_shapes = {name: list(captured[name].shape) for name in TARGETS}
    for name in TARGETS:
        if captured[name].dtype != torch.float16 or captured[name].device.type != "cuda":
            raise RuntimeError(f"Invalid captured activation for {name}")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    metadata, packed_weights = load_packed_checkpoint(CHECKPOINT_PATH)
    if metadata["model_name"] != MODEL_NAME:
        raise ValueError("Checkpoint model-name mismatch")
    if tuple(packed_weights) != TARGETS:
        raise RuntimeError("Checkpoint target ordering does not match GPU006")

    results = {}
    for name in TARGETS:
        activation = captured[name]
        info = original[name]
        packed = packed_weights[name]
        expected_shape = (int(activation.shape[0] * activation.shape[1]), info["shape"][0])

        if tuple(packed.original_shape) != tuple(info["shape"]):
            raise RuntimeError(f"Packed shape mismatch for {name}")
        if packed.packed_codes.dtype != torch.uint8 or packed.scales.dtype != torch.float16:
            raise RuntimeError(f"Packed dtype mismatch for {name}")
        if packed.padded_numel != packed.original_numel:
            raise RuntimeError(f"Unexpected padding for {name}")
        if info["shape"][1] % packed.group_size != 0 or info["shape"][1] % 4 != 0:
            raise RuntimeError(f"Unsupported packed layout for {name}")

        weight = info["weight"].cuda().contiguous()
        bias = None if info["bias"] is None else info["bias"].cuda().contiguous()
        packed_module = CUDAPackedTernaryLinear(packed, bias=info["bias"])
        packed_module = packed_module.cuda()
        packed_module._get_extension()
        torch.cuda.synchronize()

        flat_activation = activation.reshape(-1, activation.shape[-1]).contiguous()
        reference = lambda x, weight=weight, bias=bias: F.linear(x, weight, bias)
        candidate = lambda x, module=packed_module: module(x)

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        fp16 = benchmark_impl(reference, flat_activation, expected_shape)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        packed_result = benchmark_impl(candidate, flat_activation, expected_shape)

        fp16["ready_mib"] = mib(fp16["allocated_ready_bytes"])
        fp16["peak_mib"] = mib(fp16["timed_peak_allocated_bytes"])
        packed_result["ready_mib"] = mib(packed_result["allocated_ready_bytes"])
        packed_result["peak_mib"] = mib(packed_result["timed_peak_allocated_bytes"])
        results[name] = {
            "activation_shape": list(activation.shape),
            "flattened_activation_shape": list(flat_activation.shape),
            "out_features": info["shape"][0],
            "in_features": info["shape"][1],
            "group_size": int(packed.group_size),
            "packed_code_count": int(packed.packed_codes.numel()),
            "scale_count": int(packed.scales.numel()),
            "fp16": fp16,
            "packed": packed_result,
            "latency_change_percent": (packed_result["mean_ms"] - fp16["mean_ms"]) / fp16["mean_ms"] * 100.0,
            "latency_delta_ms": packed_result["mean_ms"] - fp16["mean_ms"],
            "ready_memory_delta_bytes": packed_result["allocated_ready_bytes"] - fp16["allocated_ready_bytes"],
            "peak_memory_delta_bytes": packed_result["timed_peak_allocated_bytes"] - fp16["timed_peak_allocated_bytes"],
        }
        del weight, bias, packed_module, flat_activation
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    dominant = max(TARGETS, key=lambda name: results[name]["latency_delta_ms"])
    result = {
        "experiment": "GPU006",
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
        "dominant_bottleneck_target": dominant,
        "results": results,
        "validation": {
            "target_count": len(TARGETS),
            "target_order_exact": True,
            "all_outputs_finite": all(
                results[name]["fp16"]["finite"] and results[name]["packed"]["finite"]
                for name in TARGETS
            ),
            "all_output_shapes_match": True,
            "same_captured_activation_per_pair": True,
            "warmups_exact": all(
                results[name]["fp16"]["warmup_runs"] == WARMUP_RUNS
                and results[name]["packed"]["warmup_runs"] == WARMUP_RUNS
                for name in TARGETS
            ),
            "timed_repetitions_exact": all(
                results[name]["fp16"]["timed_runs"] == TIMED_RUNS
                and results[name]["packed"]["timed_runs"] == TIMED_RUNS
                for name in TARGETS
            ),
        },
    }

    output_path = Path("results/GPU006_raw.json")
    output_path.write_text(json.dumps(result, indent=2) + "\n")

    print("GPU006 — CUDA KERNEL BOTTLENECK CHARACTERIZATION")
    print("GPU:", result["gpu"])
    print("Captured activation shapes:")
    for name in TARGETS:
        item = results[name]
        print(name, item["activation_shape"], "FP16", f'{item["fp16"]["mean_ms"]:.6f} ms', "packed", f'{item["packed"]["mean_ms"]:.6f} ms', f'{item["latency_change_percent"]:+.3f}%')
    print("Dominant bottleneck:", dominant)
    print("Raw result:", output_path)
    print("GPU006 BENCHMARK: PASS")


if __name__ == "__main__":
    main()
