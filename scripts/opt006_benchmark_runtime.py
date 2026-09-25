"""
OPT006 — branch-reduced vs table-dispatch CPU benchmark.

Controlled comparison:

branch_reduced:
    Frozen OPT005 BranchReducedPackedTernaryLinear.
    Uses comparison-mask ternary dispatch.

table_dispatch:
    OPT006 TableDispatchPackedTernaryLinear.
    Uses fixed lookup-table ternary dispatch.

All other benchmark conditions are held constant.
"""

import argparse
import gc
import json
import os
import platform
import statistics
import time
from pathlib import Path

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from packed_ternary.checkpoint import load_packed_checkpoint



from packed_ternary.table_dispatch_linear import (
    TableDispatchPackedTernaryLinear,
)

from packed_ternary.table_dispatch_runtime_loader import (
    load_table_dispatch_runtime_modules,
)

from packed_ternary.branch_reduced_linear import (
    BranchReducedPackedTernaryLinear,
)

from packed_ternary.branch_reduced_runtime_loader import (
    load_branch_reduced_runtime_modules,
)


MODEL_NAME = "Qwen/Qwen3.5-0.8B"

CHECKPOINT_PATH = Path(
    "results/ENG003_multimatrix_checkpoint.pt"
)

TARGETS = [
    "model.layers.3.self_attn.k_proj",
    "model.layers.4.linear_attn.out_proj",
    "model.layers.4.linear_attn.in_proj_qkv",
]

PROMPT = (
    "A box contains 12 red balls and 8 blue balls. "
    "How many balls are in the box?"
)


WARMUP_RUNS = 2
TIMED_RUNS = 5

SEED = 20260924


def rss_bytes():
    return psutil.Process(
        os.getpid()
    ).memory_info().rss


def mib(value):
    return value / (1024 ** 2)


def get_module(model, name):
    modules = dict(model.named_modules())

    if name not in modules:
        raise KeyError(
            f"Module not found: {name}"
        )

    return modules[name]


def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    model.eval()

    return model


def percentile(values, q):
    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    position = (
        (len(ordered) - 1) * q
    )

    lower = int(position)
    upper = min(
        lower + 1,
        len(ordered) - 1,
    )

    fraction = position - lower

    return (
        ordered[lower]
        * (1.0 - fraction)
        + ordered[upper]
        * fraction
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "branch_reduced",
            "table_dispatch",
        ],
    )

    args = parser.parse_args()

    mode = args.mode

    torch.manual_seed(SEED)

    # Freeze thread configuration explicitly.
    torch.set_num_threads(6)
    torch.set_num_interop_threads(6)

    process = psutil.Process(
        os.getpid()
    )

    gc.collect()

    rss_start = rss_bytes()

    metadata, packed_weights = (
        load_packed_checkpoint(
            CHECKPOINT_PATH
        )
    )

    assert (
        metadata["model_name"]
        == MODEL_NAME
    )

    assert (
        set(packed_weights.keys())
        == set(TARGETS)
    )

    selected_parameters = sum(
        packed_weights[name]
        .original_numel
        for name in TARGETS
    )

    packed_bytes = sum(
        packed_weights[name]
        .packed_storage_bytes
        for name in TARGETS
    )

    fp16_bytes = sum(
        packed_weights[name]
        .original_fp16_bytes
        for name in TARGETS
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

    rss_before_model = (
        rss_bytes()
    )

    model = load_model()

    rss_after_model = (
        rss_bytes()
    )

    if mode == "branch_reduced":

        result = (
            load_branch_reduced_runtime_modules(
                model=model,
                checkpoint_path=(
                    CHECKPOINT_PATH
                ),
                selected_module_names=(
                    TARGETS
                ),
                expected_model_name=(
                    MODEL_NAME
                ),
            )
        )

        expected_type = (
            BranchReducedPackedTernaryLinear
        )

    elif mode == "table_dispatch":

        result = (
            load_table_dispatch_runtime_modules(
                model=model,
                checkpoint_path=(
                    CHECKPOINT_PATH
                ),
                selected_module_names=(
                    TARGETS
                ),
                expected_model_name=(
                    MODEL_NAME
                ),
            )
        )

        expected_type = (
            TableDispatchPackedTernaryLinear
        )

    assert (
        result.replaced_matrix_count
        == 3
    )

    assert (
        result.replaced_parameter_count
        == selected_parameters
    )

    assert (
        result.persistent_packed_bytes
        == packed_bytes
    )

    assert (
        result.fp16_equivalent_bytes
        == fp16_bytes
    )

    for name in TARGETS:

        module = get_module(
            model,
            name,
        )

        assert isinstance(
            module,
            expected_type,
        )

        assert (
            "weight"
            not in dict(
                module.named_parameters()
            )
        )

    gc.collect()

    rss_ready = rss_bytes()

    # ----------------------------------------------------------
    # Warm-up.
    # ----------------------------------------------------------

    with torch.inference_mode():

        for _ in range(
            WARMUP_RUNS
        ):
            output = model(
                input_ids=input_ids,
                attention_mask=(
                    attention_mask
                ),
                use_cache=False,
            )

            # Force output access before timing continues.
            _ = float(
                output.logits[
                    0,
                    -1,
                    0,
                ].item()
            )

    del output

    gc.collect()

    rss_after_warmup = (
        rss_bytes()
    )

    # ----------------------------------------------------------
    # Timed forwards.
    # ----------------------------------------------------------

    latencies_ms = []

    rss_samples = []

    with torch.inference_mode():

        for _ in range(
            TIMED_RUNS
        ):

            start = (
                time.perf_counter()
            )

            output = model(
                input_ids=input_ids,
                attention_mask=(
                    attention_mask
                ),
                use_cache=False,
            )

            _ = float(
                output.logits[
                    0,
                    -1,
                    0,
                ].item()
            )

            end = (
                time.perf_counter()
            )

            latencies_ms.append(
                (end - start)
                * 1000.0
            )

            rss_samples.append(
                rss_bytes()
            )

    logits_shape = tuple(
        output.logits.shape
    )

    logits_dtype = str(
        output.logits.dtype
    )

    del output

    gc.collect()

    rss_final = rss_bytes()

    result = {
        "experiment": "OPT006",
        "mode": mode,

        "model": MODEL_NAME,

        "cpu": (
            platform.processor()
            or "x86_64"
        ),

        "torch_version": (
            torch.__version__
        ),

        "torch_threads": (
            torch.get_num_threads()
        ),

        "torch_interop_threads": (
            torch.get_num_interop_threads()
        ),

        "warmup_runs": (
            WARMUP_RUNS
        ),

        "timed_runs": (
            TIMED_RUNS
        ),

        "input_tokens": int(
            input_ids.shape[1]
        ),

        "selected_matrices": (
            len(TARGETS)
        ),

        "selected_parameters": (
            selected_parameters
        ),

        "selected_fp16_bytes": (
            fp16_bytes
        ),

        "selected_packed_bytes": (
            packed_bytes
        ),

        "selected_storage_compression": (
            fp16_bytes
            / packed_bytes
        ),

        "rss_start_bytes": (
            rss_start
        ),

        "rss_before_model_bytes": (
            rss_before_model
        ),

        "rss_after_model_bytes": (
            rss_after_model
        ),

        "rss_ready_bytes": (
            rss_ready
        ),

        "rss_after_warmup_bytes": (
            rss_after_warmup
        ),

        "rss_timed_max_bytes": (
            max(rss_samples)
        ),

        "rss_final_bytes": (
            rss_final
        ),

        "latencies_ms": (
            latencies_ms
        ),

        "latency_mean_ms": (
            statistics.mean(
                latencies_ms
            )
        ),

        "latency_median_ms": (
            statistics.median(
                latencies_ms
            )
        ),

        "latency_min_ms": (
            min(
                latencies_ms
            )
        ),

        "latency_max_ms": (
            max(
                latencies_ms
            )
        ),

        "latency_p90_ms": (
            percentile(
                latencies_ms,
                0.90,
            )
        ),

        "logits_shape": (
            logits_shape
        ),

        "logits_dtype": (
            logits_dtype
        ),

        "process_rss_mib": {
            "start": mib(
                rss_start
            ),
            "before_model": mib(
                rss_before_model
            ),
            "after_model": mib(
                rss_after_model
            ),
            "ready": mib(
                rss_ready
            ),
            "after_warmup": mib(
                rss_after_warmup
            ),
            "timed_max": mib(
                max(
                    rss_samples
                )
            ),
            "final": mib(
                rss_final
            ),
        },
    }

    print()
    print("=" * 90)
    print(
        f"OPT006 BENCHMARK — {mode.upper()}"
    )
    print("=" * 90)

    print(
        "Selected matrices       :",
        len(TARGETS),
    )

    print(
        "Selected parameters     :",
        selected_parameters,
    )

    print(
        "FP16 selected bytes     :",
        fp16_bytes,
    )

    print(
        "Packed selected bytes   :",
        packed_bytes,
    )

    print(
        "Storage compression     :",
        f"{fp16_bytes / packed_bytes:.6f}x",
    )

    print()
    print(
        "RSS ready MiB           :",
        f"{mib(rss_ready):.3f}",
    )

    print(
        "RSS after warmup MiB    :",
        f"{mib(rss_after_warmup):.3f}",
    )

    print(
        "RSS timed max MiB       :",
        f"{mib(max(rss_samples)):.3f}",
    )

    print()
    print(
        "Warmup runs             :",
        WARMUP_RUNS,
    )

    print(
        "Timed runs              :",
        TIMED_RUNS,
    )

    print(
        "Latency samples ms      :",
        ", ".join(
            f"{x:.3f}"
            for x
            in latencies_ms
        ),
    )

    print(
        "Latency mean ms         :",
        f"{statistics.mean(latencies_ms):.3f}",
    )

    print(
        "Latency median ms       :",
        f"{statistics.median(latencies_ms):.3f}",
    )

    print(
        "Latency min ms          :",
        f"{min(latencies_ms):.3f}",
    )

    print(
        "Latency max ms          :",
        f"{max(latencies_ms):.3f}",
    )

    print(
        "Latency p90 ms          :",
        f"{percentile(latencies_ms, 0.90):.3f}",
    )

    print()
    print(
        "Logits shape            :",
        logits_shape,
    )

    print(
        "Logits dtype            :",
        logits_dtype,
    )

    print()
    print("RESULT_JSON")
    print(
        json.dumps(
            result,
            indent=2,
        )
    )

    print()
    print("=" * 90)
    print(
        f"OPT006 {mode.upper()} BENCHMARK: PASS"
    )
    print("=" * 90)


if __name__ == "__main__":
    main()
