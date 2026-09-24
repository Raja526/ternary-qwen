from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from ternary.quantizer import groupwise_ternarize_weight

from packed_ternary.packed_weight import (
    quantize_to_packed,
    reconstruct_packed,
)

from packed_ternary.serialization import (
    save_packed_weight,
    load_packed_weight,
)


MODEL_NAME = "Qwen/Qwen3.5-0.8B"
TARGET_MODULE = "model.layers.3.self_attn.k_proj"

GROUP_SIZE = 128
THRESHOLD_MULTIPLIER = 0.50

OUTPUT_FILE = Path(
    "results/ENG002_k_proj_packed.pt"
)


def main():

    print("=" * 88)
    print("ENG002 — PACKED TERNARY SERIALIZATION VALIDATION")
    print("=" * 88)

    print()
    print("Model :", MODEL_NAME)
    print("Target:", TARGET_MODULE)

    # --------------------------------------------------------------
    # Load real Qwen FP16 weight
    # --------------------------------------------------------------

    print()
    print("Loading FP16 model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    modules = dict(
        model.named_modules()
    )

    if TARGET_MODULE not in modules:
        raise RuntimeError(
            f"Target module not found: {TARGET_MODULE}"
        )

    module = modules[
        TARGET_MODULE
    ]

    if not isinstance(
        module,
        torch.nn.Linear,
    ):
        raise TypeError(
            f"{TARGET_MODULE} is not torch.nn.Linear"
        )

    weight = (
        module
        .weight
        .detach()
    )

    print()
    print(
        "Weight shape             :",
        tuple(weight.shape),
    )

    print(
        "Weight dtype             :",
        weight.dtype,
    )

    print(
        "Weight parameters        :",
        weight.numel(),
    )

    # --------------------------------------------------------------
    # Frozen scientific reference
    # --------------------------------------------------------------

    print()
    print(
        "Running frozen scientific quantizer..."
    )

    reference = groupwise_ternarize_weight(
        weight,
        method="optimized",
        group_size=GROUP_SIZE,
        threshold_multiplier=THRESHOLD_MULTIPLIER,
    )

    reference_weight = reference[
        "reconstructed"
    ]

    # --------------------------------------------------------------
    # Create ENG001 packed representation
    # --------------------------------------------------------------

    print(
        "Creating packed representation..."
    )

    packed = quantize_to_packed(
        weight,
        group_size=GROUP_SIZE,
        threshold_multiplier=THRESHOLD_MULTIPLIER,
        scale_dtype=torch.float16,
    )

    # --------------------------------------------------------------
    # Record metadata before serialization
    # --------------------------------------------------------------

    original_metadata = {
        "shape": packed.original_shape,
        "original_numel": packed.original_numel,
        "padded_numel": packed.padded_numel,
        "group_size": packed.group_size,
        "dtype": packed.original_dtype,
        "code_dtype": packed.packed_codes.dtype,
        "scale_dtype": packed.scales.dtype,
        "code_bytes": packed.code_bytes,
        "scale_bytes": packed.scale_bytes,
        "payload_bytes": packed.packed_storage_bytes,
    }

    # --------------------------------------------------------------
    # Save
    # --------------------------------------------------------------

    print(
        "Saving packed representation..."
    )

    save_packed_weight(
        packed,
        OUTPUT_FILE,
    )

    if not OUTPUT_FILE.exists():
        raise RuntimeError(
            "Serialized output file was not created."
        )

    disk_bytes = (
        OUTPUT_FILE.stat().st_size
    )

    # --------------------------------------------------------------
    # Remove packed object before load.
    #
    # This makes the next reconstruction depend on the serialized
    # representation rather than the original PackedTernaryWeight.
    # --------------------------------------------------------------

    del packed

    # --------------------------------------------------------------
    # Load
    # --------------------------------------------------------------

    print(
        "Loading packed representation..."
    )

    loaded = load_packed_weight(
        OUTPUT_FILE
    )

    # --------------------------------------------------------------
    # Validate metadata
    # --------------------------------------------------------------

    loaded_metadata = {
        "shape": loaded.original_shape,
        "original_numel": loaded.original_numel,
        "padded_numel": loaded.padded_numel,
        "group_size": loaded.group_size,
        "dtype": loaded.original_dtype,
        "code_dtype": loaded.packed_codes.dtype,
        "scale_dtype": loaded.scales.dtype,
        "code_bytes": loaded.code_bytes,
        "scale_bytes": loaded.scale_bytes,
        "payload_bytes": loaded.packed_storage_bytes,
    }

    metadata_match = (
        original_metadata
        == loaded_metadata
    )

    # --------------------------------------------------------------
    # Reconstruct from LOADED representation
    # --------------------------------------------------------------

    print(
        "Reconstructing loaded packed weight..."
    )

    reconstructed = reconstruct_packed(
        loaded
    )

    # --------------------------------------------------------------
    # Exact comparison
    # --------------------------------------------------------------

    mismatch_count = int(
        (
            reference_weight
            != reconstructed
        )
        .sum()
        .item()
    )

    total_values = (
        reference_weight.numel()
    )

    exact_count = (
        total_values
        - mismatch_count
    )

    exact_percent = (
        100.0
        * exact_count
        / total_values
    )

    difference = (
        reference_weight.float()
        - reconstructed.float()
    ).abs()

    max_abs_difference = float(
        difference.max().item()
    )

    mean_abs_difference = float(
        difference.mean().item()
    )

    # --------------------------------------------------------------
    # Storage measurements
    # --------------------------------------------------------------

    logical_payload_bytes = (
        loaded.packed_storage_bytes
    )

    original_fp16_bytes = (
        loaded.original_numel
        * 2
    )

    serialization_overhead_bytes = (
        disk_bytes
        - logical_payload_bytes
    )

    serialization_overhead_percent = (
        100.0
        * serialization_overhead_bytes
        / logical_payload_bytes
    )

    actual_disk_ratio = (
        disk_bytes
        / original_fp16_bytes
    )

    actual_disk_compression = (
        original_fp16_bytes
        / disk_bytes
    )

    # --------------------------------------------------------------
    # Report
    # --------------------------------------------------------------

    print()
    print("-" * 88)
    print("SERIALIZED METADATA")
    print("-" * 88)

    print(
        "Original shape           :",
        loaded.original_shape,
    )

    print(
        "Original numel           :",
        loaded.original_numel,
    )

    print(
        "Padded numel             :",
        loaded.padded_numel,
    )

    print(
        "Group size               :",
        loaded.group_size,
    )

    print(
        "Original dtype           :",
        loaded.original_dtype,
    )

    print(
        "Packed-code dtype        :",
        loaded.packed_codes.dtype,
    )

    print(
        "Scale dtype              :",
        loaded.scales.dtype,
    )

    print()
    print(
        "Metadata exact match     :",
        metadata_match,
    )

    print()
    print("-" * 88)
    print("RECONSTRUCTION AFTER SAVE/LOAD")
    print("-" * 88)

    print(
        "Exact FP16 matches       :",
        exact_count,
    )

    print(
        "FP16 mismatches          :",
        mismatch_count,
    )

    print(
        "Exact match percentage   :",
        f"{exact_percent:.8f}%",
    )

    print(
        "Maximum abs difference   :",
        max_abs_difference,
    )

    print(
        "Mean abs difference      :",
        mean_abs_difference,
    )

    print()
    print("-" * 88)
    print("STORAGE")
    print("-" * 88)

    print(
        "Original FP16 bytes      :",
        original_fp16_bytes,
    )

    print(
        "Packed code bytes        :",
        loaded.code_bytes,
    )

    print(
        "FP16 scale bytes         :",
        loaded.scale_bytes,
    )

    print(
        "Logical payload bytes    :",
        logical_payload_bytes,
    )

    print(
        "Actual disk file bytes   :",
        disk_bytes,
    )

    print(
        "Serialization overhead   :",
        serialization_overhead_bytes,
        "bytes",
    )

    print(
        "Overhead vs payload      :",
        f"{serialization_overhead_percent:.6f}%",
    )

    print()
    print(
        "Logical payload ratio    :",
        f"{100.0 * logical_payload_bytes / original_fp16_bytes:.6f}%",
    )

    print(
        "Logical compression      :",
        f"{original_fp16_bytes / logical_payload_bytes:.6f}x",
    )

    print()
    print(
        "Actual disk ratio        :",
        f"{actual_disk_ratio * 100:.6f}%",
    )

    print(
        "Actual disk compression  :",
        f"{actual_disk_compression:.6f}x",
    )

    # --------------------------------------------------------------
    # Strict acceptance tests
    # --------------------------------------------------------------

    print()
    print("-" * 88)
    print("VALIDATION")
    print("-" * 88)

    assert metadata_match, (
        "Metadata changed after serialization."
    )

    assert mismatch_count == 0, (
        "Loaded packed weight does not exactly "
        "reproduce frozen FP16 reconstruction."
    )

    assert max_abs_difference == 0.0
    assert mean_abs_difference == 0.0

    assert (
        reconstructed.shape
        == reference_weight.shape
    )

    assert (
        reconstructed.dtype
        == reference_weight.dtype
    )

    assert (
        disk_bytes
        >= logical_payload_bytes
    )

    print(
        "Metadata preservation    : PASS"
    )

    print(
        "Save/load reconstruction : EXACT MATCH"
    )

    print(
        "Shape preservation       : PASS"
    )

    print(
        "Dtype preservation       : PASS"
    )

    print(
        "Disk measurement         : PASS"
    )

    print()
    print("=" * 88)
    print("ENG002 SERIALIZATION VALIDATION: PASS")
    print("=" * 88)

    del reconstructed
    del loaded
    del reference_weight
    del reference
    del model


if __name__ == "__main__":
    main()
