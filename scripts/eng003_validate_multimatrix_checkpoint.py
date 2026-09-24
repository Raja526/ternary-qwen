from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from ternary.quantizer import groupwise_ternarize_weight

from packed_ternary.packed_weight import (
    quantize_to_packed,
    reconstruct_packed,
)

from packed_ternary.checkpoint import (
    save_packed_checkpoint,
    load_packed_checkpoint,
)


MODEL_NAME = "Qwen/Qwen3.5-0.8B"

GROUP_SIZE = 128
THRESHOLD_MULTIPLIER = 0.50

TARGET_MODULES = [
    "model.layers.3.self_attn.k_proj",
    "model.layers.4.linear_attn.out_proj",
    "model.layers.4.linear_attn.in_proj_qkv",
]

OUTPUT_FILE = Path(
    "results/ENG003_multimatrix_checkpoint.pt"
)


def main():

    print("=" * 100)
    print("ENG003 — REAL QWEN MULTI-MATRIX PACKED CHECKPOINT VALIDATION")
    print("=" * 100)

    print()
    print("Model               :", MODEL_NAME)
    print("Group size          :", GROUP_SIZE)
    print("Threshold multiplier:", THRESHOLD_MULTIPLIER)
    print("Target matrices     :", len(TARGET_MODULES))

    for name in TARGET_MODULES:
        print("  -", name)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Quantize each target independently.
    #
    # Keep:
    #   1. frozen scientific reconstruction
    #   2. engineering packed representation
    # ------------------------------------------------------------------

    references = {}
    packed_weights = {}

    original_total_parameters = 0

    print()
    print("-" * 100)
    print("PACKING MATRICES")
    print("-" * 100)

    for name in TARGET_MODULES:

        if name not in modules:
            raise RuntimeError(
                f"Target module not found: {name}"
            )

        module = modules[name]

        if not isinstance(
            module,
            torch.nn.Linear,
        ):
            raise TypeError(
                f"{name} is not torch.nn.Linear"
            )

        weight = (
            module
            .weight
            .detach()
        )

        scientific = groupwise_ternarize_weight(
            weight,
            method="optimized",
            group_size=GROUP_SIZE,
            threshold_multiplier=THRESHOLD_MULTIPLIER,
        )

        references[name] = scientific[
            "reconstructed"
        ].cpu()

        packed = quantize_to_packed(
            weight,
            group_size=GROUP_SIZE,
            threshold_multiplier=THRESHOLD_MULTIPLIER,
            scale_dtype=torch.float16,
        )

        packed_weights[name] = packed

        original_total_parameters += (
            weight.numel()
        )

        print()
        print(name)
        print(
            "  shape              :",
            tuple(weight.shape),
        )
        print(
            "  parameters         :",
            weight.numel(),
        )
        print(
            "  groups             :",
            packed.num_groups,
        )
        print(
            "  padding            :",
            packed.padded_numel
            - packed.original_numel,
        )
        print(
            "  packed code bytes  :",
            packed.code_bytes,
        )
        print(
            "  scale bytes        :",
            packed.scale_bytes,
        )
        print(
            "  logical bytes      :",
            packed.packed_storage_bytes,
        )

    # ------------------------------------------------------------------
    # Pre-save aggregate storage
    # ------------------------------------------------------------------

    original_fp16_bytes = (
        original_total_parameters * 2
    )

    logical_payload_bytes = sum(
        packed.packed_storage_bytes
        for packed in packed_weights.values()
    )

    logical_code_bytes = sum(
        packed.code_bytes
        for packed in packed_weights.values()
    )

    logical_scale_bytes = sum(
        packed.scale_bytes
        for packed in packed_weights.values()
    )

    # ------------------------------------------------------------------
    # Save one checkpoint containing all matrices
    # ------------------------------------------------------------------

    print()
    print("Saving multi-matrix checkpoint...")

    save_packed_checkpoint(
        weights=packed_weights,
        path=OUTPUT_FILE,
        model_name=MODEL_NAME,
        group_size=GROUP_SIZE,
        threshold_multiplier=THRESHOLD_MULTIPLIER,
    )

    if not OUTPUT_FILE.exists():
        raise RuntimeError(
            "Checkpoint file was not created."
        )

    disk_bytes = (
        OUTPUT_FILE.stat().st_size
    )

    # ------------------------------------------------------------------
    # Remove original packed objects.
    #
    # Reconstruction below must depend on loaded checkpoint.
    # ------------------------------------------------------------------

    del packed_weights

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    print(
        "Loading multi-matrix checkpoint..."
    )

    metadata, loaded_weights = (
        load_packed_checkpoint(
            OUTPUT_FILE
        )
    )

    # ------------------------------------------------------------------
    # Check checkpoint-level metadata
    # ------------------------------------------------------------------

    expected_names = set(
        TARGET_MODULES
    )

    loaded_names = set(
        loaded_weights.keys()
    )

    checkpoint_metadata_match = (
        metadata["model_name"]
        == MODEL_NAME

        and metadata["group_size"]
        == GROUP_SIZE

        and metadata[
            "threshold_multiplier"
        ]
        == THRESHOLD_MULTIPLIER

        and metadata["weight_count"]
        == len(TARGET_MODULES)

        and loaded_names
        == expected_names
    )

    # ------------------------------------------------------------------
    # Reconstruct and compare every matrix
    # ------------------------------------------------------------------

    print()
    print("-" * 100)
    print("RECONSTRUCTION VALIDATION")
    print("-" * 100)

    total_values = 0
    total_mismatches = 0
    global_max_difference = 0.0

    matrix_results = []

    for name in TARGET_MODULES:

        reference = references[
            name
        ]

        loaded = loaded_weights[
            name
        ]

        reconstructed = reconstruct_packed(
            loaded
        )

        mismatch_count = int(
            (
                reference
                != reconstructed
            )
            .sum()
            .item()
        )

        difference = (
            reference.float()
            - reconstructed.float()
        ).abs()

        max_difference = float(
            difference.max().item()
        )

        mean_difference = float(
            difference.mean().item()
        )

        values = (
            reference.numel()
        )

        total_values += values

        total_mismatches += (
            mismatch_count
        )

        global_max_difference = max(
            global_max_difference,
            max_difference,
        )

        shape_match = (
            reconstructed.shape
            == reference.shape
        )

        dtype_match = (
            reconstructed.dtype
            == reference.dtype
        )

        matrix_pass = (
            mismatch_count == 0
            and max_difference == 0.0
            and mean_difference == 0.0
            and shape_match
            and dtype_match
        )

        matrix_results.append(
            {
                "name": name,
                "values": values,
                "mismatches":
                    mismatch_count,
                "max_difference":
                    max_difference,
                "mean_difference":
                    mean_difference,
                "shape_match":
                    shape_match,
                "dtype_match":
                    dtype_match,
                "pass":
                    matrix_pass,
            }
        )

        print()
        print(name)
        print(
            "  values              :",
            values,
        )
        print(
            "  mismatches          :",
            mismatch_count,
        )
        print(
            "  max abs difference  :",
            max_difference,
        )
        print(
            "  mean abs difference :",
            mean_difference,
        )
        print(
            "  shape match         :",
            shape_match,
        )
        print(
            "  dtype match         :",
            dtype_match,
        )
        print(
            "  result              :",
            "PASS"
            if matrix_pass
            else "FAIL",
        )

    # ------------------------------------------------------------------
    # Actual storage
    # ------------------------------------------------------------------

    serialization_overhead_bytes = (
        disk_bytes
        - logical_payload_bytes
    )

    serialization_overhead_percent = (
        100.0
        * serialization_overhead_bytes
        / logical_payload_bytes
    )

    logical_storage_ratio = (
        logical_payload_bytes
        / original_fp16_bytes
    )

    logical_compression = (
        original_fp16_bytes
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

    scale_share = (
        100.0
        * logical_scale_bytes
        / logical_payload_bytes
    )

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    print()
    print("=" * 100)
    print("CHECKPOINT METADATA")
    print("=" * 100)

    print(
        "Format version          :",
        metadata["format_version"],
    )

    print(
        "Model name              :",
        metadata["model_name"],
    )

    print(
        "Group size              :",
        metadata["group_size"],
    )

    print(
        "Threshold multiplier    :",
        metadata[
            "threshold_multiplier"
        ],
    )

    print(
        "Weight count            :",
        metadata["weight_count"],
    )

    print(
        "Metadata exact match    :",
        checkpoint_metadata_match,
    )

    print()
    print("=" * 100)
    print("AGGREGATE STORAGE")
    print("=" * 100)

    print(
        "Matrices                :",
        len(TARGET_MODULES),
    )

    print(
        "Parameters              :",
        original_total_parameters,
    )

    print(
        "Original FP16 bytes     :",
        original_fp16_bytes,
    )

    print()
    print(
        "Packed code bytes       :",
        logical_code_bytes,
    )

    print(
        "FP16 scale bytes        :",
        logical_scale_bytes,
    )

    print(
        "Logical payload bytes   :",
        logical_payload_bytes,
    )

    print(
        "Scale share of payload  :",
        f"{scale_share:.6f}%",
    )

    print()
    print(
        "Actual checkpoint bytes :",
        disk_bytes,
    )

    print(
        "Serialization overhead  :",
        serialization_overhead_bytes,
        "bytes",
    )

    print(
        "Overhead vs payload     :",
        f"{serialization_overhead_percent:.6f}%",
    )

    print()
    print(
        "Logical storage ratio   :",
        f"{logical_storage_ratio * 100:.6f}%",
    )

    print(
        "Logical compression     :",
        f"{logical_compression:.6f}x",
    )

    print()
    print(
        "Actual disk ratio       :",
        f"{actual_disk_ratio * 100:.6f}%",
    )

    print(
        "Actual disk compression :",
        f"{actual_disk_compression:.6f}x",
    )

    print()
    print("=" * 100)
    print("AGGREGATE RECONSTRUCTION")
    print("=" * 100)

    print(
        "Total values            :",
        total_values,
    )

    print(
        "Total mismatches        :",
        total_mismatches,
    )

    print(
        "Global max difference   :",
        global_max_difference,
    )

    all_matrix_pass = all(
        item["pass"]
        for item in matrix_results
    )

    print(
        "All matrices pass       :",
        all_matrix_pass,
    )

    # ------------------------------------------------------------------
    # Strict acceptance criteria
    # ------------------------------------------------------------------

    print()
    print("=" * 100)
    print("VALIDATION")
    print("=" * 100)

    assert checkpoint_metadata_match

    assert (
        original_total_parameters
        == 8_912_896
    )

    assert (
        original_fp16_bytes
        == 17_825_792
    )

    assert (
        logical_code_bytes
        == 2_228_224
    )

    assert (
        logical_scale_bytes
        == 139_264
    )

    assert (
        logical_payload_bytes
        == 2_367_488
    )

    assert (
        total_values
        == original_total_parameters
    )

    assert total_mismatches == 0

    assert (
        global_max_difference
        == 0.0
    )

    assert all_matrix_pass

    assert (
        disk_bytes
        >= logical_payload_bytes
    )

    print(
        "Checkpoint metadata     : PASS"
    )

    print(
        "Target parameter count  : PASS"
    )

    print(
        "Logical storage         : PASS"
    )

    print(
        "Matrix reconstruction   : EXACT MATCH"
    )

    print(
        "Aggregate reconstruction: EXACT MATCH"
    )

    print(
        "Disk measurement        : PASS"
    )

    print()
    print("=" * 100)
    print("ENG003 REAL MULTI-MATRIX CHECKPOINT VALIDATION: PASS")
    print("=" * 100)

    del loaded_weights
    del references
    del model


if __name__ == "__main__":
    main()
