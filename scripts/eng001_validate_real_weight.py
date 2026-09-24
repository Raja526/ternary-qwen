import torch
from transformers import AutoModelForCausalLM

from ternary.quantizer import groupwise_ternarize_weight
from packed_ternary.packed_weight import (
    quantize_to_packed,
    reconstruct_packed,
)
from packed_ternary.packing import unpack_ternary_codes


MODEL_NAME = "Qwen/Qwen3.5-0.8B"

TARGET_MODULE = "model.layers.3.self_attn.k_proj"

GROUP_SIZE = 128
THRESHOLD_MULTIPLIER = 0.50


def main():

    print("=" * 88)
    print("ENG001 — REAL QWEN PACKED TERNARY VALIDATION")
    print("=" * 88)

    print()
    print("Model :", MODEL_NAME)
    print("Target:", TARGET_MODULE)

    # ------------------------------------------------------------------
    # Load frozen FP16 model
    # ------------------------------------------------------------------

    print()
    print("Loading FP16 model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    modules = dict(model.named_modules())

    if TARGET_MODULE not in modules:
        raise RuntimeError(
            f"Target module not found: {TARGET_MODULE}"
        )

    module = modules[TARGET_MODULE]

    if not isinstance(module, torch.nn.Linear):
        raise TypeError(
            f"{TARGET_MODULE} is not torch.nn.Linear"
        )

    weight = module.weight.detach()

    print()
    print("Weight shape             :", tuple(weight.shape))
    print("Weight dtype             :", weight.dtype)
    print("Weight parameters        :", weight.numel())

    # ------------------------------------------------------------------
    # Frozen scientific reference
    # ------------------------------------------------------------------

    print()
    print("Running frozen scientific quantizer...")

    reference = groupwise_ternarize_weight(
        weight,
        method="optimized",
        group_size=GROUP_SIZE,
        threshold_multiplier=THRESHOLD_MULTIPLIER,
    )

    reference_weight = reference["reconstructed"]

    # ------------------------------------------------------------------
    # Engineering packed representation
    # ------------------------------------------------------------------

    print("Creating packed ternary representation...")

    packed = quantize_to_packed(
        weight,
        group_size=GROUP_SIZE,
        threshold_multiplier=THRESHOLD_MULTIPLIER,
        scale_dtype=torch.float16,
    )

    print("Reconstructing packed weight...")

    packed_reconstructed = reconstruct_packed(
        packed
    )

    # ------------------------------------------------------------------
    # Reconstruction comparison
    # ------------------------------------------------------------------

    mismatch_mask = (
        reference_weight
        != packed_reconstructed
    )

    mismatch_count = int(
        mismatch_mask.sum().item()
    )

    total_values = weight.numel()

    exact_count = (
        total_values - mismatch_count
    )

    exact_percent = (
        100.0
        * exact_count
        / total_values
    )

    difference = (
        reference_weight.float()
        - packed_reconstructed.float()
    ).abs()

    max_abs_difference = float(
        difference.max().item()
    )

    mean_abs_difference = float(
        difference.mean().item()
    )

    # ------------------------------------------------------------------
    # Decode packed ternary codes for statistics
    # ------------------------------------------------------------------

    unpacked_codes = unpack_ternary_codes(
        packed.packed_codes,
        packed.padded_numel,
    )

    # Ignore group padding when reporting weight-code distribution.
    valid_codes = unpacked_codes[
        :packed.original_numel
    ]

    negative_count = int(
        (valid_codes == -1).sum().item()
    )

    zero_count = int(
        (valid_codes == 0).sum().item()
    )

    positive_count = int(
        (valid_codes == 1).sum().item()
    )

    negative_percent = (
        100.0 * negative_count / total_values
    )

    zero_percent = (
        100.0 * zero_count / total_values
    )

    positive_percent = (
        100.0 * positive_count / total_values
    )

    # ------------------------------------------------------------------
    # Compare code statistics against frozen quantizer
    # ------------------------------------------------------------------

    stats_match = (
        negative_count == reference["negative"]
        and zero_count == reference["zero"]
        and positive_count == reference["positive"]
    )

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    original_fp16_bytes = (
        total_values * 2
    )

    packed_code_bytes = (
        packed.code_bytes
    )

    scale_bytes = (
        packed.scale_bytes
    )

    packed_payload_bytes = (
        packed.packed_storage_bytes
    )

    storage_ratio = (
        packed_payload_bytes
        / original_fp16_bytes
    )

    compression_factor = (
        original_fp16_bytes
        / packed_payload_bytes
    )

    scale_overhead_percent_of_packed = (
        100.0
        * scale_bytes
        / packed_payload_bytes
    )

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    print()
    print("-" * 88)
    print("QUANTIZATION")
    print("-" * 88)

    print(
        "Group size               :",
        packed.group_size,
    )

    print(
        "Groups                   :",
        packed.num_groups,
    )

    print(
        "Original numel           :",
        packed.original_numel,
    )

    print(
        "Padded numel             :",
        packed.padded_numel,
    )

    print(
        "Padding values           :",
        packed.padded_numel
        - packed.original_numel,
    )

    print()
    print(
        "Negative codes           :",
        negative_count,
        f"({negative_percent:.4f}%)",
    )

    print(
        "Zero codes               :",
        zero_count,
        f"({zero_percent:.4f}%)",
    )

    print(
        "Positive codes           :",
        positive_count,
        f"({positive_percent:.4f}%)",
    )

    print()
    print(
        "Frozen code stats match  :",
        stats_match,
    )

    print(
        "Scientific alpha mean    :",
        reference[
            "mean_reconstruction_alpha"
        ],
    )

    print(
        "Stored FP16 alpha mean   :",
        float(
            packed.scales
            .float()
            .mean()
            .item()
        ),
    )

    print()
    print("-" * 88)
    print("RECONSTRUCTION")
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
    print("PACKED PAYLOAD STORAGE")
    print("-" * 88)

    print(
        "Original FP16 bytes      :",
        original_fp16_bytes,
    )

    print(
        "Packed code bytes        :",
        packed_code_bytes,
    )

    print(
        "FP16 scale bytes         :",
        scale_bytes,
    )

    print(
        "Packed payload bytes     :",
        packed_payload_bytes,
    )

    print(
        "Storage ratio            :",
        f"{storage_ratio * 100:.6f}%",
    )

    print(
        "Payload compression      :",
        f"{compression_factor:.6f}x",
    )

    print(
        "Scale share of payload   :",
        f"{scale_overhead_percent_of_packed:.6f}%",
    )

    # ------------------------------------------------------------------
    # Strict validation
    # ------------------------------------------------------------------

    print()
    print("-" * 88)
    print("VALIDATION")
    print("-" * 88)

    assert stats_match, (
        "Packed ternary code distribution does not "
        "match frozen quantizer."
    )

    assert (
        mismatch_count == 0
    ), (
        "Packed reconstruction does not exactly "
        "match frozen FP16 reconstruction."
    )

    assert (
        max_abs_difference == 0.0
    )

    assert (
        mean_abs_difference == 0.0
    )

    assert (
        packed_reconstructed.dtype
        == reference_weight.dtype
    )

    assert (
        packed_reconstructed.shape
        == reference_weight.shape
    )

    print(
        "Ternary code statistics  : PASS"
    )

    print(
        "FP16 reconstruction      : EXACT MATCH"
    )

    print(
        "Shape                    : PASS"
    )

    print(
        "Dtype                    : PASS"
    )

    print()
    print("=" * 88)
    print("ENG001 REAL-WEIGHT VALIDATION: PASS")
    print("=" * 88)

    del reference_weight
    del packed_reconstructed
    del unpacked_codes
    del valid_codes
    del packed
    del reference
    del model


if __name__ == "__main__":
    main()
