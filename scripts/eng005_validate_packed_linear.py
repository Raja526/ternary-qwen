"""
ENG005 — Packed Ternary Linear Operator Validation

Objective
---------
Validate a correctness-first linear operator whose persistent weight
representation is:

    packed 2-bit ternary codes + FP16 group scales

rather than a full reconstructed FP16 weight Parameter.

This experiment validates:

1. Synthetic packed-linear correctness across:
   - multiple matrix shapes,
   - padded and non-padded packed weights,
   - different row chunk sizes,
   - 2-D and 3-D activations,
   - bias and no-bias cases.

2. Real Qwen k_proj correctness:
   - model.layers.3.self_attn.k_proj
   - multiple activation shapes,
   - multiple row chunk sizes.

3. Full-model integration:
   - Path A uses an ordinary nn.Linear containing the reconstructed
     ternary FP16 weight.
   - Path B replaces k_proj with PackedTernaryLinear.
   - Final logits are compared exactly.

Important scope
---------------
This is a correctness/reference implementation.

It does NOT establish:
    - inference speedup,
    - peak RAM reduction,
    - peak VRAM reduction,
    - a fused 2-bit CUDA kernel.

Temporary row chunks are reconstructed during forward execution.
"""

import gc

import torch
import torch.nn.functional as F

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from packed_ternary.packed_weight import (
    quantize_to_packed,
    reconstruct_packed,
)

from packed_ternary.packed_linear import (
    PackedTernaryLinear,
)


MODEL_NAME = "Qwen/Qwen3.5-0.8B"

TARGET = (
    "model.layers.3.self_attn.k_proj"
)

GROUP_SIZE = 128

THRESHOLD_MULTIPLIER = 0.5

REAL_ROWS_PER_CHUNK = 64

PROMPT = (
    "A box contains 12 red balls and 8 blue balls. "
    "How many balls are in the box?"
)

SEED = 20260924


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


def get_target(model):
    return (
        model
        .model
        .layers[3]
        .self_attn
        .k_proj
    )


def section(title):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def synthetic_validation():
    section(
        "PART 1 — SYNTHETIC PACKED LINEAR VALIDATION"
    )

    tests = [
        # out, in, input prefix, chunk, bias
        (16, 16, (4,), 1, False),
        (16, 16, (4,), 4, True),
        (16, 16, (4,), 16, False),

        (31, 17, (5,), 1, False),
        (31, 17, (5,), 7, True),
        (31, 17, (2, 3), 8, False),

        (33, 129, (3,), 5, True),
        (33, 129, (2, 2), 16, False),

        (257, 513, (2,), 32, False),
        (257, 513, (2, 3), 64, True),
    ]

    total_values = 0
    total_mismatches = 0
    global_max_difference = 0.0

    for index, (
        out_features,
        in_features,
        prefix,
        rows_per_chunk,
        use_bias,
    ) in enumerate(
        tests,
        start=1,
    ):

        weight = torch.randn(
            out_features,
            in_features,
            dtype=torch.float16,
        )

        bias = None

        if use_bias:
            bias = torch.randn(
                out_features,
                dtype=torch.float16,
            )

        packed = quantize_to_packed(
            weight,
            group_size=GROUP_SIZE,
            threshold_multiplier=(
                THRESHOLD_MULTIPLIER
            ),
            scale_dtype=torch.float16,
        )

        reconstructed = (
            reconstruct_packed(
                packed
            )
        )

        input_shape = (
            *prefix,
            in_features,
        )

        x = torch.randn(
            *input_shape,
            dtype=torch.float16,
        )

        reference = F.linear(
            x,
            reconstructed,
            bias,
        )

        layer = PackedTernaryLinear(
            packed_weight=packed,
            bias=bias,
            rows_per_chunk=(
                rows_per_chunk
            ),
        )

        output = layer(x)

        difference = (
            reference.float()
            - output.float()
        ).abs()

        mismatches = int(
            (
                reference
                != output
            )
            .sum()
            .item()
        )

        max_difference = float(
            difference.max().item()
        )

        mean_difference = float(
            difference.mean().item()
        )

        parameters = dict(
            layer.named_parameters()
        )

        buffers = dict(
            layer.named_buffers()
        )

        full_weight_present = (
            "weight"
            in parameters
        )

        codes_present = (
            "packed_codes"
            in buffers
        )

        scales_present = (
            "scales"
            in buffers
        )

        expected_payload = (
            packed.code_bytes
            + packed.scale_bytes
        )

        total_values += (
            output.numel()
        )

        total_mismatches += (
            mismatches
        )

        global_max_difference = max(
            global_max_difference,
            max_difference,
        )

        print()
        print(
            f"Test {index:02d}: "
            f"W=({out_features},{in_features}) "
            f"input={input_shape} "
            f"chunk={rows_per_chunk} "
            f"bias={use_bias}"
        )

        print(
            "  padding               :",
            packed.padded_numel
            - packed.original_numel,
        )

        print(
            "  output mismatches     :",
            mismatches,
        )

        print(
            "  max abs difference    :",
            max_difference,
        )

        print(
            "  mean abs difference   :",
            mean_difference,
        )

        print(
            "  full weight Parameter :",
            full_weight_present,
        )

        print(
            "  packed codes buffer   :",
            codes_present,
        )

        print(
            "  scales buffer         :",
            scales_present,
        )

        print(
            "  persistent bytes      :",
            layer.persistent_weight_bytes,
        )

        print(
            "  FP16 equivalent bytes :",
            layer.fp16_equivalent_weight_bytes,
        )

        print(
            "  compression factor    :",
            f"{layer.persistent_compression_factor:.6f}x",
        )

        assert (
            output.shape
            == reference.shape
        )

        assert (
            output.dtype
            == reference.dtype
        )

        assert mismatches == 0
        assert max_difference == 0.0

        assert not full_weight_present
        assert codes_present
        assert scales_present

        assert (
            layer.persistent_weight_bytes
            == expected_payload
        )

    print()
    print(
        "Synthetic tests         :",
        len(tests),
    )

    print(
        "Synthetic output values :",
        total_values,
    )

    print(
        "Synthetic mismatches    :",
        total_mismatches,
    )

    print(
        "Synthetic max diff      :",
        global_max_difference,
    )

    assert total_mismatches == 0
    assert global_max_difference == 0.0

    print(
        "Synthetic validation    : PASS"
    )


def real_matrix_validation(
    weight,
    bias,
    packed,
):
    section(
        "PART 2 — REAL QWEN K_PROJ OPERATOR VALIDATION"
    )

    reconstructed = (
        reconstruct_packed(
            packed
        )
    )

    inputs = [
        torch.randn(
            4,
            weight.shape[1],
            dtype=torch.float16,
        ),

        torch.randn(
            2,
            7,
            weight.shape[1],
            dtype=torch.float16,
        ),
    ]

    chunk_sizes = (
        1,
        8,
        32,
        64,
        128,
        512,
    )

    total_values = 0
    total_mismatches = 0
    global_max_difference = 0.0

    print(
        "Target                  :",
        TARGET,
    )

    print(
        "Weight shape            :",
        tuple(weight.shape),
    )

    print(
        "Parameters              :",
        weight.numel(),
    )

    print(
        "Packed payload bytes    :",
        packed.packed_storage_bytes,
    )

    print(
        "FP16 weight bytes       :",
        packed.original_fp16_bytes,
    )

    print(
        "Compression factor      :",
        f"{packed.compression_factor:.6f}x",
    )

    for input_index, x in enumerate(
        inputs,
        start=1,
    ):

        reference = F.linear(
            x,
            reconstructed,
            bias,
        )

        print()
        print(
            f"Input {input_index}: "
            f"shape={tuple(x.shape)}"
        )

        for rows_per_chunk in (
            chunk_sizes
        ):

            layer = (
                PackedTernaryLinear(
                    packed_weight=packed,
                    bias=bias,
                    rows_per_chunk=(
                        rows_per_chunk
                    ),
                )
            )

            output = layer(x)

            difference = (
                output.float()
                - reference.float()
            ).abs()

            mismatches = int(
                (
                    output
                    != reference
                )
                .sum()
                .item()
            )

            max_difference = float(
                difference.max().item()
            )

            mean_difference = float(
                difference.mean().item()
            )

            total_values += (
                output.numel()
            )

            total_mismatches += (
                mismatches
            )

            global_max_difference = max(
                global_max_difference,
                max_difference,
            )

            print(
                f"  chunk="
                f"{rows_per_chunk:<3} "
                f"mismatches="
                f"{mismatches:<6} "
                f"max_diff="
                f"{max_difference:.8f} "
                f"mean_diff="
                f"{mean_difference:.8f}"
            )

            assert (
                output.shape
                == reference.shape
            )

            assert (
                output.dtype
                == reference.dtype
            )

            assert mismatches == 0
            assert max_difference == 0.0

    print()
    print(
        "Forward configurations  :",
        len(inputs)
        * len(chunk_sizes),
    )

    print(
        "Compared output values  :",
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

    assert total_mismatches == 0
    assert global_max_difference == 0.0

    print(
        "Real operator validation:",
        "PASS",
    )


def full_model_validation(
    tokenizer,
    original_weight,
    original_bias,
    packed,
):
    section(
        "PART 3 — FULL QWEN IN-MODEL VALIDATION"
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

    print(
        "Input tokens            :",
        input_ids.shape[1],
    )

    # --------------------------------------------------------------
    # Path A
    # --------------------------------------------------------------

    print()
    print(
        "Path A: ordinary Linear "
        "with reconstructed ternary FP16 weight"
    )

    reference_model = (
        load_model()
    )

    reference_target = (
        get_target(
            reference_model
        )
    )

    reconstructed_weight = (
        reconstruct_packed(
            packed
        )
    )

    with torch.no_grad():
        reference_target.weight.copy_(
            reconstructed_weight
        )

    with torch.inference_mode():

        reference_logits = (
            reference_model(
                input_ids=input_ids,
                attention_mask=(
                    attention_mask
                ),
                use_cache=False,
            )
            .logits
            .detach()
            .cpu()
            .clone()
        )

    print(
        "Reference logits shape  :",
        tuple(
            reference_logits.shape
        ),
    )

    del reference_target
    del reference_model

    gc.collect()

    # --------------------------------------------------------------
    # Path B
    # --------------------------------------------------------------

    print()
    print(
        "Path B: PackedTernaryLinear"
    )

    packed_model = (
        load_model()
    )

    fresh_target = (
        get_target(
            packed_model
        )
    )

    fresh_mismatches = int(
        (
            fresh_target
            .weight
            .detach()
            .cpu()
            != original_weight
        )
        .sum()
        .item()
    )

    print(
        "Fresh weight mismatches :",
        fresh_mismatches,
    )

    assert fresh_mismatches == 0

    packed_layer = (
        PackedTernaryLinear(
            packed_weight=packed,
            bias=original_bias,
            rows_per_chunk=(
                REAL_ROWS_PER_CHUNK
            ),
        )
    )

    packed_model.model.layers[
        3
    ].self_attn.k_proj = (
        packed_layer
    )

    target_after = (
        get_target(
            packed_model
        )
    )

    parameters = dict(
        target_after.named_parameters()
    )

    buffers = dict(
        target_after.named_buffers()
    )

    print(
        "Replacement type        :",
        type(target_after),
    )

    print(
        "Full weight Parameter   :",
        "weight" in parameters,
    )

    print(
        "Packed codes buffer     :",
        "packed_codes" in buffers,
    )

    print(
        "Scales buffer           :",
        "scales" in buffers,
    )

    print(
        "Persistent weight bytes :",
        target_after
        .persistent_weight_bytes,
    )

    print(
        "FP16 equivalent bytes   :",
        target_after
        .fp16_equivalent_weight_bytes,
    )

    print(
        "Persistent compression  :",
        f"{target_after.persistent_compression_factor:.6f}x",
    )

    assert isinstance(
        target_after,
        PackedTernaryLinear,
    )

    assert (
        "weight"
        not in parameters
    )

    assert (
        "packed_codes"
        in buffers
    )

    assert (
        "scales"
        in buffers
    )

    with torch.inference_mode():

        packed_logits = (
            packed_model(
                input_ids=input_ids,
                attention_mask=(
                    attention_mask
                ),
                use_cache=False,
            )
            .logits
            .detach()
            .cpu()
        )

    difference = (
        packed_logits.float()
        - reference_logits.float()
    ).abs()

    mismatches = int(
        (
            packed_logits
            != reference_logits
        )
        .sum()
        .item()
    )

    max_difference = float(
        difference.max().item()
    )

    mean_difference = float(
        difference.mean().item()
    )

    print()
    print(
        "Compared logits         :",
        reference_logits.numel(),
    )

    print(
        "Logit mismatches        :",
        mismatches,
    )

    print(
        "Max abs difference      :",
        max_difference,
    )

    print(
        "Mean abs difference     :",
        mean_difference,
    )

    print(
        "Dtype match             :",
        packed_logits.dtype
        == reference_logits.dtype,
    )

    assert (
        packed_logits.shape
        == reference_logits.shape
    )

    assert (
        packed_logits.dtype
        == reference_logits.dtype
    )

    assert mismatches == 0
    assert max_difference == 0.0

    print()
    print(
        "Full-model validation   : PASS"
    )


def main():
    torch.manual_seed(
        SEED
    )

    section(
        "ENG005 — PACKED TERNARY LINEAR OPERATOR"
    )

    print(
        "Model                   :",
        MODEL_NAME,
    )

    print(
        "Target                  :",
        TARGET,
    )

    print(
        "Group size              :",
        GROUP_SIZE,
    )

    print(
        "Threshold multiplier    :",
        THRESHOLD_MULTIPLIER,
    )

    print(
        "Real rows/chunk         :",
        REAL_ROWS_PER_CHUNK,
    )

    # --------------------------------------------------------------
    # Part 1
    # --------------------------------------------------------------

    synthetic_validation()

    # Reset seed so later validation does not depend on the number of
    # random values consumed by the synthetic cases.
    torch.manual_seed(
        SEED
    )

    # --------------------------------------------------------------
    # Load one model to obtain the exact original real target.
    # --------------------------------------------------------------

    section(
        "PREPARING REAL QWEN TARGET"
    )

    model = load_model()

    target = get_target(
        model
    )

    assert isinstance(
        target,
        torch.nn.Linear,
    )

    original_weight = (
        target
        .weight
        .detach()
        .cpu()
        .clone()
    )

    original_bias = None

    if target.bias is not None:
        original_bias = (
            target
            .bias
            .detach()
            .cpu()
            .clone()
        )

    print(
        "Weight shape            :",
        tuple(
            original_weight.shape
        ),
    )

    print(
        "Weight dtype            :",
        original_weight.dtype,
    )

    print(
        "Parameters              :",
        original_weight.numel(),
    )

    print(
        "Bias present            :",
        original_bias is not None,
    )

    packed = quantize_to_packed(
        original_weight,
        group_size=GROUP_SIZE,
        threshold_multiplier=(
            THRESHOLD_MULTIPLIER
        ),
        scale_dtype=torch.float16,
    )

    print(
        "Packed payload bytes    :",
        packed.packed_storage_bytes,
    )

    print(
        "FP16 weight bytes       :",
        packed.original_fp16_bytes,
    )

    print(
        "Compression factor      :",
        f"{packed.compression_factor:.6f}x",
    )

    del target
    del model

    gc.collect()

    # --------------------------------------------------------------
    # Part 2
    # --------------------------------------------------------------

    real_matrix_validation(
        weight=original_weight,
        bias=original_bias,
        packed=packed,
    )

    # --------------------------------------------------------------
    # Part 3
    # --------------------------------------------------------------

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            MODEL_NAME
        )
    )

    full_model_validation(
        tokenizer=tokenizer,
        original_weight=(
            original_weight
        ),
        original_bias=(
            original_bias
        ),
        packed=packed,
    )

    # --------------------------------------------------------------
    # Final summary
    # --------------------------------------------------------------

    section(
        "ENG005 FINAL SUMMARY"
    )

    print(
        "Real target             :",
        TARGET,
    )

    print(
        "Target parameters       :",
        packed.original_numel,
    )

    print(
        "Persistent packed bytes :",
        packed.packed_storage_bytes,
    )

    print(
        "FP16 equivalent bytes   :",
        packed.original_fp16_bytes,
    )

    print(
        "Compression factor      :",
        f"{packed.compression_factor:.6f}x",
    )

    print(
        "Persistent full weight  :",
        False,
    )

    print(
        "Synthetic validation    : PASS"
    )

    print(
        "Real operator validation: PASS"
    )

    print(
        "Full-model validation   : PASS"
    )

    print()
    print(
        "Scope:"
    )

    print(
        "  Packed codes + scales are the "
        "persistent weight representation."
    )

    print(
        "  Temporary row chunks are reconstructed "
        "during forward execution."
    )

    print(
        "  No inference speedup is claimed."
    )

    print(
        "  No peak RAM/VRAM reduction is claimed."
    )

    print(
        "  No fused 2-bit CUDA kernel is claimed."
    )

    print()
    print("=" * 100)
    print(
        "ENG005 PACKED TERNARY LINEAR VALIDATION: PASS"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()
