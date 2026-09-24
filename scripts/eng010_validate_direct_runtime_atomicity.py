"""
ENG010 — Direct runtime loader atomicity validation.

Objective
---------
Validate that load_direct_runtime_modules():

1. Correctly replaces valid selected nn.Linear modules with
   DirectTernaryLinear.
2. Preserves packed codes + scales as the persistent representation.
3. Does not expose a persistent full FP16 weight Parameter.
4. Does not partially mutate the model if a later selected target fails.
5. Does not mutate the model when the expected model-name guard fails.

This is a synthetic loader/integration test.

It does NOT claim:
    - inference speedup,
    - peak RAM/VRAM reduction,
    - fused 2-bit execution.
"""

from pathlib import Path

import torch

from packed_ternary.checkpoint import (
    save_packed_checkpoint,
)

from packed_ternary.direct_linear import (
    DirectTernaryLinear,
)

from packed_ternary.direct_runtime_loader import (
    load_direct_runtime_modules,
)

from packed_ternary.packed_weight import (
    quantize_to_packed,
)


SEED = 20260924

GROUP_SIZE = 128

MODEL_NAME = "ENG010_SYNTHETIC_MODEL"

CHECKPOINT_PATH = Path(
    "results/ENG010_direct_atomicity_test.pt"
)


class SyntheticModel(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.first = torch.nn.Linear(
            256,
            8,
            bias=False,
            dtype=torch.float16,
        )

        self.second = torch.nn.Linear(
            256,
            8,
            bias=False,
            dtype=torch.float16,
        )

        self.control = torch.nn.Linear(
            256,
            8,
            bias=False,
            dtype=torch.float16,
        )


def create_checkpoint():
    torch.manual_seed(SEED)

    first_weight = torch.randn(
        8,
        256,
        dtype=torch.float16,
    )

    second_weight = torch.randn(
        8,
        256,
        dtype=torch.float16,
    )

    packed_first = quantize_to_packed(
        first_weight,
        group_size=GROUP_SIZE,
    )

    packed_second = quantize_to_packed(
        second_weight,
        group_size=GROUP_SIZE,
    )

    save_packed_checkpoint(
        weights={
            "first": packed_first,
            "second": packed_second,
        },
        path=CHECKPOINT_PATH,
        model_name=MODEL_NAME,
        group_size=GROUP_SIZE,
        threshold_multiplier=0.5,
    )

    return (
        packed_first,
        packed_second,
    )


def test_valid_single_replacement(
    packed_first,
):
    print()
    print("=" * 100)
    print("TEST 1 — VALID SINGLE REPLACEMENT")
    print("=" * 100)

    model = SyntheticModel()

    first_before = model.first
    second_before = model.second
    control_before = model.control

    result = load_direct_runtime_modules(
        model=model,
        checkpoint_path=CHECKPOINT_PATH,
        selected_module_names=[
            "first",
        ],
        expected_model_name=MODEL_NAME,
        rows_per_chunk=4,
    )

    print(
        "First type after        :",
        type(model.first).__name__,
    )

    print(
        "Second type after       :",
        type(model.second).__name__,
    )

    print(
        "Control type after      :",
        type(model.control).__name__,
    )

    print(
        "First replaced          :",
        model.first is not first_before,
    )

    print(
        "Second unchanged        :",
        model.second is second_before,
    )

    print(
        "Control unchanged       :",
        model.control is control_before,
    )

    print(
        "Loaded names            :",
        result.loaded_module_names,
    )

    print(
        "Replaced matrices       :",
        result.replaced_matrix_count,
    )

    print(
        "Replaced parameters     :",
        result.replaced_parameter_count,
    )

    print(
        "Persistent packed bytes :",
        result.persistent_packed_bytes,
    )

    print(
        "FP16 equivalent bytes   :",
        result.fp16_equivalent_bytes,
    )

    parameters = dict(
        model.first.named_parameters()
    )

    buffers = dict(
        model.first.named_buffers()
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

    assert isinstance(
        model.first,
        DirectTernaryLinear,
    )

    assert (
        model.first
        is not first_before
    )

    assert (
        model.second
        is second_before
    )

    assert (
        model.control
        is control_before
    )

    assert (
        result.loaded_module_names
        == ("first",)
    )

    assert (
        result.replaced_matrix_count
        == 1
    )

    assert (
        result.replaced_parameter_count
        == packed_first.original_numel
    )

    assert (
        result.persistent_packed_bytes
        == packed_first.packed_storage_bytes
    )

    assert (
        result.fp16_equivalent_bytes
        == packed_first.original_fp16_bytes
    )

    assert "weight" not in parameters
    assert "packed_codes" in buffers
    assert "scales" in buffers

    print()
    print(
        "TEST 1: PASS"
    )


def test_invalid_second_target_atomicity():
    print()
    print("=" * 100)
    print("TEST 2 — INVALID SECOND TARGET ATOMICITY")
    print("=" * 100)

    model = SyntheticModel()

    first_before = model.first
    second_before = model.second
    control_before = model.control

    # Deliberately make the second target incompatible with
    # the packed checkpoint while leaving the first valid.
    model.second = torch.nn.Linear(
        128,
        8,
        bias=False,
        dtype=torch.float16,
    )

    invalid_second_before = (
        model.second
    )

    error = None

    try:
        load_direct_runtime_modules(
            model=model,
            checkpoint_path=CHECKPOINT_PATH,
            selected_module_names=[
                "first",
                "second",
            ],
            expected_model_name=MODEL_NAME,
            rows_per_chunk=4,
        )

    except Exception as exc:
        error = exc

    print(
        "Expected error          :",
        type(error).__name__
        if error is not None
        else None,
    )

    print(
        "Error message           :",
        str(error)
        if error is not None
        else None,
    )

    print(
        "First unchanged         :",
        model.first is first_before,
    )

    print(
        "Invalid second unchanged:",
        model.second
        is invalid_second_before,
    )

    print(
        "Control unchanged       :",
        model.control
        is control_before,
    )

    assert error is not None

    assert (
        model.first
        is first_before
    )

    assert (
        model.second
        is invalid_second_before
    )

    assert (
        model.control
        is control_before
    )

    assert isinstance(
        model.first,
        torch.nn.Linear,
    )

    assert isinstance(
        model.second,
        torch.nn.Linear,
    )

    print()
    print(
        "TEST 2: PASS"
    )


def test_model_name_guard_atomicity():
    print()
    print("=" * 100)
    print("TEST 3 — MODEL-NAME GUARD ATOMICITY")
    print("=" * 100)

    model = SyntheticModel()

    first_before = model.first
    second_before = model.second
    control_before = model.control

    error = None

    try:
        load_direct_runtime_modules(
            model=model,
            checkpoint_path=CHECKPOINT_PATH,
            selected_module_names=[
                "first",
            ],
            expected_model_name=(
                "WRONG_MODEL_NAME"
            ),
            rows_per_chunk=4,
        )

    except Exception as exc:
        error = exc

    print(
        "Expected error          :",
        type(error).__name__
        if error is not None
        else None,
    )

    print(
        "Error message           :",
        str(error)
        if error is not None
        else None,
    )

    print(
        "First unchanged         :",
        model.first is first_before,
    )

    print(
        "Second unchanged        :",
        model.second is second_before,
    )

    print(
        "Control unchanged       :",
        model.control is control_before,
    )

    assert error is not None

    assert (
        model.first
        is first_before
    )

    assert (
        model.second
        is second_before
    )

    assert (
        model.control
        is control_before
    )

    print()
    print(
        "TEST 3: PASS"
    )


def main():
    print("=" * 100)
    print(
        "ENG010 — DIRECT RUNTIME LOADER ATOMICITY VALIDATION"
    )
    print("=" * 100)

    packed_first, packed_second = (
        create_checkpoint()
    )

    print(
        "Checkpoint              :",
        CHECKPOINT_PATH,
    )

    print(
        "First parameters        :",
        packed_first.original_numel,
    )

    print(
        "Second parameters       :",
        packed_second.original_numel,
    )

    test_valid_single_replacement(
        packed_first
    )

    test_invalid_second_target_atomicity()

    test_model_name_guard_atomicity()

    print()
    print("=" * 100)
    print("ENG010 ATOMICITY SUMMARY")
    print("=" * 100)

    print(
        "Valid replacement       : PASS"
    )

    print(
        "Invalid-target atomicity:",
        "PASS"
    )

    print(
        "Model-name atomicity    : PASS"
    )

    print()
    print(
        "ENG010 DIRECT RUNTIME LOADER ATOMICITY: PASS"
    )


if __name__ == "__main__":
    main()
