from pathlib import Path

import torch

from packed_ternary.checkpoint import (
    save_packed_checkpoint,
)
from packed_ternary.packed_linear import (
    PackedTernaryLinear,
)
from packed_ternary.packed_weight import (
    quantize_to_packed,
)
from packed_ternary.runtime_loader import (
    load_packed_runtime_modules,
)


CHECKPOINT_PATH = Path(
    "results/ENG006_atomicity_test.pt"
)

MODEL_NAME = "eng006-synthetic-model"

GROUP_SIZE = 128
THRESHOLD_MULTIPLIER = 0.50


class SyntheticModel(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.layer1 = torch.nn.Linear(
            16,
            16,
            bias=False,
            dtype=torch.float16,
        )

        self.layer2 = torch.nn.Linear(
            16,
            16,
            bias=False,
            dtype=torch.float16,
        )


def build_checkpoint():

    torch.manual_seed(12345)

    source = SyntheticModel()

    packed_weights = {
        "layer1": quantize_to_packed(
            source.layer1.weight,
            group_size=GROUP_SIZE,
            threshold_multiplier=(
                THRESHOLD_MULTIPLIER
            ),
        ),
        "layer2": quantize_to_packed(
            source.layer2.weight,
            group_size=GROUP_SIZE,
            threshold_multiplier=(
                THRESHOLD_MULTIPLIER
            ),
        ),
    }

    save_packed_checkpoint(
        weights=packed_weights,
        path=CHECKPOINT_PATH,
        model_name=MODEL_NAME,
        group_size=GROUP_SIZE,
        threshold_multiplier=(
            THRESHOLD_MULTIPLIER
        ),
    )


def test_valid_replacement():

    print("=" * 80)
    print("TEST 1 — VALID SINGLE-MODULE REPLACEMENT")
    print("=" * 80)

    model = SyntheticModel()

    layer1_before = model.layer1
    layer2_before = model.layer2

    result = load_packed_runtime_modules(
        model=model,
        checkpoint_path=CHECKPOINT_PATH,
        selected_module_names=[
            "layer1",
        ],
        expected_model_name=MODEL_NAME,
        rows_per_chunk=4,
    )

    assert isinstance(
        model.layer1,
        PackedTernaryLinear,
    )

    assert model.layer1 is not layer1_before

    assert model.layer2 is layer2_before

    assert (
        result.loaded_module_names
        == ("layer1",)
    )

    assert (
        result.replaced_matrix_count
        == 1
    )

    assert (
        result.replaced_parameter_count
        == 256
    )

    assert not hasattr(
        model.layer1,
        "weight",
    )

    print(
        "Replacement type        :",
        type(model.layer1).__name__,
    )

    print(
        "Loaded module names     :",
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

    print(
        "Untargeted layer intact :",
        model.layer2 is layer2_before,
    )

    print()
    print("TEST 1: PASS")
    print()


def test_atomicity():

    print("=" * 80)
    print("TEST 2 — INVALID SECOND TARGET MUST CAUSE ZERO MUTATION")
    print("=" * 80)

    model = SyntheticModel()

    layer1_before = model.layer1
    layer2_before = model.layer2

    error_seen = False

    try:
        load_packed_runtime_modules(
            model=model,
            checkpoint_path=CHECKPOINT_PATH,
            selected_module_names=[
                "layer1",
                "does_not_exist",
            ],
            expected_model_name=MODEL_NAME,
            rows_per_chunk=4,
        )

    except KeyError as exc:

        error_seen = True

        print(
            "Expected error          :",
            exc,
        )

    assert error_seen

    assert model.layer1 is layer1_before
    assert model.layer2 is layer2_before

    assert isinstance(
        model.layer1,
        torch.nn.Linear,
    )

    assert isinstance(
        model.layer2,
        torch.nn.Linear,
    )

    print(
        "Layer1 unchanged        :",
        model.layer1 is layer1_before,
    )

    print(
        "Layer2 unchanged        :",
        model.layer2 is layer2_before,
    )

    print()
    print("TEST 2: PASS")
    print()


def test_model_name_guard():

    print("=" * 80)
    print("TEST 3 — WRONG MODEL NAME MUST CAUSE ZERO MUTATION")
    print("=" * 80)

    model = SyntheticModel()

    layer1_before = model.layer1
    layer2_before = model.layer2

    error_seen = False

    try:
        load_packed_runtime_modules(
            model=model,
            checkpoint_path=CHECKPOINT_PATH,
            selected_module_names=[
                "layer1",
            ],
            expected_model_name=(
                "wrong-model-name"
            ),
            rows_per_chunk=4,
        )

    except ValueError as exc:

        error_seen = True

        print(
            "Expected error          :",
            exc,
        )

    assert error_seen

    assert model.layer1 is layer1_before
    assert model.layer2 is layer2_before

    print(
        "Layer1 unchanged        :",
        model.layer1 is layer1_before,
    )

    print(
        "Layer2 unchanged        :",
        model.layer2 is layer2_before,
    )

    print()
    print("TEST 3: PASS")
    print()


def main():

    print()
    print("=" * 80)
    print("ENG006 — PACKED RUNTIME LOADER ATOMICITY VALIDATION")
    print("=" * 80)
    print()

    build_checkpoint()

    test_valid_replacement()
    test_atomicity()
    test_model_name_guard()

    print("=" * 80)
    print("ENG006 LOADER ATOMICITY VALIDATION: PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()
