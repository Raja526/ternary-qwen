import torch

from packed_ternary.compiled_runtime_loader import (
    load_compiled_runtime_modules,
)


CHECKPOINT_PATH = "results/ENG003_multimatrix_checkpoint.pt"

VALID_TARGET = "model.layers.3.self_attn.k_proj"

INVALID_TARGET = "model.layers.999.invalid_module"


class TinyModel(torch.nn.Module):
    pass


def main():
    print("=" * 72)
    print("OPT002 COMPILED RUNTIME ATOMICITY VALIDATION")
    print("=" * 72)

    # We use the real Qwen model because checkpoint target names must match
    # the actual model hierarchy.
    from transformers import AutoModelForCausalLM

    print("Loading model...")

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-0.8B",
        torch_dtype=torch.float16,
    )

    modules_before = dict(model.named_modules())

    original_valid_module = modules_before[
        VALID_TARGET
    ]

    print(
        "Original valid module type:",
        type(original_valid_module).__name__,
    )

    failed_as_expected = False

    try:
        load_compiled_runtime_modules(
            model=model,
            checkpoint_path=CHECKPOINT_PATH,
            selected_module_names=[
                VALID_TARGET,
                INVALID_TARGET,
            ],
            expected_model_name="Qwen/Qwen3.5-0.8B",
        )

    except Exception as exc:
        failed_as_expected = True

        print(
            "Expected failure:",
            type(exc).__name__,
        )

        print(
            "Failure message:",
            str(exc),
        )

    if not failed_as_expected:
        raise RuntimeError(
            "Loader unexpectedly accepted invalid target"
        )

    modules_after = dict(model.named_modules())

    valid_module_after = modules_after[
        VALID_TARGET
    ]

    same_object = (
        valid_module_after
        is original_valid_module
    )

    still_linear = isinstance(
        valid_module_after,
        torch.nn.Linear,
    )

    print()
    print(
        "Valid target same object :",
        same_object,
    )

    print(
        "Valid target still Linear:",
        still_linear,
    )

    if not same_object:
        raise RuntimeError(
            "Atomicity failure: valid target was replaced"
        )

    if not still_linear:
        raise RuntimeError(
            "Atomicity failure: valid target type changed"
        )

    print()
    print(
        "OPT002 COMPILED RUNTIME ATOMICITY: PASS"
    )


if __name__ == "__main__":
    main()
