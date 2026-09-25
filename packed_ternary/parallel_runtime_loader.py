"""
Parallel compiled ternary runtime loader.

OPT003 objective
----------------
Load selected matrices from the frozen ENG003 packed checkpoint and replace
the corresponding nn.Linear modules with ParallelPackedTernaryLinear modules.

Persistent weight representation:

    packed 2-bit ternary codes + reconstruction scales

Unlike PackedTernaryLinear, ParallelPackedTernaryLinear does NOT reconstruct an FP16
weight matrix or FP16 weight-row chunk during forward execution.

It currently decodes the packed 2-bit representation into temporary int8
ternary codes and performs direct {-1, 0, +1} contribution accumulation.

This is a correctness-oriented engineering prototype.

This module does NOT:
    - quantize model weights,
    - modify the frozen scientific quantizer,
    - modify the ENG003 checkpoint format,
    - modify the frozen ENG006 runtime loader,
    - implement a fused 2-bit CUDA kernel,
    - claim inference acceleration,
    - claim peak RAM/VRAM reduction.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import torch

from packed_ternary.checkpoint import (
    load_packed_checkpoint,
)

from packed_ternary.parallel_linear import (
    ParallelPackedTernaryLinear,
)

from packed_ternary.packed_weight import (
    PackedTernaryWeight,
)


@dataclass
class ParallelRuntimeLoadResult:
    """
    Summary of one direct-runtime replacement operation.
    """

    checkpoint_path: str
    model_name: str
    loaded_module_names: Tuple[str, ...]
    replaced_matrix_count: int
    replaced_parameter_count: int
    persistent_packed_bytes: int
    fp16_equivalent_bytes: int


def _get_module_by_name(
    model: torch.nn.Module,
    module_name: str,
) -> torch.nn.Module:
    """
    Resolve an exact module name from model.named_modules().
    """

    modules = dict(
        model.named_modules()
    )

    if module_name not in modules:
        raise KeyError(
            "Checkpoint target module not found in model: "
            f"{module_name}"
        )

    return modules[module_name]


def _get_parent_and_child_name(
    model: torch.nn.Module,
    module_name: str,
):
    """
    Resolve the parent module and final child attribute name.
    """

    if not isinstance(
        module_name,
        str,
    ):
        raise TypeError(
            "module_name must be a string"
        )

    if not module_name:
        raise ValueError(
            "module_name cannot be empty"
        )

    parts = module_name.split(".")

    child_name = parts[-1]

    parent_path = ".".join(
        parts[:-1]
    )

    if not parent_path:
        return model, child_name

    parent = _get_module_by_name(
        model,
        parent_path,
    )

    return parent, child_name


def _validate_original_target(
    module_name: str,
    module: torch.nn.Module,
    packed: PackedTernaryWeight,
) -> None:
    """
    Validate that an existing module can be replaced by
    ParallelPackedTernaryLinear without modifying the model first.

    ENG010 currently requires:
        - 2-D packed weight,
        - no global padding,
        - each output row aligned to complete ternary groups.
    """

    if not isinstance(
        module,
        torch.nn.Linear,
    ):
        raise TypeError(
            f"{module_name} must be torch.nn.Linear before "
            "direct-runtime replacement; "
            f"got {type(module)}"
        )

    if len(
        packed.original_shape
    ) != 2:
        raise ValueError(
            f"{module_name}: packed weight must be 2-D"
        )

    expected_shape = (
        module.out_features,
        module.in_features,
    )

    if (
        tuple(packed.original_shape)
        != expected_shape
    ):
        raise ValueError(
            f"{module_name}: shape mismatch: "
            f"model={expected_shape} "
            f"checkpoint={tuple(packed.original_shape)}"
        )

    if (
        module.weight.numel()
        != packed.original_numel
    ):
        raise ValueError(
            f"{module_name}: parameter-count mismatch: "
            f"model={module.weight.numel()} "
            f"checkpoint={packed.original_numel}"
        )

    if (
        module.weight.dtype
        != packed.original_dtype
    ):
        raise TypeError(
            f"{module_name}: dtype mismatch: "
            f"model={module.weight.dtype} "
            f"checkpoint={packed.original_dtype}"
        )

    if (
        packed.padded_numel
        != packed.original_numel
    ):
        raise ValueError(
            f"{module_name}: OPT003 streaming runtime currently "
            "requires padded_numel == original_numel"
        )

    if (
        packed.original_numel
        % packed.group_size
        != 0
    ):
        raise ValueError(
            f"{module_name}: invalid packed group layout"
        )

    if (
        module.in_features
        % packed.group_size
        != 0
    ):
        raise ValueError(
            f"{module_name}: OPT003 streaming runtime requires "
            "each output row to align to complete groups; "
            f"in_features={module.in_features}, "
            f"group_size={packed.group_size}"
        )

    if (
        module.in_features
        % 4
        != 0
    ):
        raise ValueError(
            f"{module_name}: OPT003 streaming runtime requires "
            "each output row to align to packed-byte boundaries; "
            f"in_features={module.in_features}"
        )

    expected_groups = (
        packed.original_numel
        // packed.group_size
    )

    if (
        packed.scales.numel()
        != expected_groups
    ):
        raise ValueError(
            f"{module_name}: scale-count mismatch: "
            f"{packed.scales.numel()} != {expected_groups}"
        )


def _normalize_selected_names(
    selected_module_names: Iterable[str],
) -> Tuple[str, ...]:
    """
    Validate and normalize requested checkpoint subset.

    Order is preserved.
    Duplicate names are rejected.
    """

    if isinstance(
        selected_module_names,
        str,
    ):
        raise TypeError(
            "selected_module_names must be an iterable of names, "
            "not a single string"
        )

    try:
        names = tuple(
            selected_module_names
        )

    except TypeError as exc:
        raise TypeError(
            "selected_module_names must be iterable"
        ) from exc

    if not names:
        raise ValueError(
            "selected_module_names cannot be empty"
        )

    seen = set()

    for name in names:

        if not isinstance(
            name,
            str,
        ):
            raise TypeError(
                "Every selected module name must be a string"
            )

        if not name:
            raise ValueError(
                "Selected module names cannot be empty"
            )

        if name in seen:
            raise ValueError(
                f"Duplicate selected module name: {name}"
            )

        seen.add(name)

    return names


@torch.no_grad()
def load_parallel_runtime_modules(
    model: torch.nn.Module,
    checkpoint_path,
    selected_module_names: Iterable[str],
    expected_model_name: str = None,
) -> ParallelRuntimeLoadResult:
    """
    Replace selected nn.Linear modules with ParallelPackedTernaryLinear.

    Atomicity
    ---------
    Phase 1:
        Load checkpoint and validate every selected target.

    Phase 2A:
        Construct every ParallelPackedTernaryLinear replacement.

    Phase 2B:
        Install replacements only after every validation and constructor
        operation has succeeded.

    Therefore no model mutation occurs when validation or construction fails.
    """

    if not isinstance(
        model,
        torch.nn.Module,
    ):
        raise TypeError(
            "model must be a torch.nn.Module"
        )

    selected_names = (
        _normalize_selected_names(
            selected_module_names
        )
    )

    checkpoint_path = Path(
        checkpoint_path
    )

    metadata, packed_weights = (
        load_packed_checkpoint(
            checkpoint_path
        )
    )

    checkpoint_model_name = (
        metadata["model_name"]
    )

    if (
        expected_model_name is not None
        and checkpoint_model_name
        != expected_model_name
    ):
        raise ValueError(
            "Checkpoint model-name mismatch: "
            f"checkpoint={checkpoint_model_name!r}, "
            f"expected={expected_model_name!r}"
        )

    # ------------------------------------------------------------------
    # Phase 1
    #
    # Validate every selected checkpoint entry and model target.
    # No mutation occurs here.
    # ------------------------------------------------------------------

    validated = []

    for module_name in selected_names:

        if (
            module_name
            not in packed_weights
        ):
            raise KeyError(
                "Selected module is not present in packed checkpoint: "
                f"{module_name}"
            )

        packed = packed_weights[
            module_name
        ]

        module = _get_module_by_name(
            model,
            module_name,
        )

        _validate_original_target(
            module_name=module_name,
            module=module,
            packed=packed,
        )

        parent, child_name = (
            _get_parent_and_child_name(
                model,
                module_name,
            )
        )

        if not hasattr(
            parent,
            child_name,
        ):
            raise AttributeError(
                f"{module_name}: parent does not expose "
                f"child attribute {child_name!r}"
            )

        validated.append(
            (
                module_name,
                module,
                parent,
                child_name,
                packed,
            )
        )

    # ------------------------------------------------------------------
    # Phase 2A
    #
    # Construct every replacement before installing any of them.
    # ------------------------------------------------------------------

    prepared = []

    replaced_parameter_count = 0
    persistent_packed_bytes = 0
    fp16_equivalent_bytes = 0

    for (
        module_name,
        module,
        parent,
        child_name,
        packed,
    ) in validated:

        bias = None

        if module.bias is not None:
            bias = (
                module.bias
                .detach()
                .clone()
            )

        replacement = (
            ParallelPackedTernaryLinear(
                packed_weight=packed,
                bias=bias,
            )
        )

        # Packed checkpoint tensors are CPU-resident.
        #
        # Move persistent packed buffers and optional bias to the original
        # module device so replacement respects current model placement.
        target_device = (
            module.weight.device
        )

        replacement = (
            replacement.to(
                device=target_device
            )
        )

        prepared.append(
            (
                module_name,
                parent,
                child_name,
                replacement,
            )
        )

        replaced_parameter_count += (
            packed.original_numel
        )

        persistent_packed_bytes += (
            replacement
            .persistent_weight_bytes
        )

        fp16_equivalent_bytes += (
            replacement
            .fp16_equivalent_weight_bytes
        )

    # ------------------------------------------------------------------
    # Phase 2B
    #
    # Everything validated and constructed successfully.
    # Mutation starts only here.
    # ------------------------------------------------------------------

    loaded_module_names = []

    for (
        module_name,
        parent,
        child_name,
        replacement,
    ) in prepared:

        setattr(
            parent,
            child_name,
            replacement,
        )

        loaded_module_names.append(
            module_name
        )

    return ParallelRuntimeLoadResult(
        checkpoint_path=str(
            checkpoint_path
        ),
        model_name=(
            checkpoint_model_name
        ),
        loaded_module_names=tuple(
            loaded_module_names
        ),
        replaced_matrix_count=len(
            loaded_module_names
        ),
        replaced_parameter_count=(
            replaced_parameter_count
        ),
        persistent_packed_bytes=(
            persistent_packed_bytes
        ),
        fp16_equivalent_bytes=(
            fp16_equivalent_bytes
        ),
    )
