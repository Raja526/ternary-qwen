"""
Packed ternary runtime loader.

ENG006 objective
----------------
Load selected matrices from an ENG003 packed checkpoint and replace the
corresponding nn.Linear modules with PackedTernaryLinear modules.

Unlike ENG004, this loader does NOT reconstruct and copy a persistent
full FP16 weight into the original nn.Linear module.

Instead, the persistent weight representation becomes:

    packed 2-bit ternary codes + reconstruction scales

and PackedTernaryLinear reconstructs bounded temporary row chunks during
forward execution.

ENG006 intentionally supports selecting a subset of checkpoint entries.
This allows one real Qwen matrix to be validated before moving to
multi-matrix packed runtime integration in ENG007.

This module does NOT:
    - quantize model weights,
    - modify the frozen scientific quantizer,
    - modify the ENG003 checkpoint format,
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

from packed_ternary.packed_linear import (
    PackedTernaryLinear,
)

from packed_ternary.packed_weight import (
    PackedTernaryWeight,
)


@dataclass
class PackedRuntimeLoadResult:
    """
    Summary of one packed-runtime replacement operation.
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

    Example
    -------
    module_name:
        model.layers.3.self_attn.k_proj

    returns:
        parent = model.layers[3].self_attn
        child_name = "k_proj"
    """

    if not isinstance(module_name, str):
        raise TypeError(
            "module_name must be a string"
        )

    if not module_name:
        raise ValueError(
            "module_name cannot be empty"
        )

    parts = module_name.split(".")

    if len(parts) < 1:
        raise ValueError(
            f"Invalid module name: {module_name!r}"
        )

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
    PackedTernaryLinear without modifying the model first.
    """

    if not isinstance(
        module,
        torch.nn.Linear,
    ):
        raise TypeError(
            f"{module_name} must be torch.nn.Linear before "
            "packed-runtime replacement; "
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
        % packed.group_size
        != 0
    ):
        raise ValueError(
            f"{module_name}: invalid packed group layout"
        )

    expected_groups = (
        packed.padded_numel
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
    Validate and normalize the requested checkpoint subset.

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
def load_packed_runtime_modules(
    model: torch.nn.Module,
    checkpoint_path,
    selected_module_names: Iterable[str],
    expected_model_name: str = None,
    rows_per_chunk: int = 64,
) -> PackedRuntimeLoadResult:
    """
    Replace selected nn.Linear modules with PackedTernaryLinear.

    Parameters
    ----------
    model:
        Already-loaded PyTorch model.

    checkpoint_path:
        ENG003 multi-matrix packed checkpoint.

    selected_module_names:
        Exact checkpoint/module names to replace.

        ENG006 intentionally uses one name:

            model.layers.3.self_attn.k_proj

        ENG007 can later validate multiple selected names.

    expected_model_name:
        Optional exact model-name guard.

    rows_per_chunk:
        Maximum output rows reconstructed by PackedTernaryLinear
        at one time.

    Returns
    -------
    PackedRuntimeLoadResult

    Atomicity
    ---------
    The operation has two phases.

    Phase 1:
        Load checkpoint and validate every selected target.

    Phase 2:
        Construct all replacement modules, then install them.

    No model mutation occurs until every selected checkpoint entry and
    target module has passed validation.
    """

    if not isinstance(
        model,
        torch.nn.Module,
    ):
        raise TypeError(
            "model must be a torch.nn.Module"
        )

    if not isinstance(
        rows_per_chunk,
        int,
    ):
        raise TypeError(
            "rows_per_chunk must be an int"
        )

    if rows_per_chunk <= 0:
        raise ValueError(
            "rows_per_chunk must be greater than zero"
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

    # --------------------------------------------------------------
    # Phase 1:
    # Validate every selected checkpoint entry and target module.
    #
    # Nothing in the model is changed during this phase.
    # --------------------------------------------------------------

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

    # --------------------------------------------------------------
    # Phase 2A:
    # Construct all replacements before installing any of them.
    #
    # This keeps constructor failures from partially mutating model.
    # --------------------------------------------------------------

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
            PackedTernaryLinear(
                packed_weight=packed,
                bias=bias,
                rows_per_chunk=(
                    rows_per_chunk
                ),
            )
        )

        # Packed checkpoint tensors are CPU-resident.
        #
        # Move persistent packed buffers and optional bias to the
        # original module's device so replacement respects the
        # existing model placement.
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

    # --------------------------------------------------------------
    # Phase 2B:
    # All validation and construction succeeded.
    # Install replacements.
    # --------------------------------------------------------------

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

    return PackedRuntimeLoadResult(
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
