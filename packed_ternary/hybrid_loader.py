"""
Hybrid packed-ternary model loader.

ENG004 objective
----------------
Load selected packed ternary matrices from an ENG003 checkpoint,
reconstruct them to their original dtype, and inject them into the
corresponding modules of an already-loaded PyTorch model.

This module does NOT:
- quantize weights
- modify the frozen scientific quantizer
- modify the packed representation
- modify checkpoint serialization
- implement packed runtime matmul
- claim runtime VRAM reduction or inference acceleration

The resulting model still contains ordinary reconstructed FP16
parameters for the replaced matrices.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch

from packed_ternary.checkpoint import (
    load_packed_checkpoint,
)

from packed_ternary.packed_weight import (
    reconstruct_packed,
)


@dataclass
class HybridLoadResult:
    """
    Summary of one hybrid checkpoint injection operation.
    """

    checkpoint_path: str
    model_name: str
    loaded_module_names: Tuple[str, ...]
    replaced_matrix_count: int
    replaced_parameter_count: int


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
            f"Checkpoint target module not found in model: "
            f"{module_name}"
        )

    return modules[module_name]


def _validate_target_module(
    module_name: str,
    module: torch.nn.Module,
    reconstructed: torch.Tensor,
) -> None:
    """
    Validate that a reconstructed checkpoint tensor can replace
    the target module's weight without shape/dtype conversion.
    """

    if not hasattr(module, "weight"):
        raise TypeError(
            f"{module_name} does not have a weight parameter"
        )

    if not isinstance(
        module.weight,
        torch.Tensor,
    ):
        raise TypeError(
            f"{module_name}.weight is not a torch.Tensor"
        )

    if (
        tuple(module.weight.shape)
        != tuple(reconstructed.shape)
    ):
        raise ValueError(
            f"{module_name}: shape mismatch: "
            f"model={tuple(module.weight.shape)} "
            f"checkpoint={tuple(reconstructed.shape)}"
        )

    if (
        module.weight.dtype
        != reconstructed.dtype
    ):
        raise TypeError(
            f"{module_name}: dtype mismatch: "
            f"model={module.weight.dtype} "
            f"checkpoint={reconstructed.dtype}"
        )


@torch.no_grad()
def load_packed_weights_into_model(
    model: torch.nn.Module,
    checkpoint_path,
    expected_model_name: str = None,
) -> HybridLoadResult:
    """
    Load an ENG003 packed checkpoint and inject all packed matrices
    into an already-loaded PyTorch model.

    Parameters
    ----------
    model:
        Existing model whose selected weight matrices will be replaced.

    checkpoint_path:
        Path to the packed multi-matrix checkpoint.

    expected_model_name:
        Optional model identifier. If supplied, it must exactly match
        the model_name stored in the checkpoint.

    Returns
    -------
    HybridLoadResult
        Summary of the matrices and parameter count replaced.

    Important
    ---------
    Reconstruction is performed before any model mutation.

    All checkpoint entries are validated first. Only after every entry
    passes validation are the model weights modified. This prevents a
    partially modified model if a later checkpoint entry is invalid.
    """

    if not isinstance(
        model,
        torch.nn.Module,
    ):
        raise TypeError(
            "model must be a torch.nn.Module"
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
    # Reconstruct and validate EVERYTHING before modifying model.
    # --------------------------------------------------------------

    prepared: List[
        Tuple[
            str,
            torch.nn.Module,
            torch.Tensor,
        ]
    ] = []

    replaced_parameter_count = 0

    for module_name, packed in packed_weights.items():

        module = _get_module_by_name(
            model,
            module_name,
        )

        reconstructed = (
            reconstruct_packed(
                packed
            )
        )

        _validate_target_module(
            module_name,
            module,
            reconstructed,
        )

        prepared.append(
            (
                module_name,
                module,
                reconstructed,
            )
        )

        replaced_parameter_count += (
            reconstructed.numel()
        )

    if not prepared:
        raise ValueError(
            "Checkpoint contains no weights to load"
        )

    # --------------------------------------------------------------
    # Phase 2:
    # Apply only after all targets passed validation.
    # --------------------------------------------------------------

    loaded_module_names = []

    for (
        module_name,
        module,
        reconstructed,
    ) in prepared:

        target_device = (
            module.weight.device
        )

        reconstructed = (
            reconstructed.to(
                device=target_device,
                dtype=module.weight.dtype,
            )
        )

        module.weight.copy_(
            reconstructed
        )

        loaded_module_names.append(
            module_name
        )

    return HybridLoadResult(
        checkpoint_path=str(
            checkpoint_path
        ),
        model_name=checkpoint_model_name,
        loaded_module_names=tuple(
            loaded_module_names
        ),
        replaced_matrix_count=len(
            loaded_module_names
        ),
        replaced_parameter_count=(
            replaced_parameter_count
        ),
    )

