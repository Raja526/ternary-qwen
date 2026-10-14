"""
Multi-matrix packed ternary checkpoint utilities.

ENG003 objective:
    Store multiple PackedTernaryWeight objects in one checkpoint while
    preserving matrix names and all reconstruction metadata.

ENG001 and ENG002 implementations remain unchanged.
"""

from pathlib import Path
from typing import Dict

import torch

from packed_ternary.packed_weight import PackedTernaryWeight


CHECKPOINT_FORMAT_VERSION = 1


def _dtype_to_string(dtype: torch.dtype) -> str:
    mapping = {
        torch.float16: "float16",
        torch.float32: "float32",
        torch.bfloat16: "bfloat16",
    }

    if dtype not in mapping:
        raise ValueError(
            f"Unsupported dtype: {dtype}"
        )

    return mapping[dtype]


def _string_to_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }

    if name not in mapping:
        raise ValueError(
            f"Unsupported serialized dtype: {name}"
        )

    return mapping[name]


def save_packed_checkpoint(
    weights: Dict[str, PackedTernaryWeight],
    path,
    model_name: str,
    group_size: int,
    threshold_multiplier: float,
) -> None:
    """
    Save multiple packed ternary matrices into one checkpoint.
    """

    if not isinstance(weights, dict):
        raise TypeError(
            "weights must be a dictionary"
        )

    if not weights:
        raise ValueError(
            "weights cannot be empty"
        )

    if not isinstance(model_name, str):
        raise TypeError(
            "model_name must be a string"
        )

    if not model_name:
        raise ValueError(
            "model_name cannot be empty"
        )

    if group_size <= 0:
        raise ValueError(
            "group_size must be greater than zero"
        )

    serialized_weights = {}

    for name, packed in weights.items():

        if not isinstance(name, str):
            raise TypeError(
                "Weight names must be strings"
            )

        if not name:
            raise ValueError(
                "Weight names cannot be empty"
            )

        if not isinstance(
            packed,
            PackedTernaryWeight,
        ):
            raise TypeError(
                f"{name} is not a PackedTernaryWeight"
            )

        if packed.group_size != group_size:
            raise ValueError(
                f"Group-size mismatch for {name}: "
                f"{packed.group_size} != {group_size}"
            )

        serialized_weights[name] = {
            "packed_codes": (
                packed.packed_codes
                .detach()
                .cpu()
                .contiguous()
            ),

            "scales": (
                packed.scales
                .detach()
                .cpu()
                .contiguous()
            ),

            "original_shape": tuple(
                packed.original_shape
            ),

            "original_numel": int(
                packed.original_numel
            ),

            "padded_numel": int(
                packed.padded_numel
            ),

            "group_size": int(
                packed.group_size
            ),

            "original_dtype": _dtype_to_string(
                packed.original_dtype
            ),
        }

    payload = {
        "format_version":
            CHECKPOINT_FORMAT_VERSION,

        "model_name":
            model_name,

        "group_size":
            int(group_size),

        "threshold_multiplier":
            float(threshold_multiplier),

        "weights":
            serialized_weights,
    }

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        payload,
        path,
    )


def load_packed_checkpoint(
    path,
):
    """
    Load and validate a multi-matrix packed ternary checkpoint.

    Returns
    -------
    metadata:
        Dictionary containing checkpoint-level metadata.

    weights:
        Dictionary mapping module names to PackedTernaryWeight.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}"
        )

    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(payload, dict):
        raise ValueError(
            "Checkpoint payload must be a dictionary"
        )

    required_keys = {
        "format_version",
        "model_name",
        "group_size",
        "threshold_multiplier",
        "weights",
    }

    missing = (
        required_keys
        - set(payload.keys())
    )

    if missing:
        raise ValueError(
            "Checkpoint is missing keys: "
            + ", ".join(sorted(missing))
        )

    if (
        payload["format_version"]
        != CHECKPOINT_FORMAT_VERSION
    ):
        raise ValueError(
            "Unsupported checkpoint format version: "
            f"{payload['format_version']}"
        )

    model_name = payload["model_name"]
    group_size = int(
        payload["group_size"]
    )

    threshold_multiplier = float(
        payload["threshold_multiplier"]
    )

    serialized_weights = payload[
        "weights"
    ]

    if not isinstance(model_name, str):
        raise TypeError(
            "Serialized model_name must be a string"
        )

    if not model_name:
        raise ValueError(
            "Serialized model_name cannot be empty"
        )

    if group_size <= 0:
        raise ValueError(
            "Serialized group_size must be positive"
        )

    if not isinstance(
        serialized_weights,
        dict,
    ):
        raise TypeError(
            "Serialized weights must be a dictionary"
        )

    if not serialized_weights:
        raise ValueError(
            "Serialized weights cannot be empty"
        )

    restored_weights = {}

    for name, item in serialized_weights.items():

        if not isinstance(name, str):
            raise TypeError(
                "Serialized weight names must be strings"
            )

        if not isinstance(item, dict):
            raise TypeError(
                f"Serialized entry for {name} must be a dictionary"
            )

        required_weight_keys = {
            "packed_codes",
            "scales",
            "original_shape",
            "original_numel",
            "padded_numel",
            "group_size",
            "original_dtype",
        }

        missing_weight_keys = (
            required_weight_keys
            - set(item.keys())
        )

        if missing_weight_keys:
            raise ValueError(
                f"{name} is missing keys: "
                + ", ".join(
                    sorted(missing_weight_keys)
                )
            )

        packed_codes = item[
            "packed_codes"
        ]

        scales = item[
            "scales"
        ]

        if not isinstance(
            packed_codes,
            torch.Tensor,
        ):
            raise TypeError(
                f"{name}: packed_codes must be a tensor"
            )

        if packed_codes.dtype != torch.uint8:
            raise TypeError(
                f"{name}: packed_codes must use torch.uint8"
            )

        if not isinstance(
            scales,
            torch.Tensor,
        ):
            raise TypeError(
                f"{name}: scales must be a tensor"
            )

        original_shape = tuple(
            int(x)
            for x in item["original_shape"]
        )

        original_numel = int(
            item["original_numel"]
        )

        padded_numel = int(
            item["padded_numel"]
        )

        item_group_size = int(
            item["group_size"]
        )

        original_dtype = _string_to_dtype(
            item["original_dtype"]
        )

        if original_numel <= 0:
            raise ValueError(
                f"{name}: original_numel must be positive"
            )

        if item_group_size != group_size:
            raise ValueError(
                f"{name}: group-size mismatch "
                f"{item_group_size} != {group_size}"
            )

        if padded_numel < original_numel:
            raise ValueError(
                f"{name}: padded_numel is smaller "
                "than original_numel"
            )

        if padded_numel % group_size != 0:
            raise ValueError(
                f"{name}: padded_numel is not divisible "
                "by group_size"
            )

        shape_numel = 1

        for dimension in original_shape:
            shape_numel *= dimension

        if shape_numel != original_numel:
            raise ValueError(
                f"{name}: shape contains {shape_numel} values "
                f"but original_numel={original_numel}"
            )

        expected_groups = (
            padded_numel
            // group_size
        )

        if scales.numel() != expected_groups:
            raise ValueError(
                f"{name}: scale count mismatch "
                f"{scales.numel()} != {expected_groups}"
            )

        expected_code_bytes = (
            padded_numel + 3
        ) // 4

        if (
            packed_codes.numel()
            != expected_code_bytes
        ):
            raise ValueError(
                f"{name}: packed-code byte count mismatch "
                f"{packed_codes.numel()} "
                f"!= {expected_code_bytes}"
            )

        restored_weights[name] = (
            PackedTernaryWeight(
                packed_codes=(
                    packed_codes
                    .reshape(-1)
                    .contiguous()
                ),

                scales=(
                    scales
                    .reshape(-1)
                    .contiguous()
                ),

                original_shape=
                    original_shape,

                original_numel=
                    original_numel,

                padded_numel=
                    padded_numel,

                group_size=
                    item_group_size,

                original_dtype=
                    original_dtype,
            )
        )

    metadata = {
        "format_version":
            payload["format_version"],

        "model_name":
            model_name,

        "group_size":
            group_size,

        "threshold_multiplier":
            threshold_multiplier,

        "weight_count":
            len(restored_weights),
    }

    return metadata, restored_weights
