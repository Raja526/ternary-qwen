"""
Serialization utilities for PackedTernaryWeight.

ENG002 objective:
    Save the packed ternary representation to disk,
    load it back, preserve all required metadata,
    and reconstruct the original frozen ternary FP16 result.

This module does not modify the frozen scientific quantizer
or the ENG001 packing implementation.
"""

from pathlib import Path

import torch

from packed_ternary.packed_weight import PackedTernaryWeight


FORMAT_VERSION = 1


def _dtype_to_string(dtype: torch.dtype) -> str:
    """
    Convert supported torch dtype to stable string representation.
    """

    mapping = {
        torch.float16: "float16",
        torch.float32: "float32",
        torch.bfloat16: "bfloat16",
    }

    if dtype not in mapping:
        raise ValueError(
            f"Unsupported dtype for serialization: {dtype}"
        )

    return mapping[dtype]


def _string_to_dtype(name: str) -> torch.dtype:
    """
    Restore torch dtype from serialized string.
    """

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


def save_packed_weight(
    packed_weight: PackedTernaryWeight,
    path,
) -> None:
    """
    Serialize a PackedTernaryWeight to disk.

    The file contains:
        format version
        packed uint8 ternary codes
        reconstruction scales
        original shape
        original number of values
        padded number of values
        group size
        original dtype
    """

    if not isinstance(
        packed_weight,
        PackedTernaryWeight,
    ):
        raise TypeError(
            "packed_weight must be a PackedTernaryWeight"
        )

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "format_version": FORMAT_VERSION,

        "packed_codes": (
            packed_weight
            .packed_codes
            .detach()
            .cpu()
            .contiguous()
        ),

        "scales": (
            packed_weight
            .scales
            .detach()
            .cpu()
            .contiguous()
        ),

        "original_shape": tuple(
            packed_weight.original_shape
        ),

        "original_numel": int(
            packed_weight.original_numel
        ),

        "padded_numel": int(
            packed_weight.padded_numel
        ),

        "group_size": int(
            packed_weight.group_size
        ),

        "original_dtype": _dtype_to_string(
            packed_weight.original_dtype
        ),
    }

    torch.save(
        payload,
        path,
    )


def load_packed_weight(
    path,
) -> PackedTernaryWeight:
    """
    Load a PackedTernaryWeight from disk.

    Performs structural validation before returning the object.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Packed weight file not found: {path}"
        )

    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(payload, dict):
        raise ValueError(
            "Serialized packed weight must contain a dictionary."
        )

    required_keys = {
        "format_version",
        "packed_codes",
        "scales",
        "original_shape",
        "original_numel",
        "padded_numel",
        "group_size",
        "original_dtype",
    }

    missing = (
        required_keys
        - set(payload.keys())
    )

    if missing:
        raise ValueError(
            "Serialized packed weight is missing keys: "
            + ", ".join(sorted(missing))
        )

    if payload["format_version"] != FORMAT_VERSION:
        raise ValueError(
            "Unsupported packed format version: "
            f"{payload['format_version']}"
        )

    packed_codes = payload["packed_codes"]
    scales = payload["scales"]

    if not isinstance(
        packed_codes,
        torch.Tensor,
    ):
        raise TypeError(
            "Serialized packed_codes must be a tensor."
        )

    if packed_codes.dtype != torch.uint8:
        raise TypeError(
            "Serialized packed_codes must use torch.uint8."
        )

    if not isinstance(
        scales,
        torch.Tensor,
    ):
        raise TypeError(
            "Serialized scales must be a tensor."
        )

    original_shape = tuple(
        int(x)
        for x in payload["original_shape"]
    )

    original_numel = int(
        payload["original_numel"]
    )

    padded_numel = int(
        payload["padded_numel"]
    )

    group_size = int(
        payload["group_size"]
    )

    original_dtype = _string_to_dtype(
        payload["original_dtype"]
    )

    if original_numel <= 0:
        raise ValueError(
            "original_numel must be greater than zero."
        )

    if group_size <= 0:
        raise ValueError(
            "group_size must be greater than zero."
        )

    if padded_numel < original_numel:
        raise ValueError(
            "padded_numel cannot be smaller than original_numel."
        )

    if padded_numel % group_size != 0:
        raise ValueError(
            "padded_numel must be divisible by group_size."
        )

    expected_groups = (
        padded_numel // group_size
    )

    if scales.numel() != expected_groups:
        raise ValueError(
            "Serialized scale count mismatch: "
            f"{scales.numel()} != {expected_groups}"
        )

    expected_code_bytes = (
        padded_numel + 3
    ) // 4

    if packed_codes.numel() != expected_code_bytes:
        raise ValueError(
            "Serialized packed-code byte count mismatch: "
            f"{packed_codes.numel()} "
            f"!= {expected_code_bytes}"
        )

    shape_numel = 1

    for dimension in original_shape:
        shape_numel *= dimension

    if shape_numel != original_numel:
        raise ValueError(
            "Serialized shape does not match original_numel: "
            f"{shape_numel} != {original_numel}"
        )

    return PackedTernaryWeight(
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

        original_shape=original_shape,
        original_numel=original_numel,
        padded_numel=padded_numel,
        group_size=group_size,
        original_dtype=original_dtype,
    )
