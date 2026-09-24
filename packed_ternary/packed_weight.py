"""
Packed ternary weight representation.

Engineering implementation corresponding to the frozen quantization
algorithm in ternary/quantizer.py.

Scientific quantizer:
    ternary/quantizer.py

This module DOES NOT modify the frozen scientific implementation.

Storage format
--------------
- Four ternary codes per uint8 byte.
- One reconstruction scale per group.
- Default group size: 128.
- Reconstruction scales stored as FP16.

Ternary encoding
----------------
-1 -> 00
 0 -> 01
+1 -> 10
11 -> reserved
"""

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F

from packed_ternary.packing import (
    pack_ternary_codes,
    unpack_ternary_codes,
)


DEFAULT_GROUP_SIZE = 128
DEFAULT_THRESHOLD_MULTIPLIER = 0.50


@dataclass
class PackedTernaryWeight:
    """
    Compact representation of one ternarized weight tensor.

    Attributes
    ----------
    packed_codes:
        1-D uint8 tensor containing four ternary codes per byte.

    scales:
        1-D tensor containing one reconstruction scale per group.

    original_shape:
        Shape of the original weight tensor.

    original_numel:
        Number of original weight values before group padding.

    padded_numel:
        Number of values after padding to group_size.

    group_size:
        Number of weights sharing one reconstruction scale.

    original_dtype:
        Original PyTorch weight dtype.
    """

    packed_codes: torch.Tensor
    scales: torch.Tensor
    original_shape: Tuple[int, ...]
    original_numel: int
    padded_numel: int
    group_size: int
    original_dtype: torch.dtype

    @property
    def num_groups(self) -> int:
        return self.padded_numel // self.group_size

    @property
    def code_bytes(self) -> int:
        return self.packed_codes.numel() * self.packed_codes.element_size()

    @property
    def scale_bytes(self) -> int:
        return self.scales.numel() * self.scales.element_size()

    @property
    def packed_storage_bytes(self) -> int:
        """
        Payload bytes only:

            packed ternary codes
            +
            per-group reconstruction scales

        Python object metadata is intentionally excluded.
        """
        return self.code_bytes + self.scale_bytes

    @property
    def original_fp16_bytes(self) -> int:
        """
        FP16-equivalent storage for the original number of weights.

        ENG001 is currently comparing packed storage against FP16 weights.
        """
        return self.original_numel * 2

    @property
    def storage_ratio(self) -> float:
        if self.original_fp16_bytes == 0:
            return 0.0

        return (
            self.packed_storage_bytes
            / self.original_fp16_bytes
        )

    @property
    def compression_factor(self) -> float:
        if self.packed_storage_bytes == 0:
            return 0.0

        return (
            self.original_fp16_bytes
            / self.packed_storage_bytes
        )


@torch.no_grad()
def quantize_to_packed(
    weight: torch.Tensor,
    group_size: int = DEFAULT_GROUP_SIZE,
    threshold_multiplier: float = DEFAULT_THRESHOLD_MULTIPLIER,
    scale_dtype: torch.dtype = torch.float16,
) -> PackedTernaryWeight:
    """
    Apply the same optimized groupwise ternarization mathematics used by
    the frozen scientific quantizer, then store the result compactly.

    IMPORTANT
    ---------
    Code selection and reconstruction-scale calculation are performed
    in FP32 exactly like ternary/quantizer.py.

    Only AFTER reconstruction_alpha has been computed is it converted
    to scale_dtype for packed storage.

    For ENG001:
        scale_dtype = torch.float16

    This means the packed representation itself may introduce a small
    difference relative to the scientific quantizer if the scientific
    FP32 reconstruction scale is not exactly representable in FP16.

    We explicitly measure this rather than hiding it.
    """

    if not isinstance(weight, torch.Tensor):
        raise TypeError("weight must be a torch.Tensor")

    if group_size <= 0:
        raise ValueError("group_size must be greater than zero")

    if threshold_multiplier < 0:
        raise ValueError(
            "threshold_multiplier cannot be negative"
        )

    original_shape = tuple(weight.shape)

    original_numel = weight.numel()

    if original_numel == 0:
        raise ValueError(
            "Cannot pack an empty weight tensor."
        )

    # ------------------------------------------------------------------
    # Match frozen quantizer:
    #
    # weight.detach().float().reshape(-1)
    # ------------------------------------------------------------------

    flat = (
        weight
        .detach()
        .float()
        .reshape(-1)
    )

    padding = (
        group_size
        - (original_numel % group_size)
    ) % group_size

    if padding > 0:
        padded = F.pad(
            flat,
            (0, padding),
            mode="constant",
            value=0.0,
        )
    else:
        padded = flat

    padded_numel = padded.numel()

    groups = padded.reshape(
        -1,
        group_size,
    )

    # ------------------------------------------------------------------
    # Step 1 — frozen code-selection scale
    # ------------------------------------------------------------------

    initial_alpha = (
        groups
        .abs()
        .mean(
            dim=1,
            keepdim=True,
        )
    )

    # ------------------------------------------------------------------
    # Step 2 — frozen threshold
    # ------------------------------------------------------------------

    threshold = (
        threshold_multiplier
        * initial_alpha
    )

    # int8 is sufficient for engineering storage representation.
    codes = torch.zeros_like(
        groups,
        dtype=torch.int8,
    )

    codes[
        groups > threshold
    ] = 1

    codes[
        groups < -threshold
    ] = -1

    # ------------------------------------------------------------------
    # Step 3 — frozen optimized reconstruction scale
    # ------------------------------------------------------------------

    nonzero_mask = (
        codes != 0
    )

    nonzero_count = (
        nonzero_mask
        .sum(
            dim=1,
            keepdim=True,
        )
    )

    nonzero_abs_sum = (
        groups
        .abs()
        .mul(nonzero_mask)
        .sum(
            dim=1,
            keepdim=True,
        )
    )

    optimized_alpha = (
        nonzero_abs_sum
        / nonzero_count.clamp(min=1)
    )

    reconstruction_alpha = torch.where(
        nonzero_count > 0,
        optimized_alpha,
        initial_alpha,
    )

    # ------------------------------------------------------------------
    # Pack ALL padded codes.
    #
    # This is useful because each complete group can then be reconstructed
    # directly with its corresponding scale.
    # ------------------------------------------------------------------

    flat_codes = (
        codes
        .reshape(-1)
        .cpu()
        .contiguous()
    )

    packed_codes, packed_numel = (
        pack_ternary_codes(
            flat_codes
        )
    )

    if packed_numel != padded_numel:
        raise RuntimeError(
            "Internal packed code length mismatch: "
            f"{packed_numel} != {padded_numel}"
        )

    # ------------------------------------------------------------------
    # Store one scale per group.
    #
    # Scale storage is intentionally FP16 for ENG001.
    # ------------------------------------------------------------------

    scales = (
        reconstruction_alpha
        .reshape(-1)
        .to(
            device="cpu",
            dtype=scale_dtype,
        )
        .contiguous()
    )

    expected_groups = (
        padded_numel // group_size
    )

    if scales.numel() != expected_groups:
        raise RuntimeError(
            "Scale count mismatch: "
            f"{scales.numel()} != {expected_groups}"
        )

    return PackedTernaryWeight(
        packed_codes=packed_codes,
        scales=scales,
        original_shape=original_shape,
        original_numel=original_numel,
        padded_numel=padded_numel,
        group_size=group_size,
        original_dtype=weight.dtype,
    )


@torch.no_grad()
def reconstruct_packed(
    packed_weight: PackedTernaryWeight,
) -> torch.Tensor:
    """
    Reconstruct the packed ternary weight tensor.

    Reconstruction is intentionally performed in FP32:

        code * scale

    and then converted back to the original weight dtype, matching the
    final dtype behavior of the frozen scientific quantizer.
    """

    if not isinstance(
        packed_weight,
        PackedTernaryWeight,
    ):
        raise TypeError(
            "packed_weight must be a PackedTernaryWeight"
        )

    codes = unpack_ternary_codes(
        packed_weight.packed_codes,
        packed_weight.padded_numel,
    )

    expected_numel = (
        packed_weight.num_groups
        * packed_weight.group_size
    )

    if codes.numel() != expected_numel:
        raise RuntimeError(
            "Unpacked code count mismatch: "
            f"{codes.numel()} != {expected_numel}"
        )

    groups = (
        codes
        .to(torch.float32)
        .reshape(
            packed_weight.num_groups,
            packed_weight.group_size,
        )
    )

    scales = (
        packed_weight.scales
        .to(torch.float32)
        .reshape(-1, 1)
    )

    reconstructed = (
        groups
        * scales
    )

    reconstructed = (
        reconstructed
        .reshape(-1)[
            :packed_weight.original_numel
        ]
        .reshape(
            packed_weight.original_shape
        )
        .to(
            dtype=packed_weight.original_dtype
        )
    )

    return reconstructed.contiguous()
