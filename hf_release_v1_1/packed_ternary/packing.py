"""
2-bit packing utilities for ternary weights.

Encoding
--------
-1 -> 00
 0 -> 01
+1 -> 10
11 -> reserved / invalid

Four ternary values are stored in one uint8 byte.

This module is part of the engineering implementation and is intentionally
separate from the frozen scientific quantizer in ternary/quantizer.py.
"""

import torch


# Ternary value -> 2-bit representation
NEGATIVE_CODE = 0b00
ZERO_CODE = 0b01
POSITIVE_CODE = 0b10
RESERVED_CODE = 0b11


def pack_ternary_codes(codes: torch.Tensor) -> tuple[torch.Tensor, int]:
    """
    Pack {-1, 0, +1} values into 2-bit codes.

    Parameters
    ----------
    codes:
        Tensor containing only -1, 0, +1.

    Returns
    -------
    packed:
        1-D torch.uint8 tensor. Four ternary values per byte.

    original_numel:
        Number of ternary values before padding.

    Notes
    -----
    If the number of values is not divisible by four, the flattened
    sequence is padded with zero-valued ternary entries before packing.
    original_numel allows exact trimming during unpacking.
    """

    if not isinstance(codes, torch.Tensor):
        raise TypeError("codes must be a torch.Tensor")

    flat = codes.detach().reshape(-1)

    original_numel = flat.numel()

    if original_numel == 0:
        return (
            torch.empty(0, dtype=torch.uint8),
            0,
        )

    # Validate before converting representation.
    valid = (flat == -1) | (flat == 0) | (flat == 1)

    if not bool(valid.all().item()):
        invalid_values = torch.unique(flat[~valid]).cpu().tolist()

        raise ValueError(
            "Ternary codes must contain only -1, 0, +1. "
            f"Invalid values: {invalid_values}"
        )

    # Packing is intentionally performed on CPU.
    flat = flat.to(
        device="cpu",
        dtype=torch.int8,
    )

    # Map:
    # -1 -> 0
    #  0 -> 1
    # +1 -> 2
    encoded = (flat + 1).to(torch.uint8)

    remainder = encoded.numel() % 4

    if remainder != 0:
        padding_count = 4 - remainder

        # Ternary zero maps to encoded value 1.
        padding = torch.full(
            (padding_count,),
            ZERO_CODE,
            dtype=torch.uint8,
        )

        encoded = torch.cat(
            [encoded, padding],
            dim=0,
        )

    encoded = encoded.reshape(-1, 4)

    # Layout within each byte:
    #
    # bits 1:0 -> value 0
    # bits 3:2 -> value 1
    # bits 5:4 -> value 2
    # bits 7:6 -> value 3
    #
    packed = (
        encoded[:, 0]
        | (encoded[:, 1] << 2)
        | (encoded[:, 2] << 4)
        | (encoded[:, 3] << 6)
    ).to(torch.uint8)

    return packed.contiguous(), original_numel


def unpack_ternary_codes(
    packed: torch.Tensor,
    original_numel: int,
) -> torch.Tensor:
    """
    Unpack a uint8 tensor containing four 2-bit ternary codes per byte.

    Returns
    -------
    codes:
        1-D torch.int8 tensor containing exactly original_numel values
        from {-1, 0, +1}.

    Raises
    ------
    ValueError:
        If a packed 2-bit field contains reserved code 11.
    """

    if not isinstance(packed, torch.Tensor):
        raise TypeError("packed must be a torch.Tensor")

    if packed.dtype != torch.uint8:
        raise TypeError(
            f"packed must have dtype torch.uint8, got {packed.dtype}"
        )

    if not isinstance(original_numel, int):
        raise TypeError("original_numel must be an int")

    if original_numel < 0:
        raise ValueError("original_numel cannot be negative")

    packed = packed.detach().reshape(-1).cpu()

    capacity = packed.numel() * 4

    if original_numel > capacity:
        raise ValueError(
            f"original_numel={original_numel} exceeds packed "
            f"capacity={capacity}"
        )

    if original_numel == 0:
        return torch.empty(0, dtype=torch.int8)

    decoded = torch.empty(
        packed.numel() * 4,
        dtype=torch.uint8,
    )

    decoded[0::4] = packed & 0b11
    decoded[1::4] = (packed >> 2) & 0b11
    decoded[2::4] = (packed >> 4) & 0b11
    decoded[3::4] = (packed >> 6) & 0b11

    decoded = decoded[:original_numel]

    # 11 is deliberately reserved.
    if bool((decoded == RESERVED_CODE).any().item()):
        positions = (
            torch.nonzero(
                decoded == RESERVED_CODE,
                as_tuple=False,
            )
            .reshape(-1)
            .cpu()
            .tolist()
        )

        raise ValueError(
            "Packed tensor contains reserved 2-bit code 11 "
            f"at positions: {positions[:20]}"
        )

    # Reverse mapping:
    # 0 -> -1
    # 1 ->  0
    # 2 -> +1
    codes = decoded.to(torch.int8) - 1

    return codes.contiguous()


def packed_code_bytes(num_ternary_values: int) -> int:
    """
    Return the number of bytes required for 2-bit packed ternary codes.
    """

    if not isinstance(num_ternary_values, int):
        raise TypeError("num_ternary_values must be an int")

    if num_ternary_values < 0:
        raise ValueError("num_ternary_values cannot be negative")

    return (num_ternary_values + 3) // 4
