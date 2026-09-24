"""
Direct ternary linear prototype.

ENG010
------

Correctness-oriented CPU prototype for computing a linear operation from
packed ternary codes and per-group scales WITHOUT reconstructing an FP16
weight matrix or FP16 weight-row chunk.

Scope
-----
This first implementation intentionally supports row-aligned groups:

    in_features % group_size == 0

That condition is satisfied by all three frozen ENG003 matrices.

The implementation still decodes ternary codes into temporary int8 tensors.
It therefore does NOT yet represent a fused 2-bit kernel.

The important ENG010 change is narrower:

    no reconstructed FP16 weight tensor is created.

This is a correctness prototype, not an optimized inference kernel.
"""

from typing import Optional

import torch
import torch.nn as nn

from packed_ternary.packed_weight import (
    PackedTernaryWeight,
)

from packed_ternary.packing import (
    unpack_ternary_codes,
)


class DirectTernaryLinear(nn.Module):
    """
    Linear layer using direct ternary arithmetic.

    Persistent representation:

        packed 2-bit codes
        +
        one FP16 scale per group

    No persistent or temporary reconstructed FP16 weight matrix is created.
    """

    def __init__(
        self,
        packed_weight: PackedTernaryWeight,
        bias: Optional[torch.Tensor] = None,
        rows_per_chunk: int = 64,
    ):
        super().__init__()

        if not isinstance(
            packed_weight,
            PackedTernaryWeight,
        ):
            raise TypeError(
                "packed_weight must be a PackedTernaryWeight"
            )

        if len(
            packed_weight.original_shape
        ) != 2:
            raise ValueError(
                "DirectTernaryLinear requires a 2-D weight tensor"
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

        out_features = int(
            packed_weight.original_shape[0]
        )

        in_features = int(
            packed_weight.original_shape[1]
        )

        if (
            out_features
            * in_features
            != packed_weight.original_numel
        ):
            raise ValueError(
                "Packed weight shape does not match original_numel"
            )

        if (
            packed_weight.padded_numel
            != packed_weight.original_numel
        ):
            raise ValueError(
                "ENG010 prototype currently requires no group padding"
            )

        if (
            in_features
            % packed_weight.group_size
            != 0
        ):
            raise ValueError(
                "ENG010 prototype requires row-aligned quantization groups"
            )

        expected_groups = (
            packed_weight.padded_numel
            // packed_weight.group_size
        )

        if (
            packed_weight.scales.numel()
            != expected_groups
        ):
            raise ValueError(
                "Scale count does not match packed weight groups"
            )

        if bias is not None:
            if not isinstance(
                bias,
                torch.Tensor,
            ):
                raise TypeError(
                    "bias must be a torch.Tensor or None"
                )

            if bias.ndim != 1:
                raise ValueError(
                    "bias must be 1-D"
                )

            if (
                bias.numel()
                != out_features
            ):
                raise ValueError(
                    "bias length must equal out_features"
                )

        self.out_features = out_features
        self.in_features = in_features

        self.original_numel = (
            packed_weight.original_numel
        )

        self.group_size = (
            packed_weight.group_size
        )

        self.groups_per_row = (
            in_features
            // self.group_size
        )

        self.original_dtype = (
            packed_weight.original_dtype
        )

        self.rows_per_chunk = (
            rows_per_chunk
        )

        self.register_buffer(
            "packed_codes",
            packed_weight
            .packed_codes
            .detach()
            .clone()
            .contiguous(),
            persistent=True,
        )

        self.register_buffer(
            "scales",
            packed_weight
            .scales
            .detach()
            .clone()
            .contiguous(),
            persistent=True,
        )

        if bias is None:
            self.register_parameter(
                "bias",
                None,
            )
        else:
            self.bias = nn.Parameter(
                bias
                .detach()
                .clone(),
                requires_grad=False,
            )

    @property
    def persistent_weight_bytes(
        self,
    ) -> int:
        return (
            self.packed_codes.numel()
            * self.packed_codes.element_size()
            +
            self.scales.numel()
            * self.scales.element_size()
        )

    @property
    def fp16_equivalent_weight_bytes(
        self,
    ) -> int:
        return (
            self.original_numel
            * 2
        )

    @property
    def persistent_compression_factor(
        self,
    ) -> float:
        return (
            self.fp16_equivalent_weight_bytes
            / self.persistent_weight_bytes
        )

    def _decode_codes(
        self,
    ) -> torch.Tensor:
        """
        Decode packed 2-bit values into row/group layout.

        Shape:

            [out_features, groups_per_row, group_size]

        dtype:

            int8

        No FP16 weight tensor is created.
        """

        codes = unpack_ternary_codes(
            self.packed_codes,
            self.original_numel,
        )

        if (
            codes.numel()
            != self.original_numel
        ):
            raise RuntimeError(
                "Decoded code count mismatch"
            )

        return codes.reshape(
            self.out_features,
            self.groups_per_row,
            self.group_size,
        )

    def forward(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the linear operation directly from ternary codes.

        For every output row and quantization group:

            positive_sum = sum(input where code == +1)
            negative_sum = sum(input where code == -1)

            contribution =
                (positive_sum - negative_sum) * scale

        No reconstructed FP16 weight row/chunk is formed.
        """

        if (
            input.shape[-1]
            != self.in_features
        ):
            raise ValueError(
                "Input last dimension does not match in_features"
            )

        original_input_shape = (
            tuple(input.shape)
        )

        flat_input = input.reshape(
            -1,
            self.in_features,
        )

        compute_device = (
            flat_input.device
        )

        codes = (
            self
            ._decode_codes()
            .to(
                device=compute_device,
            )
        )

        scales = (
            self.scales
            .reshape(
                self.out_features,
                self.groups_per_row,
            )
            .to(
                device=compute_device,
            )
        )

        grouped_input = (
            flat_input
            .reshape(
                flat_input.shape[0],
                self.groups_per_row,
                self.group_size,
            )
        )

        outputs = []

        for row_start in range(
            0,
            self.out_features,
            self.rows_per_chunk,
        ):
            row_end = min(
                row_start
                + self.rows_per_chunk,
                self.out_features,
            )

            chunk_codes = codes[
                row_start:row_end
            ]

            chunk_scales = scales[
                row_start:row_end
            ]

            positive_mask = (
                chunk_codes == 1
            )

            negative_mask = (
                chunk_codes == -1
            )

            positive_sum = (
                grouped_input[:, None, :, :]
                .masked_fill(
                    ~positive_mask[None, :, :, :],
                    0,
                )
                .sum(
                    dim=-1,
                )
            )

            negative_sum = (
                grouped_input[:, None, :, :]
                .masked_fill(
                    ~negative_mask[None, :, :, :],
                    0,
                )
                .sum(
                    dim=-1,
                )
            )

            group_contribution = (
                positive_sum
                - negative_sum
            )

            group_contribution = (
                group_contribution
                * chunk_scales[
                    None,
                    :,
                    :,
                ]
            )

            chunk_output = (
                group_contribution
                .sum(
                    dim=-1,
                )
            )

            outputs.append(
                chunk_output
            )

        output = torch.cat(
            outputs,
            dim=-1,
        )

        if self.bias is not None:
            output = (
                output
                + self.bias
            )

        output_shape = (
            original_input_shape[:-1]
            + (self.out_features,)
        )

        return output.reshape(
            output_shape
        )
