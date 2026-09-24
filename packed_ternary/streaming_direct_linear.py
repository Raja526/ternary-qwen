"""
Streaming direct ternary linear prototype.

ENG011
------

Correctness-oriented extension of ENG010.

ENG010 keeps the compact 2-bit representation persistently, but decodes the
entire selected matrix into a temporary int8 ternary-code tensor before the
row-chunk compute loop.

ENG011 changes one variable:

    decode only the packed codes required by the current output-row chunk.

Persistent representation remains:

    packed 2-bit ternary codes
    +
    one FP16 scale per quantization group

No persistent or temporary reconstructed FP16 weight tensor is created.

Scope
-----
This implementation intentionally preserves the frozen ENG010 restrictions:

    padded_numel == original_numel
    in_features % group_size == 0

It additionally relies on:

    in_features % 4 == 0

so each output row begins and ends on packed-byte boundaries.

This is still not a fused 2-bit kernel. Each current row chunk is decoded
into a temporary int8 tensor before direct ternary arithmetic.
"""

from typing import Optional

import torch
import torch.nn as nn

from packed_ternary.packed_weight import (
    PackedTernaryWeight,
)

from packed_ternary.packing import (
    RESERVED_CODE,
)


class StreamingDirectTernaryLinear(nn.Module):
    """
    Direct ternary linear layer with chunk-local packed-code decoding.
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
                "StreamingDirectTernaryLinear requires a 2-D weight tensor"
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
                "ENG011 prototype currently requires no group padding"
            )

        if (
            in_features
            % packed_weight.group_size
            != 0
        ):
            raise ValueError(
                "ENG011 prototype requires row-aligned quantization groups"
            )

        if (
            in_features
            % 4
            != 0
        ):
            raise ValueError(
                "ENG011 prototype requires each row to align to packed bytes"
            )

        expected_groups = (
            packed_weight.original_numel
            // packed_weight.group_size
        )

        if (
            packed_weight.scales.numel()
            != expected_groups
        ):
            raise ValueError(
                "Scale count does not match packed weight groups"
            )

        expected_packed_bytes = (
            packed_weight.original_numel
            // 4
        )

        if (
            packed_weight.packed_codes.numel()
            != expected_packed_bytes
        ):
            raise ValueError(
                "Packed-code byte count does not match expected layout"
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

        self.out_features = (
            out_features
        )

        self.in_features = (
            in_features
        )

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

        self.bytes_per_row = (
            in_features
            // 4
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

    @property
    def full_decoded_code_bytes(
        self,
    ) -> int:
        """
        Size of the full-matrix int8 code tensor ENG010 would decode.
        """
        return self.original_numel

    @property
    def maximum_chunk_decoded_code_bytes(
        self,
    ) -> int:
        """
        Maximum int8 code payload decoded by one ENG011 row chunk.
        """
        return (
            min(
                self.rows_per_chunk,
                self.out_features,
            )
            * self.in_features
        )

    def _decode_row_chunk(
        self,
        row_start: int,
        row_end: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Decode only [row_start, row_end) from packed 2-bit storage.

        Returns
        -------
        torch.Tensor
            int8 tensor with shape:

                [chunk_rows, groups_per_row, group_size]
        """

        if not isinstance(
            row_start,
            int,
        ):
            raise TypeError(
                "row_start must be an int"
            )

        if not isinstance(
            row_end,
            int,
        ):
            raise TypeError(
                "row_end must be an int"
            )

        if (
            row_start < 0
            or row_end > self.out_features
            or row_start >= row_end
        ):
            raise ValueError(
                "Invalid row chunk"
            )

        byte_start = (
            row_start
            * self.bytes_per_row
        )

        byte_end = (
            row_end
            * self.bytes_per_row
        )

        packed_chunk = (
            self.packed_codes[
                byte_start:byte_end
            ]
            .to(
                device=device,
            )
        )

        expected_bytes = (
            (row_end - row_start)
            * self.bytes_per_row
        )

        if (
            packed_chunk.numel()
            != expected_bytes
        ):
            raise RuntimeError(
                "Packed chunk byte-count mismatch"
            )

        decoded = torch.empty(
            packed_chunk.numel() * 4,
            dtype=torch.uint8,
            device=device,
        )

        decoded[0::4] = (
            packed_chunk
            & 0b11
        )

        decoded[1::4] = (
            (packed_chunk >> 2)
            & 0b11
        )

        decoded[2::4] = (
            (packed_chunk >> 4)
            & 0b11
        )

        decoded[3::4] = (
            (packed_chunk >> 6)
            & 0b11
        )

        if bool(
            (
                decoded
                == RESERVED_CODE
            )
            .any()
            .item()
        ):
            raise ValueError(
                "Packed chunk contains reserved 2-bit code 11"
            )

        codes = (
            decoded.to(
                torch.int8
            )
            - 1
        )

        expected_codes = (
            (row_end - row_start)
            * self.in_features
        )

        if (
            codes.numel()
            != expected_codes
        ):
            raise RuntimeError(
                "Decoded chunk code-count mismatch"
            )

        return (
            codes
            .reshape(
                row_end - row_start,
                self.groups_per_row,
                self.group_size,
            )
            .contiguous()
        )

    def forward(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute direct ternary linear output using chunk-local decoding.

        The arithmetic intentionally follows ENG010:

            positive_sum
            negative_sum
            (positive_sum - negative_sum) * scale
            sum across groups

        The controlled ENG011 change is only when packed codes are decoded.
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

        flat_input = (
            input.reshape(
                -1,
                self.in_features,
            )
        )

        compute_device = (
            flat_input.device
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

            chunk_codes = (
                self._decode_row_chunk(
                    row_start=row_start,
                    row_end=row_end,
                    device=compute_device,
                )
            )

            chunk_scales = (
                scales[
                    row_start:row_end
                ]
            )

            positive_mask = (
                chunk_codes == 1
            )

            negative_mask = (
                chunk_codes == -1
            )

            positive_sum = (
                grouped_input[
                    :,
                    None,
                    :,
                    :,
                ]
                .masked_fill(
                    ~positive_mask[
                        None,
                        :,
                        :,
                        :,
                    ],
                    0,
                )
                .sum(
                    dim=-1,
                )
            )

            negative_sum = (
                grouped_input[
                    :,
                    None,
                    :,
                    :,
                ]
                .masked_fill(
                    ~negative_mask[
                        None,
                        :,
                        :,
                        :,
                    ],
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

        return output.reshape(
            *original_input_shape[:-1],
            self.out_features,
        )
