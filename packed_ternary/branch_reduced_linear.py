"""
Parallel compiled packed ternary linear layer.

OPT003
------

PyTorch nn.Module wrapper around the OPT003 parallel compiled CPU
packed-ternary operator.

Persistent representation:

    packed 2-bit ternary codes
    +
    one FP16 scale per quantization group

The underlying OPT003 parallel operator directly interprets packed 2-bit codes.

It does not materialize:

    full decoded int8 weight codes
    chunk decoded int8 weight codes
    positive/negative mask tensors
    reconstructed FP16 weight tensors

OPT003 adds model-integration functionality only. It does not change
the OPT001 C++ arithmetic.
"""

from typing import Optional

import torch
import torch.nn as nn

from packed_ternary.packed_weight import (
    PackedTernaryWeight,
)

from packed_ternary.opt005_extension import (
    load_opt005_extension,
)


class BranchReducedPackedTernaryLinear(nn.Module):
    """
    nn.Module wrapper for the OPT003 parallel compiled CPU operator.
    """

    def __init__(
        self,
        packed_weight: PackedTernaryWeight,
        bias: Optional[torch.Tensor] = None,
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
                "BranchReducedPackedTernaryLinear requires a 2-D weight"
            )

        out_features = int(
            packed_weight.original_shape[0]
        )

        in_features = int(
            packed_weight.original_shape[1]
        )

        if (
            out_features * in_features
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
                "OPT003 currently requires no group padding"
            )

        if (
            in_features
            % packed_weight.group_size
            != 0
        ):
            raise ValueError(
                "OPT003 requires row-aligned quantization groups"
            )

        if (
            in_features
            % 4
            != 0
        ):
            raise ValueError(
                "OPT003 requires rows aligned to packed bytes"
            )

        groups_per_row = (
            in_features
            // packed_weight.group_size
        )

        expected_scales = (
            out_features
            * groups_per_row
        )

        if (
            packed_weight.scales.numel()
            != expected_scales
        ):
            raise ValueError(
                "Scale count does not match packed matrix"
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
                "Packed-code byte count does not match matrix"
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

        self.original_numel = int(
            packed_weight.original_numel
        )

        self.group_size = int(
            packed_weight.group_size
        )

        self.groups_per_row = (
            groups_per_row
        )

        self.original_dtype = (
            packed_weight.original_dtype
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

        self._extension = None

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

    def _get_extension(self):
        if self._extension is None:
            self._extension = (
                load_opt005_extension(
                    verbose=False
                )
            )

        return self._extension

    def forward(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:

        if (
            input.shape[-1]
            != self.in_features
        ):
            raise ValueError(
                "Input last dimension does not match in_features"
            )

        if not input.device.type == "cpu":
            raise ValueError(
                "OPT003 compiled operator currently supports CPU only"
            )

        if input.dtype != torch.float16:
            raise ValueError(
                "OPT003 compiled operator currently requires FP16 input"
            )

        original_shape = tuple(
            input.shape
        )

        flat_input = (
            input.reshape(
                -1,
                self.in_features,
            )
            .contiguous()
        )

        extension = (
            self._get_extension()
        )

        output = (
            extension
            .packed_ternary_linear_cpu_parallel(
                flat_input,
                self.packed_codes,
                self.scales.reshape(-1),
                self.out_features,
                self.group_size,
            )
        )

        output = output.reshape(
            *original_shape[:-1],
            self.out_features,
        )

        if self.bias is not None:
            output = (
                output
                + self.bias
            )

        return output
