"""
Correctness-first packed ternary linear operator.

ENG005
------

This module executes a linear operation from a PackedTernaryWeight
without storing a persistent full reconstructed FP16 weight matrix.

Important
---------
The frozen packed format groups weights over the flattened weight tensor.

Therefore reconstruction must preserve those exact flattened group
boundaries.

This implementation:

    1. unpacks the compact 2-bit ternary codes,
    2. processes bounded chunks of output rows,
    3. reconstructs only the groups needed by each chunk,
    4. performs torch.nn.functional.linear on that temporary chunk,
    5. concatenates the partial outputs.

This is a correctness/reference implementation.

It is NOT:
    - a fused 2-bit CUDA kernel,
    - an optimized inference kernel,
    - evidence of inference acceleration.

The packed codes and scales remain the persistent weight representation.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from packed_ternary.packed_weight import (
    PackedTernaryWeight,
)

from packed_ternary.packing import (
    unpack_ternary_codes,
)


class PackedTernaryLinear(nn.Module):
    """
    Linear layer backed by packed ternary weights.

    Parameters
    ----------
    packed_weight:
        Packed representation of a 2-D linear weight tensor with shape

            [out_features, in_features]

    bias:
        Optional bias tensor.

    rows_per_chunk:
        Maximum number of output rows reconstructed at one time.

    Notes
    -----
    No full reconstructed weight Parameter is registered.

    packed_codes and scales are registered as buffers so they form the
    persistent representation of the layer.
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
                "PackedTernaryLinear requires a 2-D weight tensor"
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
            % packed_weight.group_size
            != 0
        ):
            raise ValueError(
                "padded_numel must be divisible by group_size"
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

            if bias.numel() != out_features:
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

        self.padded_numel = (
            packed_weight.padded_numel
        )

        self.group_size = (
            packed_weight.group_size
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
    def num_groups(self) -> int:
        return (
            self.padded_numel
            // self.group_size
        )

    @property
    def persistent_weight_bytes(
        self,
    ) -> int:
        """
        Persistent packed weight payload only.

        Includes:
            packed codes
            +
            reconstruction scales

        Bias is intentionally excluded.
        """

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

        if (
            self.persistent_weight_bytes
            == 0
        ):
            return 0.0

        return (
            self.fp16_equivalent_weight_bytes
            / self.persistent_weight_bytes
        )

    def _unpack_codes(
        self,
    ) -> torch.Tensor:
        """
        Decode the persistent 2-bit representation.

        The frozen packing implementation intentionally performs
        unpacking on CPU, so the returned int8 codes are CPU-resident.
        """

        codes = (
            unpack_ternary_codes(
                self.packed_codes,
                self.padded_numel,
            )
        )

        if (
            codes.numel()
            != self.padded_numel
        ):
            raise RuntimeError(
                "Unpacked code count mismatch"
            )

        return codes

    def _reconstruct_flat_range(
        self,
        codes: torch.Tensor,
        start: int,
        end: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Reconstruct flattened weight values in [start, end).

        The requested range may begin/end inside quantization groups.

        Exact group IDs are calculated from the global flattened
        positions so the frozen group boundaries are preserved.
        """

        if start < 0:
            raise ValueError(
                "start cannot be negative"
            )

        if end < start:
            raise ValueError(
                "end cannot be smaller than start"
            )

        if end > self.original_numel:
            raise ValueError(
                "end exceeds original_numel"
            )

        if start == end:
            return torch.empty(
                0,
                device=device,
                dtype=dtype,
            )

        first_group = (
            start
            // self.group_size
        )

        last_group = (
            (end - 1)
            // self.group_size
        )

        group_start = (
            first_group
            * self.group_size
        )

        group_end = (
            (last_group + 1)
            * self.group_size
        )

        group_codes = (
            codes[
                group_start:group_end
            ]
            .to(
                device=device,
                dtype=torch.float32,
            )
            .reshape(
                -1,
                self.group_size,
            )
        )

        group_scales = (
            self.scales[
                first_group:
                last_group + 1
            ]
            .to(
                device=device,
                dtype=torch.float32,
            )
            .reshape(
                -1,
                1,
            )
        )

        reconstructed_groups = (
            group_codes
            * group_scales
        )

        reconstructed_flat = (
            reconstructed_groups
            .reshape(-1)
        )

        local_start = (
            start
            - group_start
        )

        local_end = (
            local_start
            + (end - start)
        )

        return (
            reconstructed_flat[
                local_start:local_end
            ]
            .to(dtype=dtype)
            .contiguous()
        )

    def forward(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:
        """
        Execute packed ternary linear inference.

        A complete reconstructed weight matrix is never stored as a
        persistent parameter or buffer.

        Only rows_per_chunk output rows are reconstructed at once.
        """

        if not isinstance(
            input,
            torch.Tensor,
        ):
            raise TypeError(
                "input must be a torch.Tensor"
            )

        if input.ndim < 1:
            raise ValueError(
                "input must have at least one dimension"
            )

        if (
            input.shape[-1]
            != self.in_features
        ):
            raise ValueError(
                "Input feature dimension mismatch: "
                f"expected {self.in_features}, "
                f"got {input.shape[-1]}"
            )

        # ----------------------------------------------------------
        # Current codec intentionally unpacks on CPU.
        #
        # Codes are decoded once per forward call. Future optimized
        # kernels can replace this behavior without changing the
        # frozen packed checkpoint representation.
        # ----------------------------------------------------------

        codes = (
            self._unpack_codes()
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

            flat_start = (
                row_start
                * self.in_features
            )

            flat_end = (
                row_end
                * self.in_features
            )

            chunk_flat = (
                self._reconstruct_flat_range(
                    codes=codes,
                    start=flat_start,
                    end=flat_end,
                    device=input.device,
                    dtype=self.original_dtype,
                )
            )

            chunk_weight = (
                chunk_flat.reshape(
                    row_end - row_start,
                    self.in_features,
                )
            )

            chunk_bias = None

            if self.bias is not None:

                chunk_bias = (
                    self.bias[
                        row_start:row_end
                    ]
                    .to(
                        device=input.device,
                        dtype=input.dtype,
                    )
                )

            # Match ordinary Linear execution dtype.
            if (
                chunk_weight.dtype
                != input.dtype
            ):
                chunk_weight = (
                    chunk_weight.to(
                        dtype=input.dtype
                    )
                )

            chunk_output = F.linear(
                input,
                chunk_weight,
                chunk_bias,
            )

            outputs.append(
                chunk_output
            )

        return torch.cat(
            outputs,
            dim=-1,
        )
