"""
OPT008 explicit AVX2 packed ternary linear layer.

Controlled difference from frozen OPT006:
    extension implementation only.

Persistent packed representation, validation,
forward shape handling, scale layout, bias handling,
and operator arguments are inherited unchanged from
TableDispatchPackedTernaryLinear.
"""

from packed_ternary.table_dispatch_linear import (
    TableDispatchPackedTernaryLinear,
)

from packed_ternary.opt008_extension import (
    load_opt008_extension,
)


class AVX2PackedTernaryLinear(
    TableDispatchPackedTernaryLinear
):
    """
    OPT008 runtime wrapper.

    Only _get_extension() differs from frozen OPT006.
    """

    def _get_extension(self):
        if self._extension is None:
            self._extension = (
                load_opt008_extension(
                    verbose=False
                )
            )

        return self._extension
