"""
OPT010 AVX2 packed ternary linear layer.

Controlled difference from frozen OPT008:
    at::parallel_for grain size = 16.

All Python-side runtime behavior is inherited unchanged
from TableDispatchPackedTernaryLinear.
"""

from packed_ternary.table_dispatch_linear import (
    TableDispatchPackedTernaryLinear,
)

from packed_ternary.opt010_grain16_extension import (
    load_opt008_extension as load_opt010_extension,
)


class Grain16AVX2PackedTernaryLinear(
    TableDispatchPackedTernaryLinear
):
    """
    OPT010 grain-16 runtime wrapper.

    Only _get_extension() differs from frozen OPT008.
    """

    def _get_extension(self):
        if self._extension is None:
            self._extension = (
                load_opt010_extension(
                    verbose=False
                )
            )

        return self._extension
