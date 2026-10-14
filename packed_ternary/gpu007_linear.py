from packed_ternary.gpu004_linear import CUDAPackedTernaryLinear
from packed_ternary.gpu007_extension import load_gpu007_extension


class GPU007ParallelCUDAPackedTernaryLinear(CUDAPackedTernaryLinear):
    """GPU007 wrapper using only the parallel packed CUDA operator."""

    def _get_extension(self):
        if self._extension is None:
            self._extension = load_gpu007_extension(verbose=False)
        return self._extension
