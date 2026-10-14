from pathlib import Path

from torch.utils.cpp_extension import load


_ROOT = Path(__file__).resolve().parent
_CUDA_SOURCE = _ROOT / "csrc" / "packed_ternary_cuda_gpu007.cu"


def load_gpu007_extension(verbose: bool = False):
    if not _CUDA_SOURCE.exists():
        raise FileNotFoundError(f"GPU007 CUDA source not found: {_CUDA_SOURCE}")

    return load(
        name="gpu007_parallel_packed_ternary_cuda",
        sources=[str(_CUDA_SOURCE)],
        extra_cuda_cflags=["-gencode=arch=compute_61,code=sm_61"],
        verbose=verbose,
    )
