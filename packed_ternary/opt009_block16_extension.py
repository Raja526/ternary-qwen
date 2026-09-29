from pathlib import Path

from torch.utils.cpp_extension import load


_EXTENSION = None


def load_opt009_extension(
    verbose=False,
):
    global _EXTENSION

    if _EXTENSION is not None:
        return _EXTENSION

    root = Path(__file__).resolve().parent

    source = (
        root
        / "csrc"
        / "packed_ternary_cpu_avx2_blocked16.cpp"
    )

    _EXTENSION = load(
        name="opt009_packed_ternary_cpu_avx2_blocked16",
        sources=[str(source)],
        extra_cflags=[
            "-O2",
            "-mavx2",
            "-mno-fma",
        ],
        verbose=verbose,
    )

    return _EXTENSION
