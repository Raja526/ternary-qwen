from pathlib import Path

from torch.utils.cpp_extension import load


_EXTENSION = None


def load_opt003_extension(
    verbose=False,
):
    global _EXTENSION

    if _EXTENSION is not None:
        return _EXTENSION

    root = Path(__file__).resolve().parent

    source = (
        root
        / "csrc"
        / "packed_ternary_cpu_parallel.cpp"
    )

    _EXTENSION = load(
        name="opt003_packed_ternary_cpu_parallel",
        sources=[str(source)],
        extra_cflags=[
            "-O2",
        ],
        verbose=verbose,
    )

    return _EXTENSION
