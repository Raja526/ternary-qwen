from pathlib import Path

from torch.utils.cpp_extension import load


_EXTENSION = None


def load_opt005_extension(
    verbose=False,
):
    global _EXTENSION

    if _EXTENSION is not None:
        return _EXTENSION

    root = Path(__file__).resolve().parent

    source = (
        root
        / "csrc"
        / "packed_ternary_cpu_branch_reduced.cpp"
    )

    _EXTENSION = load(
        name="opt005_packed_ternary_cpu_branch_reduced",
        sources=[str(source)],
        extra_cflags=[
            "-O2",
        ],
        verbose=verbose,
    )

    return _EXTENSION
