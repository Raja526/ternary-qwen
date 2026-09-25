from pathlib import Path

from torch.utils.cpp_extension import load


_THIS_DIR = Path(__file__).resolve().parent

_SOURCE = (
    _THIS_DIR
    / "csrc"
    / "packed_ternary_cpu.cpp"
)


def load_opt001_extension(verbose: bool = False):
    """
    Build/load the OPT001 CPU packed-ternary extension.

    This is intentionally isolated from the frozen ENG runtime.
    """

    if not _SOURCE.exists():
        raise FileNotFoundError(
            f"OPT001 C++ source not found: {_SOURCE}"
        )

    return load(
        name="opt001_packed_ternary_cpu",
        sources=[str(_SOURCE)],
        extra_cflags=[
            "-O2",
        ],
        verbose=verbose,
    )
