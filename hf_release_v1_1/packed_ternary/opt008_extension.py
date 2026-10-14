import os
from pathlib import Path

from torch.utils.cpp_extension import load


_EXTENSION = None

_REMOTE_REPOSITORY = (
    "Raja526/qwen3.5-0.8b-selected-matrices-packed-ternary"
)
_SOURCE_NAME = "packed_ternary/csrc/packed_ternary_cpu_avx2.cpp"


def load_opt008_extension(
    verbose=False,
):
    global _EXTENSION

    if _EXTENSION is not None:
        return _EXTENSION

    root = Path(__file__).resolve().parent

    source = (
        root
        / "csrc"
        / "packed_ternary_cpu_avx2.cpp"
    )

    if not source.exists():
        from huggingface_hub import hf_hub_download

        source = Path(
            hf_hub_download(
                repo_id=_REMOTE_REPOSITORY,
                filename=_SOURCE_NAME,
                revision=os.environ.get(
                    "TERNARY_QWEN_HF_REVISION"
                ),
                local_files_only=(
                    os.environ.get("HF_HUB_OFFLINE") == "1"
                ),
            )
        )

    _EXTENSION = load(
        name="opt008_packed_ternary_cpu_avx2",
        sources=[str(source)],
        extra_cflags=[
            "-O2",
            "-mavx2",
            "-mno-fma",
        ],
        verbose=verbose,
    )

    return _EXTENSION
