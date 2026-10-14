"""Thin Transformers AutoClass factory for the selected-matrix runtime."""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from transformers import Qwen3_5ForCausalLM

from .packed_ternary_runtime import AVX2PackedTernaryLinear
from .packed_ternary_runtime import (
    load_avx2_runtime_modules,
)


BASE_MODEL_ID = "Qwen/Qwen3.5-0.8B"
BASE_MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
RELEASE_REPOSITORY_ID = (
    "Raja526/qwen3.5-0.8b-selected-matrices-packed-ternary"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "8e35aaea4e47a68363a8951fa425874a1e35929a19c277335a260bb20a9cb546"
)
EXPECTED_SELECTED_PARAMETER_COUNT = 8_912_896
TARGET_MODULES = (
    "model.layers.3.self_attn.k_proj",
    "model.layers.4.linear_attn.out_proj",
    "model.layers.4.linear_attn.in_proj_qkv",
)
MIN_TRANSFORMERS_VERSION = (5, 2, 0)


def _version_tuple(version: str) -> tuple[int, int, int]:
    values = []
    for part in version.split(".")[:3]:
        digits = "".join(character for character in part if character.isdigit())
        values.append(int(digits or 0))
    return tuple((values + [0, 0, 0])[:3])


def _require_transformers_version() -> None:
    import transformers

    if _version_tuple(transformers.__version__) < MIN_TRANSFORMERS_VERSION:
        raise RuntimeError(
            "Ternary Qwen v1.1 requires Transformers >= 5.2.0; "
            f"found {transformers.__version__}."
        )


def _require_avx2() -> None:
    machine = platform.machine().lower()
    if machine not in {"x86_64", "amd64", "x86-64"}:
        raise RuntimeError(
            "The OPT008 runtime requires an x86-64 CPU with AVX2. "
            f"Detected architecture: {machine or 'unknown'}."
        )

    system = platform.system()
    if system == "Linux":
        flags = Path("/proc/cpuinfo").read_text(errors="replace").lower()
        if "avx2" not in flags:
            raise RuntimeError("The OPT008 runtime requires CPU AVX2 support.")
        return

    if system == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.optional.avx2_0"],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(
                "Unable to verify AVX2 support on macOS."
            ) from exc
        if result.stdout.strip() != "1":
            raise RuntimeError("The OPT008 runtime requires CPU AVX2 support.")
        return

    raise RuntimeError(
        f"Unable to verify AVX2 support on {system}; refusing to continue."
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_checkpoint(source: str | os.PathLike[str], kwargs: dict[str, Any]) -> Path:
    source_path = Path(source)
    if source_path.is_dir():
        checkpoint = source_path / "ternary_selected_g128_v1.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Packed artifact not found: {checkpoint}")
        return checkpoint

    hub_kwargs = {
        key: kwargs[key]
        for key in (
            "cache_dir",
            "force_download",
            "local_files_only",
            "proxies",
            "token",
        )
        if key in kwargs
    }
    hub_kwargs["revision"] = kwargs.get("revision")
    return Path(
        hf_hub_download(
            repo_id=str(source),
            filename="ternary_selected_g128_v1.pt",
            **hub_kwargs,
        )
    )


def _validate_artifact(checkpoint: Path) -> None:
    actual = _sha256_file(checkpoint)
    if actual != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(
            "Packed artifact SHA256 mismatch: "
            f"expected {EXPECTED_CHECKPOINT_SHA256}, got {actual}."
        )


def _base_model_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    requested_dtype = kwargs.pop("dtype", kwargs.pop("torch_dtype", None))
    if requested_dtype not in (None, torch.float16, "float16"):
        raise ValueError(
            "Ternary Qwen v1.1 requires torch.float16 because the retained "
            "OPT008 runtime accepts FP16 activations only."
        )

    device_map = kwargs.pop("device_map", None)
    if device_map is not None:
        raise ValueError(
            "Ternary Qwen v1.1 is CPU-first and does not support device_map."
        )

    for key in (
        "trust_remote_code",
        "config",
        "state_dict",
        "output_loading_info",
        "adapter_kwargs",
        "_from_auto",
        "_from_pipeline",
        "revision",
        "code_revision",
    ):
        kwargs.pop(key, None)

    allowed = {
        "cache_dir",
        "force_download",
        "local_files_only",
        "proxies",
        "token",
        "low_cpu_mem_usage",
        "use_safetensors",
        "weights_only",
        "offload_state_dict",
    }
    result = {key: value for key, value in kwargs.items() if key in allowed}
    result["revision"] = BASE_MODEL_REVISION
    result["dtype"] = torch.float16
    result["trust_remote_code"] = False
    return result


def _validate_replacements(model: torch.nn.Module, load_result: Any) -> None:
    if tuple(load_result.loaded_module_names) != TARGET_MODULES:
        raise RuntimeError(
            "OPT008 did not replace exactly the three frozen target modules: "
            f"{load_result.loaded_module_names!r}"
        )
    if load_result.replaced_matrix_count != 3:
        raise RuntimeError("OPT008 replacement count is not exactly three.")
    if load_result.replaced_parameter_count != EXPECTED_SELECTED_PARAMETER_COUNT:
        raise RuntimeError(
            "OPT008 selected parameter count mismatch: "
            f"{load_result.replaced_parameter_count}"
        )
    modules = dict(model.named_modules())
    for name in TARGET_MODULES:
        if not isinstance(modules.get(name), AVX2PackedTernaryLinear):
            raise RuntimeError(
                f"Selected target {name} is not AVX2PackedTernaryLinear."
            )


class TernaryQwenForCausalLM(Qwen3_5ForCausalLM):
    """AutoClass entry point returning a patched native Qwen3.5 model."""

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        _require_transformers_version()

        if str(pretrained_model_name_or_path) == BASE_MODEL_ID:
            raise RuntimeError(
                "TernaryQwenForCausalLM cannot be used as the upstream loader; "
                "the custom repository must not self-reference the base model."
            )
        if BASE_MODEL_ID == RELEASE_REPOSITORY_ID:
            raise RuntimeError("Base and release repository identifiers collide.")

        _require_avx2()
        kwargs = dict(kwargs)
        output_loading_info = bool(kwargs.get("output_loading_info", False))
        release_revision = kwargs.get("revision") or "main"
        source_path = Path(pretrained_model_name_or_path)
        if source_path.is_dir():
            os.environ["TERNARY_QWEN_LOCAL_RUNTIME_SOURCE"] = str(
                source_path / "packed_ternary" / "csrc" / "packed_ternary_cpu_avx2.cpp"
            )
        checkpoint = _resolve_checkpoint(pretrained_model_name_or_path, kwargs)
        _validate_artifact(checkpoint)

        base_kwargs = _base_model_kwargs(kwargs)
        model = Qwen3_5ForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            *model_args,
            **base_kwargs,
        )
        if any(parameter.device.type != "cpu" for parameter in model.parameters()):
            raise RuntimeError("The native Qwen model was not loaded on CPU.")

        os.environ["TERNARY_QWEN_HF_REVISION"] = str(release_revision)
        load_result = load_avx2_runtime_modules(
            model=model,
            checkpoint_path=checkpoint,
            selected_module_names=TARGET_MODULES,
            expected_model_name=BASE_MODEL_ID,
        )
        _validate_replacements(model, load_result)

        if output_loading_info:
            return model, {
                "packed_checkpoint": str(checkpoint),
                "replaced_modules": TARGET_MODULES,
            }
        return model


__all__ = ["TernaryQwenForCausalLM"]
