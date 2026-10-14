"""Single-file OPT008 runtime used by the Transformers dynamic-module cache.

This is a packaging form of the retained CPU runtime. It intentionally has no
relative Python imports so Transformers can copy it as one remote-code file.
The packed representation and AVX2 operator call are unchanged.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load


@dataclass(frozen=True)
class PackedTernaryWeight:
    packed_codes: torch.Tensor
    scales: torch.Tensor
    original_shape: Tuple[int, ...]
    original_numel: int
    padded_numel: int
    group_size: int
    original_dtype: torch.dtype


def _string_to_dtype(name: str) -> torch.dtype:
    values = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    if name not in values:
        raise ValueError(f"Unsupported serialized dtype: {name}")
    return values[name]


def load_packed_checkpoint(path):
    payload = torch.load(
        Path(path),
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload must be a dictionary")
    required = {
        "format_version",
        "model_name",
        "group_size",
        "threshold_multiplier",
        "weights",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Checkpoint is missing keys: {sorted(missing)}")
    if payload["format_version"] != 1:
        raise ValueError(f"Unsupported checkpoint format version: {payload['format_version']}")
    if not isinstance(payload["weights"], dict) or not payload["weights"]:
        raise ValueError("Serialized weights must be a non-empty dictionary")

    restored = {}
    group_size = int(payload["group_size"])
    for name, item in payload["weights"].items():
        required_weight = {
            "packed_codes",
            "scales",
            "original_shape",
            "original_numel",
            "padded_numel",
            "group_size",
            "original_dtype",
        }
        missing_weight = required_weight - set(item)
        if missing_weight:
            raise ValueError(f"{name} is missing keys: {sorted(missing_weight)}")
        packed_codes = item["packed_codes"]
        scales = item["scales"]
        shape = tuple(int(value) for value in item["original_shape"])
        original_numel = int(item["original_numel"])
        padded_numel = int(item["padded_numel"])
        item_group_size = int(item["group_size"])
        if not isinstance(packed_codes, torch.Tensor) or packed_codes.dtype != torch.uint8:
            raise TypeError(f"{name}: packed_codes must be torch.uint8")
        if not isinstance(scales, torch.Tensor):
            raise TypeError(f"{name}: scales must be a tensor")
        shape_numel = 1
        for dimension in shape:
            shape_numel *= dimension
        if shape_numel != original_numel:
            raise ValueError(f"{name}: shape/numel mismatch")
        if item_group_size != group_size or padded_numel < original_numel:
            raise ValueError(f"{name}: invalid packed group metadata")
        if padded_numel % group_size:
            raise ValueError(f"{name}: padded_numel is not group aligned")
        if scales.numel() != padded_numel // group_size:
            raise ValueError(f"{name}: scale count mismatch")
        if packed_codes.numel() != (padded_numel + 3) // 4:
            raise ValueError(f"{name}: packed-code byte count mismatch")
        restored[name] = PackedTernaryWeight(
            packed_codes=packed_codes.reshape(-1).contiguous(),
            scales=scales.reshape(-1).contiguous(),
            original_shape=shape,
            original_numel=original_numel,
            padded_numel=padded_numel,
            group_size=item_group_size,
            original_dtype=_string_to_dtype(item["original_dtype"]),
        )
    return {
        "format_version": payload["format_version"],
        "model_name": payload["model_name"],
        "group_size": group_size,
        "threshold_multiplier": float(payload["threshold_multiplier"]),
        "weight_count": len(restored),
    }, restored


_EXTENSION = None
_REMOTE_REPOSITORY = "Raja526/qwen3.5-0.8b-selected-matrices-packed-ternary"
_SOURCE_NAME = "packed_ternary/csrc/packed_ternary_cpu_avx2.cpp"


def load_opt008_extension(verbose=False):
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION
    local_source = os.environ.get("TERNARY_QWEN_LOCAL_RUNTIME_SOURCE")
    source = Path(local_source) if local_source else Path("__missing_opt008_source__")
    if not source.exists():
        source = Path(__file__).resolve().parent / "csrc" / "packed_ternary_cpu_avx2.cpp"
    if not source.exists():
        from huggingface_hub import hf_hub_download

        source = Path(
            hf_hub_download(
                repo_id=_REMOTE_REPOSITORY,
                filename=_SOURCE_NAME,
                revision=os.environ.get("TERNARY_QWEN_HF_REVISION"),
                local_files_only=os.environ.get("HF_HUB_OFFLINE") == "1",
            )
        )
    _EXTENSION = load(
        name="opt008_packed_ternary_cpu_avx2",
        sources=[str(source)],
        extra_cflags=["-O2", "-mavx2", "-mno-fma"],
        verbose=verbose,
    )
    return _EXTENSION


class TableDispatchPackedTernaryLinear(nn.Module):
    def __init__(self, packed_weight: PackedTernaryWeight, bias: Optional[torch.Tensor] = None):
        super().__init__()
        if len(packed_weight.original_shape) != 2:
            raise ValueError("Packed ternary linear requires a 2-D weight")
        self.out_features, self.in_features = map(int, packed_weight.original_shape)
        if self.out_features * self.in_features != packed_weight.original_numel:
            raise ValueError("Packed weight shape does not match original_numel")
        if packed_weight.padded_numel != packed_weight.original_numel:
            raise ValueError("OPT008 requires no group padding")
        if self.in_features % packed_weight.group_size or self.in_features % 4:
            raise ValueError("OPT008 requires row-aligned packed groups")
        if packed_weight.scales.numel() != self.out_features * (self.in_features // packed_weight.group_size):
            raise ValueError("Scale count does not match packed matrix")
        if packed_weight.packed_codes.numel() != packed_weight.original_numel // 4:
            raise ValueError("Packed-code byte count does not match matrix")
        if bias is not None and (bias.ndim != 1 or bias.numel() != self.out_features):
            raise ValueError("Bias shape does not match packed matrix")
        self.original_numel = int(packed_weight.original_numel)
        self.group_size = int(packed_weight.group_size)
        self.groups_per_row = self.in_features // self.group_size
        self.original_dtype = packed_weight.original_dtype
        self.register_buffer("packed_codes", packed_weight.packed_codes.detach().clone().contiguous())
        self.register_buffer("scales", packed_weight.scales.detach().clone().contiguous())
        if bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(bias.detach().clone(), requires_grad=False)
        self._extension = None

    @property
    def persistent_weight_bytes(self):
        return self.packed_codes.numel() * self.packed_codes.element_size() + self.scales.numel() * self.scales.element_size()

    @property
    def fp16_equivalent_weight_bytes(self):
        return self.original_numel * 2

    def _get_extension(self):
        if self._extension is None:
            self._extension = load_opt008_extension(verbose=False)
        return self._extension

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.shape[-1] != self.in_features:
            raise ValueError("Input last dimension does not match in_features")
        if input.device.type != "cpu":
            raise ValueError("OPT008 supports CPU only")
        if input.dtype != torch.float16:
            raise ValueError("OPT008 requires FP16 input")
        original_shape = tuple(input.shape)
        flat_input = input.reshape(-1, self.in_features).contiguous()
        output = self._get_extension().packed_ternary_linear_cpu_parallel(
            flat_input,
            self.packed_codes,
            self.scales.reshape(-1),
            self.out_features,
            self.group_size,
        )
        output = output.reshape(*original_shape[:-1], self.out_features)
        if self.bias is not None:
            output = output + self.bias
        return output


class AVX2PackedTernaryLinear(TableDispatchPackedTernaryLinear):
    pass


@dataclass
class AVX2RuntimeLoadResult:
    checkpoint_path: str
    model_name: str
    loaded_module_names: Tuple[str, ...]
    replaced_matrix_count: int
    replaced_parameter_count: int


def _module(model, name):
    modules = dict(model.named_modules())
    if name not in modules:
        raise KeyError(f"Checkpoint target module not found in model: {name}")
    return modules[name]


@torch.no_grad()
def load_avx2_runtime_modules(model: torch.nn.Module, checkpoint_path, selected_module_names: Iterable[str], expected_model_name: str = None):
    names = tuple(selected_module_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("Selected module names must be non-empty and unique")
    metadata, packed_weights = load_packed_checkpoint(checkpoint_path)
    if set(packed_weights) != set(names):
        raise ValueError(
            "Packed checkpoint entries do not equal the selected target set: "
            f"{tuple(packed_weights)}"
        )
    if expected_model_name is not None and metadata["model_name"] != expected_model_name:
        raise ValueError(f"Checkpoint model-name mismatch: {metadata['model_name']!r}")
    validated = []
    for name in names:
        packed = packed_weights.get(name)
        if packed is None:
            raise KeyError(f"Selected module is not present in packed checkpoint: {name}")
        module = _module(model, name)
        if not isinstance(module, nn.Linear):
            raise TypeError(f"{name} must be torch.nn.Linear before replacement")
        expected_shape = (module.out_features, module.in_features)
        if tuple(packed.original_shape) != expected_shape:
            raise ValueError(f"{name}: model={expected_shape}, checkpoint={packed.original_shape}")
        if module.weight.dtype != packed.original_dtype:
            raise TypeError(f"{name}: model dtype={module.weight.dtype}, checkpoint dtype={packed.original_dtype}")
        if packed.padded_numel != packed.original_numel or module.in_features % packed.group_size or module.in_features % 4:
            raise ValueError(f"{name}: incompatible OPT008 packed layout")
        parent_name, child_name = name.rsplit(".", 1)
        parent = _module(model, parent_name)
        bias = None if module.bias is None else module.bias.detach().clone()
        replacement = AVX2PackedTernaryLinear(packed, bias=bias).to(device=module.weight.device)
        validated.append((name, parent, child_name, replacement))
    for name, parent, child_name, replacement in validated:
        setattr(parent, child_name, replacement)
    return AVX2RuntimeLoadResult(
        checkpoint_path=str(checkpoint_path),
        model_name=metadata["model_name"],
        loaded_module_names=names,
        replaced_matrix_count=len(names),
        replaced_parameter_count=sum(packed_weights[name].original_numel for name in names),
    )


AVX2PackedTernaryLinear.__module__ = "packed_ternary.avx2_linear"
