---
license: apache-2.0
base_model: Qwen/Qwen3.5-0.8B
tags:
  - qwen
  - ternary-quantization
  - quantization
  - model-compression
  - llm-inference
  - avx2
  - pytorch
---

# Qwen3.5-0.8B Selected-Matrix Packed Ternary Runtime

This is an independent research artifact for running `Qwen/Qwen3.5-0.8B`
with three selected weight matrices stored in a G128 packed ternary format
and executed by the retained OPT008 AVX2 CPU runtime.

This is **not** a fully ternarized Qwen model and **not** a whole-model
2-bit checkpoint. The base model remains the upstream
`Qwen/Qwen3.5-0.8B` representation. This repository contains only the
selected packed artifact and runtime code; users must obtain the base model
separately.

## What is modified

Exactly these three matrices are replaced at runtime:

| Module | Shape | Parameters |
|---|---:|---:|
| `model.layers.3.self_attn.k_proj` | `(512, 1024)` | `524,288` |
| `model.layers.4.linear_attn.out_proj` | `(1024, 2048)` | `2,097,152` |
| `model.layers.4.linear_attn.in_proj_qkv` | `(6144, 1024)` | `6,291,456` |

Total selected parameters: **8,912,896**.

All other model parameters remain in the upstream/base model.

## Packed format and storage

The artifact uses optimized group-wise ternarization with group size 128
and a threshold multiplier of 0.50. Ternary values are encoded using two
bits per value:

```text
-1 -> 00
 0 -> 01
+1 -> 10
11 -> reserved
```

Each group has one FP16 reconstruction scale. The packed selected-weight
payload is 2,367,488 bytes, compared with 17,825,792 FP16-equivalent bytes.
This is **7.529412x compression for the three selected matrices only**;
it is not a whole-model compression ratio.

The checkpoint is `ternary_selected_g128_v1.pt`. It uses the project’s
version-1 PyTorch checkpoint format and contains packed codes, scales, and
metadata, not original Qwen FP16 weights.

SHA256:

```text
8e35aaea4e47a68363a8951fa425874a1e35929a19c277335a260bb20a9cb546
```

## CPU implementation

The retained CPU implementation is **OPT008 explicit AVX2 packed-code
decode**. The loader replaces only the three selected `nn.Linear` modules
and does not silently fall back to ordinary FP16 for those modules.

The frozen research record reports two separate CPU protocols:

- Original OPT008 mean latency change: **-8.307%**.
- Fresh final validation: **-3.393% mean**, **-2.332% median**.

These are protocol-specific measurements, not a universal CPU speedup
claim. Results depend on hardware, software versions, thread settings,
prompt shape, and workload.

CUDA was experimentally implemented through GPU003/GPU004, but packed CUDA
was slower than FP16 on the tested NVIDIA Quadro P1000. CUDA code is not
included in this initial CPU-first package.

## Installation

Install a CPU-capable PyTorch build appropriate for your platform, then:

```bash
pip install -r requirements-cpu.txt
```

The first execution compiles the small OPT008 C++ extension through
PyTorch’s extension mechanism. An AVX2-capable CPU and a working C++
compiler are required.

## CPU usage

From this directory:

```bash
python infer_cpu.py \
  --prompt "What is 2 + 2?" \
  --max-new-tokens 50 \
  --timing
```

The command downloads/loads the tokenizer and the original base model from
`Qwen/Qwen3.5-0.8B`, verifies the packed artifact hash, installs OPT008 for
the three selected modules, and generates a response. The upstream base
model is never included in this repository.

For a different artifact location:

```bash
python infer_cpu.py \
  --checkpoint /path/to/ternary_selected_g128_v1.pt \
  --model-id Qwen/Qwen3.5-0.8B \
  --prompt "Explain packed ternary weights." \
  --max-new-tokens 50
```

## Limitations and intended use

This package is intended for research, inspection, and experimentation with
selected-matrix packed ternary inference. It is not presented as a general
deployment-ready quantized Qwen release.

Out of scope:

- claiming that the whole model is ternary or 2-bit;
- claiming universal CPU or GPU acceleration;
- claiming universal memory or VRAM reduction;
- using the artifact without the compatible upstream base model;
- treating the selected packed matrices as the original FP16 weights;
- assuming compatibility with arbitrary Qwen model revisions or hardware.

## Reproducibility

This package corresponds to:

- research freeze: `8af66b4621dcff440f862a3b1c13af24577e8396`
- GitHub release commit: `ad66654dce5e7fdc695f6ae2305d789e40292ec3`
- GitHub tag: `v1.0`

The source project and frozen research record are available at:

`https://github.com/Raja526/ternary-qwen`

The artifact was copied from the frozen ENG003 checkpoint without
regeneration.

## License and attribution

The independent source and packaging work is licensed under the Apache
License 2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE).

The base model is `Qwen/Qwen3.5-0.8B`, developed by the Qwen Team. Obtain
the base model from [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B)
and follow its applicable license and terms. This is not an official Qwen
or Alibaba project and does not imply endorsement, affiliation, or
sponsorship.

When citing this artifact, cite the GitHub project and identify the
selected-matrix packed ternary runtime, the base model, and the frozen
research commits listed above.
