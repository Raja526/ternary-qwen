---
license: apache-2.0
base_model: Qwen/Qwen3.5-0.8B
tags:
  - qwen
  - ternary-quantization
  - model-compression
  - llm-inference
  - avx2
  - cpu
  - transformers
---

# Ternary Qwen v1.1 — selected-matrix Transformers runtime

This repository is a CPU-first Hugging Face Transformers integration for
`Qwen/Qwen3.5-0.8B`. It contains a packed ternary artifact for exactly three
selected matrices and a thin remote-code AutoClass factory. It does **not**
contain the complete Qwen model.

## Preferred usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Raja526/qwen3.5-0.8b-selected-matrices-packed-ternary"

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True,
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
)

inputs = tokenizer("The capital of France is", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=16)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

The loader obtains the full base model from `Qwen/Qwen3.5-0.8B`, verifies the
packed artifact, and replaces only the three selected modules with the
retained `AVX2PackedTernaryLinear` implementation. The upstream model is
therefore downloaded at runtime and is not redistributed here.

## Exact scope

Selected modules:

```text
model.layers.3.self_attn.k_proj
model.layers.4.linear_attn.out_proj
model.layers.4.linear_attn.in_proj_qkv
```

Selected parameters: **8,912,896**.

The G128 packed artifact is `ternary_selected_g128_v1.pt` with SHA256:

```text
8e35aaea4e47a68363a8951fa425874a1e35929a19c277335a260bb20a9cb546
```

The selected-weight storage compression is **7.529412x**. This is only the
storage ratio for the three selected matrices; this is not whole-model 2-bit
compression and the other Qwen parameters remain in the upstream model.

## CPU and compatibility

The retained runtime is OPT008 explicit AVX2 packed-code execution. An AVX2
x86-64 CPU and a working C++ compiler are required. The first model load may
compile the small CPU extension. The integration is CPU-first, rejects
unsupported device maps and dtypes, and does not include CUDA code.

The candidate targets `transformers==5.19.0`, a stable release with native
Qwen3.5 support, Python 3.10+, and PyTorch 2.4 or newer. The upstream base
revision is pinned to:

```text
2fc06364715b967f1860aea9cf38778875588b17
```

## Security

`trust_remote_code=True` executes Python code from this repository on the
local machine. Inspect and pin the Hub repository revision for
security-sensitive or reproducible use. The code explicitly loads the
upstream identifier `Qwen/Qwen3.5-0.8B`; it never recursively loads this
repository as the base model.

The packed artifact is SHA256-verified before use. A mismatch, missing target,
shape mismatch, AVX2 failure, compiler failure, or incomplete replacement
causes a hard error. There is no silent FP16 fallback for selected modules.

## Scientific scope

The historical CPU results are protocol-specific and hardware-dependent.
They are not a universal speedup claim. This v1.1 package does not include
the CUDA implementation and makes no CUDA performance claim.

Qwen3.5-0.8B is developed by the Qwen Team and remains subject to its
applicable license and terms. This project is independent and is not an
official Qwen or Alibaba project; it does not imply endorsement, affiliation,
collaboration, or sponsorship.
