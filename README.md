# Ternary Qwen: Packed 2-Bit Ternary Inference on CPU and CUDA

This project investigates whether selected Qwen3.5-0.8B weights can be
ternarized, stored in a packed 2-bit representation, and executed directly
on CPU and CUDA without persistently reconstructing those selected weights
to FP16.

It is a research and engineering project—not a claim that ternary inference
universally outperforms FP16. The results are specific to the selected
matrices, retained implementations, protocols, and tested hardware.

## Key results

| Measurement | Result |
|---|---:|
| Selected parameters | 8,912,896 |
| FP16-equivalent selected-weight storage | 17,825,792 bytes |
| Packed selected-weight storage | 2,367,488 bytes |
| Selected-weight compression | 7.529412x |
| Retained CPU implementation | OPT008 AVX2 |
| Original CPU mean latency change | -8.307% |
| Fresh CPU validation | mean -3.393%; median -2.332% |
| Retained CUDA implementation | GPU003/GPU004 |

GPU008 full-model validation on the tested Quadro P1000:

| Prompt length | FP16 | Packed | Change |
|---:|---:|---:|---:|
| 8 tokens | 114.376822 ms | 117.373794 ms | +2.620% |
| 22 tokens | 127.558327 ms | 137.117991 ms | +7.494% |
| 47 tokens | 179.283418 ms | 198.708732 ms | +10.835% |

On CPU, a negative percentage means lower packed latency under the relevant
protocol. On GPU, a positive percentage means packed inference was slower.
The CPU measurements above are separate protocol-specific results and are
not averaged.

## Architecture / pipeline

```text
Qwen3.5-0.8B
      |
selected matrices
      |
G128 ternary quantization
      |
{-1, 0, +1}
      |
2-bit packing
      |
      +----------------------+----------------------+
      |                                             |
     CPU                                           CUDA
  OPT008 AVX2                              GPU003/GPU004
 packed decode                              packed kernel
```

## Frozen selected matrices

The frozen selected matrices are:

- `model.layers.3.self_attn.k_proj`
- `model.layers.4.linear_attn.out_proj`
- `model.layers.4.linear_attn.in_proj_qkv`

Only these selected matrices are covered by the 7.529412x storage result.
It is not a whole-model compression ratio.

## Packed representation

The representation uses group size 128 (G128):

```text
-1 -> 00
 0 -> 01
+1 -> 10
11 -> reserved/invalid
```

## CPU result

The retained CPU implementation is OPT008, using explicit AVX2 packed-code
decode. The original frozen OPT008 benchmark measured a mean latency change
of **-8.307%** relative to OPT006 under its documented protocol.

Fresh final CPU validation measured **-3.393% mean** and **-2.332% median**.
These results came from a different validation protocol, so they are
preserved separately rather than averaged. They demonstrate a
protocol-specific result, not a universal CPU speedup.

## CUDA investigation

The CUDA progression was:

- **GPU003:** correctness-first packed CUDA operator.
- **GPU004:** full-model integration.
- **GPU005:** end-to-end performance characterization.
- **GPU006:** bottleneck characterization.
- **GPU007:** one controlled parallel-K optimization; correct, but rejected
  because it was slower.
- **GPU008:** fresh retained-runtime validation.

GPU006 identified `model.layers.4.linear_attn.in_proj_qkv` as the dominant
measured bottleneck:

| Runtime | Latency |
|---|---:|
| FP16 | 0.465736 ms |
| GPU003 packed | 8.300687 ms |
| Change | +1682.274% |

For the same dominant target, GPU007 compared as follows:

| Runtime | Latency |
|---|---:|
| GPU003 | 8.262222 ms |
| GPU007 | 26.522645 ms |
| Change | +221.011% |

GPU007 was rejected because the parallel-K strategy, although correct under
the predeclared tolerance, was substantially slower for the target and
slower for all three tested matrices. GPU003 integrated through GPU004 is
the retained CUDA runtime.

The negative CUDA result is an important part of the record: GPU008 was
slower than FP16 at all three tested prompt lengths.

## What the project demonstrates

- Selected-weight storage compression.
- Direct packed execution without persistent FP16 selected weights.
- CPU packed arithmetic validation.
- CUDA packed arithmetic validation.
- Protocol-specific CPU latency improvement.
- CUDA bottleneck identification.
- Evidence-driven rejection of a slower optimization.

## What it does not claim

This project does not claim:

- universal ternary GPU speedup;
- universal CPU speedup;
- whole-model 7.529412x compression;
- universal VRAM reduction;
- superiority on modern datacenter GPUs;
- FP16/ternary model bit-exactness;
- deployment speedup from numerical ternarization alone.

## Hardware/software environment

The frozen GPU measurements were collected on:

| Component | Environment |
|---|---|
| GPU | NVIDIA Quadro P1000, 4 GB, compute capability 6.1 |
| PyTorch | 2.14.0+cu126 |
| PyTorch CUDA | 12.6 |
| CUDA compiler | 12.6 |

GPU results are specific to this hardware and software environment and
should not be assumed to generalize beyond it.

## Repository organization

- `ternary/` — ternary quantization and related scientific code.
- `packed_ternary/` — packed representation and retained CPU/CUDA runtime
  implementations.
- `scripts/` — project scripts used by the frozen record.
- `results/` — scientific, engineering, optimization, validation, and
  reproducibility artifacts.

## Reproducibility

The authoritative frozen artifacts are:

- [Final project report](results/FINAL_PROJECT_REPORT.md)
- [Final reproducibility manifest](results/FINAL_REPRODUCIBILITY_MANIFEST.json)
- [Final validation record](results/FINAL_VALIDATION.txt)

Final frozen research commit:

```text
8af66b4621dcff440f862a3b1c13af24577e8396
```

Checkpoint SHA-256:

```text
8e35aaea4e47a68363a8951fa425874a1e35929a19c277335a260bb20a9cb546
```

The repository does not promise one-command reproduction; measurements in
the frozen record were preserved and were not regenerated for this release.

## Research progression

- **EXP001–EXP026:** scientific ternary investigation.
- **ENG001–ENG013:** packed representation and runtime engineering.
- **OPT001–OPT010:** CPU optimization.
- **GPU001–GPU008:** CUDA implementation, profiling, optimization attempt,
  and validation.

## Limitations and future work

Potential future directions include optimized kernels for newer GPU
architectures, broader matrix coverage, fused packed ternary operations,
evaluation on larger models, and CPU architecture evaluation beyond the
tested environment. None of these directions is claimed as demonstrated by
the current frozen record.

## Citation

If you use this work, please cite or reference this GitHub repository and
the associated frozen research record.

## Acknowledgements

Qwen Team — [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B).

## License

This repository's independent research and engineering work is licensed
under the Apache License 2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE).
Qwen3.5-0.8B is an upstream dependency/base model developed by the Qwen
Team and licensed under the Apache License 2.0. Review the upstream model's
license when using or redistributing model artifacts. This project does not
claim ownership of Qwen or imply that it is an official Qwen or Alibaba
project.
