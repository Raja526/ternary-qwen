# Experiment 001B — Qwen3.5-0.8B GPU FP16 Baseline

## Environment

- Model: Qwen/Qwen3.5-0.8B
- Device: NVIDIA Quadro P1000
- VRAM: 3.94 GB
- Compute Capability: 6.1
- Precision: FP16
- PyTorch: 2.14.0+cu126
- CUDA Runtime: 12.6

## Model

- Parameters: 752,393,024
- Parameter Memory: 1.401 GB
- Model GPU Allocation: 1459.10 MB
- GPU Reserved After Load: 1530.00 MB
- Load Time: 2.97 seconds

## Generation Test

Prompt:

> Explain in simple terms why the sky is blue.

- Input Tokens: 11
- Generated Tokens: 50
- Generation Time: 5.55 seconds
- Throughput: 9.01 tokens/second
- Peak GPU Memory: 1495.68 MB
- Current GPU Allocation: 1467.22 MB
- Current GPU Reserved: 1558.00 MB

## Result

Qwen3.5-0.8B successfully ran in FP16 on the NVIDIA Quadro P1000.

The generated response was coherent and matched the qualitative behavior
observed in the CPU FP32 baseline.

## Baseline Comparison

| Metric | CPU FP32 | GPU FP16 |
|---|---:|---:|
| Parameter Memory | 2.803 GB | 1.401 GB |
| Generation Time | 7.00 s | 5.55 s |
| Throughput | 7.14 tok/s | 9.01 tok/s |
| Peak GPU Memory | N/A | 1495.68 MB |

## Runtime Notes

Transformers used reference PyTorch implementations because the optional
causal_conv1d and flash-linear-attention optimized packages were not installed.

These optimizations remain disabled to preserve a stable initial baseline.
