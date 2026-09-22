# Experiment 001A — Qwen3.5-0.8B CPU FP32 Baseline

## Environment

- Model: Qwen/Qwen3.5-0.8B
- Device: CPU
- Precision: FP32
- PyTorch: 2.14.0+cu126
- CUDA Runtime: 12.6
- GPU available: NVIDIA Quadro P1000
- GPU Compute Capability: 6.1

## Model

- Parameters: 752,393,024
- Parameter Memory: 2.803 GB
- Load Time: 142.15 seconds

## Generation Test

Prompt:

> Explain in simple terms why the sky is blue.

Input Tokens: 11

Generated Tokens: 50

Generation Time: 7.00 seconds

Throughput: 7.14 tokens/second

## Result

The original Qwen3.5-0.8B model loaded successfully and generated a
coherent response.

This experiment establishes the FP32 CPU reference baseline for future
ternary-weight experiments.

## Runtime Notes

Transformers used reference PyTorch implementations because the following
optional optimized packages were not installed:

- causal_conv1d
- flash-linear-attention

These optimizations were intentionally not installed for the initial
baseline.
