# Final Ternary-Qwen Research Record

Status: complete

Final parent: `e9c641cc4eafc9ed3541483ec0888f234a7f1b65` (GPU008)

This is a consolidation record. It introduces no new optimization and does
not regenerate historical measurements.

## Research history

The repository preserves the frozen scientific experiment record
EXP001–EXP026, engineering record ENG001–ENG013, CPU optimization record
OPT001–OPT010 plus final CPU validation/consolidation, and GPU001–GPU008.
Historical measurements remain in their original result files. Repeated
validation measurements are reported separately when their protocols differ.

The GPU sequence is:

1. GPU001: environment baseline.
2. GPU002: CUDA 12.6 extension toolchain.
3. GPU003: isolated packed CUDA arithmetic correctness.
4. GPU004: full-model GPU003 integration.
5. GPU005: packed full-model performance characterization.
6. GPU006: operation-level bottleneck characterization.
7. GPU007: one parallel-K CUDA optimization attempt; rejected because it
   was slower for all three matrices.
8. GPU008: fresh retained-runtime full-model validation.

## Final retained implementations

Quantization is the frozen optimized groupwise G128 ternary quantizer.
Group size is 128 and the frozen threshold multiplier is 0.50.

The packed representation is unchanged:

```text
-1 = 00
 0 = 01
+1 = 10
11 = reserved/invalid
```

The selected matrices are:

1. `model.layers.3.self_attn.k_proj`
2. `model.layers.4.linear_attn.out_proj`
3. `model.layers.4.linear_attn.in_proj_qkv`

Selected parameters: 8,912,896

Checkpoint: `results/ENG003_multimatrix_checkpoint.pt`

Checkpoint SHA-256:

`8e35aaea4e47a68363a8951fa425874a1e35929a19c277335a260bb20a9cb546`

The retained CPU implementation is OPT008 explicit AVX2 packed-code decode.
The retained CUDA implementation is the frozen GPU003 packed CUDA operator
integrated through GPU004. GPU007 is rejected and is not a deployment
implementation.

## Storage result

Selected FP16-equivalent weight storage: 17,825,792 bytes.

Selected persistent packed storage: 2,367,488 bytes.

Selected-weight storage compression: **7.529412x**.

This ratio applies only to the selected weights. It is not whole-model
compression, whole-model storage reduction, or an automatic VRAM-reduction
claim.

## CPU results

The original frozen OPT008 benchmark observed a mean latency change of
**-8.307%** relative to OPT006 under its documented protocol.

The fresh final CPU validation observed:

- mean: **-3.393%**
- median: **-2.332%**

These values are preserved separately and are not averaged. They came from
different documented protocols. The retained CPU implementation is OPT008.

Final CPU validation established bit-exact equivalence with the frozen
OPT006 packed arithmetic reference, including operator and multi-prompt
full-model validation. It does not establish bit-exact equality with an
ordinary reconstructed-FP16 implementation.

## GPU results

### GPU005

Under its Quadro P1000 protocol, the retained GPU004 packed full-model path
observed **+7.511%** mean latency versus FP16. Its observed memory differences
were -7.242 MiB ready allocation and -15.367 MiB timed peak allocation.

### GPU006

GPU006 identified the dominant operation-level bottleneck as
`model.layers.4.linear_attn.in_proj_qkv`:

- FP16: 0.465736 ms
- GPU003 packed: 8.300687 ms
- change: +1682.274%

### GPU007

GPU007 evaluated exactly one parallel-K strategy. It was correct under its
predeclared tolerance but slower for every matrix:

- `k_proj`: GPU003 0.441437 ms; GPU007 2.062623 ms; +367.253%
- `out_proj`: GPU003 1.716003 ms; GPU007 6.606163 ms; +284.974%
- `in_proj_qkv`: GPU003 8.262222 ms; GPU007 26.522645 ms; +221.011%

GPU007 is explicitly rejected and no further CUDA optimization was attempted.

### GPU008

GPU008 freshly validated ordinary FP16 full-model inference against the
retained GPU004 runtime using the frozen GPU003 operator on a Quadro P1000:

| Prompt | Tokens | FP16 mean | Packed mean | Change |
|---|---:|---:|---:|---:|
| Short | 8 | 114.376822 ms | 117.373794 ms | +2.620% |
| Medium | 22 | 127.558327 ms | 137.117991 ms | +7.494% |
| Long | 47 | 179.283418 ms | 198.708732 ms | +10.835% |

Memory observations were:

- Short: ready -7.242 MiB; timed peak -15.367 MiB.
- Medium: ready -15.367 MiB; timed peak -15.367 MiB.
- Long: ready -15.367 MiB; timed peak -15.367 MiB.

These are protocol-specific Quadro P1000 observations. They must not be
generalized to modern GPUs or universal VRAM savings.

## Correctness chain

```text
Quantizer experiments
        ↓
ENG packed representation and checkpoint
        ↓
CPU packed arithmetic
        ↓
GPU003 isolated CUDA correctness
        ↓
GPU004 full-model integration
        ↓
GPU005 performance characterization
        ↓
GPU006 bottleneck characterization
        ↓
GPU007 controlled optimization attempt
        ↓
GPU008 retained-runtime validation
```

Packed arithmetic correctness and FP16-model behavioral agreement are
distinct claims. The FP16 and packed models contain different selected
weights, so full-model top-1 agreement is diagnostic only.

## Final engineering conclusions

**A.** Ternary packing substantially reduced selected-weight storage by the
documented 7.529412x ratio.

**B.** The packed representation executes directly without retaining a
persistent ordinary FP16 selected-weight matrix.

**C.** Packed arithmetic was validated on CPU and CUDA through the frozen
correctness chain.

**D.** The retained CPU implementation showed useful protocol-specific
latency reductions: -8.307% in the original OPT008 measurement and -3.393%
mean in fresh final validation.

**E.** The retained CUDA implementation did not outperform ordinary FP16 on
the Quadro P1000 under the documented GPU005/GPU008 protocols.

**F.** GPU006 identified `in_proj_qkv` as the dominant measured operation
bottleneck.

**G.** The controlled GPU007 parallelization did not solve it; it was slower.

**H.** OPT008 is retained for CPU. GPU003 integrated through GPU004 is
retained for CUDA.

**I.** Supported claims are limited to the frozen implementations, selected
matrices, recorded checkpoint, tested protocols, and characterized hardware.

**J.** Unsupported claims are listed below.

## Claim boundaries

This record does not support:

- universal ternary GPU speedup;
- universal CPU speedup;
- whole-model 7.529412x compression;
- universal VRAM reduction;
- superiority on modern datacenter GPUs;
- FP16/ternary model bit-exactness;
- equivalence merely because `p >= 0.05`;
- accuracy improvement inferred from positive deltas;
- deployment speedup from numerical ternarization alone.

Negative performance results are valid research results. GPU007 remains a
rejected optimization, and GPU003/GPU004 remains the retained CUDA runtime.
