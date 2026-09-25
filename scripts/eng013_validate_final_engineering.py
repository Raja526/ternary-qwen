"""
ENG013 — Final engineering consolidation.

Purpose
-------
Consolidate the frozen packed-ternary engineering pipeline after ENG012
without introducing a new quantizer, checkpoint format, runtime operator,
benchmark, or optimization.

Validated chain:

    frozen ENG003 packed checkpoint
        -> checkpoint identity and metadata
        -> packed codes + FP16 scales
        -> ENG010 DirectTernaryLinear reference path
        -> ENG011 StreamingDirectTernaryLinear final path
        -> persistent packed representation checks
        -> chunk-local decoding structural checks
        -> full-model bit-exact ENG010 <-> ENG011 comparison
        -> carry forward frozen ENG012 benchmark evidence

ENG012 benchmark conclusions are read from the frozen result artifact.
ENG013 does not re-run the benchmark.

No general inference speedup, general peak RAM reduction, peak VRAM
reduction, fused 2-bit kernel, or whole-model compression claim is made.
"""

import hashlib
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from packed_ternary.checkpoint import load_packed_checkpoint
from packed_ternary.direct_linear import DirectTernaryLinear
from packed_ternary.direct_runtime_loader import (
    load_direct_runtime_modules,
)
from packed_ternary.streaming_direct_linear import (
    StreamingDirectTernaryLinear,
)
from packed_ternary.streaming_runtime_loader import (
    load_streaming_runtime_modules,
)


MODEL_NAME = "Qwen/Qwen3.5-0.8B"

CHECKPOINT_PATH = Path(
    "results/ENG003_multimatrix_checkpoint.pt"
)

ENG012_RESULT_PATH = Path(
    "results/ENG012_streaming_memory_latency_benchmark.txt"
)

EXPECTED_CHECKPOINT_SHA256 = (
    "8e35aaea4e47a68363a8951fa425874a1"
    "e35929a19c277335a260bb20a9cb546"
)

EXPECTED_CHECKPOINT_DISK_BYTES = 2_371_165

TARGETS = [
    "model.layers.3.self_attn.k_proj",
    "model.layers.4.linear_attn.out_proj",
    "model.layers.4.linear_attn.in_proj_qkv",
]

EXPECTED_PARAMETERS = 8_912_896
EXPECTED_PACKED_BYTES = 2_367_488
EXPECTED_FP16_BYTES = 17_825_792

GROUP_SIZE = 128
THRESHOLD_MULTIPLIER = 0.5
ROWS_PER_CHUNK = 64

PROMPT = (
    "A box contains 12 red balls and 8 blue balls. "
    "How many balls are in the box?"
)

SEED = 20260924


def section(title):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    model.eval()

    return model


def get_module(model, name):
    modules = dict(model.named_modules())

    if name not in modules:
        raise KeyError(
            f"Module not found: {name}"
        )

    return modules[name]


def validate_load_result(result):
    assert (
        result.loaded_module_names
        == tuple(TARGETS)
    )

    assert (
        result.replaced_matrix_count
        == len(TARGETS)
    )

    assert (
        result.replaced_parameter_count
        == EXPECTED_PARAMETERS
    )

    assert (
        result.persistent_packed_bytes
        == EXPECTED_PACKED_BYTES
    )

    assert (
        result.fp16_equivalent_bytes
        == EXPECTED_FP16_BYTES
    )


def main():
    torch.manual_seed(SEED)

    section(
        "ENG013 — FINAL PACKED TERNARY ENGINEERING CONSOLIDATION"
    )

    print(
        "Model                   :",
        MODEL_NAME,
    )

    print(
        "Checkpoint              :",
        CHECKPOINT_PATH,
    )

    print(
        "Selected matrices       :",
        len(TARGETS),
    )

    for name in TARGETS:
        print("  -", name)

    # ------------------------------------------------------------------
    # STEP 1 — Frozen checkpoint identity.
    # ------------------------------------------------------------------

    section(
        "STEP 1 — VERIFY FROZEN ENG003 CHECKPOINT IDENTITY"
    )

    assert CHECKPOINT_PATH.exists()

    checkpoint_bytes = (
        CHECKPOINT_PATH.stat().st_size
    )

    checkpoint_sha256 = (
        sha256_file(
            CHECKPOINT_PATH
        )
    )

    print(
        "Checkpoint disk bytes   :",
        checkpoint_bytes,
    )

    print(
        "Checkpoint SHA256       :",
        checkpoint_sha256,
    )

    assert (
        checkpoint_bytes
        == EXPECTED_CHECKPOINT_DISK_BYTES
    )

    assert (
        checkpoint_sha256
        == EXPECTED_CHECKPOINT_SHA256
    )

    print(
        "Checkpoint identity     : PASS"
    )

    # ------------------------------------------------------------------
    # STEP 2 — Checkpoint structure and storage.
    # ------------------------------------------------------------------

    section(
        "STEP 2 — VERIFY CHECKPOINT STRUCTURE AND STORAGE"
    )

    metadata, packed_weights = (
        load_packed_checkpoint(
            CHECKPOINT_PATH
        )
    )

    assert (
        metadata["model_name"]
        == MODEL_NAME
    )

    assert (
        metadata["group_size"]
        == GROUP_SIZE
    )

    assert (
        metadata["threshold_multiplier"]
        == THRESHOLD_MULTIPLIER
    )

    assert (
        list(packed_weights.keys())
        == TARGETS
    )

    total_parameters = sum(
        packed_weights[name].original_numel
        for name in TARGETS
    )

    total_packed_bytes = sum(
        packed_weights[name].packed_storage_bytes
        for name in TARGETS
    )

    total_fp16_bytes = sum(
        packed_weights[name].original_fp16_bytes
        for name in TARGETS
    )

    logical_compression = (
        total_fp16_bytes
        / total_packed_bytes
    )

    disk_compression = (
        total_fp16_bytes
        / checkpoint_bytes
    )

    print(
        "Checkpoint model        :",
        metadata["model_name"],
    )

    print(
        "Group size              :",
        metadata["group_size"],
    )

    print(
        "Threshold multiplier    :",
        metadata["threshold_multiplier"],
    )

    print(
        "Selected parameters     :",
        total_parameters,
    )

    print(
        "Logical packed bytes    :",
        total_packed_bytes,
    )

    print(
        "FP16 equivalent bytes   :",
        total_fp16_bytes,
    )

    print(
        "Logical compression     :",
        f"{logical_compression:.6f}x",
    )

    print(
        "Actual disk compression :",
        f"{disk_compression:.6f}x",
    )

    assert (
        total_parameters
        == EXPECTED_PARAMETERS
    )

    assert (
        total_packed_bytes
        == EXPECTED_PACKED_BYTES
    )

    assert (
        total_fp16_bytes
        == EXPECTED_FP16_BYTES
    )

    # ------------------------------------------------------------------
    # STEP 3 — Frozen input.
    # ------------------------------------------------------------------

    section(
        "STEP 3 — PREPARE FROZEN INPUT"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    encoded = tokenizer(
        PROMPT,
        return_tensors="pt",
        add_special_tokens=True,
    )

    input_ids = (
        encoded["input_ids"]
        .cpu()
    )

    attention_mask = encoded.get(
        "attention_mask"
    )

    if attention_mask is not None:
        attention_mask = (
            attention_mask.cpu()
        )

    print(
        "Input tokens            :",
        input_ids.shape[1],
    )

    assert input_ids.shape[1] == 22

    # ------------------------------------------------------------------
    # STEP 4 — ENG010 direct-compute reference.
    # ------------------------------------------------------------------

    section(
        "STEP 4 — LOAD FROZEN ENG010 DIRECT RUNTIME"
    )

    direct_model = load_model()

    direct_result = (
        load_direct_runtime_modules(
            model=direct_model,
            checkpoint_path=(
                CHECKPOINT_PATH
            ),
            selected_module_names=(
                TARGETS
            ),
            expected_model_name=(
                MODEL_NAME
            ),
            rows_per_chunk=(
                ROWS_PER_CHUNK
            ),
        )
    )

    validate_load_result(
        direct_result
    )

    for name in TARGETS:
        module = get_module(
            direct_model,
            name,
        )

        parameters = dict(
            module.named_parameters()
        )

        buffers = dict(
            module.named_buffers()
        )

        assert isinstance(
            module,
            DirectTernaryLinear,
        )

        assert (
            "weight"
            not in parameters
        )

        assert (
            "packed_codes"
            in buffers
        )

        assert (
            "scales"
            in buffers
        )

        print()
        print(
            "Direct target           :",
            name,
        )

        print(
            "Type                    :",
            type(module).__name__,
        )

        print(
            "Persistent weight param :",
            "weight" in parameters,
        )

    with torch.inference_mode():
        direct_logits = (
            direct_model(
                input_ids=input_ids,
                attention_mask=(
                    attention_mask
                ),
                use_cache=False,
            )
            .logits
            .detach()
            .cpu()
            .clone()
        )

    print()
    print(
        "Direct logits shape     :",
        tuple(direct_logits.shape),
    )

    print(
        "Direct logits dtype     :",
        direct_logits.dtype,
    )

    del direct_model

    # ------------------------------------------------------------------
    # STEP 5 — ENG011 streaming runtime.
    # ------------------------------------------------------------------

    section(
        "STEP 5 — LOAD FROZEN ENG011 STREAMING RUNTIME"
    )

    streaming_model = load_model()

    streaming_result = (
        load_streaming_runtime_modules(
            model=streaming_model,
            checkpoint_path=(
                CHECKPOINT_PATH
            ),
            selected_module_names=(
                TARGETS
            ),
            expected_model_name=(
                MODEL_NAME
            ),
            rows_per_chunk=(
                ROWS_PER_CHUNK
            ),
        )
    )

    validate_load_result(
        streaming_result
    )

    full_decode_total = 0
    chunk_decode_total = 0

    for name in TARGETS:
        module = get_module(
            streaming_model,
            name,
        )

        parameters = dict(
            module.named_parameters()
        )

        buffers = dict(
            module.named_buffers()
        )

        assert isinstance(
            module,
            StreamingDirectTernaryLinear,
        )

        assert (
            "weight"
            not in parameters
        )

        assert (
            "packed_codes"
            in buffers
        )

        assert (
            "scales"
            in buffers
        )

        full_decode_bytes = (
            module.full_decoded_code_bytes
        )

        chunk_decode_bytes = (
            module.maximum_chunk_decoded_code_bytes
        )

        decode_ratio = (
            full_decode_bytes
            / chunk_decode_bytes
        )

        full_decode_total += (
            full_decode_bytes
        )

        chunk_decode_total += (
            chunk_decode_bytes
        )

        print()
        print(
            "Streaming target        :",
            name,
        )

        print(
            "Type                    :",
            type(module).__name__,
        )

        print(
            "Persistent weight param :",
            "weight" in parameters,
        )

        print(
            "Full decode bytes       :",
            full_decode_bytes,
        )

        print(
            "Max chunk decode bytes  :",
            chunk_decode_bytes,
        )

        print(
            "Decode working-set ratio:",
            f"{decode_ratio:.6f}x",
        )

    # These are tensor-size accounting figures only.
    # They are not process RSS or allocator measurements.

    print()
    print(
        "Sum full-decode bytes   :",
        full_decode_total,
    )

    print(
        "Sum max-chunk bytes     :",
        chunk_decode_total,
    )

    print(
        "Accounting ratio        :",
        f"{full_decode_total / chunk_decode_total:.6f}x",
    )

    with torch.inference_mode():
        streaming_logits = (
            streaming_model(
                input_ids=input_ids,
                attention_mask=(
                    attention_mask
                ),
                use_cache=False,
            )
            .logits
            .detach()
            .cpu()
        )

    # ------------------------------------------------------------------
    # STEP 6 — Exact ENG010 <-> ENG011 comparison.
    # ------------------------------------------------------------------

    section(
        "STEP 6 — EXACT ENG010 <-> ENG011 FULL-MODEL COMPARISON"
    )

    assert (
        streaming_logits.shape
        == direct_logits.shape
    )

    assert (
        streaming_logits.dtype
        == direct_logits.dtype
    )

    difference = (
        streaming_logits.float()
        - direct_logits.float()
    ).abs()

    mismatches = int(
        (
            streaming_logits
            != direct_logits
        )
        .sum()
        .item()
    )

    max_difference = float(
        difference.max().item()
    )

    mean_difference = float(
        difference.mean().item()
    )

    compared_logits = (
        direct_logits.numel()
    )

    direct_top1 = (
        direct_logits.argmax(
            dim=-1
        )
    )

    streaming_top1 = (
        streaming_logits.argmax(
            dim=-1
        )
    )

    top1_matches = int(
        (
            direct_top1
            == streaming_top1
        )
        .sum()
        .item()
    )

    top1_positions = int(
        direct_top1.numel()
    )

    print(
        "Direct shape            :",
        tuple(direct_logits.shape),
    )

    print(
        "Streaming shape         :",
        tuple(streaming_logits.shape),
    )

    print(
        "Compared logits         :",
        compared_logits,
    )

    print(
        "Logit mismatches        :",
        mismatches,
    )

    print(
        "Max abs difference      :",
        max_difference,
    )

    print(
        "Mean abs difference     :",
        mean_difference,
    )

    print(
        "Bit-exact               :",
        mismatches == 0,
    )

    print(
        "Top-1 matches           :",
        f"{top1_matches}/{top1_positions}",
    )

    assert mismatches == 0
    assert max_difference == 0.0
    assert mean_difference == 0.0

    assert (
        top1_matches
        == top1_positions
    )

    # ------------------------------------------------------------------
    # STEP 7 — Carry forward frozen ENG012 benchmark.
    # ------------------------------------------------------------------

    section(
        "STEP 7 — VERIFY FROZEN ENG012 BENCHMARK EVIDENCE"
    )

    assert ENG012_RESULT_PATH.exists()

    eng012_text = (
        ENG012_RESULT_PATH.read_text()
    )

    required_eng012_lines = [
        "Persistent compression: 7.529412x",
        "Streaming mean latency change: +6.670%",
        "Timed-max RSS difference MiB (streaming - direct): -6.490",
        "GNU max RSS difference KB (streaming - direct): +18.667",
        "No general inference speedup claim.",
        "No general peak RAM reduction claim.",
        "No peak VRAM reduction claim.",
        "No fused 2-bit kernel claim.",
        "No whole-model compression claim.",
        "ENG012 STREAMING MEMORY AND LATENCY BENCHMARK: PASS",
    ]

    for line in required_eng012_lines:
        present = (
            line in eng012_text
        )

        print(
            f"{line:<72}:",
            present,
        )

        assert present

    # ------------------------------------------------------------------
    # Final consolidation.
    # ------------------------------------------------------------------

    section(
        "ENG013 FINAL ENGINEERING SUMMARY"
    )

    print(
        "Checkpoint SHA256       :",
        checkpoint_sha256,
    )

    print(
        "Selected matrices       :",
        len(TARGETS),
    )

    print(
        "Selected parameters     :",
        total_parameters,
    )

    print(
        "Logical packed bytes    :",
        total_packed_bytes,
    )

    print(
        "FP16 equivalent bytes   :",
        total_fp16_bytes,
    )

    print(
        "Logical compression     :",
        f"{logical_compression:.6f}x",
    )

    print(
        "Actual checkpoint bytes :",
        checkpoint_bytes,
    )

    print(
        "Actual disk compression :",
        f"{disk_compression:.6f}x",
    )

    print(
        "Rows per chunk          :",
        ROWS_PER_CHUNK,
    )

    print(
        "Compared logits         :",
        compared_logits,
    )

    print(
        "Logit mismatches        :",
        mismatches,
    )

    print(
        "Bit-exact ENG010/ENG011 :",
        mismatches == 0,
    )

    print(
        "Top-1 matches           :",
        f"{top1_matches}/{top1_positions}",
    )

    print()
    print("Frozen ENG012 measured evidence:")
    print(
        "  Streaming timed-max RSS difference : -6.490 MiB"
    )
    print(
        "  Streaming mean latency change       : +6.670%"
    )
    print(
        "  GNU max RSS difference              : +18.667 KB"
    )

    print()
    print("Engineering conclusion:")
    print(
        "  The frozen three-matrix packed checkpoint can be loaded into"
    )
    print(
        "  Qwen3.5-0.8B as ENG011 streaming direct-compute modules,"
    )
    print(
        "  retaining the 7.529412x selected-weight persistent storage"
    )
    print(
        "  compression while avoiding ENG010's full selected-matrix"
    )
    print(
        "  temporary int8 code decode."
    )
    print(
        "  On the frozen 22-token full-model validation, ENG011 is"
    )
    print(
        "  bit-exact with ENG010."
    )
    print(
        "  ENG012 measured lower timed-max RSS for streaming in its"
    )
    print(
        "  controlled CPU runs, with a 6.670% mean latency increase;"
    )
    print(
        "  GNU maximum RSS was effectively unchanged."
    )

    print()
    print("Scope:")
    print(
        "  Compression applies only to the three selected weights."
    )
    print(
        "  Decoded-code byte counts are tensor-size accounting only."
    )
    print(
        "  No whole-model compression claim."
    )
    print(
        "  No general inference speedup claim."
    )
    print(
        "  No general peak RAM reduction claim."
    )
    print(
        "  No peak VRAM reduction claim."
    )
    print(
        "  No fused 2-bit kernel claim."
    )

    section(
        "ENG013 FINAL ENGINEERING CONSOLIDATION: PASS"
    )


if __name__ == "__main__":
    main()
