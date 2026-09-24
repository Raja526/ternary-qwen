"""
ENG009 — Final end-to-end packed ternary engineering validation.

Purpose
-------
Consolidate the already-frozen engineering pipeline without introducing a
new quantizer, checkpoint format, runtime operator, or optimization.

Validated chain:

    frozen ENG003 checkpoint
        -> checkpoint integrity
        -> packed codes + scales
        -> exact packed reconstruction
        -> ordinary FP16 reference model
        -> packed runtime model
        -> three PackedTernaryLinear replacements
        -> full-model forward
        -> exact logit comparison

ENG008 benchmark conclusions are carried forward from the frozen result file;
ENG009 does not re-benchmark performance.

No inference speedup, peak RAM/VRAM reduction, fused 2-bit CUDA kernel, or
whole-model compression claim is made.
"""

import hashlib
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from packed_ternary.checkpoint import load_packed_checkpoint
from packed_ternary.packed_linear import PackedTernaryLinear
from packed_ternary.packed_weight import reconstruct_packed
from packed_ternary.runtime_loader import load_packed_runtime_modules


MODEL_NAME = "Qwen/Qwen3.5-0.8B"

CHECKPOINT_PATH = Path(
    "results/ENG003_multimatrix_checkpoint.pt"
)

ENG008_RESULT_PATH = Path(
    "results/ENG008_memory_latency_benchmark.txt"
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


def main():
    torch.manual_seed(SEED)

    section(
        "ENG009 — FINAL END-TO-END PACKED TERNARY VALIDATION"
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
        "Runtime targets         :",
        len(TARGETS),
    )

    for name in TARGETS:
        print("  -", name)

    # --------------------------------------------------------------
    # STEP 1 — Frozen checkpoint identity.
    # --------------------------------------------------------------

    section(
        "STEP 1 — VERIFY FROZEN CHECKPOINT IDENTITY"
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

    # --------------------------------------------------------------
    # STEP 2 — Checkpoint structure and logical storage.
    # --------------------------------------------------------------

    section(
        "STEP 2 — VERIFY CHECKPOINT STRUCTURE AND STORAGE"
    )

    metadata, packed_weights = (
        load_packed_checkpoint(
            CHECKPOINT_PATH
        )
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
        metadata[
            "threshold_multiplier"
        ],
    )

    print(
        "Matrix count            :",
        len(packed_weights),
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
        metadata[
            "threshold_multiplier"
        ]
        == THRESHOLD_MULTIPLIER
    )

    assert (
        list(packed_weights.keys())
        == TARGETS
    )

    total_parameters = 0
    total_packed_bytes = 0
    total_fp16_bytes = 0

    for name in TARGETS:
        packed = packed_weights[name]

        total_parameters += (
            packed.original_numel
        )

        total_packed_bytes += (
            packed.packed_storage_bytes
        )

        total_fp16_bytes += (
            packed.original_fp16_bytes
        )

        print()
        print(
            "Target                  :",
            name,
        )

        print(
            "Shape                   :",
            packed.original_shape,
        )

        print(
            "Parameters              :",
            packed.original_numel,
        )

        print(
            "Packed bytes            :",
            packed.packed_storage_bytes,
        )

        print(
            "FP16 bytes              :",
            packed.original_fp16_bytes,
        )

        print(
            "Compression             :",
            f"{packed.compression_factor:.6f}x",
        )

    logical_compression = (
        total_fp16_bytes
        / total_packed_bytes
    )

    disk_compression = (
        total_fp16_bytes
        / checkpoint_bytes
    )

    checkpoint_overhead = (
        checkpoint_bytes
        - total_packed_bytes
    )

    print()
    print(
        "Total parameters        :",
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
        "Checkpoint overhead     :",
        checkpoint_overhead,
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

    # --------------------------------------------------------------
    # STEP 3 — Tokenization.
    # --------------------------------------------------------------

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

    # --------------------------------------------------------------
    # STEP 4 — Reference model.
    #
    # Same packed checkpoint is reconstructed into persistent ordinary
    # FP16 Linear weights.
    # --------------------------------------------------------------

    section(
        "STEP 4 — RECONSTRUCT FP16 REFERENCE PATH"
    )

    reference_model = load_model()

    for name in TARGETS:
        module = get_module(
            reference_model,
            name,
        )

        packed = packed_weights[name]

        assert isinstance(
            module,
            torch.nn.Linear,
        )

        reconstructed = (
            reconstruct_packed(
                packed
            )
        )

        assert (
            tuple(reconstructed.shape)
            == tuple(module.weight.shape)
        )

        assert (
            reconstructed.dtype
            == module.weight.dtype
        )

        with torch.no_grad():
            module.weight.copy_(
                reconstructed
            )

        mismatch_count = int(
            (
                module.weight
                .detach()
                .cpu()
                != reconstructed
            )
            .sum()
            .item()
        )

        print()
        print(
            "Reference target        :",
            name,
        )

        print(
            "Reconstruction mismatch :",
            mismatch_count,
        )

        assert mismatch_count == 0

        del reconstructed

    with torch.inference_mode():
        reference_logits = (
            reference_model(
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
        "Reference logits shape  :",
        tuple(reference_logits.shape),
    )

    print(
        "Reference logits dtype  :",
        reference_logits.dtype,
    )

    del reference_model

    # --------------------------------------------------------------
    # STEP 5 — Packed runtime path.
    # --------------------------------------------------------------

    section(
        "STEP 5 — LOAD THREE PACKED RUNTIME MODULES"
    )

    runtime_model = load_model()

    original_objects = {}

    for name in TARGETS:
        module = get_module(
            runtime_model,
            name,
        )

        assert isinstance(
            module,
            torch.nn.Linear,
        )

        original_objects[name] = module

    load_result = (
        load_packed_runtime_modules(
            model=runtime_model,
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

    print(
        "Loaded module names     :",
        load_result.loaded_module_names,
    )

    print(
        "Replaced matrices       :",
        load_result.replaced_matrix_count,
    )

    print(
        "Replaced parameters     :",
        load_result.replaced_parameter_count,
    )

    print(
        "Persistent packed bytes :",
        load_result.persistent_packed_bytes,
    )

    print(
        "FP16 equivalent bytes   :",
        load_result.fp16_equivalent_bytes,
    )

    assert (
        load_result.loaded_module_names
        == tuple(TARGETS)
    )

    assert (
        load_result.replaced_matrix_count
        == 3
    )

    assert (
        load_result.replaced_parameter_count
        == EXPECTED_PARAMETERS
    )

    assert (
        load_result.persistent_packed_bytes
        == EXPECTED_PACKED_BYTES
    )

    assert (
        load_result.fp16_equivalent_bytes
        == EXPECTED_FP16_BYTES
    )

    for name in TARGETS:
        module = get_module(
            runtime_model,
            name,
        )

        parameters = dict(
            module.named_parameters()
        )

        buffers = dict(
            module.named_buffers()
        )

        print()
        print(
            "Runtime target          :",
            name,
        )

        print(
            "Type                    :",
            type(module).__name__,
        )

        print(
            "Object replaced         :",
            module
            is not original_objects[name],
        )

        print(
            "Persistent weight param :",
            "weight" in parameters,
        )

        print(
            "Packed codes buffer     :",
            "packed_codes" in buffers,
        )

        print(
            "Scales buffer           :",
            "scales" in buffers,
        )

        assert isinstance(
            module,
            PackedTernaryLinear,
        )

        assert (
            module
            is not original_objects[name]
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

    # --------------------------------------------------------------
    # STEP 6 — Full-model exact comparison.
    # --------------------------------------------------------------

    section(
        "STEP 6 — FULL-MODEL EXACT LOGIT COMPARISON"
    )

    with torch.inference_mode():
        runtime_logits = (
            runtime_model(
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

    assert (
        runtime_logits.shape
        == reference_logits.shape
    )

    assert (
        runtime_logits.dtype
        == reference_logits.dtype
    )

    difference = (
        runtime_logits.float()
        - reference_logits.float()
    ).abs()

    mismatches = int(
        (
            runtime_logits
            != reference_logits
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

    print(
        "Reference shape         :",
        tuple(reference_logits.shape),
    )

    print(
        "Runtime shape           :",
        tuple(runtime_logits.shape),
    )

    print(
        "Compared logits         :",
        reference_logits.numel(),
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
        "Dtype match             :",
        runtime_logits.dtype
        == reference_logits.dtype,
    )

    assert mismatches == 0
    assert max_difference == 0.0
    assert mean_difference == 0.0

    # --------------------------------------------------------------
    # STEP 7 — Carry forward frozen ENG008 benchmark conclusion.
    #
    # We validate the result artifact exists and contains the scoped
    # conclusions. We do NOT re-run the benchmark here.
    # --------------------------------------------------------------

    section(
        "STEP 7 — VERIFY ENG008 BENCHMARK CONCLUSION"
    )

    assert ENG008_RESULT_PATH.exists()

    eng008_text = (
        ENG008_RESULT_PATH.read_text()
    )

    required_eng008_lines = [
        "Selected-weight compression: 7.529412x",
        "Mean latency change: -0.342%",
        "No inference speedup claim.",
        "No peak RAM reduction claim.",
        "No peak VRAM reduction claim.",
        "ENG008 CPU MEMORY AND LATENCY BENCHMARK: PASS",
    ]

    for line in required_eng008_lines:
        present = (
            line in eng008_text
        )

        print(
            f"{line:<60}:",
            present,
        )

        assert present

    # --------------------------------------------------------------
    # Final consolidation.
    # --------------------------------------------------------------

    section(
        "ENG009 FINAL ENGINEERING SUMMARY"
    )

    print(
        "Checkpoint SHA256       :",
        checkpoint_sha256,
    )

    print(
        "Selected matrices       :",
        3,
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
        "Runtime replacements    :",
        load_result.replaced_matrix_count,
    )

    print(
        "Persistent FP16 weights :",
        0,
    )

    print(
        "Compared logits         :",
        reference_logits.numel(),
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

    print()
    print("Engineering conclusion:")
    print(
        "  The frozen three-matrix checkpoint can be stored compactly,"
    )
    print(
        "  loaded into a fresh Qwen model as packed runtime modules,"
    )
    print(
        "  and executed without persistent full FP16 weights in those"
    )
    print(
        "  three modules while matching the same packed matrices"
    )
    print(
        "  reconstructed into ordinary FP16 Linear weights exactly."
    )

    print()
    print("Scope:")
    print(
        "  Demonstrated compression applies to the three selected weights."
    )
    print(
        "  No whole-model compression claim."
    )
    print(
        "  No inference speedup claim."
    )
    print(
        "  No peak RAM reduction claim."
    )
    print(
        "  No peak VRAM reduction claim."
    )
    print(
        "  No fused 2-bit CUDA kernel claim."
    )

    section(
        "ENG009 END-TO-END ENGINEERING VALIDATION: PASS"
    )


if __name__ == "__main__":
    main()
