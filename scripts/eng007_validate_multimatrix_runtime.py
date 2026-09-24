"""
ENG007 — Real Qwen multi-matrix packed-runtime validation.

Controlled change from ENG006:
    ENG006 runtime-loaded one matrix from the ENG003 checkpoint.
    ENG007 runtime-loads all three matrices from the same checkpoint.

Path A:
    Reconstruct all three ENG003 packed matrices into ordinary FP16
    nn.Linear weights.

Path B:
    Load the same three packed matrices using the unchanged ENG006
    runtime loader and execute them through PackedTernaryLinear.

Everything else remains fixed.

No inference speedup, peak RAM/VRAM reduction, fused 2-bit CUDA kernel,
or whole-model compression is claimed.
"""

import gc
from pathlib import Path

import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from packed_ternary.checkpoint import (
    load_packed_checkpoint,
)

from packed_ternary.packed_linear import (
    PackedTernaryLinear,
)

from packed_ternary.packed_weight import (
    reconstruct_packed,
)

from packed_ternary.runtime_loader import (
    load_packed_runtime_modules,
)


MODEL_NAME = "Qwen/Qwen3.5-0.8B"

CHECKPOINT_PATH = Path(
    "results/ENG003_multimatrix_checkpoint.pt"
)

TARGETS = [
    "model.layers.3.self_attn.k_proj",
    "model.layers.4.linear_attn.out_proj",
    "model.layers.4.linear_attn.in_proj_qkv",
]

REAL_ROWS_PER_CHUNK = 64

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


def load_model():

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            MODEL_NAME,
            dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
    )

    model.eval()

    return model


def get_module(
    model,
    name,
):

    modules = dict(
        model.named_modules()
    )

    if name not in modules:
        raise KeyError(
            f"Module not found: {name}"
        )

    return modules[name]


def main():

    torch.manual_seed(
        SEED
    )

    section(
        "ENG007 — REAL QWEN MULTI-MATRIX PACKED RUNTIME"
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
        print(
            "  -",
            name,
        )

    print(
        "Rows/chunk              :",
        REAL_ROWS_PER_CHUNK,
    )

    # --------------------------------------------------------------
    # STEP 1
    # Load the frozen ENG003 checkpoint.
    # --------------------------------------------------------------

    section(
        "STEP 1 — LOAD AND VERIFY ENG003 CHECKPOINT"
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
        "Checkpoint group size   :",
        metadata["group_size"],
    )

    print(
        "Threshold multiplier    :",
        metadata[
            "threshold_multiplier"
        ],
    )

    print(
        "Checkpoint matrices     :",
        len(packed_weights),
    )

    for name in packed_weights:
        print(
            "  -",
            name,
        )

    assert (
        metadata["model_name"]
        == MODEL_NAME
    )

    assert (
        set(packed_weights.keys())
        == set(TARGETS)
    )

    total_parameters = 0
    total_packed_bytes = 0
    total_fp16_bytes = 0

    print()
    print("Per-matrix storage:")

    for name in TARGETS:

        packed = packed_weights[
            name
        ]

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

    aggregate_compression = (
        total_fp16_bytes
        / total_packed_bytes
    )

    print()
    print(
        "Total parameters        :",
        total_parameters,
    )

    print(
        "Total packed bytes      :",
        total_packed_bytes,
    )

    print(
        "Total FP16 bytes        :",
        total_fp16_bytes,
    )

    print(
        "Aggregate compression   :",
        f"{aggregate_compression:.6f}x",
    )

    # --------------------------------------------------------------
    # Tokenization — same prompt/setup as ENG005 and ENG006.
    # --------------------------------------------------------------

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            MODEL_NAME
        )
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

    attention_mask = (
        encoded.get(
            "attention_mask"
        )
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
    # STEP 2 — PATH A
    #
    # Reconstruct all three packed matrices into ordinary Linear
    # weights.
    # --------------------------------------------------------------

    section(
        "STEP 2 — PATH A: THREE RECONSTRUCTED FP16 LINEAR WEIGHTS"
    )

    reference_model = (
        load_model()
    )

    for name in TARGETS:

        module = get_module(
            reference_model,
            name,
        )

        packed = packed_weights[
            name
        ]

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
            tuple(
                reconstructed.shape
            )
            == tuple(
                module.weight.shape
            )
        )

        assert (
            reconstructed.dtype
            == module.weight.dtype
        )

        with torch.no_grad():

            module.weight.copy_(
                reconstructed
            )

        mismatches = int(
            (
                module
                .weight
                .detach()
                .cpu()
                != reconstructed
            )
            .sum()
            .item()
        )

        print()
        print(
            "Loaded                  :",
            name,
        )

        print(
            "Type                    :",
            type(module).__name__,
        )

        print(
            "Parameters              :",
            module.weight.numel(),
        )

        print(
            "Reconstruction mismatch :",
            mismatches,
        )

        assert mismatches == 0

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
        tuple(
            reference_logits.shape
        ),
    )

    print(
        "Reference logits dtype  :",
        reference_logits.dtype,
    )

    del reference_model

    gc.collect()

    # --------------------------------------------------------------
    # STEP 3 — PATH B
    #
    # Same checkpoint, same packed matrices, unchanged ENG006
    # runtime loader — but select all three entries.
    # --------------------------------------------------------------

    section(
        "STEP 3 — PATH B: THREE PACKED RUNTIME MODULES"
    )

    runtime_model = (
        load_model()
    )

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

        original_objects[
            name
        ] = module

        print(
            "Before loader          :",
            name,
            "->",
            type(module).__name__,
        )

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
                REAL_ROWS_PER_CHUNK
            ),
        )
    )

    print()
    print(
        "Loaded module names     :",
        load_result
        .loaded_module_names,
    )

    print(
        "Replaced matrices       :",
        load_result
        .replaced_matrix_count,
    )

    print(
        "Replaced parameters     :",
        load_result
        .replaced_parameter_count,
    )

    print(
        "Loader packed bytes     :",
        load_result
        .persistent_packed_bytes,
    )

    print(
        "Loader FP16 bytes       :",
        load_result
        .fp16_equivalent_bytes,
    )

    assert (
        load_result
        .loaded_module_names
        == tuple(TARGETS)
    )

    assert (
        load_result
        .replaced_matrix_count
        == len(TARGETS)
    )

    assert (
        load_result
        .replaced_parameter_count
        == total_parameters
    )

    assert (
        load_result
        .persistent_packed_bytes
        == total_packed_bytes
    )

    assert (
        load_result
        .fp16_equivalent_bytes
        == total_fp16_bytes
    )

    # --------------------------------------------------------------
    # Inspect all three replacements.
    # --------------------------------------------------------------

    inspected_packed_bytes = 0
    inspected_fp16_bytes = 0

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
            "After loader           :",
            name,
        )

        print(
            "Type                    :",
            type(module).__name__,
        )

        print(
            "Object replaced         :",
            module
            is not original_objects[
                name
            ],
        )

        print(
            "Full weight Parameter   :",
            "weight"
            in parameters,
        )

        print(
            "Packed codes buffer     :",
            "packed_codes"
            in buffers,
        )

        print(
            "Scales buffer           :",
            "scales"
            in buffers,
        )

        print(
            "Persistent bytes        :",
            module
            .persistent_weight_bytes,
        )

        print(
            "FP16 equivalent bytes   :",
            module
            .fp16_equivalent_weight_bytes,
        )

        print(
            "Compression             :",
            f"{module.persistent_compression_factor:.6f}x",
        )

        assert isinstance(
            module,
            PackedTernaryLinear,
        )

        assert (
            module
            is not original_objects[
                name
            ]
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

        inspected_packed_bytes += (
            module
            .persistent_weight_bytes
        )

        inspected_fp16_bytes += (
            module
            .fp16_equivalent_weight_bytes
        )

    assert (
        inspected_packed_bytes
        == total_packed_bytes
    )

    assert (
        inspected_fp16_bytes
        == total_fp16_bytes
    )

    # --------------------------------------------------------------
    # STEP 4 — Full-model comparison.
    # --------------------------------------------------------------

    section(
        "STEP 4 — FULL-MODEL LOGIT COMPARISON"
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
        "Reference logits shape  :",
        tuple(
            reference_logits.shape
        ),
    )

    print(
        "Runtime logits shape    :",
        tuple(
            runtime_logits.shape
        ),
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

    assert (
        runtime_logits.shape
        == reference_logits.shape
    )

    assert (
        runtime_logits.dtype
        == reference_logits.dtype
    )

    assert mismatches == 0
    assert max_difference == 0.0

    # --------------------------------------------------------------
    # Final scoped result.
    # --------------------------------------------------------------

    section(
        "ENG007 FINAL SUMMARY"
    )

    print(
        "Runtime matrices loaded :",
        load_result
        .replaced_matrix_count,
    )

    print(
        "Runtime parameters      :",
        load_result
        .replaced_parameter_count,
    )

    print(
        "Persistent packed bytes :",
        load_result
        .persistent_packed_bytes,
    )

    print(
        "FP16 equivalent bytes   :",
        load_result
        .fp16_equivalent_bytes,
    )

    print(
        "Compression factor      :",
        f"{aggregate_compression:.6f}x",
    )

    print(
        "Persistent full weights :",
        False,
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

    print()
    print("Scope:")
    print(
        "  All three ENG003 packed matrices were runtime-loaded."
    )
    print(
        "  Persistent representation is packed codes + scales."
    )
    print(
        "  Temporary row chunks are reconstructed during forward execution."
    )
    print(
        "  Exact comparison is against the same three packed matrices"
    )
    print(
        "  reconstructed into ordinary FP16 Linear weights."
    )
    print(
        "  No inference speedup is claimed."
    )
    print(
        "  No peak RAM/VRAM reduction is claimed."
    )
    print(
        "  No fused 2-bit CUDA kernel is claimed."
    )
    print(
        "  No whole-model compression is claimed."
    )

    section(
        "ENG007 REAL QWEN MULTI-MATRIX PACKED RUNTIME: PASS"
    )


if __name__ == "__main__":
    main()
