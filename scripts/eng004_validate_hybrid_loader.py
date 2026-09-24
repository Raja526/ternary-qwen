"""
ENG004 — Real Qwen hybrid-loader validation.

Objective
---------
Verify that loading the ENG003 packed multi-matrix checkpoint through
packed_ternary.hybrid_loader produces exactly the same selected FP16
weights as direct application of the frozen scientific ternarizer.

The experiment also verifies:
    - exactly three expected matrices are replaced
    - exactly 8,912,896 parameters are replaced
    - selected non-target control matrices remain unchanged
    - deterministic forward outputs match the direct-reference path

Important
---------
The hybrid-loaded model reconstructs packed weights back into ordinary
FP16 PyTorch parameters.

ENG004 therefore does NOT demonstrate:
    - packed runtime execution
    - runtime VRAM reduction
    - inference acceleration
"""

import gc
import hashlib
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from ternary.quantizer import (
    groupwise_ternarize_weight,
)

from packed_ternary.checkpoint import (
    load_packed_checkpoint,
)

from packed_ternary.packed_weight import (
    reconstruct_packed,
)

from packed_ternary.hybrid_loader import (
    load_packed_weights_into_model,
)


MODEL_NAME = "Qwen/Qwen3.5-0.8B"

CHECKPOINT_PATH = Path(
    "results/ENG003_multimatrix_checkpoint.pt"
)

EXPECTED_CHECKPOINT_SHA256 = (
    "8e35aaea4e47a68363a8951fa425874a1"
    "e35929a19c277335a260bb20a9cb546"
)

GROUP_SIZE = 128
THRESHOLD_MULTIPLIER = 0.50

TARGET_MODULES = (
    "model.layers.3.self_attn.k_proj",
    "model.layers.4.linear_attn.out_proj",
    "model.layers.4.linear_attn.in_proj_qkv",
)

EXPECTED_TARGET_PARAMETERS = 8_912_896


# Controls deliberately include:
#   - same layer / different self-attention projection
#   - same linear-attention layer / different projection
#   - MLP weight
#
# None are checkpoint targets.
CONTROL_MODULES = (
    "model.layers.3.self_attn.q_proj",
    "model.layers.4.linear_attn.in_proj_z",
    "model.layers.4.mlp.gate_proj",
)


PROMPT = (
    "A box contains 12 red balls and 8 blue balls. "
    "How many balls are in the box?"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def get_linear_module(
    model,
    name,
):
    modules = dict(
        model.named_modules()
    )

    if name not in modules:
        raise RuntimeError(
            f"Module not found: {name}"
        )

    module = modules[name]

    if not isinstance(
        module,
        torch.nn.Linear,
    ):
        raise TypeError(
            f"{name} is not torch.nn.Linear"
        )

    return module


def exact_tensor_comparison(
    left,
    right,
):
    if left.shape != right.shape:
        return {
            "shape_match": False,
            "dtype_match":
                left.dtype == right.dtype,
            "mismatches": None,
            "max_difference": None,
            "mean_difference": None,
        }

    difference = (
        left.float()
        - right.float()
    ).abs()

    mismatches = int(
        (
            left != right
        ).sum().item()
    )

    return {
        "shape_match": True,
        "dtype_match":
            left.dtype == right.dtype,
        "mismatches":
            mismatches,
        "max_difference":
            float(
                difference.max().item()
            ),
        "mean_difference":
            float(
                difference.mean().item()
            ),
    }


def load_fp16_model():
    return (
        AutoModelForCausalLM
        .from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
    )


def main():

    torch.manual_seed(
        20260924
    )

    print("=" * 100)
    print("ENG004 — REAL QWEN HYBRID LOADER VALIDATION")
    print("=" * 100)

    print()
    print(
        "Model                    :",
        MODEL_NAME,
    )

    print(
        "Checkpoint               :",
        CHECKPOINT_PATH,
    )

    print(
        "Group size               :",
        GROUP_SIZE,
    )

    print(
        "Threshold multiplier     :",
        THRESHOLD_MULTIPLIER,
    )

    print(
        "Expected target matrices :",
        len(TARGET_MODULES),
    )

    print(
        "Expected target params   :",
        EXPECTED_TARGET_PARAMETERS,
    )

    # ==============================================================
    # Checkpoint identity
    # ==============================================================

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"ENG003 checkpoint missing: "
            f"{CHECKPOINT_PATH}"
        )

    checkpoint_sha256 = (
        sha256_file(
            CHECKPOINT_PATH
        )
    )

    print()
    print("-" * 100)
    print("CHECKPOINT IDENTITY")
    print("-" * 100)

    print(
        "Expected SHA-256 :",
        EXPECTED_CHECKPOINT_SHA256,
    )

    print(
        "Actual SHA-256   :",
        checkpoint_sha256,
    )

    checksum_match = (
        checkpoint_sha256
        == EXPECTED_CHECKPOINT_SHA256
    )

    print(
        "Checksum match   :",
        checksum_match,
    )

    assert checksum_match

    # ==============================================================
    # Inspect checkpoint itself
    # ==============================================================

    checkpoint_metadata, checkpoint_weights = (
        load_packed_checkpoint(
            CHECKPOINT_PATH
        )
    )

    checkpoint_names = tuple(
        checkpoint_weights.keys()
    )

    print()
    print("-" * 100)
    print("CHECKPOINT CONTENT")
    print("-" * 100)

    print(
        "Checkpoint model :",
        checkpoint_metadata[
            "model_name"
        ],
    )

    print(
        "Weight count     :",
        checkpoint_metadata[
            "weight_count"
        ],
    )

    for name in checkpoint_names:
        print("  -", name)

    assert (
        checkpoint_metadata[
            "model_name"
        ]
        == MODEL_NAME
    )

    assert (
        checkpoint_metadata[
            "group_size"
        ]
        == GROUP_SIZE
    )

    assert (
        checkpoint_metadata[
            "threshold_multiplier"
        ]
        == THRESHOLD_MULTIPLIER
    )

    assert (
        set(checkpoint_names)
        == set(TARGET_MODULES)
    )

    # Keep exact checkpoint reconstructions as an independent
    # comparison reference.
    checkpoint_references = {}

    for name in TARGET_MODULES:
        checkpoint_references[name] = (
            reconstruct_packed(
                checkpoint_weights[
                    name
                ]
            )
            .cpu()
            .clone()
        )

    del checkpoint_weights

    # ==============================================================
    # Tokenizer / deterministic input
    # ==============================================================

    print()
    print("Loading tokenizer...")

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
        "Input token count        :",
        input_ids.shape[1],
    )

    # ==============================================================
    # PATH A — DIRECT FROZEN SCIENTIFIC QUANTIZER
    # ==============================================================

    print()
    print("=" * 100)
    print("PATH A — DIRECT FROZEN QUANTIZER")
    print("=" * 100)

    reference_model = (
        load_fp16_model()
    )

    reference_model.eval()

    # Store untouched controls from the original model.
    original_controls = {}

    for name in CONTROL_MODULES:

        module = get_linear_module(
            reference_model,
            name,
        )

        original_controls[name] = (
            module.weight
            .detach()
            .cpu()
            .clone()
        )

    direct_target_weights = {}

    direct_parameter_count = 0

    with torch.no_grad():

        for name in TARGET_MODULES:

            module = get_linear_module(
                reference_model,
                name,
            )

            result = (
                groupwise_ternarize_weight(
                    module.weight,
                    method="optimized",
                    group_size=GROUP_SIZE,
                    threshold_multiplier=(
                        THRESHOLD_MULTIPLIER
                    ),
                )
            )

            reconstructed = (
                result[
                    "reconstructed"
                ]
            )

            direct_parameter_count += (
                reconstructed.numel()
            )

            module.weight.copy_(
                reconstructed
            )

            direct_target_weights[
                name
            ] = (
                module.weight
                .detach()
                .cpu()
                .clone()
            )

    print(
        "Direct matrices replaced :",
        len(direct_target_weights),
    )

    print(
        "Direct params replaced   :",
        direct_parameter_count,
    )

    assert (
        direct_parameter_count
        == EXPECTED_TARGET_PARAMETERS
    )

    # --------------------------------------------------------------
    # Direct scientific reconstruction vs checkpoint reconstruction
    # --------------------------------------------------------------

    print()
    print("-" * 100)
    print("DIRECT vs PACKED CHECKPOINT WEIGHTS")
    print("-" * 100)

    for name in TARGET_MODULES:

        comparison = (
            exact_tensor_comparison(
                direct_target_weights[
                    name
                ],
                checkpoint_references[
                    name
                ],
            )
        )

        print()
        print(name)

        print(
            "  mismatches          :",
            comparison[
                "mismatches"
            ],
        )

        print(
            "  max abs difference  :",
            comparison[
                "max_difference"
            ],
        )

        print(
            "  mean abs difference :",
            comparison[
                "mean_difference"
            ],
        )

        assert (
            comparison[
                "shape_match"
            ]
        )

        assert (
            comparison[
                "dtype_match"
            ]
        )

        assert (
            comparison[
                "mismatches"
            ]
            == 0
        )

        assert (
            comparison[
                "max_difference"
            ]
            == 0.0
        )

    # --------------------------------------------------------------
    # Reference forward pass
    # --------------------------------------------------------------

    print()
    print(
        "Running reference forward pass..."
    )

    with torch.inference_mode():

        reference_output = (
            reference_model(
                input_ids=input_ids,
                attention_mask=(
                    attention_mask
                ),
                use_cache=False,
            )
        )

        reference_logits = (
            reference_output.logits
            .detach()
            .cpu()
            .clone()
        )

    print(
        "Reference logits shape   :",
        tuple(
            reference_logits.shape
        ),
    )

    # Free first full model before loading second.
    del reference_output
    del reference_model

    gc.collect()

    # ==============================================================
    # PATH B — PACKED CHECKPOINT HYBRID LOADER
    # ==============================================================

    print()
    print("=" * 100)
    print("PATH B — PACKED CHECKPOINT HYBRID LOADER")
    print("=" * 100)

    hybrid_model = (
        load_fp16_model()
    )

    hybrid_model.eval()

    # --------------------------------------------------------------
    # Verify controls match original before injection.
    # --------------------------------------------------------------

    print()
    print("-" * 100)
    print("PRE-INJECTION CONTROL CHECK")
    print("-" * 100)

    for name in CONTROL_MODULES:

        module = get_linear_module(
            hybrid_model,
            name,
        )

        comparison = (
            exact_tensor_comparison(
                module.weight
                .detach()
                .cpu(),
                original_controls[
                    name
                ],
            )
        )

        print(
            f"{name:<55} "
            f"mismatches="
            f"{comparison['mismatches']}"
        )

        assert (
            comparison[
                "mismatches"
            ]
            == 0
        )

    # --------------------------------------------------------------
    # Apply hybrid loader.
    # --------------------------------------------------------------

    print()
    print(
        "Applying packed checkpoint..."
    )

    load_result = (
        load_packed_weights_into_model(
            hybrid_model,
            CHECKPOINT_PATH,
            expected_model_name=(
                MODEL_NAME
            ),
        )
    )

    print()
    print(
        "Loader matrix count      :",
        load_result
        .replaced_matrix_count,
    )

    print(
        "Loader parameter count   :",
        load_result
        .replaced_parameter_count,
    )

    print(
        "Loader module names      :",
        load_result
        .loaded_module_names,
    )

    assert (
        load_result
        .replaced_matrix_count
        == len(TARGET_MODULES)
    )

    assert (
        load_result
        .replaced_parameter_count
        == EXPECTED_TARGET_PARAMETERS
    )

    assert (
        set(
            load_result
            .loaded_module_names
        )
        == set(TARGET_MODULES)
    )

    # ==============================================================
    # TARGET WEIGHT COMPARISON
    # ==============================================================

    print()
    print("=" * 100)
    print("TARGET WEIGHT VALIDATION")
    print("=" * 100)

    total_target_values = 0
    total_target_mismatches = 0
    global_target_max_difference = 0.0

    for name in TARGET_MODULES:

        module = get_linear_module(
            hybrid_model,
            name,
        )

        hybrid_weight = (
            module.weight
            .detach()
            .cpu()
        )

        direct_weight = (
            direct_target_weights[
                name
            ]
        )

        comparison = (
            exact_tensor_comparison(
                hybrid_weight,
                direct_weight,
            )
        )

        total_target_values += (
            hybrid_weight.numel()
        )

        total_target_mismatches += (
            comparison[
                "mismatches"
            ]
        )

        global_target_max_difference = max(
            global_target_max_difference,
            comparison[
                "max_difference"
            ],
        )

        print()
        print(name)

        print(
            "  values              :",
            hybrid_weight.numel(),
        )

        print(
            "  mismatches          :",
            comparison[
                "mismatches"
            ],
        )

        print(
            "  max abs difference  :",
            comparison[
                "max_difference"
            ],
        )

        print(
            "  mean abs difference :",
            comparison[
                "mean_difference"
            ],
        )

        print(
            "  shape match         :",
            comparison[
                "shape_match"
            ],
        )

        print(
            "  dtype match         :",
            comparison[
                "dtype_match"
            ],
        )

        assert (
            comparison[
                "shape_match"
            ]
        )

        assert (
            comparison[
                "dtype_match"
            ]
        )

        assert (
            comparison[
                "mismatches"
            ]
            == 0
        )

        assert (
            comparison[
                "max_difference"
            ]
            == 0.0
        )

    # ==============================================================
    # UNTOUCHED CONTROL VALIDATION
    # ==============================================================

    print()
    print("=" * 100)
    print("UNTOUCHED CONTROL VALIDATION")
    print("=" * 100)

    total_control_changes = 0

    for name in CONTROL_MODULES:

        module = get_linear_module(
            hybrid_model,
            name,
        )

        comparison = (
            exact_tensor_comparison(
                module.weight
                .detach()
                .cpu(),
                original_controls[
                    name
                ],
            )
        )

        total_control_changes += (
            comparison[
                "mismatches"
            ]
        )

        print()
        print(name)

        print(
            "  changed values       :",
            comparison[
                "mismatches"
            ],
        )

        print(
            "  max abs difference   :",
            comparison[
                "max_difference"
            ],
        )

        assert (
            comparison[
                "mismatches"
            ]
            == 0
        )

        assert (
            comparison[
                "max_difference"
            ]
            == 0.0
        )

    # ==============================================================
    # HYBRID FORWARD PASS
    # ==============================================================

    print()
    print(
        "Running hybrid forward pass..."
    )

    with torch.inference_mode():

        hybrid_output = (
            hybrid_model(
                input_ids=input_ids,
                attention_mask=(
                    attention_mask
                ),
                use_cache=False,
            )
        )

        hybrid_logits = (
            hybrid_output.logits
            .detach()
            .cpu()
        )

    # ==============================================================
    # LOGIT COMPARISON
    # ==============================================================

    print()
    print("=" * 100)
    print("FORWARD OUTPUT VALIDATION")
    print("=" * 100)

    logit_comparison = (
        exact_tensor_comparison(
            reference_logits,
            hybrid_logits,
        )
    )

    print(
        "Reference shape         :",
        tuple(
            reference_logits.shape
        ),
    )

    print(
        "Hybrid shape            :",
        tuple(
            hybrid_logits.shape
        ),
    )

    print(
        "Logit mismatches        :",
        logit_comparison[
            "mismatches"
        ],
    )

    print(
        "Max abs logit difference:",
        logit_comparison[
            "max_difference"
        ],
    )

    print(
        "Mean abs logit difference:",
        logit_comparison[
            "mean_difference"
        ],
    )

    print(
        "Shape match             :",
        logit_comparison[
            "shape_match"
        ],
    )

    print(
        "Dtype match             :",
        logit_comparison[
            "dtype_match"
        ],
    )

    # ==============================================================
    # FINAL ACCEPTANCE
    # ==============================================================

    print()
    print("=" * 100)
    print("FINAL ACCEPTANCE")
    print("=" * 100)

    print(
        "Checkpoint checksum     :",
        "PASS",
    )

    print(
        "Target matrices         :",
        len(TARGET_MODULES),
    )

    print(
        "Target parameters       :",
        total_target_values,
    )

    print(
        "Target mismatches       :",
        total_target_mismatches,
    )

    print(
        "Target global max diff  :",
        global_target_max_difference,
    )

    print(
        "Untouched control changes:",
        total_control_changes,
    )

    assert (
        total_target_values
        == EXPECTED_TARGET_PARAMETERS
    )

    assert (
        total_target_mismatches
        == 0
    )

    assert (
        global_target_max_difference
        == 0.0
    )

    assert (
        total_control_changes
        == 0
    )

    # Because both paths contain exactly the same FP16 model
    # parameters and run the same deterministic CPU forward path,
    # require exact logits.
    assert (
        logit_comparison[
            "shape_match"
        ]
    )

    assert (
        logit_comparison[
            "dtype_match"
        ]
    )

    assert (
        logit_comparison[
            "mismatches"
        ]
        == 0
    )

    assert (
        logit_comparison[
            "max_difference"
        ]
        == 0.0
    )

    print(
        "Direct vs checkpoint    : EXACT MATCH"
    )

    print(
        "Hybrid target injection : EXACT MATCH"
    )

    print(
        "Untouched controls      : EXACT MATCH"
    )

    print(
        "Forward logits          : EXACT MATCH"
    )

    print()
    print("=" * 100)
    print("ENG004 REAL HYBRID LOADER VALIDATION: PASS")
    print("=" * 100)


if __name__ == "__main__":
    main()
