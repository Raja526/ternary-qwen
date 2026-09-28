#!/usr/bin/env bash
set -euo pipefail

echo "=========================================="
echo " OPT008 Stage 3 — Full-Model Equivalence"
echo "=========================================="

EXPECTED_BRANCH="opt008-avx2-ternary-compute"
EXPECTED_PARENT="cd2f52b"

echo
echo "[1/5] Checking experiment state..."

if [[ "$(git branch --show-current)" != "$EXPECTED_BRANCH" ]]; then
    echo "ERROR: Expected branch $EXPECTED_BRANCH"
    exit 1
fi

if [[ "$(git rev-parse HEAD)" != "$(git rev-parse "$EXPECTED_PARENT")" ]]; then
    echo "ERROR: HEAD moved away from frozen OPT007 parent."
    git log -1 --oneline
    exit 1
fi

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "ERROR: .venv is not active."
    exit 1
fi

if [[ ! -s results/OPT008_operator_equivalence.txt ]]; then
    echo "ERROR: Gate-1 report missing."
    exit 1
fi

grep -q \
"PASS: OPT008 operator is bit-exact with frozen OPT006." \
results/OPT008_operator_equivalence.txt || {
    echo "ERROR: Gate 1 has not passed."
    exit 1
}

if [[ ! -s results/OPT008_avx2_machine_code.txt ]]; then
    echo "ERROR: Stage-2B machine-code report missing."
    exit 1
fi

grep -q \
"Explicit AVX2 instructions are present" \
results/OPT008_avx2_machine_code.txt || {
    echo "ERROR: AVX2 machine-code gate has not passed."
    exit 1
}

echo "PASS: prerequisite gates confirmed."

# ---------------------------------------------------------
# 2. Create OPT008 linear wrapper.
# ---------------------------------------------------------

echo
echo "[2/5] Creating AVX2 runtime layer..."

cat > packed_ternary/avx2_linear.py <<'PY'
"""
OPT008 explicit AVX2 packed ternary linear layer.

Controlled difference from frozen OPT006:
    extension implementation only.

Persistent packed representation, validation,
forward shape handling, scale layout, bias handling,
and operator arguments are inherited unchanged from
TableDispatchPackedTernaryLinear.
"""

from packed_ternary.table_dispatch_linear import (
    TableDispatchPackedTernaryLinear,
)

from packed_ternary.opt008_extension import (
    load_opt008_extension,
)


class AVX2PackedTernaryLinear(
    TableDispatchPackedTernaryLinear
):
    """
    OPT008 runtime wrapper.

    Only _get_extension() differs from frozen OPT006.
    """

    def _get_extension(self):
        if self._extension is None:
            self._extension = (
                load_opt008_extension(
                    verbose=False
                )
            )

        return self._extension
PY

# ---------------------------------------------------------
# 3. Create OPT008 runtime loader from frozen OPT006.
# ---------------------------------------------------------

echo
echo "[3/5] Creating AVX2 runtime loader..."

cp \
    packed_ternary/table_dispatch_runtime_loader.py \
    packed_ternary/avx2_runtime_loader.py

python - <<'PY'
from pathlib import Path

p = Path(
    "packed_ternary/avx2_runtime_loader.py"
)

s = p.read_text()

s = s.replace(
    "from packed_ternary.table_dispatch_linear import (\n"
    "    TableDispatchPackedTernaryLinear,\n"
    ")",
    "from packed_ternary.avx2_linear import (\n"
    "    AVX2PackedTernaryLinear,\n"
    ")",
)

s = s.replace(
    "TableDispatchPackedTernaryLinear",
    "AVX2PackedTernaryLinear",
)

s = s.replace(
    "TableDispatchRuntimeLoadResult",
    "AVX2RuntimeLoadResult",
)

s = s.replace(
    "load_table_dispatch_runtime_modules",
    "load_avx2_runtime_modules",
)

p.write_text(s)

print(
    "PASS: AVX2 runtime loader created "
    "from frozen OPT006 loader."
)
PY

grep -q \
"class AVX2RuntimeLoadResult" \
packed_ternary/avx2_runtime_loader.py || {
    echo "ERROR: AVX2 load-result class missing."
    exit 1
}

grep -q \
"def load_avx2_runtime_modules" \
packed_ternary/avx2_runtime_loader.py || {
    echo "ERROR: AVX2 loader function missing."
    exit 1
}

grep -q \
"AVX2PackedTernaryLinear" \
packed_ternary/avx2_runtime_loader.py || {
    echo "ERROR: AVX2 layer not used by loader."
    exit 1
}

echo "PASS: runtime integration structure."

# ---------------------------------------------------------
# 4. Create strict full-model validator.
# ---------------------------------------------------------

echo
echo "[4/5] Creating full-model validation..."

cat > results/opt008_validate_full_model.py <<'PY'
import gc
from pathlib import Path

import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from packed_ternary.table_dispatch_runtime_loader import (
    load_table_dispatch_runtime_modules,
)

from packed_ternary.avx2_runtime_loader import (
    load_avx2_runtime_modules,
)


MODEL_NAME = "Qwen/Qwen3.5-0.8B"

CHECKPOINT_PATH = Path(
    "results/ENG003_multimatrix_checkpoint.pt"
)

TARGETS = (
    "model.layers.3.self_attn.k_proj",
    "model.layers.4.linear_attn.out_proj",
    "model.layers.4.linear_attn.in_proj_qkv",
)

PROMPT = (
    "A box contains 12 red balls and 8 blue balls. "
    "How many balls are in the box?"
)

SEED = 20260925
CPU_THREADS = 6


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


def run_model(
    model,
    input_ids,
    attention_mask,
):
    kwargs = {
        "input_ids": input_ids,
    }

    if attention_mask is not None:
        kwargs["attention_mask"] = (
            attention_mask
        )

    with torch.inference_mode():
        logits = model(
            **kwargs
        ).logits

    return (
        logits
        .detach()
        .cpu()
        .clone()
    )


def main():
    torch.manual_seed(SEED)

    torch.set_num_threads(
        CPU_THREADS
    )

    section(
        "OPT008 — FULL-MODEL AVX2 EQUIVALENCE"
    )

    print(
        "Reference                : OPT006 table dispatch"
    )
    print(
        "Candidate                : OPT008 explicit AVX2 decode"
    )
    print(
        "Model                    :",
        MODEL_NAME,
    )
    print(
        "Checkpoint               :",
        CHECKPOINT_PATH,
    )
    print(
        "Selected matrices        :",
        len(TARGETS),
    )
    print(
        "CPU threads              :",
        torch.get_num_threads(),
    )

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
        "Input tokens             :",
        input_ids.shape[1],
    )

    # --------------------------------------------------
    # PATH A — frozen OPT006.
    # --------------------------------------------------

    section(
        "PATH A — FROZEN OPT006 TABLE DISPATCH"
    )

    reference_model = load_model()

    reference_load = (
        load_table_dispatch_runtime_modules(
            model=reference_model,
            checkpoint_path=CHECKPOINT_PATH,
            selected_module_names=TARGETS,
            expected_model_name=MODEL_NAME,
        )
    )

    print(
        "Replaced matrices        :",
        reference_load.replaced_matrix_count,
    )
    print(
        "Replaced parameters      :",
        reference_load.replaced_parameter_count,
    )
    print(
        "Persistent packed bytes  :",
        reference_load.persistent_packed_bytes,
    )
    print(
        "FP16 equivalent bytes    :",
        reference_load.fp16_equivalent_bytes,
    )

    reference_logits = run_model(
        reference_model,
        input_ids,
        attention_mask,
    )

    print(
        "Reference logits shape   :",
        tuple(reference_logits.shape),
    )

    del reference_model
    gc.collect()

    # --------------------------------------------------
    # PATH B — OPT008.
    # --------------------------------------------------

    section(
        "PATH B — OPT008 EXPLICIT AVX2"
    )

    candidate_model = load_model()

    candidate_load = (
        load_avx2_runtime_modules(
            model=candidate_model,
            checkpoint_path=CHECKPOINT_PATH,
            selected_module_names=TARGETS,
            expected_model_name=MODEL_NAME,
        )
    )

    print(
        "Replaced matrices        :",
        candidate_load.replaced_matrix_count,
    )
    print(
        "Replaced parameters      :",
        candidate_load.replaced_parameter_count,
    )
    print(
        "Persistent packed bytes  :",
        candidate_load.persistent_packed_bytes,
    )
    print(
        "FP16 equivalent bytes    :",
        candidate_load.fp16_equivalent_bytes,
    )

    # Structural gate.
    assert (
        reference_load.replaced_matrix_count
        == candidate_load.replaced_matrix_count
        == 3
    )

    assert (
        reference_load.replaced_parameter_count
        == candidate_load.replaced_parameter_count
        == 8_912_896
    )

    assert (
        reference_load.persistent_packed_bytes
        == candidate_load.persistent_packed_bytes
    )

    assert (
        reference_load.fp16_equivalent_bytes
        == candidate_load.fp16_equivalent_bytes
    )

    candidate_logits = run_model(
        candidate_model,
        input_ids,
        attention_mask,
    )

    print(
        "Candidate logits shape   :",
        tuple(candidate_logits.shape),
    )

    del candidate_model
    gc.collect()

    # --------------------------------------------------
    # Strict comparison.
    # --------------------------------------------------

    section(
        "OPT006 VS OPT008 LOGIT COMPARISON"
    )

    assert (
        reference_logits.shape
        == candidate_logits.shape
    )

    assert (
        reference_logits.dtype
        == candidate_logits.dtype
    )

    compared = int(
        reference_logits.numel()
    )

    mismatches = int(
        (
            reference_logits
            != candidate_logits
        )
        .sum()
        .item()
    )

    difference = (
        candidate_logits.float()
        - reference_logits.float()
    )

    max_abs = float(
        difference
        .abs()
        .max()
        .item()
    )

    mean_abs = float(
        difference
        .abs()
        .double()
        .mean()
        .item()
    )

    bit_exact = torch.equal(
        reference_logits,
        candidate_logits,
    )

    reference_top1 = (
        reference_logits.argmax(
            dim=-1
        )
    )

    candidate_top1 = (
        candidate_logits.argmax(
            dim=-1
        )
    )

    top1_matches = int(
        (
            reference_top1
            == candidate_top1
        )
        .sum()
        .item()
    )

    top1_total = int(
        reference_top1.numel()
    )

    print(
        "Compared logits          :",
        compared,
    )
    print(
        "Exact mismatches         :",
        mismatches,
    )
    print(
        "Max abs difference       :",
        max_abs,
    )
    print(
        "Mean abs difference      :",
        mean_abs,
    )
    print(
        "Bit-exact logits         :",
        bit_exact,
    )
    print(
        "Top-1 matches            :",
        f"{top1_matches}/{top1_total}",
    )

    if mismatches != 0:
        raise RuntimeError(
            "FAIL: OPT008 full-model logits "
            "are not bit-exact with OPT006."
        )

    if max_abs != 0.0:
        raise RuntimeError(
            "FAIL: non-zero maximum difference."
        )

    if mean_abs != 0.0:
        raise RuntimeError(
            "FAIL: non-zero mean difference."
        )

    if not bit_exact:
        raise RuntimeError(
            "FAIL: torch.equal returned False."
        )

    if top1_matches != top1_total:
        raise RuntimeError(
            "FAIL: top-1 predictions differ."
        )

    section(
        "OPT008 GATE 2 — PASS"
    )

    print(
        "OPT008 full-model runtime is bit-exact "
        "with frozen OPT006."
    )


if __name__ == "__main__":
    main()
PY

# ---------------------------------------------------------
# 5. Run full-model gate.
# ---------------------------------------------------------

echo
echo "[5/5] Running strict full-model equivalence..."

PYTHONPATH="$PWD" \
python results/opt008_validate_full_model.py \
| tee results/OPT008_full_model_equivalence.txt

grep -q \
"OPT008 full-model runtime is bit-exact with frozen OPT006." \
results/OPT008_full_model_equivalence.txt || {
    echo
    echo "ERROR: full-model equivalence gate failed."
    exit 1
}

echo
echo "=========================================="
echo " OPT008 STAGE 3 COMPLETE"
echo "=========================================="
echo
echo "PASS:"
echo "  OPT008 full-model runtime is bit-exact"
echo "  with frozen OPT006."
echo
echo "Report:"
echo "  results/OPT008_full_model_equivalence.txt"
echo
echo "IMPORTANT:"
echo "  No performance benchmark performed."
echo "  No commit or push performed."
