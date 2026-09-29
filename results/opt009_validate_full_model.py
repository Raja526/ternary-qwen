import gc
import inspect

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import packed_ternary.avx2_runtime_loader as ref_loader
import packed_ternary.blocked_avx2_runtime_loader as cand_loader


MODEL_ID = "Qwen/Qwen3.5-0.8B"

CHECKPOINT = (
    "results/"
    "ENG003_multimatrix_checkpoint.pt"
)

SELECTED_MODULE_NAMES = (
    "model.layers.3.self_attn.k_proj",
    "model.layers.4.linear_attn.out_proj",
    "model.layers.4.linear_attn.in_proj_qkv",
)

PROMPT = (
    "A box contains 12 red balls and 8 blue balls. "
    "How many balls are in the box?"
)

CPU_THREADS = 6
SEED = 20260925


def find_loader(module):
    """
    Find the public runtime-loader function without assuming
    a specific historical function name.
    """
    preferred = [
        "load_blocked_avx2_runtime",
        "load_avx2_runtime",
        "load_runtime",
        "load_packed_runtime",
        "load_packed_ternary_runtime",
    ]

    for name in preferred:
        fn = getattr(module, name, None)

        if callable(fn):
            return fn

    candidates = []

    for name, obj in vars(module).items():
        if (
            callable(obj)
            and name.startswith("load_")
            and not name.startswith("load_model")
        ):
            candidates.append(
                (name, obj)
            )

    if len(candidates) == 1:
        return candidates[0][1]

    raise RuntimeError(
        "Could not uniquely identify runtime loader in "
        f"{module.__name__}. Candidates: "
        f"{[name for name, _ in candidates]}"
    )


def call_loader(
    loader_fn,
    model,
):
    """
    Call the frozen loader while adapting only to its
    existing argument names.
    """
    sig = inspect.signature(loader_fn)

    kwargs = {}

    for name in sig.parameters:
        lname = name.lower()

        if lname in {
            "model",
            "base_model",
        }:
            kwargs[name] = model

        elif lname in {
            "checkpoint",
            "checkpoint_path",
            "path",
            "packed_checkpoint",
            "packed_checkpoint_path",
        }:
            kwargs[name] = CHECKPOINT

    try:
        result = loader_fn(
            **kwargs
        )
    except TypeError:
        # Fallback for simple positional loaders.
        try:
            result = loader_fn(
                model,
                CHECKPOINT,
            )
        except TypeError:
            result = loader_fn(
                model,
            )

    # Most loaders mutate and return the model or return
    # metadata alongside it. The original model reference
    # remains valid in either case.
    if isinstance(
        result,
        torch.nn.Module,
    ):
        return result, result

    return model, result


def extract_stats(
    result,
):
    """
    Extract replacement/storage statistics when available.
    This is descriptive only and is not part of correctness.
    """
    stats = {}

    if isinstance(result, dict):
        stats.update(result)

    elif isinstance(
        result,
        (tuple, list),
    ):
        for item in result:
            if isinstance(item, dict):
                stats.update(item)

    return stats


def print_stats(
    label,
    stats,
):
    print()
    print(label)

    wanted = [
        "replaced_matrices",
        "replaced_parameters",
        "persistent_packed_bytes",
        "fp16_equivalent_bytes",
    ]

    for key in wanted:
        if key in stats:
            print(
                f"{key}:",
                stats[key],
            )


def load_base_model():
    torch.manual_seed(SEED)

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
        )
    )

    model.eval()
    model.to("cpu")

    return model


def run_path(
    loader_module,
    input_ids,
):
    model = load_base_model()

    if loader_module is ref_loader:
        loader_fn = (
            loader_module
            .load_avx2_runtime_modules
        )
    elif loader_module is cand_loader:
        loader_fn = (
            loader_module
            .load_blocked_avx2_runtime_modules
        )
    else:
        raise RuntimeError(
            f"Unknown runtime loader module: "
            f"{loader_module.__name__}"
        )

    loader_result = loader_fn(
        model=model,
        checkpoint_path=CHECKPOINT,
        selected_module_names=SELECTED_MODULE_NAMES,
        expected_model_name=MODEL_ID,
    )

    stats = extract_stats(
        loader_result
    )

    with torch.inference_mode():
        logits = (
            model(
                input_ids=input_ids,
            )
            .logits
            .detach()
            .cpu()
            .clone()
        )

    del model
    gc.collect()

    return logits, stats


def main():
    torch.set_num_threads(
        CPU_THREADS
    )

    try:
        torch.set_num_interop_threads(
            CPU_THREADS
        )
    except RuntimeError:
        pass

    torch.manual_seed(SEED)

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            MODEL_ID
        )
    )

    encoded = tokenizer(
        PROMPT,
        return_tensors="pt",
    )

    input_ids = encoded[
        "input_ids"
    ]

    print(
        "=" * 72
    )
    print(
        "OPT009 — FULL-MODEL BLOCKING EQUIVALENCE"
    )
    print(
        "=" * 72
    )

    print()
    print(
        "Reference : OPT008 explicit AVX2 decode"
    )
    print(
        "Candidate : OPT009 AVX2 + output-row blocking"
    )
    print(
        "Block rows: 8"
    )
    print(
        "Model     :",
        MODEL_ID,
    )
    print(
        "CPU threads:",
        CPU_THREADS,
    )
    print(
        "Input tokens:",
        int(input_ids.numel()),
    )

    print()
    print(
        "Running PATH A — frozen OPT008..."
    )

    ref_logits, ref_stats = (
        run_path(
            ref_loader,
            input_ids,
        )
    )

    print_stats(
        "PATH A statistics:",
        ref_stats,
    )

    print()
    print(
        "Running PATH B — OPT009 blocked..."
    )

    cand_logits, cand_stats = (
        run_path(
            cand_loader,
            input_ids,
        )
    )

    print_stats(
        "PATH B statistics:",
        cand_stats,
    )

    if (
        ref_logits.shape
        != cand_logits.shape
    ):
        raise RuntimeError(
            "FAIL: logits shapes differ."
        )

    exact = (
        ref_logits
        == cand_logits
    )

    mismatches = int(
        (~exact)
        .sum()
        .item()
    )

    diff = (
        cand_logits.float()
        - ref_logits.float()
    ).abs()

    max_abs = float(
        diff.max().item()
    )

    mean_abs = float(
        diff.mean().item()
    )

    ref_top1 = (
        ref_logits.argmax(
            dim=-1
        )
    )

    cand_top1 = (
        cand_logits.argmax(
            dim=-1
        )
    )

    top1_matches = int(
        (
            ref_top1
            == cand_top1
        )
        .sum()
        .item()
    )

    top1_total = int(
        ref_top1.numel()
    )

    compared = int(
        ref_logits.numel()
    )

    bit_exact = (
        mismatches == 0
        and max_abs == 0.0
    )

    print()
    print(
        "=" * 72
    )
    print(
        "COMPARISON"
    )
    print(
        "=" * 72
    )

    print(
        "Logits shape      :",
        tuple(ref_logits.shape),
    )
    print(
        "Compared logits   :",
        compared,
    )
    print(
        "Exact mismatches  :",
        mismatches,
    )
    print(
        "Max abs error     :",
        max_abs,
    )
    print(
        "Mean abs error    :",
        mean_abs,
    )
    print(
        "Bit-exact logits  :",
        bit_exact,
    )
    print(
        "Top-1 matches     :",
        f"{top1_matches}/{top1_total}",
    )

    if not bit_exact:
        raise RuntimeError(
            "FAIL: OPT009 full-model logits "
            "are not bit-exact with OPT008."
        )

    if (
        top1_matches
        != top1_total
    ):
        raise RuntimeError(
            "FAIL: top-1 behavior differs."
        )

    print()
    print(
        "PASS: OPT009 full-model behavior "
        "is bit-exact with frozen OPT008."
    )


if __name__ == "__main__":
    main()
