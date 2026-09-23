import gc
import importlib.util
import json
import math
import os
import time

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


# =============================================================================
# EXPERIMENT 024
#
# Safe-Core Higher-Order Interaction Study
#
# Frozen core:
#   k_proj
#   out_proj
#   in_proj_qkv
#
# Conditions:
#   A = core + q_proj      (Exp020 Step-4 reproducibility anchor)
#   B = core + gate_proj
#   C = core + down_proj
#   D = core + up_proj
#
# Exactly ONE matrix changes relative to the same three-matrix core.
# =============================================================================


EXP18_RESULT = "results/018_arc_challenge_predictions.json"

RESULT_JSON = "results/024_safe_core_interactions_arc.json"
RESULT_TXT = "results/024_safe_core_interactions_arc_summary.txt"


# =============================================================================
# LOAD EXP018 IMPLEMENTATION
# =============================================================================

spec = importlib.util.spec_from_file_location(
    "exp18",
    "scripts/18_arc_challenge_validation.py",
)

exp18 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp18)


# =============================================================================
# FROZEN CORE
# =============================================================================

CORE = [
    "model.layers.3.self_attn.k_proj",
    "model.layers.4.linear_attn.out_proj",
    "model.layers.4.linear_attn.in_proj_qkv",
]


CONDITIONS = [
    {
        "id": "A",
        "name": "core_plus_q_proj",
        "added_short": "q_proj",
        "added_name": "model.layers.3.self_attn.q_proj",
    },
    {
        "id": "B",
        "name": "core_plus_gate_proj",
        "added_short": "gate_proj",
        "added_name": "model.layers.6.mlp.gate_proj",
    },
    {
        "id": "C",
        "name": "core_plus_down_proj",
        "added_short": "down_proj",
        "added_name": "model.layers.6.mlp.down_proj",
    },
    {
        "id": "D",
        "name": "core_plus_up_proj",
        "added_short": "up_proj",
        "added_name": "model.layers.6.mlp.up_proj",
    },
]


EXPECTED_MIXED_SMALL = {
    "model.layers.3.self_attn.q_proj",
    "model.layers.3.self_attn.k_proj",
    "model.layers.3.self_attn.v_proj",
    "model.layers.3.self_attn.o_proj",
    "model.layers.4.linear_attn.out_proj",
    "model.layers.4.linear_attn.in_proj_qkv",
    "model.layers.4.linear_attn.in_proj_z",
    "model.layers.6.mlp.gate_proj",
    "model.layers.6.mlp.up_proj",
    "model.layers.6.mlp.down_proj",
}


# =============================================================================
# MCNEMAR
# =============================================================================

def exact_mcnemar(b, c):

    n = b + c

    if n == 0:
        return 1.0

    k = min(b, c)

    one_tail = sum(
        math.comb(n, i)
        for i in range(k + 1)
    ) / (2 ** n)

    return min(
        1.0,
        2.0 * one_tail,
    )


# =============================================================================
# BASELINE
# =============================================================================

def load_fp16_baseline():

    with open(
        EXP18_RESULT,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    fp16 = data["fp16"]

    predictions = {
        row["id"]: row
        for row in fp16["predictions"]
    }

    return {
        "correct": fp16["correct"],
        "total": fp16["total"],
        "accuracy": fp16["accuracy"],
        "predictions": predictions,
    }


# =============================================================================
# MATRIX DISCOVERY
# =============================================================================

def discover_mixed_small(model):

    discovered = {}

    modules = exp18.discover_modules(model)

    for name, module, component, block in modules:

        discovered[name] = {
            "name": name,
            "module": module,
            "component": component,
            "block": block,
            "parameters": module.weight.numel(),
            "shape": list(module.weight.shape),
        }

    return discovered


# =============================================================================
# VALIDATION
# =============================================================================

def validate_configuration(discovered):

    actual = set(discovered.keys())

    if actual != EXPECTED_MIXED_SMALL:

        missing = sorted(
            EXPECTED_MIXED_SMALL - actual
        )

        unexpected = sorted(
            actual - EXPECTED_MIXED_SMALL
        )

        raise RuntimeError(
            "MIXED_SMALL discovery mismatch.\n"
            f"Missing: {missing}\n"
            f"Unexpected: {unexpected}"
        )

    if len(CORE) != 3:
        raise RuntimeError(
            f"Expected exactly 3 core matrices, found {len(CORE)}."
        )

    if len(set(CORE)) != 3:
        raise RuntimeError(
            "Duplicate matrix in CORE."
        )

    for name in CORE:

        if name not in discovered:
            raise RuntimeError(
                f"Core matrix not discovered: {name}"
            )

    if len(CONDITIONS) != 4:
        raise RuntimeError(
            f"Expected 4 conditions, found {len(CONDITIONS)}."
        )

    ids = [
        condition["id"]
        for condition in CONDITIONS
    ]

    if len(ids) != len(set(ids)):
        raise RuntimeError(
            "Duplicate condition ID."
        )

    added_names = [
        condition["added_name"]
        for condition in CONDITIONS
    ]

    if len(added_names) != len(set(added_names)):
        raise RuntimeError(
            "Duplicate added matrix."
        )

    for condition in CONDITIONS:

        added = condition["added_name"]

        if added not in discovered:
            raise RuntimeError(
                f"Added matrix not discovered: {added}"
            )

        if added in CORE:
            raise RuntimeError(
                f"Added matrix already belongs to core: {added}"
            )

        matrices = CORE + [added]

        if len(matrices) != 4:
            raise RuntimeError(
                "Condition does not contain exactly four matrices."
            )

        if len(set(matrices)) != 4:
            raise RuntimeError(
                f"Duplicate matrix in condition {condition['id']}."
            )

        if set(matrices) - EXPECTED_MIXED_SMALL:
            raise RuntimeError(
                f"Unexpected matrix in condition {condition['id']}."
            )


# =============================================================================
# TERNARIZATION
# =============================================================================

def ternarize_condition(
    model,
    matrix_names,
):

    discovered = discover_mixed_small(
        model
    )

    validate_configuration(
        discovered
    )

    modules = []

    for name in matrix_names:

        item = discovered[name]

        modules.append(
            (
                item["name"],
                item["module"],
                item["component"],
                item["block"],
            )
        )

    stats = exp18.ternarize_modules(
        modules,
        method="optimized",
    )

    return stats, discovered


# =============================================================================
# ARC EVALUATION
# =============================================================================

def evaluate_configuration(
    model,
    tokenizer,
    dataset,
    baseline,
):

    predictions = []

    correct = 0

    both_correct = 0
    fp16_only = 0
    ternary_only = 0
    both_wrong = 0

    changed_predictions = 0

    start = time.perf_counter()

    for index, example in enumerate(
        dataset,
        start=1,
    ):

        result = exp18.evaluate_question(
            model,
            tokenizer,
            example,
        )

        predictions.append(
            result
        )

        question_id = result["id"]

        if question_id not in baseline["predictions"]:
            raise RuntimeError(
                f"Question {question_id} missing from FP16 baseline."
            )

        fp = baseline["predictions"][question_id]

        fp_correct = bool(
            fp["correct"]
        )

        ternary_correct = bool(
            result["correct"]
        )

        if ternary_correct:
            correct += 1

        if fp["prediction"] != result["prediction"]:
            changed_predictions += 1

        if fp_correct and ternary_correct:
            both_correct += 1

        elif fp_correct and not ternary_correct:
            fp16_only += 1

        elif not fp_correct and ternary_correct:
            ternary_only += 1

        else:
            both_wrong += 1

        print(
            f"[{index:03d}/{len(dataset)}] "
            f"id={question_id} "
            f"gold={result['answer_key']} "
            f"fp16={fp['prediction']} "
            f"ternary={result['prediction']} "
            f"{'CORRECT' if ternary_correct else 'WRONG'}"
        )

    elapsed = (
        time.perf_counter()
        - start
    )

    total = len(dataset)

    accuracy = (
        correct / total
    )

    accuracy_difference = (
        accuracy
        - baseline["accuracy"]
    )

    prediction_agreement = (
        1.0
        - changed_predictions / total
    )

    p_value = exact_mcnemar(
        fp16_only,
        ternary_only,
    )

    return {
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "accuracy_difference": accuracy_difference,

        "both_correct": both_correct,
        "fp16_only": fp16_only,
        "ternary_only": ternary_only,
        "both_wrong": both_wrong,

        "changed_predictions": changed_predictions,
        "prediction_agreement": prediction_agreement,

        "mcnemar_exact_p": p_value,

        "time_seconds": elapsed,

        "predictions": predictions,
    }


# =============================================================================
# CHECKPOINT
# =============================================================================

def load_checkpoint():

    if not os.path.exists(
        RESULT_JSON
    ):
        return {
            "experiment": 24,
            "results": [],
        }

    with open(
        RESULT_JSON,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def save_checkpoint(data):

    temp_path = (
        RESULT_JSON + ".tmp"
    )

    with open(
        temp_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )

    os.replace(
        temp_path,
        RESULT_JSON,
    )


# =============================================================================
# SUMMARY
# =============================================================================

def write_summary(
    baseline,
    results,
):

    ordered = sorted(
        results,
        key=lambda row: row["condition_id"],
    )

    lines = []

    lines.append("=" * 142)

    lines.append(
        "EXPERIMENT 024 — SAFE-CORE HIGHER-ORDER INTERACTION STUDY"
    )

    lines.append("=" * 142)

    lines.append("")

    lines.append(
        f"FP16 baseline: "
        f"{baseline['correct']}/"
        f"{baseline['total']} "
        f"({100 * baseline['accuracy']:.2f}%)"
    )

    lines.append("")

    lines.append(
        "Frozen core: k_proj + out_proj + in_proj_qkv"
    )

    lines.append("")

    header = (
        f"{'ID':<4} "
        f"{'Added matrix':<14} "
        f"{'Matrices':>8} "
        f"{'Params':>11} "
        f"{'Cov%':>8} "
        f"{'Correct':>8} "
        f"{'Acc%':>8} "
        f"{'Delta':>9} "
        f"{'Agree%':>9} "
        f"{'Changed':>8} "
        f"{'FPonly':>7} "
        f"{'Tonly':>7} "
        f"{'McNemar p':>12}"
    )

    lines.append(header)
    lines.append("-" * 142)

    for row in ordered:

        lines.append(
            f"{row['condition_id']:<4} "
            f"{row['added_short']:<14} "
            f"{row['matrix_count']:>8} "
            f"{row['parameters']:>11,d} "
            f"{row['coverage_percent']:>8.4f} "
            f"{row['correct']:>8} "
            f"{100 * row['accuracy']:>8.2f} "
            f"{100 * row['accuracy_difference']:>+8.2f} "
            f"{100 * row['prediction_agreement']:>8.2f} "
            f"{row['changed_predictions']:>8} "
            f"{row['fp16_only']:>7} "
            f"{row['ternary_only']:>7} "
            f"{row['mcnemar_exact_p']:>12.6g}"
        )

    text = "\n".join(lines)

    with open(
        RESULT_TXT,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(text + "\n")

    print()
    print(text)


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 100)
    print(
        "EXPERIMENT 024 — SAFE-CORE HIGHER-ORDER INTERACTION STUDY"
    )
    print("=" * 100)

    baseline = load_fp16_baseline()

    print()
    print(
        f"Frozen FP16 baseline: "
        f"{baseline['correct']}/"
        f"{baseline['total']} "
        f"= {100 * baseline['accuracy']:.2f}%"
    )

    dataset = load_dataset(
        exp18.DATASET_ID,
        exp18.DATASET_CONFIG,
        split=exp18.DATASET_SPLIT,
    )

    if len(dataset) != baseline["total"]:
        raise RuntimeError(
            "Dataset size does not match Exp018 baseline."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        exp18.MODEL_ID
    )

    # -------------------------------------------------------------------------
    # Structural sanity check
    # -------------------------------------------------------------------------

    validation_model = exp18.load_model()

    discovered = discover_mixed_small(
        validation_model
    )

    validate_configuration(
        discovered
    )

    total_model_parameters = sum(
        p.numel()
        for p in validation_model.parameters()
    )

    core_parameters = sum(
        discovered[name]["parameters"]
        for name in CORE
    )

    print()
    print("STRUCTURAL CHECK")
    print("-" * 100)

    print(
        f"Core matrices          : {len(CORE)}"
    )

    print(
        f"Core parameters        : {core_parameters:,}"
    )

    print(
        f"Core coverage          : "
        f"{100 * core_parameters / total_model_parameters:.4f}%"
    )

    print(
        f"Conditions             : {len(CONDITIONS)}"
    )

    print(
        f"Total model parameters : {total_model_parameters:,}"
    )

    print()
    print("CONDITIONS")
    print("-" * 100)

    for condition in CONDITIONS:

        matrix_names = (
            CORE
            + [condition["added_name"]]
        )

        parameters = sum(
            discovered[name]["parameters"]
            for name in matrix_names
        )

        coverage = (
            100.0
            * parameters
            / total_model_parameters
        )

        print(
            f"{condition['id']}  "
            f"core + {condition['added_short']:<10} "
            f"matrices={len(matrix_names)}  "
            f"params={parameters:,}  "
            f"coverage={coverage:.4f}%"
        )

    exp18.release_model(
        validation_model
    )

    # -------------------------------------------------------------------------
    # Checkpoint
    # -------------------------------------------------------------------------

    checkpoint = load_checkpoint()

    checkpoint["model"] = exp18.MODEL_ID

    checkpoint["dataset"] = {
        "id": exp18.DATASET_ID,
        "config": exp18.DATASET_CONFIG,
        "split": exp18.DATASET_SPLIT,
        "questions": len(dataset),
    }

    checkpoint["fp16_baseline"] = {
        "correct": baseline["correct"],
        "total": baseline["total"],
        "accuracy": baseline["accuracy"],
    }

    checkpoint["core"] = CORE
    checkpoint["conditions"] = CONDITIONS

    completed = {
        row["condition_id"]
        for row in checkpoint["results"]
    }

    # -------------------------------------------------------------------------
    # Evaluate four conditions
    # -------------------------------------------------------------------------

    for condition in CONDITIONS:

        condition_id = condition["id"]

        if condition_id in completed:

            print()
            print(
                f"Condition {condition_id} already completed. Skipping."
            )

            continue

        matrix_names = (
            CORE
            + [condition["added_name"]]
        )

        print()
        print("=" * 100)

        print(
            f"CONDITION {condition_id} — "
            f"CORE + {condition['added_short']}"
        )

        print("=" * 100)

        # Fresh FP16 model for every condition.
        model = exp18.load_model()

        quant_stats, current_discovered = (
            ternarize_condition(
                model,
                matrix_names,
            )
        )

        expected_parameters = sum(
            current_discovered[name]["parameters"]
            for name in matrix_names
        )

        if (
            quant_stats["parameters"]
            != expected_parameters
        ):
            raise RuntimeError(
                "Quantized parameter count mismatch."
            )

        coverage = (
            100.0
            * expected_parameters
            / total_model_parameters
        )

        print()
        print(
            f"Matrices         : {len(matrix_names)}"
        )

        print(
            f"Parameters       : {expected_parameters:,}"
        )

        print(
            f"Coverage         : {coverage:.4f}%"
        )

        print(
            f"Mean error       : "
            f"{quant_stats['mean_error']:.9f}"
        )

        print(
            f"Max error        : "
            f"{quant_stats['max_error']:.9f}"
        )

        print(
            f"Mean scale ratio : "
            f"{quant_stats['mean_scale_ratio']:.9f}"
        )

        evaluation = evaluate_configuration(
            model,
            tokenizer,
            dataset,
            baseline,
        )

        exp18.release_model(
            model
        )

        row = {
            "condition_id": condition_id,
            "condition_name": condition["name"],

            "added_short": condition["added_short"],
            "added_name": condition["added_name"],

            "matrix_names": matrix_names,
            "matrix_count": len(matrix_names),

            "parameters": expected_parameters,
            "coverage_percent": coverage,

            "mean_error": quant_stats["mean_error"],
            "max_error": quant_stats["max_error"],
            "mean_scale_ratio":
                quant_stats["mean_scale_ratio"],

            **evaluation,
        }

        checkpoint["results"].append(
            row
        )

        save_checkpoint(
            checkpoint
        )

        print()
        print(
            f"CONDITION {condition_id} RESULT"
        )

        print(
            f"Correct              : "
            f"{evaluation['correct']}/{evaluation['total']}"
        )

        print(
            f"Accuracy             : "
            f"{100 * evaluation['accuracy']:.2f}%"
        )

        print(
            f"Delta vs FP16        : "
            f"{100 * evaluation['accuracy_difference']:+.2f} pp"
        )

        print(
            f"Prediction agreement : "
            f"{100 * evaluation['prediction_agreement']:.2f}%"
        )

        print(
            f"Changed predictions  : "
            f"{evaluation['changed_predictions']}"
        )

        print(
            f"FP16 only correct    : "
            f"{evaluation['fp16_only']}"
        )

        print(
            f"Ternary only correct : "
            f"{evaluation['ternary_only']}"
        )

        print(
            f"McNemar exact p      : "
            f"{evaluation['mcnemar_exact_p']:.12g}"
        )

        write_summary(
            baseline,
            checkpoint["results"],
        )

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_summary(
        baseline,
        checkpoint["results"],
    )

    print()
    print(
        f"Detailed results: {RESULT_JSON}"
    )

    print(
        f"Summary: {RESULT_TXT}"
    )


if __name__ == "__main__":
    main()
