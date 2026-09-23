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
# EXPERIMENT 020
#
# Progressive combination of relatively behavior-preserving matrices identified
# by Experiment 019.
#
# Scientific question:
# Can individually tolerable ternary matrices be combined while preserving
# FP16 ARC-Challenge behavior?
#
# Important:
#   - Same frozen Exp018 evaluation protocol
#   - Same frozen ternary quantizer
#   - Same Exp018 FP16 baseline
#   - Fresh FP16 model for EVERY cumulative step
#   - Each step independently ternarizes the full cumulative matrix set
#   - Checkpoint after every completed step
# =============================================================================


EXP18_RESULT = "results/018_arc_challenge_predictions.json"

RESULT_JSON = "results/020_progressive_safe_arc.json"
RESULT_TXT = "results/020_progressive_safe_arc_summary.txt"


# =============================================================================
# LOAD EXP018
# =============================================================================

spec = importlib.util.spec_from_file_location(
    "exp18",
    "scripts/18_arc_challenge_validation.py",
)

exp18 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp18)


# =============================================================================
# FROZEN PROGRESSIVE ORDER
# =============================================================================

PROGRESSIVE_ORDER = [
    "model.layers.3.self_attn.k_proj",
    "model.layers.4.linear_attn.out_proj",
    "model.layers.4.linear_attn.in_proj_qkv",
    "model.layers.3.self_attn.q_proj",
    "model.layers.6.mlp.gate_proj",
    "model.layers.6.mlp.down_proj",
    "model.layers.6.mlp.up_proj",
]


# Matrices intentionally excluded based on Exp019 individual ablation.

EXCLUDED_MATRICES = [
    "model.layers.3.self_attn.v_proj",
    "model.layers.3.self_attn.o_proj",
    "model.layers.4.linear_attn.in_proj_z",
]


# =============================================================================
# STATISTICS
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
# FROZEN FP16 BASELINE
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
# MODEL MATRIX DISCOVERY
# =============================================================================

def discover_mixed_small(model):

    discovered = {}

    modules = exp18.discover_modules(
        model
    )

    for (
        name,
        module,
        component,
        block,
    ) in modules:

        discovered[name] = {
            "name": name,
            "module": module,
            "component": component,
            "block": block,
            "parameters":
                module.weight.numel(),
            "shape":
                list(module.weight.shape),
        }

    return discovered


def validate_matrix_configuration(
    discovered,
):

    expected = set(
        PROGRESSIVE_ORDER
        + EXCLUDED_MATRICES
    )

    actual = set(
        discovered.keys()
    )

    if expected != actual:

        missing = sorted(
            expected - actual
        )

        unexpected = sorted(
            actual - expected
        )

        raise RuntimeError(
            "MIXED_SMALL matrix mismatch.\n"
            f"Missing: {missing}\n"
            f"Unexpected: {unexpected}"
        )

    if len(PROGRESSIVE_ORDER) != 7:
        raise RuntimeError(
            "Expected exactly 7 progressive matrices."
        )

    if len(EXCLUDED_MATRICES) != 3:
        raise RuntimeError(
            "Expected exactly 3 excluded matrices."
        )

    if len(actual) != 10:
        raise RuntimeError(
            f"Expected 10 MIXED_SMALL matrices, "
            f"found {len(actual)}."
        )


# =============================================================================
# TERNARIZE CUMULATIVE SET
# =============================================================================

def ternarize_matrix_set(
    model,
    matrix_names,
):

    discovered = discover_mixed_small(
        model
    )

    validate_matrix_configuration(
        discovered
    )

    modules = []

    for name in matrix_names:

        if name not in discovered:
            raise RuntimeError(
                f"Matrix not found: {name}"
            )

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
# EVALUATION
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

        if result["correct"]:
            correct += 1

        question_id = result["id"]

        if question_id not in baseline["predictions"]:
            raise RuntimeError(
                f"Question {question_id} missing "
                f"from Exp018 FP16 baseline."
            )

        fp = baseline[
            "predictions"
        ][question_id]

        fp_correct = bool(
            fp["correct"]
        )

        ternary_correct = bool(
            result["correct"]
        )

        if (
            fp["prediction"]
            != result["prediction"]
        ):
            changed_predictions += 1

        if (
            fp_correct
            and ternary_correct
        ):
            both_correct += 1

        elif (
            fp_correct
            and not ternary_correct
        ):
            fp16_only += 1

        elif (
            not fp_correct
            and ternary_correct
        ):
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
        "correct":
            correct,

        "total":
            total,

        "accuracy":
            accuracy,

        "accuracy_difference":
            accuracy_difference,

        "both_correct":
            both_correct,

        "fp16_only":
            fp16_only,

        "ternary_only":
            ternary_only,

        "both_wrong":
            both_wrong,

        "changed_predictions":
            changed_predictions,

        "prediction_agreement":
            prediction_agreement,

        "mcnemar_exact_p":
            p_value,

        "time_seconds":
            elapsed,

        "predictions":
            predictions,
    }


# =============================================================================
# CHECKPOINTING
# =============================================================================

def load_checkpoint():

    if not os.path.exists(
        RESULT_JSON
    ):

        return {
            "experiment": 20,
            "results": [],
        }

    with open(
        RESULT_JSON,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def save_checkpoint(data):

    temporary = (
        RESULT_JSON + ".tmp"
    )

    with open(
        temporary,
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
        temporary,
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
        key=lambda row:
            row["step"],
    )

    lines = []

    lines.append(
        "=" * 125
    )

    lines.append(
        "EXPERIMENT 020 — PROGRESSIVE SAFE-MATRIX ARC-CHALLENGE VALIDATION"
    )

    lines.append(
        "=" * 125
    )

    lines.append("")

    lines.append(
        f"FP16 baseline: "
        f"{baseline['correct']}/"
        f"{baseline['total']} "
        f"({100 * baseline['accuracy']:.2f}%)"
    )

    lines.append("")

    lines.append(
        "Excluded from progression:"
    )

    for name in EXCLUDED_MATRICES:
        lines.append(
            f"  - {name}"
        )

    lines.append("")

    header = (
        f"{'Step':<5} "
        f"{'Added matrix':<18} "
        f"{'Matrices':>8} "
        f"{'Params':>11} "
        f"{'Cov%':>8} "
        f"{'Correct':>9} "
        f"{'Acc%':>8} "
        f"{'Delta':>9} "
        f"{'Agree%':>9} "
        f"{'Changed':>8} "
        f"{'FPonly':>8} "
        f"{'Tonly':>7} "
        f"{'McNemar p':>14}"
    )

    lines.append(
        header
    )

    lines.append(
        "-" * 125
    )

    for row in ordered:

        added = (
            row["added_matrix"]
            .split(".")[-1]
        )

        lines.append(
            f"{row['step']:<5} "
            f"{added:<18} "
            f"{row['matrix_count']:>8} "
            f"{row['parameters']:>11,d} "
            f"{row['coverage_percent']:>8.4f} "
            f"{row['correct']:>9} "
            f"{100 * row['accuracy']:>8.2f} "
            f"{100 * row['accuracy_difference']:>+8.2f} "
            f"{100 * row['prediction_agreement']:>8.2f} "
            f"{row['changed_predictions']:>8} "
            f"{row['fp16_only']:>8} "
            f"{row['ternary_only']:>7} "
            f"{row['mcnemar_exact_p']:>14.6g}"
        )

    text = "\n".join(
        lines
    )

    with open(
        RESULT_TXT,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            text + "\n"
        )

    print()
    print(text)


# =============================================================================
# MAIN
# =============================================================================

def main():

    print(
        "=" * 90
    )

    print(
        "EXPERIMENT 020 — PROGRESSIVE SAFE-MATRIX ARC VALIDATION"
    )

    print(
        "=" * 90
    )

    # -------------------------------------------------------------------------
    # Baseline
    # -------------------------------------------------------------------------

    baseline = (
        load_fp16_baseline()
    )

    print()
    print(
        f"Frozen FP16 baseline: "
        f"{baseline['correct']}/"
        f"{baseline['total']} "
        f"= "
        f"{100 * baseline['accuracy']:.2f}%"
    )

    # -------------------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------------------

    dataset = load_dataset(
        exp18.DATASET_ID,
        exp18.DATASET_CONFIG,
        split=exp18.DATASET_SPLIT,
    )

    if len(dataset) != baseline["total"]:

        raise RuntimeError(
            "Dataset size does not match "
            "Exp018 FP16 baseline."
        )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            exp18.MODEL_ID
        )
    )

    # -------------------------------------------------------------------------
    # Validate frozen matrix configuration once before experiment.
    # -------------------------------------------------------------------------

    validation_model = (
        exp18.load_model()
    )

    discovered = (
        discover_mixed_small(
            validation_model
        )
    )

    validate_matrix_configuration(
        discovered
    )

    total_model_parameters = sum(
        parameter.numel()
        for parameter
        in validation_model.parameters()
    )

    print()
    print(
        f"Total model parameters: "
        f"{total_model_parameters:,}"
    )

    print()
    print(
        "Progressive order:"
    )

    cumulative_parameters = 0

    for index, name in enumerate(
        PROGRESSIVE_ORDER,
        start=1,
    ):

        params = (
            discovered[name][
                "parameters"
            ]
        )

        cumulative_parameters += params

        coverage = (
            100.0
            * cumulative_parameters
            / total_model_parameters
        )

        print(
            f"{index}. "
            f"{name} "
            f"| +{params:,} "
            f"| cumulative="
            f"{cumulative_parameters:,} "
            f"| coverage="
            f"{coverage:.4f}%"
        )

    print()
    print(
        "Excluded matrices:"
    )

    for name in EXCLUDED_MATRICES:

        print(
            f"  {name}"
        )

    exp18.release_model(
        validation_model
    )

    # -------------------------------------------------------------------------
    # Checkpoint
    # -------------------------------------------------------------------------

    checkpoint = (
        load_checkpoint()
    )

    checkpoint[
        "model"
    ] = exp18.MODEL_ID

    checkpoint[
        "dataset"
    ] = {
        "id":
            exp18.DATASET_ID,

        "config":
            exp18.DATASET_CONFIG,

        "split":
            exp18.DATASET_SPLIT,

        "questions":
            len(dataset),
    }

    checkpoint[
        "fp16_baseline"
    ] = {
        "correct":
            baseline["correct"],

        "total":
            baseline["total"],

        "accuracy":
            baseline["accuracy"],
    }

    checkpoint[
        "progressive_order"
    ] = PROGRESSIVE_ORDER

    checkpoint[
        "excluded_matrices"
    ] = EXCLUDED_MATRICES

    completed_steps = {
        row["step"]
        for row
        in checkpoint["results"]
    }

    # -------------------------------------------------------------------------
    # Progressive experiment
    # -------------------------------------------------------------------------

    for step in range(
        1,
        len(PROGRESSIVE_ORDER) + 1,
    ):

        if step in completed_steps:

            print()
            print(
                f"STEP {step} already completed. "
                f"Skipping."
            )

            continue

        matrix_names = (
            PROGRESSIVE_ORDER[:step]
        )

        added_matrix = (
            PROGRESSIVE_ORDER[
                step - 1
            ]
        )

        print()
        print(
            "=" * 90
        )

        print(
            f"STEP {step}/"
            f"{len(PROGRESSIVE_ORDER)}"
        )

        print(
            f"Added matrix: "
            f"{added_matrix}"
        )

        print(
            "=" * 90
        )

        print()
        print(
            "Cumulative matrices:"
        )

        for name in matrix_names:
            print(
                f"  {name}"
            )

        # ---------------------------------------------------------------------
        # Fresh FP16 model for every step.
        # ---------------------------------------------------------------------

        model = (
            exp18.load_model()
        )

        # ---------------------------------------------------------------------
        # Ternarize entire cumulative set independently.
        # ---------------------------------------------------------------------

        quant_stats, current_discovered = (
            ternarize_matrix_set(
                model,
                matrix_names,
            )
        )

        expected_parameters = sum(
            current_discovered[
                name
            ]["parameters"]
            for name
            in matrix_names
        )

        if (
            quant_stats["parameters"]
            != expected_parameters
        ):

            raise RuntimeError(
                "Quantized parameter count mismatch: "
                f"expected "
                f"{expected_parameters:,}, "
                f"got "
                f"{quant_stats['parameters']:,}"
            )

        coverage = (
            100.0
            * expected_parameters
            / total_model_parameters
        )

        print()
        print(
            f"Matrix count     : "
            f"{len(matrix_names)}"
        )

        print(
            f"Parameters       : "
            f"{expected_parameters:,}"
        )

        print(
            f"Coverage         : "
            f"{coverage:.4f}%"
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

        # ---------------------------------------------------------------------
        # ARC benchmark
        # ---------------------------------------------------------------------

        evaluation = (
            evaluate_configuration(
                model,
                tokenizer,
                dataset,
                baseline,
            )
        )

        exp18.release_model(
            model
        )

        row = {
            "step":
                step,

            "added_matrix":
                added_matrix,

            "matrix_names":
                list(matrix_names),

            "matrix_count":
                len(matrix_names),

            "parameters":
                expected_parameters,

            "coverage_percent":
                coverage,

            "mean_error":
                quant_stats[
                    "mean_error"
                ],

            "max_error":
                quant_stats[
                    "max_error"
                ],

            "mean_scale_ratio":
                quant_stats[
                    "mean_scale_ratio"
                ],

            **evaluation,
        }

        checkpoint[
            "results"
        ].append(
            row
        )

        save_checkpoint(
            checkpoint
        )

        print()
        print(
            f"STEP {step} RESULT"
        )

        print(
            f"Correct              : "
            f"{evaluation['correct']}/"
            f"{evaluation['total']}"
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

        # Update human-readable summary after every completed step.

        write_summary(
            baseline,
            checkpoint["results"],
        )

        gc.collect()
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------------

    write_summary(
        baseline,
        checkpoint["results"],
    )

    print()
    print(
        f"Detailed results: "
        f"{RESULT_JSON}"
    )

    print(
        f"Summary: "
        f"{RESULT_TXT}"
    )


if __name__ == "__main__":
    main()
