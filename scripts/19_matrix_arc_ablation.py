import gc
import json
import math
import os
import time

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

import importlib.util

# =============================================================================
# EXPERIMENT 019
#
# Matrix-level ARC-Challenge ablation of MIXED_SMALL.
#
# Each run:
#   1. Loads a fresh FP16 model
#   2. Ternarizes exactly ONE matrix
#   3. Evaluates all 299 ARC-Challenge validation questions
#   4. Compares against frozen Exp018 FP16 predictions
#   5. Saves immediately
#
# Exp018 evaluation protocol and frozen quantizer are reused unchanged.
# =============================================================================

EXP18_RESULT = "results/018_arc_challenge_predictions.json"

RESULT_JSON = "results/019_matrix_arc_ablation.json"
RESULT_TXT = "results/019_matrix_arc_ablation_summary.txt"


# =============================================================================
# LOAD EXP018 CODE
# =============================================================================

spec = importlib.util.spec_from_file_location(
    "exp18",
    "scripts/18_arc_challenge_validation.py",
)

exp18 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp18)


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
# LOAD FROZEN FP16 BASELINE
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
# DISCOVER THE EXACT MIXED_SMALL MATRICES
# =============================================================================

def discover_target_names():

    model = exp18.load_model()

    modules = exp18.discover_modules(
        model
    )

    targets = []

    for (
        name,
        module,
        component,
        block,
    ) in modules:

        targets.append(
            {
                "name": name,
                "component": component,
                "block": block,
                "parameters":
                    module.weight.numel(),
                "shape":
                    list(module.weight.shape),
            }
        )

    exp18.release_model(model)

    return targets


# =============================================================================
# FIND ONE MATRIX IN A FRESH MODEL
# =============================================================================

def find_matrix(
    model,
    target_name,
):

    for name, module in model.named_modules():

        if name == target_name:

            if not isinstance(
                module,
                torch.nn.Linear,
            ):
                raise RuntimeError(
                    f"{target_name} is not Linear"
                )

            return module

    raise RuntimeError(
        f"Could not find matrix: "
        f"{target_name}"
    )


# =============================================================================
# TERNARIZE EXACTLY ONE MATRIX
# =============================================================================

def ternarize_one_matrix(
    model,
    target,
):

    module = find_matrix(
        model,
        target["name"],
    )

    # ternarize_modules expects tuples with:
    # name, module, component, block

    modules = [
        (
            target["name"],
            module,
            target["component"],
            target["block"],
        )
    ]

    stats = exp18.ternarize_modules(
        modules,
        method="optimized",
    )

    return stats


# =============================================================================
# EVALUATE ONE MATRIX
# =============================================================================

def evaluate_single_matrix(
    model,
    tokenizer,
    dataset,
    target,
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

        fp = baseline[
            "predictions"
        ][result["id"]]

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
            f"gold={result['answer_key']} "
            f"fp16={fp['prediction']} "
            f"ternary={result['prediction']} "
            f"{'CORRECT' if result['correct'] else 'WRONG'}"
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
            "experiment": 19,
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
        key=lambda x:
            x["accuracy"],
        reverse=True,
    )

    lines = []

    lines.append(
        "=" * 120
    )

    lines.append(
        "EXPERIMENT 019 — MATRIX-LEVEL ARC-CHALLENGE ABLATION"
    )

    lines.append(
        "=" * 120
    )

    lines.append("")

    lines.append(
        f"FP16 baseline: "
        f"{baseline['correct']}/"
        f"{baseline['total']} "
        f"({100 * baseline['accuracy']:.2f}%)"
    )

    lines.append("")

    header = (
        f"{'#':<3} "
        f"{'Component':<12} "
        f"{'Block':<6} "
        f"{'Matrix':<18} "
        f"{'Params':>10} "
        f"{'Cov%':>8} "
        f"{'Correct':>9} "
        f"{'Acc%':>8} "
        f"{'Delta':>9} "
        f"{'Agree%':>9} "
        f"{'FPonly':>8} "
        f"{'Tonly':>7} "
        f"{'McNemar p':>14}"
    )

    lines.append(
        header
    )

    lines.append(
        "-" * 120
    )

    for index, row in enumerate(
        ordered,
        start=1,
    ):

        short_name = (
            row["matrix_name"]
            .split(".")[-1]
        )

        lines.append(
            f"{index:<3} "
            f"{row['component']:<12} "
            f"{row['block']:<6} "
            f"{short_name:<18} "
            f"{row['parameters']:>10,d} "
            f"{row['coverage_percent']:>8.4f} "
            f"{row['correct']:>9} "
            f"{100 * row['accuracy']:>8.2f} "
            f"{100 * row['accuracy_difference']:>+8.2f} "
            f"{100 * row['prediction_agreement']:>8.2f} "
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
        f.write(text + "\n")

    print()
    print(text)


# =============================================================================
# MAIN
# =============================================================================

def main():

    print(
        "=" * 80
    )

    print(
        "EXPERIMENT 019 — MATRIX-LEVEL ARC ABLATION"
    )

    print(
        "=" * 80
    )

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

    dataset = load_dataset(
        exp18.DATASET_ID,
        exp18.DATASET_CONFIG,
        split=exp18.DATASET_SPLIT,
    )

    if len(dataset) != baseline["total"]:
        raise RuntimeError(
            "Dataset size does not match "
            "Exp018 baseline."
        )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            exp18.MODEL_ID
        )
    )

    targets = (
        discover_target_names()
    )

    print()
    print(
        f"MIXED_SMALL matrices: "
        f"{len(targets)}"
    )

    if len(targets) != 10:
        raise RuntimeError(
            f"Expected 10 matrices, "
            f"found {len(targets)}"
        )

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

    completed = {
        row["matrix_name"]
        for row
        in checkpoint["results"]
    }

    total_model_parameters = None

    for target_index, target in enumerate(
        targets,
        start=1,
    ):

        name = target["name"]

        if name in completed:

            print()
            print(
                f"[{target_index}/"
                f"{len(targets)}] "
                f"SKIPPING COMPLETED:"
            )

            print(
                name
            )

            continue

        print()
        print(
            "=" * 80
        )

        print(
            f"MATRIX "
            f"{target_index}/"
            f"{len(targets)}"
        )

        print(
            name
        )

        print(
            "=" * 80
        )

        # -------------------------------------------------------------
        # Fresh model for every matrix.
        # -------------------------------------------------------------

        model = exp18.load_model()

        if total_model_parameters is None:

            total_model_parameters = sum(
                parameter.numel()
                for parameter
                in model.parameters()
            )

        # -------------------------------------------------------------
        # Ternarize exactly one matrix.
        # -------------------------------------------------------------

        quant_stats = (
            ternarize_one_matrix(
                model,
                target,
            )
        )

        coverage = (
            100.0
            * target["parameters"]
            / total_model_parameters
        )

        print()
        print(
            f"Parameters       : "
            f"{target['parameters']:,}"
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

        # -------------------------------------------------------------
        # ARC evaluation.
        # -------------------------------------------------------------

        evaluation = (
            evaluate_single_matrix(
                model,
                tokenizer,
                dataset,
                target,
                baseline,
            )
        )

        exp18.release_model(
            model
        )

        row = {
            "matrix_name":
                name,

            "component":
                target["component"],

            "block":
                target["block"],

            "shape":
                target["shape"],

            "parameters":
                target["parameters"],

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
            f"RESULT: "
            f"{evaluation['correct']}/"
            f"{evaluation['total']} "
            f"= "
            f"{100 * evaluation['accuracy']:.2f}%"
        )

        print(
            f"Delta vs FP16: "
            f"{100 * evaluation['accuracy_difference']:+.2f} pp"
        )

        print(
            f"Prediction agreement: "
            f"{100 * evaluation['prediction_agreement']:.2f}%"
        )

        print(
            f"McNemar exact p: "
            f"{evaluation['mcnemar_exact_p']:.12g}"
        )

        # Summary is updated after every matrix too.

        write_summary(
            baseline,
            checkpoint["results"],
        )

        gc.collect()
        torch.cuda.empty_cache()

    # =========================================================================
    # FINAL
    # =========================================================================

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
