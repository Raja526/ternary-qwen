import gc
import importlib.util
import itertools
import json
import math
import os
import time

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


# =============================================================================
# EXPERIMENT 022
#
# Pairwise compatibility study among the seven candidate matrices from Exp020.
#
# Scientific question:
# How does pairwise joint ternarization affect functional behavior among
# individually characterized matrices, and can pairwise compatibility help
# explain the cumulative degradation observed in Exp020?
#
# Frozen:
#   - Exp018 ARC-Challenge evaluation protocol
#   - Exp018 FP16 predictions
#   - optimized groupwise ternary quantizer
#   - fresh FP16 model for every pair
# =============================================================================


EXP18_RESULT = "results/018_arc_challenge_predictions.json"
EXP19_RESULT = "results/019_matrix_arc_ablation.json"

RESULT_JSON = "results/022_pairwise_compatibility_arc.json"
RESULT_TXT = "results/022_pairwise_compatibility_arc_summary.txt"


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
# SEVEN CANDIDATE MATRICES
# =============================================================================

CANDIDATES = [
    {
        "short": "k_proj",
        "name": "model.layers.3.self_attn.k_proj",
    },
    {
        "short": "out_proj",
        "name": "model.layers.4.linear_attn.out_proj",
    },
    {
        "short": "in_proj_qkv",
        "name": "model.layers.4.linear_attn.in_proj_qkv",
    },
    {
        "short": "q_proj",
        "name": "model.layers.3.self_attn.q_proj",
    },
    {
        "short": "gate_proj",
        "name": "model.layers.6.mlp.gate_proj",
    },
    {
        "short": "down_proj",
        "name": "model.layers.6.mlp.down_proj",
    },
    {
        "short": "up_proj",
        "name": "model.layers.6.mlp.up_proj",
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
# GENERATE EXACTLY 21 UNIQUE PAIRS
# =============================================================================

PAIRS = []

for index, (a, b) in enumerate(
    itertools.combinations(CANDIDATES, 2),
    start=1,
):
    PAIRS.append(
        {
            "id": index,
            "a_short": a["short"],
            "b_short": b["short"],
            "a_name": a["name"],
            "b_name": b["name"],
            "matrices": [
                a["name"],
                b["name"],
            ],
        }
    )


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
# LOAD FP16 BASELINE
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
# LOAD EXP019 INDIVIDUAL RESULTS
# =============================================================================

def load_individual_results():

    with open(
        EXP19_RESULT,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    # Exp019 stores one result per individual matrix.
    rows = data["results"]

    lookup = {}

    for row in rows:

        matrix_name = row.get("matrix_name")

        # Compatibility with alternate field naming if present.
        if matrix_name is None:
            matrix_name = row.get("name")

        if matrix_name is None:
            raise RuntimeError(
                "Could not locate matrix name field "
                "in Exp019 result row."
            )

        lookup[matrix_name] = row

    missing = [
        candidate["name"]
        for candidate in CANDIDATES
        if candidate["name"] not in lookup
    ]

    if missing:
        raise RuntimeError(
            "Candidate matrices missing from Exp019: "
            f"{missing}"
        )

    return lookup


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
# CONFIGURATION VALIDATION
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
            "MIXED_SMALL matrix mismatch.\n"
            f"Missing: {missing}\n"
            f"Unexpected: {unexpected}"
        )

    if len(CANDIDATES) != 7:
        raise RuntimeError(
            f"Expected 7 candidates, "
            f"found {len(CANDIDATES)}."
        )

    candidate_names = [
        candidate["name"]
        for candidate in CANDIDATES
    ]

    if len(candidate_names) != len(set(candidate_names)):
        raise RuntimeError(
            "Duplicate candidate matrix."
        )

    candidate_short = [
        candidate["short"]
        for candidate in CANDIDATES
    ]

    if len(candidate_short) != len(set(candidate_short)):
        raise RuntimeError(
            "Duplicate candidate short name."
        )

    if len(PAIRS) != 21:
        raise RuntimeError(
            f"Expected 21 pairs, "
            f"found {len(PAIRS)}."
        )

    canonical_pairs = set()

    for pair in PAIRS:

        a = pair["a_name"]
        b = pair["b_name"]

        if a == b:
            raise RuntimeError(
                f"Self-pair found: {a}"
            )

        if a not in discovered:
            raise RuntimeError(
                f"Unknown matrix: {a}"
            )

        if b not in discovered:
            raise RuntimeError(
                f"Unknown matrix: {b}"
            )

        canonical = tuple(
            sorted([a, b])
        )

        if canonical in canonical_pairs:
            raise RuntimeError(
                f"Duplicate/reversed pair: "
                f"{canonical}"
            )

        canonical_pairs.add(
            canonical
        )

    if len(canonical_pairs) != 21:
        raise RuntimeError(
            "Unique pair count is not 21."
        )


# =============================================================================
# TERNARIZATION
# =============================================================================

def ternarize_pair(
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
                f"Question {question_id} missing "
                "from Exp018 FP16 baseline."
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

        if ternary_correct:
            correct += 1

        if (
            fp["prediction"]
            != result["prediction"]
        ):
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
# INDIVIDUAL DELTA EXTRACTION
# =============================================================================

def get_accuracy_delta(
    row,
    baseline_accuracy,
):

    if "accuracy_difference" in row:
        return float(
            row["accuracy_difference"]
        )

    if "accuracy" in row:
        return (
            float(row["accuracy"])
            - baseline_accuracy
        )

    if (
        "correct" in row
        and "total" in row
    ):
        return (
            float(row["correct"])
            / float(row["total"])
            - baseline_accuracy
        )

    raise RuntimeError(
        "Cannot determine individual "
        "accuracy delta from Exp019 row."
    )


# =============================================================================
# CHECKPOINT
# =============================================================================

def load_checkpoint():

    if not os.path.exists(
        RESULT_JSON
    ):
        return {
            "experiment": 22,
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
        key=lambda row:
            row["pair_id"],
    )

    lines = []

    lines.append("=" * 157)

    lines.append(
        "EXPERIMENT 022 — PAIRWISE MATRIX COMPATIBILITY ON ARC-CHALLENGE"
    )

    lines.append("=" * 157)

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
        f"{'Matrix A':<13} "
        f"{'Matrix B':<13} "
        f"{'Params':>11} "
        f"{'Cov%':>8} "
        f"{'Correct':>8} "
        f"{'Acc%':>8} "
        f"{'Delta':>9} "
        f"{'Agree%':>9} "
        f"{'Changed':>8} "
        f"{'FPonly':>7} "
        f"{'Tonly':>7} "
        f"{'p':>11} "
        f"{'I_acc':>9}"
    )

    lines.append(header)
    lines.append("-" * 157)

    for row in ordered:

        lines.append(
            f"{row['pair_id']:<3} "
            f"{row['a_short']:<13} "
            f"{row['b_short']:<13} "
            f"{row['parameters']:>11,d} "
            f"{row['coverage_percent']:>8.4f} "
            f"{row['correct']:>8} "
            f"{100 * row['accuracy']:>8.2f} "
            f"{100 * row['accuracy_difference']:>+8.2f} "
            f"{100 * row['prediction_agreement']:>8.2f} "
            f"{row['changed_predictions']:>8} "
            f"{row['fp16_only']:>7} "
            f"{row['ternary_only']:>7} "
            f"{row['mcnemar_exact_p']:>11.4g} "
            f"{100 * row['accuracy_interaction']:>+8.2f}"
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

    print("=" * 95)
    print(
        "EXPERIMENT 022 — PAIRWISE MATRIX COMPATIBILITY"
    )
    print("=" * 95)

    # -------------------------------------------------------------------------
    # Baseline and Exp019 individual data
    # -------------------------------------------------------------------------

    baseline = (
        load_fp16_baseline()
    )

    individual = (
        load_individual_results()
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
            "Dataset size mismatch with "
            "Exp018 baseline."
        )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            exp18.MODEL_ID
        )
    )

    # -------------------------------------------------------------------------
    # Validate configuration
    # -------------------------------------------------------------------------

    validation_model = (
        exp18.load_model()
    )

    discovered = (
        discover_mixed_small(
            validation_model
        )
    )

    validate_configuration(
        discovered
    )

    total_model_parameters = sum(
        parameter.numel()
        for parameter
        in validation_model.parameters()
    )

    print()
    print(
        f"Candidate matrices       : "
        f"{len(CANDIDATES)}"
    )

    print(
        f"Unique pair combinations : "
        f"{len(PAIRS)}"
    )

    print(
        f"Total model parameters   : "
        f"{total_model_parameters:,}"
    )

    print()
    print("PAIR CONFIGURATION")
    print("-" * 95)

    for pair in PAIRS:

        parameters = (
            discovered[
                pair["a_name"]
            ]["parameters"]
            +
            discovered[
                pair["b_name"]
            ]["parameters"]
        )

        coverage = (
            100.0
            * parameters
            / total_model_parameters
        )

        print(
            f"{pair['id']:02d}. "
            f"{pair['a_short']:<13} + "
            f"{pair['b_short']:<13} "
            f"params={parameters:,} "
            f"coverage={coverage:.4f}%"
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

    checkpoint["model"] = (
        exp18.MODEL_ID
    )

    checkpoint["dataset"] = {
        "id": exp18.DATASET_ID,
        "config": exp18.DATASET_CONFIG,
        "split": exp18.DATASET_SPLIT,
        "questions": len(dataset),
    }

    checkpoint["fp16_baseline"] = {
        "correct":
            baseline["correct"],

        "total":
            baseline["total"],

        "accuracy":
            baseline["accuracy"],
    }

    checkpoint["candidates"] = (
        CANDIDATES
    )

    checkpoint["pairs"] = (
        PAIRS
    )

    completed = {
        row["pair_id"]
        for row
        in checkpoint["results"]
    }

    # -------------------------------------------------------------------------
    # Evaluate all 21 pairs
    # -------------------------------------------------------------------------

    for pair in PAIRS:

        pair_id = pair["id"]

        if pair_id in completed:

            print()
            print(
                f"Pair {pair_id:02d} "
                "already completed. Skipping."
            )

            continue

        print()
        print("=" * 95)

        print(
            f"PAIR {pair_id:02d}/21 — "
            f"{pair['a_short']} + "
            f"{pair['b_short']}"
        )

        print("=" * 95)

        # Fresh model for every pair.
        model = (
            exp18.load_model()
        )

        quant_stats, current_discovered = (
            ternarize_pair(
                model,
                pair["matrices"],
            )
        )

        expected_parameters = sum(
            current_discovered[
                name
            ]["parameters"]
            for name
            in pair["matrices"]
        )

        if (
            quant_stats["parameters"]
            != expected_parameters
        ):
            raise RuntimeError(
                "Quantized parameter count "
                "mismatch."
            )

        coverage = (
            100.0
            * expected_parameters
            / total_model_parameters
        )

        print()
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
        # ARC
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

        # ---------------------------------------------------------------------
        # Interaction calculation
        #
        # I_acc(A,B) =
        #     delta(A+B) - [delta(A) + delta(B)]
        # ---------------------------------------------------------------------

        delta_a = get_accuracy_delta(
            individual[
                pair["a_name"]
            ],
            baseline["accuracy"],
        )

        delta_b = get_accuracy_delta(
            individual[
                pair["b_name"]
            ],
            baseline["accuracy"],
        )

        expected_additive_delta = (
            delta_a + delta_b
        )

        accuracy_interaction = (
            evaluation[
                "accuracy_difference"
            ]
            - expected_additive_delta
        )

        row = {
            "pair_id":
                pair_id,

            "a_short":
                pair["a_short"],

            "b_short":
                pair["b_short"],

            "a_name":
                pair["a_name"],

            "b_name":
                pair["b_name"],

            "matrix_names":
                pair["matrices"],

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

            "individual_delta_a":
                delta_a,

            "individual_delta_b":
                delta_b,

            "expected_additive_delta":
                expected_additive_delta,

            "accuracy_interaction":
                accuracy_interaction,

            **evaluation,
        }

        checkpoint[
            "results"
        ].append(row)

        save_checkpoint(
            checkpoint
        )

        print()
        print(
            f"PAIR {pair_id:02d} RESULT"
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

        print(
            f"Individual delta A   : "
            f"{100 * delta_a:+.2f} pp"
        )

        print(
            f"Individual delta B   : "
            f"{100 * delta_b:+.2f} pp"
        )

        print(
            f"Additive expectation : "
            f"{100 * expected_additive_delta:+.2f} pp"
        )

        print(
            f"Accuracy interaction : "
            f"{100 * accuracy_interaction:+.2f} pp"
        )

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
