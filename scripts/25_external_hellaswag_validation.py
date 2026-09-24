import gc
import json
import math
import time
from pathlib import Path

import torch
from datasets import load_dataset
from scipy.stats import binomtest
from transformers import AutoModelForCausalLM, AutoTokenizer

from ternary.quantizer import groupwise_ternarize_weight


# =============================================================================
# EXPERIMENT 025 — EXTERNAL HELLASWAG VALIDATION
#
# Model:
#     Qwen/Qwen3.5-0.8B
#
# Dataset:
#     Rowan/hellaswag
#     validation split
#
# Frozen evaluation subset:
#     First 2,000 validation examples
#
# Scoring:
#     Mean conditional log probability per continuation token
#
# Candidate execution:
#     Four candidate endings scored SEQUENTIALLY.
#
# Reason:
#     Sequential scoring avoids CUDA OOM on the 4 GB Quadro P1000.
#     It does NOT change the frozen scoring definition.
#
# Configurations:
#     FP16
#     K_ONLY
#     SAFE_CORE
#     CORE_PLUS_Q
#
# Frozen ternary quantizer:
#     method = "optimized"
#     group_size = 128
#     threshold_multiplier = 0.5
#
# IMPORTANT:
#     Numerical ternarization is reconstructed into FP16.
#     This experiment does NOT implement packed ternary storage
#     and does NOT claim inference speedup or VRAM reduction.
# =============================================================================


# =============================================================================
# EXPERIMENT SETTINGS
# =============================================================================

MODEL_ID = "Qwen/Qwen3.5-0.8B"

DATASET_ID = "Rowan/hellaswag"

SPLIT = "validation"

SUBSET_SIZE = 2000


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

DTYPE = torch.float16


# =============================================================================
# OUTPUT FILES
# =============================================================================

RESULT_DIR = Path("results")

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


CHECKPOINT_PATH = (
    RESULT_DIR
    / "025_hellaswag_2000_checkpoint.json"
)


FINAL_JSON_PATH = (
    RESULT_DIR
    / "025_external_hellaswag_2000_validation.json"
)


SUMMARY_PATH = (
    RESULT_DIR
    / "025_external_hellaswag_2000_validation_summary.txt"
)


# =============================================================================
# FROZEN CONFIGURATIONS
# =============================================================================

CONFIGURATIONS = {

    "FP16": [],

    "K_ONLY": [
        "model.layers.3.self_attn.k_proj",
    ],

    "SAFE_CORE": [
        "model.layers.3.self_attn.k_proj",
        "model.layers.4.linear_attn.out_proj",
        "model.layers.4.linear_attn.in_proj_qkv",
    ],

    "CORE_PLUS_Q": [
        "model.layers.3.self_attn.k_proj",
        "model.layers.4.linear_attn.out_proj",
        "model.layers.4.linear_attn.in_proj_qkv",
        "model.layers.3.self_attn.q_proj",
    ],
}


# =============================================================================
# MEMORY MANAGEMENT
# =============================================================================

def clear_memory():

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model():

    print(
        f"Loading model: {MODEL_ID}"
    )

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            MODEL_ID,
            torch_dtype=DTYPE,
        )
    )

    model.to(DEVICE)

    model.eval()

    return model


# =============================================================================
# MODULE LOOKUP
# =============================================================================

def get_module(
    model,
    module_name,
):

    current = model

    for part in module_name.split("."):

        if part.isdigit():

            current = current[
                int(part)
            ]

        else:

            current = getattr(
                current,
                part,
            )

    return current


# =============================================================================
# FROZEN TERNARY QUANTIZATION
# =============================================================================

def ternarize_configuration(
    model,
    matrix_names,
):

    """
    Apply the frozen Exp025 ternary quantizer.

    Settings:

        method = optimized
        group_size = 128
        threshold_multiplier = 0.5
    """

    stats = []

    with torch.no_grad():

        for name in matrix_names:

            print(
                f"Ternarizing: {name}"
            )

            module = get_module(
                model,
                name,
            )

            original_weight = (
                module.weight.data
            )

            result = (
                groupwise_ternarize_weight(
                    original_weight,
                    method="optimized",
                    group_size=128,
                    threshold_multiplier=0.5,
                )
            )

            reconstructed = (
                result["reconstructed"]
            )

            module.weight.data.copy_(
                reconstructed.to(
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                )
            )

            parameters = (
                module.weight.numel()
            )

            total_codes = (
                result["negative"]
                + result["zero"]
                + result["positive"]
            )

            if total_codes != parameters:

                raise RuntimeError(
                    f"Code-count mismatch for "
                    f"{name}: "
                    f"{total_codes:,} != "
                    f"{parameters:,}"
                )

            stats.append({

                "matrix_name":
                    name,

                "parameters":
                    parameters,

                "negative":
                    result["negative"],

                "zero":
                    result["zero"],

                "positive":
                    result["positive"],

                "mean_initial_alpha":
                    result[
                        "mean_initial_alpha"
                    ],

                "mean_reconstruction_alpha":
                    result[
                        "mean_reconstruction_alpha"
                    ],

                "scale_ratio":
                    result[
                        "scale_ratio"
                    ],
            })

    return stats


# =============================================================================
# HELLASWAG INPUT PREPARATION
# =============================================================================

def prepare_example(
    tokenizer,
    example,
):

    context = (
        example["ctx"]
        .strip()
    )

    if not context:

        raise RuntimeError(
            "Empty HellaSwag context."
        )

    endings = (
        example["endings"]
    )

    if len(endings) != 4:

        raise RuntimeError(
            "Expected four HellaSwag endings, "
            f"found {len(endings)}."
        )

    context_ids = tokenizer(
        context,
        add_special_tokens=False,
    )["input_ids"]

    if len(context_ids) == 0:

        raise RuntimeError(
            "Context produced zero tokens."
        )

    rows = []

    for (
        ending_index,
        ending,
    ) in enumerate(endings):

        continuation = ending

        # ---------------------------------------------------------
        # Frozen concatenation rule:
        #
        # Insert exactly one space only when neither side already
        # provides whitespace at the context/ending boundary.
        # ---------------------------------------------------------

        if (
            context
            and continuation
            and not context[-1].isspace()
            and not continuation[0].isspace()
        ):

            continuation = (
                " " + continuation
            )

        full_text = (
            context
            + continuation
        )

        full_ids = tokenizer(
            full_text,
            add_special_tokens=False,
        )["input_ids"]

        # ---------------------------------------------------------
        # Exp025A exhaustive validation:
        #
        # HellaSwag validation examples : 10,042
        # Context-ending pairs          : 40,168
        # Boundary mismatches           : 0
        #
        # Keep the assertion active during Exp025.
        # ---------------------------------------------------------

        if (
            full_ids[
                :len(context_ids)
            ]
            != context_ids
        ):

            raise RuntimeError(
                "Token boundary mismatch "
                "encountered."
            )

        continuation_length = (
            len(full_ids)
            - len(context_ids)
        )

        if continuation_length <= 0:

            raise RuntimeError(
                "Continuation produced "
                "zero additional tokens."
            )

        rows.append({

            "ending_index":
                ending_index,

            "input_ids":
                full_ids,

            "context_length":
                len(context_ids),

            "continuation_length":
                continuation_length,
        })

    return rows


# =============================================================================
# HELLASWAG SCORING
# =============================================================================

@torch.inference_mode()
def score_example(
    model,
    tokenizer,
    example,
):

    """
    Score the four HellaSwag candidate endings sequentially.

    Frozen scoring definition:

        score =
            mean conditional log probability
            per continuation token

        prediction =
            ending with highest mean log probability

    Sequential candidate execution is used only to reduce
    GPU-memory requirements.
    """

    rows = prepare_example(
        tokenizer,
        example,
    )

    scores = []

    for row in rows:

        input_ids = torch.tensor(
            row["input_ids"],
            dtype=torch.long,
            device=DEVICE,
        ).unsqueeze(0)

        outputs = model(
            input_ids=input_ids,
            use_cache=False,
        )

        logits = (
            outputs.logits[0]
            .float()
        )

        context_length = (
            row[
                "context_length"
            ]
        )

        continuation_length = (
            row[
                "continuation_length"
            ]
        )

        start_token = (
            context_length
        )

        end_token = (
            context_length
            + continuation_length
        )

        continuation_ids = (
            input_ids[
                0,
                start_token:end_token,
            ]
        )

        # ---------------------------------------------------------
        # Causal alignment:
        #
        # logits[t - 1] predicts token[t]
        # ---------------------------------------------------------

        prediction_logits = (
            logits[
                start_token - 1:
                end_token - 1,
                :
            ]
        )

        if (
            prediction_logits.shape[0]
            != continuation_ids.shape[0]
        ):

            raise RuntimeError(
                "Continuation/logit "
                "alignment mismatch."
            )

        log_probs = (
            torch.log_softmax(
                prediction_logits,
                dim=-1,
            )
        )

        token_log_probs = (
            log_probs
            .gather(
                dim=-1,
                index=(
                    continuation_ids
                    .unsqueeze(-1)
                ),
            )
            .squeeze(-1)
        )

        total_logprob = (
            token_log_probs
            .sum()
            .item()
        )

        token_count = (
            continuation_ids
            .numel()
        )

        mean_logprob = (
            total_logprob
            / token_count
        )

        if not math.isfinite(
            mean_logprob
        ):

            raise RuntimeError(
                "Non-finite continuation "
                "score encountered."
            )

        scores.append({

            "ending_index":
                row["ending_index"],

            "token_count":
                token_count,

            "total_logprob":
                total_logprob,

            "mean_logprob":
                mean_logprob,
        })

        # ---------------------------------------------------------
        # Explicitly release the large vocabulary tensors before
        # evaluating the next candidate ending.
        # ---------------------------------------------------------

        del token_log_probs
        del log_probs
        del prediction_logits
        del continuation_ids
        del logits
        del outputs
        del input_ids

    predicted = max(
        scores,
        key=lambda row:
            row["mean_logprob"],
    )["ending_index"]

    gold = int(
        example["label"]
    )

    if gold not in {
        0,
        1,
        2,
        3,
    }:

        raise RuntimeError(
            f"Unexpected gold label: "
            f"{gold}"
        )

    return {

        "prediction":
            predicted,

        "gold":
            gold,

        "correct":
            predicted == gold,

        "scores":
            scores,
    }


# =============================================================================
# CONFIGURATION EVALUATION
# =============================================================================

def evaluate_configuration(
    model,
    tokenizer,
    dataset,
    config_name,
):

    predictions = []

    correct = 0

    total = len(dataset)

    start = (
        time.perf_counter()
    )

    for index, example in enumerate(
        dataset
    ):

        result = score_example(
            model,
            tokenizer,
            example,
        )

        if result["correct"]:

            correct += 1

        predictions.append({

            "index":
                index,

            "gold":
                result["gold"],

            "prediction":
                result["prediction"],

            "correct":
                result["correct"],
        })

        # ---------------------------------------------------------
        # Progress report every 100 examples.
        # ---------------------------------------------------------

        if (
            (index + 1) % 100 == 0
            or index + 1 == total
        ):

            elapsed = (
                time.perf_counter()
                - start
            )

            accuracy = (
                100.0
                * correct
                / (index + 1)
            )

            examples_per_second = (
                (index + 1)
                / elapsed
                if elapsed > 0
                else 0.0
            )

            remaining = (
                total
                - (index + 1)
            )

            eta_seconds = (
                remaining
                / examples_per_second
                if examples_per_second > 0
                else 0.0
            )

            print(
                f"[{config_name:<11}] "
                f"{index + 1:>5}/{total}  "
                f"correct={correct:>5}  "
                f"acc={accuracy:6.2f}%  "
                f"speed={examples_per_second:6.2f} ex/s  "
                f"elapsed={elapsed / 60:6.1f}m  "
                f"ETA={eta_seconds / 60:6.1f}m"
            )

    elapsed = (
        time.perf_counter()
        - start
    )

    accuracy = (
        100.0
        * correct
        / total
    )

    return {

        "correct":
            correct,

        "total":
            total,

        "accuracy":
            accuracy,

        "time_seconds":
            elapsed,

        "examples_per_second":
            (
                total / elapsed
                if elapsed > 0
                else 0.0
            ),

        "predictions":
            predictions,
    }


# =============================================================================
# PAIRED STATISTICS
# =============================================================================

def paired_statistics(
    fp16_result,
    ternary_result,
):

    fp = (
        fp16_result[
            "predictions"
        ]
    )

    tr = (
        ternary_result[
            "predictions"
        ]
    )

    if len(fp) != len(tr):

        raise RuntimeError(
            "Prediction counts differ."
        )

    both_correct = 0

    fp16_only = 0

    ternary_only = 0

    both_wrong = 0

    changed = 0

    for a, b in zip(
        fp,
        tr,
    ):

        if (
            a["index"]
            != b["index"]
            or
            a["gold"]
            != b["gold"]
        ):

            raise RuntimeError(
                "Prediction alignment "
                "mismatch."
            )

        if (
            a["prediction"]
            != b["prediction"]
        ):

            changed += 1

        if (
            a["correct"]
            and b["correct"]
        ):

            both_correct += 1

        elif (
            a["correct"]
            and not b["correct"]
        ):

            fp16_only += 1

        elif (
            not a["correct"]
            and b["correct"]
        ):

            ternary_only += 1

        else:

            both_wrong += 1

    discordant = (
        fp16_only
        + ternary_only
    )

    if discordant == 0:

        p_value = 1.0

    else:

        p_value = (
            binomtest(
                min(
                    fp16_only,
                    ternary_only,
                ),
                n=discordant,
                p=0.5,
                alternative="two-sided",
            )
            .pvalue
        )

    total = len(fp)

    agreement = (
        100.0
        * (total - changed)
        / total
    )

    return {

        "both_correct":
            both_correct,

        "fp16_only":
            fp16_only,

        "ternary_only":
            ternary_only,

        "both_wrong":
            both_wrong,

        "discordant_correctness_pairs":
            discordant,

        "changed_predictions":
            changed,

        "prediction_agreement_percent":
            agreement,

        "mcnemar_exact_p":
            p_value,
    }


# =============================================================================
# JSON HELPERS
# =============================================================================

def save_json(
    path,
    data,
):

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
        )


def load_checkpoint():

    if not CHECKPOINT_PATH.exists():

        return {}

    print(
        f"Loading checkpoint: "
        f"{CHECKPOINT_PATH}"
    )

    with open(
        CHECKPOINT_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


# =============================================================================
# SUMMARY
# =============================================================================

def build_summary(
    results,
):

    fp16 = (
        results[
            "FP16"
        ]
    )

    total_model_params = (
        results[
            "_metadata"
        ][
            "total_model_parameters"
        ]
    )

    lines = []

    lines.append(
        "=" * 150
    )

    lines.append(
        "EXPERIMENT 025 — "
        "EXTERNAL HELLASWAG VALIDATION"
    )

    lines.append(
        "=" * 150
    )

    lines.append("")

    lines.append(
        f"Model       : {MODEL_ID}"
    )

    lines.append(
        f"Dataset     : {DATASET_ID}"
    )

    lines.append(
        f"Split       : {SPLIT}"
    )

    lines.append(
        "Subset      : first 2,000 validation examples"
    )

    lines.append(
        f"Questions   : "
        f"{fp16['total']:,}"
    )

    lines.append("")

    lines.append(
        "Scoring     : mean conditional "
        "log probability per continuation token"
    )

    lines.append(
        "Execution   : four candidate endings "
        "scored sequentially"
    )

    lines.append("")

    lines.append(
        f"FP16 baseline: "
        f"{fp16['correct']}/"
        f"{fp16['total']} "
        f"({fp16['accuracy']:.2f}%)"
    )

    lines.append("")

    header = (

        f"{'Configuration':<18} "

        f"{'Matrices':>8} "

        f"{'Params':>12} "

        f"{'Cov%':>8} "

        f"{'Correct':>9} "

        f"{'Acc%':>8} "

        f"{'Delta':>9} "

        f"{'Agree%':>9} "

        f"{'Changed':>9} "

        f"{'FPonly':>8} "

        f"{'Tonly':>8} "

        f"{'McNemar p':>13}"
    )

    lines.append(
        header
    )

    lines.append(
        "-" * 150
    )

    for name in [

        "K_ONLY",
        "SAFE_CORE",
        "CORE_PLUS_Q",

    ]:

        row = (
            results[name]
        )

        params = (
            row[
                "ternary_parameters"
            ]
        )

        coverage = (
            100.0
            * params
            / total_model_params
        )

        delta = (
            row["accuracy"]
            - fp16["accuracy"]
        )

        lines.append(

            f"{name:<18} "

            f"{row['matrix_count']:>8} "

            f"{params:>12,} "

            f"{coverage:>8.4f} "

            f"{row['correct']:>9} "

            f"{row['accuracy']:>8.2f} "

            f"{delta:>+9.2f} "

            f"{row['prediction_agreement_percent']:>9.2f} "

            f"{row['changed_predictions']:>9} "

            f"{row['fp16_only']:>8} "

            f"{row['ternary_only']:>8} "

            f"{row['mcnemar_exact_p']:>13.6g}"
        )

    return "\n".join(
        lines
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print(
        "=" * 110
    )

    print(
        "EXPERIMENT 025 — "
        "EXTERNAL HELLASWAG VALIDATION"
    )

    print(
        "=" * 110
    )

    print()

    print(
        f"Model       : "
        f"{MODEL_ID}"
    )

    print(
        f"Dataset     : "
        f"{DATASET_ID}"
    )

    print(
        f"Split       : "
        f"{SPLIT}"
    )

    print(
        f"Subset size : "
        f"{SUBSET_SIZE:,}"
    )

    print(
        f"Device      : "
        f"{DEVICE}"
    )

    print(
        f"Model dtype : "
        f"{DTYPE}"
    )

    print()

    # -------------------------------------------------------------------------
    # TOKENIZER
    # -------------------------------------------------------------------------

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            MODEL_ID
        )
    )

    # -------------------------------------------------------------------------
    # FULL DATASET
    # -------------------------------------------------------------------------

    full_dataset = (
        load_dataset(
            DATASET_ID,
            split=SPLIT,
        )
    )

    print(
        f"Full validation examples : "
        f"{len(full_dataset):,}"
    )

    if (
        len(full_dataset)
        < SUBSET_SIZE
    ):

        raise RuntimeError(
            f"Dataset contains only "
            f"{len(full_dataset):,} examples; "
            f"{SUBSET_SIZE:,} are required."
        )

    # -------------------------------------------------------------------------
    # FROZEN DETERMINISTIC SUBSET
    # -------------------------------------------------------------------------

    dataset = (
        full_dataset.select(
            range(
                SUBSET_SIZE
            )
        )
    )

    print(
        f"Experiment subset        : "
        f"{len(dataset):,}"
    )

    print(
        "Subset selection         : "
        "first 2,000 validation examples"
    )

    # -------------------------------------------------------------------------
    # CHECKPOINT
    # -------------------------------------------------------------------------

    results = (
        load_checkpoint()
    )

    # -------------------------------------------------------------------------
    # METADATA
    # -------------------------------------------------------------------------

    if "_metadata" not in results:

        print()

        print(
            "Collecting model metadata..."
        )

        model = (
            load_model()
        )

        total_model_parameters = sum(
            parameter.numel()
            for parameter
            in model.parameters()
        )

        print(
            f"Total model parameters: "
            f"{total_model_parameters:,}"
        )

        results[
            "_metadata"
        ] = {

            "experiment":
                "025",

            "model":
                MODEL_ID,

            "dataset":
                DATASET_ID,

            "split":
                SPLIT,

            "full_validation_examples":
                len(full_dataset),

            "examples":
                len(dataset),

            "subset_size":
                SUBSET_SIZE,

            "subset_selection":
                "first_2000_validation_examples",

            "total_model_parameters":
                total_model_parameters,

            "scoring":
                (
                    "mean continuation "
                    "conditional log probability"
                ),

            "candidate_execution":
                "sequential",

            "quantizer_method":
                "optimized",

            "quantizer_group_size":
                128,

            "quantizer_threshold_multiplier":
                0.5,

            "boundary_check_pairs":
                40168,

            "boundary_check_mismatches":
                0,
        }

        del model

        clear_memory()

        save_json(
            CHECKPOINT_PATH,
            results,
        )

    # -------------------------------------------------------------------------
    # SAFETY CHECK AGAINST WRONG CHECKPOINT
    # -------------------------------------------------------------------------

    metadata = (
        results[
            "_metadata"
        ]
    )

    if (
        metadata.get(
            "subset_size"
        )
        != SUBSET_SIZE
    ):

        raise RuntimeError(
            "Checkpoint subset size does not "
            "match current Exp025 configuration."
        )

    if (
        metadata.get(
            "subset_selection"
        )
        !=
        "first_2000_validation_examples"
    ):

        raise RuntimeError(
            "Checkpoint subset selection does "
            "not match current Exp025 configuration."
        )

    # -------------------------------------------------------------------------
    # CONFIGURATION LOOP
    # -------------------------------------------------------------------------

    for (
        config_name,
        matrix_names,
    ) in CONFIGURATIONS.items():

        if config_name in results:

            print()

            print(
                f"Skipping {config_name}: "
                "already completed in checkpoint."
            )

            continue

        print()

        print(
            "=" * 110
        )

        print(
            f"CONFIGURATION: "
            f"{config_name}"
        )

        print(
            "=" * 110
        )

        # ---------------------------------------------------------
        # Fresh FP16 model for every configuration.
        # ---------------------------------------------------------

        model = (
            load_model()
        )

        quantizer_stats = []

        if matrix_names:

            quantizer_stats = (
                ternarize_configuration(
                    model,
                    matrix_names,
                )
            )

        # ---------------------------------------------------------
        # Coverage
        # ---------------------------------------------------------

        ternary_parameters = sum(

            get_module(
                model,
                name,
            ).weight.numel()

            for name
            in matrix_names
        )

        total_model_parameters = (
            results[
                "_metadata"
            ][
                "total_model_parameters"
            ]
        )

        coverage = (
            100.0
            * ternary_parameters
            / total_model_parameters
        )

        print()

        print(
            f"Matrices ternarized : "
            f"{len(matrix_names)}"
        )

        print(
            f"Ternary parameters  : "
            f"{ternary_parameters:,}"
        )

        print(
            f"Coverage            : "
            f"{coverage:.4f}%"
        )

        print()

        # ---------------------------------------------------------
        # Evaluation
        # ---------------------------------------------------------

        result = (
            evaluate_configuration(
                model,
                tokenizer,
                dataset,
                config_name,
            )
        )

        result[
            "matrix_names"
        ] = matrix_names

        result[
            "matrix_count"
        ] = len(
            matrix_names
        )

        result[
            "ternary_parameters"
        ] = (
            ternary_parameters
        )

        result[
            "coverage_percent"
        ] = coverage

        result[
            "quantizer_stats"
        ] = (
            quantizer_stats
        )

        results[
            config_name
        ] = result

        # ---------------------------------------------------------
        # Checkpoint only after a COMPLETE configuration.
        # ---------------------------------------------------------

        save_json(
            CHECKPOINT_PATH,
            results,
        )

        print()

        print(
            f"{config_name} COMPLETE"
        )

        print(
            f"Correct  : "
            f"{result['correct']}/"
            f"{result['total']}"
        )

        print(
            f"Accuracy : "
            f"{result['accuracy']:.2f}%"
        )

        print(
            f"Time     : "
            f"{result['time_seconds']:.2f} sec"
        )

        print(
            f"Speed    : "
            f"{result['examples_per_second']:.2f} "
            "examples/sec"
        )

        print(
            "Checkpoint saved."
        )

        del model

        clear_memory()

    # -------------------------------------------------------------------------
    # COMPLETION CHECK
    # -------------------------------------------------------------------------

    for required in (
        CONFIGURATIONS
    ):

        if required not in results:

            raise RuntimeError(
                "Missing completed "
                f"configuration: "
                f"{required}"
            )

    # -------------------------------------------------------------------------
    # PAIRED COMPARISONS
    # -------------------------------------------------------------------------

    fp16 = (
        results[
            "FP16"
        ]
    )

    for config_name in [

        "K_ONLY",
        "SAFE_CORE",
        "CORE_PLUS_Q",

    ]:

        paired = (
            paired_statistics(
                fp16,
                results[
                    config_name
                ],
            )
        )

        results[
            config_name
        ].update(
            paired
        )

        results[
            config_name
        ][
            "accuracy_difference"
        ] = (

            results[
                config_name
            ][
                "accuracy"
            ]

            - fp16[
                "accuracy"
            ]
        )

    # -------------------------------------------------------------------------
    # FINAL JSON
    # -------------------------------------------------------------------------

    save_json(
        FINAL_JSON_PATH,
        results,
    )

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------

    summary = (
        build_summary(
            results
        )
    )

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            summary
        )

        f.write(
            "\n"
        )

    print()

    print(
        summary
    )

    print()

    print(
        f"Detailed results : "
        f"{FINAL_JSON_PATH}"
    )

    print(
        f"Summary          : "
        f"{SUMMARY_PATH}"
    )

    print(
        f"Checkpoint       : "
        f"{CHECKPOINT_PATH}"
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
