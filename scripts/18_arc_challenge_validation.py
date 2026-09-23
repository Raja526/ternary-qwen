import gc
import json
import time
from collections import Counter

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

from ternary.quantizer import ternarize_modules


# =============================================================================
# EXPERIMENT 018
# ARC-Challenge validation
#
# FP16 vs MIXED_SMALL_3_83
#
# Multiple-choice evaluation:
#   - Present question + all choices
#   - Use Qwen chat template
#   - Disable thinking
#   - Score answer labels directly
#
# Quantizer and ternary target configuration are frozen.
# =============================================================================

MODEL_ID = "Qwen/Qwen3.5-0.8B"

DEVICE = "cuda"
DTYPE = torch.float16

MIN_WEIGHT_PARAMS = 100_000

DATASET_ID = "allenai/ai2_arc"
DATASET_CONFIG = "ARC-Challenge"
DATASET_SPLIT = "validation"

RESULT_JSON = "results/018_arc_challenge_predictions.json"


# =============================================================================
# FROZEN MIXED_SMALL CONFIGURATION
# =============================================================================

TARGET_BLOCKS = {
    "mlp": [6],
    "linear_attn": [4],
    "self_attn": [3],
}


# =============================================================================
# TARGET DISCOVERY
# =============================================================================

def get_block_index(name):

    parts = name.split(".")

    try:
        position = parts.index("layers")
        return int(parts[position + 1])

    except (ValueError, IndexError):
        return None


def get_component(name, module):

    if not isinstance(module, torch.nn.Linear):
        return None

    if module.weight.numel() < MIN_WEIGHT_PARAMS:
        return None

    lower_name = name.lower()

    if ".mlp." in lower_name:
        return "mlp"

    if ".linear_attn." in lower_name:
        return "linear_attn"

    if ".self_attn." in lower_name:
        return "self_attn"

    return None


def is_target_matrix(name, module):

    component = get_component(
        name,
        module,
    )

    if component is None:
        return False

    block = get_block_index(name)

    if block is None:
        return False

    return block in TARGET_BLOCKS.get(
        component,
        [],
    )


def discover_modules(model):

    modules = []

    for name, module in model.named_modules():

        if not is_target_matrix(
            name,
            module,
        ):
            continue

        modules.append(
            (
                name,
                module,
                get_component(name, module),
                get_block_index(name),
            )
        )

    return modules


# =============================================================================
# MODEL
# =============================================================================

def load_model():

    gc.collect()
    torch.cuda.empty_cache()

    model = (
        Qwen3_5ForCausalLM
        .from_pretrained(
            MODEL_ID,
            torch_dtype=DTYPE,
            device_map=DEVICE,
        )
    )

    model.eval()

    return model


def release_model(model):

    model.to("cpu")
    del model

    gc.collect()
    gc.collect()

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


# =============================================================================
# MULTIPLE-CHOICE PROMPT
# =============================================================================

def build_user_prompt(example):

    question = example["question"]

    labels = example[
        "choices"
    ]["label"]

    texts = example[
        "choices"
    ]["text"]

    lines = [
        "Choose the best answer to the following "
        "multiple-choice question.",
        "",
        f"Question: {question}",
        "",
    ]

    for label, text in zip(
        labels,
        texts,
    ):
        lines.append(
            f"{label}. {text}"
        )

    lines.extend(
        [
            "",
            "Respond with only the answer label.",
        ]
    )

    return "\n".join(lines)


def build_prompt(
    tokenizer,
    example,
):

    user_prompt = build_user_prompt(
        example
    )

    messages = [
        {
            "role": "user",
            "content": user_prompt,
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    return prompt


# =============================================================================
# LABEL SCORING
# =============================================================================

@torch.no_grad()
def get_next_token_logits(
    model,
    tokenizer,
    prompt,
):

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )

    input_ids = (
        inputs["input_ids"]
        .to(DEVICE)
    )

    attention_mask = (
        inputs["attention_mask"]
        .to(DEVICE)
    )

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )

    logits = outputs.logits[
        0,
        -1,
        :
    ].float()

    return logits


def get_label_token_id(
    tokenizer,
    label,
):

    # At generation time the assistant begins with the label.
    #
    # We explicitly require each ARC label to map to exactly
    # one token. If this assumption fails, the experiment stops
    # rather than silently using an invalid score.

    token_ids = tokenizer.encode(
        label,
        add_special_tokens=False,
    )

    if len(token_ids) != 1:

        raise RuntimeError(
            f"Label {label!r} does not map to "
            f"exactly one token: {token_ids}"
        )

    return token_ids[0]


@torch.no_grad()
def evaluate_question(
    model,
    tokenizer,
    example,
):

    prompt = build_prompt(
        tokenizer,
        example,
    )

    logits = get_next_token_logits(
        model,
        tokenizer,
        prompt,
    )

    log_probs = F.log_softmax(
        logits,
        dim=-1,
    )

    labels = [
        str(label)
        for label
        in example["choices"]["label"]
    ]

    label_scores = []

    for label in labels:

        token_id = get_label_token_id(
            tokenizer,
            label,
        )

        logprob = (
            log_probs[token_id]
            .item()
        )

        label_scores.append(
            {
                "label": label,
                "token_id": token_id,
                "logprob": logprob,
            }
        )

    predicted = max(
        label_scores,
        key=lambda item:
            item["logprob"],
    )["label"]

    answer_key = str(
        example["answerKey"]
    )

    return {
        "id": example["id"],
        "question": example["question"],
        "choices": example["choices"],
        "answer_key": answer_key,
        "prediction": predicted,
        "correct":
            predicted == answer_key,
        "label_scores":
            label_scores,
    }


# =============================================================================
# CONFIGURATION EVALUATION
# =============================================================================

def evaluate_configuration(
    name,
    model,
    tokenizer,
    dataset,
):

    print()
    print(
        "=" * 70
    )

    print(
        f"CONFIGURATION: {name}"
    )

    print(
        "=" * 70
    )

    results = []

    correct = 0

    start = time.perf_counter()

    for index, example in enumerate(
        dataset,
        start=1,
    ):

        result = evaluate_question(
            model,
            tokenizer,
            example,
        )

        results.append(
            result
        )

        if result["correct"]:
            correct += 1

        status = (
            "CORRECT"
            if result["correct"]
            else "WRONG"
        )

        print(
            f"[{index:03d}/{len(dataset)}] "
            f"{example['id']} "
            f"gold={result['answer_key']} "
            f"pred={result['prediction']} "
            f"{status}"
        )

    elapsed = (
        time.perf_counter()
        - start
    )

    accuracy = (
        correct
        / len(dataset)
    )

    print()

    print(
        f"Correct  : "
        f"{correct}/{len(dataset)}"
    )

    print(
        f"Accuracy : "
        f"{100 * accuracy:.2f}%"
    )

    print(
        f"Time     : "
        f"{elapsed:.2f} sec"
    )

    return {
        "name": name,
        "correct": correct,
        "total": len(dataset),
        "accuracy": accuracy,
        "time_seconds": elapsed,
        "predictions": results,
    }


# =============================================================================
# PAIRED ANALYSIS
# =============================================================================

def paired_analysis(
    fp16_result,
    ternary_result,
):

    fp16_by_id = {
        item["id"]: item
        for item
        in fp16_result["predictions"]
    }

    ternary_by_id = {
        item["id"]: item
        for item
        in ternary_result["predictions"]
    }

    counts = Counter()

    changed_predictions = 0

    rows = []

    for question_id in fp16_by_id:

        fp = fp16_by_id[
            question_id
        ]

        tq = ternary_by_id[
            question_id
        ]

        if (
            fp["prediction"]
            != tq["prediction"]
        ):
            changed_predictions += 1

        if (
            fp["correct"]
            and tq["correct"]
        ):
            category = "both_correct"

        elif (
            fp["correct"]
            and not tq["correct"]
        ):
            category = "fp16_only"

        elif (
            not fp["correct"]
            and tq["correct"]
        ):
            category = "ternary_only"

        else:
            category = "both_wrong"

        counts[
            category
        ] += 1

        rows.append(
            {
                "id": question_id,
                "answer_key":
                    fp["answer_key"],
                "fp16_prediction":
                    fp["prediction"],
                "ternary_prediction":
                    tq["prediction"],
                "pair_category":
                    category,
            }
        )

    total = len(rows)

    prediction_agreement = (
        1.0
        - changed_predictions / total
    )

    accuracy_difference = (
        ternary_result["accuracy"]
        - fp16_result["accuracy"]
    )

    return {
        "both_correct":
            counts["both_correct"],
        "fp16_only":
            counts["fp16_only"],
        "ternary_only":
            counts["ternary_only"],
        "both_wrong":
            counts["both_wrong"],
        "changed_predictions":
            changed_predictions,
        "prediction_agreement":
            prediction_agreement,
        "accuracy_difference":
            accuracy_difference,
        "rows":
            rows,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():

    print(
        "Experiment 018"
    )

    print(
        "ARC-Challenge label-scoring validation"
    )

    print(
        f"Model: {MODEL_ID}"
    )

    dataset = load_dataset(
        DATASET_ID,
        DATASET_CONFIG,
        split=DATASET_SPLIT,
    )

    print(
        f"Questions: {len(dataset)}"
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            MODEL_ID
        )
    )

    # Verify label-token assumption before model evaluation.

    all_labels = sorted(
        {
            str(label)
            for example in dataset
            for label
            in example["choices"]["label"]
        }
    )

    print(
        f"ARC labels found: "
        f"{all_labels}"
    )

    for label in all_labels:

        token_id = get_label_token_id(
            tokenizer,
            label,
        )

        print(
            f"Label {label!r} -> "
            f"token_id {token_id}"
        )

    # -------------------------------------------------------------------------
    # FP16
    # -------------------------------------------------------------------------

    model = load_model()

    fp16_result = (
        evaluate_configuration(
            "FP16",
            model,
            tokenizer,
            dataset,
        )
    )

    release_model(
        model
    )

    # -------------------------------------------------------------------------
    # MIXED_SMALL
    # -------------------------------------------------------------------------

    model = load_model()

    modules = discover_modules(
        model
    )

    print()
    print(
        "TERNARY CONFIGURATION"
    )

    print(
        f"Target matrices     : "
        f"{len(modules)}"
    )

    quant_stats = ternarize_modules(
        modules,
        method="optimized",
    )

    total_model_parameters = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    coverage = (
        100.0
        * quant_stats["parameters"]
        / total_model_parameters
    )

    print(
        f"Ternary parameters  : "
        f"{quant_stats['parameters']:,}"
    )

    print(
        f"Coverage            : "
        f"{coverage:.4f}%"
    )

    print(
        f"Mean error          : "
        f"{quant_stats['mean_error']:.9f}"
    )

    print(
        f"Max error           : "
        f"{quant_stats['max_error']:.9f}"
    )

    print(
        f"Mean scale ratio    : "
        f"{quant_stats['mean_scale_ratio']:.9f}"
    )

    ternary_result = (
        evaluate_configuration(
            "MIXED_SMALL_3_83",
            model,
            tokenizer,
            dataset,
        )
    )

    release_model(
        model
    )

    # -------------------------------------------------------------------------
    # Paired comparison
    # -------------------------------------------------------------------------

    paired = paired_analysis(
        fp16_result,
        ternary_result,
    )

    print()
    print(
        "=" * 70
    )

    print(
        "EXPERIMENT 018 — FINAL SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        f"\nQuestions             : "
        f"{len(dataset)}"
    )

    print(
        f"\nFP16 correct          : "
        f"{fp16_result['correct']}/"
        f"{fp16_result['total']}"
    )

    print(
        f"FP16 accuracy         : "
        f"{100 * fp16_result['accuracy']:.2f}%"
    )

    print(
        f"\nTernary correct       : "
        f"{ternary_result['correct']}/"
        f"{ternary_result['total']}"
    )

    print(
        f"Ternary accuracy      : "
        f"{100 * ternary_result['accuracy']:.2f}%"
    )

    print(
        f"\nAccuracy difference   : "
        f"{100 * paired['accuracy_difference']:+.2f} pp"
    )

    print(
        "\nPAIRED OUTCOMES"
    )

    print(
        f"Both correct          : "
        f"{paired['both_correct']}"
    )

    print(
        f"FP16 only correct     : "
        f"{paired['fp16_only']}"
    )

    print(
        f"Ternary only correct  : "
        f"{paired['ternary_only']}"
    )

    print(
        f"Both wrong            : "
        f"{paired['both_wrong']}"
    )

    print(
        f"\nChanged predictions   : "
        f"{paired['changed_predictions']}"
    )

    print(
        f"Prediction agreement  : "
        f"{100 * paired['prediction_agreement']:.2f}%"
    )

    # -------------------------------------------------------------------------
    # Machine-readable result
    # -------------------------------------------------------------------------

    output = {
        "experiment": 18,

        "model": MODEL_ID,

        "evaluation_method":
            "chat_template_next_token_label_logprob",

        "dataset": {
            "id": DATASET_ID,
            "config": DATASET_CONFIG,
            "split": DATASET_SPLIT,
            "questions": len(dataset),
        },

        "ternary": {
            "target_blocks":
                TARGET_BLOCKS,

            "target_matrices":
                len(modules),

            "parameters":
                quant_stats["parameters"],

            "coverage_percent":
                coverage,

            "mean_error":
                quant_stats["mean_error"],

            "max_error":
                quant_stats["max_error"],

            "mean_scale_ratio":
                quant_stats[
                    "mean_scale_ratio"
                ],
        },

        "fp16":
            fp16_result,

        "mixed_small":
            ternary_result,

        "paired":
            paired,
    }

    with open(
        RESULT_JSON,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"\nDetailed results saved to: "
        f"{RESULT_JSON}"
    )


if __name__ == "__main__":
    main()
