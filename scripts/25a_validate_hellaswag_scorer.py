import gc
import math
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen3.5-0.8B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_INSPECT = 10


def load_model():

    print(f"Loading model: {MODEL_ID}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
    )

    model.to(DEVICE)
    model.eval()

    return model


def build_context(example):
    """
    HellaSwag contains:
        ctx
        endings[4]
        label

    We score each ending conditioned on ctx.
    """

    context = example["ctx"].strip()

    if not context:
        raise RuntimeError("Empty HellaSwag context.")

    return context


@torch.inference_mode()
def score_continuation(
    model,
    tokenizer,
    context,
    continuation,
):
    """
    Score continuation tokens conditioned on context.

    Returns:
        total_logprob
        mean_logprob
        continuation_token_count

    mean_logprob is used for prediction so answer length does not
    automatically dominate the comparison.
    """

    # Important:
    # tokenize context and context+continuation together so that token
    # boundaries correspond to the actual concatenated sequence.

    context_text = context

    continuation_text = continuation

    # HellaSwag endings normally continue the context naturally.
    # Insert one separating space when necessary.
    if (
        context_text
        and continuation_text
        and not context_text[-1].isspace()
        and not continuation_text[0].isspace()
    ):
        continuation_text = " " + continuation_text

    full_text = (
        context_text
        + continuation_text
    )

    context_ids = tokenizer(
        context_text,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"]

    full_ids = tokenizer(
        full_text,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"]

    context_length = context_ids.shape[1]
    full_length = full_ids.shape[1]

    if full_length <= context_length:
        raise RuntimeError(
            "Continuation produced no additional tokens."
        )

    full_ids = full_ids.to(DEVICE)

    outputs = model(
        input_ids=full_ids,
        use_cache=False,
    )

    logits = outputs.logits

    # logits[:, t-1] predicts token t.
    #
    # If context has N tokens, the first continuation token has
    # token index N and is predicted by logits index N-1.

    continuation_ids = full_ids[
        0,
        context_length:
    ]

    prediction_logits = logits[
        0,
        context_length - 1:
        full_length - 1,
        :
    ]

    if (
        prediction_logits.shape[0]
        != continuation_ids.shape[0]
    ):
        raise RuntimeError(
            "Continuation/logit alignment mismatch: "
            f"logits={prediction_logits.shape[0]}, "
            f"tokens={continuation_ids.shape[0]}"
        )

    log_probs = torch.log_softmax(
        prediction_logits.float(),
        dim=-1,
    )

    token_log_probs = log_probs.gather(
        dim=-1,
        index=continuation_ids.unsqueeze(-1),
    ).squeeze(-1)

    total_logprob = (
        token_log_probs.sum().item()
    )

    token_count = (
        continuation_ids.numel()
    )

    mean_logprob = (
        total_logprob / token_count
    )

    return {
        "total_logprob": total_logprob,
        "mean_logprob": mean_logprob,
        "token_count": token_count,
    }


def evaluate_example(
    model,
    tokenizer,
    example,
):

    context = build_context(example)

    endings = example["endings"]

    if len(endings) != 4:
        raise RuntimeError(
            f"Expected 4 endings, found {len(endings)}."
        )

    scores = []

    for index, ending in enumerate(endings):

        result = score_continuation(
            model,
            tokenizer,
            context,
            ending,
        )

        scores.append({
            "index": index,
            "ending": ending,
            **result,
        })

    predicted = max(
        scores,
        key=lambda row: row["mean_logprob"],
    )["index"]

    gold = int(example["label"])

    return {
        "context": context,
        "scores": scores,
        "prediction": predicted,
        "gold": gold,
        "correct": predicted == gold,
    }


def main():

    print("=" * 100)
    print("EXP025A — HELLASWAG FP16 SCORER VALIDATION")
    print("=" * 100)

    dataset = load_dataset(
        "Rowan/hellaswag",
        split="validation",
    )

    print()
    print(f"Validation examples : {len(dataset):,}")

    if len(dataset) == 0:
        raise RuntimeError(
            "HellaSwag validation split is empty."
        )

    first = dataset[0]

    required_fields = {
        "ctx",
        "endings",
        "label",
    }

    missing = (
        required_fields
        - set(first.keys())
    )

    if missing:
        raise RuntimeError(
            f"Missing required fields: {missing}"
        )

    print(
        f"Fields              : {list(first.keys())}"
    )

    print(
        f"Inspection examples : {NUM_INSPECT}"
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_ID
        )
    )

    model = load_model()

    correct = 0

    start = time.perf_counter()

    for index in range(NUM_INSPECT):

        example = dataset[index]

        result = evaluate_example(
            model,
            tokenizer,
            example,
        )

        if result["correct"]:
            correct += 1

        print()
        print("=" * 100)

        print(
            f"EXAMPLE {index + 1}/{NUM_INSPECT}"
        )

        print("=" * 100)

        print()
        print("CONTEXT:")
        print(result["context"])

        print()
        print(
            f"GOLD       : {result['gold']}"
        )

        print(
            f"PREDICTION : {result['prediction']}"
        )

        print(
            f"CORRECT    : {result['correct']}"
        )

        print()
        print("CANDIDATES")
        print("-" * 100)

        for row in result["scores"]:

            marker = ""

            if row["index"] == result["gold"]:
                marker += " GOLD"

            if row["index"] == result["prediction"]:
                marker += " PRED"

            print(
                f"[{row['index']}] "
                f"tokens={row['token_count']:>3}  "
                f"total={row['total_logprob']:>10.4f}  "
                f"mean={row['mean_logprob']:>9.4f}"
                f"{marker}"
            )

            print(
                f"    {row['ending']}"
            )

    elapsed = (
        time.perf_counter()
        - start
    )

    print()
    print("=" * 100)
    print("SCORER SANITY SUMMARY")
    print("=" * 100)

    print(
        f"Examples inspected : {NUM_INSPECT}"
    )

    print(
        f"Correct            : "
        f"{correct}/{NUM_INSPECT}"
    )

    print(
        f"Elapsed seconds    : {elapsed:.2f}"
    )

    print()
    print(
        "NOTE: This 10-example accuracy is NOT an experimental result."
    )

    print(
        "Its purpose is only to validate dataset parsing, "
        "token alignment, continuation scoring, and prediction logic."
    )

    del model

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
