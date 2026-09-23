import gc
import re
import time
from collections import defaultdict

import torch
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

from ternary.quantizer import ternarize_modules


# =============================================================================
# EXPERIMENT 017
# Formal validation of MIXED_SMALL (~3.83%)
# =============================================================================

MODEL_ID = "Qwen/Qwen3.5-0.8B"

DEVICE = "cuda"
DTYPE = torch.float16

MIN_WEIGHT_PARAMS = 100_000

EXACT_MAX_NEW_TOKENS = 32
OPEN_MAX_NEW_TOKENS = 80

TARGET_BLOCKS = {
    "mlp": [6],
    "linear_attn": [4],
    "self_attn": [3],
}


# =============================================================================
# EVALUATION SET
# Same 20 prompts used in Experiment 016
# =============================================================================

EVALUATION_SET = [

    # Arithmetic
    {
        "id": "A1",
        "category": "arithmetic",
        "prompt": (
            "Answer with only the final number. "
            "What is 24 multiplied by 17?"
        ),
        "expected": "408",
    },
    {
        "id": "A2",
        "category": "arithmetic",
        "prompt": (
            "Answer with only the final number. "
            "What is 432 divided by 18?"
        ),
        "expected": "24",
    },
    {
        "id": "A3",
        "category": "arithmetic",
        "prompt": (
            "Answer with only the final number. "
            "What is 37 plus 58?"
        ),
        "expected": "95",
    },
    {
        "id": "A4",
        "category": "arithmetic",
        "prompt": (
            "Answer with only the final number. "
            "What is 91 minus 46?"
        ),
        "expected": "45",
    },

    # Word problems
    {
        "id": "W1",
        "category": "word_problem",
        "prompt": (
            "Answer with only the final number. "
            "A bus travels 55 kilometers per hour for 4 hours. "
            "How many kilometers does it travel?"
        ),
        "expected": "220",
    },
    {
        "id": "W2",
        "category": "word_problem",
        "prompt": (
            "Answer with only the final number. "
            "A runner covers 8 kilometers per hour for 3 hours. "
            "How many kilometers does the runner cover?"
        ),
        "expected": "24",
    },
    {
        "id": "W3",
        "category": "word_problem",
        "prompt": (
            "Answer with only the final number. "
            "A box contains 12 rows of 9 apples. "
            "How many apples are there?"
        ),
        "expected": "108",
    },

    # Factual
    {
        "id": "F1",
        "category": "factual",
        "prompt": (
            "Answer with only one word. "
            "What planet is known as the Red Planet?"
        ),
        "expected": "mars",
    },
    {
        "id": "F2",
        "category": "factual",
        "prompt": (
            "Answer with only one word. "
            "What gas do humans need to breathe to survive?"
        ),
        "expected": "oxygen",
    },
    {
        "id": "F3",
        "category": "factual",
        "prompt": (
            "Answer with only one word. "
            "What is the process by which plants make food using sunlight?"
        ),
        "expected": "photosynthesis",
    },

    # Science
    {
        "id": "S1",
        "category": "science",
        "prompt":
            "Explain in simple terms why objects fall toward the ground.",
        "expected": None,
    },
    {
        "id": "S2",
        "category": "science",
        "prompt":
            "Explain in simple terms why ice floats on liquid water.",
        "expected": None,
    },
    {
        "id": "S3",
        "category": "science",
        "prompt":
            "Explain in simple terms how rain forms.",
        "expected": None,
    },
    {
        "id": "S4",
        "category": "science",
        "prompt": (
            "Explain in simple terms why the Moon appears to change shape "
            "during the month."
        ),
        "expected": None,
    },

    # General
    {
        "id": "G1",
        "category": "general",
        "prompt": (
            "Explain in simple terms why regular sleep is important "
            "for the human body."
        ),
        "expected": None,
    },
    {
        "id": "G2",
        "category": "general",
        "prompt": (
            "Explain in simple terms why washing your hands helps "
            "prevent illness."
        ),
        "expected": None,
    },
    {
        "id": "G3",
        "category": "general",
        "prompt":
            "Explain in simple terms why saving money can be useful.",
        "expected": None,
    },

    # Instruction following
    {
        "id": "I1",
        "category": "instruction",
        "prompt": (
            "Write exactly three short benefits of reading books. "
            "Use a numbered list."
        ),
        "expected": None,
    },
    {
        "id": "I2",
        "category": "instruction",
        "prompt": (
            "Give exactly three examples of renewable energy sources. "
            "Use a numbered list."
        ),
        "expected": None,
    },
    {
        "id": "I3",
        "category": "instruction",
        "prompt":
            "In two short sentences, explain why exercise is useful.",
        "expected": None,
    },
]


# =============================================================================
# ARCHITECTURE HELPERS
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
# EXACT-ANSWER SCORING
# =============================================================================

def normalize_word(text):
    return re.sub(
        r"[^a-z]",
        "",
        text.lower(),
    )


def score_numeric(text, expected):

    stripped = text.strip()

    # Strict requested format:
    # only the final number.
    format_compliant = bool(
        re.fullmatch(
            r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)",
            stripped,
        )
    )

    expected_value = float(expected)

    candidates = []

    # Direct numeric response.
    if format_compliant:
        candidates.append(
            float(stripped)
        )

    # Completed arithmetic equations.
    equation_matches = re.findall(
        r"(?:=|equals)\s*"
        r"([+-]?(?:\d+(?:\.\d+)?|\.\d+))",
        stripped,
        flags=re.IGNORECASE,
    )

    for value in equation_matches:
        candidates.append(
            float(value)
        )

    # Explicit final-answer wording.
    answer_matches = re.findall(
        r"(?:answer|result|total|distance|there are|covers?)"
        r"\s*(?:is|=|:)?\s*"
        r"([+-]?(?:\d+(?:\.\d+)?|\.\d+))",
        stripped,
        flags=re.IGNORECASE,
    )

    for value in answer_matches:
        candidates.append(
            float(value)
        )

    if not candidates:
        return (
            "NO_ANSWER",
            None,
            format_compliant,
        )

    answer = candidates[-1]

    if abs(
        answer - expected_value
    ) < 1e-9:
        status = "CORRECT"
    else:
        status = "WRONG"

    return (
        status,
        answer,
        format_compliant,
    )


def score_word(text, expected):

    stripped = text.strip()

    format_compliant = bool(
        re.fullmatch(
            r"[A-Za-z]+",
            stripped,
        )
    )

    expected_norm = normalize_word(
        expected
    )

    direct = normalize_word(
        stripped
    )

    if direct == expected_norm:
        return (
            "CORRECT",
            expected_norm,
            format_compliant,
        )

    words = re.findall(
        r"[A-Za-z]+",
        stripped.lower(),
    )

    matches = [
        word
        for word in words
        if normalize_word(word)
        == expected_norm
    ]

    if matches:
        return (
            "CORRECT",
            expected_norm,
            format_compliant,
        )

    if not words:
        return (
            "NO_ANSWER",
            None,
            format_compliant,
        )

    # A direct single-word response that is not the expected answer
    # is an explicit wrong answer.
    if len(words) == 1:
        return (
            "WRONG",
            normalize_word(words[0]),
            format_compliant,
        )

    # For longer text, avoid inventing an answer from arbitrary prose.
    return (
        "NO_ANSWER",
        None,
        format_compliant,
    )


def score_exact(text, item):

    if item["category"] in {
        "arithmetic",
        "word_problem",
    }:
        return score_numeric(
            text,
            item["expected"],
        )

    return score_word(
        text,
        item["expected"],
    )


# =============================================================================
# SIMPLE REPETITION MEASURE FOR OPEN-ENDED OUTPUTS
# =============================================================================

def repetition_score(text):

    words = re.findall(
        r"\b\w+\b",
        text.lower(),
    )

    if len(words) < 2:
        return 0.0

    bigrams = list(
        zip(
            words,
            words[1:],
        )
    )

    if not bigrams:
        return 0.0

    unique = len(
        set(bigrams)
    )

    return (
        1.0
        - unique / len(bigrams)
    )


# =============================================================================
# GENERATION
# =============================================================================

@torch.no_grad()
def generate(
    model,
    tokenizer,
    item,
):

    exact = (
        item["expected"]
        is not None
    )

    max_new_tokens = (
        EXACT_MAX_NEW_TOKENS
        if exact
        else OPEN_MAX_NEW_TOKENS
    )

    messages = [
        {
            "role": "user",
            "content": item["prompt"],
        }
    ]

    prompt_text = (
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )

    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value
        in inputs.items()
    }

    torch.cuda.synchronize()

    start = time.perf_counter()

    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    torch.cuda.synchronize()

    elapsed = (
        time.perf_counter()
        - start
    )

    input_tokens = (
        inputs["input_ids"].shape[1]
    )

    generated = output[
        0,
        input_tokens:,
    ]

    generated_tokens = (
        generated.shape[0]
    )

    text = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    ).strip()

    speed = (
        generated_tokens / elapsed
        if elapsed > 0
        else 0.0
    )

    del output
    del generated

    for value in inputs.values():
        del value

    return {
        "text": text,
        "tokens": generated_tokens,
        "time": elapsed,
        "tok_per_sec": speed,
    }


# =============================================================================
# RUN CONFIGURATION
# =============================================================================

def run_configuration(
    config_name,
    use_ternary,
    tokenizer,
):

    print(
        "\n"
        "============================================================"
    )
    print(
        f"CONFIGURATION: {config_name}"
    )
    print(
        "============================================================"
    )

    model = load_model()

    quant_stats = None

    if use_ternary:

        modules = discover_modules(
            model
        )

        print(
            f"\nTarget matrices: "
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
            f"Ternary parameters : "
            f"{quant_stats['parameters']:,}"
        )

        print(
            f"Coverage           : "
            f"{coverage:.4f}%"
        )

        print(
            f"Mean error         : "
            f"{quant_stats['mean_error']:.9f}"
        )

        print(
            f"Max error          : "
            f"{quant_stats['max_error']:.9f}"
        )

        print(
            f"Mean scale ratio   : "
            f"{quant_stats['mean_scale_ratio']:.9f}"
        )

    results = []

    for item in EVALUATION_SET:

        generated = generate(
            model,
            tokenizer,
            item,
        )

        record = {
            "id": item["id"],
            "category": item["category"],
            "prompt": item["prompt"],
            "expected": item["expected"],
            "output": generated["text"],
            "tokens": generated["tokens"],
            "time": generated["time"],
            "tok_per_sec":
                generated["tok_per_sec"],
        }

        if item["expected"] is not None:

            (
                status,
                extracted,
                format_compliant,
            ) = score_exact(
                generated["text"],
                item,
            )

            record[
                "correctness"
            ] = status

            record[
                "extracted_answer"
            ] = extracted

            record[
                "format_compliant"
            ] = format_compliant

        else:

            record[
                "repetition_score"
            ] = repetition_score(
                generated["text"]
            )

        results.append(record)

        print(
            f"\n[{item['id']}] "
            f"{item['category']}"
        )

        print(
            f"PROMPT: {item['prompt']}"
        )

        print(
            f"OUTPUT: "
            f"{generated['text']!r}"
        )

        if item["expected"] is not None:

            print(
                f"EXPECTED: "
                f"{item['expected']}"
            )

            print(
                f"CORRECTNESS: "
                f"{record['correctness']}"
            )

            print(
                f"EXTRACTED: "
                f"{record['extracted_answer']}"
            )

            print(
                f"FORMAT COMPLIANT: "
                f"{record['format_compliant']}"
            )

        else:

            print(
                f"REPETITION SCORE: "
                f"{record['repetition_score']:.4f}"
            )

        print(
            f"TOKENS: "
            f"{generated['tokens']}"
        )

        print(
            f"TOK/S: "
            f"{generated['tok_per_sec']:.2f}"
        )

    release_model(
        model
    )

    return {
        "name": config_name,
        "quant_stats": quant_stats,
        "results": results,
    }


# =============================================================================
# SUMMARY
# =============================================================================

def summarize(configuration):

    exact = [
        item
        for item
        in configuration["results"]
        if item["expected"] is not None
    ]

    open_ended = [
        item
        for item
        in configuration["results"]
        if item["expected"] is None
    ]

    correct = sum(
        item["correctness"] == "CORRECT"
        for item in exact
    )

    wrong = sum(
        item["correctness"] == "WRONG"
        for item in exact
    )

    no_answer = sum(
        item["correctness"] == "NO_ANSWER"
        for item in exact
    )

    compliant = sum(
        item["format_compliant"]
        for item in exact
    )

    category_stats = defaultdict(
        lambda: {
            "correct": 0,
            "wrong": 0,
            "no_answer": 0,
            "total": 0,
        }
    )

    for item in exact:

        stats = category_stats[
            item["category"]
        ]

        stats["total"] += 1

        if (
            item["correctness"]
            == "CORRECT"
        ):
            stats["correct"] += 1

        elif (
            item["correctness"]
            == "WRONG"
        ):
            stats["wrong"] += 1

        else:
            stats["no_answer"] += 1

    mean_repetition = (
        sum(
            item["repetition_score"]
            for item in open_ended
        )
        / len(open_ended)
    )

    total_tokens = sum(
        item["tokens"]
        for item
        in configuration["results"]
    )

    total_time = sum(
        item["time"]
        for item
        in configuration["results"]
    )

    aggregate_speed = (
        total_tokens / total_time
        if total_time > 0
        else 0.0
    )

    return {
        "correct": correct,
        "wrong": wrong,
        "no_answer": no_answer,
        "total_exact": len(exact),
        "answer_accuracy":
            correct / len(exact),
        "format_compliance":
            compliant / len(exact),
        "mean_repetition":
            mean_repetition,
        "aggregate_speed":
            aggregate_speed,
        "category_stats":
            dict(category_stats),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            MODEL_ID
        )
    )

    configurations = []

    configurations.append(
        run_configuration(
            "FP16",
            False,
            tokenizer,
        )
    )

    configurations.append(
        run_configuration(
            "MIXED_SMALL_3_83",
            True,
            tokenizer,
        )
    )

    print(
        "\n\n"
        "============================================================"
    )
    print(
        "EXPERIMENT 017 — FINAL SUMMARY"
    )
    print(
        "============================================================"
    )

    for configuration in configurations:

        summary = summarize(
            configuration
        )

        print(
            f"\n{configuration['name']}"
        )

        print(
            f"Correct            : "
            f"{summary['correct']}/"
            f"{summary['total_exact']}"
        )

        print(
            f"Wrong              : "
            f"{summary['wrong']}"
        )

        print(
            f"No answer          : "
            f"{summary['no_answer']}"
        )

        print(
            f"Answer accuracy    : "
            f"{100 * summary['answer_accuracy']:.2f}%"
        )

        print(
            f"Format compliance : "
            f"{100 * summary['format_compliance']:.2f}%"
        )

        print(
            f"Open mean repetition: "
            f"{summary['mean_repetition']:.4f}"
        )

        print(
            f"Aggregate tok/s    : "
            f"{summary['aggregate_speed']:.2f}"
        )

        print(
            "Exact categories:"
        )

        for (
            category,
            stats,
        ) in summary[
            "category_stats"
        ].items():

            print(
                f"  {category:<14} "
                f"C={stats['correct']} "
                f"W={stats['wrong']} "
                f"N={stats['no_answer']} "
                f"T={stats['total']}"
            )


if __name__ == "__main__":
    main()
