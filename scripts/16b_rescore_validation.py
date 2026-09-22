import re
from pathlib import Path
from collections import defaultdict


# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_FILE = Path(
    "results/016_formal_validation.txt"
)

OUTPUT_FILE = Path(
    "results/016b_rescored_validation.txt"
)


# =============================================================================
# EXPECTED EXACT ANSWERS
# =============================================================================

EXPECTED = {
    "A1": {
        "category": "arithmetic",
        "answer": "408",
        "type": "numeric",
    },
    "A2": {
        "category": "arithmetic",
        "answer": "24",
        "type": "numeric",
    },
    "A3": {
        "category": "arithmetic",
        "answer": "95",
        "type": "numeric",
    },
    "A4": {
        "category": "arithmetic",
        "answer": "45",
        "type": "numeric",
    },

    "W1": {
        "category": "word_problem",
        "answer": "220",
        "type": "numeric",
    },
    "W2": {
        "category": "word_problem",
        "answer": "24",
        "type": "numeric",
    },
    "W3": {
        "category": "word_problem",
        "answer": "108",
        "type": "numeric",
    },

    "F1": {
        "category": "factual",
        "answer": "mars",
        "type": "word",
    },
    "F2": {
        "category": "factual",
        "answer": "oxygen",
        "type": "word",
    },
    "F3": {
        "category": "factual",
        "answer": "photosynthesis",
        "type": "word",
    },
}


CASE_NAMES = [
    "FP16",
    "BLOCK_GUIDED_7_67",
    "ACT_GUIDED_7_73",
]


# =============================================================================
# REMOVE THINKING SECTION
# =============================================================================
#
# Preferred scoring target:
#
# <think>
# reasoning...
# </think>
# FINAL ANSWER
#
# If </think> exists, score only text after it.
#
# If there is no closing think tag, use the entire response.
# =============================================================================

def remove_thinking(text):

    if "</think>" in text:

        return text.split(
            "</think>",
            1,
        )[1].strip()

    return text.strip()


# =============================================================================
# CLEAN TEXT
# =============================================================================

def normalize_text(text):

    text = text.lower()

    text = text.replace(
        ",",
        "",
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# =============================================================================
# NUMERIC EXTRACTION
# =============================================================================
#
# We prefer explicit answer-style expressions.
#
# Examples:
#
#   24 * 17 = 408
#                   -> 408
#
#   The answer is 408
#                   -> 408
#
#   Therefore, 408
#                   -> 408
#
# If no explicit answer marker exists, use the LAST numeric value.
#
# This fixes Experiment 016's bug where:
#
#   24 * 17 = 408
#
# was incorrectly scored using the FIRST number ("24").
# =============================================================================

def extract_numeric_answer(text):

    scoring_text = remove_thinking(
        text
    )

    normalized = normalize_text(
        scoring_text
    )


    # -------------------------------------------------------------------------
    # 1. Explicit "answer is X"
    # -------------------------------------------------------------------------

    patterns = [
        r"(?:final\s+answer|answer)\s*(?:is|=|:)?\s*(-?\d+(?:\.\d+)?)",

        r"(?:therefore|thus|so)\s*[,:]?\s*(-?\d+(?:\.\d+)?)",

        r"=\s*(-?\d+(?:\.\d+)?)",
    ]


    for pattern in patterns:

        matches = re.findall(
            pattern,
            normalized,
            flags=re.IGNORECASE,
        )

        if matches:

            return matches[-1]


    # -------------------------------------------------------------------------
    # 2. Otherwise use the LAST number
    # -------------------------------------------------------------------------

    numbers = re.findall(
        r"-?\d+(?:\.\d+)?",
        normalized,
    )


    if numbers:

        return numbers[-1]


    return None


# =============================================================================
# WORD EXTRACTION
# =============================================================================
#
# For factual one-word questions we first check the final non-empty line.
#
# Then fall back to the final alphabetic word.
# =============================================================================

def extract_word_answer(text):

    scoring_text = remove_thinking(
        text
    )


    lines = [
        line.strip()
        for line
        in scoring_text.splitlines()
        if line.strip()
    ]


    # -------------------------------------------------------------------------
    # Prefer a line containing exactly one word
    # -------------------------------------------------------------------------

    for line in reversed(lines):

        words = re.findall(
            r"[A-Za-z]+",
            line.lower(),
        )

        if len(words) == 1:

            return words[0]


    # -------------------------------------------------------------------------
    # Otherwise inspect full response
    # -------------------------------------------------------------------------

    words = re.findall(
        r"[A-Za-z]+",
        scoring_text.lower(),
    )


    if words:

        return words[-1]


    return None


# =============================================================================
# SCORE ONE RESPONSE
# =============================================================================

def score_response(
    text,
    expected_info,
):

    expected = (
        expected_info["answer"]
        .strip()
        .lower()
    )


    answer_type = (
        expected_info["type"]
    )


    if answer_type == "numeric":

        extracted = (
            extract_numeric_answer(
                text
            )
        )

    elif answer_type == "word":

        extracted = (
            extract_word_answer(
                text
            )
        )

    else:

        raise ValueError(
            f"Unknown answer type: "
            f"{answer_type}"
        )


    if extracted is None:

        return {
            "extracted": None,
            "correct": False,
        }


    extracted = (
        extracted
        .strip()
        .lower()
    )


    return {
        "extracted":
            extracted,

        "correct":
            extracted == expected,
    }


# =============================================================================
# PARSE EXPERIMENT 016 OUTPUT
# =============================================================================

def parse_results(text):

    results = defaultdict(dict)

    current_case = None


    # -------------------------------------------------------------------------
    # Split by lines but retain prompt blocks
    # -------------------------------------------------------------------------

    lines = text.splitlines()

    i = 0


    while i < len(lines):

        line = lines[i]


        # ---------------------------------------------------------------------
        # Detect validation case
        # ---------------------------------------------------------------------

        if "VALIDATION CASE" in line:

            for case_name in CASE_NAMES:

                if case_name in line:

                    current_case = (
                        case_name
                    )

                    break


        # ---------------------------------------------------------------------
        # Detect exact-answer prompt
        # ---------------------------------------------------------------------

        prompt_match = re.match(
            r"^\[(A[1-4]|W[1-3]|F[1-3])\]\s+",
            line.strip(),
        )


        if (
            prompt_match
            and current_case is not None
        ):

            prompt_id = (
                prompt_match.group(1)
            )


            # -----------------------------------------------------------------
            # Search for "Output:"
            # -----------------------------------------------------------------

            j = i + 1


            while (
                j < len(lines)
                and lines[j].strip()
                != "Output:"
            ):

                # Stop if another prompt starts unexpectedly
                if re.match(
                    r"^\[[A-Z]\d+\]",
                    lines[j].strip(),
                ):

                    break

                j += 1


            if (
                j >= len(lines)
                or lines[j].strip()
                != "Output:"
            ):

                print(
                    f"WARNING: Could not find "
                    f"Output for "
                    f"{current_case} "
                    f"{prompt_id}"
                )

                i += 1
                continue


            # -----------------------------------------------------------------
            # Collect output until experiment metadata begins
            # -----------------------------------------------------------------

            j += 1

            output_lines = []


            while j < len(lines):

                stripped = (
                    lines[j].strip()
                )


                if stripped.startswith(
                    "Exact    :"
                ):

                    break


                if stripped.startswith(
                    "Repetition score"
                ):

                    break


                if stripped.startswith(
                    "Generated tokens"
                ):

                    break


                if re.match(
                    r"^\[[A-Z]\d+\]",
                    stripped,
                ):

                    break


                output_lines.append(
                    lines[j]
                )

                j += 1


            output_text = (
                "\n".join(
                    output_lines
                )
                .strip()
            )


            results[
                current_case
            ][
                prompt_id
            ] = output_text


            i = j
            continue


        i += 1


    return results


# =============================================================================
# LOAD FILE
# =============================================================================

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"Could not find: "
        f"{INPUT_FILE}"
    )


text = INPUT_FILE.read_text(
    encoding="utf-8",
    errors="replace",
)


print(
    "=" * 120
)

print(
    "EXPERIMENT 016B — "
    "DETERMINISTIC RESCORING OF "
    "EXPERIMENT 016"
)

print(
    "=" * 120
)


print(
    f"\nInput file : "
    f"{INPUT_FILE}"
)

print(
    f"Output file: "
    f"{OUTPUT_FILE}"
)


# =============================================================================
# PARSE
# =============================================================================

parsed = parse_results(
    text
)


# =============================================================================
# VALIDATE PARSING
# =============================================================================

print(
    "\nPARSE VALIDATION"
)

print(
    "-" * 120
)


for case_name in CASE_NAMES:

    found = len(
        parsed.get(
            case_name,
            {}
        )
    )


    print(
        f"{case_name:<24}: "
        f"{found}/10 exact-answer "
        f"responses found"
    )


    if found != 10:

        missing = (
            set(EXPECTED)
            - set(
                parsed.get(
                    case_name,
                    {}
                )
            )
        )


        raise RuntimeError(
            f"{case_name}: expected 10 "
            f"responses but found {found}. "
            f"Missing: {sorted(missing)}"
        )


# =============================================================================
# RESCORE
# =============================================================================

case_summaries = {}


output_lines = []


def emit(line=""):

    print(line)

    output_lines.append(
        line
    )


emit()
emit("=" * 120)

emit(
    "RESCORED EXACT-ANSWER RESULTS"
)

emit("=" * 120)


for case_name in CASE_NAMES:

    emit()
    emit("#" * 120)

    emit(
        f"CASE — {case_name}"
    )

    emit("#" * 120)


    total = 0
    correct = 0

    category_stats = defaultdict(
        lambda: {
            "correct": 0,
            "total": 0,
        }
    )

    prompt_scores = {}


    for prompt_id, info in (
        EXPECTED.items()
    ):

        response = (
            parsed[
                case_name
            ][
                prompt_id
            ]
        )


        score = score_response(
            response,
            info,
        )


        total += 1


        category = (
            info["category"]
        )


        category_stats[
            category
        ][
            "total"
        ] += 1


        if score["correct"]:

            correct += 1

            category_stats[
                category
            ][
                "correct"
            ] += 1


        prompt_scores[
            prompt_id
        ] = score


        emit()

        emit(
            f"[{prompt_id}] "
            f"{category}"
        )

        emit(
            f"Expected  : "
            f"{info['answer']}"
        )

        emit(
            f"Extracted : "
            f"{score['extracted']}"
        )

        emit(
            "Result    : "
            + (
                "PASS"
                if score["correct"]
                else "FAIL"
            )
        )


        # Include compact response for auditability
        compact_response = re.sub(
            r"\s+",
            " ",
            response,
        ).strip()


        if len(compact_response) > 250:

            compact_response = (
                compact_response[:250]
                + "..."
            )


        emit(
            f"Response  : "
            f"{compact_response}"
        )


    accuracy = (
        correct
        / total
        * 100
    )


    emit()
    emit("-" * 120)

    emit(
        f"Correct   : "
        f"{correct}/{total}"
    )

    emit(
        f"Accuracy  : "
        f"{accuracy:.2f}%"
    )


    for category in [
        "arithmetic",
        "word_problem",
        "factual",
    ]:

        stats = (
            category_stats[
                category
            ]
        )


        category_accuracy = (
            stats["correct"]
            / stats["total"]
            * 100
        )


        emit(
            f"{category:<14}: "
            f"{stats['correct']}/"
            f"{stats['total']} "
            f"({category_accuracy:.2f}%)"
        )


    case_summaries[
        case_name
    ] = {
        "correct":
            correct,

        "total":
            total,

        "accuracy":
            accuracy,

        "categories":
            dict(
                category_stats
            ),

        "prompts":
            prompt_scores,
    }


# =============================================================================
# CROSS-CONFIGURATION SUMMARY
# =============================================================================

emit()
emit()
emit("=" * 120)

emit(
    "CROSS-CONFIGURATION "
    "RESCORING SUMMARY"
)

emit("=" * 120)


emit(
    f"{'Configuration':<24}"
    f"{'Correct':>12}"
    f"{'Accuracy':>14}"
    f"{'Arithmetic':>16}"
    f"{'Word Problems':>18}"
    f"{'Factual':>14}"
)


emit(
    "-" * 120
)


for case_name in CASE_NAMES:

    summary = (
        case_summaries[
            case_name
        ]
    )


    arithmetic = (
        summary[
            "categories"
        ][
            "arithmetic"
        ]
    )

    word_problem = (
        summary[
            "categories"
        ][
            "word_problem"
        ]
    )

    factual = (
        summary[
            "categories"
        ][
            "factual"
        ]
    )


    emit(
        f"{case_name:<24}"
        f"{summary['correct']:>5}/"
        f"{summary['total']:<6}"
        f"{summary['accuracy']:>13.2f}%"
        f"{arithmetic['correct']:>7}/"
        f"{arithmetic['total']:<8}"
        f"{word_problem['correct']:>8}/"
        f"{word_problem['total']:<9}"
        f"{factual['correct']:>6}/"
        f"{factual['total']:<7}"
    )


# =============================================================================
# PER-PROMPT COMPARISON
# =============================================================================

emit()
emit()
emit("=" * 120)

emit(
    "PER-PROMPT COMPARISON"
)

emit("=" * 120)


emit(
    f"{'Prompt':<10}"
    f"{'Expected':<20}"
    + "".join(
        f"{case:>24}"
        for case
        in CASE_NAMES
    )
)


emit(
    "-" * 120
)


for prompt_id, info in (
    EXPECTED.items()
):

    row = (
        f"{prompt_id:<10}"
        f"{info['answer']:<20}"
    )


    for case_name in CASE_NAMES:

        score = (
            case_summaries[
                case_name
            ][
                "prompts"
            ][
                prompt_id
            ]
        )


        if score["correct"]:

            cell = (
                f"PASS ({score['extracted']})"
            )

        else:

            cell = (
                f"FAIL ({score['extracted']})"
            )


        row += (
            f"{cell:>24}"
        )


    emit(
        row
    )


# =============================================================================
# RETENTION RELATIVE TO FP16
# =============================================================================

emit()
emit()
emit("=" * 120)

emit(
    "RETENTION RELATIVE TO FP16"
)

emit("=" * 120)


fp16_correct = (
    case_summaries[
        "FP16"
    ][
        "correct"
    ]
)


for case_name in [
    "BLOCK_GUIDED_7_67",
    "ACT_GUIDED_7_73",
]:

    ternary_correct = (
        case_summaries[
            case_name
        ][
            "correct"
        ]
    )


    if fp16_correct > 0:

        retention = (
            ternary_correct
            / fp16_correct
            * 100
        )

        emit(
            f"{case_name:<24}: "
            f"{ternary_correct}/"
            f"{fp16_correct} "
            f"= {retention:.2f}% "
            f"of FP16 exact-answer count"
        )

    else:

        emit(
            f"{case_name:<24}: "
            "undefined because FP16 "
            "correct count is zero"
        )


# =============================================================================
# SCIENTIFIC NOTES
# =============================================================================

emit()
emit()
emit("=" * 120)

emit(
    "SCIENTIFIC NOTES"
)

emit("=" * 120)


emit(
    "1. Experiment 016B does not run or "
    "modify the language model."
)

emit(
    "2. It deterministically rescores the "
    "responses already generated by "
    "Experiment 016."
)

emit(
    "3. When </think> is present, only text "
    "after the closing tag is preferred for "
    "answer extraction."
)

emit(
    "4. Numeric scoring prefers explicit "
    "answer expressions and otherwise uses "
    "the final numeric value rather than the "
    "first numeric value."
)

emit(
    "5. The open-ended responses are not "
    "rescored automatically here."
)

emit(
    "6. Exact-answer retention is not a "
    "complete model-quality metric and must "
    "be interpreted together with the "
    "repetition and qualitative results "
    "from Experiment 016."
)


emit()
emit("=" * 120)

emit(
    "EXPERIMENT 016B COMPLETE"
)

emit("=" * 120)


# =============================================================================
# SAVE
# =============================================================================

OUTPUT_FILE.write_text(
    "\n".join(
        output_lines
    )
    + "\n",
    encoding="utf-8",
)


print(
    f"\nSaved rescored results to: "
    f"{OUTPUT_FILE}"
)
