import re
from pathlib import Path
from collections import defaultdict


# =============================================================================
# EXPERIMENT 016C
#
# Final deterministic rescoring of Experiment 016.
#
# IMPORTANT:
# - Does NOT load the model.
# - Does NOT use GPU.
# - Does NOT regenerate responses.
# - Reads the existing Experiment 016 output.
# - Extracts the FIRST explicit answer produced by the model.
# - Reports PASS / FAIL / UNRESOLVED.
# =============================================================================


INPUT_FILE = Path(
    "results/016_formal_validation.txt"
)

OUTPUT_FILE = Path(
    "results/016c_final_rescore.txt"
)


# =============================================================================
# EXPECTED ANSWERS
# =============================================================================

EXPECTED = {

    # -------------------------------------------------------------------------
    # Arithmetic
    # -------------------------------------------------------------------------

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


    # -------------------------------------------------------------------------
    # Word problems
    # -------------------------------------------------------------------------

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


    # -------------------------------------------------------------------------
    # Factual
    # -------------------------------------------------------------------------

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
# TEXT NORMALIZATION
# =============================================================================

def normalize_numeric(value):

    if value is None:
        return None

    value = value.strip()

    value = value.replace(
        ",",
        "",
    )

    value = value.rstrip(
        "."
    )

    return value


def compact_text(text):

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


# =============================================================================
# NUMERIC ANSWER EXTRACTION
# =============================================================================
#
# Priority:
#
# 1. First explicit:
#       Answer: 408
#       Answer is 408
#       Final answer: 408
#       Final number is 408
#
# 2. First completed equation:
#       24 * 17 = 408
#
# 3. First direct numeric-only response:
#       16
#
# Otherwise:
#       UNRESOLVED
#
# IMPORTANT:
# We intentionally DO NOT use the last number in the response.
# =============================================================================

def extract_numeric_answer(text):

    normalized = compact_text(
        text
    )


    # =========================================================================
    # PRIORITY 1 — EXPLICIT ANSWER PHRASES
    # =========================================================================

    explicit_patterns = [

        # Answer: 408
        # Answer = 408
        # Answer is 408
        r"\banswer\s*(?:is|=|:)\s*"
        r"\*{0,2}(-?\d+(?:\.\d+)?)\*{0,2}",

        # Final answer: 408
        # Final answer is 408
        r"\bfinal\s+answer\s*(?:is|=|:)\s*"
        r"\*{0,2}(-?\d+(?:\.\d+)?)\*{0,2}",

        # Final number is 45
        # Final number: 45
        r"\bfinal\s+number\s*(?:is|=|:)\s*"
        r"\*{0,2}(-?\d+(?:\.\d+)?)\*{0,2}",
    ]


    candidates = []


    for pattern in explicit_patterns:

        match = re.search(
            pattern,
            normalized,
            flags=re.IGNORECASE,
        )

        if match:

            candidates.append(
                (
                    match.start(),
                    normalize_numeric(
                        match.group(1)
                    ),
                    "explicit_answer",
                )
            )


    if candidates:

        candidates.sort(
            key=lambda x: x[0]
        )

        _, value, method = (
            candidates[0]
        )

        return {
            "value": value,
            "method": method,
        }


    # =========================================================================
    # PRIORITY 2 — FIRST COMPLETED EQUATION
    # =========================================================================
    #
    # Examples:
    #
    # 24 * 17 = 408
    # 37 + 58 = 95
    # 432 / 18 = 24
    # 91 - 46 = 45
    # 55 \times 4 = 220
    #
    # We capture the first number immediately after "=".
    # =========================================================================

    equation_match = re.search(
        r"=\s*"
        r"\*{0,2}"
        r"(-?\d+(?:\.\d+)?)"
        r"\*{0,2}",
        normalized,
    )


    if equation_match:

        return {
            "value":
                normalize_numeric(
                    equation_match.group(1)
                ),

            "method":
                "first_equation",
        }


    # =========================================================================
    # PRIORITY 3 — DIRECT NUMERIC RESPONSE
    # =========================================================================
    #
    # Examples:
    #
    # 16
    # 24.
    #
    # This only succeeds if the response STARTS as a direct number and the
    # immediate response structure looks like an answer rather than reasoning.
    # =========================================================================

    direct_match = re.match(
        r"^\s*"
        r"\*{0,2}"
        r"(-?\d+(?:\.\d+)?)"
        r"\*{0,2}"
        r"(?:\s*(?:km|kilometers?|apples?)\b)?"
        r"(?:[.!]|\s|$)",
        normalized,
        flags=re.IGNORECASE,
    )


    if direct_match:

        return {
            "value":
                normalize_numeric(
                    direct_match.group(1)
                ),

            "method":
                "direct_numeric",
        }


    # =========================================================================
    # NO RELIABLE NUMERIC ANSWER FOUND
    # =========================================================================

    return {
        "value": None,
        "method": "unresolved",
    }


# =============================================================================
# FACTUAL WORD ANSWER EXTRACTION
# =============================================================================
#
# These prompts explicitly requested one word.
#
# We use a conservative strategy:
#
# 1. Explicit answer phrase:
#       Answer: Mars
#       Answer is oxygen
#
# 2. Direct one-word response:
#       Photosynthesis
#
# 3. Expected answer appears in a clear answer statement:
#       "The planet ... is Mars"
#       "... is called photosynthesis"
#
# Otherwise UNRESOLVED.
#
# We DO NOT simply search anywhere inside chain-of-thought because that could
# incorrectly score reasoning text as the final answer.
# =============================================================================

def extract_word_answer(
    text,
    expected,
):

    normalized = compact_text(
        text
    )

    expected_lower = (
        expected.lower()
    )


    # =========================================================================
    # PRIORITY 1 — EXPLICIT ANSWER
    # =========================================================================

    explicit_patterns = [

        r"\banswer\s*(?:is|=|:)\s*"
        r"\*{0,2}([A-Za-z]+)\*{0,2}",

        r"\bfinal\s+answer\s*(?:is|=|:)\s*"
        r"\*{0,2}([A-Za-z]+)\*{0,2}",
    ]


    candidates = []


    for pattern in explicit_patterns:

        match = re.search(
            pattern,
            normalized,
            flags=re.IGNORECASE,
        )

        if match:

            candidates.append(
                (
                    match.start(),
                    match.group(1).lower(),
                    "explicit_answer",
                )
            )


    if candidates:

        candidates.sort(
            key=lambda x: x[0]
        )

        _, value, method = (
            candidates[0]
        )

        return {
            "value": value,
            "method": method,
        }


    # =========================================================================
    # PRIORITY 2 — DIRECT ONE-WORD RESPONSE
    # =========================================================================

    direct_match = re.fullmatch(
        r"\s*"
        r"\*{0,2}"
        r"([A-Za-z]+)"
        r"\*{0,2}"
        r"[.!]?"
        r"\s*",
        normalized,
    )


    if direct_match:

        return {
            "value":
                direct_match
                .group(1)
                .lower(),

            "method":
                "direct_word",
        }


    # =========================================================================
    # PRIORITY 3 — CLEAR ANSWER STATEMENT CONTAINING EXPECTED WORD
    # =========================================================================
    #
    # Examples:
    #
    # The planet known as the Red Planet is Mars.
    #
    # The gas humans need to breathe is oxygen.
    #
    # The process ... is called photosynthesis.
    #
    # This is intentionally narrower than simply:
    #
    #     if expected in text
    #
    # because expected words can appear inside reasoning.
    # =========================================================================

    escaped_expected = re.escape(
        expected_lower
    )


    clear_patterns = [

        rf"\bis\s+(?:called\s+)?"
        rf"\*{{0,2}}{escaped_expected}\*{{0,2}}\b",

        rf"\bcalled\s+"
        rf"\*{{0,2}}{escaped_expected}\*{{0,2}}\b",

        rf"\bknown\s+as\s+"
        rf"\*{{0,2}}{escaped_expected}\*{{0,2}}\b",
    ]


    for pattern in clear_patterns:

        match = re.search(
            pattern,
            normalized,
            flags=re.IGNORECASE,
        )

        if match:

            return {
                "value":
                    expected_lower,

                "method":
                    "clear_answer_statement",
            }


    # =========================================================================
    # UNRESOLVED
    # =========================================================================

    return {
        "value": None,
        "method": "unresolved",
    }


# =============================================================================
# SCORE RESPONSE
# =============================================================================

def score_response(
    text,
    expected_info,
):

    expected = (
        expected_info[
            "answer"
        ]
        .strip()
        .lower()
    )


    answer_type = (
        expected_info[
            "type"
        ]
    )


    if answer_type == "numeric":

        extraction = (
            extract_numeric_answer(
                text
            )
        )


    elif answer_type == "word":

        extraction = (
            extract_word_answer(
                text,
                expected,
            )
        )


    else:

        raise ValueError(
            f"Unknown answer type: "
            f"{answer_type}"
        )


    extracted = (
        extraction["value"]
    )


    method = (
        extraction["method"]
    )


    if extracted is None:

        status = "UNRESOLVED"

    elif extracted.lower() == expected:

        status = "PASS"

    else:

        status = "FAIL"


    return {
        "expected":
            expected,

        "extracted":
            extracted,

        "method":
            method,

        "status":
            status,
    }


# =============================================================================
# PARSE EXPERIMENT 016 OUTPUT
# =============================================================================

def parse_results(text):

    results = defaultdict(
        dict
    )


    current_case = None

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
            # Find Output:
            # -----------------------------------------------------------------

            j = i + 1


            while j < len(lines):

                stripped = (
                    lines[j].strip()
                )


                if stripped == "Output:":

                    break


                if re.match(
                    r"^\[[A-Z]\d+\]",
                    stripped,
                ):

                    break


                j += 1


            if (
                j >= len(lines)
                or lines[j].strip()
                != "Output:"
            ):

                print(
                    "WARNING: Could not "
                    "find Output for "
                    f"{current_case} "
                    f"{prompt_id}"
                )

                i += 1
                continue


            # -----------------------------------------------------------------
            # Collect response
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


                if stripped.startswith(
                    "Generation time"
                ):

                    break


                if stripped.startswith(
                    "Tokens/sec"
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
# LOAD INPUT
# =============================================================================

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"Could not find "
        f"{INPUT_FILE}"
    )


raw_text = INPUT_FILE.read_text(
    encoding="utf-8",
    errors="replace",
)


print(
    "=" * 130
)

print(
    "EXPERIMENT 016C — "
    "FINAL DETERMINISTIC RESCORING"
)

print(
    "=" * 130
)


print(
    f"\nInput  : "
    f"{INPUT_FILE}"
)

print(
    f"Output : "
    f"{OUTPUT_FILE}"
)


# =============================================================================
# PARSE
# =============================================================================

parsed = parse_results(
    raw_text
)


print(
    "\nPARSE VALIDATION"
)

print(
    "-" * 130
)


for case_name in CASE_NAMES:

    responses = (
        parsed.get(
            case_name,
            {}
        )
    )


    found = len(
        responses
    )


    print(
        f"{case_name:<24}: "
        f"{found}/10 exact responses"
    )


    if found != 10:

        missing = (
            set(EXPECTED)
            - set(responses)
        )


        raise RuntimeError(
            f"{case_name}: expected "
            f"10 responses but found "
            f"{found}. Missing: "
            f"{sorted(missing)}"
        )


# =============================================================================
# OUTPUT HELPER
# =============================================================================

output_lines = []


def emit(line=""):

    print(line)

    output_lines.append(
        line
    )


# =============================================================================
# RESCORE
# =============================================================================

case_summaries = {}


emit()
emit("=" * 130)

emit(
    "FINAL RESCORED EXACT-ANSWER RESULTS"
)

emit("=" * 130)


for case_name in CASE_NAMES:

    emit()
    emit("#" * 130)

    emit(
        f"CASE — {case_name}"
    )

    emit("#" * 130)


    counts = {
        "PASS": 0,
        "FAIL": 0,
        "UNRESOLVED": 0,
    }


    category_stats = defaultdict(
        lambda: {
            "PASS": 0,
            "FAIL": 0,
            "UNRESOLVED": 0,
            "TOTAL": 0,
        }
    )


    prompt_results = {}


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


        result = score_response(
            response,
            info,
        )


        status = (
            result["status"]
        )


        category = (
            info["category"]
        )


        counts[
            status
        ] += 1


        category_stats[
            category
        ][
            status
        ] += 1


        category_stats[
            category
        ][
            "TOTAL"
        ] += 1


        prompt_results[
            prompt_id
        ] = result


        emit()

        emit(
            f"[{prompt_id}] "
            f"{category}"
        )


        emit(
            f"Expected   : "
            f"{result['expected']}"
        )


        emit(
            f"Extracted  : "
            f"{result['extracted']}"
        )


        emit(
            f"Method     : "
            f"{result['method']}"
        )


        emit(
            f"Status     : "
            f"{status}"
        )


        response_compact = (
            compact_text(
                response
            )
        )


        if len(
            response_compact
        ) > 300:

            response_compact = (
                response_compact[
                    :300
                ]
                + "..."
            )


        emit(
            f"Response   : "
            f"{response_compact}"
        )


    # =========================================================================
    # SUMMARY
    # =========================================================================

    total = sum(
        counts.values()
    )


    resolved = (
        counts["PASS"]
        + counts["FAIL"]
    )


    # -------------------------------------------------------------------------
    # Strict accuracy:
    #
    # PASS / all 10 prompts.
    #
    # Unresolved prompts remain visible in denominator.
    # -------------------------------------------------------------------------

    strict_accuracy = (
        counts["PASS"]
        / total
        * 100
        if total > 0
        else 0.0
    )


    # -------------------------------------------------------------------------
    # Resolved-only accuracy:
    #
    # PASS / (PASS + FAIL)
    #
    # Reported separately so unresolved extraction is transparent.
    # -------------------------------------------------------------------------

    resolved_accuracy = (
        counts["PASS"]
        / resolved
        * 100
        if resolved > 0
        else 0.0
    )


    emit()
    emit("-" * 130)


    emit(
        f"PASS              : "
        f"{counts['PASS']}"
    )

    emit(
        f"FAIL              : "
        f"{counts['FAIL']}"
    )

    emit(
        f"UNRESOLVED        : "
        f"{counts['UNRESOLVED']}"
    )

    emit(
        f"TOTAL             : "
        f"{total}"
    )

    emit(
        f"Strict accuracy   : "
        f"{strict_accuracy:.2f}%"
    )

    emit(
        f"Resolved accuracy : "
        f"{resolved_accuracy:.2f}%"
    )


    emit()

    emit(
        "Category breakdown:"
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


        emit(
            f"{category:<14} "
            f"PASS={stats['PASS']} "
            f"FAIL={stats['FAIL']} "
            f"UNRESOLVED="
            f"{stats['UNRESOLVED']} "
            f"TOTAL={stats['TOTAL']}"
        )


    case_summaries[
        case_name
    ] = {
        "counts":
            counts,

        "total":
            total,

        "resolved":
            resolved,

        "strict_accuracy":
            strict_accuracy,

        "resolved_accuracy":
            resolved_accuracy,

        "categories":
            {
                category:
                    dict(values)

                for category, values
                in category_stats.items()
            },

        "prompts":
            prompt_results,
    }


# =============================================================================
# CROSS-CONFIGURATION SUMMARY
# =============================================================================

emit()
emit()
emit("=" * 145)

emit(
    "CROSS-CONFIGURATION SUMMARY"
)

emit("=" * 145)


emit(
    f"{'Configuration':<24}"
    f"{'PASS':>8}"
    f"{'FAIL':>8}"
    f"{'UNRES':>8}"
    f"{'Strict Acc':>14}"
    f"{'Resolved Acc':>16}"
)


emit(
    "-" * 145
)


for case_name in CASE_NAMES:

    summary = (
        case_summaries[
            case_name
        ]
    )


    counts = (
        summary[
            "counts"
        ]
    )


    emit(
        f"{case_name:<24}"
        f"{counts['PASS']:>8}"
        f"{counts['FAIL']:>8}"
        f"{counts['UNRESOLVED']:>8}"
        f"{summary['strict_accuracy']:>13.2f}%"
        f"{summary['resolved_accuracy']:>15.2f}%"
    )


# =============================================================================
# PER-PROMPT COMPARISON
# =============================================================================

emit()
emit()
emit("=" * 145)

emit(
    "PER-PROMPT FINAL COMPARISON"
)

emit("=" * 145)


emit(
    f"{'Prompt':<10}"
    f"{'Expected':<20}"
    + "".join(
        f"{case_name:>30}"
        for case_name
        in CASE_NAMES
    )
)


emit(
    "-" * 145
)


for prompt_id, info in (
    EXPECTED.items()
):

    row = (
        f"{prompt_id:<10}"
        f"{info['answer']:<20}"
    )


    for case_name in CASE_NAMES:

        result = (
            case_summaries[
                case_name
            ][
                "prompts"
            ][
                prompt_id
            ]
        )


        extracted = (
            result["extracted"]
            if result["extracted"]
            is not None
            else "-"
        )


        cell = (
            f"{result['status']} "
            f"({extracted})"
        )


        row += (
            f"{cell:>30}"
        )


    emit(
        row
    )


# =============================================================================
# RETENTION RELATIVE TO FP16
# =============================================================================

emit()
emit()
emit("=" * 145)

emit(
    "EXACT-TASK RETENTION RELATIVE TO FP16"
)

emit("=" * 145)


fp16_pass = (
    case_summaries[
        "FP16"
    ][
        "counts"
    ][
        "PASS"
    ]
)


emit(
    f"FP16 PASS count: "
    f"{fp16_pass}"
)


for case_name in [
    "BLOCK_GUIDED_7_67",
    "ACT_GUIDED_7_73",
]:

    ternary_pass = (
        case_summaries[
            case_name
        ][
            "counts"
        ][
            "PASS"
        ]
    )


    if fp16_pass > 0:

        retention = (
            ternary_pass
            / fp16_pass
            * 100
        )


        emit(
            f"{case_name:<24}: "
            f"{ternary_pass}/"
            f"{fp16_pass} "
            f"= {retention:.2f}%"
        )


    else:

        emit(
            f"{case_name:<24}: "
            "undefined because "
            "FP16 PASS count is zero"
        )


# =============================================================================
# AGREEMENT / DISAGREEMENT ANALYSIS
# =============================================================================

emit()
emit()
emit("=" * 145)

emit(
    "FP16 VS TERNARY PER-PROMPT AGREEMENT"
)

emit("=" * 145)


for case_name in [
    "BLOCK_GUIDED_7_67",
    "ACT_GUIDED_7_73",
]:

    both_pass = 0

    fp16_pass_ternary_not = 0

    fp16_not_ternary_pass = 0

    neither_pass = 0


    for prompt_id in EXPECTED:

        fp16_status = (
            case_summaries[
                "FP16"
            ][
                "prompts"
            ][
                prompt_id
            ][
                "status"
            ]
        )


        ternary_status = (
            case_summaries[
                case_name
            ][
                "prompts"
            ][
                prompt_id
            ][
                "status"
            ]
        )


        fp16_ok = (
            fp16_status
            == "PASS"
        )


        ternary_ok = (
            ternary_status
            == "PASS"
        )


        if (
            fp16_ok
            and ternary_ok
        ):

            both_pass += 1


        elif (
            fp16_ok
            and not ternary_ok
        ):

            fp16_pass_ternary_not += 1


        elif (
            not fp16_ok
            and ternary_ok
        ):

            fp16_not_ternary_pass += 1


        else:

            neither_pass += 1


    emit()

    emit(
        case_name
    )


    emit(
        f"Both PASS                 : "
        f"{both_pass}"
    )

    emit(
        f"FP16 PASS / ternary not   : "
        f"{fp16_pass_ternary_not}"
    )

    emit(
        f"FP16 not / ternary PASS   : "
        f"{fp16_not_ternary_pass}"
    )

    emit(
        f"Neither PASS              : "
        f"{neither_pass}"
    )


# =============================================================================
# INTERPRETATION RULES
# =============================================================================

emit()
emit()
emit("=" * 145)

emit(
    "INTERPRETATION RULES"
)

emit("=" * 145)


emit(
    "1. 016C does not run or modify "
    "the language model."
)

emit(
    "2. It rescored the exact-answer "
    "responses already generated in "
    "Experiment 016."
)

emit(
    "3. The first explicit answer is "
    "preferred over later repetition "
    "or degeneration."
)

emit(
    "4. PASS means the first reliably "
    "extractable answer matches the "
    "expected answer."
)

emit(
    "5. FAIL means a reliably "
    "extractable answer exists but "
    "is incorrect."
)

emit(
    "6. UNRESOLVED means the evaluator "
    "could not reliably identify an "
    "answer; it must not automatically "
    "be interpreted as a model error."
)

emit(
    "7. Strict accuracy uses all exact "
    "prompts in the denominator."
)

emit(
    "8. Resolved accuracy excludes "
    "UNRESOLVED cases and is reported "
    "separately."
)

emit(
    "9. Exact-answer performance must "
    "be interpreted together with the "
    "repetition and qualitative "
    "open-ended behavior from "
    "Experiment 016."
)

emit(
    "10. Throughput from Experiment 016 "
    "is not ternary acceleration because "
    "the ternary weights were reconstructed "
    "into FP16 tensors."
)


emit()
emit("=" * 145)

emit(
    "EXPERIMENT 016C COMPLETE"
)

emit("=" * 145)


# =============================================================================
# SAVE RESULTS
# =============================================================================

OUTPUT_FILE.write_text(
    "\n".join(
        output_lines
    )
    + "\n",
    encoding="utf-8",
)


print(
    f"\nSaved final rescoring to: "
    f"{OUTPUT_FILE}"
)
