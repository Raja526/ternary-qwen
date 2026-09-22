import re
from pathlib import Path
from collections import defaultdict


# =============================================================================
# EXPERIMENT 016E
#
# Final deterministic scoring of Experiment 016D outputs.
#
# NO MODEL LOAD
# NO GPU
# NO REGENERATION
#
# Two independent dimensions:
#
# 1. Answer correctness
#       CORRECT
#       WRONG
#       NO_ANSWER
#
# 2. Format compliance
#       COMPLIANT
#       NON_COMPLIANT
#
# Example:
#
#   "45"
#       CORRECT + COMPLIANT
#
#   "91 - 46 = 45"
#       CORRECT + NON_COMPLIANT
#
#   "37 + 58 = 91"
#       WRONG + NON_COMPLIANT
#
#   "To find the total distance..."
#       NO_ANSWER + NON_COMPLIANT
# =============================================================================


INPUT_FILE = Path(
    "results/016d_clean_exact_validation.txt"
)

OUTPUT_FILE = Path(
    "results/016e_final_scoring.txt"
)


# =============================================================================
# EXPECTED ANSWERS
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
# HELPERS
# =============================================================================

def compact_text(text):

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def clean_markdown(text):

    text = text.replace(
        "**",
        "",
    )

    text = text.replace(
        "__",
        "",
    )

    return text.strip()


def normalize_number(value):

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


# =============================================================================
# FORMAT COMPLIANCE
# =============================================================================
#
# Numeric:
#
#     408
#     408.
#
# accepted as compliant.
#
# Because the prompt explicitly says:
#
#     "Answer with only the final number."
#
#
# Word:
#
#     Mars
#     Oxygen
#     Photosynthesis
#
# accepted.
#
# Anything containing an equation/explanation is NON_COMPLIANT.
# =============================================================================

def check_format_compliance(
    text,
    answer_type,
):

    cleaned = clean_markdown(
        text
    ).strip()


    if answer_type == "numeric":

        match = re.fullmatch(
            r"\s*"
            r"-?\d+(?:\.\d+)?"
            r"[.!]?"
            r"\s*",
            cleaned,
        )

        return (
            "COMPLIANT"
            if match
            else "NON_COMPLIANT"
        )


    if answer_type == "word":

        match = re.fullmatch(
            r"\s*"
            r"[A-Za-z]+"
            r"[.!]?"
            r"\s*",
            cleaned,
        )

        return (
            "COMPLIANT"
            if match
            else "NON_COMPLIANT"
        )


    raise ValueError(
        f"Unknown answer type: "
        f"{answer_type}"
    )


# =============================================================================
# NUMERIC ANSWER EXTRACTION
# =============================================================================
#
# This is intentionally conservative.
#
# Priority:
#
# 1. Direct numeric response
#
#       45
#
# 2. Explicit answer expression
#
#       Answer: 45
#       Answer is 45
#       final answer = 45
#
# 3. Completed arithmetic equation
#
#       91 - 46 = 45
#       24 × 17 = 408
#       432 ÷ 18 = 24
#
# We DO NOT extract arbitrary numbers from unfinished prose.
#
# Example:
#
#   "The box contains 12 rows, each with 9 apples. Therefore,"
#
# becomes:
#
#   NO_ANSWER
#
# rather than 12 or 9.
# =============================================================================

def extract_numeric_answer(text):

    cleaned = clean_markdown(
        text
    )

    compact = compact_text(
        cleaned
    )


    # =========================================================================
    # 1. DIRECT NUMERIC RESPONSE
    # =========================================================================

    direct = re.fullmatch(
        r"\s*"
        r"(-?\d+(?:\.\d+)?)"
        r"[.!]?"
        r"\s*",
        compact,
    )


    if direct:

        return {
            "answer":
                normalize_number(
                    direct.group(1)
                ),

            "method":
                "direct_numeric",
        }


    # =========================================================================
    # 2. EXPLICIT ANSWER
    # =========================================================================

    explicit_patterns = [

        r"\bfinal\s+answer\s*"
        r"(?:is|=|:)\s*"
        r"(-?\d+(?:\.\d+)?)",

        r"\banswer\s*"
        r"(?:is|=|:)\s*"
        r"(-?\d+(?:\.\d+)?)",

        r"\bfinal\s+number\s*"
        r"(?:is|=|:)\s*"
        r"(-?\d+(?:\.\d+)?)",
    ]


    candidates = []


    for pattern in explicit_patterns:

        match = re.search(
            pattern,
            compact,
            flags=re.IGNORECASE,
        )

        if match:

            candidates.append(
                (
                    match.start(),
                    normalize_number(
                        match.group(1)
                    ),
                    "explicit_answer",
                )
            )


    if candidates:

        candidates.sort(
            key=lambda item: item[0]
        )

        _, answer, method = (
            candidates[0]
        )

        return {
            "answer": answer,
            "method": method,
        }


    # =========================================================================
    # 3. COMPLETED EQUATION
    # =========================================================================
    #
    # Important:
    #
    # We require something arithmetic-looking BEFORE "=".
    #
    # This avoids treating:
    #
    #     Speed = 55
    #
    # as the answer.
    #
    # Supported examples:
    #
    #     24 * 17 = 408
    #     24 × 17 = 408
    #     432 / 18 = 24
    #     432 ÷ 18 = 24
    #     37 + 58 = 95
    #     91 - 46 = 45
    #     24 multiplied by 17 equals 408
    # =============================================================================

    symbolic_equation = re.search(
        r"(-?\d+(?:\.\d+)?)"
        r"\s*"
        r"(?:"
        r"\+|-|\*|/|×|÷"
        r")"
        r"\s*"
        r"(-?\d+(?:\.\d+)?)"
        r"\s*"
        r"=\s*"
        r"(-?\d+(?:\.\d+)?)",
        compact,
    )


    if symbolic_equation:

        return {
            "answer":
                normalize_number(
                    symbolic_equation.group(3)
                ),

            "method":
                "completed_equation",
        }


    # =========================================================================
    # Natural-language multiplication
    # =========================================================================

    multiplied = re.search(
        r"(-?\d+(?:\.\d+)?)"
        r"\s+multiplied\s+by\s+"
        r"(-?\d+(?:\.\d+)?)"
        r"\s+(?:equals|is)\s+"
        r"(-?\d+(?:\.\d+)?)",
        compact,
        flags=re.IGNORECASE,
    )


    if multiplied:

        return {
            "answer":
                normalize_number(
                    multiplied.group(3)
                ),

            "method":
                "completed_equation_words",
        }


    # =========================================================================
    # Natural-language division
    # =========================================================================

    divided = re.search(
        r"(-?\d+(?:\.\d+)?)"
        r"\s+divided\s+by\s+"
        r"(-?\d+(?:\.\d+)?)"
        r"\s+(?:equals|is)\s+"
        r"(-?\d+(?:\.\d+)?)",
        compact,
        flags=re.IGNORECASE,
    )


    if divided:

        return {
            "answer":
                normalize_number(
                    divided.group(3)
                ),

            "method":
                "completed_equation_words",
        }


    # =========================================================================
    # Natural-language plus
    # =========================================================================

    plus = re.search(
        r"(-?\d+(?:\.\d+)?)"
        r"\s+plus\s+"
        r"(-?\d+(?:\.\d+)?)"
        r"\s+(?:equals|is)\s+"
        r"(-?\d+(?:\.\d+)?)",
        compact,
        flags=re.IGNORECASE,
    )


    if plus:

        return {
            "answer":
                normalize_number(
                    plus.group(3)
                ),

            "method":
                "completed_equation_words",
        }


    # =========================================================================
    # Natural-language minus
    # =========================================================================

    minus = re.search(
        r"(-?\d+(?:\.\d+)?)"
        r"\s+minus\s+"
        r"(-?\d+(?:\.\d+)?)"
        r"\s+(?:equals|is)\s+"
        r"(-?\d+(?:\.\d+)?)",
        compact,
        flags=re.IGNORECASE,
    )


    if minus:

        return {
            "answer":
                normalize_number(
                    minus.group(3)
                ),

            "method":
                "completed_equation_words",
        }


    # =========================================================================
    # NO RELIABLE ANSWER
    # =========================================================================

    return {
        "answer": None,
        "method": "no_answer",
    }


# =============================================================================
# WORD ANSWER EXTRACTION
# =============================================================================
#
# 016D factual outputs are short because thinking was disabled.
#
# We support:
#
#     Mars
#     Earth
#     Oxygen
#     Photosynthesis
#
# Also support explicit:
#
#     Answer: Mars
#
# But we do NOT search for the expected answer anywhere in arbitrary prose.
# =============================================================================

def extract_word_answer(text):

    cleaned = clean_markdown(
        text
    )

    compact = compact_text(
        cleaned
    )


    # =========================================================================
    # DIRECT ONE-WORD RESPONSE
    # =========================================================================

    direct = re.fullmatch(
        r"\s*"
        r"([A-Za-z]+)"
        r"[.!]?"
        r"\s*",
        compact,
    )


    if direct:

        return {
            "answer":
                direct.group(1).lower(),

            "method":
                "direct_word",
        }


    # =========================================================================
    # EXPLICIT ANSWER
    # =========================================================================

    explicit = re.search(
        r"\b(?:final\s+answer|answer)"
        r"\s*(?:is|=|:)\s*"
        r"([A-Za-z]+)",
        compact,
        flags=re.IGNORECASE,
    )


    if explicit:

        return {
            "answer":
                explicit.group(1).lower(),

            "method":
                "explicit_answer",
        }


    return {
        "answer": None,
        "method": "no_answer",
    }


# =============================================================================
# SCORE ONE RESPONSE
# =============================================================================

def score_response(
    text,
    info,
):

    expected = (
        info["answer"]
        .strip()
        .lower()
    )


    answer_type = (
        info["type"]
    )


    format_status = (
        check_format_compliance(
            text,
            answer_type,
        )
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
                text
            )
        )


    else:

        raise ValueError(
            f"Unknown answer type: "
            f"{answer_type}"
        )


    extracted = (
        extraction[
            "answer"
        ]
    )


    if extracted is None:

        correctness = (
            "NO_ANSWER"
        )


    elif (
        extracted.lower()
        == expected
    ):

        correctness = (
            "CORRECT"
        )


    else:

        correctness = (
            "WRONG"
        )


    return {
        "expected":
            expected,

        "extracted":
            extracted,

        "method":
            extraction[
                "method"
            ],

        "correctness":
            correctness,

        "format":
            format_status,
    }


# =============================================================================
# PARSE EXPERIMENT 016D
# =============================================================================

def parse_016d(text):

    results = defaultdict(
        dict
    )


    current_case = None

    current_prompt = None


    lines = text.splitlines()

    i = 0


    while i < len(lines):

        stripped = (
            lines[i].strip()
        )


        # ---------------------------------------------------------------------
        # Detect case
        # ---------------------------------------------------------------------

        if stripped.startswith(
            "VALIDATION CASE —"
        ):

            possible_case = (
                stripped
                .split(
                    "VALIDATION CASE —",
                    1,
                )[1]
                .strip()
            )


            if (
                possible_case
                in CASE_NAMES
            ):

                current_case = (
                    possible_case
                )


        # ---------------------------------------------------------------------
        # Detect prompt
        # ---------------------------------------------------------------------

        prompt_match = re.match(
            r"^\[(A[1-4]|W[1-3]|F[1-3])\]",
            stripped,
        )


        if (
            prompt_match
            and current_case
            is not None
        ):

            current_prompt = (
                prompt_match.group(1)
            )


        # ---------------------------------------------------------------------
        # Output line
        #
        # Experiment 016D uses repr():
        #
        # Output   : '408'
        #
        # Output   : '432 / 18 = 24'
        #
        # We safely parse the Python string literal with ast.literal_eval.
        # ---------------------------------------------------------------------

        if (
            stripped.startswith(
                "Output   :"
            )
            and current_case
            is not None
            and current_prompt
            is not None
        ):

            literal = (
                stripped
                .split(
                    "Output   :",
                    1,
                )[1]
                .strip()
            )


            import ast


            try:

                output_text = (
                    ast.literal_eval(
                        literal
                    )
                )


            except Exception as exc:

                raise RuntimeError(
                    f"Could not parse output "
                    f"for {current_case} "
                    f"{current_prompt}: "
                    f"{literal}"
                ) from exc


            results[
                current_case
            ][
                current_prompt
            ] = output_text


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


raw_text = INPUT_FILE.read_text(
    encoding="utf-8",
    errors="replace",
)


print(
    "=" * 140
)

print(
    "EXPERIMENT 016E — "
    "FINAL ANSWER-CORRECTNESS + "
    "FORMAT-COMPLIANCE SCORING"
)

print(
    "=" * 140
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

parsed = parse_016d(
    raw_text
)


print(
    "\nPARSE VALIDATION"
)

print(
    "-" * 140
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
        f"{found}/10 responses"
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
            f"{case_name}: expected "
            f"10 responses, found "
            f"{found}. Missing: "
            f"{sorted(missing)}"
        )


# =============================================================================
# OUTPUT
# =============================================================================

output_lines = []


def emit(line=""):

    print(line)

    output_lines.append(
        str(line)
    )


# =============================================================================
# SCORE ALL CASES
# =============================================================================

summaries = {}


emit()
emit(
    "=" * 140
)

emit(
    "FINAL SCORING RESULTS"
)

emit(
    "=" * 140
)


for case_name in CASE_NAMES:

    emit()
    emit(
        "#" * 140
    )

    emit(
        f"CASE — {case_name}"
    )

    emit(
        "#" * 140
    )


    correctness_counts = {
        "CORRECT": 0,
        "WRONG": 0,
        "NO_ANSWER": 0,
    }


    format_counts = {
        "COMPLIANT": 0,
        "NON_COMPLIANT": 0,
    }


    category_stats = defaultdict(
        lambda: {
            "CORRECT": 0,
            "WRONG": 0,
            "NO_ANSWER": 0,
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


        correctness = (
            result[
                "correctness"
            ]
        )


        format_status = (
            result[
                "format"
            ]
        )


        correctness_counts[
            correctness
        ] += 1


        format_counts[
            format_status
        ] += 1


        category = (
            info[
                "category"
            ]
        )


        category_stats[
            category
        ][
            correctness
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
            f"Expected     : "
            f"{result['expected']}"
        )


        emit(
            f"Output       : "
            f"{repr(response)}"
        )


        emit(
            f"Extracted    : "
            f"{result['extracted']}"
        )


        emit(
            f"Method       : "
            f"{result['method']}"
        )


        emit(
            f"Correctness  : "
            f"{correctness}"
        )


        emit(
            f"Format       : "
            f"{format_status}"
        )


    # =========================================================================
    # CASE SUMMARY
    # =========================================================================

    total = len(
        EXPECTED
    )


    answer_accuracy = (
        correctness_counts[
            "CORRECT"
        ]
        / total
        * 100
    )


    format_compliance = (
        format_counts[
            "COMPLIANT"
        ]
        / total
        * 100
    )


    answered = (
        correctness_counts[
            "CORRECT"
        ]
        + correctness_counts[
            "WRONG"
        ]
    )


    answered_accuracy = (
        correctness_counts[
            "CORRECT"
        ]
        / answered
        * 100
        if answered > 0
        else 0.0
    )


    emit()
    emit(
        "-" * 140
    )


    emit(
        "ANSWER CORRECTNESS"
    )


    emit(
        f"CORRECT           : "
        f"{correctness_counts['CORRECT']}"
    )


    emit(
        f"WRONG             : "
        f"{correctness_counts['WRONG']}"
    )


    emit(
        f"NO_ANSWER         : "
        f"{correctness_counts['NO_ANSWER']}"
    )


    emit(
        f"TOTAL             : "
        f"{total}"
    )


    emit(
        f"Answer accuracy   : "
        f"{answer_accuracy:.2f}%"
    )


    emit(
        f"Answered accuracy : "
        f"{answered_accuracy:.2f}%"
    )


    emit()

    emit(
        "FORMAT COMPLIANCE"
    )


    emit(
        f"COMPLIANT         : "
        f"{format_counts['COMPLIANT']}"
    )


    emit(
        f"NON_COMPLIANT     : "
        f"{format_counts['NON_COMPLIANT']}"
    )


    emit(
        f"Compliance rate   : "
        f"{format_compliance:.2f}%"
    )


    emit()

    emit(
        "CATEGORY BREAKDOWN"
    )


    for category in [
        "arithmetic",
        "word_problem",
        "factual",
    ]:

        values = (
            category_stats[
                category
            ]
        )


        emit(
            f"{category:<14} "
            f"CORRECT="
            f"{values['CORRECT']} "
            f"WRONG="
            f"{values['WRONG']} "
            f"NO_ANSWER="
            f"{values['NO_ANSWER']} "
            f"TOTAL="
            f"{values['TOTAL']}"
        )


    summaries[
        case_name
    ] = {
        "correctness":
            correctness_counts,

        "format":
            format_counts,

        "answer_accuracy":
            answer_accuracy,

        "answered_accuracy":
            answered_accuracy,

        "format_compliance":
            format_compliance,

        "prompts":
            prompt_results,
    }


# =============================================================================
# CROSS-CONFIGURATION SUMMARY
# =============================================================================

emit()
emit()
emit(
    "=" * 155
)

emit(
    "CROSS-CONFIGURATION SUMMARY"
)

emit(
    "=" * 155
)


emit(
    f"{'Configuration':<24}"
    f"{'Correct':>10}"
    f"{'Wrong':>10}"
    f"{'NoAns':>10}"
    f"{'Answer Acc':>14}"
    f"{'Answered Acc':>16}"
    f"{'Format Comp':>16}"
)


emit(
    "-" * 155
)


for case_name in CASE_NAMES:

    summary = (
        summaries[
            case_name
        ]
    )


    correctness = (
        summary[
            "correctness"
        ]
    )


    emit(
        f"{case_name:<24}"
        f"{correctness['CORRECT']:>10}"
        f"{correctness['WRONG']:>10}"
        f"{correctness['NO_ANSWER']:>10}"
        f"{summary['answer_accuracy']:>13.2f}%"
        f"{summary['answered_accuracy']:>15.2f}%"
        f"{summary['format_compliance']:>15.2f}%"
    )


# =============================================================================
# PER-PROMPT COMPARISON
# =============================================================================

emit()
emit()
emit(
    "=" * 155
)

emit(
    "PER-PROMPT ANSWER CORRECTNESS"
)

emit(
    "=" * 155
)


emit(
    f"{'Prompt':<10}"
    f"{'Expected':<20}"
    + "".join(
        f"{case_name:>32}"
        for case_name
        in CASE_NAMES
    )
)


emit(
    "-" * 155
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
            summaries[
                case_name
            ][
                "prompts"
            ][
                prompt_id
            ]
        )


        extracted = (
            result[
                "extracted"
            ]
            if result[
                "extracted"
            ]
            is not None
            else "-"
        )


        cell = (
            f"{result['correctness']} "
            f"({extracted})"
        )


        row += (
            f"{cell:>32}"
        )


    emit(
        row
    )


# =============================================================================
# PER-PROMPT FORMAT COMPLIANCE
# =============================================================================

emit()
emit()
emit(
    "=" * 155
)

emit(
    "PER-PROMPT FORMAT COMPLIANCE"
)

emit(
    "=" * 155
)


emit(
    f"{'Prompt':<10}"
    + "".join(
        f"{case_name:>32}"
        for case_name
        in CASE_NAMES
    )
)


emit(
    "-" * 155
)


for prompt_id in EXPECTED:

    row = (
        f"{prompt_id:<10}"
    )


    for case_name in CASE_NAMES:

        format_status = (
            summaries[
                case_name
            ][
                "prompts"
            ][
                prompt_id
            ][
                "format"
            ]
        )


        row += (
            f"{format_status:>32}"
        )


    emit(
        row
    )


# =============================================================================
# RETENTION
# =============================================================================

emit()
emit()
emit(
    "=" * 155
)

emit(
    "ANSWER-CORRECTNESS RETENTION RELATIVE TO FP16"
)

emit(
    "=" * 155
)


fp16_correct = (
    summaries[
        "FP16"
    ][
        "correctness"
    ][
        "CORRECT"
    ]
)


emit(
    f"FP16 correct answers: "
    f"{fp16_correct}/10"
)


for case_name in [
    "BLOCK_GUIDED_7_67",
    "ACT_GUIDED_7_73",
]:

    ternary_correct = (
        summaries[
            case_name
        ][
            "correctness"
        ][
            "CORRECT"
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
            f"= {retention:.2f}%"
        )


    else:

        emit(
            f"{case_name:<24}: "
            "undefined"
        )


# =============================================================================
# FP16 VS TERNARY AGREEMENT
# =============================================================================

emit()
emit()
emit(
    "=" * 155
)

emit(
    "FP16 VS TERNARY CORRECTNESS AGREEMENT"
)

emit(
    "=" * 155
)


for case_name in [
    "BLOCK_GUIDED_7_67",
    "ACT_GUIDED_7_73",
]:

    both_correct = 0

    fp16_only = 0

    ternary_only = 0

    neither = 0


    for prompt_id in EXPECTED:

        fp16_correct_prompt = (
            summaries[
                "FP16"
            ][
                "prompts"
            ][
                prompt_id
            ][
                "correctness"
            ]
            == "CORRECT"
        )


        ternary_correct_prompt = (
            summaries[
                case_name
            ][
                "prompts"
            ][
                prompt_id
            ][
                "correctness"
            ]
            == "CORRECT"
        )


        if (
            fp16_correct_prompt
            and ternary_correct_prompt
        ):

            both_correct += 1


        elif (
            fp16_correct_prompt
            and not ternary_correct_prompt
        ):

            fp16_only += 1


        elif (
            not fp16_correct_prompt
            and ternary_correct_prompt
        ):

            ternary_only += 1


        else:

            neither += 1


    emit()

    emit(
        case_name
    )


    emit(
        f"Both correct       : "
        f"{both_correct}"
    )

    emit(
        f"FP16 only correct  : "
        f"{fp16_only}"
    )

    emit(
        f"Ternary only       : "
        f"{ternary_only}"
    )

    emit(
        f"Neither correct    : "
        f"{neither}"
    )


# =============================================================================
# SCIENTIFIC INTERPRETATION NOTES
# =============================================================================

emit()
emit()
emit(
    "=" * 155
)

emit(
    "SCIENTIFIC INTERPRETATION NOTES"
)

emit(
    "=" * 155
)


emit(
    "1. Experiment 016E does not run "
    "the model or modify any weights."
)

emit(
    "2. It scores only the responses "
    "already generated by Experiment 016D."
)

emit(
    "3. Answer correctness and format "
    "compliance are independent metrics."
)

emit(
    "4. CORRECT means a reliable final "
    "answer was present and matched the "
    "expected answer."
)

emit(
    "5. WRONG means a reliable answer "
    "was present but did not match the "
    "expected answer."
)

emit(
    "6. NO_ANSWER means the generated "
    "text ended before a reliable final "
    "answer could be identified."
)

emit(
    "7. COMPLIANT means the model followed "
    "the requested final-number or "
    "one-word output format."
)

emit(
    "8. NON_COMPLIANT does not necessarily "
    "mean the underlying answer was wrong."
)

emit(
    "9. Answer accuracy is CORRECT / all "
    "10 exact-answer prompts."
)

emit(
    "10. Answered accuracy excludes "
    "NO_ANSWER and therefore must be "
    "reported separately from overall "
    "answer accuracy."
)

emit(
    "11. These exact-answer metrics should "
    "be interpreted together with the "
    "broader repetition and open-ended "
    "generation findings from Experiment 016."
)

emit(
    "12. Tokens/sec from Experiment 016D "
    "must not be interpreted as packed "
    "ternary acceleration because the "
    "ternary matrices were reconstructed "
    "into FP16 tensors."
)


emit()
emit(
    "=" * 155
)

emit(
    "EXPERIMENT 016E COMPLETE"
)

emit(
    "=" * 155
)


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
    f"\nSaved final scoring to: "
    f"{OUTPUT_FILE}"
)
