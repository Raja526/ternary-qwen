import json
import math
from pathlib import Path
from statistics import mean

from scipy.stats import pearsonr, spearmanr


# =============================================================================
# EXPERIMENT 026 — FINAL STATISTICAL CONSOLIDATION
#
# Consolidates frozen Experiments 018–025.
#
# NO new model evaluation.
# NO new ternarization.
# NO new matrix/configuration search.
#
# UNIT CONVENTIONS
# ----------------
#
# Exp019–024 ARC files:
#
#   accuracy              fraction
#   accuracy_difference   fraction
#   prediction_agreement  fraction
#   accuracy_interaction  fraction
#
# Therefore:
#
#   0.55518  -> 55.518%
#  -0.07692  -> -7.692 percentage points
#
# ARC coverage_percent is ALREADY percentage units.
#
#
# Exp025 HellaSwag:
#
#   accuracy                       percentage
#   accuracy_difference            percentage points
#   prediction_agreement_percent   percentage
#
# Examples:
#
#   accuracy = 41.35
#   accuracy_difference = +0.05
#   prediction_agreement_percent = 94.05
#
# IMPORTANT:
# Exp025 accuracy_difference MUST NOT be multiplied by 100.
# =============================================================================


RESULT_DIR = Path("results")


INPUT_FILES = {
    "exp018": RESULT_DIR / "018_arc_challenge_predictions.json",
    "exp019": RESULT_DIR / "019_matrix_arc_ablation.json",
    "exp020": RESULT_DIR / "020_progressive_safe_arc.json",
    "exp021": RESULT_DIR / "021_qproj_interaction_arc.json",
    "exp022": RESULT_DIR / "022_pairwise_compatibility_arc.json",
    "exp023": RESULT_DIR / "023_compatibility_guided_selection.json",
    "exp024": RESULT_DIR / "024_safe_core_interactions_arc.json",
    "exp025": RESULT_DIR / "025_external_hellaswag_2000_validation.json",
}


OUTPUT_JSON = (
    RESULT_DIR
    / "026_final_statistical_consolidation.json"
)

OUTPUT_TEXT = (
    RESULT_DIR
    / "026_final_statistical_consolidation.txt"
)


# =============================================================================
# DESCRIPTIVE THRESHOLDS
#
# These are descriptive flags only.
# They are NOT equivalence margins.
# They are NOT universal definitions of "safe".
# =============================================================================


ACCURACY_PRESERVATION_THRESHOLD_PP = 1.0

BEHAVIOR_AGREEMENT_THRESHOLD_PERCENT = 90.0

LOW_COVERAGE_THRESHOLD_PERCENT = 1.0


# =============================================================================
# BASIC HELPERS
# =============================================================================


def load_json(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


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
            allow_nan=False,
        )


def is_finite_number(value):

    return (
        isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def short_matrix_name(name):

    return name.split(".")[-1]


# =============================================================================
# UNIT NORMALIZATION
# =============================================================================


def arc_proportion_to_percent(value):
    """
    Exp019–024 accuracy/agreement fields are stored
    as fractions.

    Example:
        0.5551839 -> 55.51839%
    """

    return float(value) * 100.0


def arc_delta_to_pp(value):
    """
    Exp019–024 accuracy differences/interactions
    are stored as fractions.

    Example:
        -0.076923 -> -7.6923 percentage points
    """

    return float(value) * 100.0


def hellaswag_percent(value):
    """
    Exp025 accuracy and prediction_agreement_percent
    are already percentage values.

    Example:
        41.35 -> 41.35%
        94.05 -> 94.05%
    """

    return float(value)


def hellaswag_delta_pp(value):
    """
    Exp025 accuracy_difference is ALREADY stored
    in percentage-point units.

    Examples:
        +0.15 -> +0.15 pp
        +0.05 -> +0.05 pp
        -0.80 -> -0.80 pp

    DO NOT multiply this field by 100.
    """

    return float(value)


# =============================================================================
# CORRELATION
# =============================================================================


def calculate_correlation(
    x_values,
    y_values,
):

    pairs = []

    for x, y in zip(
        x_values,
        y_values,
    ):

        if (
            is_finite_number(x)
            and is_finite_number(y)
        ):

            pairs.append(
                (
                    float(x),
                    float(y),
                )
            )

    if len(pairs) < 3:

        return {
            "n": len(pairs),
            "pearson_r": None,
            "pearson_p": None,
            "spearman_rho": None,
            "spearman_p": None,
        }

    x = [
        pair[0]
        for pair in pairs
    ]

    y = [
        pair[1]
        for pair in pairs
    ]

    if (
        len(set(x)) < 2
        or len(set(y)) < 2
    ):

        return {
            "n": len(pairs),
            "pearson_r": None,
            "pearson_p": None,
            "spearman_rho": None,
            "spearman_p": None,
        }

    pearson = pearsonr(
        x,
        y,
    )

    spearman = spearmanr(
        x,
        y,
    )

    return {
        "n":
            len(pairs),

        "pearson_r":
            float(pearson.statistic),

        "pearson_p":
            float(pearson.pvalue),

        "spearman_rho":
            float(spearman.statistic),

        "spearman_p":
            float(spearman.pvalue),
    }


# =============================================================================
# LOAD EXPERIMENTS
# =============================================================================


def load_all_experiments():

    data = {}

    for name, path in INPUT_FILES.items():

        print(
            f"Loading {name}: {path}"
        )

        data[name] = (
            load_json(path)
        )

    return data


# =============================================================================
# VALIDATION
# =============================================================================


def validate_arc_baseline(
    data,
    experiment_name,
):

    baseline = (
        data.get(
            "fp16_baseline"
        )
    )

    if baseline is None:
        return

    if baseline["total"] != 299:

        raise RuntimeError(
            f"{experiment_name}: "
            f"expected ARC total=299, "
            f"found {baseline['total']}"
        )

    if baseline["correct"] != 166:

        raise RuntimeError(
            f"{experiment_name}: "
            f"expected ARC FP16 correct=166, "
            f"found {baseline['correct']}"
        )

    expected_accuracy = (
        100.0
        * baseline["correct"]
        / baseline["total"]
    )

    stored_accuracy = (
        arc_proportion_to_percent(
            baseline[
                "accuracy"
            ]
        )
    )

    if abs(
        stored_accuracy
        - expected_accuracy
    ) > 0.01:

        raise RuntimeError(
            f"{experiment_name}: "
            "ARC FP16 accuracy inconsistent "
            "with correct/total. "
            f"stored={stored_accuracy:.10f}%, "
            f"calculated={expected_accuracy:.10f}%"
        )


def validate_inputs(data):

    print()
    print(
        "Validating frozen experiment inputs..."
    )

    # -------------------------------------------------------------------------
    # Exp018 ARC
    # -------------------------------------------------------------------------

    exp018 = data["exp018"]

    if (
        exp018["fp16"]["correct"] != 166
        or
        exp018["fp16"]["total"] != 299
    ):

        raise RuntimeError(
            "Exp018 ARC FP16 baseline mismatch."
        )

    # -------------------------------------------------------------------------
    # Exp019–024
    # -------------------------------------------------------------------------

    for name in [
        "exp019",
        "exp020",
        "exp021",
        "exp022",
        "exp024",
    ]:

        validate_arc_baseline(
            data[name],
            name,
        )

    # -------------------------------------------------------------------------
    # Exp023
    # -------------------------------------------------------------------------

    exp023_baseline = (
        data[
            "exp023"
        ][
            "fp16_baseline"
        ]
    )

    if (
        exp023_baseline["correct"] != 166
        or
        exp023_baseline["total"] != 299
    ):

        raise RuntimeError(
            "Exp023 ARC baseline mismatch."
        )

    # -------------------------------------------------------------------------
    # Exp025
    # -------------------------------------------------------------------------

    exp025 = data["exp025"]

    metadata = (
        exp025[
            "_metadata"
        ]
    )

    if metadata["subset_size"] != 2000:

        raise RuntimeError(
            "Exp025 subset size mismatch."
        )

    if (
        exp025[
            "FP16"
        ][
            "total"
        ]
        != 2000
    ):

        raise RuntimeError(
            "Exp025 FP16 total mismatch."
        )

    if (
        exp025[
            "FP16"
        ][
            "correct"
        ]
        != 826
    ):

        raise RuntimeError(
            "Exp025 FP16 correct mismatch."
        )

    # -------------------------------------------------------------------------
    # Critical Exp025 unit sanity checks
    # -------------------------------------------------------------------------

    expected_hellaswag = {
        "K_ONLY": {
            "correct": 829,
            "accuracy": 41.45,
            "delta": 0.15,
            "agreement": 98.65,
            "changed": 27,
        },

        "SAFE_CORE": {
            "correct": 827,
            "accuracy": 41.35,
            "delta": 0.05,
            "agreement": 94.05,
            "changed": 119,
        },

        "CORE_PLUS_Q": {
            "correct": 810,
            "accuracy": 40.50,
            "delta": -0.80,
            "agreement": 92.00,
            "changed": 160,
        },
    }

    for name, expected in (
        expected_hellaswag.items()
    ):

        row = exp025[name]

        if row["correct"] != expected["correct"]:

            raise RuntimeError(
                f"Exp025 {name}: correct mismatch."
            )

        if row["changed_predictions"] != expected["changed"]:

            raise RuntimeError(
                f"Exp025 {name}: "
                "changed_predictions mismatch."
            )

        if abs(
            hellaswag_percent(
                row["accuracy"]
            )
            - expected["accuracy"]
        ) > 0.01:

            raise RuntimeError(
                f"Exp025 {name}: "
                "accuracy unit/value mismatch."
            )

        if abs(
            hellaswag_delta_pp(
                row[
                    "accuracy_difference"
                ]
            )
            - expected["delta"]
        ) > 0.01:

            raise RuntimeError(
                f"Exp025 {name}: "
                "accuracy_difference unit/value mismatch. "
                f"Found="
                f"{row['accuracy_difference']}"
            )

        if abs(
            hellaswag_percent(
                row[
                    "prediction_agreement_percent"
                ]
            )
            - expected["agreement"]
        ) > 0.01:

            raise RuntimeError(
                f"Exp025 {name}: "
                "prediction agreement mismatch."
            )

    print(
        "INPUT VALIDATION: PASS"
    )


# =============================================================================
# NORMALIZED ARC ROW
# =============================================================================


def make_arc_row(
    experiment,
    category,
    label,
    row,
    matrix_names=None,
    matrix_count=None,
):

    if matrix_names is None:

        matrix_names = (
            row.get(
                "matrix_names",
                [],
            )
        )

    if matrix_count is None:

        matrix_count = (
            row.get(
                "matrix_count",
                len(matrix_names),
            )
        )

    return {
        "experiment":
            experiment,

        "category":
            category,

        "label":
            label,

        "matrix_names":
            matrix_names,

        "matrix_count":
            matrix_count,

        # Already percentage units.
        "coverage_percent":
            float(
                row[
                    "coverage_percent"
                ]
            ),

        # Numerical reconstruction metrics.
        "mean_error":
            float(
                row[
                    "mean_error"
                ]
            ),

        "max_error":
            float(
                row[
                    "max_error"
                ]
            ),

        # Fraction -> percentage.
        "accuracy_percent":
            arc_proportion_to_percent(
                row[
                    "accuracy"
                ]
            ),

        # Fraction -> percentage points.
        "accuracy_difference_pp":
            arc_delta_to_pp(
                row[
                    "accuracy_difference"
                ]
            ),

        # Fraction -> percentage.
        "prediction_agreement_percent":
            arc_proportion_to_percent(
                row[
                    "prediction_agreement"
                ]
            ),

        "changed_predictions":
            int(
                row[
                    "changed_predictions"
                ]
            ),

        "mcnemar_exact_p":
            float(
                row[
                    "mcnemar_exact_p"
                ]
            ),
    }


# =============================================================================
# CONSOLIDATED ARC OBSERVATIONS
# =============================================================================


def build_arc_observations(data):

    observations = []

    # -------------------------------------------------------------------------
    # Exp019 — individual matrices
    # -------------------------------------------------------------------------

    for row in (
        data[
            "exp019"
        ][
            "results"
        ]
    ):

        observations.append(
            make_arc_row(
                experiment=19,
                category="individual_matrix",
                label=short_matrix_name(
                    row[
                        "matrix_name"
                    ]
                ),
                row=row,
                matrix_names=[
                    row[
                        "matrix_name"
                    ]
                ],
                matrix_count=1,
            )
        )

    # -------------------------------------------------------------------------
    # Exp020 — progressive combinations
    # -------------------------------------------------------------------------

    for row in (
        data[
            "exp020"
        ][
            "results"
        ]
    ):

        observations.append(
            make_arc_row(
                experiment=20,
                category="progressive_combination",
                label=(
                    f"step_{row['step']}"
                ),
                row=row,
            )
        )

    # -------------------------------------------------------------------------
    # Exp021 — q_proj interaction experiments
    # -------------------------------------------------------------------------

    for row in (
        data[
            "exp021"
        ][
            "results"
        ]
    ):

        observations.append(
            make_arc_row(
                experiment=21,
                category="qproj_interaction",
                label=row[
                    "condition_name"
                ],
                row=row,
            )
        )

    # -------------------------------------------------------------------------
    # Exp022 — pairwise combinations
    # -------------------------------------------------------------------------

    for row in (
        data[
            "exp022"
        ][
            "results"
        ]
    ):

        result = make_arc_row(
            experiment=22,
            category="pairwise_combination",
            label=(
                f"{row['a_short']}"
                f"+"
                f"{row['b_short']}"
            ),
            row=row,
            matrix_count=2,
        )

        result[
            "accuracy_interaction_pp"
        ] = (
            arc_delta_to_pp(
                row[
                    "accuracy_interaction"
                ]
            )
        )

        result[
            "individual_delta_a_pp"
        ] = (
            arc_delta_to_pp(
                row[
                    "individual_delta_a"
                ]
            )
        )

        result[
            "individual_delta_b_pp"
        ] = (
            arc_delta_to_pp(
                row[
                    "individual_delta_b"
                ]
            )
        )

        result[
            "expected_additive_delta_pp"
        ] = (
            arc_delta_to_pp(
                row[
                    "expected_additive_delta"
                ]
            )
        )

        observations.append(
            result
        )

    # -------------------------------------------------------------------------
    # Exp024 — SAFE_CORE + one
    # -------------------------------------------------------------------------

    for row in (
        data[
            "exp024"
        ][
            "results"
        ]
    ):

        observations.append(
            make_arc_row(
                experiment=24,
                category="safe_core_interaction",
                label=row[
                    "condition_name"
                ],
                row=row,
            )
        )

    return observations


# =============================================================================
# RQ1 — COVERAGE
# =============================================================================


def analyze_rq1(
    observations,
):

    coverage = [
        row[
            "coverage_percent"
        ]
        for row in observations
    ]

    abs_delta = [
        abs(
            row[
                "accuracy_difference_pp"
            ]
        )
        for row in observations
    ]

    disagreement = [
        100.0
        - row[
            "prediction_agreement_percent"
        ]
        for row in observations
    ]

    low_coverage = [
        row
        for row in observations
        if (
            row[
                "coverage_percent"
            ]
            <=
            LOW_COVERAGE_THRESHOLD_PERCENT
        )
    ]

    nontrivial = [
        row
        for row in low_coverage
        if (
            abs(
                row[
                    "accuracy_difference_pp"
                ]
            )
            >
            ACCURACY_PRESERVATION_THRESHOLD_PP
            or
            row[
                "prediction_agreement_percent"
            ]
            <
            BEHAVIOR_AGREEMENT_THRESHOLD_PERCENT
        )
    ]

    strongest = None

    if low_coverage:

        strongest = max(
            low_coverage,
            key=lambda row:
                abs(
                    row[
                        "accuracy_difference_pp"
                    ]
                ),
        )

    return {
        "question":
            (
                "Does low ternarized parameter coverage "
                "guarantee functional preservation?"
            ),

        "low_coverage_threshold_percent":
            LOW_COVERAGE_THRESHOLD_PERCENT,

        "observations_total":
            len(observations),

        "low_coverage_observations":
            len(low_coverage),

        "low_coverage_with_nontrivial_effect":
            len(nontrivial),

        "coverage_vs_absolute_accuracy_change":
            calculate_correlation(
                coverage,
                abs_delta,
            ),

        "coverage_vs_prediction_disagreement":
            calculate_correlation(
                coverage,
                disagreement,
            ),

        "strongest_low_coverage_accuracy_effect":
            strongest,

        "descriptive_conclusion":
            (
                "Low parameter coverage does not "
                "guarantee functional preservation."
            ),
    }


# =============================================================================
# RQ2 — INTERACTIONS
# =============================================================================


def analyze_rq2(data):

    exp022 = data["exp022"]
    exp023 = data["exp023"]
    exp024 = data["exp024"]

    pairs = (
        exp022[
            "results"
        ]
    )

    strongest_negative = min(
        pairs,
        key=lambda row:
            row[
                "accuracy_interaction"
            ],
    )

    strongest_positive = max(
        pairs,
        key=lambda row:
            row[
                "accuracy_interaction"
            ],
    )

    higher_order = []

    for row in (
        exp024[
            "results"
        ]
    ):

        higher_order.append({
            "added_matrix":
                row[
                    "added_short"
                ],

            "accuracy_difference_pp":
                arc_delta_to_pp(
                    row[
                        "accuracy_difference"
                    ]
                ),

            "prediction_agreement_percent":
                arc_proportion_to_percent(
                    row[
                        "prediction_agreement"
                    ]
                ),

            "changed_predictions":
                int(
                    row[
                        "changed_predictions"
                    ]
                ),

            "mcnemar_exact_p":
                float(
                    row[
                        "mcnemar_exact_p"
                    ]
                ),
        })

    return {
        "question":
            (
                "Does individual or pairwise "
                "compatibility guarantee larger-"
                "combination preservation?"
            ),

        "strongest_negative_pairwise_interaction":
            {
                "pair":
                    (
                        f"{strongest_negative['a_short']}"
                        "+"
                        f"{strongest_negative['b_short']}"
                    ),

                "individual_delta_a_pp":
                    arc_delta_to_pp(
                        strongest_negative[
                            "individual_delta_a"
                        ]
                    ),

                "individual_delta_b_pp":
                    arc_delta_to_pp(
                        strongest_negative[
                            "individual_delta_b"
                        ]
                    ),

                "expected_additive_delta_pp":
                    arc_delta_to_pp(
                        strongest_negative[
                            "expected_additive_delta"
                        ]
                    ),

                "observed_delta_pp":
                    arc_delta_to_pp(
                        strongest_negative[
                            "accuracy_difference"
                        ]
                    ),

                "accuracy_interaction_pp":
                    arc_delta_to_pp(
                        strongest_negative[
                            "accuracy_interaction"
                        ]
                    ),

                "prediction_agreement_percent":
                    arc_proportion_to_percent(
                        strongest_negative[
                            "prediction_agreement"
                        ]
                    ),

                "mcnemar_exact_p":
                    float(
                        strongest_negative[
                            "mcnemar_exact_p"
                        ]
                    ),
            },

        "strongest_positive_pairwise_interaction":
            {
                "pair":
                    (
                        f"{strongest_positive['a_short']}"
                        "+"
                        f"{strongest_positive['b_short']}"
                    ),

                "observed_delta_pp":
                    arc_delta_to_pp(
                        strongest_positive[
                            "accuracy_difference"
                        ]
                    ),

                "accuracy_interaction_pp":
                    arc_delta_to_pp(
                        strongest_positive[
                            "accuracy_interaction"
                        ]
                    ),
            },

        "compatible_edges":
            exp023[
                "compatible_edges"
            ],

        "maximum_clique_size":
            exp023[
                "maximum_clique_size"
            ],

        "maximum_cliques":
            exp023[
                "maximum_cliques"
            ],

        "retrospective_exp20":
            exp023[
                "retrospective_exp20"
            ],

        "safe_core_plus_one":
            higher_order,

        "descriptive_conclusion":
            (
                "Individual and pairwise compatibility "
                "does not guarantee preservation in "
                "larger combinations."
            ),
    }


# =============================================================================
# RQ3 — ACCURACY VS BEHAVIOR
# =============================================================================


def analyze_rq3(
    observations,
    data,
):

    arc_cases = []

    for row in observations:

        if (
            abs(
                row[
                    "accuracy_difference_pp"
                ]
            )
            <=
            ACCURACY_PRESERVATION_THRESHOLD_PP
            and
            row[
                "prediction_agreement_percent"
            ]
            <
            BEHAVIOR_AGREEMENT_THRESHOLD_PERCENT
        ):

            arc_cases.append(
                row
            )

    exp025 = data["exp025"]

    hellaswag_cases = []

    for name in [
        "K_ONLY",
        "SAFE_CORE",
        "CORE_PLUS_Q",
    ]:

        row = exp025[name]

        delta_pp = (
            hellaswag_delta_pp(
                row[
                    "accuracy_difference"
                ]
            )
        )

        agreement = (
            hellaswag_percent(
                row[
                    "prediction_agreement_percent"
                ]
            )
        )

        if (
            abs(delta_pp)
            <=
            ACCURACY_PRESERVATION_THRESHOLD_PP
        ):

            hellaswag_cases.append({
                "configuration":
                    name,

                "accuracy_difference_pp":
                    delta_pp,

                "prediction_agreement_percent":
                    agreement,

                "changed_predictions":
                    int(
                        row[
                            "changed_predictions"
                        ]
                    ),

                "fp16_only":
                    int(
                        row[
                            "fp16_only"
                        ]
                    ),

                "ternary_only":
                    int(
                        row[
                            "ternary_only"
                        ]
                    ),

                "mcnemar_exact_p":
                    float(
                        row[
                            "mcnemar_exact_p"
                        ]
                    ),
            })

    return {
        "question":
            (
                "Does aggregate accuracy preservation "
                "imply prediction-level behavioral "
                "preservation?"
            ),

        "accuracy_preservation_threshold_pp":
            ACCURACY_PRESERVATION_THRESHOLD_PP,

        "behavior_agreement_threshold_percent":
            BEHAVIOR_AGREEMENT_THRESHOLD_PERCENT,

        "arc_accuracy_preserving_behavior_changing_cases":
            arc_cases,

        "hellaswag_accuracy_preserving_cases":
            hellaswag_cases,

        "descriptive_conclusion":
            (
                "Similar aggregate accuracy can coexist "
                "with substantial prediction-level "
                "changes."
            ),
    }


# =============================================================================
# RQ4 — CROSS-BENCHMARK GENERALIZATION
# =============================================================================


def analyze_rq4(data):

    exp020 = data["exp020"]
    exp024 = data["exp024"]
    exp025 = data["exp025"]

    # -------------------------------------------------------------------------
    # SAFE_CORE
    # -------------------------------------------------------------------------

    expected_core = {
        "model.layers.3.self_attn.k_proj",
        "model.layers.4.linear_attn.out_proj",
        "model.layers.4.linear_attn.in_proj_qkv",
    }

    safe_core_arc = None

    for row in (
        exp020[
            "results"
        ]
    ):

        if (
            set(
                row[
                    "matrix_names"
                ]
            )
            == expected_core
        ):

            safe_core_arc = row
            break

    if safe_core_arc is None:

        raise RuntimeError(
            "ARC SAFE_CORE not found."
        )

    # -------------------------------------------------------------------------
    # CORE_PLUS_Q
    # -------------------------------------------------------------------------

    core_plus_q_arc = None

    for row in (
        exp024[
            "results"
        ]
    ):

        if (
            row[
                "added_short"
            ]
            == "q_proj"
        ):

            core_plus_q_arc = row
            break

    if core_plus_q_arc is None:

        raise RuntimeError(
            "ARC CORE_PLUS_Q not found."
        )

    # -------------------------------------------------------------------------
    # HellaSwag
    # -------------------------------------------------------------------------

    safe_core_hs = (
        exp025[
            "SAFE_CORE"
        ]
    )

    core_plus_q_hs = (
        exp025[
            "CORE_PLUS_Q"
        ]
    )

    k_only_hs = (
        exp025[
            "K_ONLY"
        ]
    )

    # -------------------------------------------------------------------------
    # Accuracy
    # -------------------------------------------------------------------------

    safe_core_arc_accuracy = (
        arc_proportion_to_percent(
            safe_core_arc[
                "accuracy"
            ]
        )
    )

    core_plus_q_arc_accuracy = (
        arc_proportion_to_percent(
            core_plus_q_arc[
                "accuracy"
            ]
        )
    )

    safe_core_hs_accuracy = (
        hellaswag_percent(
            safe_core_hs[
                "accuracy"
            ]
        )
    )

    core_plus_q_hs_accuracy = (
        hellaswag_percent(
            core_plus_q_hs[
                "accuracy"
            ]
        )
    )

    # -------------------------------------------------------------------------
    # Adding q_proj relative to SAFE_CORE
    # -------------------------------------------------------------------------

    arc_q_increment = (
        core_plus_q_arc_accuracy
        - safe_core_arc_accuracy
    )

    hs_q_increment = (
        core_plus_q_hs_accuracy
        - safe_core_hs_accuracy
    )

    return {
        "question":
            (
                "Do ternarization sensitivity effects "
                "generalize across ARC and HellaSwag?"
            ),

        "SAFE_CORE": {
            "ARC": {
                "questions":
                    int(
                        safe_core_arc[
                            "total"
                        ]
                    ),

                "accuracy_percent":
                    safe_core_arc_accuracy,

                "accuracy_difference_pp":
                    arc_delta_to_pp(
                        safe_core_arc[
                            "accuracy_difference"
                        ]
                    ),

                "prediction_agreement_percent":
                    arc_proportion_to_percent(
                        safe_core_arc[
                            "prediction_agreement"
                        ]
                    ),

                "changed_predictions":
                    int(
                        safe_core_arc[
                            "changed_predictions"
                        ]
                    ),

                "mcnemar_exact_p":
                    float(
                        safe_core_arc[
                            "mcnemar_exact_p"
                        ]
                    ),
            },

            "HellaSwag": {
                "questions":
                    int(
                        safe_core_hs[
                            "total"
                        ]
                    ),

                "accuracy_percent":
                    safe_core_hs_accuracy,

                "accuracy_difference_pp":
                    hellaswag_delta_pp(
                        safe_core_hs[
                            "accuracy_difference"
                        ]
                    ),

                "prediction_agreement_percent":
                    hellaswag_percent(
                        safe_core_hs[
                            "prediction_agreement_percent"
                        ]
                    ),

                "changed_predictions":
                    int(
                        safe_core_hs[
                            "changed_predictions"
                        ]
                    ),

                "mcnemar_exact_p":
                    float(
                        safe_core_hs[
                            "mcnemar_exact_p"
                        ]
                    ),
            },
        },

        "CORE_PLUS_Q": {
            "ARC": {
                "questions":
                    int(
                        core_plus_q_arc[
                            "total"
                        ]
                    ),

                "accuracy_percent":
                    core_plus_q_arc_accuracy,

                "accuracy_difference_pp":
                    arc_delta_to_pp(
                        core_plus_q_arc[
                            "accuracy_difference"
                        ]
                    ),

                "prediction_agreement_percent":
                    arc_proportion_to_percent(
                        core_plus_q_arc[
                            "prediction_agreement"
                        ]
                    ),

                "changed_predictions":
                    int(
                        core_plus_q_arc[
                            "changed_predictions"
                        ]
                    ),

                "mcnemar_exact_p":
                    float(
                        core_plus_q_arc[
                            "mcnemar_exact_p"
                        ]
                    ),
            },

            "HellaSwag": {
                "questions":
                    int(
                        core_plus_q_hs[
                            "total"
                        ]
                    ),

                "accuracy_percent":
                    core_plus_q_hs_accuracy,

                "accuracy_difference_pp":
                    hellaswag_delta_pp(
                        core_plus_q_hs[
                            "accuracy_difference"
                        ]
                    ),

                "prediction_agreement_percent":
                    hellaswag_percent(
                        core_plus_q_hs[
                            "prediction_agreement_percent"
                        ]
                    ),

                "changed_predictions":
                    int(
                        core_plus_q_hs[
                            "changed_predictions"
                        ]
                    ),

                "mcnemar_exact_p":
                    float(
                        core_plus_q_hs[
                            "mcnemar_exact_p"
                        ]
                    ),
            },
        },

        "HELLASWAG_K_ONLY": {
            "accuracy_percent":
                hellaswag_percent(
                    k_only_hs[
                        "accuracy"
                    ]
                ),

            "accuracy_difference_pp":
                hellaswag_delta_pp(
                    k_only_hs[
                        "accuracy_difference"
                    ]
                ),

            "prediction_agreement_percent":
                hellaswag_percent(
                    k_only_hs[
                        "prediction_agreement_percent"
                    ]
                ),

            "changed_predictions":
                int(
                    k_only_hs[
                        "changed_predictions"
                    ]
                ),

            "mcnemar_exact_p":
                float(
                    k_only_hs[
                        "mcnemar_exact_p"
                    ]
                ),
        },

        "qproj_increment_relative_to_safe_core_pp": {
            "ARC":
                arc_q_increment,

            "HellaSwag":
                hs_q_increment,
        },

        "protocol_note":
            (
                "ARC and HellaSwag use different frozen "
                "evaluation protocols. Their raw accuracies "
                "should not be compared as if they measured "
                "the same task."
            ),

        "descriptive_conclusion":
            (
                "Adding q_proj to SAFE_CORE shifts accuracy "
                "downward on both benchmarks, but the "
                "magnitude differs substantially. Functional "
                "sensitivity is therefore task dependent."
            ),
    }


# =============================================================================
# RECONSTRUCTION ERROR ANALYSIS
# =============================================================================


def analyze_reconstruction_error(
    observations,
):

    rows = [
        row
        for row in observations
        if is_finite_number(
            row.get(
                "mean_error"
            )
        )
    ]

    errors = [
        row[
            "mean_error"
        ]
        for row in rows
    ]

    abs_delta = [
        abs(
            row[
                "accuracy_difference_pp"
            ]
        )
        for row in rows
    ]

    disagreement = [
        100.0
        - row[
            "prediction_agreement_percent"
        ]
        for row in rows
    ]

    return {
        "observations":
            len(rows),

        "mean_reconstruction_error":
            (
                mean(errors)
                if errors
                else None
            ),

        "error_vs_absolute_accuracy_change":
            calculate_correlation(
                errors,
                abs_delta,
            ),

        "error_vs_prediction_disagreement":
            calculate_correlation(
                errors,
                disagreement,
            ),

        "interpretation":
            (
                "These correlations are descriptive only. "
                "The configurations overlap and are not "
                "statistically independent."
            ),
    }


# =============================================================================
# FINAL EVIDENCE
# =============================================================================


def build_final_evidence():

    return [
        {
            "id": "E1",

            "statement":
                (
                    "Parameter coverage alone is not a "
                    "sufficient predictor of functional "
                    "preservation."
                ),
        },

        {
            "id": "E2",

            "statement":
                (
                    "Matrix sensitivity is location- and "
                    "interaction-dependent."
                ),
        },

        {
            "id": "E3",

            "statement":
                (
                    "Individual or pairwise compatibility "
                    "does not guarantee compatibility in "
                    "larger combinations."
                ),
        },

        {
            "id": "E4",

            "statement":
                (
                    "Aggregate accuracy preservation does "
                    "not imply prediction-level behavioral "
                    "preservation."
                ),
        },

        {
            "id": "E5",

            "statement":
                (
                    "Functional sensitivity is benchmark "
                    "dependent."
                ),
        },

        {
            "id": "E6",

            "statement":
                (
                    "The current experiments demonstrate "
                    "numerical ternarization behavior, not "
                    "deployment-level compression or "
                    "inference acceleration."
                ),
        },
    ]


# =============================================================================
# REPORT
# =============================================================================


def build_report(
    rq1,
    rq2,
    rq3,
    rq4,
    reconstruction,
    evidence,
):

    lines = []

    def add(text=""):
        lines.append(text)

    # -------------------------------------------------------------------------
    # Header
    # -------------------------------------------------------------------------

    add("=" * 110)

    add(
        "EXPERIMENT 026 — "
        "FINAL STATISTICAL CONSOLIDATION"
    )

    add("=" * 110)

    add()

    add(
        "Frozen experiments : 018–025"
    )

    add(
        "Model              : Qwen/Qwen3.5-0.8B"
    )

    add(
        "New model runs     : NO"
    )

    add(
        "New ternarization  : NO"
    )

    add(
        "New search         : NO"
    )

    # -------------------------------------------------------------------------
    # RQ1
    # -------------------------------------------------------------------------

    add()
    add()

    add(
        "RQ1 — PARAMETER COVERAGE"
    )

    add("-" * 110)

    add(
        rq1[
            "question"
        ]
    )

    add()

    add(
        f"ARC observations analyzed : "
        f"{rq1['observations_total']}"
    )

    add(
        f"Low coverage threshold    : "
        f"<= "
        f"{rq1['low_coverage_threshold_percent']:.2f}%"
    )

    add(
        f"Low coverage observations : "
        f"{rq1['low_coverage_observations']}"
    )

    add(
        "Low coverage with nontrivial effect: "
        f"{rq1['low_coverage_with_nontrivial_effect']}"
    )

    strongest = (
        rq1[
            "strongest_low_coverage_accuracy_effect"
        ]
    )

    if strongest:

        add()

        add(
            "Strongest absolute accuracy effect "
            "at <=1% coverage:"
        )

        add(
            f"  Exp        : "
            f"{strongest['experiment']}"
        )

        add(
            f"  Condition  : "
            f"{strongest['label']}"
        )

        add(
            f"  Coverage   : "
            f"{strongest['coverage_percent']:.4f}%"
        )

        add(
            f"  Delta      : "
            f"{strongest['accuracy_difference_pp']:+.2f} pp"
        )

        add(
            f"  Agreement  : "
            f"{strongest['prediction_agreement_percent']:.2f}%"
        )

    add()

    add(
        "Conclusion: "
        + rq1[
            "descriptive_conclusion"
        ]
    )

    # -------------------------------------------------------------------------
    # RQ2
    # -------------------------------------------------------------------------

    add()
    add()

    add(
        "RQ2 — INTERACTION / COMBINATION SAFETY"
    )

    add("-" * 110)

    add(
        rq2[
            "question"
        ]
    )

    add()

    negative = (
        rq2[
            "strongest_negative_pairwise_interaction"
        ]
    )

    add(
        "Strongest negative descriptive "
        "pairwise interaction:"
    )

    add(
        f"  Pair                    : "
        f"{negative['pair']}"
    )

    add(
        f"  Individual delta A      : "
        f"{negative['individual_delta_a_pp']:+.2f} pp"
    )

    add(
        f"  Individual delta B      : "
        f"{negative['individual_delta_b_pp']:+.2f} pp"
    )

    add(
        f"  Expected additive delta : "
        f"{negative['expected_additive_delta_pp']:+.2f} pp"
    )

    add(
        f"  Observed pair delta     : "
        f"{negative['observed_delta_pp']:+.2f} pp"
    )

    add(
        f"  Interaction deviation   : "
        f"{negative['accuracy_interaction_pp']:+.2f} pp"
    )

    add(
        f"  Agreement               : "
        f"{negative['prediction_agreement_percent']:.2f}%"
    )

    add()

    add(
        f"Compatible pairwise edges : "
        f"{len(rq2['compatible_edges'])}"
    )

    add(
        f"Maximum clique size       : "
        f"{rq2['maximum_clique_size']}"
    )

    add(
        f"Maximum clique(s)         : "
        f"{rq2['maximum_cliques']}"
    )

    add()

    add(
        "SAFE_CORE + one matrix:"
    )

    for row in (
        rq2[
            "safe_core_plus_one"
        ]
    ):

        add(
            f"  {row['added_matrix']:<12} "
            f"delta="
            f"{row['accuracy_difference_pp']:+6.2f} pp  "
            f"agreement="
            f"{row['prediction_agreement_percent']:6.2f}%  "
            f"changed="
            f"{row['changed_predictions']:3d}  "
            f"p="
            f"{row['mcnemar_exact_p']:.6g}"
        )

    add()

    add(
        "Conclusion: "
        + rq2[
            "descriptive_conclusion"
        ]
    )

    # -------------------------------------------------------------------------
    # RQ3
    # -------------------------------------------------------------------------

    add()
    add()

    add(
        "RQ3 — ACCURACY VS BEHAVIOR"
    )

    add("-" * 110)

    add(
        rq3[
            "question"
        ]
    )

    add()

    add(
        "Accuracy-preservation threshold: "
        f"|delta| <= "
        f"{rq3['accuracy_preservation_threshold_pp']:.2f} pp"
    )

    add(
        "Behavior-change flag: "
        f"agreement < "
        f"{rq3['behavior_agreement_threshold_percent']:.2f}%"
    )

    add()

    add(
        "ARC accuracy-preserving but "
        "behavior-changing cases:"
    )

    arc_cases = (
        rq3[
            "arc_accuracy_preserving_behavior_changing_cases"
        ]
    )

    if not arc_cases:

        add(
            "  None."
        )

    else:

        for row in arc_cases:

            add(
                f"  Exp{row['experiment']:03d} "
                f"{row['label']:<25} "
                f"delta="
                f"{row['accuracy_difference_pp']:+6.2f} pp  "
                f"agreement="
                f"{row['prediction_agreement_percent']:6.2f}%  "
                f"changed="
                f"{row['changed_predictions']:3d}"
            )

    add()

    add(
        "HellaSwag configurations with "
        "|delta| <= 1 pp:"
    )

    hs_cases = (
        rq3[
            "hellaswag_accuracy_preserving_cases"
        ]
    )

    if not hs_cases:

        add(
            "  None."
        )

    else:

        for row in hs_cases:

            add(
                f"  {row['configuration']:<14} "
                f"delta="
                f"{row['accuracy_difference_pp']:+6.2f} pp  "
                f"agreement="
                f"{row['prediction_agreement_percent']:6.2f}%  "
                f"changed="
                f"{row['changed_predictions']:4d}  "
                f"p="
                f"{row['mcnemar_exact_p']:.6g}"
            )

    add()

    add(
        "Conclusion: "
        + rq3[
            "descriptive_conclusion"
        ]
    )

    # -------------------------------------------------------------------------
    # RQ4
    # -------------------------------------------------------------------------

    add()
    add()

    add(
        "RQ4 — CROSS-BENCHMARK GENERALIZATION"
    )

    add("-" * 110)

    add(
        rq4[
            "question"
        ]
    )

    for config in [
        "SAFE_CORE",
        "CORE_PLUS_Q",
    ]:

        add()
        add(config)

        for benchmark in [
            "ARC",
            "HellaSwag",
        ]:

            row = (
                rq4[
                    config
                ][
                    benchmark
                ]
            )

            add(
                f"  {benchmark:<10} "
                f"accuracy="
                f"{row['accuracy_percent']:6.2f}%  "
                f"delta="
                f"{row['accuracy_difference_pp']:+6.2f} pp  "
                f"agreement="
                f"{row['prediction_agreement_percent']:6.2f}%  "
                f"changed="
                f"{row['changed_predictions']:4d}  "
                f"p="
                f"{row['mcnemar_exact_p']:.6g}"
            )

    add()

    increment = (
        rq4[
            "qproj_increment_relative_to_safe_core_pp"
        ]
    )

    add(
        "Observed effect of adding q_proj "
        "to SAFE_CORE:"
    )

    add(
        f"  ARC       : "
        f"{increment['ARC']:+.2f} pp"
    )

    add(
        f"  HellaSwag : "
        f"{increment['HellaSwag']:+.2f} pp"
    )

    add()

    add(
        "Protocol note: "
        + rq4[
            "protocol_note"
        ]
    )

    add()

    add(
        "Conclusion: "
        + rq4[
            "descriptive_conclusion"
        ]
    )

    # -------------------------------------------------------------------------
    # Reconstruction error
    # -------------------------------------------------------------------------

    add()
    add()

    add(
        "RECONSTRUCTION ERROR — "
        "DESCRIPTIVE ASSOCIATIONS"
    )

    add("-" * 110)

    add(
        f"Observations: "
        f"{reconstruction['observations']}"
    )

    correlation_accuracy = (
        reconstruction[
            "error_vs_absolute_accuracy_change"
        ]
    )

    correlation_behavior = (
        reconstruction[
            "error_vs_prediction_disagreement"
        ]
    )

    add()

    add(
        "Mean reconstruction error vs "
        "|accuracy delta|:"
    )

    add(
        f"  Pearson r    = "
        f"{correlation_accuracy['pearson_r']}"
    )

    add(
        f"  Pearson p    = "
        f"{correlation_accuracy['pearson_p']}"
    )

    add(
        f"  Spearman rho = "
        f"{correlation_accuracy['spearman_rho']}"
    )

    add(
        f"  Spearman p   = "
        f"{correlation_accuracy['spearman_p']}"
    )

    add()

    add(
        "Mean reconstruction error vs "
        "prediction disagreement:"
    )

    add(
        f"  Pearson r    = "
        f"{correlation_behavior['pearson_r']}"
    )

    add(
        f"  Pearson p    = "
        f"{correlation_behavior['pearson_p']}"
    )

    add(
        f"  Spearman rho = "
        f"{correlation_behavior['spearman_rho']}"
    )

    add(
        f"  Spearman p   = "
        f"{correlation_behavior['spearman_p']}"
    )

    add()

    add(
        reconstruction[
            "interpretation"
        ]
    )

    # -------------------------------------------------------------------------
    # Evidence
    # -------------------------------------------------------------------------

    add()
    add()

    add(
        "FINAL EVIDENCE-SUPPORTED CONCLUSIONS"
    )

    add("-" * 110)

    for item in evidence:

        add(
            f"{item['id']}. "
            f"{item['statement']}"
        )

    # -------------------------------------------------------------------------
    # Limitations
    # -------------------------------------------------------------------------

    add()
    add()

    add(
        "LIMITATIONS"
    )

    add("-" * 110)

    limitations = [
        (
            "Results are specific to "
            "Qwen/Qwen3.5-0.8B and the tested matrices."
        ),

        (
            "The frozen quantizer uses optimized-scale "
            "groupwise ternarization, group size 128, "
            "and threshold multiplier 0.5."
        ),

        (
            "ARC-Challenge uses 299 validation questions."
        ),

        (
            "HellaSwag Exp025 uses the deterministic first "
            "2,000 validation examples rather than all "
            "10,042 validation examples."
        ),

        (
            "HellaSwag conclusions apply under the frozen "
            "length-normalized conditional continuation "
            "log-likelihood protocol used in Exp025."
        ),

        (
            "ARC and HellaSwag use different frozen "
            "evaluation protocols."
        ),

        (
            "McNemar p >= 0.05 does not establish "
            "statistical equivalence or prove preservation."
        ),

        (
            "Positive accuracy deltas do not establish "
            "that ternarization improves the model."
        ),

        (
            "The pairwise accuracy-interaction metric is "
            "a descriptive deviation from additive accuracy "
            "deltas, not a formal statistical interaction."
        ),

        (
            "Correlation analyses combine overlapping and "
            "heterogeneous configurations and therefore "
            "should be interpreted descriptively."
        ),

        (
            "No universal safe ternarization coverage "
            "threshold was established."
        ),

        (
            "Ternary values are reconstructed into FP16; "
            "packed ternary storage, custom kernels, actual "
            "VRAM compression, and inference acceleration "
            "were not evaluated."
        ),
    ]

    for index, item in enumerate(
        limitations,
        start=1,
    ):

        add(
            f"{index}. {item}"
        )

    add()
    add("=" * 110)

    add(
        "END OF EXPERIMENT 026"
    )

    add("=" * 110)

    return "\n".join(
        lines
    )


# =============================================================================
# MAIN
# =============================================================================


def main():

    print("=" * 100)

    print(
        "EXPERIMENT 026 — "
        "FINAL STATISTICAL CONSOLIDATION"
    )

    print("=" * 100)

    print()

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------

    data = (
        load_all_experiments()
    )

    # -------------------------------------------------------------------------
    # Validate
    # -------------------------------------------------------------------------

    validate_inputs(
        data
    )

    # -------------------------------------------------------------------------
    # Build normalized observations
    # -------------------------------------------------------------------------

    print()

    print(
        "Building normalized ARC observations..."
    )

    observations = (
        build_arc_observations(
            data
        )
    )

    print(
        f"ARC observations: "
        f"{len(observations)}"
    )

    # -------------------------------------------------------------------------
    # Analysis
    # -------------------------------------------------------------------------

    print()

    print(
        "Analyzing RQ1..."
    )

    rq1 = (
        analyze_rq1(
            observations
        )
    )

    print(
        "Analyzing RQ2..."
    )

    rq2 = (
        analyze_rq2(
            data
        )
    )

    print(
        "Analyzing RQ3..."
    )

    rq3 = (
        analyze_rq3(
            observations,
            data,
        )
    )

    print(
        "Analyzing RQ4..."
    )

    rq4 = (
        analyze_rq4(
            data
        )
    )

    print(
        "Analyzing reconstruction-error associations..."
    )

    reconstruction = (
        analyze_reconstruction_error(
            observations
        )
    )

    evidence = (
        build_final_evidence()
    )

    # -------------------------------------------------------------------------
    # Output JSON
    # -------------------------------------------------------------------------

    output = {
        "experiment":
            26,

        "title":
            "Final Statistical Consolidation",

        "model":
            "Qwen/Qwen3.5-0.8B",

        "source_experiments":
            [
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
            ],

        "new_model_evaluation":
            False,

        "new_ternarization":
            False,

        "new_configuration_search":
            False,

        "unit_normalization":
            {
                "exp019_024_accuracy":
                    "fraction -> percent",

                "exp019_024_accuracy_difference":
                    "fraction -> percentage points",

                "exp019_024_prediction_agreement":
                    "fraction -> percent",

                "exp022_accuracy_interaction":
                    "fraction -> percentage points",

                "exp019_024_coverage_percent":
                    "already percentage units; unchanged",

                "exp025_accuracy":
                    "already percent; unchanged",

                "exp025_accuracy_difference":
                    (
                        "already percentage points; "
                        "unchanged"
                    ),

                "exp025_prediction_agreement_percent":
                    "already percent; unchanged",

                "reconstruction_error":
                    "unchanged",
            },

        "descriptive_thresholds":
            {
                "accuracy_preservation_pp":
                    ACCURACY_PRESERVATION_THRESHOLD_PP,

                "behavior_agreement_percent":
                    BEHAVIOR_AGREEMENT_THRESHOLD_PERCENT,

                "low_coverage_percent":
                    LOW_COVERAGE_THRESHOLD_PERCENT,
            },

        "arc_observation_count":
            len(observations),

        "arc_observations":
            observations,

        "rq1":
            rq1,

        "rq2":
            rq2,

        "rq3":
            rq3,

        "rq4":
            rq4,

        "reconstruction_error_analysis":
            reconstruction,

        "final_evidence":
            evidence,
    }

    save_json(
        OUTPUT_JSON,
        output,
    )

    # -------------------------------------------------------------------------
    # Text report
    # -------------------------------------------------------------------------

    report = (
        build_report(
            rq1,
            rq2,
            rq3,
            rq4,
            reconstruction,
            evidence,
        )
    )

    with open(
        OUTPUT_TEXT,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(report)
        f.write("\n")

    # -------------------------------------------------------------------------
    # Terminal
    # -------------------------------------------------------------------------

    print()
    print(report)
    print()

    print(
        f"JSON report : "
        f"{OUTPUT_JSON}"
    )

    print(
        f"Text report : "
        f"{OUTPUT_TEXT}"
    )


if __name__ == "__main__":
    main()
