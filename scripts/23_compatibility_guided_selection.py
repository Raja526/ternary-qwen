import json
import itertools
from pathlib import Path


EXP19_PATH = Path("results/019_matrix_arc_ablation.json")
EXP20_PATH = Path("results/020_progressive_safe_arc.json")
EXP22_PATH = Path("results/022_pairwise_compatibility_arc.json")

OUTPUT_JSON = Path("results/023_compatibility_guided_selection.json")
OUTPUT_TXT = Path("results/023_compatibility_guided_selection_summary.txt")


# =============================================================================
# FROZEN COMPATIBILITY RULE
# =============================================================================
#
# A pair is considered compatible when ALL are true:
#
#   |accuracy difference vs FP16| <= 2.0 percentage points
#   prediction agreement         >= 80%
#   exact McNemar p              >= 0.05
#
# I_acc is retained as a descriptive diagnostic but is NOT part of the rule.
# =============================================================================

MAX_ABS_DELTA = 0.020
MIN_AGREEMENT = 0.80
MIN_MCNEMAR_P = 0.05


CANDIDATES = [
    "k_proj",
    "out_proj",
    "in_proj_qkv",
    "q_proj",
    "gate_proj",
    "down_proj",
    "up_proj",
]


FULL_NAMES = {
    "k_proj": "model.layers.3.self_attn.k_proj",
    "out_proj": "model.layers.4.linear_attn.out_proj",
    "in_proj_qkv": "model.layers.4.linear_attn.in_proj_qkv",
    "q_proj": "model.layers.3.self_attn.q_proj",
    "gate_proj": "model.layers.6.mlp.gate_proj",
    "down_proj": "model.layers.6.mlp.down_proj",
    "up_proj": "model.layers.6.mlp.up_proj",
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pair_key(a, b):
    return tuple(sorted((a, b)))


def is_compatible(row):
    return (
        abs(row["accuracy_difference"]) <= MAX_ABS_DELTA
        and row["prediction_agreement"] >= MIN_AGREEMENT
        and row["mcnemar_exact_p"] >= MIN_MCNEMAR_P
    )


def find_all_cliques(compatible_edges):
    """
    Enumerate all cliques of size >= 2 among the seven candidates.
    With only seven nodes, exhaustive enumeration is simple and exact.
    """

    cliques = []

    for size in range(2, len(CANDIDATES) + 1):
        for subset in itertools.combinations(CANDIDATES, size):

            edges = list(itertools.combinations(subset, 2))

            if all(
                pair_key(a, b) in compatible_edges
                for a, b in edges
            ):
                cliques.append(list(subset))

    return cliques


def find_maximal_cliques(cliques):
    """
    A clique is maximal if it is not a strict subset of another clique.
    """

    maximal = []

    clique_sets = [set(c) for c in cliques]

    for i, clique in enumerate(clique_sets):

        contained = False

        for j, other in enumerate(clique_sets):

            if i == j:
                continue

            if clique < other:
                contained = True
                break

        if not contained:
            maximal.append(cliques[i])

    return maximal


def find_exp20_result(exp20, target_names):
    """
    Search Exp020 for the cumulative configuration containing exactly
    target_names.

    This is used only for retrospective comparison. Exp020 is already
    frozen and is NOT used to define compatibility.
    """

    target = set(target_names)

    for row in exp20["results"]:

        names = row.get("matrix_names")

        if names is None:
            names = row.get("matrices")

        if names is None:
            continue

        if set(names) == target:
            return row

    return None


def main():

    exp19 = load_json(EXP19_PATH)
    exp20 = load_json(EXP20_PATH)
    exp22 = load_json(EXP22_PATH)

    baseline = exp22["fp16_baseline"]

    print("=" * 120)
    print("EXPERIMENT 023 — COMPATIBILITY-GUIDED SELECTION ANALYSIS")
    print("=" * 120)

    print()
    print(
        f"FP16 baseline: "
        f"{baseline['correct']}/{baseline['total']} "
        f"({100 * baseline['accuracy']:.2f}%)"
    )

    print()
    print("FROZEN COMPATIBILITY RULE")
    print("-" * 120)
    print(
        f"|Delta accuracy| <= {100 * MAX_ABS_DELTA:.2f} pp"
    )
    print(
        f"Prediction agreement >= {100 * MIN_AGREEMENT:.2f}%"
    )
    print(
        f"McNemar exact p >= {MIN_MCNEMAR_P}"
    )

    # -------------------------------------------------------------------------
    # Individual lookup — Exp019
    # -------------------------------------------------------------------------

    individual_lookup = {
        row["matrix_name"]: row
        for row in exp19["results"]
    }

    missing_individual = [
        FULL_NAMES[short]
        for short in CANDIDATES
        if FULL_NAMES[short] not in individual_lookup
    ]

    if missing_individual:
        raise RuntimeError(
            f"Missing Exp019 matrices: {missing_individual}"
        )

    # -------------------------------------------------------------------------
    # Pair lookup — Exp022
    # -------------------------------------------------------------------------

    pair_rows = {}
    compatible_edges = set()

    for row in exp22["results"]:

        a = row["a_short"]
        b = row["b_short"]

        key = pair_key(a, b)

        if key in pair_rows:
            raise RuntimeError(
                f"Duplicate Exp022 pair: {key}"
            )

        compatible = is_compatible(row)

        pair_rows[key] = {
            **row,
            "compatible": compatible,
        }

        if compatible:
            compatible_edges.add(key)

    expected_pairs = (
        len(CANDIDATES)
        * (len(CANDIDATES) - 1)
        // 2
    )

    if len(pair_rows) != expected_pairs:
        raise RuntimeError(
            f"Expected {expected_pairs} Exp022 pairs, "
            f"found {len(pair_rows)}."
        )

    # -------------------------------------------------------------------------
    # Print all pair classifications
    # -------------------------------------------------------------------------

    print()
    print("PAIRWISE COMPATIBILITY")
    print("-" * 120)

    print(
        f"{'#':<3} "
        f"{'Matrix A':<13} "
        f"{'Matrix B':<13} "
        f"{'Delta':>9} "
        f"{'Agree%':>9} "
        f"{'p':>11} "
        f"{'I_acc':>9} "
        f"{'Result':>12}"
    )

    print("-" * 120)

    ordered_rows = sorted(
        pair_rows.values(),
        key=lambda x: x["pair_id"],
    )

    for row in ordered_rows:

        status = (
            "COMPATIBLE"
            if row["compatible"]
            else "REJECT"
        )

        print(
            f"{row['pair_id']:<3} "
            f"{row['a_short']:<13} "
            f"{row['b_short']:<13} "
            f"{100 * row['accuracy_difference']:>+8.2f} "
            f"{100 * row['prediction_agreement']:>8.2f} "
            f"{row['mcnemar_exact_p']:>11.4g} "
            f"{100 * row['accuracy_interaction']:>+8.2f} "
            f"{status:>12}"
        )

    # -------------------------------------------------------------------------
    # Graph statistics
    # -------------------------------------------------------------------------

    degrees = {
        node: 0
        for node in CANDIDATES
    }

    for a, b in compatible_edges:
        degrees[a] += 1
        degrees[b] += 1

    print()
    print("COMPATIBILITY GRAPH")
    print("-" * 120)

    print(
        f"Nodes              : {len(CANDIDATES)}"
    )

    print(
        f"Possible edges     : {expected_pairs}"
    )

    print(
        f"Compatible edges   : {len(compatible_edges)}"
    )

    print(
        f"Rejected edges     : "
        f"{expected_pairs - len(compatible_edges)}"
    )

    print()
    print("Node degrees:")

    for node in CANDIDATES:
        print(
            f"  {node:<13}: {degrees[node]}"
        )

    # -------------------------------------------------------------------------
    # Cliques
    # -------------------------------------------------------------------------

    cliques = find_all_cliques(
        compatible_edges
    )

    maximal_cliques = find_maximal_cliques(
        cliques
    )

    maximum_size = max(
        (len(c) for c in cliques),
        default=1,
    )

    maximum_cliques = [
        c
        for c in cliques
        if len(c) == maximum_size
    ]

    print()
    print("CLIQUES")
    print("-" * 120)

    for size in range(2, maximum_size + 1):

        same_size = [
            c
            for c in cliques
            if len(c) == size
        ]

        print(
            f"Size {size}: {len(same_size)}"
        )

        for clique in same_size:
            print(
                "  "
                + " + ".join(clique)
            )

    print()
    print(
        f"Maximum clique size: "
        f"{maximum_size}"
    )

    print()
    print("Maximum clique(s):")

    for clique in maximum_cliques:
        print(
            "  "
            + " + ".join(clique)
        )

    print()
    print("Maximal clique(s):")

    for clique in maximal_cliques:
        print(
            "  "
            + " + ".join(clique)
        )

    # -------------------------------------------------------------------------
    # Retrospective Exp020 validation
    # -------------------------------------------------------------------------

    retrospective = []

    print()
    print("RETROSPECTIVE CHECK AGAINST FROZEN EXP020")
    print("-" * 120)

    for clique in maximum_cliques:

        full_names = [
            FULL_NAMES[node]
            for node in clique
        ]

        row = find_exp20_result(
            exp20,
            full_names,
        )

        record = {
            "clique": clique,
            "matrix_names": full_names,
            "exp20_found": row is not None,
        }

        if row is not None:

            record.update({
                "correct":
                    row["correct"],

                "total":
                    row["total"],

                "accuracy":
                    row["accuracy"],

                "accuracy_difference":
                    row["accuracy_difference"],

                "prediction_agreement":
                    row["prediction_agreement"],

                "changed_predictions":
                    row["changed_predictions"],

                "mcnemar_exact_p":
                    row["mcnemar_exact_p"],
            })

            print(
                f"{' + '.join(clique)}"
            )

            print(
                f"  Exp020 accuracy : "
                f"{100 * row['accuracy']:.2f}%"
            )

            print(
                f"  Delta           : "
                f"{100 * row['accuracy_difference']:+.2f} pp"
            )

            print(
                f"  Agreement       : "
                f"{100 * row['prediction_agreement']:.2f}%"
            )

            print(
                f"  Changed         : "
                f"{row['changed_predictions']}"
            )

            print(
                f"  McNemar p       : "
                f"{row['mcnemar_exact_p']:.6g}"
            )

        else:

            print(
                f"{' + '.join(clique)}"
            )

            print(
                "  Exact configuration not found "
                "in Exp020."
            )

        retrospective.append(record)

    # -------------------------------------------------------------------------
    # Build machine-readable result
    # -------------------------------------------------------------------------

    output = {
        "experiment": 23,

        "description":
            "Offline compatibility-guided selection analysis "
            "using frozen Exp019, Exp020, and Exp022 results.",

        "fp16_baseline":
            baseline,

        "compatibility_rule": {
            "max_abs_accuracy_difference":
                MAX_ABS_DELTA,

            "min_prediction_agreement":
                MIN_AGREEMENT,

            "min_mcnemar_exact_p":
                MIN_MCNEMAR_P,

            "accuracy_interaction_used_for_filtering":
                False,
        },

        "candidates":
            CANDIDATES,

        "full_names":
            FULL_NAMES,

        "pair_count":
            expected_pairs,

        "compatible_edge_count":
            len(compatible_edges),

        "compatible_edges": [
            list(edge)
            for edge in sorted(
                compatible_edges
            )
        ],

        "degrees":
            degrees,

        "all_cliques":
            cliques,

        "maximal_cliques":
            maximal_cliques,

        "maximum_clique_size":
            maximum_size,

        "maximum_cliques":
            maximum_cliques,

        "retrospective_exp20":
            retrospective,

        "pair_results": [
            {
                "pair_id":
                    row["pair_id"],

                "a_short":
                    row["a_short"],

                "b_short":
                    row["b_short"],

                "accuracy_difference":
                    row["accuracy_difference"],

                "prediction_agreement":
                    row["prediction_agreement"],

                "mcnemar_exact_p":
                    row["mcnemar_exact_p"],

                "accuracy_interaction":
                    row["accuracy_interaction"],

                "compatible":
                    row["compatible"],
            }
            for row in ordered_rows
        ],
    }

    OUTPUT_JSON.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
        )

    # -------------------------------------------------------------------------
    # Text summary
    # -------------------------------------------------------------------------

    summary = []

    summary.append(
        "EXPERIMENT 023 — COMPATIBILITY-GUIDED SELECTION ANALYSIS"
    )

    summary.append("=" * 80)
    summary.append("")

    summary.append(
        f"FP16 baseline: "
        f"{baseline['correct']}/{baseline['total']} "
        f"({100 * baseline['accuracy']:.2f}%)"
    )

    summary.append("")

    summary.append("Frozen compatibility rule:")

    summary.append(
        f"  |Delta accuracy| <= "
        f"{100 * MAX_ABS_DELTA:.2f} pp"
    )

    summary.append(
        f"  prediction agreement >= "
        f"{100 * MIN_AGREEMENT:.2f}%"
    )

    summary.append(
        f"  McNemar exact p >= "
        f"{MIN_MCNEMAR_P}"
    )

    summary.append("")

    summary.append(
        f"Compatible edges: "
        f"{len(compatible_edges)}/{expected_pairs}"
    )

    summary.append(
        f"Maximum clique size: "
        f"{maximum_size}"
    )

    summary.append("")

    summary.append(
        "Maximum clique(s):"
    )

    for clique in maximum_cliques:
        summary.append(
            "  "
            + " + ".join(clique)
        )

    summary.append("")
    summary.append(
        "Retrospective Exp020 check:"
    )

    for row in retrospective:

        summary.append(
            "  "
            + " + ".join(row["clique"])
        )

        if row["exp20_found"]:

            summary.append(
                f"    accuracy="
                f"{100 * row['accuracy']:.2f}%"
            )

            summary.append(
                f"    delta="
                f"{100 * row['accuracy_difference']:+.2f} pp"
            )

            summary.append(
                f"    agreement="
                f"{100 * row['prediction_agreement']:.2f}%"
            )

            summary.append(
                f"    changed="
                f"{row['changed_predictions']}"
            )

            summary.append(
                f"    McNemar p="
                f"{row['mcnemar_exact_p']:.6g}"
            )

        else:
            summary.append(
                "    not present in Exp020"
            )

    with open(
        OUTPUT_TXT,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "\n".join(summary)
            + "\n"
        )

    print()
    print("=" * 120)
    print("OUTPUT")
    print("=" * 120)

    print(
        f"JSON    : {OUTPUT_JSON}"
    )

    print(
        f"Summary : {OUTPUT_TXT}"
    )


if __name__ == "__main__":
    main()
