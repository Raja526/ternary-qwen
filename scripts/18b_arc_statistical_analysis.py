import json
import math

RESULT_FILE = "results/018_arc_challenge_predictions.json"


def exact_mcnemar(b, c):
    """
    Exact two-sided McNemar test using the binomial distribution.

    b = FP16 correct, ternary wrong
    c = FP16 wrong, ternary correct
    """

    n = b + c
    k = min(b, c)

    one_tail = sum(
        math.comb(n, i)
        for i in range(k + 1)
    ) / (2 ** n)

    return min(1.0, 2.0 * one_tail)


with open(RESULT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)


paired = data["paired"]

b = paired["fp16_only"]
c = paired["ternary_only"]

discordant = b + c

p_exact = exact_mcnemar(b, c)

chi_square_cc = (
    (abs(b - c) - 1) ** 2
    / discordant
)


print("=" * 60)
print("EXPERIMENT 018B — ARC-CHALLENGE PAIRED STATISTICAL ANALYSIS")
print("=" * 60)

print()
print(f"FP16 only correct     : {b}")
print(f"Ternary only correct  : {c}")
print(f"Discordant pairs      : {discordant}")

print()
print(f"Exact McNemar p-value : {p_exact:.12g}")
print(f"Chi-square (CC)       : {chi_square_cc:.6f}")

print()

if p_exact < 0.05:
    print("Result                 : Statistically significant")
else:
    print("Result                 : Not statistically significant")
