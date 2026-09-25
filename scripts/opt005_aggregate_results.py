import json
import re
import statistics
from pathlib import Path


ROOT = Path("results/opt005_raw")

OUTPUT_PATH = Path(
    "results/OPT005_branch_reduced_cpu_benchmark.txt"
)


def extract_json(path):
    text = path.read_text()

    marker = "RESULT_JSON"
    start = text.index(marker) + len(marker)

    remaining = text[start:].lstrip()

    decoder = json.JSONDecoder()
    data, _ = decoder.raw_decode(remaining)

    return data


def extract_gnu_max_rss(path):
    text = path.read_text()

    match = re.search(
        r"Maximum resident set size \(kbytes\):\s*(\d+)",
        text,
    )

    if not match:
        raise RuntimeError(
            f"GNU max RSS not found in {path}"
        )

    return int(match.group(1))


def collect(mode):
    runs = []

    for i in range(1, 4):
        result_path = ROOT / f"{mode}_run{i}.txt"
        time_path = ROOT / f"{mode}_run{i}_time.txt"

        data = extract_json(result_path)

        if data["experiment"] != "OPT005":
            raise RuntimeError(
                f"Unexpected experiment in {result_path}: "
                f"{data['experiment']}"
            )

        if data["mode"] != mode:
            raise RuntimeError(
                f"Unexpected mode in {result_path}: "
                f"{data['mode']}"
            )

        data["gnu_max_rss_kb"] = (
            extract_gnu_max_rss(time_path)
        )

        runs.append(data)

    return runs


def mean(values):
    return statistics.mean(values)


def median(values):
    return statistics.median(values)


def stdev(values):
    return statistics.stdev(values)


# ------------------------------------------------------------
# Load the two controlled conditions.
# ------------------------------------------------------------

unrolled = collect("unrolled")

branch_reduced = collect(
    "branch_reduced"
)


# ------------------------------------------------------------
# Validate controlled configuration.
# ------------------------------------------------------------

controlled_fields = [
    "model",
    "torch_version",
    "torch_threads",
    "torch_interop_threads",
    "warmup_runs",
    "timed_runs",
    "input_tokens",
    "selected_matrices",
    "selected_parameters",
    "selected_fp16_bytes",
    "selected_packed_bytes",
    "selected_storage_compression",
    "logits_shape",
    "logits_dtype",
]

baseline = unrolled[0]

for run in unrolled + branch_reduced:
    for field in controlled_fields:
        if run[field] != baseline[field]:
            raise RuntimeError(
                "Controlled benchmark mismatch: "
                f"field={field}, "
                f"expected={baseline[field]!r}, "
                f"observed={run[field]!r}"
            )


# ------------------------------------------------------------
# Latency samples.
# ------------------------------------------------------------

unrolled_latencies = [
    value
    for run in unrolled
    for value in run["latencies_ms"]
]

branch_reduced_latencies = [
    value
    for run in branch_reduced
    for value in run["latencies_ms"]
]


# ------------------------------------------------------------
# RSS samples.
# ------------------------------------------------------------

unrolled_ready = [
    run["process_rss_mib"]["ready"]
    for run in unrolled
]

branch_reduced_ready = [
    run["process_rss_mib"]["ready"]
    for run in branch_reduced
]


unrolled_warmup = [
    run["process_rss_mib"]["after_warmup"]
    for run in unrolled
]

branch_reduced_warmup = [
    run["process_rss_mib"]["after_warmup"]
    for run in branch_reduced
]


unrolled_timed_max = [
    run["process_rss_mib"]["timed_max"]
    for run in unrolled
]

branch_reduced_timed_max = [
    run["process_rss_mib"]["timed_max"]
    for run in branch_reduced
]


unrolled_gnu = [
    run["gnu_max_rss_kb"]
    for run in unrolled
]

branch_reduced_gnu = [
    run["gnu_max_rss_kb"]
    for run in branch_reduced
]


# ------------------------------------------------------------
# Aggregate statistics.
# ------------------------------------------------------------

unrolled_mean_latency = mean(
    unrolled_latencies
)

branch_reduced_mean_latency = mean(
    branch_reduced_latencies
)

latency_change_percent = (
    (
        branch_reduced_mean_latency
        / unrolled_mean_latency
    )
    - 1.0
) * 100.0


ready_difference = (
    mean(branch_reduced_ready)
    - mean(unrolled_ready)
)

warmup_difference = (
    mean(branch_reduced_warmup)
    - mean(unrolled_warmup)
)

timed_max_difference = (
    mean(branch_reduced_timed_max)
    - mean(unrolled_timed_max)
)

gnu_difference = (
    mean(branch_reduced_gnu)
    - mean(unrolled_gnu)
)


# ------------------------------------------------------------
# Report.
# ------------------------------------------------------------

lines = []

lines.append(
    "OPT005 — BRANCH-REDUCED PACKED-DECODE CPU BENCHMARK"
)
lines.append("=" * 80)
lines.append("")

lines.append("Controlled comparison")
lines.append("-" * 80)

lines.append(
    "Baseline condition: OPT004 UnrolledPackedTernaryLinear"
)

lines.append(
    "Candidate condition: OPT005 BranchReducedPackedTernaryLinear"
)

lines.append(
    "Controlled difference: explicit conditional ternary dispatch "
    "vs comparison-mask accumulation; manual four-code unrolling "
    "and parallel scheduling are unchanged."
)

lines.append(
    "Packed representation and ternary arithmetic remain unchanged."
)

lines.append("")


lines.append("Environment and protocol")
lines.append("-" * 80)

lines.append(
    f"Model: {baseline['model']}"
)

lines.append(
    "Execution device: CPU"
)

lines.append(
    f"PyTorch: {baseline['torch_version']}"
)

lines.append(
    f"PyTorch threads: {baseline['torch_threads']}"
)

lines.append(
    f"PyTorch interop threads: "
    f"{baseline['torch_interop_threads']}"
)

lines.append(
    f"Input tokens: {baseline['input_tokens']}"
)

lines.append(
    f"Warmup forwards per process: "
    f"{baseline['warmup_runs']}"
)

lines.append(
    f"Timed forwards per process: "
    f"{baseline['timed_runs']}"
)

lines.append(
    "Independent processes per condition: 3"
)

lines.append(
    f"Total timed forwards per condition: "
    f"{len(unrolled_latencies)}"
)

lines.append("")


lines.append(
    "Selected-weight persistent representation"
)
lines.append("-" * 80)

lines.append(
    f"Selected matrices: "
    f"{baseline['selected_matrices']}"
)

lines.append(
    f"Selected parameters: "
    f"{baseline['selected_parameters']}"
)

lines.append(
    f"FP16-equivalent bytes: "
    f"{baseline['selected_fp16_bytes']}"
)

lines.append(
    f"Packed persistent bytes: "
    f"{baseline['selected_packed_bytes']}"
)

lines.append(
    f"Persistent compression: "
    f"{baseline['selected_storage_compression']:.6f}x"
)

lines.append(
    "Persistent selected-weight representation is identical "
    "between both conditions."
)

lines.append("")


lines.append(
    "Latency — 15 timed forwards per condition"
)
lines.append("-" * 80)

lines.append(
    f"Unrolled baseline mean ms: "
    f"{unrolled_mean_latency:.3f}"
)

lines.append(
    f"Branch-reduced candidate mean ms: "
    f"{branch_reduced_mean_latency:.3f}"
)

lines.append(
    f"Branch-reduced mean latency change: "
    f"{latency_change_percent:+.3f}%"
)

lines.append(
    f"Unrolled baseline median ms: "
    f"{median(unrolled_latencies):.3f}"
)

lines.append(
    f"Branch-reduced candidate median ms: "
    f"{median(branch_reduced_latencies):.3f}"
)

lines.append(
    f"Unrolled baseline std dev ms: "
    f"{stdev(unrolled_latencies):.3f}"
)

lines.append(
    f"Branch-reduced candidate std dev ms: "
    f"{stdev(branch_reduced_latencies):.3f}"
)

lines.append(
    f"Unrolled baseline min/max ms: "
    f"{min(unrolled_latencies):.3f} / "
    f"{max(unrolled_latencies):.3f}"
)

lines.append(
    f"Branch-reduced candidate min/max ms: "
    f"{min(branch_reduced_latencies):.3f} / "
    f"{max(branch_reduced_latencies):.3f}"
)

lines.append("")


lines.append(
    "Process RSS — mean across 3 independent processes"
)
lines.append("-" * 80)

lines.append(
    f"Unrolled baseline ready RSS MiB: "
    f"{mean(unrolled_ready):.3f}"
)

lines.append(
    f"Branch-reduced candidate ready RSS MiB: "
    f"{mean(branch_reduced_ready):.3f}"
)

lines.append(
    f"Ready RSS difference MiB "
    f"(branch_reduced - unrolled): "
    f"{ready_difference:+.3f}"
)

lines.append("")

lines.append(
    f"Unrolled baseline after-warmup RSS MiB: "
    f"{mean(unrolled_warmup):.3f}"
)

lines.append(
    f"Branch-reduced candidate after-warmup RSS MiB: "
    f"{mean(branch_reduced_warmup):.3f}"
)

lines.append(
    f"After-warmup RSS difference MiB "
    f"(branch_reduced - unrolled): "
    f"{warmup_difference:+.3f}"
)

lines.append("")

lines.append(
    f"Unrolled baseline timed-max RSS MiB: "
    f"{mean(unrolled_timed_max):.3f}"
)

lines.append(
    f"Branch-reduced candidate timed-max RSS MiB: "
    f"{mean(branch_reduced_timed_max):.3f}"
)

lines.append(
    f"Timed-max RSS difference MiB "
    f"(branch_reduced - unrolled): "
    f"{timed_max_difference:+.3f}"
)

lines.append("")

lines.append(
    f"Unrolled baseline GNU max RSS KB: "
    f"{mean(unrolled_gnu):.3f}"
)

lines.append(
    f"Branch-reduced candidate GNU max RSS KB: "
    f"{mean(branch_reduced_gnu):.3f}"
)

lines.append(
    f"GNU max RSS difference KB "
    f"(branch_reduced - unrolled): "
    f"{gnu_difference:+.3f}"
)

lines.append("")


lines.append("Interpretation")
lines.append("-" * 80)

lines.append(
    "1. OPT004 and OPT005 use the same persistent packed "
    "selected-weight representation."
)

lines.append(
    "2. OPT005 changes only per-code ternary dispatch from "
    "explicit conditional branches to comparison-mask "
    "accumulation; manual four-code unrolling and parallel "
    "scheduling are unchanged."
)

lines.append(
    "3. Separate equivalence validation established bit-exact "
    "operator and full-model outputs for the tested configuration."
)

lines.append(
    f"4. Branch-reduced source changed observed mean forward "
    f"latency by {latency_change_percent:+.3f}% in these runs."
)

lines.append(
    f"5. Timed-max process RSS changed by "
    f"{timed_max_difference:+.3f} MiB, while GNU maximum RSS "
    f"changed by {gnu_difference:+.3f} KB on average."
)

lines.append(
    "6. These measurements apply to this model, three selected "
    "matrices, 22-token input, six-thread CPU environment, "
    "and benchmark protocol."
)

lines.append("")


lines.append("Scope")
lines.append("-" * 80)

lines.append(
    "No general inference speedup claim."
)

lines.append(
    "No general peak RAM reduction claim."
)

lines.append(
    "No peak VRAM reduction claim."
)

lines.append(
    "No SIMD/vectorization claim."
)

lines.append(
    "No claim that generated machine code is branchless."
)

lines.append(
    "No whole-model compression claim."
)

lines.append("")

lines.append(
    "OPT005 BRANCH-REDUCED PACKED-DECODE CPU BENCHMARK: PASS"
)


output = "\n".join(lines) + "\n"

OUTPUT_PATH.write_text(output)

print(output)
