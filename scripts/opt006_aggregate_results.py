import json
import re
import statistics
from pathlib import Path


ROOT = Path("results/opt006_raw")

OUTPUT_PATH = Path(
    "results/OPT006_table_dispatch_cpu_benchmark.txt"
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

        if data["experiment"] != "OPT006":
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

branch_reduced = collect(
    "branch_reduced"
)

table_dispatch = collect(
    "table_dispatch"
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

baseline = branch_reduced[0]

for run in branch_reduced + table_dispatch:
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

branch_reduced_latencies = [
    value
    for run in branch_reduced
    for value in run["latencies_ms"]
]

table_dispatch_latencies = [
    value
    for run in table_dispatch
    for value in run["latencies_ms"]
]


# ------------------------------------------------------------
# RSS samples.
# ------------------------------------------------------------

branch_reduced_ready = [
    run["process_rss_mib"]["ready"]
    for run in branch_reduced
]

table_dispatch_ready = [
    run["process_rss_mib"]["ready"]
    for run in table_dispatch
]


branch_reduced_warmup = [
    run["process_rss_mib"]["after_warmup"]
    for run in branch_reduced
]

table_dispatch_warmup = [
    run["process_rss_mib"]["after_warmup"]
    for run in table_dispatch
]


branch_reduced_timed_max = [
    run["process_rss_mib"]["timed_max"]
    for run in branch_reduced
]

table_dispatch_timed_max = [
    run["process_rss_mib"]["timed_max"]
    for run in table_dispatch
]


branch_reduced_gnu = [
    run["gnu_max_rss_kb"]
    for run in branch_reduced
]

table_dispatch_gnu = [
    run["gnu_max_rss_kb"]
    for run in table_dispatch
]


# ------------------------------------------------------------
# Aggregate statistics.
# ------------------------------------------------------------

branch_reduced_mean_latency = mean(
    branch_reduced_latencies
)

table_dispatch_mean_latency = mean(
    table_dispatch_latencies
)

latency_change_percent = (
    (
        table_dispatch_mean_latency
        / branch_reduced_mean_latency
    )
    - 1.0
) * 100.0


ready_difference = (
    mean(table_dispatch_ready)
    - mean(branch_reduced_ready)
)

warmup_difference = (
    mean(table_dispatch_warmup)
    - mean(branch_reduced_warmup)
)

timed_max_difference = (
    mean(table_dispatch_timed_max)
    - mean(branch_reduced_timed_max)
)

gnu_difference = (
    mean(table_dispatch_gnu)
    - mean(branch_reduced_gnu)
)


# ------------------------------------------------------------
# Report.
# ------------------------------------------------------------

lines = []

lines.append(
    "OPT006 — TABLE-DISPATCH PACKED-DECODE CPU BENCHMARK"
)
lines.append("=" * 80)
lines.append("")

lines.append("Controlled comparison")
lines.append("-" * 80)

lines.append(
    "Baseline condition: OPT005 BranchReducedPackedTernaryLinear"
)

lines.append(
    "Candidate condition: OPT006 TableDispatchPackedTernaryLinear"
)

lines.append(
    "Controlled difference: comparison-mask ternary dispatch "
    "vs fixed lookup-table ternary dispatch; manual four-code "
    "unrolling and parallel scheduling are unchanged."
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
    f"{len(branch_reduced_latencies)}"
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
    f"Branch-reduced baseline mean ms: "
    f"{branch_reduced_mean_latency:.3f}"
)

lines.append(
    f"Table-dispatch candidate mean ms: "
    f"{table_dispatch_mean_latency:.3f}"
)

lines.append(
    f"Table-dispatch mean latency change: "
    f"{latency_change_percent:+.3f}%"
)

lines.append(
    f"Branch-reduced baseline median ms: "
    f"{median(branch_reduced_latencies):.3f}"
)

lines.append(
    f"Table-dispatch candidate median ms: "
    f"{median(table_dispatch_latencies):.3f}"
)

lines.append(
    f"Branch-reduced baseline std dev ms: "
    f"{stdev(branch_reduced_latencies):.3f}"
)

lines.append(
    f"Table-dispatch candidate std dev ms: "
    f"{stdev(table_dispatch_latencies):.3f}"
)

lines.append(
    f"Branch-reduced baseline min/max ms: "
    f"{min(branch_reduced_latencies):.3f} / "
    f"{max(branch_reduced_latencies):.3f}"
)

lines.append(
    f"Table-dispatch candidate min/max ms: "
    f"{min(table_dispatch_latencies):.3f} / "
    f"{max(table_dispatch_latencies):.3f}"
)

lines.append("")


lines.append(
    "Process RSS — mean across 3 independent processes"
)
lines.append("-" * 80)

lines.append(
    f"Branch-reduced baseline ready RSS MiB: "
    f"{mean(branch_reduced_ready):.3f}"
)

lines.append(
    f"Table-dispatch candidate ready RSS MiB: "
    f"{mean(table_dispatch_ready):.3f}"
)

lines.append(
    f"Ready RSS difference MiB "
    f"(table_dispatch - branch_reduced): "
    f"{ready_difference:+.3f}"
)

lines.append("")

lines.append(
    f"Branch-reduced baseline after-warmup RSS MiB: "
    f"{mean(branch_reduced_warmup):.3f}"
)

lines.append(
    f"Table-dispatch candidate after-warmup RSS MiB: "
    f"{mean(table_dispatch_warmup):.3f}"
)

lines.append(
    f"After-warmup RSS difference MiB "
    f"(table_dispatch - branch_reduced): "
    f"{warmup_difference:+.3f}"
)

lines.append("")

lines.append(
    f"Branch-reduced baseline timed-max RSS MiB: "
    f"{mean(branch_reduced_timed_max):.3f}"
)

lines.append(
    f"Table-dispatch candidate timed-max RSS MiB: "
    f"{mean(table_dispatch_timed_max):.3f}"
)

lines.append(
    f"Timed-max RSS difference MiB "
    f"(table_dispatch - branch_reduced): "
    f"{timed_max_difference:+.3f}"
)

lines.append("")

lines.append(
    f"Branch-reduced baseline GNU max RSS KB: "
    f"{mean(branch_reduced_gnu):.3f}"
)

lines.append(
    f"Table-dispatch candidate GNU max RSS KB: "
    f"{mean(table_dispatch_gnu):.3f}"
)

lines.append(
    f"GNU max RSS difference KB "
    f"(table_dispatch - branch_reduced): "
    f"{gnu_difference:+.3f}"
)

lines.append("")


lines.append("Interpretation")
lines.append("-" * 80)

lines.append(
    "1. OPT005 and OPT006 use the same persistent packed "
    "selected-weight representation."
)

lines.append(
    "2. OPT006 changes only the source-level per-code ternary "
    "dispatch from comparison-mask accumulation to fixed "
    "lookup-table dispatch; manual four-code unrolling and "
    "parallel scheduling are unchanged."
)

lines.append(
    "3. Separate equivalence validation established bit-exact "
    "operator and full-model outputs for the tested configuration."
)

lines.append(
    f"4. Table-dispatch source changed observed mean forward "
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
    "No claim about the exact generated instruction sequence "
    "without assembly inspection."
)

lines.append(
    "No claim that the compiler implements the source LUT as "
    "a physical memory-table lookup."
)

lines.append(
    "No whole-model compression claim."
)

lines.append("")

lines.append(
    "OPT006 TABLE-DISPATCH PACKED-DECODE CPU BENCHMARK: PASS"
)


output = "\n".join(lines) + "\n"

OUTPUT_PATH.write_text(output)

print(output)
