import json
import re
import statistics
from pathlib import Path


ROOT = Path("results/opt004_raw")

OUTPUT_PATH = Path(
    "results/OPT004_unrolled_cpu_benchmark.txt"
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

        if data["experiment"] != "OPT004":
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


unrolled = collect("unrolled")
parallel = collect("parallel")


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

baseline = parallel[0]

for run in parallel + unrolled:
    for field in controlled_fields:
        if run[field] != baseline[field]:
            raise RuntimeError(
                "Controlled benchmark mismatch: "
                f"field={field}, "
                f"expected={baseline[field]!r}, "
                f"observed={run[field]!r}"
            )


parallel_latencies = [
    value
    for run in parallel
    for value in run["latencies_ms"]
]

unrolled_latencies = [
    value
    for run in unrolled
    for value in run["latencies_ms"]
]

parallel_ready = [
    run["process_rss_mib"]["ready"]
    for run in parallel
]

unrolled_ready = [
    run["process_rss_mib"]["ready"]
    for run in unrolled
]

parallel_warmup = [
    run["process_rss_mib"]["after_warmup"]
    for run in parallel
]

unrolled_warmup = [
    run["process_rss_mib"]["after_warmup"]
    for run in unrolled
]

parallel_timed_max = [
    run["process_rss_mib"]["timed_max"]
    for run in parallel
]

unrolled_timed_max = [
    run["process_rss_mib"]["timed_max"]
    for run in unrolled
]

parallel_gnu = [
    run["gnu_max_rss_kb"]
    for run in parallel
]

unrolled_gnu = [
    run["gnu_max_rss_kb"]
    for run in unrolled
]

parallel_mean_latency = mean(
    parallel_latencies
)

unrolled_mean_latency = mean(
    unrolled_latencies
)

latency_change_percent = (
    (
        unrolled_mean_latency
        / parallel_mean_latency
    )
    - 1.0
) * 100.0


ready_difference = (
    mean(unrolled_ready)
    - mean(parallel_ready)
)

warmup_difference = (
    mean(unrolled_warmup)
    - mean(parallel_warmup)
)

timed_max_difference = (
    mean(unrolled_timed_max)
    - mean(parallel_timed_max)
)

gnu_difference = (
    mean(unrolled_gnu)
    - mean(parallel_gnu)
)


lines = []

lines.append(
    "OPT004 — UNROLLED PACKED-DECODE CPU BENCHMARK"
)
lines.append("=" * 80)
lines.append("")

lines.append("Controlled comparison")
lines.append("-" * 80)
lines.append(
    "Baseline condition: OPT003 ParallelPackedTernaryLinear"
)
lines.append(
    "Candidate condition: OPT004 UnrolledPackedTernaryLinear"
)
lines.append(
    "Controlled difference: looped four-code packed decode "
    "vs manually unrolled four-code packed decode; "
    "parallel scheduling is unchanged."
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
    f"{len(parallel_latencies)}"
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
    f"Parallel baseline mean ms: "
    f"{parallel_mean_latency:.3f}"
)
lines.append(
    f"Unrolled candidate mean ms: "
    f"{unrolled_mean_latency:.3f}"
)
lines.append(
    f"Unrolled mean latency change: "
    f"{latency_change_percent:+.3f}%"
)

lines.append(
    f"Parallel baseline median ms: "
    f"{median(parallel_latencies):.3f}"
)
lines.append(
    f"Unrolled candidate median ms: "
    f"{median(unrolled_latencies):.3f}"
)

lines.append(
    f"Parallel baseline std dev ms: "
    f"{stdev(parallel_latencies):.3f}"
)
lines.append(
    f"Unrolled candidate std dev ms: "
    f"{stdev(unrolled_latencies):.3f}"
)

lines.append(
    f"Parallel baseline min/max ms: "
    f"{min(parallel_latencies):.3f} / "
    f"{max(parallel_latencies):.3f}"
)
lines.append(
    f"Unrolled candidate min/max ms: "
    f"{min(unrolled_latencies):.3f} / "
    f"{max(unrolled_latencies):.3f}"
)
lines.append("")


lines.append(
    "Process RSS — mean across 3 independent processes"
)
lines.append("-" * 80)

lines.append(
    f"Parallel baseline ready RSS MiB: "
    f"{mean(parallel_ready):.3f}"
)
lines.append(
    f"Unrolled candidate ready RSS MiB: "
    f"{mean(unrolled_ready):.3f}"
)
lines.append(
    f"Ready RSS difference MiB "
    f"(unrolled - parallel): "
    f"{ready_difference:+.3f}"
)
lines.append("")

lines.append(
    f"Parallel baseline after-warmup RSS MiB: "
    f"{mean(parallel_warmup):.3f}"
)
lines.append(
    f"Unrolled candidate after-warmup RSS MiB: "
    f"{mean(unrolled_warmup):.3f}"
)
lines.append(
    f"After-warmup RSS difference MiB "
    f"(unrolled - parallel): "
    f"{warmup_difference:+.3f}"
)
lines.append("")

lines.append(
    f"Parallel baseline timed-max RSS MiB: "
    f"{mean(parallel_timed_max):.3f}"
)
lines.append(
    f"Unrolled candidate timed-max RSS MiB: "
    f"{mean(unrolled_timed_max):.3f}"
)
lines.append(
    f"Timed-max RSS difference MiB "
    f"(unrolled - parallel): "
    f"{timed_max_difference:+.3f}"
)
lines.append("")

lines.append(
    f"Parallel baseline GNU max RSS KB: "
    f"{mean(parallel_gnu):.3f}"
)
lines.append(
    f"Unrolled candidate GNU max RSS KB: "
    f"{mean(unrolled_gnu):.3f}"
)
lines.append(
    f"GNU max RSS difference KB "
    f"(unrolled - parallel): "
    f"{gnu_difference:+.3f}"
)
lines.append("")


lines.append("Interpretation")
lines.append("-" * 80)

lines.append(
    "1. OPT003 and OPT004 use the same persistent packed "
    "selected-weight representation."
)

lines.append(
    "2. OPT004 changes only the inner four-code packed decode "
    "from a four-iteration loop to explicit manual unrolling; "
    "parallel scheduling is unchanged."
)

lines.append(
    "3. Separate equivalence validation established bit-exact "
    "operator and full-model outputs for the tested configuration."
)

lines.append(
    f"4. Manual four-code unrolling changed mean forward latency by "
    f"{latency_change_percent:+.3f}% in these runs."
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
    "No whole-model compression claim."
)
lines.append("")

lines.append(
    "OPT004 UNROLLED PACKED-DECODE CPU BENCHMARK: PASS"
)


output = "\n".join(lines) + "\n"

OUTPUT_PATH.write_text(output)

print(output)
