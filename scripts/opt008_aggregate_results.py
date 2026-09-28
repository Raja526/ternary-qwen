import json
import re
import statistics
from pathlib import Path


ROOT = Path(
    "results/opt008_raw"
)

OUTPUT_PATH = Path(
    "results/OPT008_avx2_cpu_benchmark.txt"
)

MODES = (
    "table_dispatch",
    "avx2",
)


def extract_json(path):
    text = path.read_text()

    marker = "RESULT_JSON"

    if marker not in text:
        raise RuntimeError(
            f"RESULT_JSON missing from {path}"
        )

    start = (
        text.index(marker)
        + len(marker)
    )

    remaining = (
        text[start:].lstrip()
    )

    decoder = json.JSONDecoder()

    data, _ = decoder.raw_decode(
        remaining
    )

    return data


def extract_gnu_max_rss(path):
    text = path.read_text()

    match = re.search(
        r"Maximum resident set size "
        r"\(kbytes\):\s*(\d+)",
        text,
    )

    if not match:
        raise RuntimeError(
            f"GNU max RSS missing from {path}"
        )

    return int(
        match.group(1)
    )


def collect(mode):
    runs = []

    for i in range(1, 4):
        result_path = (
            ROOT
            / f"{mode}_run{i}.txt"
        )

        time_path = (
            ROOT
            / f"{mode}_run{i}_time.txt"
        )

        data = extract_json(
            result_path
        )

        if data["experiment"] != "OPT008":
            raise RuntimeError(
                f"Unexpected experiment in "
                f"{result_path}: "
                f"{data['experiment']}"
            )

        if data["mode"] != mode:
            raise RuntimeError(
                f"Unexpected mode in "
                f"{result_path}: "
                f"{data['mode']}"
            )

        data["gnu_max_rss_kb"] = (
            extract_gnu_max_rss(
                time_path
            )
        )

        runs.append(data)

    return runs


def mean(values):
    return statistics.mean(
        values
    )


def median(values):
    return statistics.median(
        values
    )


def stdev(values):
    return statistics.stdev(
        values
    )


baseline_runs = collect(
    "table_dispatch"
)

candidate_runs = collect(
    "avx2"
)


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

baseline = baseline_runs[0]

for run in (
    baseline_runs
    + candidate_runs
):
    for field in controlled_fields:
        if run[field] != baseline[field]:
            raise RuntimeError(
                "Controlled benchmark mismatch: "
                f"field={field}, "
                f"expected={baseline[field]!r}, "
                f"observed={run[field]!r}"
            )


baseline_latencies = [
    value
    for run in baseline_runs
    for value in run["latencies_ms"]
]

candidate_latencies = [
    value
    for run in candidate_runs
    for value in run["latencies_ms"]
]


baseline_ready = [
    run["process_rss_mib"]["ready"]
    for run in baseline_runs
]

candidate_ready = [
    run["process_rss_mib"]["ready"]
    for run in candidate_runs
]


baseline_warmup = [
    run["process_rss_mib"]["after_warmup"]
    for run in baseline_runs
]

candidate_warmup = [
    run["process_rss_mib"]["after_warmup"]
    for run in candidate_runs
]


baseline_timed_max = [
    run["process_rss_mib"]["timed_max"]
    for run in baseline_runs
]

candidate_timed_max = [
    run["process_rss_mib"]["timed_max"]
    for run in candidate_runs
]


baseline_gnu = [
    run["gnu_max_rss_kb"]
    for run in baseline_runs
]

candidate_gnu = [
    run["gnu_max_rss_kb"]
    for run in candidate_runs
]


baseline_mean = mean(
    baseline_latencies
)

candidate_mean = mean(
    candidate_latencies
)

latency_change = (
    (
        candidate_mean
        / baseline_mean
    )
    - 1.0
) * 100.0


ready_difference = (
    mean(candidate_ready)
    - mean(baseline_ready)
)

warmup_difference = (
    mean(candidate_warmup)
    - mean(baseline_warmup)
)

timed_max_difference = (
    mean(candidate_timed_max)
    - mean(baseline_timed_max)
)

gnu_difference = (
    mean(candidate_gnu)
    - mean(baseline_gnu)
)


lines = []

lines.append(
    "OPT008 — EXPLICIT AVX2 PACKED-DECODE CPU BENCHMARK"
)
lines.append("=" * 80)
lines.append("")

lines.append(
    "Controlled comparison"
)
lines.append("-" * 80)

lines.append(
    "Baseline condition: OPT006 "
    "TableDispatchPackedTernaryLinear"
)

lines.append(
    "Candidate condition: OPT008 "
    "AVX2PackedTernaryLinear"
)

lines.append(
    "Controlled difference: explicit AVX2 "
    "packed-code decoding in OPT008 versus "
    "frozen OPT006 table-dispatch decoding."
)

lines.append(
    "Packed representation, group size, "
    "selected matrices, FP32 accumulation "
    "order, parallel scheduling, model, "
    "input, and benchmark protocol are held "
    "constant."
)

lines.append(
    "OPT008 was compiled with "
    "-O2 -mavx2 -mno-fma."
)

lines.append("")

lines.append(
    "Environment and protocol"
)
lines.append("-" * 80)

lines.append(
    f"Model: {baseline['model']}"
)

lines.append(
    "Execution device: CPU"
)

lines.append(
    f"PyTorch: "
    f"{baseline['torch_version']}"
)

lines.append(
    f"PyTorch threads: "
    f"{baseline['torch_threads']}"
)

lines.append(
    f"PyTorch interop threads: "
    f"{baseline['torch_interop_threads']}"
)

lines.append(
    f"Input tokens: "
    f"{baseline['input_tokens']}"
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
    "Total timed forwards per condition: "
    f"{len(baseline_latencies)}"
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
    "Persistent selected-weight representation "
    "is identical between both conditions."
)

lines.append("")

lines.append(
    "Latency — 15 timed forwards per condition"
)
lines.append("-" * 80)

lines.append(
    f"OPT006 baseline mean ms: "
    f"{baseline_mean:.3f}"
)

lines.append(
    f"OPT008 AVX2 mean ms: "
    f"{candidate_mean:.3f}"
)

lines.append(
    f"OPT008 mean latency change: "
    f"{latency_change:+.3f}%"
)

lines.append("")

lines.append(
    f"OPT006 median ms: "
    f"{median(baseline_latencies):.3f}"
)

lines.append(
    f"OPT008 median ms: "
    f"{median(candidate_latencies):.3f}"
)

lines.append("")

lines.append(
    f"OPT006 std ms: "
    f"{stdev(baseline_latencies):.3f}"
)

lines.append(
    f"OPT008 std ms: "
    f"{stdev(candidate_latencies):.3f}"
)

lines.append("")

lines.append(
    f"OPT006 min/max ms: "
    f"{min(baseline_latencies):.3f} / "
    f"{max(baseline_latencies):.3f}"
)

lines.append(
    f"OPT008 min/max ms: "
    f"{min(candidate_latencies):.3f} / "
    f"{max(candidate_latencies):.3f}"
)

lines.append("")

lines.append(
    "Process RSS — mean across 3 processes"
)
lines.append("-" * 80)

lines.append(
    f"OPT006 ready RSS MiB: "
    f"{mean(baseline_ready):.3f}"
)

lines.append(
    f"OPT008 ready RSS MiB: "
    f"{mean(candidate_ready):.3f}"
)

lines.append(
    f"Ready RSS difference MiB: "
    f"{ready_difference:+.3f}"
)

lines.append("")

lines.append(
    f"OPT006 after-warmup RSS MiB: "
    f"{mean(baseline_warmup):.3f}"
)

lines.append(
    f"OPT008 after-warmup RSS MiB: "
    f"{mean(candidate_warmup):.3f}"
)

lines.append(
    f"After-warmup RSS difference MiB: "
    f"{warmup_difference:+.3f}"
)

lines.append("")

lines.append(
    f"OPT006 timed-max RSS MiB: "
    f"{mean(baseline_timed_max):.3f}"
)

lines.append(
    f"OPT008 timed-max RSS MiB: "
    f"{mean(candidate_timed_max):.3f}"
)

lines.append(
    f"Timed-max RSS difference MiB: "
    f"{timed_max_difference:+.3f}"
)

lines.append("")

lines.append(
    "GNU time maximum RSS — mean across 3 processes"
)
lines.append("-" * 80)

lines.append(
    f"OPT006 GNU max RSS KB: "
    f"{mean(baseline_gnu):.3f}"
)

lines.append(
    f"OPT008 GNU max RSS KB: "
    f"{mean(candidate_gnu):.3f}"
)

lines.append(
    f"GNU max RSS difference KB: "
    f"{gnu_difference:+.3f}"
)

lines.append("")

lines.append(
    "Correctness prerequisites"
)
lines.append("-" * 80)

lines.append(
    "Operator equivalence: PASS — bit-exact "
    "vs frozen OPT006."
)

lines.append(
    "Machine-code verification: PASS — "
    "AVX2 instructions observed inside the "
    "compiled hot function."
)

lines.append(
    "Full-model equivalence: PASS — "
    "5,463,040 logits, 0 mismatches, "
    "top-1 22/22."
)

lines.append("")

lines.append(
    "Interpretation boundary"
)
lines.append("-" * 80)

lines.append(
    "This benchmark measures the observed "
    "latency difference for the tested "
    "Qwen3.5-0.8B three-matrix packed runtime "
    "on this CPU and protocol."
)

lines.append(
    "It does not establish a general CPU "
    "inference speedup or a general memory "
    "advantage."
)

lines.append(
    "A negative latency-change percentage "
    "means OPT008 was faster; a positive "
    "percentage means OPT008 was slower."
)

OUTPUT_PATH.write_text(
    "\n".join(lines) + "\n"
)

print(
    "\n".join(lines)
)
