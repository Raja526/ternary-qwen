import json
import re
import statistics
from pathlib import Path


ROOT = Path("/tmp/eng012_repeats")

OUTPUT_PATH = Path(
    "results/ENG012_streaming_memory_latency_benchmark.txt"
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
        result_path = ROOT / f"{mode}_{i}.txt"
        time_path = ROOT / f"{mode}_{i}_time.txt"

        data = extract_json(result_path)

        if data["experiment"] != "ENG012":
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


direct = collect("direct")
streaming = collect("streaming")


# ------------------------------------------------------------------
# Validate controlled configuration.
# ------------------------------------------------------------------

controlled_fields = [
    "model",
    "torch_version",
    "torch_threads",
    "torch_interop_threads",
    "warmup_runs",
    "timed_runs",
    "input_tokens",
    "rows_per_chunk",
    "selected_matrices",
    "selected_parameters",
    "selected_fp16_bytes",
    "selected_packed_bytes",
    "selected_storage_compression",
    "logits_shape",
    "logits_dtype",
]

baseline = direct[0]

for run in direct + streaming:
    for field in controlled_fields:
        if run[field] != baseline[field]:
            raise RuntimeError(
                "Controlled benchmark mismatch: "
                f"field={field}, "
                f"expected={baseline[field]!r}, "
                f"observed={run[field]!r}"
            )


direct_latencies = [
    value
    for run in direct
    for value in run["latencies_ms"]
]

streaming_latencies = [
    value
    for run in streaming
    for value in run["latencies_ms"]
]


direct_ready = [
    run["process_rss_mib"]["ready"]
    for run in direct
]

streaming_ready = [
    run["process_rss_mib"]["ready"]
    for run in streaming
]


direct_warmup = [
    run["process_rss_mib"]["after_warmup"]
    for run in direct
]

streaming_warmup = [
    run["process_rss_mib"]["after_warmup"]
    for run in streaming
]


direct_timed_max = [
    run["process_rss_mib"]["timed_max"]
    for run in direct
]

streaming_timed_max = [
    run["process_rss_mib"]["timed_max"]
    for run in streaming
]


direct_gnu = [
    run["gnu_max_rss_kb"]
    for run in direct
]

streaming_gnu = [
    run["gnu_max_rss_kb"]
    for run in streaming
]


selected_parameters = baseline[
    "selected_parameters"
]

fp16_bytes = baseline[
    "selected_fp16_bytes"
]

packed_bytes = baseline[
    "selected_packed_bytes"
]

compression = baseline[
    "selected_storage_compression"
]

rows_per_chunk = baseline[
    "rows_per_chunk"
]


direct_mean_latency = mean(
    direct_latencies
)

streaming_mean_latency = mean(
    streaming_latencies
)

latency_change_percent = (
    (
        streaming_mean_latency
        / direct_mean_latency
    )
    - 1.0
) * 100.0


ready_difference = (
    mean(streaming_ready)
    - mean(direct_ready)
)

warmup_difference = (
    mean(streaming_warmup)
    - mean(direct_warmup)
)

timed_max_difference = (
    mean(streaming_timed_max)
    - mean(direct_timed_max)
)

gnu_difference = (
    mean(streaming_gnu)
    - mean(direct_gnu)
)


lines = []

lines.append(
    "ENG012 — STREAMING MEMORY AND LATENCY BENCHMARK"
)
lines.append("=" * 80)
lines.append("")

lines.append("Controlled comparison")
lines.append("-" * 80)
lines.append(
    "Direct condition: ENG010 DirectTernaryLinear"
)
lines.append(
    "Streaming condition: ENG011 StreamingDirectTernaryLinear"
)
lines.append(
    "Controlled difference: full selected-matrix int8 decode "
    "vs output-row-chunk int8 decode."
)
lines.append(
    "Direct ternary arithmetic otherwise remains unchanged."
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
    f"Rows per chunk: {rows_per_chunk}"
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
    f"{len(direct_latencies)}"
)
lines.append("")


lines.append("Selected-weight persistent representation")
lines.append("-" * 80)
lines.append(
    f"Selected matrices: {baseline['selected_matrices']}"
)
lines.append(
    f"Selected parameters: {selected_parameters}"
)
lines.append(
    f"FP16-equivalent bytes: {fp16_bytes}"
)
lines.append(
    f"Packed persistent bytes: {packed_bytes}"
)
lines.append(
    f"Persistent compression: {compression:.6f}x"
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
    f"Direct mean ms: "
    f"{direct_mean_latency:.3f}"
)
lines.append(
    f"Streaming mean ms: "
    f"{streaming_mean_latency:.3f}"
)
lines.append(
    f"Streaming mean latency change: "
    f"{latency_change_percent:+.3f}%"
)
lines.append(
    f"Direct median ms: "
    f"{median(direct_latencies):.3f}"
)
lines.append(
    f"Streaming median ms: "
    f"{median(streaming_latencies):.3f}"
)
lines.append(
    f"Direct std dev ms: "
    f"{stdev(direct_latencies):.3f}"
)
lines.append(
    f"Streaming std dev ms: "
    f"{stdev(streaming_latencies):.3f}"
)
lines.append(
    f"Direct min/max ms: "
    f"{min(direct_latencies):.3f} / "
    f"{max(direct_latencies):.3f}"
)
lines.append(
    f"Streaming min/max ms: "
    f"{min(streaming_latencies):.3f} / "
    f"{max(streaming_latencies):.3f}"
)
lines.append("")


lines.append(
    "Process RSS — mean across 3 independent processes"
)
lines.append("-" * 80)

lines.append(
    f"Direct ready RSS MiB: "
    f"{mean(direct_ready):.3f}"
)
lines.append(
    f"Streaming ready RSS MiB: "
    f"{mean(streaming_ready):.3f}"
)
lines.append(
    f"Ready RSS difference MiB "
    f"(streaming - direct): "
    f"{ready_difference:+.3f}"
)
lines.append("")

lines.append(
    f"Direct after-warmup RSS MiB: "
    f"{mean(direct_warmup):.3f}"
)
lines.append(
    f"Streaming after-warmup RSS MiB: "
    f"{mean(streaming_warmup):.3f}"
)
lines.append(
    f"After-warmup RSS difference MiB "
    f"(streaming - direct): "
    f"{warmup_difference:+.3f}"
)
lines.append("")

lines.append(
    f"Direct timed-max RSS MiB: "
    f"{mean(direct_timed_max):.3f}"
)
lines.append(
    f"Streaming timed-max RSS MiB: "
    f"{mean(streaming_timed_max):.3f}"
)
lines.append(
    f"Timed-max RSS difference MiB "
    f"(streaming - direct): "
    f"{timed_max_difference:+.3f}"
)
lines.append("")

lines.append(
    f"Direct GNU max RSS KB: "
    f"{mean(direct_gnu):.3f}"
)
lines.append(
    f"Streaming GNU max RSS KB: "
    f"{mean(streaming_gnu):.3f}"
)
lines.append(
    f"GNU max RSS difference KB "
    f"(streaming - direct): "
    f"{gnu_difference:+.3f}"
)
lines.append("")


lines.append("Interpretation")
lines.append("-" * 80)
lines.append(
    "1. ENG010 and ENG011 use the same persistent packed "
    "selected-weight representation."
)
lines.append(
    "2. ENG011 changes temporary code decoding from a complete "
    "selected-matrix int8 decode to output-row-chunk decoding."
)
lines.append(
    "3. The RSS measurements above determine whether that smaller "
    "temporary decoded-code working set is visible at the process level "
    "under this controlled CPU benchmark."
)
lines.append(
    "4. The latency measurements quantify the observed CPU cost or "
    "benefit of streaming chunk-local decoding relative to ENG010."
)
lines.append(
    "5. These measurements apply to this model, three selected matrices, "
    "22-token input, rows_per_chunk=64, CPU environment, and benchmark "
    "protocol."
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
    "No fused 2-bit kernel claim."
)
lines.append(
    "No whole-model compression claim."
)
lines.append("")

lines.append(
    "ENG012 STREAMING MEMORY AND LATENCY BENCHMARK: PASS"
)


output = "\n".join(lines) + "\n"

OUTPUT_PATH.write_text(output)

print(output)
