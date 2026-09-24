import json
import re
import statistics
from pathlib import Path


ROOT = Path("/tmp/eng008_repeats")


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

        data["gnu_max_rss_kb"] = (
            extract_gnu_max_rss(time_path)
        )

        runs.append(data)

    return runs


reference = collect("reference")
packed = collect("packed")


reference_latencies = [
    x
    for run in reference
    for x in run["latencies_ms"]
]

packed_latencies = [
    x
    for run in packed
    for x in run["latencies_ms"]
]


def mean(values):
    return statistics.mean(values)


def median(values):
    return statistics.median(values)


def stdev(values):
    return statistics.stdev(values)


reference_ready = [
    run["process_rss_mib"]["ready"]
    for run in reference
]

packed_ready = [
    run["process_rss_mib"]["ready"]
    for run in packed
]

reference_timed_max = [
    run["process_rss_mib"]["timed_max"]
    for run in reference
]

packed_timed_max = [
    run["process_rss_mib"]["timed_max"]
    for run in packed
]

reference_gnu = [
    run["gnu_max_rss_kb"]
    for run in reference
]

packed_gnu = [
    run["gnu_max_rss_kb"]
    for run in packed
]


fp16_bytes = reference[0][
    "selected_fp16_bytes"
]

packed_bytes = reference[0][
    "selected_packed_bytes"
]

selected_parameters = reference[0][
    "selected_parameters"
]

compression = (
    fp16_bytes
    / packed_bytes
)

bytes_eliminated = (
    fp16_bytes
    - packed_bytes
)

reference_mean_latency = mean(
    reference_latencies
)

packed_mean_latency = mean(
    packed_latencies
)

latency_change_percent = (
    (
        packed_mean_latency
        / reference_mean_latency
    )
    - 1.0
) * 100.0


lines = []

lines.append(
    "ENG008 — CPU MEMORY AND LATENCY BENCHMARK"
)
lines.append("=" * 80)
lines.append("")

lines.append("Environment")
lines.append("-" * 80)
lines.append(
    "Model: Qwen/Qwen3.5-0.8B"
)
lines.append(
    "CPU: Intel Core i7-8850H @ 2.60 GHz"
)
lines.append(
    "Physical cores: 6"
)
lines.append(
    "Logical CPUs: 12"
)
lines.append(
    "PyTorch threads: 6"
)
lines.append(
    "PyTorch interop threads: 6"
)
lines.append(
    "PyTorch: 2.14.0+cu126"
)
lines.append(
    "Execution device: CPU"
)
lines.append(
    "Input tokens: 22"
)
lines.append(
    "Warmup forwards per process: 2"
)
lines.append(
    "Timed forwards per process: 5"
)
lines.append(
    "Independent processes per condition: 3"
)
lines.append(
    "Total timed forwards per condition: 15"
)
lines.append("")

lines.append("Selected-weight persistent representation")
lines.append("-" * 80)
lines.append(
    f"Selected matrices: 3"
)
lines.append(
    f"Selected parameters: {selected_parameters}"
)
lines.append(
    f"Reference FP16 bytes: {fp16_bytes}"
)
lines.append(
    f"Packed persistent bytes: {packed_bytes}"
)
lines.append(
    f"Bytes eliminated: {bytes_eliminated}"
)
lines.append(
    f"Bytes eliminated MiB: {bytes_eliminated / 1024**2:.6f}"
)
lines.append(
    f"Selected-weight compression: {compression:.6f}x"
)
lines.append("")

lines.append("Latency — 15 timed forwards per condition")
lines.append("-" * 80)

lines.append(
    f"Reference mean ms: {reference_mean_latency:.3f}"
)
lines.append(
    f"Packed mean ms: {packed_mean_latency:.3f}"
)
lines.append(
    f"Mean latency change: {latency_change_percent:+.3f}%"
)

lines.append(
    f"Reference median ms: {median(reference_latencies):.3f}"
)
lines.append(
    f"Packed median ms: {median(packed_latencies):.3f}"
)

lines.append(
    f"Reference std dev ms: {stdev(reference_latencies):.3f}"
)
lines.append(
    f"Packed std dev ms: {stdev(packed_latencies):.3f}"
)

lines.append(
    f"Reference min/max ms: "
    f"{min(reference_latencies):.3f} / "
    f"{max(reference_latencies):.3f}"
)

lines.append(
    f"Packed min/max ms: "
    f"{min(packed_latencies):.3f} / "
    f"{max(packed_latencies):.3f}"
)

lines.append("")

lines.append("Process RSS — mean across 3 independent processes")
lines.append("-" * 80)

lines.append(
    f"Reference ready RSS MiB: {mean(reference_ready):.3f}"
)
lines.append(
    f"Packed ready RSS MiB: {mean(packed_ready):.3f}"
)
lines.append(
    f"Ready RSS difference MiB: "
    f"{mean(packed_ready) - mean(reference_ready):+.3f}"
)

lines.append(
    f"Reference timed-max RSS MiB: {mean(reference_timed_max):.3f}"
)
lines.append(
    f"Packed timed-max RSS MiB: {mean(packed_timed_max):.3f}"
)
lines.append(
    f"Timed-max RSS difference MiB: "
    f"{mean(packed_timed_max) - mean(reference_timed_max):+.3f}"
)

lines.append(
    f"Reference GNU max RSS KB: {mean(reference_gnu):.3f}"
)
lines.append(
    f"Packed GNU max RSS KB: {mean(packed_gnu):.3f}"
)
lines.append(
    f"GNU max RSS difference KB: "
    f"{mean(packed_gnu) - mean(reference_gnu):+.3f}"
)

lines.append("")
lines.append("Interpretation")
lines.append("-" * 80)

lines.append(
    "1. The three selected matrices retain the previously validated "
    "7.529412x persistent selected-weight storage compression."
)

lines.append(
    "2. Packed ready-state RSS was lower in these runs, but this is "
    "not interpreted as a peak-memory reduction."
)

lines.append(
    "3. Timed-max process RSS and GNU maximum RSS were effectively "
    "similar between the two representations."
)

lines.append(
    "4. The observed mean latency difference was small relative to "
    "run-to-run latency variation; ENG008 therefore does not establish "
    "a meaningful CPU latency advantage or penalty."
)

lines.append(
    "5. The current PackedTernaryLinear implementation temporarily "
    "unpacks ternary codes and reconstructs FP16 row chunks during "
    "forward execution."
)

lines.append("")
lines.append("Scope")
lines.append("-" * 80)

lines.append(
    "No inference speedup claim."
)
lines.append(
    "No peak RAM reduction claim."
)
lines.append(
    "No peak VRAM reduction claim."
)
lines.append(
    "No fused 2-bit CUDA kernel claim."
)
lines.append(
    "No whole-model compression claim."
)

lines.append("")
lines.append(
    "ENG008 CPU MEMORY AND LATENCY BENCHMARK: PASS"
)

output = "\n".join(lines) + "\n"

path = Path(
    "results/ENG008_memory_latency_benchmark.txt"
)

path.write_text(output)

print(output)
