import json
import re
import statistics
from pathlib import Path


RAW_DIR = Path("results/opt010_raw")
OUTPUT = Path("results/OPT010_final_summary.txt")

MODES = (
    ("avx2", "OPT008 grain-1"),
    ("grain16", "OPT010 grain-16"),
    ("grain64", "OPT010 grain-64"),
    ("grain256", "OPT010 grain-256"),
)


def extract_latencies(path):
    text = path.read_text()

    match = re.search(
        r'"latencies_ms"\s*:\s*\[(.*?)\]',
        text,
        flags=re.S,
    )

    if not match:
        raise RuntimeError(
            f"latencies_ms missing: {path}"
        )

    values = [
        float(x)
        for x in re.findall(
            r"[0-9]+\.[0-9]+",
            match.group(1),
        )
    ]

    if len(values) != 5:
        raise RuntimeError(
            f"{path}: expected 5 latencies, "
            f"got {len(values)}"
        )

    if "BENCHMARK: PASS" not in text:
        raise RuntimeError(
            f"benchmark did not pass: {path}"
        )

    return values


def extract_process_rss(path):
    text = path.read_text()

    match = re.search(
        r"RSS timed max MiB\s*:\s*"
        r"([0-9.]+)",
        text,
    )

    if not match:
        raise RuntimeError(
            f"RSS timed max missing: {path}"
        )

    return float(match.group(1))


def extract_gnu_rss(path):
    text = path.read_text()

    match = re.search(
        r"Maximum resident set size "
        r"\(kbytes\):\s*([0-9]+)",
        text,
    )

    if not match:
        raise RuntimeError(
            f"GNU max RSS missing: {path}"
        )

    return int(match.group(1))


results = {}

for mode, label in MODES:
    latencies = []
    process_rss = []
    gnu_rss = []

    for run in range(1, 4):
        log_path = (
            RAW_DIR
            / f"{mode}_run{run}.log"
        )

        time_path = (
            RAW_DIR
            / f"{mode}_run{run}.time.txt"
        )

        latencies.extend(
            extract_latencies(log_path)
        )

        process_rss.append(
            extract_process_rss(log_path)
        )

        gnu_rss.append(
            extract_gnu_rss(time_path)
        )

    if len(latencies) != 15:
        raise RuntimeError(
            f"{mode}: expected 15 total samples"
        )

    results[mode] = {
        "label": label,
        "latencies": latencies,
        "mean": statistics.mean(latencies),
        "median": statistics.median(latencies),
        "stdev": statistics.stdev(latencies),
        "min": min(latencies),
        "max": max(latencies),
        "process_rss_mean": statistics.mean(
            process_rss
        ),
        "gnu_rss_mean_kb": statistics.mean(
            gnu_rss
        ),
    }


baseline = results["avx2"]["mean"]

for mode in results:
    results[mode]["delta_percent"] = (
        (
            results[mode]["mean"]
            / baseline
        )
        - 1.0
    ) * 100.0


lines = []

lines.append(
    "OPT010 — THREAD-SCHEDULING / "
    "GRAIN-SIZE FINAL SUMMARY"
)
lines.append("=" * 78)
lines.append("")

lines.append(
    "Controlled variable:"
)
lines.append(
    "  at::parallel_for grain-size argument"
)
lines.append("")

lines.append(
    "Frozen reference:"
)
lines.append(
    "  OPT008 explicit AVX2, grain size 1"
)
lines.append("")

lines.append(
    "Candidates:"
)
lines.append(
    "  grain size 16, 64, 256"
)
lines.append("")

lines.append(
    "Protocol:"
)
lines.append(
    "  CPU threads             : 6"
)
lines.append(
    "  Inter-op threads        : 6"
)
lines.append(
    "  Warmups/process         : 2"
)
lines.append(
    "  Timed forwards/process  : 5"
)
lines.append(
    "  Independent processes   : 3/condition"
)
lines.append(
    "  Samples/condition       : 15"
)
lines.append(
    "  Total timed forwards    : 60"
)
lines.append("")

lines.append(
    "Correctness:"
)
lines.append(
    "  Operator gate           : PASS"
)
lines.append(
    "  Full-model gate         : PASS"
)
lines.append(
    "  Compared logits         : 5,463,040/candidate"
)
lines.append(
    "  Exact mismatches        : 0"
)
lines.append(
    "  Max abs difference      : 0.0"
)
lines.append(
    "  Mean abs difference     : 0.0"
)
lines.append(
    "  Top-1 matches           : 22/22"
)
lines.append("")

lines.append(
    "Latency results:"
)
lines.append(
    f"{'Mode':<22}"
    f"{'N':>5}"
    f"{'Mean':>12}"
    f"{'Median':>12}"
    f"{'Std':>12}"
    f"{'Min':>12}"
    f"{'Max':>12}"
    f"{'Delta':>12}"
)

lines.append("-" * 99)

for mode, _ in MODES:
    r = results[mode]

    delta = (
        "baseline"
        if mode == "avx2"
        else f"{r['delta_percent']:+.3f}%"
    )

    lines.append(
        f"{r['label']:<22}"
        f"{15:>5}"
        f"{r['mean']:>12.3f}"
        f"{r['median']:>12.3f}"
        f"{r['stdev']:>12.3f}"
        f"{r['min']:>12.3f}"
        f"{r['max']:>12.3f}"
        f"{delta:>12}"
    )

lines.append("")
lines.append(
    "Memory:"
)

for mode, _ in MODES:
    r = results[mode]

    lines.append(
        f"  {r['label']:<20} "
        f"timed-max RSS mean "
        f"{r['process_rss_mean']:.3f} MiB; "
        f"GNU max RSS mean "
        f"{r['gnu_rss_mean_kb']:.3f} KB"
    )

lines.append("")
lines.append(
    "Decision:"
)
lines.append(
    "  Retain frozen OPT008 grain size 1."
)
lines.append("")

lines.append(
    "Interpretation:"
)
lines.append(
    "  Under the frozen OPT010 benchmark protocol, "
    "grain sizes 16, 64, and 256"
)
lines.append(
    "  preserved exact behavior but increased "
    "observed mean latency by 2.112%,"
)
lines.append(
    "  1.203%, and 1.307%, respectively, versus "
    "frozen OPT008 grain size 1."
)
lines.append(
    "  No meaningful memory advantage was observed."
)
lines.append("")

lines.append(
    "Claim boundary:"
)
lines.append(
    "  These are protocol-specific observations for "
    "the tested model, hardware,"
)
lines.append(
    "  six-thread configuration, selected matrices, "
    "and workload. They do not"
)
lines.append(
    "  establish that larger parallel_for grain-size "
    "arguments are universally slower."
)

OUTPUT.write_text(
    "\n".join(lines) + "\n"
)

print(
    OUTPUT.read_text()
)
