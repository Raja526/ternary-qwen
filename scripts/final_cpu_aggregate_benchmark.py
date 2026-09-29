import json
import re
import statistics
from pathlib import Path


ROOT = Path("results/final_cpu_benchmark_raw")


def load_result(path):
    text = path.read_text()

    match = re.search(
        r"RESULT_JSON\s*(\{.*?\})\s*"
        r"={10,}\s*OPT008",
        text,
        flags=re.S,
    )

    if not match:
        raise RuntimeError(
            f"Could not extract RESULT_JSON from {path}"
        )

    return json.loads(match.group(1))


def collect(prefix):
    results = []

    for run in range(1, 4):
        path = ROOT / f"{prefix}_run{run}.log"
        result = load_result(path)

        if len(result["latencies_ms"]) != 5:
            raise RuntimeError(
                f"{path}: expected 5 timed samples"
            )

        results.append(result)

    return results


def summarize(results):
    latencies = [
        value
        for result in results
        for value in result["latencies_ms"]
    ]

    ready = [
        result["process_rss_mib"]["ready"]
        for result in results
    ]

    warmup = [
        result["process_rss_mib"]["after_warmup"]
        for result in results
    ]

    timed_max = [
        result["process_rss_mib"]["timed_max"]
        for result in results
    ]

    return {
        "latencies": latencies,
        "mean": statistics.mean(latencies),
        "median": statistics.median(latencies),
        "stdev": statistics.stdev(latencies),
        "min": min(latencies),
        "max": max(latencies),
        "ready_rss": statistics.mean(ready),
        "warmup_rss": statistics.mean(warmup),
        "timed_max_rss": statistics.mean(timed_max),
    }


def gnu_max_rss(prefix):
    values = []

    for run in range(1, 4):
        path = ROOT / f"{prefix}_run{run}.time.txt"
        text = path.read_text()

        match = re.search(
            r"Maximum resident set size "
            r"\(kbytes\):\s*(\d+)",
            text,
        )

        if not match:
            raise RuntimeError(
                f"Could not find GNU max RSS in {path}"
            )

        values.append(int(match.group(1)))

    return statistics.mean(values)


def show(label, summary):
    print(label)
    print("-" * 72)
    print(
        "samples             :",
        len(summary["latencies"]),
    )
    print(
        "mean ms             :",
        f'{summary["mean"]:.3f}',
    )
    print(
        "median ms           :",
        f'{summary["median"]:.3f}',
    )
    print(
        "sample std ms       :",
        f'{summary["stdev"]:.3f}',
    )
    print(
        "min ms              :",
        f'{summary["min"]:.3f}',
    )
    print(
        "max ms              :",
        f'{summary["max"]:.3f}',
    )
    print(
        "mean ready RSS MiB  :",
        f'{summary["ready_rss"]:.3f}',
    )
    print(
        "mean warmup RSS MiB :",
        f'{summary["warmup_rss"]:.3f}',
    )
    print(
        "mean timed RSS MiB  :",
        f'{summary["timed_max_rss"]:.3f}',
    )
    print()


def main():
    baseline_results = collect("opt006")
    candidate_results = collect("opt008")

    baseline = summarize(baseline_results)
    candidate = summarize(candidate_results)

    show("OPT006 — TABLE DISPATCH", baseline)
    show("OPT008 — AVX2", candidate)

    latency_change = (
        (candidate["mean"] - baseline["mean"])
        / baseline["mean"]
        * 100.0
    )

    median_change = (
        (candidate["median"] - baseline["median"])
        / baseline["median"]
        * 100.0
    )

    ready_diff = (
        candidate["ready_rss"]
        - baseline["ready_rss"]
    )

    warmup_diff = (
        candidate["warmup_rss"]
        - baseline["warmup_rss"]
    )

    timed_diff = (
        candidate["timed_max_rss"]
        - baseline["timed_max_rss"]
    )

    baseline_gnu = gnu_max_rss("opt006")
    candidate_gnu = gnu_max_rss("opt008")

    print("=" * 72)
    print("FINAL CPU VALIDATION — GATE 4")
    print("=" * 72)

    print(
        "OPT006 pooled mean ms       :",
        f"{baseline['mean']:.3f}",
    )
    print(
        "OPT008 pooled mean ms       :",
        f"{candidate['mean']:.3f}",
    )
    print(
        "Mean latency change         :",
        f"{latency_change:+.3f}%",
    )

    print(
        "OPT006 pooled median ms     :",
        f"{baseline['median']:.3f}",
    )
    print(
        "OPT008 pooled median ms     :",
        f"{candidate['median']:.3f}",
    )
    print(
        "Median latency change       :",
        f"{median_change:+.3f}%",
    )

    print(
        "Ready RSS difference MiB    :",
        f"{ready_diff:+.3f}",
    )
    print(
        "Warmup RSS difference MiB   :",
        f"{warmup_diff:+.3f}",
    )
    print(
        "Timed-max RSS diff MiB      :",
        f"{timed_diff:+.3f}",
    )

    print(
        "OPT006 GNU max RSS mean KB  :",
        f"{baseline_gnu:.3f}",
    )
    print(
        "OPT008 GNU max RSS mean KB  :",
        f"{candidate_gnu:.3f}",
    )
    print(
        "GNU max RSS difference KB   :",
        f"{candidate_gnu - baseline_gnu:+.3f}",
    )

    print()
    print("Original frozen OPT008 result:")
    print("  OPT006 mean               : 2175.356 ms")
    print("  OPT008 mean               : 1994.648 ms")
    print("  observed change           : -8.307%")

    print()
    print("Fresh final-validation result:")
    print(
        "  observed change           :",
        f"{latency_change:+.3f}%",
    )


if __name__ == "__main__":
    main()
