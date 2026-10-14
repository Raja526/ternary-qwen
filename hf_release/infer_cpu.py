#!/usr/bin/env python3
"""Run Qwen3.5-0.8B with the retained OPT008 packed CPU modules."""

import argparse
import hashlib
import platform
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from packed_ternary.avx2_runtime_loader import load_avx2_runtime_modules


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = BASE_DIR / "ternary_selected_g128_v1.pt"
EXPECTED_SHA256 = "8e35aaea4e47a68363a8951fa425874a1e35929a19c277335a260bb20a9cb546"
TARGETS = (
    "model.layers.3.self_attn.k_proj",
    "model.layers.4.linear_attn.out_proj",
    "model.layers.4.linear_attn.in_proj_qkv",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def has_avx2() -> bool:
    if platform.system() == "Linux":
        try:
            flags = Path("/proc/cpuinfo").read_text(errors="replace").lower()
            return "avx2" in flags
        except OSError:
            pass
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the three-matrix packed ternary CPU runtime."
    )
    parser.add_argument("--prompt", required=True, help="Text prompt")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Packed artifact path",
    )
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen3.5-0.8B",
        help="Upstream model ID to download/load",
    )
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--timing", action="store_true", help="Print generation latency")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    checkpoint = args.checkpoint.expanduser()
    if not checkpoint.is_absolute():
        checkpoint = Path.cwd() / checkpoint
    if not checkpoint.exists():
        raise FileNotFoundError(f"Packed checkpoint not found: {checkpoint}")

    actual_sha256 = sha256_file(checkpoint)
    if actual_sha256 != EXPECTED_SHA256:
        raise RuntimeError(
            "Checkpoint SHA256 mismatch: "
            f"expected {EXPECTED_SHA256}, got {actual_sha256}"
        )

    if not has_avx2():
        raise RuntimeError("This retained OPT008 runtime requires an AVX2-capable CPU.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.eval()

    try:
        load_result = load_avx2_runtime_modules(
            model=model,
            checkpoint_path=checkpoint,
            selected_module_names=TARGETS,
            expected_model_name=args.model_id,
        )
    except Exception as exc:
        raise RuntimeError(
            "OPT008 installation failed; no FP16 fallback is provided."
        ) from exc

    if tuple(load_result.loaded_module_names) != TARGETS:
        raise RuntimeError("OPT008 did not install exactly the three selected modules.")

    encoded = tokenizer(args.prompt, return_tensors="pt", add_special_tokens=True)
    start = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=args.max_new_tokens,
        )
    elapsed = time.perf_counter() - start

    print(tokenizer.decode(generated[0], skip_special_tokens=True))
    if args.timing:
        print(f"generation_latency_seconds: {elapsed:.6f}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
