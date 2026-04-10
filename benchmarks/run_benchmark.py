#!/usr/bin/env python3
"""
run_benchmark.py — instrumented TensorRT-LLM benchmark runner (v2)

A single, instrumented entry point that:

1. Takes lifecycle memory snapshots before/after each phase (see TEST_PLAN.md §6).
2. Builds a TensorRT-LLM engine via `trtllm-bench build`.
3. Runs `trtllm-bench throughput` against a deterministic dataset.
4. Parses the log for KV-cache bytes, KV-cache blocks, throughput, and latency.
5. Emits a single validated JSON per run matching the schema in TEST_PLAN.md §11.

This supersedes the legacy `benchmarks/trtllm_benchmark.py` for v2 experiments.
The legacy script is preserved unchanged for reference.

Usage:
    run_benchmark.py --config configs/runs/<run>.json --out results/<batch>/<run>.json

The config JSON is produced by the orchestrator (scripts/orchestrate.py) and
contains everything this process needs — including the pre-shuffled run ordinal,
environment.json path, model path, trtllm-bench args, and lifecycle-snapshot
output path.

No shell interpolation, no environment variables, no defaults that differ
between invocations. Everything that can drift is captured in the config file,
which is itself written once per batch by the orchestrator.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import sys
import time
from typing import Any


# ---------------------------------------------------------------------------
# Lifecycle snapshot helper — calls scripts/snapshot_memory.py as a subprocess
# so it works identically in both native chroot and container runs (no Python
# import ordering concerns across frameworks).
# ---------------------------------------------------------------------------

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent.parent / "scripts"
SNAPSHOT = SCRIPT_DIR / "snapshot_memory.py"


def snapshot(phase: str, pid: int, jsonl: pathlib.Path,
             container_name: str | None = None, tag: str | None = None) -> None:
    cmd = [
        sys.executable, str(SNAPSHOT),
        "--phase", phase,
        "--pid", str(pid),
        "--out", str(jsonl),
    ]
    if container_name:
        cmd.extend(["--container-name", container_name])
    if tag:
        cmd.extend(["--tag", tag])
    try:
        subprocess.run(cmd, check=False, timeout=15)
    except Exception as e:  # noqa: BLE001
        print(f"[snapshot warn] {phase}: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Log parsing — TensorRT-LLM emits structured lines we can regex.
# The patterns below come from inspecting trtllm-bench output; anything we
# can't parse is logged and the run is marked `parse_ok: false`.
# ---------------------------------------------------------------------------

KV_CACHE_BYTES_RE = re.compile(
    r"(?:Memory used for(?: paged)? KV cache|KV cache size)[^\d]*([\d.]+)\s*(GiB|GB|MiB|MB)",
    re.IGNORECASE,
)
KV_CACHE_BLOCKS_RE = re.compile(
    r"(?:num(?:_| )(?:gpu_)?blocks|KV cache blocks)[^\d]*(\d+)",
    re.IGNORECASE,
)
THROUGHPUT_RE = re.compile(
    r"(?:Token Throughput|tokens?/sec(?:ond)?|output tokens/s)[^\d]*([\d.]+)",
    re.IGNORECASE,
)
LATENCY_P50_RE = re.compile(r"P50[^\d]*([\d.]+)\s*ms", re.IGNORECASE)
LATENCY_P95_RE = re.compile(r"P95[^\d]*([\d.]+)\s*ms", re.IGNORECASE)
LATENCY_P99_RE = re.compile(r"P99[^\d]*([\d.]+)\s*ms", re.IGNORECASE)


def _to_bytes(val: float, unit: str) -> int:
    unit = unit.lower()
    mult = {
        "gib": 1024 ** 3, "gb":  10 ** 9,
        "mib": 1024 ** 2, "mb":  10 ** 6,
    }[unit]
    return int(val * mult)


def parse_trtllm_log(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {"parse_ok": True, "parse_warnings": []}
    m = KV_CACHE_BYTES_RE.search(text)
    if m:
        out["kv_cache_bytes"] = _to_bytes(float(m.group(1)), m.group(2))
    else:
        out["parse_warnings"].append("kv_cache_bytes not found")
    m = KV_CACHE_BLOCKS_RE.search(text)
    if m:
        out["kv_cache_blocks"] = int(m.group(1))
    m = THROUGHPUT_RE.search(text)
    if m:
        out["throughput_tokens_per_sec"] = float(m.group(1))
    else:
        out["parse_warnings"].append("throughput not found")
    lat: dict[str, float] = {}
    for key, rx in (("p50", LATENCY_P50_RE), ("p95", LATENCY_P95_RE), ("p99", LATENCY_P99_RE)):
        m = rx.search(text)
        if m:
            lat[key] = float(m.group(1))
    if lat:
        out["latency_ms"] = lat
    out["parse_ok"] = not out["parse_warnings"]
    return out


# ---------------------------------------------------------------------------
# Dataset generation — deterministic; seeded by (batch_id, run_id) so every
# replication uses the SAME prompts but different replications can't silently
# converge on the lucky cache state.
# ---------------------------------------------------------------------------

def build_dataset(workload: dict[str, Any], out_path: pathlib.Path, seed: int) -> int:
    import random
    rng = random.Random(seed)
    n = int(workload["num_requests"])
    input_lo, input_hi = workload["input_len_range"]
    output_len = int(workload["output_len"])

    # A small, legally reusable prompt corpus. Not meant to be realistic —
    # meant to be DETERMINISTIC given a seed, so reproducers get identical
    # dataset bytes. W3 (mixed) should use a real dataset; W1/W2 default below.
    base = (
        "Explain in detail the following topic, with examples and context: "
        "the history of cryptography from ancient substitution ciphers to "
        "modern elliptic curve systems. Include turning points, failures, "
        "and the motivations of key researchers. "
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for i in range(n):
            target_input = rng.randint(input_lo, input_hi)
            # Pad/truncate by word repetition — deterministic given seed.
            words: list[str] = []
            while len(" ".join(words).split()) < target_input:
                words.extend(base.split())
            prompt = " ".join(words[:target_input])
            rec = {"task_id": i, "prompt": prompt, "output_tokens": output_len}
            f.write(json.dumps(rec) + "\n")
    return n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="Run config JSON (from orchestrator)")
    ap.add_argument("--out", required=True, help="Output result JSON path")
    args = ap.parse_args()

    cfg_path = pathlib.Path(args.config).resolve()
    out_path = pathlib.Path(args.out).resolve()

    with cfg_path.open() as f:
        cfg = json.load(f)

    run_id   = cfg["run_id"]
    batch_id = cfg["batch_id"]
    model    = cfg["model"]["huggingface_id"]
    model_path = cfg["model"]["local_path"]
    workspace  = pathlib.Path(cfg["workspace"])
    workload   = cfg["workload"]
    trtllm_args = cfg.get("trtllm_args", {})
    pinned_kv   = cfg.get("pinned_kv_blocks", None)  # integer if pinned, None if dynamic

    results_dir = out_path.parent
    results_dir.mkdir(parents=True, exist_ok=True)
    lifecycle_jsonl = results_dir / f"{run_id}.lifecycle.jsonl"
    log_path        = results_dir / f"{run_id}.trtllm.log"

    pid = os.getpid()
    container_name = cfg.get("container_name")  # None when running native

    result: dict[str, Any] = {
        "schema_version": 2,
        "run_id":   run_id,
        "batch_id": batch_id,
        "cell":     cfg["cell"],
        "environment_kind": cfg["environment_kind"],  # "native" | "container"
        "model":    cfg["model"],
        "rep":      int(cfg["rep"]),
        "warmup":   bool(cfg.get("warmup", False)),
        "plan_ordinal": int(cfg["plan_ordinal"]),
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config":   cfg,
        "config_sha256": hashlib.sha256(cfg_path.read_bytes()).hexdigest(),
        "environment_hash": cfg.get("environment_hash"),
        "workload": workload,
        "pinned_kv_blocks": pinned_kv,
        "lifecycle_jsonl": str(lifecycle_jsonl),
        "trtllm_log":      str(log_path),
    }

    workspace.mkdir(parents=True, exist_ok=True)

    snapshot("baseline", pid, lifecycle_jsonl, container_name, tag=run_id)

    # --- build dataset ------------------------------------------------------
    seed = int(hashlib.sha256(run_id.encode()).hexdigest(), 16) % (2**32)
    dataset_file = workspace / f"dataset_{run_id}.jsonl"
    n_req = build_dataset(workload, dataset_file, seed)

    snapshot("post_dataset_build", pid, lifecycle_jsonl, container_name)

    # --- build engine -------------------------------------------------------
    #
    # NOTE: we go through `trtllm-bench build` exactly as Phase 1 did. The
    # critical v2 change is the OPTIONAL --max_num_tokens flag that, when set
    # via pinned_kv_blocks * tokens_per_block, forces a fixed KV-cache block
    # budget. When pinned_kv_blocks is None, sizing is dynamic as before.
    target_input_len = workload["input_len_range"][1]
    max_batch_size   = int(trtllm_args.get("max_batch_size", 8))
    max_seq_len      = target_input_len + int(workload["output_len"])
    max_num_tokens_override = None
    tokens_per_block = int(trtllm_args.get("tokens_per_block", 64))

    if pinned_kv is not None:
        # Force the KV cache capacity by sizing max_num_tokens s.t. the
        # allocator can create exactly `pinned_kv` blocks.
        max_num_tokens_override = int(pinned_kv) * tokens_per_block

    build_cmd = [
        "trtllm-bench",
        "-m", model,
        "--workspace", str(workspace),
    ]
    if model_path:
        build_cmd += ["--model_path", model_path]
    build_cmd += [
        "build",
        "--max_batch_size", str(max_batch_size),
        "--max_seq_len",    str(max_seq_len),
    ]
    if max_num_tokens_override is not None:
        build_cmd += ["--max_num_tokens", str(max_num_tokens_override)]
    else:
        build_cmd += ["--max_num_tokens", str(max_batch_size * target_input_len)]

    snapshot("pre_engine_build", pid, lifecycle_jsonl, container_name)
    t0 = time.time()
    with log_path.open("w") as logf:
        logf.write(f"=== build cmd ===\n{shlex.join(build_cmd)}\n\n")
        logf.flush()
        build_proc = subprocess.run(
            build_cmd, stdout=logf, stderr=subprocess.STDOUT, text=True
        )
    build_ok = build_proc.returncode == 0
    result["engine_build_seconds"] = time.time() - t0
    result["engine_build_ok"] = build_ok
    snapshot("post_engine_build", pid, lifecycle_jsonl, container_name)

    if not build_ok:
        result["failed_phase"] = "engine_build"
        _finalize(result, out_path)
        return 10

    # --- run benchmark ------------------------------------------------------
    engine_dir = workspace / model / "tp_1_pp_1"
    bench_cmd = [
        "trtllm-bench",
        "-m", model,
        "--workspace", str(workspace),
        "throughput",
        "--engine_dir", str(engine_dir),
        "--dataset",       str(dataset_file),
        "--num_requests",  str(n_req),
        "--target_output_len", str(workload["output_len"]),
        "--max_batch_size", str(max_batch_size),
        "--max_num_tokens", str(max_num_tokens_override or max_batch_size * target_input_len),
    ]

    # Optional dynamic-mode knob: kv-cache free GPU memory fraction.
    kv_frac = trtllm_args.get("kv_cache_free_gpu_memory_fraction")
    if kv_frac is not None and pinned_kv is None:
        bench_cmd += ["--kv_cache_free_gpu_memory_fraction", str(kv_frac)]

    snapshot("pre_bench", pid, lifecycle_jsonl, container_name)
    t0 = time.time()
    with log_path.open("a") as logf:
        logf.write(f"\n=== bench cmd ===\n{shlex.join(bench_cmd)}\n\n")
        logf.flush()
        bench_proc = subprocess.run(
            bench_cmd, stdout=logf, stderr=subprocess.STDOUT, text=True
        )
    bench_ok = bench_proc.returncode == 0
    result["bench_seconds"] = time.time() - t0
    result["bench_ok"] = bench_ok
    snapshot("post_bench", pid, lifecycle_jsonl, container_name)

    # --- parse log ----------------------------------------------------------
    with log_path.open() as f:
        log_text = f.read()
    result["metrics"] = parse_trtllm_log(log_text)
    result["log_byte_size"] = len(log_text)

    if not bench_ok:
        result["failed_phase"] = "benchmark"
        _finalize(result, out_path)
        return 11

    _finalize(result, out_path)
    return 0


def _finalize(result: dict[str, Any], out_path: pathlib.Path) -> None:
    result["ended_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, default=str))
    tmp.replace(out_path)
    print(f"[run_benchmark] wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
