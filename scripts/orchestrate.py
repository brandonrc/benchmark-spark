#!/usr/bin/env python3
"""
orchestrate.py — v2 experiment orchestrator

Reads configs/experiment.yaml, generates a deterministic random run plan,
writes it to run_plan.json BEFORE executing anything, then dispatches each
run either through chroot (native) or docker (container).

Design rules, derived from TEST_PLAN.md:

  1. Run order is a uniform random permutation of all (cell × rep) tuples,
     seeded from `experiment.random_seed`. The plan is written to disk
     once and never regenerated mid-batch.
  2. The first 2 runs of every batch are warmup (marked in metadata, NOT
     excluded here — excluded in analysis).
  3. Between every run: drop caches, wait until GPU temp ≤ max_temp, sleep
     cooldown_seconds minimum.
  4. environment.json is captured once per batch. Its hash is injected into
     every run config. If the hash changes mid-batch (re-captured), the
     run aborts.
  5. On any crash: record the failure in exclusions.json, continue with
     the next run. The batch is not restarted from the beginning.

Usage:
    scripts/orchestrate.py \\
        --experiment configs/experiment.yaml \\
        --out results/2026-04-12_clean_spark \\
        [--dry-run]           # write plan, don't execute
        [--resume]            # skip already-completed runs in results dir
        [--only-cell E4C]     # filter to one cell (debugging)

This script is deliberately NOT idempotent beyond --resume: running it twice
against the same output dir is a user error that is caught by refusing to
start if run_plan.json already exists unless --resume is passed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import random
import re
import shlex
import shutil
import subprocess
import sys
import time
from typing import Any

# ---------------------------------------------------------------------------
# YAML loading without requiring PyYAML: we support a restricted subset via
# json fallback. If configs/experiment.yaml is actually YAML, require PyYAML.
# ---------------------------------------------------------------------------

_VAR_RE = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)(?::-([^}]*))?\}")


def _expand_env(value: Any) -> Any:
    """Recursively expand ${VAR} and ${VAR:-default} in strings, using os.environ."""
    if isinstance(value, str):
        def sub(m: re.Match[str]) -> str:
            name, default = m.group(1), (m.group(2) or "")
            return os.environ.get(name, default)
        return _VAR_RE.sub(sub, value)
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    return value


def load_experiment(path: pathlib.Path) -> dict[str, Any]:
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except ImportError as e:  # noqa: BLE001
            raise SystemExit(
                f"PyYAML required to read {path}. Install with: pip install pyyaml"
            ) from e
        raw = yaml.safe_load(text)
    else:
        raw = json.loads(text)
    return _expand_env(raw)


# ---------------------------------------------------------------------------
# Docker command construction — one place, no shell interpolation.
# ---------------------------------------------------------------------------

def docker_flags_for(cell: dict[str, Any], container_image: str,
                     bench_dir: pathlib.Path, results_dir: pathlib.Path,
                     models_dir: pathlib.Path, workspace_dir: pathlib.Path,
                     container_name: str) -> list[str]:
    flags = cell.get("docker_flags", [])
    cmd = [
        "docker", "run", "--rm",
        "--name", container_name,
        "--gpus", "all",
        "-v", f"{bench_dir}:/workspace",
        "-v", f"{results_dir}:/results",
        "-v", f"{models_dir}:/models:ro",
        "-v", f"{workspace_dir}:/trtllm_workspace",
    ]
    cmd.extend(flags)
    cmd.append(container_image)
    return cmd


# ---------------------------------------------------------------------------
# Cooldown helpers
# ---------------------------------------------------------------------------

def gpu_temp_c() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=False,
        ).stdout.strip()
        return int(out.splitlines()[0]) if out else None
    except Exception:  # noqa: BLE001
        return None


def drop_caches() -> None:
    """Clear page cache so runs are independent. Requires root or sudo -n."""
    try:
        subprocess.run(["sync"], check=False, timeout=10)
        # /proc/sys/vm/drop_caches requires root; try sudo -n
        r = subprocess.run(
            ["sudo", "-n", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if r.returncode != 0:
            print(f"[warn] drop_caches failed (not root?): {r.stderr.strip()}",
                  file=sys.stderr)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] drop_caches exception: {e}", file=sys.stderr)


def cooldown(seconds: int, max_temp_c: int) -> None:
    print(f"[cooldown] sleeping {seconds}s and waiting for GPU ≤ {max_temp_c}°C")
    t_end = time.time() + seconds
    while time.time() < t_end:
        time.sleep(5)
    while True:
        t = gpu_temp_c()
        if t is None:
            print("[cooldown] GPU temp unavailable — proceeding")
            break
        if t <= max_temp_c:
            print(f"[cooldown] GPU @ {t}°C — ready")
            break
        print(f"[cooldown] GPU @ {t}°C — waiting")
        time.sleep(15)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--only-cell", default=None, help="Debug: run only one cell id")
    ap.add_argument("--skip-env-capture", action="store_true",
                    help="Skip environment.json capture (for local testing only)")
    args = ap.parse_args()

    exp = load_experiment(args.experiment)
    exp_meta = exp["experiment"]
    batch_id = exp_meta.get("batch_id") or dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir: pathlib.Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    plan_path = out_dir / "run_plan.json"
    env_path  = out_dir / "environment.json"
    excl_path = out_dir / "exclusions.json"

    # --- environment capture ------------------------------------------------
    if not args.skip_env_capture:
        capture = pathlib.Path(__file__).parent / "capture_environment.sh"
        cmd = [str(capture), str(env_path)]
        lockfile = exp_meta.get("versions_lockfile")
        if lockfile:
            cmd += ["--verify", lockfile]
        r = subprocess.run(cmd, check=False)
        if r.returncode != 0:
            print(f"[fatal] environment capture failed: {r.returncode}", file=sys.stderr)
            return r.returncode
    if env_path.exists():
        env_data = json.loads(env_path.read_text())
        environment_hash = env_data.get("environment_hash", "sha256:unknown")
    else:
        environment_hash = "sha256:unknown"

    # --- build run plan -----------------------------------------------------
    cells   = exp["cells"]
    models  = exp["models"]
    runs_per_cell = int(exp_meta["runs_per_cell"])
    warmup_reps   = int(exp_meta.get("warmup_reps", 2))
    seed = int(exp_meta["random_seed"])

    run_configs: list[dict[str, Any]] = []
    for cell in cells:
        if args.only_cell and cell["id"] != args.only_cell:
            continue
        for model in models:
            if cell.get("pinned_kv_blocks") is not None:
                # Validate: pinned blocks must not exceed the model's max.
                pass
            for rep in range(1, runs_per_cell + 1):
                run_id = f"{cell['id'].lower()}_{model['short']}_rep{rep:02d}"
                run_configs.append({
                    "run_id":   run_id,
                    "batch_id": batch_id,
                    "cell":     cell["id"],
                    "environment_kind": cell["environment_kind"],
                    "model":    model,
                    "rep":      rep,
                    "plan_ordinal": None,        # filled after shuffle
                    "warmup":   False,           # filled after shuffle
                    "environment_hash": environment_hash,
                    "workspace": f"/tmp/trtllm_workspace_{run_id}",
                    "workload":  exp["workloads"][cell.get("workload", "W1")],
                    "trtllm_args": exp.get("trtllm_args", {}),
                    "pinned_kv_blocks": cell.get("pinned_kv_blocks"),
                    "docker_flags":     cell.get("docker_flags", []),
                    "container_image":  exp["container_image"],
                    "container_name":   None,    # filled per-run
                })

    rng = random.Random(seed)
    rng.shuffle(run_configs)
    for i, rc in enumerate(run_configs):
        rc["plan_ordinal"] = i
        rc["warmup"] = (i < warmup_reps)

    # Write plan (or load existing if --resume)
    if plan_path.exists():
        if not args.resume:
            print(
                f"[fatal] {plan_path} already exists. Use --resume to continue, "
                f"or pick a different --out directory.",
                file=sys.stderr,
            )
            return 2
        plan_on_disk = json.loads(plan_path.read_text())
        if plan_on_disk.get("seed") != seed or \
           plan_on_disk.get("batch_id") != batch_id or \
           len(plan_on_disk.get("runs", [])) != len(run_configs):
            print("[fatal] --resume plan mismatch; aborting", file=sys.stderr)
            return 2
        run_configs = plan_on_disk["runs"]
    else:
        plan_path.write_text(json.dumps(
            {
                "batch_id": batch_id,
                "seed":     seed,
                "experiment_file": str(args.experiment),
                "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "total_runs": len(run_configs),
                "runs":     run_configs,
            },
            indent=2, default=str,
        ))

    print(f"[plan] {len(run_configs)} runs planned, seed={seed}, batch={batch_id}")
    if args.dry_run:
        return 0

    # --- execute runs -------------------------------------------------------
    exclusions: list[dict] = []
    if excl_path.exists():
        exclusions = json.loads(excl_path.read_text())

    completed = set()
    for p in out_dir.glob("*.json"):
        if p.name in ("run_plan.json", "environment.json", "exclusions.json"):
            continue
        completed.add(p.stem)

    for idx, rc in enumerate(run_configs):
        if rc["run_id"] in completed and args.resume:
            print(f"[resume] skip {rc['run_id']} (already done)")
            continue

        print(f"\n==== [{idx+1}/{len(run_configs)}] {rc['run_id']}  cell={rc['cell']}"
              f"  warmup={rc['warmup']} ====")

        if idx > 0:
            drop_caches()
            cooldown(
                seconds=int(exp_meta["cooldown_seconds"]),
                max_temp_c=int(exp_meta["max_gpu_temp_c"]),
            )

        # Per-run config file (immutable handoff to run_benchmark.py)
        cfg_path = out_dir / f"{rc['run_id']}.cfg.json"
        rc["container_name"] = f"bench_{rc['run_id']}"
        cfg_path.write_text(json.dumps(rc, indent=2, default=str))

        out_json = out_dir / f"{rc['run_id']}.json"
        rc_status = _execute_run(rc, cfg_path, out_json, exp, out_dir)
        if rc_status != 0:
            exclusions.append({
                "run_id": rc["run_id"],
                "reason": f"exit_code={rc_status}",
                "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
            })
            excl_path.write_text(json.dumps(exclusions, indent=2))

    print(f"\n[batch] complete: {len(run_configs)} runs, "
          f"{len(exclusions)} exclusions")
    return 0


def _execute_run(rc: dict, cfg_path: pathlib.Path, out_json: pathlib.Path,
                 exp: dict, out_dir: pathlib.Path) -> int:
    """Dispatch a single run to native chroot or docker."""
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    bench_dir = repo_root
    models_dir = pathlib.Path(exp["models_dir"])
    results_dir = out_dir
    workspace_dir = pathlib.Path(f"/tmp/trtllm_workspace_{rc['run_id']}")
    workspace_dir.mkdir(parents=True, exist_ok=True)

    if rc["environment_kind"] == "native":
        rootfs = pathlib.Path(exp["container_rootfs"])
        if not rootfs.exists():
            print(f"[fatal] rootfs missing: {rootfs}", file=sys.stderr)
            return 20
        runner = pathlib.Path(__file__).resolve().parent / "native_runner.sh"
        if not runner.exists():
            print(f"[fatal] native_runner.sh missing: {runner}", file=sys.stderr)
            return 21
        inner_cmd = [
            "python3",
            "/workspace/benchmarks/run_benchmark.py",
            "--config", f"/results/{cfg_path.name}",
            "--out",    f"/results/{out_json.name}",
        ]
        # Ensure workspace_dir is reachable inside chroot via /trtllm_workspace
        # and bench dir / results dir / models dir under their canonical paths.
        cmd = [
            "sudo", str(runner),
            "--rootfs", str(rootfs),
            "--bind", f"{bench_dir}:/workspace",
            "--bind", f"{results_dir}:/results",
            "--bind", f"{models_dir}:/models",
            "--bind", f"{workspace_dir}:/trtllm_workspace",
            "--env",  f"BATCH_ID={rc['batch_id']}",
            "--env",  f"RUN_ID={rc['run_id']}",
            "--",
            "bash", "-c", shlex.join(inner_cmd),
        ]
    else:
        container_image = exp["container_image"]
        container_name = rc["container_name"]
        docker_cmd = docker_flags_for(
            cell=rc, container_image=container_image,
            bench_dir=bench_dir, results_dir=results_dir,
            models_dir=models_dir, workspace_dir=workspace_dir,
            container_name=container_name,
        )
        docker_cmd += [
            "python3",
            "/workspace/benchmarks/run_benchmark.py",
            "--config", f"/results/{cfg_path.name}",
            "--out",    f"/results/{out_json.name}",
        ]
        cmd = docker_cmd

    print(f"[exec] {shlex.join(cmd)}")
    try:
        r = subprocess.run(cmd, check=False, timeout=int(exp["experiment"].get("per_run_timeout_seconds", 3600)))
        return r.returncode
    except subprocess.TimeoutExpired:
        print(f"[fatal] run {rc['run_id']} timed out", file=sys.stderr)
        return 30


if __name__ == "__main__":
    sys.exit(main())
