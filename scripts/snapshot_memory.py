#!/usr/bin/env python3
"""
snapshot_memory.py

Captures a single memory lifecycle snapshot: /proc/meminfo, nvidia-smi free
memory, cgroup memory.current/peak, target PID smaps_rollup, and docker stats
(if the target lives in a container).

This is called at each lifecycle phase defined in TEST_PLAN.md section 6 so
we can see WHERE free memory diverges between native and container runs —
which is the key diagnostic for the probe-artifact hypothesis.

Usage:
    python3 snapshot_memory.py --phase <name> --pid <pid> --out <jsonl_path> [--container-name <name>]

Appends a single JSON line to <jsonl_path>. Safe to call many times per run.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import subprocess
import sys


def _run(cmd: list[str], timeout: int = 5) -> str:
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        ).stdout
    except Exception as e:  # noqa: BLE001
        return f"__error__: {e}"


def read_meminfo() -> dict[str, int]:
    """Return /proc/meminfo as {field: kB_int}."""
    out: dict[str, int] = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                parts = line.split(":", 1)
                if len(parts) != 2:
                    continue
                k = parts[0].strip()
                v = parts[1].strip().split()
                if v and v[0].isdigit():
                    out[k] = int(v[0])  # kB for most fields
    except OSError:
        pass
    return out


def read_smaps_rollup(pid: int | None) -> dict[str, int]:
    """Process-level memory summary. Requires /proc/<pid>/smaps_rollup."""
    if pid is None or pid <= 0:
        return {}
    path = f"/proc/{pid}/smaps_rollup"
    out: dict[str, int] = {}
    try:
        with open(path) as f:
            for line in f:
                parts = line.split(":", 1)
                if len(parts) != 2:
                    continue
                k = parts[0].strip()
                v = parts[1].strip().split()
                if v and v[0].isdigit():
                    out[k] = int(v[0])  # kB
    except OSError:
        pass
    return out


def read_status_vm(pid: int | None) -> dict[str, int]:
    """VmRSS, VmPeak, VmSize, VmHWM from /proc/<pid>/status."""
    if pid is None or pid <= 0:
        return {}
    out: dict[str, int] = {}
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith(("VmRSS:", "VmPeak:", "VmSize:", "VmHWM:", "VmData:", "VmSwap:")):
                    k, _, v = line.partition(":")
                    toks = v.strip().split()
                    if toks and toks[0].isdigit():
                        out[k.strip()] = int(toks[0])  # kB
    except OSError:
        pass
    return out


def read_cgroup_memory(pid: int | None) -> dict[str, int | str]:
    """Read cgroup v2 memory.current / memory.peak / memory.max for the pid's cgroup."""
    if pid is None or pid <= 0:
        return {}
    out: dict[str, int | str] = {}
    try:
        with open(f"/proc/{pid}/cgroup") as f:
            line = f.readline().strip()
        # cgroup v2 line looks like: "0::/system.slice/docker-<id>.scope"
        parts = line.split("::", 1)
        if len(parts) != 2:
            return {"_parse_error": line}
        rel = parts[1].lstrip("/")
        cg_path = pathlib.Path("/sys/fs/cgroup") / rel
        out["_cgroup_path"] = str(cg_path)
        for name in ("memory.current", "memory.peak", "memory.max",
                     "memory.swap.current", "memory.stat"):
            p = cg_path / name
            try:
                data = p.read_text().strip()
                if name == "memory.stat":
                    stat: dict[str, int] = {}
                    for sline in data.splitlines():
                        toks = sline.split()
                        if len(toks) == 2 and toks[1].lstrip("-").isdigit():
                            stat[toks[0]] = int(toks[1])
                    out[name] = stat  # type: ignore[assignment]
                else:
                    try:
                        out[name] = int(data)
                    except ValueError:
                        out[name] = data  # 'max'
            except OSError:
                pass
    except OSError as e:
        out["_error"] = str(e)
    return out


def read_nvidia_smi() -> dict[str, int | float | str]:
    """Single snapshot of GPU memory/util/temp/clocks."""
    fields = (
        "memory.total,memory.used,memory.free,"
        "utilization.gpu,utilization.memory,"
        "temperature.gpu,power.draw,"
        "clocks.current.sm,clocks.max.sm"
    )
    raw = _run(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"]
    )
    if not raw or raw.startswith("__error__"):
        return {"_raw": raw}
    row = raw.strip().splitlines()[0]
    vals = [v.strip() for v in row.split(",")]
    keys = fields.split(",")
    out: dict[str, int | float | str] = {}
    for k, v in zip(keys, vals):
        # try numeric
        try:
            if "." in v:
                out[k] = float(v)
            else:
                out[k] = int(v)
        except ValueError:
            out[k] = v
    return out


def read_nvidia_smi_pids() -> list[dict]:
    """Per-PID GPU memory accounting. Crucial for distinguishing framework RSS
    from `nvidia-smi memory.used` double-counting on UMA."""
    raw = _run([
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ])
    if not raw or raw.startswith("__error__"):
        return []
    out: list[dict] = []
    for line in raw.strip().splitlines():
        toks = [t.strip() for t in line.split(",")]
        if len(toks) < 3:
            continue
        try:
            out.append({"pid": int(toks[0]), "name": toks[1], "used_mb": int(toks[2])})
        except ValueError:
            continue
    return out


def read_docker_stats(container_name: str | None) -> dict | None:
    if not container_name:
        return None
    raw = _run([
        "docker", "stats", "--no-stream", "--format",
        "{{json .}}",
        container_name,
    ], timeout=10)
    if not raw or raw.startswith("__error__"):
        return {"_error": raw}
    try:
        return json.loads(raw.strip().splitlines()[0])
    except Exception as e:  # noqa: BLE001
        return {"_parse_error": str(e), "_raw": raw}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", required=True, help="Lifecycle phase label")
    ap.add_argument("--pid", type=int, default=None,
                    help="Target PID to read smaps_rollup / cgroup from")
    ap.add_argument("--container-name", default=None,
                    help="Docker container name for docker stats (optional)")
    ap.add_argument("--out", required=True, help="JSONL file to append to")
    ap.add_argument("--tag", default=None, help="Arbitrary tag stored with snapshot")
    args = ap.parse_args()

    snapshot = {
        "ts":    dt.datetime.now(dt.timezone.utc).isoformat(),
        "phase": args.phase,
        "tag":   args.tag,
        "pid":   args.pid,
        "meminfo_kB":    read_meminfo(),
        "proc_status":   read_status_vm(args.pid),
        "smaps_rollup":  read_smaps_rollup(args.pid),
        "cgroup_memory": read_cgroup_memory(args.pid),
        "nvidia_smi":    read_nvidia_smi(),
        "nvidia_smi_pids": read_nvidia_smi_pids(),
        "docker_stats":  read_docker_stats(args.container_name),
    }

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a") as f:
        f.write(json.dumps(snapshot, default=str))
        f.write("\n")

    # Short human line on stdout so orchestrator logs show progress
    mi = snapshot["meminfo_kB"]
    nv = snapshot["nvidia_smi"]
    free_sys_gib = mi.get("MemAvailable", 0) / 1024 / 1024
    free_gpu_mib = nv.get("memory.free", 0)
    print(
        f"[snapshot {args.phase}] sys MemAvailable={free_sys_gib:.2f} GiB "
        f"nvsmi_free={free_gpu_mib} MiB pid={args.pid}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
