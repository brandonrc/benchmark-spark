#!/usr/bin/env bash
#
# capture_environment.sh
#
# Captures every controlled variable from TEST_PLAN.md section 5 into a single
# JSON file. Run this ONCE per batch, before any benchmark, and keep the output
# alongside the results. Optionally verifies against configs/versions.lock.
#
# Usage:
#   scripts/capture_environment.sh <output_file> [--verify configs/versions.lock]
#
# Exit codes:
#   0  success
#   2  missing required tool
#   3  verification against lockfile failed (see stderr)
#   4  GPU not accessible
#
# All shelling out is contained here so the rest of the benchmark can treat
# environment.json as opaque.

set -euo pipefail

OUT="${1:-environment.json}"
LOCKFILE=""
if [ "${2:-}" = "--verify" ] && [ -n "${3:-}" ]; then
    LOCKFILE="$3"
fi

# --- helpers ---------------------------------------------------------------

need() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "ERROR: required tool not found: $1" >&2
        exit 2
    }
}

json_escape() {
    # stdin → JSON-escaped string (no surrounding quotes)
    python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))'
}

emit_kv() {
    # emit_kv <key> <value-stdin>
    local key="$1"
    local val
    val="$(cat | json_escape)"
    printf '  %s: %s' "\"$key\"" "$val"
}

# --- prereqs ---------------------------------------------------------------

need python3
need nvidia-smi
need uname
need awk
need grep
need sed
need lsb_release || true   # optional

if ! nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi failed — GPU not accessible" >&2
    exit 4
fi

# --- collect ---------------------------------------------------------------

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# Hostname / timestamp / user
HOSTNAME_="$(hostname)"
TIMESTAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
USER_="$(id -un)"

# Kernel / OS
KERNEL="$(uname -srmpio 2>/dev/null || uname -a)"
OS_RELEASE="$(cat /etc/os-release 2>/dev/null || true)"
LSB_RELEASE="$(lsb_release -a 2>/dev/null || true)"

# CPU / NUMA
CPU_INFO="$(lscpu 2>/dev/null || true)"
NUMA_INFO="$(numactl --hardware 2>/dev/null || true)"

# Memory / hugepages
MEMINFO="$(cat /proc/meminfo)"
HUGEPAGES="$(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || true)"
DEFRAG="$(cat /sys/kernel/mm/transparent_hugepage/defrag 2>/dev/null || true)"
ULIMIT="$(bash -c 'ulimit -a')"
VM_SYSCTLS="$(sysctl -a 2>/dev/null | grep -E '^vm\.|^kernel\.shm|^kernel\.sched_|^net\.core\.rmem_' || true)"

# Cgroup version
CGROUP_V=""
if [ -f /sys/fs/cgroup/cgroup.controllers ]; then
    CGROUP_V="v2"
elif [ -d /sys/fs/cgroup/memory ]; then
    CGROUP_V="v1"
else
    CGROUP_V="unknown"
fi

# NVIDIA driver + GPU details
NVSMI_QUERY="$(nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total,persistence_mode,clocks.current.sm,clocks.max.sm,temperature.gpu,power.draw --format=csv)"
NVSMI_TOPO="$(nvidia-smi topo -m 2>/dev/null || true)"

# CUDA toolkit (may not be installed on host; try both locations)
CUDA_VERSION=""
if [ -f /usr/local/cuda/version.json ]; then
    CUDA_VERSION="$(cat /usr/local/cuda/version.json)"
elif [ -f /usr/local/cuda/version.txt ]; then
    CUDA_VERSION="$(cat /usr/local/cuda/version.txt)"
fi
NVCC_V="$(nvcc --version 2>/dev/null || true)"

# NVIDIA Container Toolkit
NCT_VERSION="$(nvidia-container-cli --version 2>/dev/null || true)"
NCT_INFO="$(nvidia-container-cli info 2>/dev/null || true)"
CDI_SPECS=""
if [ -d /etc/cdi ]; then
    CDI_SPECS="$(ls -la /etc/cdi 2>/dev/null || true)"
fi

# Docker / containerd / runc
DOCKER_V="$(docker --version 2>/dev/null || true)"
DOCKER_INFO="$(docker info 2>/dev/null || true)"
CONTAINERD_V="$(containerd --version 2>/dev/null || true)"
RUNC_V="$(runc --version 2>/dev/null || true)"

# Container image digest (for the spark-single-gpu-dev image)
IMAGE="${CONTAINER_IMAGE:-nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev}"
IMAGE_DIGEST="$(docker image inspect --format '{{index .RepoDigests 0}}' "$IMAGE" 2>/dev/null || true)"
IMAGE_ID="$(docker image inspect --format '{{.Id}}' "$IMAGE" 2>/dev/null || true)"
IMAGE_CREATED="$(docker image inspect --format '{{.Created}}' "$IMAGE" 2>/dev/null || true)"

# Extracted rootfs hash (optional)
CONTAINER_ROOTFS="${CONTAINER_ROOTFS:-$HOME/container-rootfs}"
ROOTFS_HASH=""
if [ -d "$CONTAINER_ROOTFS" ]; then
    # hash the list of files + mtimes + sizes (full content hash would take forever)
    ROOTFS_HASH="$(find "$CONTAINER_ROOTFS" -type f -printf '%p %s %T@\n' 2>/dev/null | sort | sha256sum | awk '{print $1}' || true)"
fi

# Git state of this benchmark repo
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GIT_SHA="$(cd "$REPO_ROOT" && git rev-parse HEAD 2>/dev/null || echo unknown)"
GIT_DIRTY="$(cd "$REPO_ROOT" && git status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
GIT_REMOTE="$(cd "$REPO_ROOT" && git remote get-url origin 2>/dev/null || true)"

# --- write JSON ------------------------------------------------------------

python3 - "$OUT" <<PYEOF
import json, os, sys, hashlib, datetime

data = {
    "schema_version": 2,
    "hostname":   ${HOSTNAME_@Q},
    "timestamp":  ${TIMESTAMP@Q},
    "user":       ${USER_@Q},
    "kernel":     ${KERNEL@Q},
    "cgroup_version": ${CGROUP_V@Q},
    "os_release": """${OS_RELEASE}""",
    "lsb_release": """${LSB_RELEASE}""",
    "cpu_info":   """${CPU_INFO}""",
    "numa_info":  """${NUMA_INFO}""",
    "meminfo":    """${MEMINFO}""",
    "transparent_hugepage_enabled": ${HUGEPAGES@Q},
    "transparent_hugepage_defrag":  ${DEFRAG@Q},
    "ulimit":     """${ULIMIT}""",
    "vm_sysctls": """${VM_SYSCTLS}""",
    "nvidia_smi_query": """${NVSMI_QUERY}""",
    "nvidia_smi_topo":  """${NVSMI_TOPO}""",
    "cuda_version":     """${CUDA_VERSION}""",
    "nvcc_version":     """${NVCC_V}""",
    "nvidia_container_toolkit_version": ${NCT_VERSION@Q},
    "nvidia_container_toolkit_info":    """${NCT_INFO}""",
    "cdi_specs":        """${CDI_SPECS}""",
    "docker_version":   ${DOCKER_V@Q},
    "docker_info":      """${DOCKER_INFO}""",
    "containerd_version": ${CONTAINERD_V@Q},
    "runc_version":     ${RUNC_V@Q},
    "container_image":  ${IMAGE@Q},
    "container_image_digest":  ${IMAGE_DIGEST@Q},
    "container_image_id":      ${IMAGE_ID@Q},
    "container_image_created": ${IMAGE_CREATED@Q},
    "container_rootfs_path": ${CONTAINER_ROOTFS@Q},
    "container_rootfs_hash": ${ROOTFS_HASH@Q},
    "git_sha":     ${GIT_SHA@Q},
    "git_dirty_files": int(${GIT_DIRTY:-0} or 0),
    "git_remote":  ${GIT_REMOTE@Q},
}

# Environment hash — stable identifier for this batch's controlled state.
# Hash the critical fields only; full blob includes verbose logs that change
# on every invocation (docker info contains time-since-start etc.).
critical = {
    "kernel": data["kernel"],
    "cgroup_version": data["cgroup_version"],
    "nvidia_smi_query": data["nvidia_smi_query"],
    "nvidia_container_toolkit_version": data["nvidia_container_toolkit_version"],
    "docker_version": data["docker_version"],
    "container_image_digest": data["container_image_digest"],
    "container_rootfs_hash": data["container_rootfs_hash"],
    "git_sha": data["git_sha"],
    "transparent_hugepage_enabled": data["transparent_hugepage_enabled"],
}
data["environment_hash"] = "sha256:" + hashlib.sha256(
    json.dumps(critical, sort_keys=True).encode()
).hexdigest()

out_path = sys.argv[1]
os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
with open(out_path, "w") as f:
    json.dump(data, f, indent=2, sort_keys=True)
print(f"environment captured → {out_path}")
print(f"environment_hash = {data['environment_hash']}")
PYEOF

# --- verify against lockfile (optional) ------------------------------------

if [ -n "$LOCKFILE" ]; then
    if [ ! -f "$LOCKFILE" ]; then
        echo "ERROR: lockfile not found: $LOCKFILE" >&2
        exit 3
    fi
    python3 - "$OUT" "$LOCKFILE" <<'PYEOF'
import json, sys, re

env_path, lock_path = sys.argv[1], sys.argv[2]
with open(env_path) as f:
    env = json.load(f)
with open(lock_path) as f:
    lock = {}
    for line in f:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        lock[k.strip()] = v.strip().strip('"').strip("'")

def check(key, needle, haystack):
    if not needle:
        return True
    if needle in str(haystack):
        return True
    print(f"LOCKFILE MISMATCH on {key}: expected substring '{needle}' in '{haystack}'", file=sys.stderr)
    return False

ok = True
ok &= check("kernel",            lock.get("kernel_contains"),            env.get("kernel", ""))
ok &= check("driver",            lock.get("nvidia_driver_contains"),     env.get("nvidia_smi_query", ""))
ok &= check("cgroup_version",    lock.get("cgroup_version"),             env.get("cgroup_version", ""))
ok &= check("docker_version",    lock.get("docker_version_contains"),    env.get("docker_version", ""))
ok &= check("container_image",   lock.get("container_image"),            env.get("container_image", ""))
ok &= check("container_digest",  lock.get("container_image_digest"),     env.get("container_image_digest", ""))
ok &= check("nct_version",       lock.get("nct_version_contains"),       env.get("nvidia_container_toolkit_version", ""))

if not ok:
    sys.exit(3)
print("lockfile verification OK")
PYEOF
fi
