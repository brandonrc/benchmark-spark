#!/usr/bin/env bash
#
# preflight.sh — verify a clean DGX Spark is ready for a benchmark batch.
#
# Runs a tiered battery of checks. Any FAIL will abort the batch later, so
# catching them here (in seconds) saves you hours of wasted GPU time.
#
# Exit codes:
#   0  all checks passed
#   1  one or more hard FAILs (must fix)
#   2  one or more soft WARNs (recommended but not blocking)
#
# Usage:
#     scripts/preflight.sh [--experiment configs/experiment.yaml]

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP_FILE="${REPO_ROOT}/configs/experiment.yaml"

while [ $# -gt 0 ]; do
    case "$1" in
        --experiment) EXP_FILE="$2"; shift 2 ;;
        --help|-h) sed -n '2,15p' "$0" >&2; exit 2 ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done

BENCH_CONTAINER_IMAGE="${BENCH_CONTAINER_IMAGE:-nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev}"
BENCH_CONTAINER_ROOTFS="${BENCH_CONTAINER_ROOTFS:-$HOME/container-rootfs}"
BENCH_MODELS_DIR="${BENCH_MODELS_DIR:-/data/models/huggingface}"

GREEN="\033[0;32m"; RED="\033[0;31m"; YELLOW="\033[1;33m"; BLUE="\033[0;34m"; NC="\033[0m"

FAILS=0
WARNS=0

section() { printf "\n${BLUE}=== %s ===${NC}\n" "$*"; }
ok()      { printf "${GREEN}  ✓${NC} %s\n" "$*"; }
fail()    { printf "${RED}  ✗ FAIL${NC} %s\n" "$*"; FAILS=$((FAILS+1)); }
warn()    { printf "${YELLOW}  ! WARN${NC} %s\n" "$*"; WARNS=$((WARNS+1)); }
info()    { printf "    %s\n" "$*"; }

need() {
    # need <description> <command...>
    local desc="$1"; shift
    if "$@" >/dev/null 2>&1; then
        ok "$desc"
    else
        fail "$desc"
    fi
}

# -------------------------------------------------------------------------

section "Hardware & OS"

if [ "$(uname -m)" = "aarch64" ]; then
    ok "aarch64 CPU"
else
    fail "expected aarch64, got $(uname -m)"
fi

if [ -r /etc/os-release ] && grep -q "24.04" /etc/os-release; then
    ok "Ubuntu 24.04"
else
    warn "not Ubuntu 24.04 — TEST_PLAN.md expects 24.04"
fi

if [ -f /sys/fs/cgroup/cgroup.controllers ]; then
    if grep -q memory /sys/fs/cgroup/cgroup.controllers; then
        ok "cgroup v2 with memory controller"
    else
        fail "cgroup v2 present but memory controller missing"
    fi
else
    fail "cgroup v2 not detected"
fi

# GPU access
if nvidia-smi >/dev/null 2>&1; then
    ok "nvidia-smi works"
    NVSMI="$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1)"
    info "$NVSMI"
else
    fail "nvidia-smi failed"
fi

# Free disk
if [ -d /data ]; then
    FREE_GB="$(df -BG /data | awk 'NR==2{print $4}' | tr -d 'G')"
    if [ "$FREE_GB" -ge 300 ]; then
        ok "/data has ${FREE_GB}G free"
    else
        warn "/data only has ${FREE_GB}G free (≥ 300 recommended for weights + results)"
    fi
fi

# -------------------------------------------------------------------------

section "Required tools"

need "docker CLI"                command -v docker
need "nvidia-container-cli"      command -v nvidia-container-cli
need "python3 ≥ 3.10"            python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"
need "git"                       command -v git
need "jq"                        command -v jq
need "curl"                      command -v curl
need "huggingface-cli"           command -v huggingface-cli

# python deps (on host — venv must be activated)
for mod in numpy scipy yaml pandas; do
    if python3 -c "import $mod" >/dev/null 2>&1; then
        ok "python: $mod"
    else
        warn "python: $mod missing (activate bench venv: source ~/bench-venv/bin/activate)"
    fi
done

# -------------------------------------------------------------------------

section "Docker runtime"

if docker info >/dev/null 2>&1; then
    ok "docker daemon reachable"
else
    fail "docker info failed (not in docker group? run: newgrp docker)"
fi

if docker run --rm --gpus all "$BENCH_CONTAINER_IMAGE" nvidia-smi >/dev/null 2>&1; then
    ok "nvidia container runtime works (--gpus all OK)"
else
    fail "docker + --gpus all failed (check nvidia-container-toolkit install)"
fi

# Image pulled
if docker image inspect "$BENCH_CONTAINER_IMAGE" >/dev/null 2>&1; then
    ok "container image present: $BENCH_CONTAINER_IMAGE"
    DIGEST="$(docker image inspect --format '{{index .RepoDigests 0}}' "$BENCH_CONTAINER_IMAGE" 2>/dev/null || echo '')"
    info "digest: $DIGEST"
else
    fail "container image not pulled — run: docker pull $BENCH_CONTAINER_IMAGE"
fi

# -------------------------------------------------------------------------

section "Native chroot (rootfs)"

if [ -d "$BENCH_CONTAINER_ROOTFS" ] && [ -x "$BENCH_CONTAINER_ROOTFS/usr/bin/python3" ]; then
    ok "rootfs populated at $BENCH_CONTAINER_ROOTFS"
else
    fail "rootfs missing or incomplete — run: scripts/extract_container_rootfs.sh"
fi

for d in workspace results models trtllm_workspace; do
    if [ -d "$BENCH_CONTAINER_ROOTFS/$d" ]; then
        ok "bind-mount target: $BENCH_CONTAINER_ROOTFS/$d"
    else
        warn "bind-mount target missing: $BENCH_CONTAINER_ROOTFS/$d (bootstrap_clean_spark.sh creates these)"
    fi
done

# -------------------------------------------------------------------------

section "Sudoers / drop_caches"

if sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches' >/dev/null 2>&1; then
    ok "passwordless drop_caches works"
else
    fail "passwordless drop_caches NOT configured — rerun scripts/bootstrap_clean_spark.sh, or batch will be unusably noisy"
fi

if sudo -n true >/dev/null 2>&1; then
    ok "passwordless sudo cached"
else
    warn "sudo credentials not cached — run: sudo -v (orchestrator will prompt on native runs)"
fi

# -------------------------------------------------------------------------

section "Models"

if [ -f "$EXP_FILE" ]; then
    ok "experiment file: $EXP_FILE"
else
    fail "experiment file missing: $EXP_FILE"
fi

MODELS_OK=1
for path in \
    "$BENCH_MODELS_DIR/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
    "$BENCH_MODELS_DIR/Qwen/Qwen2.5-72B-Instruct" \
; do
    if [ -d "$path" ] && [ -n "$(ls -A "$path" 2>/dev/null)" ]; then
        SIZE="$(du -sh "$path" 2>/dev/null | awk '{print $1}')"
        ok "model: $path ($SIZE)"
    else
        fail "model missing or empty: $path"
        MODELS_OK=0
    fi
done
if [ "$MODELS_OK" = "0" ]; then
    info "hint: huggingface-cli download <id> --local-dir <path>"
fi

# -------------------------------------------------------------------------

section "GPU state"

if command -v nvidia-smi >/dev/null 2>&1; then
    PERSIST="$(nvidia-smi --query-gpu=persistence_mode --format=csv,noheader | head -1)"
    if [ "$PERSIST" = "Enabled" ]; then
        ok "persistence mode: Enabled"
    else
        warn "persistence mode: $PERSIST — recommended: sudo nvidia-smi --persistence-mode=1"
    fi

    CUR="$(nvidia-smi --query-gpu=clocks.current.sm --format=csv,noheader,nounits | head -1)"
    MAX="$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits | head -1)"
    if [ "$CUR" = "$MAX" ] && [ -n "$CUR" ]; then
        ok "GPU clock locked at ${CUR} MHz"
    else
        warn "GPU clock NOT locked (cur=$CUR MHz, max=$MAX MHz) — recommended: sudo nvidia-smi --lock-gpu-clocks=$MAX,$MAX"
    fi

    TEMP="$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1)"
    if [ -n "$TEMP" ] && [ "$TEMP" -le 50 ]; then
        ok "GPU temp: ${TEMP}°C (cool start)"
    else
        warn "GPU temp: ${TEMP}°C (let it cool below 45°C before batch start)"
    fi

    # Check for other processes on the GPU
    NPROC="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)"
    if [ "$NPROC" = "0" ]; then
        ok "no other GPU processes"
    else
        fail "$NPROC process(es) already running on the GPU — kill them before starting"
        nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>&1 | sed 's/^/    /'
    fi
fi

# -------------------------------------------------------------------------

section "Versions lockfile"

LOCK="$REPO_ROOT/configs/versions.lock"
if [ -f "$LOCK" ]; then
    if "$REPO_ROOT/scripts/capture_environment.sh" /tmp/preflight_env.$$.json --verify "$LOCK" >/dev/null 2>&1; then
        ok "environment matches versions.lock"
    else
        fail "environment mismatches versions.lock — run capture_environment.sh with --verify for details"
    fi
    rm -f /tmp/preflight_env.$$.json
else
    warn "configs/versions.lock missing — create one and pin the stack"
fi

# -------------------------------------------------------------------------

section "Summary"

if [ "$FAILS" = "0" ] && [ "$WARNS" = "0" ]; then
    printf "${GREEN}  ALL CHECKS PASSED${NC} — you're ready to run the batch.\n"
    exit 0
elif [ "$FAILS" = "0" ]; then
    printf "${YELLOW}  $WARNS warning(s)${NC}, 0 failures — batch will run but with caveats.\n"
    exit 2
else
    printf "${RED}  $FAILS failure(s)${NC}, $WARNS warning(s) — fix the failures before running a batch.\n"
    exit 1
fi
