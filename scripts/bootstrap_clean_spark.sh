#!/usr/bin/env bash
#
# bootstrap_clean_spark.sh — one-shot setup for a clean DGX Spark.
#
# This script gets a fresh machine to the point where you can run:
#
#     scripts/preflight.sh                 # sanity check
#     scripts/orchestrate.py ...           # start a batch
#
# What it does (idempotent — safe to re-run):
#
#   1. Verifies you're on aarch64 + Ubuntu 24.04 + have nvidia-smi
#   2. Installs pinned apt packages (docker.io, nvidia-container-toolkit, jq, python3-venv)
#   3. Enables + starts the docker daemon
#   4. Adds the invoking user to the `docker` group
#   5. Installs a sudoers drop-in so orchestrate.py can call
#      `sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches'` without a password
#      (REQUIRED for run-independence; see TEST_PLAN.md §8)
#   6. Pulls the TensorRT-LLM container image and prints its digest
#      (you must paste this digest into configs/versions.lock)
#   7. Runs scripts/extract_container_rootfs.sh to populate $BENCH_CONTAINER_ROOTFS
#   8. Creates the results and models directories
#   9. Installs Python host-side dependencies into a venv at ~/bench-venv
#
# What it does NOT do (you have to do these yourself):
#
#   - Download HuggingFace model weights (needs your HF token; see REPRODUCING.md §5)
#   - Pin huggingface_revision in configs/experiment.yaml (record after download)
#   - Lock GPU clocks (see REPRODUCING.md §7)
#   - Edit configs/versions.lock with the image digest you just pulled
#
# Usage:
#     sudo -v   # cache credentials once
#     bash scripts/bootstrap_clean_spark.sh
#
# Environment variables you can override:
#     BENCH_CONTAINER_IMAGE   default: nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev
#     BENCH_CONTAINER_ROOTFS  default: $HOME/container-rootfs
#     BENCH_MODELS_DIR        default: /data/models/huggingface
#     BENCH_RESULTS_DIR       default: $HOME/benchmark-spark/results
#     BENCH_VENV              default: $HOME/bench-venv
#     SKIP_APT=1              skip apt install (debugging)
#     SKIP_IMAGE_PULL=1       skip docker pull (if air-gapped)
#     SKIP_ROOTFS=1           skip rootfs extract
#     SKIP_VENV=1             skip venv setup
#

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BENCH_CONTAINER_IMAGE="${BENCH_CONTAINER_IMAGE:-nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev}"
BENCH_CONTAINER_ROOTFS="${BENCH_CONTAINER_ROOTFS:-$HOME/container-rootfs}"
BENCH_MODELS_DIR="${BENCH_MODELS_DIR:-/data/models/huggingface}"
BENCH_RESULTS_DIR="${BENCH_RESULTS_DIR:-$REPO_ROOT/results}"
BENCH_VENV="${BENCH_VENV:-$HOME/bench-venv}"

SKIP_APT="${SKIP_APT:-0}"
SKIP_IMAGE_PULL="${SKIP_IMAGE_PULL:-0}"
SKIP_ROOTFS="${SKIP_ROOTFS:-0}"
SKIP_VENV="${SKIP_VENV:-0}"

BLUE="\033[0;34m"; GREEN="\033[0;32m"; YELLOW="\033[1;33m"; RED="\033[0;31m"; NC="\033[0m"

say()  { printf "${BLUE}==>${NC} %s\n" "$*"; }
ok()   { printf "${GREEN}✓${NC} %s\n" "$*"; }
warn() { printf "${YELLOW}!${NC} %s\n" "$*"; }
die()  { printf "${RED}ERROR${NC}: %s\n" "$*" >&2; exit 1; }

# -------------------------------------------------------------------------
# 0. Sanity
# -------------------------------------------------------------------------

say "Step 0/9 — sanity checks"

[ "$(uname -m)" = "aarch64" ] || die "expected aarch64, got $(uname -m)"
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found — is this a DGX Spark?"
if ! nvidia-smi >/dev/null 2>&1; then die "nvidia-smi failed — GPU not accessible"; fi
if [ -r /etc/os-release ]; then
    . /etc/os-release
    case "$VERSION_ID" in
        24.04) ok "Ubuntu $VERSION_ID detected" ;;
        *) warn "Ubuntu $VERSION_ID detected; test plan expects 24.04" ;;
    esac
fi
ok "GPU accessible: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# -------------------------------------------------------------------------
# 1. APT install pinned packages
# -------------------------------------------------------------------------

if [ "$SKIP_APT" = "1" ]; then
    warn "SKIP_APT=1, skipping apt install"
else
    say "Step 1/9 — installing apt packages"
    sudo apt-get update -y
    sudo apt-get install -y \
        docker.io \
        nvidia-container-toolkit \
        jq \
        python3-pip \
        python3-venv \
        curl \
        git
    ok "apt packages installed"
fi

# -------------------------------------------------------------------------
# 2. Docker daemon
# -------------------------------------------------------------------------

say "Step 2/9 — docker daemon"
sudo systemctl enable docker >/dev/null 2>&1 || true
sudo systemctl start docker
sudo systemctl is-active docker >/dev/null || die "docker failed to start"

# Configure NVIDIA Container Toolkit runtime if not already done
if ! docker info 2>/dev/null | grep -q "nvidia"; then
    say "  configuring nvidia-ctk runtime"
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
fi
ok "docker: $(docker --version)"
ok "nvidia-container-cli: $(nvidia-container-cli --version 2>&1 | head -1)"

# -------------------------------------------------------------------------
# 3. Docker group membership
# -------------------------------------------------------------------------

say "Step 3/9 — docker group membership"
if id -nG "$USER" | grep -qw docker; then
    ok "$USER already in docker group"
else
    sudo usermod -aG docker "$USER"
    warn "$USER added to docker group — you must log out and back in before docker commands work without sudo"
fi

# -------------------------------------------------------------------------
# 4. Sudoers drop-in for drop_caches
# -------------------------------------------------------------------------

say "Step 4/9 — sudoers drop-in for drop_caches"
SUDOERS_FILE="/etc/sudoers.d/99-benchmark-spark-drop-caches"
DROP_LINE="$USER ALL=(ALL) NOPASSWD: /usr/bin/sh -c echo\\ 3\\ >\\ /proc/sys/vm/drop_caches"
if [ -f "$SUDOERS_FILE" ] && sudo grep -qF "$DROP_LINE" "$SUDOERS_FILE" 2>/dev/null; then
    ok "sudoers drop-in already installed"
else
    echo "# Installed by benchmark-spark bootstrap_clean_spark.sh" \
        | sudo tee "$SUDOERS_FILE" >/dev/null
    echo "$DROP_LINE" | sudo tee -a "$SUDOERS_FILE" >/dev/null
    sudo chmod 0440 "$SUDOERS_FILE"
    if sudo visudo -c -f "$SUDOERS_FILE" >/dev/null; then
        ok "sudoers drop-in installed at $SUDOERS_FILE"
    else
        sudo rm -f "$SUDOERS_FILE"
        die "sudoers drop-in failed visudo check; refusing to leave broken sudoers"
    fi
fi

# -------------------------------------------------------------------------
# 5. Pull container image
# -------------------------------------------------------------------------

if [ "$SKIP_IMAGE_PULL" = "1" ]; then
    warn "SKIP_IMAGE_PULL=1, skipping"
else
    say "Step 5/9 — pulling container image"
    say "  image: $BENCH_CONTAINER_IMAGE"
    docker pull "$BENCH_CONTAINER_IMAGE" || die "docker pull failed"
    DIGEST="$(docker image inspect --format '{{index .RepoDigests 0}}' "$BENCH_CONTAINER_IMAGE" 2>/dev/null || echo '')"
    if [ -n "$DIGEST" ]; then
        ok "image digest: $DIGEST"
        echo
        warn "ACTION REQUIRED: paste this digest into configs/versions.lock"
        warn "    container_image_digest = \"$DIGEST\""
        echo
    fi
fi

# -------------------------------------------------------------------------
# 6. Extract container rootfs for the native chroot side
# -------------------------------------------------------------------------

if [ "$SKIP_ROOTFS" = "1" ]; then
    warn "SKIP_ROOTFS=1, skipping"
else
    say "Step 6/9 — extracting container rootfs to $BENCH_CONTAINER_ROOTFS"
    if [ -d "$BENCH_CONTAINER_ROOTFS" ] && [ -f "$BENCH_CONTAINER_ROOTFS/usr/bin/python3" ]; then
        ok "rootfs already extracted at $BENCH_CONTAINER_ROOTFS"
    else
        CONTAINER_IMAGE="$BENCH_CONTAINER_IMAGE" \
        EXTRACT_DIR="$BENCH_CONTAINER_ROOTFS" \
            bash "$REPO_ROOT/scripts/extract_container_rootfs.sh"
    fi
    # Pre-create bind-mount targets so native_runner.sh's --bind just works
    sudo mkdir -p \
        "$BENCH_CONTAINER_ROOTFS/workspace" \
        "$BENCH_CONTAINER_ROOTFS/results" \
        "$BENCH_CONTAINER_ROOTFS/models" \
        "$BENCH_CONTAINER_ROOTFS/trtllm_workspace"
    ok "rootfs bind-mount targets created"
fi

# -------------------------------------------------------------------------
# 7. Create directories
# -------------------------------------------------------------------------

say "Step 7/9 — creating directories"
mkdir -p "$BENCH_RESULTS_DIR"
sudo mkdir -p "$BENCH_MODELS_DIR"
# Make the models dir owned by the user so huggingface-cli download works
# without sudo. Safe — it's a dedicated models cache.
sudo chown "$USER:$USER" "$BENCH_MODELS_DIR" || true
ok "results dir: $BENCH_RESULTS_DIR"
ok "models dir:  $BENCH_MODELS_DIR"

# -------------------------------------------------------------------------
# 8. Python venv for host-side orchestration / analysis
# -------------------------------------------------------------------------

if [ "$SKIP_VENV" = "1" ]; then
    warn "SKIP_VENV=1, skipping"
else
    say "Step 8/9 — python venv at $BENCH_VENV"
    if [ ! -d "$BENCH_VENV" ]; then
        python3 -m venv "$BENCH_VENV"
    fi
    # shellcheck source=/dev/null
    source "$BENCH_VENV/bin/activate"
    pip install -q -U pip
    pip install -q -r "$REPO_ROOT/requirements.txt"
    pip install -q huggingface_hub
    deactivate
    ok "venv ready — activate with: source $BENCH_VENV/bin/activate"
fi

# -------------------------------------------------------------------------
# 9. Final instructions
# -------------------------------------------------------------------------

say "Step 9/9 — next steps"
cat <<NEXT

Bootstrap complete. To finish the setup:

  1. If you were added to the docker group for the first time, LOG OUT
     and back in (or run: newgrp docker) so docker commands work without sudo.

  2. Activate the Python venv:
       source $BENCH_VENV/bin/activate

  3. Edit configs/versions.lock and paste the container_image_digest
     printed above.

  4. Download model weights (requires HF token):
       huggingface-cli login
       huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \\
           --local-dir $BENCH_MODELS_DIR/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
       huggingface-cli download Qwen/Qwen2.5-72B-Instruct \\
           --local-dir $BENCH_MODELS_DIR/Qwen/Qwen2.5-72B-Instruct

  5. Record the HF commit SHAs into configs/experiment.yaml
     (huggingface_revision: <sha> for each model).

  6. Lock GPU clocks for noise reduction:
       sudo nvidia-smi --persistence-mode=1
       sudo nvidia-smi --lock-gpu-clocks=1950,1950

  7. Run the preflight check (MANDATORY before a batch):
       scripts/preflight.sh

  8. Dry-run the orchestrator to inspect the plan:
       scripts/orchestrate.py --experiment configs/experiment.yaml \\
           --out /tmp/dry --dry-run --skip-env-capture

  9. Start the real batch:
       scripts/orchestrate.py --experiment configs/experiment.yaml \\
           --out results/\$(date +%F)_clean_spark

See REPRODUCING.md for the complete runbook.
NEXT
