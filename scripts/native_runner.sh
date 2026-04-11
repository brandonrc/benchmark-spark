#!/usr/bin/env bash
#
# native_runner.sh — run a command inside the extracted TensorRT-LLM rootfs
# with user-supplied bind mounts and env vars.
#
# This is the v2 replacement for the original run_in_rootfs.sh shipped by
# extract_container_rootfs.sh. The v1 runner hard-codes mounts and doesn't
# accept --bind / --env, so orchestrate.py cannot use it directly.
#
# Usage:
#     sudo scripts/native_runner.sh \
#         --rootfs /home/you/container-rootfs \
#         --bind /path/on/host:/path/in/chroot \
#         --bind /another/host/path:/another/chroot/path \
#         --env  KEY=VALUE \
#         --env  ANOTHER=VALUE \
#         -- \
#         bash -c 'command to run inside chroot'
#
# Notes:
#   - Requires root (mounts + chroot).
#   - Always mounts /proc, /sys (rbind), /dev (rbind), /dev/pts into the rootfs.
#   - Cleans up ALL its mounts on exit, even on crash. It is safe to re-run
#     after an interrupted batch; existing mounts are detected and reused.
#   - The CUDA / TensorRT / Python env vars are set to match the container
#     image. They can be overridden with --env.

set -euo pipefail

# ---- defaults -------------------------------------------------------------

ROOTFS="${CONTAINER_ROOTFS:-$HOME/container-rootfs}"
BIND_MOUNTS=()          # array of "host:chroot" strings
EXTRA_ENVS=()           # array of "KEY=VALUE" strings
CMD=()                  # command to run inside chroot
DRY_RUN="0"

# ---- arg parse ------------------------------------------------------------

usage() {
    sed -n '2,27p' "$0" >&2
    exit 2
}

while [ $# -gt 0 ]; do
    case "$1" in
        --rootfs)
            ROOTFS="$2"; shift 2 ;;
        --bind)
            BIND_MOUNTS+=("$2"); shift 2 ;;
        --env)
            EXTRA_ENVS+=("$2"); shift 2 ;;
        --dry-run)
            DRY_RUN="1"; shift ;;
        --help|-h)
            usage ;;
        --)
            shift; CMD=("$@"); break ;;
        *)
            echo "native_runner.sh: unknown flag: $1" >&2
            usage ;;
    esac
done

if [ ${#CMD[@]} -eq 0 ]; then
    echo "native_runner.sh: missing command after --" >&2
    usage
fi

if [ ! -d "$ROOTFS" ]; then
    echo "native_runner.sh: rootfs not found: $ROOTFS" >&2
    exit 3
fi

if [ "$EUID" -ne 0 ] && [ "$DRY_RUN" -eq 0 ]; then
    echo "native_runner.sh: must run as root (use sudo)" >&2
    exit 4
fi

# ---- mount bookkeeping ----------------------------------------------------
# We track every mount we create so we can unmount them in reverse order
# on EXIT.  Existing mounts (from a prior interrupted run) are NOT unmounted
# so concurrent runs in other shells don't get broken.

CREATED_MOUNTS=()

cleanup() {
    local rc=$?
    set +e
    for (( idx=${#CREATED_MOUNTS[@]}-1 ; idx>=0 ; idx-- )); do
        local mp="${CREATED_MOUNTS[$idx]}"
        umount "$mp" 2>/dev/null || umount -l "$mp" 2>/dev/null || true
    done
    exit $rc
}
trap cleanup EXIT INT TERM

ensure_mount() {
    # ensure_mount <source> <mode> <target>
    # mode ∈ {proc, rbind, bind}
    local src="$1"
    local mode="$2"
    local tgt="$3"

    mkdir -p "$tgt"
    if mountpoint -q "$tgt"; then
        # already mounted — don't touch or track
        return 0
    fi

    case "$mode" in
        proc)
            mount -t proc /proc "$tgt" ;;
        rbind)
            mount --rbind "$src" "$tgt"
            mount --make-rslave "$tgt" ;;
        bind)
            mount --bind "$src" "$tgt" ;;
        *)
            echo "native_runner.sh: bad mount mode: $mode" >&2
            return 1 ;;
    esac
    CREATED_MOUNTS+=("$tgt")
}

# ---- base mounts ----------------------------------------------------------

if [ "$DRY_RUN" -eq 0 ]; then
    ensure_mount /proc    proc  "$ROOTFS/proc"
    ensure_mount /sys     rbind "$ROOTFS/sys"
    ensure_mount /dev     rbind "$ROOTFS/dev"
    ensure_mount /dev/pts rbind "$ROOTFS/dev/pts"
    ensure_mount /run     rbind "$ROOTFS/run"
fi

# ---- user bind mounts -----------------------------------------------------

for spec in "${BIND_MOUNTS[@]}"; do
    host="${spec%%:*}"
    inner="${spec#*:}"
    if [ -z "$host" ] || [ -z "$inner" ] || [ "$host" = "$inner" ]; then
        echo "native_runner.sh: bad --bind spec: $spec" >&2
        exit 5
    fi
    if [ ! -e "$host" ]; then
        echo "native_runner.sh: bind source missing: $host" >&2
        exit 6
    fi
    if [ "$DRY_RUN" -eq 0 ]; then
        ensure_mount "$host" bind "$ROOTFS$inner"
    else
        echo "[dry-run] would bind $host -> $ROOTFS$inner"
    fi
done

# ---- env vars -------------------------------------------------------------

# Default env — matches extract_container_rootfs.sh so the chroot'd python
# finds TensorRT-LLM libraries.
CHROOT_ENV=(
    "HOME=/root"
    "PATH=/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
    "LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/tensorrt_libs:/usr/local/lib/python3.12/dist-packages/tensorrt:/usr/local/cuda/lib64:/usr/local/lib:/usr/lib/aarch64-linux-gnu"
    "PYTHONPATH=/usr/local/lib/python3.12/dist-packages"
    "CUDA_HOME=/usr/local/cuda"
    "BENCHMARK_ENVIRONMENT_KIND=native"
)
for e in "${EXTRA_ENVS[@]}"; do
    CHROOT_ENV+=("$e")
done

# ---- execute --------------------------------------------------------------

if [ "$DRY_RUN" -eq 1 ]; then
    echo "[dry-run] chroot $ROOTFS /usr/bin/env -i \\"
    for e in "${CHROOT_ENV[@]}"; do
        echo "    $e \\"
    done
    printf '    %s\n' "${CMD[*]}"
    exit 0
fi

chroot "$ROOTFS" /usr/bin/env -i "${CHROOT_ENV[@]}" "${CMD[@]}"
