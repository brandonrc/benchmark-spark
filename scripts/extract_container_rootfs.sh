#!/bin/bash
#
# Extract Container Root Filesystem
# Extracts the container's filesystem to run binaries natively (bare metal)
#

set -e

CONTAINER_IMAGE="${CONTAINER_IMAGE:-nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev}"
EXTRACT_DIR="${EXTRACT_DIR:-$HOME/container-rootfs}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "========================================"
echo "  Extract Container Filesystem"
echo "========================================"
echo "Container: ${CONTAINER_IMAGE}"
echo "Extract to: ${EXTRACT_DIR}"
echo ""
echo "This will extract the container's binaries to run natively"
echo "on bare metal (no Docker overhead)"
echo ""

# Check architecture match
HOST_ARCH=$(uname -m)
CONTAINER_ARCH=$(docker run --rm "${CONTAINER_IMAGE}" uname -m 2>&1 | tail -n1)

echo "Host architecture: ${HOST_ARCH}"
echo "Container architecture: ${CONTAINER_ARCH}"

if [ "${HOST_ARCH}" != "${CONTAINER_ARCH}" ]; then
    echo -e "${RED}ERROR: Architecture mismatch!${NC}"
    echo "Cannot run container binaries on different architecture"
    exit 1
fi

echo -e "${GREEN}✓ Architectures match - extraction is safe${NC}"
echo ""

# Create extraction directory
mkdir -p "${EXTRACT_DIR}"

# Export container filesystem
echo "Step 1: Creating temporary container..."
TEMP_CONTAINER=$(docker create "${CONTAINER_IMAGE}")
echo "Container ID: ${TEMP_CONTAINER}"

echo ""
echo "Step 2: Exporting container filesystem..."
echo "This may take several minutes..."

docker export "${TEMP_CONTAINER}" | tar -C "${EXTRACT_DIR}" -xf -

echo -e "${GREEN}✓ Filesystem extracted${NC}"

# Cleanup temporary container
echo ""
echo "Step 3: Cleaning up temporary container..."
docker rm "${TEMP_CONTAINER}" >/dev/null
echo -e "${GREEN}✓ Cleanup complete${NC}"

# Create chroot wrapper script
echo ""
echo "Step 4: Creating runner script..."

RUNNER_SCRIPT="${EXTRACT_DIR}/run_in_rootfs.sh"
cat > "${RUNNER_SCRIPT}" << 'EOFRUNNER'
#!/bin/bash
#
# Run commands in extracted container rootfs
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Mount necessary host filesystems
mount_host_fs() {
    echo "Mounting host filesystems..."

    # Only mount if not already mounted
    mountpoint -q "${SCRIPT_DIR}/proc" || sudo mount -t proc /proc "${SCRIPT_DIR}/proc"
    mountpoint -q "${SCRIPT_DIR}/sys" || sudo mount --rbind /sys "${SCRIPT_DIR}/sys"
    mountpoint -q "${SCRIPT_DIR}/dev" || sudo mount --rbind /dev "${SCRIPT_DIR}/dev"

    # Mount host GPU devices
    mkdir -p "${SCRIPT_DIR}/dev/dri"

    echo "✓ Host filesystems mounted"
}

# Unmount filesystems
unmount_host_fs() {
    echo "Unmounting host filesystems..."
    sudo umount "${SCRIPT_DIR}/proc" 2>/dev/null || true
    sudo umount "${SCRIPT_DIR}/sys" 2>/dev/null || true
    sudo umount "${SCRIPT_DIR}/dev" 2>/dev/null || true
    echo "✓ Cleanup complete"
}

# Trap to cleanup on exit
trap unmount_host_fs EXIT

# Mount host fs
mount_host_fs

# Set environment variables (include all important library paths from container)
export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/tensorrt_libs:/usr/local/lib/python3.12/dist-packages/tensorrt:/usr/local/cuda/lib64:/usr/local/lib:/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin
export PYTHONPATH=/usr/local/lib/python3.12/dist-packages

# Run command in chroot
echo ""
echo "Running in bare metal (no container)..."
sudo chroot "${SCRIPT_DIR}" /usr/bin/env -i \
    HOME=/root \
    PATH="${PATH}" \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
    PYTHONPATH="${PYTHONPATH}" \
    CUDA_HOME=/usr/local/cuda \
    "$@"

EXITCODE=$?

# Cleanup
unmount_host_fs
trap - EXIT

exit $EXITCODE
EOFRUNNER

chmod +x "${RUNNER_SCRIPT}"

# Create Python wrapper
PYTHON_WRAPPER="${EXTRACT_DIR}/run_python.sh"
cat > "${PYTHON_WRAPPER}" << EOFPYTHON
#!/bin/bash
# Run Python from extracted rootfs
SCRIPT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
"\${SCRIPT_DIR}/run_in_rootfs.sh" /usr/bin/python3 "\$@"
EOFPYTHON

chmod +x "${PYTHON_WRAPPER}"

# Create activation script
ACTIVATE_SCRIPT="${EXTRACT_DIR}/activate.sh"
cat > "${ACTIVATE_SCRIPT}" << EOFACTIVATE
#!/bin/bash
# Activate extracted container environment

ROOTFS_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"

echo "========================================"
echo "  Container Rootfs Environment"
echo "========================================"
echo "Location: \${ROOTFS_DIR}"
echo ""
echo "This environment uses the container's binaries"
echo "running directly on bare metal (no Docker overhead)"
echo ""
echo "Commands:"
echo "  \${ROOTFS_DIR}/run_python.sh <script.py>  - Run Python scripts"
echo "  \${ROOTFS_DIR}/run_in_rootfs.sh <cmd>     - Run any command"
echo ""
EOFACTIVATE

chmod +x "${ACTIVATE_SCRIPT}"

# Print summary
echo ""
echo "========================================"
echo "  Extraction Complete!"
echo "========================================"
echo ""
echo "Root filesystem: ${EXTRACT_DIR}"
echo "Size: $(du -sh ${EXTRACT_DIR} | cut -f1)"
echo ""
echo "To run Python from extracted container:"
echo "  ${PYTHON_WRAPPER} script.py"
echo ""
echo "To run any command:"
echo "  ${RUNNER_SCRIPT} <command>"
echo ""
echo "Next steps:"
echo "  1. Update run_native_benchmark.sh to use ${PYTHON_WRAPPER}"
echo "  2. Run benchmarks on bare metal with container binaries"
echo ""
