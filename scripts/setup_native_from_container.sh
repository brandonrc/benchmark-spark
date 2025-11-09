#!/bin/bash
#
# Extract Container Environment for Native Use
# This script extracts Python packages and settings from the container
# to create a matching native environment
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

CONTAINER_IMAGE="${CONTAINER_IMAGE:-nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev}"
INSTALL_DIR="${INSTALL_DIR:-$HOME/tensorrt-llm-native}"
PYTHON_VERSION="3.12"  # Container uses Python 3.12

echo "========================================"
echo "  Extract Container Environment"
echo "========================================"
echo "Container: ${CONTAINER_IMAGE}"
echo "Install directory: ${INSTALL_DIR}"
echo ""

# Create installation directory
mkdir -p "${INSTALL_DIR}"
cd "${INSTALL_DIR}"

# Step 1: Extract pip freeze from container
echo "Step 1: Extracting package list from container..."
PACKAGES_FILE="${INSTALL_DIR}/container_packages.txt"

docker run --rm "${CONTAINER_IMAGE}" bash -c "pip freeze 2>/dev/null" | grep -E "^[a-zA-Z0-9_-]+[=<>]" > "${PACKAGES_FILE}"
echo -e "${GREEN}✓ Extracted $(wc -l < ${PACKAGES_FILE}) packages${NC}"

# Step 2: Extract Python version from container
echo ""
echo "Step 2: Checking container Python version..."
CONTAINER_PYTHON=$(docker run --rm "${CONTAINER_IMAGE}" bash -c "python --version 2>&1" | grep "Python" | awk '{print $2}')
echo "Container Python: ${CONTAINER_PYTHON}"

# Check if we have matching Python version
if ! command -v python${PYTHON_VERSION} &> /dev/null; then
    echo -e "${YELLOW}Python ${PYTHON_VERSION} not found. Checking for python3...${NC}"
    if command -v python3 &> /dev/null; then
        HOST_PYTHON=$(python3 --version | awk '{print $2}')
        echo "Host Python: ${HOST_PYTHON}"
        if [ "${HOST_PYTHON:0:4}" != "${CONTAINER_PYTHON:0:4}" ]; then
            echo -e "${YELLOW}WARNING: Version mismatch. Container: ${CONTAINER_PYTHON}, Host: ${HOST_PYTHON}${NC}"
            echo "This may cause compatibility issues."
        fi
        PYTHON_CMD="python3"
    else
        echo -e "${RED}ERROR: Python 3 not found${NC}"
        exit 1
    fi
else
    PYTHON_CMD="python${PYTHON_VERSION}"
fi

# Step 3: Create virtual environment
echo ""
echo "Step 3: Creating virtual environment..."
VENV_DIR="${INSTALL_DIR}/venv"

if [ -d "${VENV_DIR}" ]; then
    echo -e "${YELLOW}Virtual environment already exists. Removing...${NC}"
    rm -rf "${VENV_DIR}"
fi

${PYTHON_CMD} -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
echo -e "${GREEN}✓ Virtual environment created${NC}"

# Step 4: Upgrade pip to match container
echo ""
echo "Step 4: Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Step 5: Extract CUDA info from container
echo ""
echo "Step 5: Checking CUDA versions..."
CONTAINER_CUDA=$(docker run --rm "${CONTAINER_IMAGE}" bash -c "nvcc --version 2>&1" | grep "release" | sed -n 's/.*release \([0-9.]*\).*/\1/p')
echo "Container CUDA: ${CONTAINER_CUDA}"

if command -v nvcc &> /dev/null; then
    HOST_CUDA=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9.]*\).*/\1/p')
    echo "Host CUDA: ${HOST_CUDA}"

    if [ "${CONTAINER_CUDA}" != "${HOST_CUDA}" ]; then
        echo -e "${YELLOW}CUDA version mismatch!${NC}"
        CUDA_12_9_PATH="/usr/local/cuda-${CONTAINER_CUDA}"

        if [ -d "${CUDA_12_9_PATH}" ]; then
            echo -e "${GREEN}✓ CUDA ${CONTAINER_CUDA} found at ${CUDA_12_9_PATH}${NC}"
            export CUDA_HOME="${CUDA_12_9_PATH}"
            export PATH="${CUDA_12_9_PATH}/bin:$PATH"
            export LD_LIBRARY_PATH="${CUDA_12_9_PATH}/lib64:$LD_LIBRARY_PATH"
        else
            echo -e "${YELLOW}CUDA ${CONTAINER_CUDA} not found on host.${NC}"
            echo "Install with: sudo apt-get install -y cuda-toolkit-${CONTAINER_CUDA//./-}"
            echo "Continuing with host CUDA ${HOST_CUDA}..."
        fi
    fi
else
    echo -e "${RED}ERROR: nvcc not found on host${NC}"
    exit 1
fi

# Step 6: Filter and install packages
echo ""
echo "Step 6: Installing packages from container..."
echo "This may take a while..."

# Filter out packages that might cause issues or are system-specific
FILTERED_PACKAGES="${INSTALL_DIR}/filtered_packages.txt"
cat "${PACKAGES_FILE}" | \
    grep -v "^nvidia-" | \
    grep -v "^triton" | \
    grep -v "^-e git" | \
    grep -v "@ file://" > "${FILTERED_PACKAGES}" || true

# Try to install packages
pip install -r "${FILTERED_PACKAGES}" || {
    echo -e "${YELLOW}Some packages failed to install. This is normal.${NC}"
    echo "Continuing..."
}

# Step 7: Install key packages explicitly
echo ""
echo "Step 7: Installing key packages..."

# PyTorch (from container's torch version)
TORCH_VERSION=$(grep "^torch==" "${PACKAGES_FILE}" | cut -d'=' -f3 || echo "")
if [ -n "${TORCH_VERSION}" ]; then
    echo "Installing PyTorch ${TORCH_VERSION}..."
    pip install "torch==${TORCH_VERSION}" torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 || {
        echo -e "${YELLOW}Failed to install exact version, trying latest...${NC}"
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    }
else
    echo -e "${YELLOW}Could not determine torch version, installing latest...${NC}"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
fi

# Install numpy and other critical packages
echo "Installing critical packages..."
pip install numpy pandas pyyaml

# TensorRT-LLM (if available as package)
echo "Checking for TensorRT-LLM..."
if grep -q "tensorrt-llm" "${PACKAGES_FILE}"; then
    TRTLLM_VERSION=$(grep "tensorrt-llm" "${PACKAGES_FILE}" | head -n1 | cut -d'=' -f3 || echo "")
    if [ -n "${TRTLLM_VERSION}" ]; then
        pip install "tensorrt-llm==${TRTLLM_VERSION}" || echo -e "${YELLOW}TensorRT-LLM ${TRTLLM_VERSION} not available via pip${NC}"
    fi
fi

# Step 8: Verify installation
echo ""
echo "Step 8: Verifying installation..."

python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo -e "${GREEN}✓ PyTorch CUDA OK${NC}"
else
    echo -e "${RED}ERROR: PyTorch CUDA not available${NC}"
fi

# Step 9: Create activation script
echo ""
echo "Step 9: Creating activation script..."

ACTIVATE_SCRIPT="${INSTALL_DIR}/activate.sh"
cat > "${ACTIVATE_SCRIPT}" << EOF
#!/bin/bash
# Activate native environment (extracted from container)

SCRIPT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"

# Activate virtual environment
source "\${SCRIPT_DIR}/venv/bin/activate"

# Set up CUDA ${CONTAINER_CUDA} to match container
if [ -d "/usr/local/cuda-${CONTAINER_CUDA}" ]; then
    export CUDA_HOME="/usr/local/cuda-${CONTAINER_CUDA}"
    export LD_LIBRARY_PATH="/usr/local/cuda-${CONTAINER_CUDA}/lib64:\$LD_LIBRARY_PATH"
    export PATH="/usr/local/cuda-${CONTAINER_CUDA}/bin:\$PATH"
    echo "Using CUDA ${CONTAINER_CUDA} (matching container)"
else
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH
    export PATH=/usr/local/cuda/bin:\$PATH
    echo "Using system default CUDA"
fi

echo "Container-matched native environment activated"
echo "Python: \$(which python)"
echo "CUDA: \${CUDA_HOME:-/usr/local/cuda}"
echo "To deactivate, run: deactivate"
EOF

chmod +x "${ACTIVATE_SCRIPT}"
echo -e "${GREEN}✓ Activation script created${NC}"

# Print summary
echo ""
echo "========================================"
echo "  Setup Complete!"
echo "========================================"
echo ""
echo "Installation directory: ${INSTALL_DIR}"
echo "Activation script: ${ACTIVATE_SCRIPT}"
echo ""
echo "To use this environment:"
echo "  source ${ACTIVATE_SCRIPT}"
echo ""
echo "Container packages: ${PACKAGES_FILE}"
echo "Filtered packages: ${FILTERED_PACKAGES}"
echo ""
echo "Container specs:"
echo "  Python: ${CONTAINER_PYTHON}"
echo "  CUDA: ${CONTAINER_CUDA}"
echo ""
