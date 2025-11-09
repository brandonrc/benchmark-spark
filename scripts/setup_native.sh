#!/bin/bash
#
# Native TensorRT-LLM Setup Script
# Installs TensorRT-LLM and dependencies for native execution
#

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

INSTALL_DIR="${INSTALL_DIR:-$HOME/tensorrt-llm}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

echo "========================================"
echo "  TensorRT-LLM Native Setup"
echo "========================================"
echo "Install directory: ${INSTALL_DIR}"
echo "Python version: ${PYTHON_VERSION}"
echo ""

# Check prerequisites
echo "Checking prerequisites..."

# Check CUDA
if ! command -v nvcc &> /dev/null; then
    echo -e "${RED}ERROR: CUDA (nvcc) not found. Please install CUDA Toolkit 12.x${NC}"
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9.]*\).*/\1/p')
echo -e "${GREEN}✓ CUDA ${CUDA_VERSION} found${NC}"

# Check if CUDA version is 13.x - we need 12.9 to match the container
CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
CUDA_MINOR=$(echo $CUDA_VERSION | cut -d. -f2)
CUDA_12_9_PATH="/usr/local/cuda-12.9"

if [ "$CUDA_MAJOR" == "13" ] || [ "$CUDA_VERSION" != "12.9" ]; then
    echo -e "${YELLOW}Note: Container uses CUDA 12.9. Current CUDA is ${CUDA_VERSION}${NC}"

    # Check if CUDA 12.9 is already installed
    if [ -d "$CUDA_12_9_PATH" ]; then
        echo -e "${GREEN}✓ CUDA 12.9 found at ${CUDA_12_9_PATH}${NC}"
    else
        echo -e "${YELLOW}CUDA 12.9 not found. Would you like to install it? (y/n)${NC}"
        echo "This will install CUDA 12.9 alongside your existing CUDA ${CUDA_VERSION}"
        echo "Commands to run:"
        echo "  sudo apt-get update"
        echo "  sudo apt-get install -y cuda-toolkit-12-9"
        read -p "Install CUDA 12.9? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "Installing CUDA 12.9 toolkit..."
            sudo apt-get update
            sudo apt-get install -y cuda-toolkit-12-9

            if [ ! -d "$CUDA_12_9_PATH" ]; then
                echo -e "${RED}ERROR: CUDA 12.9 installation failed${NC}"
                echo "Continuing with CUDA ${CUDA_VERSION} (will use PyTorch CUDA 12.1)"
            else
                echo -e "${GREEN}✓ CUDA 12.9 installed successfully${NC}"
            fi
        else
            echo "Skipping CUDA 12.9 installation. Will use PyTorch with CUDA 12.1 support."
        fi
    fi

    # If CUDA 12.9 exists, use it
    if [ -d "$CUDA_12_9_PATH" ]; then
        export CUDA_HOME="$CUDA_12_9_PATH"
        export PATH="$CUDA_12_9_PATH/bin:$PATH"
        export LD_LIBRARY_PATH="$CUDA_12_9_PATH/lib64:$LD_LIBRARY_PATH"
        echo -e "${GREEN}✓ Environment configured to use CUDA 12.9${NC}"
        CUDA_VERSION="12.9"
    fi
fi

# Check nvidia-smi
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}ERROR: nvidia-smi not found. Please install NVIDIA drivers${NC}"
    exit 1
fi

echo -e "${GREEN}✓ NVIDIA drivers OK${NC}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# Check Python
if ! command -v python${PYTHON_VERSION} &> /dev/null; then
    echo -e "${YELLOW}WARNING: Python ${PYTHON_VERSION} not found. Trying 'python3'...${NC}"
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}ERROR: Python 3 not found${NC}"
        exit 1
    fi
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python${PYTHON_VERSION}"
fi

PYTHON_ACTUAL_VERSION=$(${PYTHON_CMD} --version | cut -d' ' -f2)
echo -e "${GREEN}✓ Python ${PYTHON_ACTUAL_VERSION} found${NC}"

# Check git
if ! command -v git &> /dev/null; then
    echo -e "${RED}ERROR: git not found. Please install git${NC}"
    exit 1
fi

echo -e "${GREEN}✓ git OK${NC}"
echo ""

# Create installation directory
mkdir -p "${INSTALL_DIR}"
cd "${INSTALL_DIR}"

# Create virtual environment
VENV_DIR="${INSTALL_DIR}/venv"
if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating Python virtual environment..."
    ${PYTHON_CMD} -m venv "${VENV_DIR}"
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi

# Activate virtual environment
source "${VENV_DIR}/bin/activate"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install PyTorch
echo ""
# Determine which PyTorch CUDA version to use
CUDA_MAJOR_MINOR=$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.')
if [ "$CUDA_VERSION" == "12.9" ]; then
    echo "Installing PyTorch with CUDA 12.1 support (closest to CUDA 12.9)..."
    PYTORCH_INDEX="cu121"
elif [ "$CUDA_MAJOR" == "13" ]; then
    echo "Installing PyTorch with CUDA 12.1 support (compatible with CUDA 13)..."
    PYTORCH_INDEX="cu121"
else
    echo "Installing PyTorch with CUDA ${CUDA_MAJOR_MINOR} support..."
    PYTORCH_INDEX="cu${CUDA_MAJOR_MINOR}"
fi

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/${PYTORCH_INDEX}

# Verify PyTorch CUDA
echo ""
echo "Verifying PyTorch CUDA support..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo -e "${RED}ERROR: PyTorch CUDA not available${NC}"
    exit 1
fi

echo -e "${GREEN}✓ PyTorch with CUDA installed successfully${NC}"

# Install TensorFlow (for matmul benchmark)
echo ""
echo "Installing TensorFlow..."
pip install tensorflow[and-cuda]

# Install additional dependencies
echo ""
echo "Installing additional dependencies..."
pip install numpy pandas matplotlib seaborn pyyaml

# Clone TensorRT-LLM (if not already present)
TENSORRT_LLM_DIR="${INSTALL_DIR}/TensorRT-LLM"
if [ ! -d "${TENSORRT_LLM_DIR}" ]; then
    echo ""
    echo "Cloning TensorRT-LLM repository..."
    git clone https://github.com/NVIDIA/TensorRT-LLM.git "${TENSORRT_LLM_DIR}"
    cd "${TENSORRT_LLM_DIR}"

    # Checkout a stable version (adjust as needed)
    echo "Checking out latest stable release..."
    git checkout $(git describe --tags $(git rev-list --tags --max-count=1) 2>/dev/null || echo "main")
else
    echo -e "${GREEN}✓ TensorRT-LLM repository already exists${NC}"
    cd "${TENSORRT_LLM_DIR}"
fi

# Build TensorRT-LLM
echo ""
echo -e "${YELLOW}Building TensorRT-LLM (this may take 30-60 minutes)...${NC}"
echo "Log file: ${INSTALL_DIR}/build.log"

# Install TensorRT-LLM dependencies
pip install -r requirements.txt

# Build
python scripts/build_wheel.py --clean --trt_root /usr/local/tensorrt 2>&1 | tee "${INSTALL_DIR}/build.log"

# Install the built wheel
WHEEL_FILE=$(find build -name "tensorrt_llm-*.whl" | head -n1)
if [ -z "${WHEEL_FILE}" ]; then
    echo -e "${RED}ERROR: TensorRT-LLM wheel not found${NC}"
    echo "Please check ${INSTALL_DIR}/build.log for errors"
    exit 1
fi

pip install "${WHEEL_FILE}"

echo -e "${GREEN}✓ TensorRT-LLM built and installed successfully${NC}"

# Verify installation
echo ""
echo "Verifying TensorRT-LLM installation..."
if python -c "import tensorrt_llm; print(f'TensorRT-LLM version: {tensorrt_llm.__version__}')" 2>/dev/null; then
    echo -e "${GREEN}✓ TensorRT-LLM import successful${NC}"
else
    echo -e "${YELLOW}WARNING: TensorRT-LLM import failed. You may need to set environment variables.${NC}"
fi

# Create activation script
ACTIVATE_SCRIPT="${INSTALL_DIR}/activate.sh"
cat > "${ACTIVATE_SCRIPT}" << 'EOF'
#!/bin/bash
# Activate TensorRT-LLM native environment

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate virtual environment
source "${SCRIPT_DIR}/venv/bin/activate"

# Set up CUDA 12.9 if available (to match container CUDA 12.9)
if [ -d "/usr/local/cuda-12.9" ]; then
    export CUDA_HOME="/usr/local/cuda-12.9"
    export LD_LIBRARY_PATH="/usr/local/cuda-12.9/lib64:$LD_LIBRARY_PATH"
    export PATH="/usr/local/cuda-12.9/bin:$PATH"
    echo "Using CUDA 12.9 (matching container)"
else
    # Fallback to default CUDA
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    export PATH=/usr/local/cuda/bin:$PATH
    echo "Using system default CUDA"
fi

echo "TensorRT-LLM native environment activated"
echo "Python: $(which python)"
echo "CUDA: ${CUDA_HOME:-/usr/local/cuda}"
echo "To deactivate, run: deactivate"
EOF

chmod +x "${ACTIVATE_SCRIPT}"

# Print summary
echo ""
echo "========================================"
echo "  Installation Complete!"
echo "========================================"
echo ""
echo "Installation directory: ${INSTALL_DIR}"
echo ""
echo "To use the native environment:"
echo "  source ${ACTIVATE_SCRIPT}"
echo ""
echo "To run native benchmarks:"
echo "  source ${ACTIVATE_SCRIPT}"
echo "  cd /path/to/benchmark-spark"
echo "  ./scripts/run_native_benchmark.sh"
echo ""
echo "Python packages installed:"
pip list | grep -E "torch|tensorflow|tensorrt"
echo ""
