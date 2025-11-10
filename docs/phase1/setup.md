# Setup Guide

## Prerequisites

### Hardware
- NVIDIA DGX Spark system (or similar GPU-equipped server)
- Minimum 1 GPU with 12GB+ VRAM
- 32GB+ system RAM recommended

### Software
- Ubuntu 22.04 LTS (or similar Linux distribution)
- NVIDIA GPU drivers (version 535.x or later)
- CUDA Toolkit 12.x
- Docker Engine with NVIDIA Container Toolkit
- Python 3.10+
- Git

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/benchmark-spark.git
cd benchmark-spark
```

### 2. Run Container Benchmarks (Easiest)

Container benchmarks require minimal setup:

```bash
# Pull the NVIDIA container (first time only)
docker pull nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev

# Run all benchmarks
./scripts/run_container_benchmark.sh all

# Or run individual benchmarks
./scripts/run_container_benchmark.sh matmul
./scripts/run_container_benchmark.sh llm
```

Results will be saved to `results/container/`

### 3. Setup Native Environment

Native setup requires building TensorRT-LLM from source (takes 30-60 minutes):

```bash
# Run automated setup script
./scripts/setup_native.sh

# The script will:
# - Check prerequisites
# - Create Python virtual environment
# - Install PyTorch and TensorFlow
# - Clone and build TensorRT-LLM
# - Install all dependencies
```

**Environment Variables:**

```bash
# Customize installation location (optional)
export INSTALL_DIR="$HOME/tensorrt-llm"
export PYTHON_VERSION="3.10"

./scripts/setup_native.sh
```

### 4. Run Native Benchmarks

```bash
# Activate native environment
source $HOME/tensorrt-llm/activate.sh

# Run all benchmarks
./scripts/run_native_benchmark.sh all

# Or run individual benchmarks
./scripts/run_native_benchmark.sh matmul
./scripts/run_native_benchmark.sh llm
```

Results will be saved to `results/native/`

### 5. Compare Results

```bash
# Install analysis dependencies
pip install pandas matplotlib seaborn scipy

# Run comparison
python analysis/compare_results.py

# View results
cat results/comparison/comparison_report.md
```

## Detailed Setup

### Docker Setup

1. **Install Docker:**
   ```bash
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   sudo usermod -aG docker $USER
   # Log out and back in
   ```

2. **Install NVIDIA Container Toolkit:**
   ```bash
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
     sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

3. **Verify Installation:**
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
   ```

### Native Environment Setup

#### Option 1: Automated (Recommended)

```bash
./scripts/setup_native.sh
```

#### Option 2: Manual

1. **Install CUDA Toolkit:**
   ```bash
   # Download from https://developer.nvidia.com/cuda-downloads
   # Or use package manager
   sudo apt-get install cuda-toolkit-12-1
   ```

2. **Create Virtual Environment:**
   ```bash
   python3.10 -m venv ~/tensorrt-llm-env
   source ~/tensorrt-llm-env/bin/activate
   ```

3. **Install PyTorch:**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

4. **Install TensorFlow:**
   ```bash
   pip install tensorflow[and-cuda]
   ```

5. **Clone and Build TensorRT-LLM:**
   ```bash
   git clone https://github.com/NVIDIA/TensorRT-LLM.git
   cd TensorRT-LLM
   pip install -r requirements.txt
   python scripts/build_wheel.py --clean
   pip install build/tensorrt_llm-*.whl
   ```

## Verification

### Check GPU Access

```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA
nvcc --version

# Check PyTorch CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check TensorFlow CUDA
python -c "import tensorflow as tf; print(f'GPUs: {tf.config.list_physical_devices(\"GPU\")}')"
```

### Run Simple Test

```bash
# Container test
docker run --rm --gpus all \
  -v $(pwd):/workspace \
  nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev \
  python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"

# Native test
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
```

## Troubleshooting

### Docker Issues

**Problem:** `docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]`

**Solution:**
```bash
sudo systemctl restart docker
# Or reinstall nvidia-container-toolkit
```

**Problem:** Permission denied

**Solution:**
```bash
sudo usermod -aG docker $USER
# Log out and back in
```

### CUDA Issues

**Problem:** PyTorch cannot find CUDA

**Solution:**
```bash
# Check CUDA installation
ls /usr/local/cuda/

# Add to ~/.bashrc
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

source ~/.bashrc
```

**Problem:** CUDA version mismatch

**Solution:**
```bash
# Check versions
nvidia-smi  # Driver CUDA version
nvcc --version  # Toolkit version
python -c "import torch; print(torch.version.cuda)"  # PyTorch version

# Reinstall PyTorch with matching version
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### TensorRT-LLM Build Issues

**Problem:** Build fails with compiler errors

**Solution:**
```bash
# Install build dependencies
sudo apt-get install build-essential cmake ninja-build

# Use specific CUDA version
export CUDA_HOME=/usr/local/cuda-12.1
python scripts/build_wheel.py --clean --trt_root /usr/local/tensorrt
```

**Problem:** Out of memory during build

**Solution:**
```bash
# Limit parallel jobs
export MAX_JOBS=4
python scripts/build_wheel.py --clean
```

## Configuration

### Benchmark Configuration

Edit `benchmarks/config.yaml` to customize:

```yaml
# Test configurations
llm_tests:
  - id: "custom_test"
    batch_size: 8
    input_length: 1024
    output_length: 256

# Execution parameters
execution:
  warmup_iterations: 10
  measurement_iterations: 100
  cooldown_seconds: 30
```

### GPU Monitoring

Adjust monitoring interval in runner scripts:

```bash
export GPU_MONITOR_INTERVAL=0.5  # Sample every 0.5 seconds
./scripts/run_container_benchmark.sh
```

## Next Steps

1. Run baseline benchmarks: `./scripts/run_container_benchmark.sh all`
2. Setup native environment: `./scripts/setup_native.sh`
3. Run native benchmarks: `./scripts/run_native_benchmark.sh all`
4. Compare results: `python analysis/compare_results.py`
5. Review report: `cat results/comparison/comparison_report.md`

## Getting Help

- Check logs in `results/` directories
- Review GPU metrics CSV files
- Enable debug mode: `set -x` in bash scripts
- Check NVIDIA documentation: https://docs.nvidia.com/
